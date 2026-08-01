"""ピンボール／ブロック崩しのテスト。

モデル無しで通る範囲 (物理・ラダー・シーケンサ・隠れ状態・可視化) を検証する。
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tetris_vla.pinball import (
    BLOCK_ROWS,
    FIELD_H,
    FIELD_W,
    N_SEG,
    PADDLE_Y,
    Ball,
    LadderPLC,
    PinballConfig,
    PinballWorld,
    Rung,
    SeqState,
    Sequencer,
    build_ladder,
)
from tetris_vla.pinball_agents import (
    HeuristicStrategist,
    MockVLAPaddle,
    ScriptedPaddle,
    StackConfig,
    predict_landing_x,
    render_pinball,
    run_stack,
)


def _stack(seed=3, ticks=2200, skill=0.9):
    w = PinballWorld(PinballConfig(seed=seed, max_ticks=ticks))
    res = run_stack(w, MockVLAPaddle(skill=skill), HeuristicStrategist(),
                    StackConfig(vlm_every=150, verbose=False))
    return w, res


# --- ラダーロジック -------------------------------------------------------


def test_rung_series_and_parallel() -> None:
    r = Rung("t", [["A", "B"], ["C"]], "Y")
    assert r.evaluate({"A": True, "B": True}) is True      # 直列 AND
    assert r.evaluate({"A": True, "B": False}) is False
    assert r.evaluate({"C": True}) is True                  # 並列 OR
    assert r.evaluate({}) is False


def test_rung_negated_contact() -> None:
    r = Rung("t", [["A", "!B"]], "Y")
    assert r.evaluate({"A": True, "B": False}) is True
    assert r.evaluate({"A": True, "B": True}) is False


def test_rung_self_seal_latches() -> None:
    """自己保持。条件が消えてもコイルが立ったままになる (実機のラッチ)。"""
    r = Rung("t", [["SET"]], "Y", seal=True)
    assert r.evaluate({"SET": True}) is True
    assert r.evaluate({"SET": False, "Y": True}) is True
    assert r.evaluate({"SET": False, "Y": False}) is False


def test_plc_evaluates_rungs_in_order() -> None:
    """上のラングが書いたコイルを下のラングが同一スキャン内で読める。

    これが「ラダーは読み切れる」ことの根拠なので、順序依存は仕様として固定する。
    """
    plc = LadderPLC([Rung("1", [["IN"]], "M1"), Rung("2", [["M1"]], "OUT")])
    bits = plc.scan({"IN": True})
    assert bits["M1"] and bits["OUT"], "1 スキャンで最下段まで伝播するはず"

    rev = LadderPLC([Rung("2", [["M1"]], "OUT"), Rung("1", [["IN"]], "M1")])
    bits = rev.scan({"IN": True})
    assert bits["M1"] and not bits["OUT"], "順序を逆にすると 1 スキャン遅れる"


def test_demo_ladder_interlocks() -> None:
    plc = LadderPLC(build_ladder())
    # ドローン飛行中はゲート操作を禁止
    bits = plc.scan({"drone_active": True, "ball_lower": True, "stall_timer": True,
                     "ball_upper": False})
    assert bits["gate_inhibit"] is True
    # 危険域にボールがあれば修理許可は取り消される
    bits = plc.scan({"jig_broken": True, "ball_in_danger": True, "drone_active": False,
                     "ball_upper": False, "ball_lower": True, "stall_timer": False})
    assert bits["repair_cancel"] is True
    # 危険域外なら許可が立つ
    plc2 = LadderPLC(build_ladder())
    bits = plc2.scan({"jig_broken": True, "ball_in_danger": False, "drone_active": False})
    assert bits["repair_permit"] is True and not bits["repair_cancel"]


def test_ladder_is_deterministic() -> None:
    a = LadderPLC(build_ladder())
    b = LadderPLC(build_ladder())
    seqs = [{"ball_upper": i % 2 == 0, "ball_lower": i % 3 == 0,
             "stall_timer": i > 5, "jig_broken": i > 8} for i in range(12)]
    assert [a.scan(s) for s in seqs] == [b.scan(s) for s in seqs]


# --- 隠れ状態 -------------------------------------------------------------


def test_durability_is_hidden_from_telemetry() -> None:
    """耐久度はテレメトリに出さない。打数だけが観測できる。"""
    w = PinballWorld(PinballConfig(seed=1))
    tel = w.telemetry()
    for j, t in zip(w.jigs, tel["jigs"]):
        assert "hits" in t and "broken" in t
        assert "durability" not in t, "隠れ状態が漏れている"
        assert "max_durability" not in t


def test_durability_is_not_drawn_by_default() -> None:
    """モデルに渡す画像に耐久度を描いてはいけない。"""
    w = PinballWorld(PinballConfig(seed=1))
    plain = np.asarray(render_pinball(w).convert("L"))
    shown = np.asarray(render_pinball(w, show_hidden=True).convert("L"))
    assert not np.array_equal(plain, shown), "show_hidden が効いていない"


def test_jig_becomes_pass_through_only_at_zero() -> None:
    w = PinballWorld(PinballConfig(seed=1))
    j = w.jigs[0]
    j.durability = 2
    b = Ball(x=j.x, y=j.y - j.r - 0.4, vx=0.0, vy=6.0)
    w.balls = [b]
    for _ in range(2):
        b.x, b.y, b.vx, b.vy = j.x, j.y - j.r - 0.2, 0.0, 6.0
        w._bounce_jigs(b)
    assert j.broken is True
    assert j.hits == 2
    # 素通りになったら弾かない
    b.x, b.y, b.vy = j.x, j.y - j.r - 0.2, 6.0
    before = b.vy
    w._bounce_jigs(b)
    assert b.vy == before


# --- ブロック崩しのルール -------------------------------------------------


def test_paddle_segment_maps_to_block_column() -> None:
    """当たった位置のセグメント = 崩れる列。この対応がこの題材の中心。"""
    w = PinballWorld(PinballConfig(seed=1))
    p = w.paddle
    for seg in range(N_SEG):
        assert p.segment_of(p.segment_center(seg)) == seg


def test_hitting_a_segment_breaks_that_column_only() -> None:
    w = PinballWorld(PinballConfig(seed=1))
    seg = 2
    before = [int(w.blocks[:, c].sum()) for c in range(N_SEG)]
    w._break_block(seg)
    after = [int(w.blocks[:, c].sum()) for c in range(N_SEG)]
    assert after[seg] == before[seg] - 1
    assert all(after[c] == before[c] for c in range(N_SEG) if c != seg)
    assert w.broken_per_col[seg] == 1


def test_ball_on_paddle_breaks_the_matching_column() -> None:
    w = PinballWorld(PinballConfig(seed=1))
    p = w.paddle
    seg = 4
    b = Ball(x=p.segment_center(seg), y=PADDLE_Y - 0.1, vx=0.0, vy=5.0)
    w.balls = [b]
    w._paddle_and_blocks(b)
    assert w.broken_per_col[seg] == 1
    assert b.vy < 0, "跳ね返っていない"


def test_score_rewards_evenness() -> None:
    w = PinballWorld(PinballConfig(seed=1))
    for _ in range(2):
        for c in range(N_SEG):
            w._break_block(c)
    even = w.score()
    # 同じ総数 (2*N_SEG=12) を、少数の列に集中させた場合と比べる。
    # 1 列は BLOCK_ROWS しか無いので、容量を超えない配り方にすること
    w2 = PinballWorld(PinballConfig(seed=1))
    for c in range(2 * N_SEG // BLOCK_ROWS):
        for _ in range(BLOCK_ROWS):
            w2._break_block(c)
    biased = w2.score()
    assert even["blocks_broken"] == biased["blocks_broken"]
    assert even["evenness"] > biased["evenness"]
    assert even["score"] > biased["score"], "同じ崩し数なら均等なほうが高得点"


# --- 救済機とシーケンサ ---------------------------------------------------


def test_drone_repair_restores_the_jig() -> None:
    w = PinballWorld(PinballConfig(seed=1))
    j = w.jigs[1]
    j.durability, j.broken, j.hits = 0, True, 9
    w.balls = []
    # 経路上の健全ジグに触れると失敗するのが仕様なので、
    # ここでは「成功する経路」だけを切り出して検証する
    w.jigs = [j]
    w.drone.active, w.drone.target = True, j.jid
    w.drone.x, w.drone.y = j.x, j.y + 3.0
    for _ in range(400):
        w._drone()
        if not w.drone.active:
            break
    assert not j.broken and j.durability == j.max_durability and j.hits == 0


def test_drone_fails_when_it_touches_a_healthy_jig() -> None:
    w = PinballWorld(PinballConfig(seed=1))
    tgt, blocker = w.jigs[0], w.jigs[1]
    tgt.broken = True
    w.balls = []
    w.drone.active, w.drone.target = True, tgt.jid
    w.drone.x, w.drone.y = blocker.x, blocker.y + 1.2   # 健全ジグの真横から出発
    w._drone()
    assert w.drone.failed_reason and "ジグ" in w.drone.failed_reason
    assert not w.drone.active


def test_drone_fails_when_it_touches_the_ball() -> None:
    w = PinballWorld(PinballConfig(seed=1))
    tgt = w.jigs[0]
    tgt.broken = True
    w.drone.active, w.drone.target = True, tgt.jid
    w.drone.x, w.drone.y = 12.0, 20.0
    w.balls = [Ball(x=12.0, y=19.6, vx=0, vy=0)]
    w._drone()
    assert w.drone.failed_reason == "ボールに接触"


def test_sequencer_needs_ladder_permission_before_dispatch() -> None:
    """AI が何と言おうと、ラダーの許可なしにドローンは出せない。"""
    w = PinballWorld(PinballConfig(seed=1))
    w.jigs[0].broken = True
    seq = Sequencer()
    seq.vlm_advice = {"repair_ok": True}
    for _ in range(6):
        seq.step(w.tick, w, {"repair_permit": False, "repair_cancel": False})
    assert seq.state in (SeqState.RUN, SeqState.FAULT)
    assert not w.drone.active, "許可なしで発進してはいけない"


def test_vlm_can_veto_but_not_command() -> None:
    """VLM の助言は拒否権として働くが、発進を強制はできない。"""
    w = PinballWorld(PinballConfig(seed=1))
    w.jigs[0].broken = True
    seq = Sequencer()
    seq.vlm_advice = {"repair_ok": False}
    bits = {"repair_permit": True, "repair_cancel": False}
    for _ in range(8):
        seq.step(w.tick, w, bits)
    assert not w.drone.active, "VLM が危険と言ったら出さない"
    assert any("VLM" in e.reason for e in seq.events)


def test_full_stack_runs_and_exercises_repair() -> None:
    w, res = _stack(seed=3)
    assert res["ladder_scans"] > 100
    assert res["score"]["blocks_broken"] > 0
    assert len(res["seq_events"]) > 0
    kinds = {e["kind"] for e in res["world_events"]}
    assert "jig_broken" in kinds, "ジグが 1 つも壊れない設定では隠れ状態が試せない"


def test_stack_is_deterministic() -> None:
    a = _stack(seed=7)[1]["score"]
    b = _stack(seed=7)[1]["score"]
    assert a == b


def test_scripted_teacher_beats_a_clumsy_vla() -> None:
    """腕が落ちると素直に成績が落ちる = 評価装置として機能している。"""
    good = _stack(seed=3, skill=0.95)[1]["score"]["score"]
    bad = _stack(seed=3, skill=0.2)[1]["score"]["score"]
    assert good > bad, (good, bad)


def test_landing_prediction_is_sane() -> None:
    w = PinballWorld(PinballConfig(seed=2))
    w.balls = [Ball(x=5.0, y=10.0, vx=0.0, vy=8.0)]
    x = predict_landing_x(w)
    assert x is not None and 0 <= x <= FIELD_W


# --- 層のクロック ---------------------------------------------------------


def test_layers_run_at_their_own_clocks() -> None:
    """ラダーは毎 tick、VLA は行動列ぶん、VLM は最も疎。"""
    w, res = _stack(seed=3)
    vla = [c for c in res["calls"] if c["layer"] == "vla"]
    vlm = [c for c in res["calls"] if c["layer"] == "vlm"]
    assert res["ladder_scans"] >= len(res["frames"]) - 2
    assert len(vla) > len(vlm), "VLA のほうが高頻度で呼ばれるはず"
    assert all(c["output"]["chunk_len"] > 1 for c in vla), "VLA は行動列を返す"


def test_vla_call_records_its_inputs_and_outputs() -> None:
    _, res = _stack(seed=3)
    c = next(c for c in res["calls"] if c["layer"] == "vla")
    assert "state6" in c["inputs"] and len(c["inputs"]["state6"]) == 6
    assert "target_seg" in c["inputs"]
    assert c["output"]["chunk_len"] >= 1


# --- 可視化 ---------------------------------------------------------------


def test_svg_and_html_are_self_contained() -> None:
    from tetris_vla.pinballviz import pinball_html, pinball_svg

    _, res = _stack(seed=3)
    svg = pinball_svg(res)
    ET.fromstring(svg)
    assert "<animateTransform" in svg and "<script" not in svg
    doc = pinball_html(res, svg)
    for key in ("ラダーロジック", "シーケンサの遷移", "AI 層の入出力", "隠れ耐久度"):
        assert key in doc, key
    for text in (svg, doc):
        refs = re.findall(r'(?:src|href|xlink:href)\s*=\s*["\']([^"\']+)', text)
        assert not [r for r in refs if r.startswith(("http://", "https://", "//"))]


def test_html_documents_every_ladder_rung() -> None:
    from tetris_vla.pinballviz import pinball_html, pinball_svg

    _, res = _stack(seed=3)
    doc = pinball_html(res, pinball_svg(res))
    for r in build_ladder():
        assert r.name in doc, f"{r.name} が HTML に出ていない"
        assert r.comment in doc
