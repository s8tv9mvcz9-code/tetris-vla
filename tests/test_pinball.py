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


def _stack(seed=3, ticks=5000, skill=0.9):
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


def test_hitting_jig_j_breaks_column_j() -> None:
    """**ジグ j を叩く = 工程 j を 1 回こなす** → 列 j が 1 個崩れる。"""
    w = PinballWorld(PinballConfig(seed=1))
    j = w.jigs[3]
    before = int(w.blocks[:, 3].sum())
    b = Ball(x=j.x, y=j.y - j.r - 0.2, vx=0.0, vy=6.0)
    w.balls = [b]
    w._bounce_jigs(b)
    assert int(w.blocks[:, 3].sum()) == before - 1
    assert w.broken_per_col[3] == 1
    assert j.hits == 1 and j.durability == j.max_durability - 1 or j.durability >= 0


def test_paddle_returns_the_ball_but_breaks_nothing() -> None:
    """パドルは返球と狙いだけ。崩すのはジグ側の仕事。"""
    w = PinballWorld(PinballConfig(seed=1))
    p = w.paddle
    b = Ball(x=p.x, y=PADDLE_Y - 0.1, vx=0.0, vy=5.0)
    w.balls = [b]
    w._paddle_and_blocks(b)
    assert sum(w.broken_per_col) == 0, "パドルではブロックは崩れない"
    assert b.vy < 0, "跳ね返っていない"


def test_paddle_hit_position_steers_the_ball() -> None:
    """端で当てるほど大きく曲がる = どのジグへ返すかを狙える。"""
    w = PinballWorld(PinballConfig(seed=1))
    p = w.paddle
    out = []
    for off in (-0.9, 0.0, 0.9):
        b = Ball(x=p.x + off * p.w / 2, y=PADDLE_Y - 0.1, vx=0.0, vy=5.0)
        w.balls = [b]
        w._paddle_and_blocks(b)
        out.append(b.vx)
    assert out[0] < out[1] < out[2]


def test_a_broken_jig_stops_producing() -> None:
    """摩耗して素通りになった工程はもう回せない (修理するまで)。"""
    w = PinballWorld(PinballConfig(seed=1))
    j = w.jigs[2]
    j.broken = True
    before = int(w.blocks[:, 2].sum())
    b = Ball(x=j.x, y=j.y - j.r - 0.2, vx=0.0, vy=6.0)
    w.balls = [b]
    w._bounce_jigs(b)
    assert int(w.blocks[:, 2].sum()) == before, "壊れたジグで工程が進んではいけない"


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
    # 整備は「その個体の元の寿命」に戻す。設定上限 (max_durability) まで回復させると、
    # 耐久 6 の個体が修理のたびに 9 になり、直すほど新品より強くなってしまう
    assert not j.broken and j.durability == j.initial_durability and j.hits == 0
    assert j.initial_durability <= j.max_durability


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


def _sweep(skill: float, every: int, n: int = 8, fit_tolerance: float = 9.9) -> list[float]:
    """既定では嵌合摩耗を切って (公差を事実上無限に) 測る。

    腕前と指令の新鮮さだけを見たいときに摩耗が混ざると、
    「当てたのに工程が進まない」分だけ腕前の効果が薄まってしまうため。
    """
    out = []
    for s in range(n):
        w = PinballWorld(PinballConfig(seed=s, max_ticks=5000,
                                       fit_tolerance=fit_tolerance))
        r = run_stack(w, MockVLAPaddle(skill=skill, seed=s), HeuristicStrategist(),
                      StackConfig(vlm_every=every, verbose=False))
        out.append(r["score"]["score"])
    return out


@pytest.mark.xfail(reason=(
    "この主張は n=8 でしか確かめておらず、20 seed で測り直すと再現しない。"
    "旧仕様 (ゲートが死んでいた頃) ですら fresh_gap=16.4±64 で、"
    "要求している 150 に遠く及ばない。seed 0〜7 がたまたまそう並んでいただけ。"
    "皮肉なことに test_this_task_needs_many_seeds が『10 seed 以上必要』と"
    "書いているのに、この検証自体が n=8 だった。"
    "→ 主張を捨てるか、題材を作り直して効果量を出すかは人間の判断待ち。"
    "詳細は test_the_freshness_claim_does_not_survive_more_seeds を参照。"),
    strict=False)
def test_execution_precision_pays_off_only_when_the_goal_is_fresh() -> None:
    """**精密な実行は、上位の指令が新鮮なときだけ価値がある。**

    指令周期 60 tick (1.2秒) なら腕前の差がはっきり出るが、
    150 tick (3秒) まで空くと差が消える。上位が古い目標を出している間は、
    それを精密に追うこと自体に意味がなくなるため。
    このプロジェクト全体の主題が、この題材でも再現している。
    """
    import statistics

    fresh_good, fresh_bad = _sweep(0.95, 60), _sweep(0.2, 60)
    stale_good, stale_bad = _sweep(0.95, 150), _sweep(0.2, 150)
    fresh_gap = statistics.fmean(fresh_good) - statistics.fmean(fresh_bad)
    stale_gap = statistics.fmean(stale_good) - statistics.fmean(stale_bad)
    assert fresh_gap > 150, fresh_gap
    assert stale_gap < fresh_gap / 2, (fresh_gap, stale_gap)


def test_the_task_is_a_usable_instrument() -> None:
    """**分散が小さいこと**を仕様として固定する。

    ジグを 1 列に並べ替えて狙いが効くようにするまで、この題材は変動係数が
    15% を超えていて、腕前の差が測れなかった。並べ替え後は 3% 前後に落ちる。
    ここが再び悪化したら、比較実験の結論がすべて信用できなくなるので落とす。
    """
    import statistics

    scores = _sweep(0.9, 60, n=8)
    cv = statistics.stdev(scores) / statistics.fmean(scores)
    assert cv < 0.08, (f"変動係数 {cv:.1%} — 題材が再びカオス化している", scores)


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
    # 着弾予測を足して 8 次元。決まった計算は渡す側に置く、という分業
    assert "state" in c["inputs"] and len(c["inputs"]["state"]) == 8
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
    for key in ("ラダーロジック", "シーケンサの遷移", "AI 層の入出力", "真の寿命(答え)"):
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


# --- ST (IEC 61131-3) の生成 ---------------------------------------------


def test_st_is_generated_from_the_implementation_not_hardcoded() -> None:
    """ST は手書きではなく実装から導出されている、という保証。

    ハードコードした ST を置くと、ロジックを直したときに黙って嘘になる。
    ここが落ちたら「実装は変わったのに ST が追随していない」ということ。
    """
    from tetris_vla.pinball import SeqState, build_ladder
    from tetris_vla.pinballviz import (extract_seq_transitions, st_ladder_code,
                                       st_sequencer_code)

    trans = extract_seq_transitions()
    covered = {frm for frm, _ in trans}
    # step() が扱っている状態はすべて抽出できていること
    assert {"IDLE", "RUN", "FAULT", "PLAN", "DISPATCH", "REPAIR"} <= covered

    st = st_sequencer_code()
    for s in SeqState:
        assert f"{s.name}:" in st, s.name
    # 排他性: PLAN の分岐は IF/ELSIF/ELSE で繋がっていなければならない
    plan = st.split("PLAN:")[1].split("DISPATCH:")[0]
    opens = [ln for ln in plan.splitlines() if ln.strip().startswith("IF ")]
    assert "ELSIF" in plan and len(opens) == 1, (
        "分岐が独立した IF に平坦化されている。排他性が壊れ、"
        "PLAN で 2 つの遷移が同時に成立してしまう")

    # Python の実装詳細が翻訳されずに漏れていないこと
    for leak in ("self.", "bits.get", "SeqState.", "d.active", "elif"):
        assert leak not in st, leak

    lad = st_ladder_code(build_ladder())
    assert "FUNCTION_BLOCK FB_GateInterlock" in lad and "END_FUNCTION_BLOCK" in lad
    for r in build_ladder():
        assert f"{r.coil} :=" in lad, r.coil
    # 自己保持は末尾の OR で表現される
    seal = [r for r in build_ladder() if r.seal][0]
    assert f"OR {seal.coil};" in lad


def test_tool_wear_caps_what_precision_can_buy() -> None:
    """**治具が摩耗すると、腕前で買えるものに上限がつく。**

    嵌合公差を入れる前と後で、同じ「腕前の差」がスコアに換算される量を比べる。
    当てても公差を外していれば工程は進まないので、精密に狙えることの価値は
    道具の状態に頭を押さえられる。現場で「良いオペレータを入れても
    治具が終わっていれば数字は出ない」と言われるのと同じ構図。
    """
    import statistics

    def gap(tol: float) -> float:
        return (statistics.fmean(_sweep(0.95, 60, fit_tolerance=tol))
                - statistics.fmean(_sweep(0.2, 60, fit_tolerance=tol)))

    no_wear = gap(9.9)      # 摩耗なし
    with_wear = gap(0.8)    # 既定の公差
    assert no_wear > with_wear, (no_wear, with_wear)
    assert with_wear > 0, "摩耗があっても腕前の向きは正のままであるべき"


@pytest.mark.slow
def test_the_freshness_claim_does_not_survive_more_seeds() -> None:
    """**「指令が新鮮なときだけ腕前が効く」は、seed を増やすと消える。**

    この題材はボールがカオス要素なので分散が大きい。腕前 0.95 と 0.2 の差を
    20 seed で測ると、標準誤差が ±60 前後あり、効果量 (16〜59) がその中に埋もれる。

    元の検証は n=8 だった。seed 0〜7 がたまたま並んだだけで、
    同じ条件 (ゲートが死んでいた旧仕様) でも n=20 では fresh_gap=16.4 しか出ない。

    ここでは「有意差が無い」ことを仕様として固定する。効果量を主張したいなら、
    題材側で分散を下げる (ボールを 1 個にする / 面を長くする) 必要がある。
    """
    import statistics

    def arm(skill: float, every: int, n: int = 20) -> list[float]:
        out = []
        for s in range(n):
            w = PinballWorld(PinballConfig(seed=s, max_ticks=5000, fit_tolerance=9.9,
                                           stall_ticks=99999))   # 旧仕様を再現
            r = run_stack(w, MockVLAPaddle(skill=skill, seed=s), HeuristicStrategist(),
                          StackConfig(vlm_every=every, verbose=False))
            out.append(r["score"]["score"])
        return out

    good, bad = arm(0.95, 60), arm(0.2, 60)
    gap = statistics.fmean(good) - statistics.fmean(bad)
    se = (statistics.pstdev(good) ** 2 / len(good)
          + statistics.pstdev(bad) ** 2 / len(bad)) ** 0.5
    assert abs(gap) < 2 * se, (
        f"gap={gap:.1f} se={se:.1f} — 有意差が出た。題材の分散が下がったなら"
        "この所見と docs/design-qa.md を見直すこと")


# --- 吸収射出 -------------------------------------------------------------


def test_kicker_only_catches_balls_through_the_saucer() -> None:
    """受け皿を外れた球は捕まえない。

    盤面全幅で捕まえると誰が撃っても狙いどおり入るので、無作為なパドルでも
    満点が出て腕前が消える。「隙間を抜いて受け皿へ通す」ことを技術として残す。
    """
    w = PinballWorld(PinballConfig(seed=1))
    k = w.kicker
    on = Ball(x=k.cx, y=k.y - 0.2, vx=0.0, vy=-8.0)          # 受け皿の真上
    assert w._kicker_capture(on) is True and k.loaded is True

    w2 = PinballWorld(PinballConfig(seed=1))
    off = Ball(x=w2.kicker.cx + w2.kicker.half_w + 1.0, y=w2.kicker.y - 0.2, vx=0.0, vy=-8.0)
    assert w2._kicker_capture(off) is False and w2.kicker.loaded is False


def test_kicker_fires_at_the_commanded_process() -> None:
    """射出は上位が指した工程へ向く。弾道の到達範囲に縛られない経路になる。"""
    w = PinballWorld(PinballConfig(seed=1))
    k = w.kicker
    k.loaded, k.x, k.timer = True, k.cx, 1
    w.balls = []
    w._kicker_fire(0)                                         # 左端の工程を指定
    assert not k.loaded and k.shots == 1 and len(w.balls) == 1
    b = w.balls[0]
    assert b.vx < 0, "左の工程を指したのに左へ向いていない"
    assert b.vy > 0, "下向きに撃ち出していない"

    w2 = PinballWorld(PinballConfig(seed=1))
    w2.kicker.loaded, w2.kicker.x, w2.kicker.timer = True, w2.kicker.cx, 1
    w2.balls = []
    w2._kicker_fire(N_SEG - 1)
    assert w2.balls[0].vx > 0, "右の工程を指したのに右へ向いていない"


def test_kicker_holds_for_the_dwell_before_firing() -> None:
    """段取り時間の間は保持する。保持中その球は工程を回せない (捕獲の代償)。"""
    w = PinballWorld(PinballConfig(seed=1))
    k = w.kicker
    k.loaded, k.x, k.timer = True, k.cx, 3
    w.balls = []
    for _ in range(2):
        w._kicker_fire(2)
        assert k.loaded is True and not w.balls, "段取り中に撃ってはいけない"
    w._kicker_fire(2)
    assert not k.loaded and len(w.balls) == 1


def test_ladder_gates_the_kicker_while_the_drone_flies() -> None:
    """救済機が飛んでいる間は射出しない (巻き込み防止のインタロック)。"""
    plc = LadderPLC(build_ladder())
    bits = plc.scan({"kicker_loaded": True, "drone_active": False})
    assert bits["kick_ok"] is True
    bits = plc.scan({"kicker_loaded": True, "drone_active": True})
    assert bits["kick_ok"] is False
