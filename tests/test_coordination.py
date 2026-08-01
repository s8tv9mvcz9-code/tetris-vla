"""協調モード (自律 / A2A / 独裁) と、その比較・ベンチマークのテスト。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tetris_vla.agents import (
    PERSONAS,
    HeuristicAgent,
    HeuristicDictator,
    Intent,
    JointObservation,
    MockBackend,
    MockConfig,
    PieceInfo,
    VLADictator,
    parse_joint_intents,
    plan_joint,
)
from tetris_vla.engine import KIND_ID, ActivePiece, Engine, EngineConfig
from tetris_vla.render import GHOST, RenderConfig, decode, render
from tetris_vla.runtime import RuntimeConfig, Session


def _cfgs(active=3, ticks=400, seed=5, **kw):
    e = EngineConfig(active_pieces=active, seed=seed)
    r = RuntimeConfig(max_ticks=ticks, render=RenderConfig(), compute_oracle=False, **kw)
    return e, r


# --- ゴースト (A2A の通信路) --------------------------------------------


def test_ghosts_decode_as_a_separate_plane_without_breaking_the_board() -> None:
    """予定地の描画が確定/落下ブロックのデコードを壊さないこと。

    A2A の通信をスナップショットの中に閉じ込める設計が成立する前提。
    """
    eng = Engine(EngineConfig(active_pieces=2, seed=3))
    for _ in range(40):
        eng.step({})
    snap = eng.snapshot()
    cfg = RenderConfig(show_ghosts=True)
    ghosts = [(18, 0), (18, 1), (19, 0), (19, 1)]

    plain = decode(render(snap, ego_pid=None, cfg=RenderConfig()), snap.width, snap.height)
    withg = decode(render(snap, ego_pid=None, cfg=cfg, ghost_cells=ghosts),
                   snap.width, snap.height, cfg)

    assert np.array_equal(plain.settled, withg.settled)
    assert np.array_equal(plain.active, withg.active)
    for r, c in ghosts:
        if not snap.board[r, c]:
            assert withg.ghost[r, c], "予定地がゴースト面に出ていない"
    assert not plain.ghost.any()


def test_ghosts_are_not_drawn_over_occupied_cells() -> None:
    eng = Engine(EngineConfig(active_pieces=1, seed=3))
    eng.board[19, :] = KIND_ID["T"]
    snap = eng.snapshot()
    cfg = RenderConfig(show_ghosts=True)
    dec = decode(render(snap, cfg=cfg, ghost_cells=[(19, c) for c in range(10)]),
                 snap.width, snap.height, cfg)
    assert not dec.ghost.any()
    assert (dec.settled[19] == KIND_ID["T"]).all()


def test_ghosts_are_disabled_unless_show_ghosts() -> None:
    eng = Engine(EngineConfig(active_pieces=1, seed=3))
    snap = eng.snapshot()
    cfg = RenderConfig(show_ghosts=False)
    dec = decode(render(snap, cfg=cfg, ghost_cells=[(19, 0)]), snap.width, snap.height, cfg)
    assert not dec.ghost.any()


def test_planner_avoids_declared_ghosts() -> None:
    """他人の宣言が見えたら、そこを避けた手に変わる。"""
    from tetris_vla.agents import plan_from_decoded
    from tetris_vla.render import Decoded

    settled = np.zeros((20, 10), dtype=np.uint8)
    empty = np.zeros((20, 10), dtype=np.uint8)
    no_ghost = Decoded(settled=settled, active=empty, ego=np.zeros((20, 10), bool))
    base = plan_from_decoded(no_ghost, "O", PERSONAS["neutral"])[0]

    ghost = np.zeros((20, 10), dtype=bool)
    ghost[18:20, base.col:base.col + 2] = True
    blocked = plan_from_decoded(
        Decoded(settled=settled, active=empty, ego=np.zeros((20, 10), bool), ghost=ghost),
        "O", PERSONAS["neutral"],
    )[0]
    assert blocked.col != base.col or blocked.y < base.y


# --- 独裁モード ----------------------------------------------------------


def _joint_obs(eng: Engine) -> JointObservation:
    snap = eng.snapshot()
    cfg = RenderConfig()
    img = render(snap, ego_pid=None, cfg=cfg)
    from tetris_vla.render import to_png

    infos = tuple(
        PieceInfo(pid=p.pid, kind=p.kind, persona=PERSONAS["neutral"],
                  fall_period=p.fall_period, left_col=p.left_col,
                  top_row=min(r for r, _ in p.cells()))
        for p in sorted(eng.active, key=lambda q: q.pid)
    )
    return JointObservation(png=to_png(img), image=img, pieces=infos, tick=snap.tick,
                            width=snap.width, height=snap.height, render_cfg=cfg)


def test_joint_plan_never_assigns_two_pieces_to_the_same_cell() -> None:
    """独裁者の割り当ては構造的に衝突しない。"""
    from tetris_vla.engine import ROT_CELLS, cell_span, drop_row

    settled = np.zeros((20, 10), dtype=np.uint8)
    pieces = tuple(
        PieceInfo(pid=i, kind=k, persona=PERSONAS["neutral"], fall_period=4,
                  left_col=3, top_row=i)
        for i, k in enumerate(["O", "O", "I", "T"])
    )
    intents, placements = plan_joint(settled, pieces)
    assert len(intents) == len(pieces)
    seen: set[tuple[int, int]] = set()
    for pid, pl in placements.items():
        cells = {(pl.y + dy, pl.x + dx) for dy, dx in ROT_CELLS[pieces[pid].kind][pl.rot]}
        assert not (cells & seen), f"pid={pid} の割り当てが他と重なった"
        seen |= cells


def test_dictator_agent_returns_one_intent_per_piece() -> None:
    eng = Engine(EngineConfig(active_pieces=3, seed=7))
    for _ in range(60):
        eng.step({})
    jobs = _joint_obs(eng)
    jd = HeuristicDictator().decide_joint(jobs)
    assert set(jd.intents) == {p.pid for p in jobs.pieces}
    assert not jd.errors


def test_dictator_costs_one_inference_call_for_all_pieces() -> None:
    e, r = _cfgs(active=3, ticks=400)
    auto = Session(e, r, HeuristicAgent()).run()
    dict_res = Session(e, r, HeuristicDictator()).run()
    assert dict_res.coordination == "dictator"
    assert auto.coordination == "autonomous"
    # 同じ設置数あたりの推論回数が減っている
    assert dict_res.calls_per_piece < auto.calls_per_piece


@pytest.mark.parametrize(
    "raw,expect_ok,expect_err",
    [
        ('{"moves":[{"pid":0,"col":2,"rot":0},{"pid":1,"col":6,"rot":1}]}', 2, 0),
        ('{"moves":[{"pid":0,"col":2,"rot":0}]}', 1, 1),          # 1 体分だけ欠落
        ('{"moves":[{"pid":0,"col":99,"rot":0},{"pid":1,"col":6,"rot":1}]}', 2, 1),
        ("I cannot assign the pieces.", 0, 2),
        ("garbage", 0, 2),
        ("", 0, 2),
    ],
)
def test_parse_joint_survives_partial_breakage(raw, expect_ok, expect_err) -> None:
    pieces = (
        PieceInfo(0, "O", PERSONAS["neutral"], 4, 0, 0),
        PieceInfo(1, "T", PERSONAS["neutral"], 4, 0, 0),
    )
    intents, errors = parse_joint_intents(raw, pieces, 10)
    assert len(intents) == expect_ok
    assert len(errors) == expect_err


def test_dictator_vla_mock_runs_end_to_end() -> None:
    e, r = _cfgs(active=3, ticks=400)
    agent = VLADictator(MockBackend(MockConfig(skill=0.9), seed=2))
    res = Session(e, r, agent).run()
    assert res.coordination == "dictator"
    assert res.pieces_locked > 0
    assert res.model_calls > 0


def test_dictator_mode_requires_a_joint_agent() -> None:
    e, r = _cfgs(active=2, ticks=100, coordination="dictator")
    with pytest.raises(ValueError):
        Session(e, r, HeuristicAgent())


# --- モードの強弱 --------------------------------------------------------


def test_coordination_improves_outcomes_at_equal_brainpower() -> None:
    """頭脳は同じ Dellacherie のまま、意図の共有度だけを変えて比較する。

    自律 < A2A < 独裁 の順で穴が減るのが、この PoC の中心的な主張。
    """
    e = EngineConfig(active_pieces=3, seed=1)
    base = RuntimeConfig(max_ticks=900, render=RenderConfig(), compute_oracle=False)

    auto = Session(e, base, HeuristicAgent()).run()
    a2a = Session(
        e, replace(base, coordination="a2a", render=RenderConfig(show_ghosts=True)),
        HeuristicAgent(),
    ).run()
    dictator = Session(e, base, HeuristicDictator()).run()

    assert a2a.holes_per_piece < auto.holes_per_piece, (auto.holes_per_piece, a2a.holes_per_piece)
    assert dictator.holes_per_piece < a2a.holes_per_piece
    assert dictator.lines_cleared >= auto.lines_cleared


def test_a2a_declarations_are_published_and_retired() -> None:
    e, r = _cfgs(active=3, ticks=200, coordination="a2a")
    r = replace(r, render=RenderConfig(show_ghosts=True))
    sess = Session(e, r, HeuristicAgent())
    sess.run()
    # lock されたピースの宣言は取り下げられている
    assert all(sess.engine.piece(pid) is not None for pid in sess.declared)


def test_a2a_text_channel_reaches_the_prompt() -> None:
    from tetris_vla.agents import build_prompt

    captured = []

    class _Spy:
        name = "spy"

        def decide(self, obs):
            captured.append(obs)
            from tetris_vla.agents import Decision

            return Decision(intent=Intent(col=0, rot=0), latency_s=0.0)

    e, r = _cfgs(active=3, ticks=200, coordination="a2a", a2a_text_channel=True)
    r = replace(r, render=RenderConfig(show_ghosts=True))
    Session(e, r, _Spy()).run()
    assert any(obs.declarations for obs in captured), "テキスト宣言が届いていない"
    obs = next(o for o in captured if o.declarations)
    assert "宣言" in build_prompt(obs)


def test_single_piece_makes_all_modes_identical() -> None:
    """active=1 なら協調の余地がないので結果は一致するはず (健全性チェック)。"""
    e = EngineConfig(active_pieces=1, seed=3)
    base = RuntimeConfig(max_ticks=600, render=RenderConfig(), compute_oracle=False)
    auto = Session(e, base, HeuristicAgent()).run()
    dictator = Session(e, base, HeuristicDictator()).run()
    assert auto.lines_cleared == dictator.lines_cleared
    assert auto.final_holes == dictator.final_holes


# --- ベンチマーク --------------------------------------------------------


def test_benchmark_tiers_are_monotonically_harder() -> None:
    from tetris_vla.benchmark import TIERS

    for a, b in zip(TIERS, TIERS[1:]):
        assert b.active_pieces >= a.active_pieces
        assert b.fall_period <= a.fall_period
        assert b.tick_seconds <= a.tick_seconds
        assert b.latency_budget_s < a.latency_budget_s, "許容レイテンシは単調に厳しくなるべき"


def test_score_is_bounded_and_monotone() -> None:
    from tetris_vla.benchmark import Score
    from tetris_vla.runtime import EpisodeResult

    worst = Score.of(EpisodeResult(pieces_locked=0), 20)
    assert 0 <= worst.total <= 1000

    best = EpisodeResult(pieces_locked=20, lines_cleared=8, final_holes=0, settled_on_target=20)
    best.lines_per_piece = 0.4
    best.holes_per_piece = 0.0
    best.settle_rate = 1.0
    assert Score.of(best, 20).total == pytest.approx(1000.0)

    mid = EpisodeResult(pieces_locked=10)
    mid.lines_per_piece, mid.holes_per_piece, mid.settle_rate = 0.2, 0.75, 0.5
    assert worst.total < Score.of(mid, 20).total < 1000


def test_benchmark_runs_a_single_tier() -> None:
    from tetris_vla.benchmark import TIER_BY_NAME, run_benchmark
    from tetris_vla.evaluate import AgentSpec

    bench = run_benchmark(
        AgentSpec("heuristic", "heuristic"), coordination="autonomous",
        seeds=(0,), tiers=(TIER_BY_NAME["T2"],), stop_on_fail=False,
    )
    assert len(bench.tiers) == 1
    tr = bench.tiers[0]
    assert 0 <= tr.score <= 1000
    assert tr.latency_budget_s > 0
    assert tr.mean_latency_s > 0, "ヒューリスティックの探索時間は実測される (=CPU ベンチになる)"


def test_stop_after_pieces_normalizes_episode_length() -> None:
    e = EngineConfig(active_pieces=3, seed=2)
    r = RuntimeConfig(max_ticks=2000, stop_after_pieces=12,
                      render=RenderConfig(), compute_oracle=False)
    res = Session(e, r, HeuristicAgent()).run()
    assert 12 <= res.pieces_locked <= 12 + e.active_pieces


def test_max_model_calls_caps_inference() -> None:
    e, r = _cfgs(active=3, ticks=600)
    r = replace(r, max_model_calls=10)
    res = Session(e, r, HeuristicAgent()).run()
    assert res.model_calls <= 10


# --- 途中判断の比較 ------------------------------------------------------


def test_compare_captures_scenes_with_multiple_pieces_in_flight() -> None:
    from tetris_vla.compare import capture_scenes

    scenes = capture_scenes(seed=1, active=3, n_scenes=3, first_at=100, every=80)
    assert len(scenes) >= 2
    assert all(len(s.pieces) >= 2 for s in scenes)


def test_autonomous_agreement_with_the_reference_causes_conflicts() -> None:
    """PoC の核心: 参照と一致することが、そのまま衝突の原因になる。

    全員が同じ地形を見て同じ最良手を選ぶので、狙いが重なる。
    """
    from tetris_vla.compare import (
        capture_scenes,
        conflict_count,
        disagreement,
        verdict_autonomous,
        verdict_dictator,
        verdict_oracle,
    )

    scenes = capture_scenes(seed=1, active=3, n_scenes=3, first_at=120, every=90)
    auto_conflicts = dictator_conflicts = 0
    total_disagree = 0
    for s in scenes:
        ref = verdict_oracle(s)
        auto = verdict_autonomous(s)
        dictator = verdict_dictator(s)
        assert disagreement(ref, auto) == [], "自律は地形のみの最良手と一致するはず"
        auto_conflicts += conflict_count(s, auto)
        dictator_conflicts += conflict_count(s, dictator)
        total_disagree += len(disagreement(ref, dictator))
    assert auto_conflicts > 0, "自律では狙いの重複が起きるはず"
    assert dictator_conflicts == 0, "独裁は衝突しない割り当てを作るはず"
    assert total_disagree > 0, "独裁は衝突回避のため参照から意図的にずれるはず"


def test_compare_html_is_self_contained() -> None:
    from tetris_vla.compare import capture_scenes, default_verdicts, to_html

    scenes = capture_scenes(seed=1, active=3, n_scenes=2, first_at=120, every=90)
    verdicts = [default_verdicts(s, vla_skill=0.6) for s in scenes]
    doc = to_html(scenes, verdicts)
    assert "data:image/png;base64," in doc
    assert "http://" not in doc and "https://" not in doc, "外部参照があってはいけない"
    assert doc.count("<img") >= len(scenes) * 4
