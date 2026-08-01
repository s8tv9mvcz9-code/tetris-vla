"""ランタイムのテスト: 論理時計、待ち行列、反射制御、決定性、観測の遮断。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tetris_vla.agents import (
    PERSONAS,
    Decision,
    HeuristicAgent,
    Intent,
    MockBackend,
    MockConfig,
    Observation,
    VLAAgent,
)
from tetris_vla.engine import Action, ActivePiece, Engine, EngineConfig
from tetris_vla.render import RenderConfig
from tetris_vla.runtime import RuntimeConfig, Session, on_target, reflex


def _cfgs(**kw):
    e = EngineConfig(active_pieces=kw.pop("active", 2), seed=kw.pop("seed", 0),
                     fall_period=kw.pop("fall_period", 4))
    r = RuntimeConfig(max_ticks=kw.pop("max_ticks", 120), render=RenderConfig(),
                      compute_oracle=kw.pop("compute_oracle", True), **kw)
    return e, r


# --- 反射コントローラ ------------------------------------------------------


def _act(p, intent, legal=None) -> Action:
    return reflex(p, intent, legal)[0]


def test_reflex_rotates_first_then_translates() -> None:
    p = ActivePiece(pid=0, kind="T", rot=0, x=4, y=3, fall_period=4)
    assert _act(p, Intent(col=1, rot=1)) == Action.ROT_CW
    p.rot = 1
    assert _act(p, Intent(col=1, rot=1)) == Action.LEFT
    p.x = 0
    assert _act(p, Intent(col=p.left_col, rot=1)) == Action.NONE
    assert _act(p, Intent(col=p.left_col + 1, rot=1)) == Action.RIGHT
    assert _act(p, Intent(col=p.left_col - 1, rot=1)) == Action.LEFT


def test_reflex_picks_the_shorter_rotation_direction() -> None:
    p = ActivePiece(pid=0, kind="T", rot=0, x=4, y=3, fall_period=4)
    assert _act(p, Intent(col=p.left_col, rot=3)) == Action.ROT_CCW
    assert _act(p, Intent(col=p.left_col, rot=1)) == Action.ROT_CW


def test_reflex_never_touches_gravity() -> None:
    """エージェントは操舵しかできない: 落下系のアクションは出さない。"""
    p = ActivePiece(pid=0, kind="T", rot=0, x=4, y=3, fall_period=4)
    for rot in range(4):
        for col in range(8):
            assert _act(p, Intent(col=col, rot=rot)) not in (Action.SOFT_DROP, Action.HARD_DROP)


def test_reflex_none_intent_is_noop() -> None:
    p = ActivePiece(pid=0, kind="T", rot=0, x=4, y=3, fall_period=4)
    assert _act(p, None) == Action.NONE
    assert not on_target(p, None)


def test_reflex_waits_instead_of_grinding_into_an_obstacle() -> None:
    """塞がれている方向へは突っ込まず待つ。待たされたことは報告する。"""
    p = ActivePiece(pid=0, kind="T", rot=1, x=4, y=3, fall_period=4)
    intent = Intent(col=p.left_col - 1, rot=1)
    action, blocked = reflex(p, intent, legal=lambda a: a != Action.LEFT)
    assert action == Action.NONE
    assert blocked is True


def test_reflex_routes_around_a_blocked_180_rotation() -> None:
    """180 度回したいだけなら、塞がれた向きの逆回りで到達できる。"""
    p = ActivePiece(pid=0, kind="T", rot=0, x=4, y=3, fall_period=4)
    action, blocked = reflex(p, Intent(col=p.left_col, rot=2),
                             legal=lambda a: a != Action.ROT_CW)
    assert action == Action.ROT_CCW
    assert blocked is False


def test_blocked_ticks_are_reported_in_concurrent_mode() -> None:
    e, r = _cfgs(active=4, max_ticks=400, seed=21, fall_period=2)
    e = replace(e, spawn_interval=1)
    r = replace(r, compute_oracle=False, decision_budget_per_piece=8, replan_period=4)
    res = Session(e, r, HeuristicAgent()).run()
    assert res.blocked_ticks > 0, "同時落下なら進路干渉が起きるはず"
    # 反射層は発行時点では必ず合法な手しか出さない。残る競合は
    # 「同 tick に先着した他ピースに割り込まれた」ものだけなので、ごく少数のはず。
    assert res.simultaneity_conflicts <= 0.05 * max(1, res.actions_issued), (
        res.simultaneity_conflicts, res.actions_issued
    )


def test_single_piece_mode_has_no_interference_at_all() -> None:
    e, r = _cfgs(active=1, max_ticks=400, seed=4)
    res = Session(e, r, HeuristicAgent()).run()
    assert (res.stall_ticks, res.force_locks, res.blocked_ticks,
            res.simultaneity_conflicts) == (0, 0, 0, 0)


# --- 論理時計と待ち行列 ----------------------------------------------------


class _FixedLatencyAgent:
    """レイテンシ固定のダミー。tick 換算の検証用。"""

    name = "fixed"

    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s
        self.calls = 0

    def decide(self, obs: Observation) -> Decision:
        self.calls += 1
        return Decision(intent=Intent(col=0, rot=0, why="fixed"), latency_s=self.latency_s)


def test_latency_becomes_staleness_in_ticks() -> None:
    """1.0 秒 / tick_seconds=0.25 -> 4 tick 遅れて意図が届く。"""
    e, r = _cfgs(active=1, max_ticks=80, compute_oracle=False)
    r = replace(r, tick_seconds=0.25, replan_period=1000, decision_budget_per_piece=1)
    sess = Session(e, r, _FixedLatencyAgent(1.0))
    res = sess.run()
    assert res.decisions > 0
    delivered = [rec for rec in res.records if rec.delivered]
    assert delivered
    assert all(rec.staleness == 4 for rec in delivered), [rec.staleness for rec in delivered]
    # 終了時に推論中だったものは staleness 未定義として区別される
    assert all(rec.staleness == -1 for rec in res.records if not rec.delivered)


def test_zero_latency_agent_gets_fresh_information() -> None:
    e, r = _cfgs(active=1, max_ticks=60, compute_oracle=False)
    r = replace(r, tick_seconds=0.25, replan_period=1000, decision_budget_per_piece=1)
    sess = Session(e, r, _FixedLatencyAgent(0.0))
    res = sess.run()
    assert res.decisions > 0
    assert all(rec.staleness == 0 for rec in res.records)
    assert res.undelivered_decisions == 0


def test_single_worker_serializes_concurrent_requests() -> None:
    """3 ピースが同時に判断を求めたら、2 個は待ち行列で待つ。"""
    e, r = _cfgs(active=3, max_ticks=150, compute_oracle=False)
    e = replace(e, spawn_interval=1, fall_period=8)
    r = replace(r, tick_seconds=0.2, inference_workers=1, replan_period=1000,
                decision_budget_per_piece=1)
    sess = Session(e, r, _FixedLatencyAgent(1.0))
    res = sess.run()
    assert res.decisions >= 3
    assert max(rec.queue_wait for rec in res.records) > 0, "待ち行列が発生していない"


def test_more_workers_reduce_queue_wait() -> None:
    e, r = _cfgs(active=3, max_ticks=150, compute_oracle=False)
    e = replace(e, spawn_interval=1, fall_period=8)
    base = replace(r, tick_seconds=0.2, replan_period=1000, decision_budget_per_piece=1)

    one = Session(e, replace(base, inference_workers=1), _FixedLatencyAgent(1.0)).run()
    three = Session(e, replace(base, inference_workers=3), _FixedLatencyAgent(1.0)).run()
    assert three.mean_queue_wait < one.mean_queue_wait


def test_slow_inference_produces_decisions_that_arrive_too_late() -> None:
    """推論が落下より遅いと、答えが届く前にピースが lock される。

    PoC の中心的な現象なので、指標として必ず観測できる必要がある。
    """
    e, r = _cfgs(active=1, max_ticks=200, compute_oracle=False, fall_period=1)
    e = replace(e, spawn_interval=1)
    r = replace(r, tick_seconds=0.05, replan_period=1, decision_budget_per_piece=99)
    res = Session(e, r, _FixedLatencyAgent(2.0)).run()  # 2.0s = 40 tick 相当
    assert res.late_decisions + res.undelivered_decisions + res.dropped_requests > 0
    assert res.pieces_locked > 0


def test_fast_inference_wastes_nothing() -> None:
    e, r = _cfgs(active=2, max_ticks=200, compute_oracle=False)
    res = Session(e, r, _FixedLatencyAgent(0.0)).run()
    assert res.late_decisions == 0
    assert res.undelivered_decisions == 0


def test_decision_budget_is_respected() -> None:
    e, r = _cfgs(active=2, max_ticks=200, compute_oracle=False)
    r = replace(r, decision_budget_per_piece=2, replan_period=1)
    sess = Session(e, r, _FixedLatencyAgent(0.0))
    res = sess.run()
    per_piece: dict[int, int] = {}
    for rec in res.records:
        per_piece[rec.pid] = per_piece.get(rec.pid, 0) + 1
    assert per_piece and max(per_piece.values()) <= 2


# --- 観測の遮断 ------------------------------------------------------------


def test_agent_receives_pixels_only() -> None:
    """Observation にエンジン/盤面配列への参照が含まれないことを構造的に確認。"""
    fields = set(Observation.__dataclass_fields__)
    assert "board" not in fields and "engine" not in fields and "snapshot" not in fields
    assert {"png", "image", "ego_pid", "ego_kind", "persona"} <= fields

    captured: list[Observation] = []

    class _Spy:
        name = "spy"

        def decide(self, obs):
            captured.append(obs)
            return Decision(intent=None, latency_s=0.0)

    e, r = _cfgs(active=1, max_ticks=40, compute_oracle=False)
    Session(e, r, _Spy()).run()
    assert captured
    for obs in captured:
        assert isinstance(obs.png, (bytes, bytearray)) and len(obs.png) > 0
        assert obs.image.size == r.render.image_size(e.width, e.height)


# --- エピソード全体 --------------------------------------------------------


def test_heuristic_beats_random_on_holes() -> None:
    e, r = _cfgs(active=1, max_ticks=300, seed=2)
    r = replace(r, replan_period=4, decision_budget_per_piece=8)
    from tetris_vla.agents import RandomAgent

    heur = Session(e, r, HeuristicAgent()).run()
    rand = Session(e, r, RandomAgent(seed=2)).run()
    assert heur.final_holes <= rand.final_holes
    assert heur.mean_regret is not None and heur.mean_regret < 0.05


def test_heuristic_agreement_is_near_perfect() -> None:
    """オラクルと同じ探索なので一致率はほぼ 1.0 になるべき (指標の健全性チェック)。"""
    e, r = _cfgs(active=2, max_ticks=200, seed=9)
    res = Session(e, r, HeuristicAgent()).run()
    assert res.agreement_top1 is not None
    assert res.agreement_top1 > 0.95, res.agreement_top1


def test_episode_is_deterministic_given_a_seed() -> None:
    def run() -> tuple:
        e, r = _cfgs(active=3, max_ticks=200, seed=13)
        agent = VLAAgent(MockBackend(MockConfig(), seed=13))
        res = Session(e, r, agent).run()
        return (res.lines_cleared, res.pieces_locked, res.ticks,
                res.final_holes, res.decisions, res.agreement_top1)

    assert run() == run()


def test_multi_piece_run_reports_interference() -> None:
    """同時落下では停滞や強制 lock が実際に起きる (= 干渉が存在する)。"""
    e, r = _cfgs(active=4, max_ticks=400, seed=21, fall_period=2)
    e = replace(e, spawn_interval=1, stall_limit=6)
    r = replace(r, compute_oracle=False, decision_budget_per_piece=8, replan_period=4)
    res = Session(e, r, HeuristicAgent()).run()
    assert res.pieces_locked > 5
    assert res.stall_ticks > 0, "同時落下なのに干渉が一度も起きていない"


def test_active_pieces_one_is_classic_tetris() -> None:
    e, r = _cfgs(active=1, max_ticks=300, seed=4)
    res = Session(e, r, HeuristicAgent()).run()
    assert res.stall_ticks == 0, "単一ピースでは他ピースとの干渉は起きない"
    assert res.force_locks == 0


def test_jsonl_log_roundtrips_through_replay(tmp_path) -> None:
    from tetris_vla.agents import ReplayBackend

    log = tmp_path / "run.jsonl"
    e, r = _cfgs(active=2, max_ticks=150, seed=31)
    r = replace(r, log_path=str(log))
    agent = VLAAgent(MockBackend(MockConfig(), seed=31))
    first = Session(e, r, agent).run()
    assert log.exists() and first.decisions > 0

    e2, r2 = _cfgs(active=2, max_ticks=150, seed=31)
    replayed = Session(e2, r2, VLAAgent(ReplayBackend.from_jsonl(str(log)))).run()
    assert replayed.lines_cleared == first.lines_cleared
    assert replayed.pieces_locked == first.pieces_locked
    assert replayed.agreement_top1 == first.agreement_top1
