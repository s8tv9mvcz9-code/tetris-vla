"""エージェント層のテスト: パース堅牢性、探索の妥当性、ペルソナの効き。"""

from __future__ import annotations

import numpy as np
import pytest

from tetris_vla.agents import (
    PERSONAS,
    HeuristicAgent,
    Intent,
    MockBackend,
    MockConfig,
    Observation,
    RandomAgent,
    VLAAgent,
    col_range,
    parse_intent,
    plan_placements,
)
from tetris_vla.engine import KIND_ID, Engine, EngineConfig
from tetris_vla.render import RenderConfig, render, to_png


def _obs(eng: Engine, persona_name: str = "neutral", cfg: RenderConfig | None = None) -> Observation:
    cfg = cfg or RenderConfig()
    p = eng.active[0]
    snap = eng.snapshot()
    img = render(snap, ego_pid=p.pid, cfg=cfg)
    return Observation(
        png=to_png(img),
        image=img,
        ego_pid=p.pid,
        ego_kind=p.kind,
        persona=PERSONAS[persona_name],
        tick=snap.tick,
        width=snap.width,
        height=snap.height,
        fall_period=p.fall_period,
        render_cfg=cfg,
    )


# --- パース ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expect_col,expect_err",
    [
        ('{"col": 3, "rot": 1, "why": "flat"}', 3, None),
        ('```json\n{"col": 0, "rot": 0}\n```', 0, None),
        ('ええと… {"col": 5, "rot": 2, "why": "穴を{避ける}"} です', 5, None),
        ('col: 4, rot: 1', 4, None),                      # JSON 崩れからの救出
        ('{"col": 99, "rot": 0}', 6, "out_of_range"),      # T なら最大 7 列目
        ('I cannot determine the board state.', None, "refusal"),
        ('申し訳ありませんが分かりません', None, "refusal"),
        ('the piece looks nice today', None, "parse_fail"),
        ('', None, "backend_error"),
        ('{"rot": 1}', None, "parse_fail"),
    ],
)
def test_parse_intent_handles_real_failure_modes(raw, expect_col, expect_err) -> None:
    intent, err = parse_intent(raw, "T", 10)
    assert err == expect_err
    if expect_col is None:
        assert intent is None
    else:
        assert intent is not None
        # out_of_range はクランプされる
        lo, hi = col_range("T", intent.rot, 10)
        assert lo <= intent.col <= hi
        if expect_err != "out_of_range":
            assert intent.col == expect_col


def test_parse_normalizes_redundant_rotations() -> None:
    """O ピースに rot=3 が来ても有効な回転に丸める。"""
    intent, err = parse_intent('{"col": 2, "rot": 3}', "O", 10)
    assert err is None
    assert intent.rot == 0


# --- 探索 -----------------------------------------------------------------


def test_planner_prefers_filling_a_gap_over_creating_a_hole() -> None:
    """1 マス空いた最下段に O を置ける状況で、穴を作る手を選ばない。"""
    b = np.zeros((20, 10), dtype=np.uint8)
    b[19, :] = KIND_ID["T"]
    b[18, :] = KIND_ID["T"]
    b[19, 4] = b[19, 5] = 0
    b[18, 4] = b[18, 5] = 0
    plans = plan_placements(b, "O", PERSONAS["neutral"])
    assert plans[0].col == 4, "空いた穴にぴったり嵌める手が最良になるはず"
    assert plans[0].features["holes"] == 0


def test_planner_covers_every_reachable_column() -> None:
    b = np.zeros((20, 10), dtype=np.uint8)
    plans = plan_placements(b, "T", PERSONAS["neutral"])
    cols_by_rot: dict[int, set[int]] = {}
    for p in plans:
        cols_by_rot.setdefault(p.rot, set()).add(p.col)
    for rot, cols in cols_by_rot.items():
        lo, hi = col_range("T", rot, 10)
        assert cols == set(range(lo, hi + 1))


def test_personas_change_the_chosen_placement() -> None:
    """「意思」が実際に行動を変えることの検証。左派と右派は違う列を選ぶ。"""
    b = np.zeros((20, 10), dtype=np.uint8)
    left = plan_placements(b, "O", PERSONAS["left"])[0]
    right = plan_placements(b, "O", PERSONAS["right"])[0]
    assert left.col < right.col
    assert left.col == 0 and right.col == 8


def test_nohole_persona_pays_height_to_avoid_holes() -> None:
    b = np.zeros((20, 10), dtype=np.uint8)
    b[19, :] = KIND_ID["T"]
    b[19, 3] = 0            # 深さ 1 の隙間
    b[18, 0] = KIND_ID["T"]
    neutral = plan_placements(b, "I", PERSONAS["neutral"])[0]
    nohole = plan_placements(b, "I", PERSONAS["nohole"])[0]
    assert nohole.features["holes"] <= neutral.features["holes"]


# --- エージェント ---------------------------------------------------------


def test_heuristic_only_sees_pixels() -> None:
    """ヒューリスティックは画像から盤面を復元して判断する (Engine を参照しない)。"""
    eng = Engine(EngineConfig(active_pieces=1, seed=3))
    for _ in range(30):
        eng.step({})
    obs = _obs(eng)
    d = HeuristicAgent().decide(obs)
    assert d.intent is not None
    lo, hi = col_range(obs.ego_kind, d.intent.rot, obs.width)
    assert lo <= d.intent.col <= hi
    assert d.error is None


def test_random_agent_always_returns_legal_columns() -> None:
    eng = Engine(EngineConfig(active_pieces=1, seed=4))
    agent = RandomAgent(seed=1)
    for _ in range(50):
        eng.step({})
        if not eng.active:
            continue
        obs = _obs(eng)
        d = agent.decide(obs)
        lo, hi = col_range(obs.ego_kind, d.intent.rot, obs.width)
        assert lo <= d.intent.col <= hi


def test_mock_backend_reproduces_the_configured_failure_modes() -> None:
    """モックは「良い手だけ返すダミー」ではなく、失敗も混ぜることを確認。"""
    eng = Engine(EngineConfig(active_pieces=1, seed=5))
    for _ in range(20):
        eng.step({})
    obs = _obs(eng)
    agent = VLAAgent(MockBackend(MockConfig(skill=0.5, p_malformed=0.2, p_refuse=0.2,
                                            p_out_of_range=0.2), seed=0))
    errors = set()
    for _ in range(200):
        errors.add(agent.decide(obs).error)
    assert {"parse_fail", "refusal", "out_of_range"} <= errors
    assert None in errors, "正常系も出るはず"


def test_mock_skill_one_matches_the_oracle() -> None:
    """skill=1.0・失敗率 0 なら、モックはヒューリスティックと同じ手を出す。"""
    eng = Engine(EngineConfig(active_pieces=1, seed=6))
    for _ in range(25):
        eng.step({})
    obs = _obs(eng)
    mock = VLAAgent(MockBackend(MockConfig(skill=1.0, p_malformed=0, p_refuse=0,
                                           p_out_of_range=0), seed=0))
    heur = HeuristicAgent()
    assert mock.decide(obs).intent.key() == heur.decide(obs).intent.key()


def test_mock_latency_is_positive_and_varies() -> None:
    eng = Engine(EngineConfig(active_pieces=1, seed=7))
    for _ in range(10):
        eng.step({})
    obs = _obs(eng)
    agent = VLAAgent(MockBackend(MockConfig(latency_s=1.2), seed=0))
    lats = [agent.decide(obs).latency_s for _ in range(20)]
    assert all(l > 0 for l in lats)
    assert len(set(lats)) > 1, "レイテンシはゆらぐべき"


def test_persona_reaches_the_agent_through_the_observation() -> None:
    eng = Engine(EngineConfig(active_pieces=1, seed=8))
    for _ in range(30):
        eng.step({})
    left = HeuristicAgent().decide(_obs(eng, "left")).intent
    right = HeuristicAgent().decide(_obs(eng, "right")).intent
    assert left.col <= right.col
