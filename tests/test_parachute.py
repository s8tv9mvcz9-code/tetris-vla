"""パラシュート降下モードのテスト: 物理の健全性、評価軸の分離、記録の往復。

SmolVLA を要する部分は tests/test_smolvla.py に分けてある (lerobot が無くても
このファイルは通る)。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import numpy as np
import pytest

from tetris_vla.parachute import (
    Lander,
    MockFlightBackend,
    MockFlightConfig,
    PDPilot,
    ParachuteConfig,
    ParachuteWorld,
    RandomPilot,
    Thrust,
    VLAPilot,
    build_flight_prompt,
    fly,
    landing_regret,
    load_flight,
    make_terrain,
    parse_flight,
    render_flight,
    save_flight,
    score_flight,
    spawn_lander,
    target_column,
)
from tetris_vla.render import RenderConfig, decode


class Coast:
    """何もしない操縦者。評価軸が「無為」を強く評価していないかの対照。"""

    name = "coast"

    def decide(self, obs):
        from tetris_vla.parachute import FlightDecision

        return FlightDecision(Thrust.COAST, None, "")


def _world(seed: int = 0, cfg: ParachuteConfig | None = None):
    cfg = cfg or ParachuteConfig()
    board = make_terrain(cfg, seed)
    lander = spawn_lander(cfg, "O", seed, board)
    return ParachuteWorld(cfg, board, lander), cfg


# --- 物理 -----------------------------------------------------------------


def test_vertical_speed_approaches_terminal_velocity() -> None:
    """抗力があるので落下速度は終端速度に漸近し、それ以上速くならない。"""
    cfg = ParachuteConfig(wind=0.0, gust=0.0, height=200)
    board = np.zeros((200, 10), dtype=np.uint8)
    w = ParachuteWorld(cfg, board, Lander("O", 0, 4.0, 0.0, fuel=0))
    for _ in range(200):
        w.step(Thrust.COAST)
        if w.landed:
            break
    assert w.lander.vy == pytest.approx(cfg.terminal_vy, rel=0.05)
    assert w.lander.vy < cfg.terminal_vy * 1.01


def test_vent_makes_it_fall_faster_than_terminal() -> None:
    """ベントは抗力を下げるので、通常の終端速度を超える (取り返しのつかない加速)。"""
    cfg = ParachuteConfig(wind=0.0, gust=0.0, height=200)
    board = np.zeros((200, 10), dtype=np.uint8)
    w = ParachuteWorld(cfg, board, Lander("O", 0, 4.0, 0.0, fuel=0))
    for _ in range(120):
        w.step(Thrust.VENT)
    assert w.lander.vy > cfg.terminal_vy * 1.5


def test_thrust_consumes_fuel_and_stops_when_empty() -> None:
    cfg = ParachuteConfig(fuel=5, wind=0.0, gust=0.0)
    w, _ = _world(0, cfg)
    for _ in range(12):
        w.step(Thrust.RIGHT)
    assert w.lander.fuel == 0
    assert w.fuel_used == 5, "燃料切れ後は噴射しても消費しない"


def test_coast_costs_no_fuel() -> None:
    w, _ = _world(1)
    start = w.lander.fuel
    for _ in range(20):
        w.step(Thrust.COAST)
    assert w.lander.fuel == start


def test_wind_pushes_a_passive_lander_sideways() -> None:
    """風があるので「何もしない」は必ず流される = 制御に意味が生まれる。"""
    cfg = ParachuteConfig(wind=1.5, gust=0.0)
    w, _ = _world(2, cfg)
    x0 = w.lander.x
    for _ in range(25):
        if w.landed:
            break
        w.step(Thrust.COAST)
    assert w.lander.x > x0 + 0.5


def test_horizontal_speed_is_bounded_by_drag() -> None:
    cfg = ParachuteConfig(wind=0.0, gust=0.0, fuel=10**6, height=200)
    board = np.zeros((200, 10), dtype=np.uint8)
    w = ParachuteWorld(cfg, board, Lander("O", 0, 0.0, 0.0, fuel=10**6))
    for _ in range(200):
        w.step(Thrust.RIGHT)
        w.lander.x = 0.0   # 壁に貼り付かないよう位置だけ戻して速度の上限を見る
    assert w.lander.vx == pytest.approx(cfg.terminal_vx, rel=0.05)


def test_walls_stop_horizontal_motion_without_penetration() -> None:
    cfg = ParachuteConfig(wind=0.0, gust=0.0)
    w, _ = _world(3, cfg)
    w.lander.x, w.lander.vx = 0.0, 0.0
    for _ in range(15):
        if w.landed:
            break
        w.step(Thrust.LEFT)
    assert w.lander.x >= 0.0
    assert w.wall_hits > 0
    assert w.lander.vx == 0.0


def test_lander_never_overlaps_terrain() -> None:
    cfg = ParachuteConfig()
    rng = np.random.default_rng(0)
    for seed in range(8):
        w, _ = _world(seed, cfg)
        while not w.landed and not w.timeout:
            w.step(Thrust(int(rng.integers(0, 4))))
            for r, c in w.lander.cells():
                if 0 <= r < cfg.height and 0 <= c < cfg.width:
                    assert w.board[r, c] == 0, f"seed{seed}: 地形にめり込んだ"


def test_landing_records_impact_speed_then_freezes() -> None:
    cfg = ParachuteConfig(wind=0.0, gust=0.0)
    w, _ = _world(4, cfg)
    while not w.landed and not w.timeout:
        w.step(Thrust.COAST)
    assert w.landed
    assert w.impact_vy > 0
    assert w.lander.vy == 0.0 and w.lander.vx == 0.0
    n = w.tick
    w.step(Thrust.RIGHT)
    assert w.tick == n, "着地後は進まない"


def test_every_seed_terminates() -> None:
    cfg = ParachuteConfig()
    for seed in range(12):
        w, _ = _world(seed, cfg)
        while not w.landed and not w.timeout:
            w.step(Thrust.COAST)
        assert w.landed or w.timeout
        assert w.tick <= cfg.max_ticks


def test_spawn_is_placed_away_from_the_target() -> None:
    """真上に落とすと「何もしない」が正解になってしまうので、必ず横断させる。"""
    cfg = ParachuteConfig()
    gaps = []
    for seed in range(12):
        board = make_terrain(cfg, seed)
        for kind in "IOTSZJL":
            L = spawn_lander(cfg, kind, seed, board)
            tgt = target_column(board, kind, L.rot)
            if tgt is not None:
                gaps.append(abs(L.col - tgt))
    assert statistics.fmean(gaps) >= 3.0, gaps


def test_flight_is_deterministic_for_a_seed() -> None:
    def run():
        r = fly(PDPilot(), ParachuteConfig(), seed=7, keep_png=False)
        return (r.landing_col, r.ticks, round(r.impact_vy, 6), r.fuel_left, r.score)

    assert run() == run()


# --- 評価軸 ---------------------------------------------------------------


def test_score_axes_are_bounded() -> None:
    cfg = ParachuteConfig()
    crash = {"timeout": True, "landed": False, "col_error": None, "impact_vy": 9.0,
             "impact_vx": 9.0, "holes_created": 9, "fuel_left": 0, "regret": 1.0}
    total, parts = score_flight(crash, cfg)
    assert total == 0.0 and all(v == 0.0 for v in parts.values())

    best = {"timeout": False, "landed": True, "col_error": 0, "impact_vy": 0.0,
            "impact_vx": 0.0, "holes_created": 0, "fuel_left": cfg.fuel, "regret": 0.0}
    total, parts = score_flight(best, cfg)
    assert total == pytest.approx(1000.0)
    assert all(0.0 <= v <= 1.0 for v in parts.values())


def test_doing_nothing_does_not_beat_controlling() -> None:
    """「無為」が制御に勝ってしまうと、この課題は評価装置として無意味になる。"""
    cfg = ParachuteConfig()
    pd = [fly(PDPilot(), cfg, seed=s, keep_png=False).score for s in range(12)]
    coast = [fly(Coast(), cfg, seed=s, keep_png=False).score for s in range(12)]
    rand = [fly(RandomPilot(seed=s), cfg, seed=s, keep_png=False).score for s in range(12)]
    assert statistics.fmean(pd) > statistics.fmean(coast) + 100
    assert statistics.fmean(pd) > statistics.fmean(rand) + 100


def test_pd_pilot_reaches_the_reference_column() -> None:
    cfg = ParachuteConfig()
    errs = [fly(PDPilot(), cfg, seed=s, keep_png=False).col_error for s in range(12)]
    assert all(e is not None for e in errs)
    assert statistics.fmean(errs) < 0.6, errs


def test_landing_regret_is_zero_at_the_reference_placement() -> None:
    cfg = ParachuteConfig()
    board = make_terrain(cfg, 5)
    L = spawn_lander(cfg, "O", 5, board)
    tgt = target_column(board, "O", L.rot)
    assert landing_regret(board, "O", L.rot, tgt) == pytest.approx(0.0)
    far = 0 if tgt > 4 else 9
    assert landing_regret(board, "O", L.rot, far) >= 0.0


# --- レイテンシが成績に効くこと -------------------------------------------


def _mock_runs(latency: float, seeds: range, skill: float = 1.0):
    cfg = ParachuteConfig()
    out = []
    for s in seeds:
        be = MockFlightBackend(
            MockFlightConfig(skill=skill, latency_s=latency, latency_sigma=0.05,
                             p_malformed=0, p_refuse=0), seed=s)
        out.append(fly(VLAPilot(be), cfg, seed=s, keep_png=False))
    return out


def test_latency_is_charged_as_drift() -> None:
    """遅い判断は、その分だけ機体を流す。VLA/VLM 比較の土台になる性質。

    1 seed だと「判断が減って燃料が残る」ぶんで逆転しうるので、
    統計的な主張として複数 seed の平均で検定する。
    """
    seeds = range(14)
    fast = _mock_runs(0.05, seeds)
    slow = _mock_runs(4.0, seeds)
    assert statistics.fmean(r.decisions for r in slow) < \
        statistics.fmean(r.decisions for r in fast), "遅いほど判断回数が減る"
    assert statistics.fmean(r.score for r in slow) + 80 < \
        statistics.fmean(r.score for r in fast), (
            statistics.fmean(r.score for r in slow), statistics.fmean(r.score for r in fast))


def test_latency_degrades_score_monotonically_in_aggregate() -> None:
    """レイテンシを上げていくと平均点は単調に近い形で下がる。"""
    seeds = range(12)
    means = [statistics.fmean(r.score for r in _mock_runs(l, seeds))
             for l in (0.05, 1.0, 4.0)]
    assert means[0] > means[1] > means[2], means


def test_slow_pilot_still_lands() -> None:
    """レイテンシが大きくても落下自体は続くので、必ず決着する。"""
    cfg = ParachuteConfig()
    r = fly(VLAPilot(MockFlightBackend(
        MockFlightConfig(latency_s=8.0, latency_sigma=0.05), seed=2)), cfg, seed=2, keep_png=False)
    assert r.landed or r.timeout


# --- 応答のパース ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expect_act,expect_err",
    [
        ('{"thrust":"left","target_col":3,"why":"flat"}', Thrust.LEFT, None),
        ('{"thrust":"right","target_col":0}', Thrust.RIGHT, None),
        ('{"thrust":"none"}', Thrust.COAST, None),
        ('{"thrust":"vent","target_col":5}', Thrust.VENT, None),
        ('```json\n{"thrust":"left","target_col":2}\n```', Thrust.LEFT, None),
        ('thrust: right, blah', Thrust.RIGHT, None),
        ('{"thrust":"left","target_col":99}', Thrust.LEFT, "out_of_range"),
        ('I cannot determine the terrain.', None, "refusal"),
        ('the sky is nice', None, "parse_fail"),
        ('', None, "backend_error"),
        ('{"thrust":"sideways"}', None, "parse_fail"),
    ],
)
def test_parse_flight_handles_real_failure_modes(raw, expect_act, expect_err) -> None:
    d, err = parse_flight(raw, 10)
    assert err == expect_err
    if expect_act is None:
        assert d is None
    else:
        assert d.action == expect_act
        if d.target_col is not None:
            assert 0 <= d.target_col < 10


def test_unparseable_response_falls_back_to_not_burning_fuel() -> None:
    cfg = ParachuteConfig()

    class Broken:
        name = "broken"

        def generate(self, prompt, obs):
            return "???", 0.1

    d = VLAPilot(Broken()).decide(_obs(cfg))
    assert d.action == Thrust.COAST, "読めない時に噴射すると燃料を捨てることになる"
    assert d.error == "parse_fail"


def _obs(cfg: ParachuteConfig):
    from tetris_vla.parachute import FlightObservation, to_png

    w, _ = _world(0, cfg)
    for _ in range(5):
        w.step(Thrust.COAST)
    img = render_flight(w, cfg, RenderConfig())
    L = w.lander
    return FlightObservation(png=to_png(img), image=img, kind=L.kind, rot=L.rot, x=L.x, y=L.y,
                             vx=L.vx, vy=L.vy, fuel=L.fuel, tick=w.tick, t=w.t,
                             width=cfg.width, height=cfg.height, cfg=cfg,
                             render_cfg=RenderConfig())


@pytest.mark.parametrize("lang", ["en", "ja"])
def test_prompt_contains_the_sensor_values(lang) -> None:
    obs = _obs(ParachuteConfig())
    p = build_flight_prompt(obs, lang)
    assert str(obs.fuel) in p
    assert "thrust" in p and "target_col" in p
    assert "vent" in p


# --- 描画 -----------------------------------------------------------------


def test_terrain_stays_readable_from_pixels_while_flying() -> None:
    """地形は画像から厳密に読める。機体や軌跡を重ねてもそこは壊れない。"""
    cfg = ParachuteConfig()
    rc = RenderConfig()
    w, _ = _world(6, cfg)
    for _ in range(18):
        w.step(Thrust.RIGHT)
    dec = decode(render_flight(w, cfg, rc, Thrust.RIGHT), cfg.width, cfg.height, rc)
    covered = {(r, c) for r, c in w.lander.cells()}
    for r in range(cfg.height):
        for c in range(cfg.width):
            if (r, c) in covered:
                continue   # 機体が真上を覆っているセルは見えなくて当然
            assert dec.settled[r, c] == w.board[r, c], f"({r},{c}) の地形読み取りが不一致"


def test_pd_pilot_only_uses_pixels_for_the_terrain() -> None:
    """PD も VLA と同じ画像を見る。違うのは決め方だけ。"""
    cfg = ParachuteConfig()
    obs = _obs(cfg)
    d = PDPilot().decide(obs)
    assert d.target_col is not None
    assert 0 <= d.target_col < cfg.width


# --- 記録 -----------------------------------------------------------------


def test_flight_log_roundtrip(tmp_path) -> None:
    cfg = ParachuteConfig()
    res = fly(PDPilot(), cfg, seed=3)
    path = tmp_path / "flight.jsonl"
    save_flight(res, str(path), cfg)
    data = load_flight(str(path))

    assert data["end"]["score"] == res.score
    assert len(data["decisions"]) == len(res.decision_log)
    assert len(data["frames"]) == len(res.frames)
    assert np.array_equal(data["board"], res.board)
    assert data["decisions"][0]["png"], "モデルに渡した画像そのものが残っていること"
    assert "state" in data["decisions"][0]


def test_svg_and_html_are_self_contained(tmp_path) -> None:
    import re

    from tetris_vla.svganim import flight_html, flight_svg

    cfg = ParachuteConfig()
    res = fly(PDPilot(), cfg, seed=3)
    path = tmp_path / "f.jsonl"
    save_flight(res, str(path), cfg)
    data = load_flight(str(path))

    svg = flight_svg(data)
    import xml.etree.ElementTree as ET

    ET.fromstring(svg)                       # 妥当な XML か
    assert "<animateTransform" in svg        # 機体が動くか
    assert "<script" not in svg
    doc = flight_html(data, svg)
    assert "data:image/png;base64," in doc   # 入力画像が埋め込まれているか
    for text in (svg, doc):
        refs = re.findall(r'(?:src|href|xlink:href)\s*=\s*["\']([^"\']+)', text)
        assert not [r for r in refs if r.startswith(("http://", "https://", "//"))]


def test_live_flight_prints_inputs_and_decisions() -> None:
    import io

    buf = io.StringIO()
    from tetris_vla.parachute import live_flight

    res = live_flight(PDPilot(), ParachuteConfig(), seed=3, decision_period=12, stream=buf)
    out = buf.getvalue()
    assert "[入力]" in out and "[解釈]" in out and "[コスト]" in out
    assert "漂流" in out
    assert f"{res.score:.0f}点" in out
