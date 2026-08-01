"""複数機同時降下のテスト。モデルを使わない部分（物理・記録・可視化）を検証する。

qwen / SmolVLA の実機呼び出しは scripts/flock_demo.py 側の仕事なので、
ここではダミーの指令と行動列で配管が通ることを確かめる。
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tetris_vla.flock import (
    FlockConfig,
    FlockWorld,
    ModelCall,
    QwenCommander,
    flock_ascii,
    load_flock,
    render_flock,
    save_flock,
    FlockResult,
)
from tetris_vla.parachute import Lander, ParachuteConfig, Thrust, make_terrain
from tetris_vla.render import RenderConfig, decode


def _world(n=3, seed=2):
    cfg = ParachuteConfig()
    fcfg = FlockConfig(n_landers=n, spawn_gap_ticks=0, max_ticks=200)
    w = FlockWorld(cfg, fcfg, make_terrain(cfg, seed), seed=seed)
    for _ in range(n):
        w._maybe_spawn()
        w._last_spawn = -10**9
    return w, cfg, fcfg


# --- 物理 -----------------------------------------------------------------


def test_multiple_landers_spawn_and_are_distinct() -> None:
    w, _, _ = _world(3)
    assert len(w.landers) == 3
    assert len({id(L) for L in w.landers.values()}) == 3


def test_landers_never_overlap_each_other_or_terrain() -> None:
    w, cfg, _ = _world(3, seed=5)
    rng = np.random.default_rng(0)
    while not w.done:
        w.step({pid: Thrust(int(rng.integers(0, 4))) for pid in w.landers})
        seen: set[tuple[int, int]] = set()
        for pid, L in w.landers.items():
            for r, c in w._cells(L, L.x, L.y):
                assert (r, c) not in seen, "機体同士が重なった"
                seen.add((r, c))
                if 0 <= r < cfg.height and 0 <= c < cfg.width:
                    assert w.board[r, c] == 0, "地形にめり込んだ"


def test_a_landed_piece_becomes_terrain_for_the_others() -> None:
    """先に着いた機はその場で地形になるので、後続の着地面が変わる。"""
    w, cfg, _ = _world(3, seed=4)
    before = int((w.board > 0).sum())
    while not w.done:
        w.step({})
    assert w.landed, "誰も着地していない"
    assert int((w.board > 0).sum()) > before


def test_every_seed_terminates() -> None:
    for seed in range(6):
        w, _, fcfg = _world(3, seed)
        while not w.done:
            w.step({})
        assert len(w.landed) == fcfg.n_landers or w.tick >= fcfg.max_ticks


def test_landing_records_impact_and_holes() -> None:
    w, _, _ = _world(2, seed=7)
    while not w.done:
        w.step({})
    for info in w.landed.values():
        assert {"kind", "col", "impact_vy", "holes_created", "soft"} <= set(info)
        assert info["holes_created"] >= 0


# --- 描画 -----------------------------------------------------------------


def test_terrain_is_still_decodable_with_several_landers_drawn() -> None:
    """複数機と軌跡を重ねても、地形の読み取りは壊れない。"""
    w, cfg, _ = _world(3, seed=6)
    for _ in range(14):
        w.step({})
    rc = RenderConfig()
    dec = decode(render_flock(w, cfg, rc), cfg.width, cfg.height, rc)
    covered = {c for L in w.landers.values() for c in w._cells(L, L.x, L.y)}
    for r in range(cfg.height):
        for c in range(cfg.width):
            if (r, c) in covered:
                continue
            assert dec.settled[r, c] == w.board[r, c]


def test_ego_ring_marks_exactly_one_lander() -> None:
    """VLA には「自分がどれか」を白枠で示す。機体は小数座標に描かれるので、
    格子で探るのではなく白画素の増分で確かめる。"""
    from tetris_vla.render import EGO_RING

    w, cfg, _ = _world(3, seed=6)
    for _ in range(10):
        w.step({})
    rc = RenderConfig()

    def white(img) -> int:
        px = np.asarray(img.convert("RGB"))
        return int(((px[..., 0] == EGO_RING[0]) & (px[..., 1] == EGO_RING[1])
                    & (px[..., 2] == EGO_RING[2])).sum())

    plain = white(render_flock(w, cfg, rc, ego_pid=None, labels=False))
    counts = [white(render_flock(w, cfg, rc, ego_pid=pid, labels=False))
              for pid in sorted(w.landers)]
    assert plain == 0, "ego 指定なしでは白枠は出ない"
    assert all(c > 0 for c in counts), "指定した機に白枠が出ていない"
    # どの機を指定してもリング 1 個ぶん (機体の形は同じ 4 セル) で、桁は揃う
    assert max(counts) < 4 * rc.cell_px * rc.cell_px, "白枠が塗り潰しになっている"


def test_ascii_view_shows_targets() -> None:
    w, _, _ = _world(2, seed=3)
    txt = flock_ascii(w, {sorted(w.landers)[0]: 4})
    assert "VLM が指令した目標列" in txt
    assert txt.count("\n") >= 20


# --- 上位 VLM のプロンプトと応答パース ------------------------------------


def test_commander_prompt_lists_pieces_in_order() -> None:
    w, _, _ = _world(3, seed=2)
    p = QwenCommander(unload_after=False).build_prompt(w)
    assert "IN THE SAME ORDER" in p
    for i in range(1, len(w.landers) + 1):
        assert f"  {i}. shape" in p
    assert '{"cols":[' in p


@pytest.mark.parametrize(
    "raw,expect_len,expect_err",
    [
        ('{"cols":[1,5,8]}', 3, None),
        ('{"columns":[0,0,9]}', 3, None),
        ('{"cols":[1,5]}', 2, None),            # 足りなくても前から当てる
        ('{"cols":[1,5,8,9,9]}', 3, None),      # 多い分は捨てる
        ('cols are 2, 4 and 6', 3, "salvaged"), # 崩れても数字を拾う
        ('no idea', 0, "parse_fail"),
    ],
)
def test_commander_parses_positional_answers(raw, expect_len, expect_err, monkeypatch) -> None:
    """pid を書かせず位置で対応させる。存在しない pid を作られる事故を防ぐため。"""
    w, _, _ = _world(3, seed=2)
    cmd = QwenCommander(unload_after=False)

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"response": raw}

    monkeypatch.setattr(cmd._client, "post", lambda *a, **k: _Resp())
    out, prompt, got, err = cmd(w, b"")
    assert len(out) == expect_len
    assert err == expect_err
    assert all(0 <= v < w.cfg.width for v in out.values())
    assert set(out) <= set(w.landers), "存在しない機に指令が付いてはいけない"


def test_commander_reports_backend_failure_without_raising(monkeypatch) -> None:
    w, _, _ = _world(2, seed=2)
    cmd = QwenCommander(unload_after=False)

    def _boom(*a, **k):
        raise RuntimeError("接続不可")

    monkeypatch.setattr(cmd._client, "post", _boom)
    out, _, _, err = cmd(w, b"")
    assert out == {} and err.startswith("backend_error")


# --- 記録と可視化 ---------------------------------------------------------


def _fake_result(seed=2):
    w, cfg, fcfg = _world(3, seed)
    calls = []
    seq = 0
    while not w.done:
        if w.tick % 20 == 0 and w.landers:
            seq += 1
            calls.append(ModelCall(
                seq=seq, model="qwen2.5vl:3b", role="どこへ", tick=w.tick,
                t=round(w.t, 2), pids=sorted(w.landers), latency_s=2.0, drift_ticks=4,
                prompt="p", raw='{"cols":[1,5,8]}',
                parsed={"assign": {pid: 3 for pid in sorted(w.landers)}},
                png_b64=None, inputs={"roster": {}}))
        if w.tick % 13 == 0 and w.landers:
            seq += 1
            pid = sorted(w.landers)[0]
            calls.append(ModelCall(
                seq=seq, model="smolvla", role="どうやって", tick=w.tick, t=round(w.t, 2),
                pids=[pid], latency_s=0.4, drift_ticks=1, prompt="task",
                raw='{"chunk_len":50}', parsed={"target_col": 3, "plan12": [1, 0, 2, 3] * 3},
                png_b64=None, inputs={"state6": [0.0] * 6}))
        w.step({})
    per = {pid: {**info, "regret": 0.2, "oracle_col": 3, "commanded_col": 3, "col_error": 1}
           for pid, info in w.landed.items()}
    summary = {"landed": len(w.landed), "n_landers": fcfg.n_landers, "ticks": w.tick,
               "sim_seconds": round(w.t, 2), "slowmo": fcfg.slowmo, "holes_created": 1,
               "soft_landings": len(w.landed), "mean_regret": 0.2, "mean_col_error": 1.0,
               "vlm_calls": sum(1 for c in calls if c.role == "どこへ"),
               "vla_calls": sum(1 for c in calls if c.role == "どうやって"),
               "vlm_latency_mean": 2.0, "vla_latency_mean": 0.4, "wall_seconds": 10.0,
               "per_lander": per}
    return FlockResult(seeds=seed, frames=w.frames, calls=calls, landed=w.landed,
                       board0=make_terrain(cfg, seed), board1=w.board,
                       spawn_order=w.spawn_order, targets_history=[], summary=summary), cfg, fcfg


def test_flock_log_roundtrip(tmp_path) -> None:
    res, cfg, fcfg = _fake_result()
    path = tmp_path / "flock.jsonl"
    save_flock(res, str(path), cfg, fcfg)
    data = load_flock(str(path))
    assert len(data["calls"]) == len(res.calls)
    assert len(data["frames"]) == len(res.frames)
    assert np.array_equal(data["board"], res.board0)
    assert data["end"]["landed"] == res.summary["landed"]
    assert data["start"]["spawn_order"]


def test_flock_svg_and_html_are_self_contained(tmp_path) -> None:
    from tetris_vla.flockviz import flock_html, flock_svg

    res, cfg, fcfg = _fake_result()
    path = tmp_path / "f.jsonl"
    save_flock(res, str(path), cfg, fcfg)
    data = load_flock(str(path))

    svg = flock_svg(data)
    ET.fromstring(svg)
    assert svg.count("<animateTransform") == fcfg.n_landers, "各機が動くこと"
    doc = flock_html(data, svg)
    assert "依頼タイミング" in doc and "各呼び出しの入出力" in doc
    for text in (svg, doc):
        refs = re.findall(r'(?:src|href|xlink:href)\s*=\s*["\']([^"\']+)', text)
        assert not [r for r in refs if r.startswith(("http://", "https://", "//"))]
        assert "<script" not in text


def test_html_shows_both_models_and_their_timing(tmp_path) -> None:
    from tetris_vla.flockviz import flock_html, flock_svg

    res, cfg, fcfg = _fake_result()
    path = tmp_path / "f.jsonl"
    save_flock(res, str(path), cfg, fcfg)
    data = load_flock(str(path))
    doc = flock_html(data, flock_svg(data))
    assert "qwen2.5vl:3b" in doc and "smolvla" in doc
    assert "どこへ" in doc and "どうやって" in doc
    assert "漂流" in doc
