"""レンダラ/デコーダの往復同一性テスト。

これが通らないと「全エージェントが同じピクセルを見ている」という PoC の公平性主張が
検証されていないことになる。
"""

from __future__ import annotations

import numpy as np
import pytest

from tetris_vla.engine import KIND_ID, KINDS, Action, ActivePiece, Engine, EngineConfig
from tetris_vla.render import (
    BASE,
    BG,
    EGO_RING,
    FOG,
    GRID,
    HEADER_BG,
    HEADER_FG,
    SETTLED,
    UNKNOWN_KIND,
    RenderConfig,
    decode,
    render,
    to_ascii,
    to_png,
    to_text_board,
)


def test_all_cell_colors_are_pairwise_distinct() -> None:
    """中心ピクセル 1 点でデコードするので、色が衝突していたら即破綻する。"""
    colors = [BG, FOG, EGO_RING, GRID, HEADER_BG, HEADER_FG]
    colors += list(BASE.values()) + list(SETTLED.values())
    assert len(set(colors)) == len(colors), "セル色が衝突している"


def test_cell_px_lower_bound_is_enforced() -> None:
    with pytest.raises(ValueError):
        RenderConfig(cell_px=6)


def _random_snapshot(seed: int, width: int = 10, height: int = 20):
    rng = np.random.default_rng(seed)
    eng = Engine(EngineConfig(width=width, height=height, active_pieces=3, seed=seed))
    # 確定ブロックをランダムに敷く (満杯行は作らない)
    board = np.zeros((height, width), dtype=np.uint8)
    for r in range(height // 2, height):
        for c in range(width):
            if rng.random() < 0.55:
                board[r, c] = int(rng.integers(1, 8))
        board[r, int(rng.integers(0, width))] = 0  # 必ず 1 マス空ける
    eng.board = board

    # 空きセルに落下ピースを置く
    eng.active.clear()
    placed = 0
    pid = 0
    while placed < 3 and pid < 60:
        kind = KINDS[int(rng.integers(0, 7))]
        rot = int(rng.integers(0, 4))
        x = int(rng.integers(0, width - 3))
        y = int(rng.integers(0, height - 4))
        if eng.can_place(kind, rot, x, y, None):
            eng.active.append(ActivePiece(pid=pid, kind=kind, rot=rot, x=x, y=y, fall_period=3))
            placed += 1
        pid += 1
    return eng.snapshot()


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("cell_px", [8, 12, 16])
def test_render_decode_roundtrip_is_exact(seed: int, cell_px: int) -> None:
    snap = _random_snapshot(seed)
    cfg = RenderConfig(cell_px=cell_px)
    ego = snap.active[0].pid if snap.active else None
    img = render(snap, ego_pid=ego, cfg=cfg)
    dec = decode(img, snap.width, snap.height, cfg)

    assert np.array_equal(dec.settled, snap.board), "確定ブロックの復元が一致しない"

    expected_active = np.zeros_like(snap.board)
    expected_ego = np.zeros(snap.board.shape, dtype=bool)
    for p in snap.active:
        for r, c in p.cells():
            if 0 <= r < snap.height and 0 <= c < snap.width:
                expected_active[r, c] = KIND_ID[p.kind]
                if p.pid == ego:
                    expected_ego[r, c] = True
    assert np.array_equal(dec.active, expected_active), "落下ブロックの復元が一致しない"
    assert np.array_equal(dec.ego, expected_ego), "自機マーカーの復元が一致しない"


def test_image_size_is_small_enough_for_a_local_vlm() -> None:
    snap = _random_snapshot(0)
    img = render(snap, cfg=RenderConfig(cell_px=12))
    assert (img.width, img.height) == (120, 254)  # 120x240 + header 14
    assert len(to_png(img)) < 6000, "prefill を抑えるため PNG は小さく保つ"


def test_fog_hides_other_piece_kinds_but_not_positions() -> None:
    snap = _random_snapshot(3)
    ego = snap.active[0].pid
    cfg = RenderConfig(fog_other=True)
    dec = decode(render(snap, ego_pid=ego, cfg=cfg), snap.width, snap.height, cfg)
    others = [p for p in snap.active if p.pid != ego]
    assert others
    for p in others:
        for r, c in p.cells():
            if 0 <= r < snap.height and 0 <= c < snap.width:
                assert dec.active[r, c] == UNKNOWN_KIND, "霧では種類が見えないはず"
                assert dec.occupancy[r, c], "位置は見えるはず"
    # 自機は霧の影響を受けない
    for p in snap.active:
        if p.pid == ego:
            for r, c in p.cells():
                if 0 <= r < snap.height and 0 <= c < snap.width:
                    assert dec.active[r, c] == KIND_ID[p.kind]


def test_header_band_is_ignored_by_decoder() -> None:
    """ヘッダ有無でデコード結果が変わらない = ラベルは公平性を壊さない。"""
    snap = _random_snapshot(5)
    a = RenderConfig(labels=True)
    b = RenderConfig(labels=False)
    da = decode(render(snap, ego_pid=None, cfg=a), snap.width, snap.height, a)
    db = decode(render(snap, ego_pid=None, cfg=b), snap.width, snap.height, b)
    assert np.array_equal(da.settled, db.settled)
    assert np.array_equal(da.active, db.active)


def test_ascii_and_text_views_render() -> None:
    snap = _random_snapshot(1)
    ego = snap.active[0].pid
    txt = to_ascii(snap, ego)
    assert f"t={snap.tick}" in txt
    assert txt.count("\n") >= snap.height
    dec = decode(render(snap, ego_pid=ego), snap.width, snap.height)
    tb = to_text_board(dec)
    assert "@" in tb, "自機はテキスト表現でも識別できる"
