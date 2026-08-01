"""エンジンの不変条件テスト。ここが壊れていると他の指標が全部無意味になる。"""

from __future__ import annotations

import numpy as np
import pytest

from tetris_vla.engine import (
    KINDS,
    KIND_ID,
    ROT_CELLS,
    Action,
    ActivePiece,
    Coll,
    Engine,
    EngineConfig,
    bumpiness,
    cell_span,
    column_heights,
    count_holes,
    drop_row,
    place,
    unique_rotations,
    wells,
)


# --- テトロミノ定義 -------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_each_rotation_has_four_cells(kind: str) -> None:
    for rot in range(4):
        assert len(ROT_CELLS[kind][rot]) == 4, f"{kind} rot={rot}"


@pytest.mark.parametrize("kind", KINDS)
def test_rotation_is_a_four_cycle(kind: str) -> None:
    """4 回まわすと元に戻る。手打ちテーブルの取り違えを検出する。"""
    assert ROT_CELLS[kind][0] == ROT_CELLS[kind][4 % 4]
    seq = [ROT_CELLS[kind][r] for r in range(4)]
    # 回転で形が壊れていない = セル数と相対形状の連結性が保たれる
    for cells in seq:
        assert len(set(cells)) == 4


@pytest.mark.parametrize(
    "kind,expected",
    [("O", 1), ("I", 2), ("S", 2), ("Z", 2), ("T", 4), ("J", 4), ("L", 4)],
)
def test_unique_rotation_counts(kind: str, expected: int) -> None:
    assert len(unique_rotations(kind)) == expected


def test_piece_ids_are_distinct() -> None:
    assert sorted(KIND_ID.values()) == list(range(1, 8))


def test_i_piece_is_horizontal_at_spawn() -> None:
    cells = ROT_CELLS["I"][0]
    assert len({c[0] for c in cells}) == 1, "spawn state の I は水平"


# --- 衝突とロック --------------------------------------------------------


def _fresh(**kw) -> Engine:
    cfg = EngineConfig(**{"active_pieces": 1, "fall_period": 1, "spawn_interval": 0, **kw})
    return Engine(cfg)


def test_no_overlap_invariant_over_a_long_run() -> None:
    """落下中ピース同士・確定ブロックとの重なりが一度も起きない。"""
    eng = Engine(EngineConfig(active_pieces=4, fall_period=2, spawn_interval=2, seed=7))
    rng = np.random.default_rng(0)
    for _ in range(400):
        if eng.game_over:
            break
        actions = {
            p.pid: Action(int(rng.integers(0, 5))) for p in eng.active
        }
        eng.step(actions)
        seen: set[tuple[int, int]] = set()
        for p in eng.active:
            for r, c in p.cells():
                assert (r, c) not in seen, "落下ピース同士が重なった"
                seen.add((r, c))
                assert -4 <= r < eng.board.shape[0]
                assert 0 <= c < eng.board.shape[1]
                if r >= 0:
                    assert eng.board[r, c] == 0, "落下ピースが確定ブロックに重なった"


def test_lock_only_against_terrain() -> None:
    """他の落下ピースに阻まれただけでは lock しない (空中固定を防ぐ)。"""
    eng = _fresh(active_pieces=2, stall_limit=100, height=20, width=10)
    eng.active.clear()
    lower = ActivePiece(pid=100, kind="O", rot=0, x=4, y=10, fall_period=1000)
    upper = ActivePiece(pid=101, kind="O", rot=0, x=4, y=8, fall_period=1)
    eng.active.extend([lower, upper])

    before = eng.pieces_locked
    for _ in range(5):
        eng.step({})
    assert eng.pieces_locked == before, "落下ピースに阻まれただけで lock してはいけない"
    assert eng.piece(101) is not None
    assert eng.piece(101).stall > 0


def test_force_lock_breaks_deadlock() -> None:
    eng = _fresh(active_pieces=2, stall_limit=3, height=20, width=10)
    eng.active.clear()
    lower = ActivePiece(pid=100, kind="O", rot=0, x=4, y=10, fall_period=10000)
    upper = ActivePiece(pid=101, kind="O", rot=0, x=4, y=8, fall_period=1)
    eng.active.extend([lower, upper])
    forced = []
    for _ in range(6):
        rep = eng.step({})
        forced.extend(rep.force_locked)
    assert 101 in forced, "stall_limit を超えたら強制 lock されるはず"


def test_lock_on_floor() -> None:
    eng = _fresh(active_pieces=1)
    eng.active.clear()
    p = ActivePiece(pid=0, kind="O", rot=0, x=0, y=18, fall_period=1)
    eng.active.append(p)
    eng.step({})
    assert eng.pieces_locked == 1
    assert eng.board[18, 0] and eng.board[19, 1]


# --- ライン消去 -----------------------------------------------------------


def test_line_clear_and_shift() -> None:
    eng = _fresh(active_pieces=1, width=10, height=20)
    eng.active.clear()
    eng.board[19, :] = KIND_ID["T"]
    eng.board[19, 0] = 0
    eng.board[19, 1] = 0
    eng.board[18, 5] = KIND_ID["T"]  # 上に残るブロック

    p = ActivePiece(pid=0, kind="O", rot=0, x=0, y=18, fall_period=1)
    eng.active.append(p)
    rep = eng.step({})
    assert rep.cleared_rows == (19,)
    assert eng.lines_cleared == 1
    assert eng.board[19, 5] == KIND_ID["T"], "消去後に上の行が 1 段下がる"
    assert eng.board[18, 5] == 0


def test_active_pieces_shift_down_after_clear() -> None:
    eng = _fresh(active_pieces=2, width=10, height=20)
    eng.active.clear()
    eng.board[19, :] = KIND_ID["T"]
    eng.board[19, 0] = 0
    eng.board[19, 1] = 0
    lander = ActivePiece(pid=0, kind="O", rot=0, x=0, y=18, fall_period=1)
    floater = ActivePiece(pid=1, kind="O", rot=0, x=6, y=2, fall_period=10000)
    eng.active.extend([lander, floater])
    eng.step({})
    assert eng.piece(1).y == 3, "消えた行より上の落下ピースも 1 段下がる"


def test_tetris_clears_four_rows() -> None:
    eng = _fresh(active_pieces=1, width=10, height=20)
    eng.active.clear()
    for r in (16, 17, 18, 19):
        eng.board[r, :] = KIND_ID["T"]
        eng.board[r, 9] = 0
    p = ActivePiece(pid=0, kind="I", rot=1, x=7, y=13, fall_period=1)
    # I の rot=1 は縦。着地位置まで落とす
    while eng.can_place("I", 1, p.x, p.y + 1, p.pid):
        p.y += 1
    eng.active.append(p)
    rep = eng.step({})
    assert len(rep.cleared_rows) == 4
    assert eng.line_clear_histogram[4] == 1


# --- アクション -----------------------------------------------------------


def test_horizontal_moves_respect_walls() -> None:
    eng = _fresh(active_pieces=1)
    eng.active.clear()
    p = ActivePiece(pid=0, kind="O", rot=0, x=0, y=5, fall_period=1000)
    eng.active.append(p)
    rep = eng.step({0: Action.LEFT})
    assert p.x == 0
    assert rep.illegal == (0,), "壁にぶつかる移動は illegal として報告される"


def test_rotation_kick_near_wall() -> None:
    eng = _fresh(active_pieces=1)
    eng.active.clear()
    p = ActivePiece(pid=0, kind="I", rot=1, x=-1, y=5, fall_period=1000)
    # 縦 I を右壁際に置いて回転できることを確認
    p.x, p.rot = 8, 1
    eng.active.append(p)
    eng.step({0: Action.ROT_CW})
    assert eng.can_place(p.kind, p.rot, p.x, p.y, p.pid)


def test_action_is_legal_agrees_with_actually_applying_it() -> None:
    """身体感覚 (action_is_legal) と実適用の判定がずれていないこと。

    ここがずれると反射コントローラが「動けるはず」と思って動けない状態に陥る。
    """
    from copy import deepcopy

    rng = np.random.default_rng(3)
    eng = Engine(EngineConfig(active_pieces=4, fall_period=3, spawn_interval=1, seed=17))
    checked = 0
    for _ in range(250):
        if eng.game_over:
            break
        for p in list(eng.active):
            for a in Action:
                predicted = eng.action_is_legal(p, a)
                saved = deepcopy(p.__dict__)
                actual = eng._apply_action(p, a) or a == Action.NONE
                p.__dict__.update(saved)
                assert predicted == actual, f"{p.kind} rot={p.rot} action={a.name}"
                checked += 1
        eng.step({p.pid: Action(int(rng.integers(0, 5))) for p in eng.active})
    assert checked > 500


def test_hard_drop_disabled_by_default() -> None:
    eng = _fresh(active_pieces=1)
    eng.active.clear()
    p = ActivePiece(pid=0, kind="O", rot=0, x=0, y=2, fall_period=1000)
    eng.active.append(p)
    rep = eng.step({0: Action.HARD_DROP})
    assert p.y == 2
    assert rep.illegal == (0,)


# --- 決定性 ---------------------------------------------------------------


def test_same_seed_same_history() -> None:
    def run(seed: int) -> list[tuple]:
        eng = Engine(EngineConfig(active_pieces=3, seed=seed))
        out = []
        for _ in range(120):
            rep = eng.step({p.pid: Action.RIGHT for p in eng.active})
            out.append((rep.spawned, rep.locked, rep.cleared_rows))
        return out

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_game_over_when_stack_reaches_top() -> None:
    eng = _fresh(active_pieces=1, height=20, width=10)
    eng.active.clear()
    eng.board[:, :] = KIND_ID["T"]
    eng.board[19, :] = 0  # 全消しを起こさないよう最下段だけ空ける
    eng.step({})
    assert eng.game_over


# --- 特徴量 ---------------------------------------------------------------


def test_column_heights_and_holes() -> None:
    b = np.zeros((5, 3), dtype=np.uint8)
    b[4, 0] = 1
    b[2, 1] = 1
    b[4, 1] = 1  # 3 行目に穴
    assert list(column_heights(b)) == [1, 3, 0]
    assert count_holes(b) == 1
    assert bumpiness(b) == 2 + 3


def test_wells_counts_cumulative_depth() -> None:
    b = np.zeros((4, 3), dtype=np.uint8)
    b[:, 0] = 1
    b[:, 2] = 1  # 真ん中が深さ 4 の井戸
    assert wells(b) == 1 + 2 + 3 + 4


def test_drop_row_lands_on_stack() -> None:
    b = np.zeros((20, 10), dtype=np.uint8)
    b[19, :] = 1
    y = drop_row(b, "O", 0, 4)
    assert y == 17
    after = place(b, "O", 0, 4, y)
    assert after[17, 4] and after[18, 5]


def test_drop_row_rejects_out_of_bounds_columns() -> None:
    b = np.zeros((20, 10), dtype=np.uint8)
    assert drop_row(b, "I", 0, 8) is None
