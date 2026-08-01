"""tetris_vla.engine — マルチエージェント・テトリスのコア。

de-facto standard な Tetris ルール (SRS 準拠のテトロミノ形状 / 7-bag randomizer /
重力落下 + lock / ライン消去) を実装しつつ、1点だけ意図的に拡張している:

    盤面上に *同時に N>1 個の active piece (落下中ピース)* が存在しうる。
    各ピースは独立したエージェントに所有される。

このモジュールは純粋かつ決定的である。seed と action 列が同じなら実行は完全に再現する。
wall-clock も I/O も一切参照しない。

座標系:
    row  r: 0 が天井、H-1 が床
    col  c: 0 が左端、W-1 が右端
    piece の (x, y) は「回転ボックスの左上」の盤面座標。r<0 (天井より上) は許容される。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------
# テトロミノ定義
# --------------------------------------------------------------------------

#: 回転状態 0 (spawn state) の形状。SRS の spawn state に一致させてある。
#: I は 4x4、O は 2x2、それ以外は 3x3 のボックスに収める。
SHAPES: dict[str, tuple[tuple[int, ...], ...]] = {
    "I": ((0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 0, 0), (0, 0, 0, 0)),
    "O": ((1, 1), (1, 1)),
    "T": ((0, 1, 0), (1, 1, 1), (0, 0, 0)),
    "S": ((0, 1, 1), (1, 1, 0), (0, 0, 0)),
    "Z": ((1, 1, 0), (0, 1, 1), (0, 0, 0)),
    "J": ((1, 0, 0), (1, 1, 1), (0, 0, 0)),
    "L": ((0, 0, 1), (1, 1, 1), (0, 0, 0)),
}

KINDS: tuple[str, ...] = ("I", "O", "T", "S", "Z", "J", "L")

#: 盤面配列に書き込む ID。0 は空セル。
KIND_ID: dict[str, int] = {k: i + 1 for i, k in enumerate(KINDS)}
ID_KIND: dict[int, str] = {v: k for k, v in KIND_ID.items()}


def _build_rotations(mat: Sequence[Sequence[int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    """行列を 90 度ずつ時計回りに回して 4 つの回転状態のセル集合を作る。

    手打ちの回転テーブルはバグの温床なので、行列回転から機械的に生成する。
    これにより「各回転がちょうど 4 セル」「回転が 4-cycle をなす」が構造的に保証される。
    """
    m = np.array(mat, dtype=np.uint8)
    out: list[tuple[tuple[int, int], ...]] = []
    for _ in range(4):
        cells = tuple(sorted((int(y), int(x)) for y, x in zip(*np.nonzero(m))))
        out.append(cells)
        m = np.rot90(m, -1)  # k=-1 は時計回り
    return tuple(out)


#: ROT_CELLS[kind][rot] -> ((dy, dx), ...)  ボックス左上からの相対座標
ROT_CELLS: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    k: _build_rotations(v) for k, v in SHAPES.items()
}
#: BOX[kind] -> 回転ボックスの一辺
BOX: dict[str, int] = {k: len(v) for k, v in SHAPES.items()}


def cell_span(kind: str, rot: int) -> tuple[int, int, int, int]:
    """(min_dy, max_dy, min_dx, max_dx) を返す。"""
    cells = ROT_CELLS[kind][rot % 4]
    dys = [c[0] for c in cells]
    dxs = [c[1] for c in cells]
    return min(dys), max(dys), min(dxs), max(dxs)


def unique_rotations(kind: str) -> list[int]:
    """見た目が重複しない回転状態のリスト (O は 1 個、I/S/Z は 2 個、他は 4 個)。"""
    seen: dict[tuple[tuple[int, int], ...], int] = {}
    for r in range(4):
        # 左上詰めに正規化してから比較する
        cells = ROT_CELLS[kind][r]
        min_dy = min(c[0] for c in cells)
        min_dx = min(c[1] for c in cells)
        norm = tuple(sorted((dy - min_dy, dx - min_dx) for dy, dx in cells))
        if norm not in seen:
            seen[norm] = r
    return sorted(seen.values())


# --------------------------------------------------------------------------
# アクション / 衝突種別
# --------------------------------------------------------------------------


class Action(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    ROT_CW = 3
    ROT_CCW = 4
    SOFT_DROP = 5
    HARD_DROP = 6


class Coll(IntEnum):
    """衝突の種類。TERRAIN と ACTIVE の区別が同時多ピース制御の肝。"""

    FREE = 0
    ACTIVE = 1  # 他の *落下中* ピースにのみ阻まれている -> lock しない (stall)
    TERRAIN = 2  # 壁・床・確定ブロック -> lock する


#: 回転時に試す簡易キック (SRS の完全な kick table ではなく、実装バグを避けた簡略版)
KICKS: tuple[tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1))


# --------------------------------------------------------------------------
# 状態
# --------------------------------------------------------------------------


@dataclass
class ActivePiece:
    """落下中ピース = 1 エージェント。"""

    pid: int
    kind: str
    rot: int
    x: int
    y: int
    fall_period: int  # 何 tick に 1 セル落ちるか (= 決められた時速)
    fall_accum: int = 0
    stall: int = 0  # 他の落下ピースに阻まれ続けた tick 数
    spawn_tick: int = 0
    persona: str = "default"

    def cells(self) -> tuple[tuple[int, int], ...]:
        return tuple((self.y + dy, self.x + dx) for dy, dx in ROT_CELLS[self.kind][self.rot % 4])

    @property
    def left_col(self) -> int:
        """占有セルの最左列。エージェントが指定する target column の基準。"""
        return self.x + cell_span(self.kind, self.rot)[2]

    @property
    def bottom_row(self) -> int:
        return self.y + cell_span(self.kind, self.rot)[1]


@dataclass
class Snapshot:
    """レンダラ / エージェントに渡す読み取り専用ビュー。"""

    tick: int
    board: np.ndarray  # (H, W) uint8, 確定ブロックのみ
    active: tuple[ActivePiece, ...]
    lines_cleared: int
    game_over: bool

    @property
    def height(self) -> int:
        return int(self.board.shape[0])

    @property
    def width(self) -> int:
        return int(self.board.shape[1])


@dataclass
class TickReport:
    tick: int
    spawned: tuple[int, ...] = ()
    locked: tuple[int, ...] = ()
    #: lock した瞬間のピース状態 (「意図した場所に収まれたか」の判定に使う)
    locked_pieces: tuple[ActivePiece, ...] = ()
    force_locked: tuple[int, ...] = ()
    cleared_rows: tuple[int, ...] = ()
    stalls: int = 0
    illegal: tuple[int, ...] = ()  # 不正でリジェクトされた action を出した pid
    game_over: bool = False


@dataclass
class EngineConfig:
    width: int = 10
    height: int = 20
    #: 同時に落下するピース数。1 にすると古典的なテトリスと同じ挙動になる。
    active_pieces: int = 3
    #: 落下速度: 何 tick に 1 セル落ちるか。小さいほど速い。
    fall_period: int = 4
    #: ピースごとの速度ばらつき (fall_period ± jitter)。「各ピースの決められた時速」
    fall_period_jitter: int = 1
    #: 連続 spawn の最小間隔 (tick)
    spawn_interval: int = 5
    #: 他の落下ピースに阻まれ続けた場合に強制 lock するまでの tick 数 (デッドロック防止)
    stall_limit: int = 10
    allow_hard_drop: bool = False
    seed: int = 0


# --------------------------------------------------------------------------
# エンジン本体
# --------------------------------------------------------------------------


class Engine:
    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        c = self.cfg
        self.rng = random.Random(c.seed)
        self.board = np.zeros((c.height, c.width), dtype=np.uint8)
        self.active: list[ActivePiece] = []
        self.tick = 0
        self.lines_cleared = 0
        self.pieces_locked = 0
        self.line_clear_histogram = [0, 0, 0, 0, 0]  # index = 同時消し行数
        self.game_over = False
        self._bag: list[str] = []
        self._next_pid = 0
        self._last_spawn_tick = -10**9
        self._spawn_cursor = 0
        self._maybe_spawn()  # tick 0 で最初の 1 個を出す

    # -- 内部ヘルパ --------------------------------------------------------

    def _refill_bag(self) -> None:
        bag = list(KINDS)
        self.rng.shuffle(bag)
        self._bag.extend(bag)

    def _peek_kind(self) -> str:
        if not self._bag:
            self._refill_bag()
        return self._bag[0]

    def _pop_kind(self) -> str:
        k = self._peek_kind()
        self._bag.pop(0)
        return k

    def _active_cells(self, ignore_pid: int | None) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for p in self.active:
            if p.pid == ignore_pid:
                continue
            out.update(p.cells())
        return out

    def check(self, kind: str, rot: int, x: int, y: int, ignore_pid: int | None) -> Coll:
        """指定配置が置けるか。TERRAIN > ACTIVE > FREE の優先度で返す。"""
        H, W = self.board.shape
        others = self._active_cells(ignore_pid)
        hit_active = False
        for dy, dx in ROT_CELLS[kind][rot % 4]:
            r, c = y + dy, x + dx
            if c < 0 or c >= W or r >= H:
                return Coll.TERRAIN
            if r >= 0 and self.board[r, c]:
                return Coll.TERRAIN
            if (r, c) in others:
                hit_active = True
        return Coll.ACTIVE if hit_active else Coll.FREE

    def can_place(self, kind: str, rot: int, x: int, y: int, ignore_pid: int | None = None) -> bool:
        return self.check(kind, rot, x, y, ignore_pid) == Coll.FREE

    # -- action の適用 -----------------------------------------------------

    def action_is_legal(self, p: ActivePiece, a: Action) -> bool:
        """そのアクションが今このピースに適用できるか (盤面を変更しない)。

        これは「体の感覚」であって盤面の特権的知識ではない。反射コントローラが
        塞がれた方向へ毎 tick 突っ込むのを防ぐために使う。
        _apply_action と判定が一致することは tests で検証している。
        """
        if a == Action.NONE:
            return True
        if a in (Action.LEFT, Action.RIGHT):
            dx = -1 if a == Action.LEFT else 1
            return self.can_place(p.kind, p.rot, p.x + dx, p.y, p.pid)
        if a in (Action.ROT_CW, Action.ROT_CCW):
            nrot = (p.rot + (1 if a == Action.ROT_CW else -1)) % 4
            return any(
                self.can_place(p.kind, nrot, p.x + kx, p.y + ky, p.pid) for kx, ky in KICKS
            )
        if a == Action.SOFT_DROP:
            return self.can_place(p.kind, p.rot, p.x, p.y + 1, p.pid)
        if a == Action.HARD_DROP:
            return self.cfg.allow_hard_drop and self.can_place(
                p.kind, p.rot, p.x, p.y + 1, p.pid
            )
        return False

    def _apply_action(self, p: ActivePiece, a: Action) -> bool:
        """1 tick に 1 アクション。適用できたら True。"""
        if a == Action.NONE:
            return True
        if a in (Action.LEFT, Action.RIGHT):
            dx = -1 if a == Action.LEFT else 1
            if self.can_place(p.kind, p.rot, p.x + dx, p.y, p.pid):
                p.x += dx
                return True
            return False
        if a in (Action.ROT_CW, Action.ROT_CCW):
            nrot = (p.rot + (1 if a == Action.ROT_CW else -1)) % 4
            for kx, ky in KICKS:
                if self.can_place(p.kind, nrot, p.x + kx, p.y + ky, p.pid):
                    p.rot, p.x, p.y = nrot, p.x + kx, p.y + ky
                    return True
            return False
        if a == Action.SOFT_DROP:
            if self.can_place(p.kind, p.rot, p.x, p.y + 1, p.pid):
                p.y += 1
                p.fall_accum = 0
                return True
            return False
        if a == Action.HARD_DROP:
            if not self.cfg.allow_hard_drop:
                return False
            moved = False
            while self.can_place(p.kind, p.rot, p.x, p.y + 1, p.pid):
                p.y += 1
                moved = True
            p.fall_accum = p.fall_period  # 次の重力処理で即 lock 判定に入る
            return moved
        return False

    # -- lock / line clear -------------------------------------------------

    def _lock(self, p: ActivePiece) -> None:
        pid_val = KIND_ID[p.kind]
        for r, c in p.cells():
            if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]:
                self.board[r, c] = pid_val
        self.pieces_locked += 1

    def _clear_lines(self) -> tuple[int, ...]:
        full = [r for r in range(self.board.shape[0]) if bool(self.board[r].all())]
        if not full:
            return ()
        keep = [r for r in range(self.board.shape[0]) if r not in set(full)]
        new = np.zeros_like(self.board)
        new[len(full):] = self.board[keep]
        self.board = new
        n = len(full)
        self.lines_cleared += n
        self.line_clear_histogram[min(n, 4)] += 1

        # 消えた行より上にいる落下中ピースも一緒に下げる
        for p in self.active:
            shift = sum(1 for r in full if r > p.bottom_row)
            while shift > 0:
                if self.can_place(p.kind, p.rot, p.x, p.y + shift, p.pid):
                    p.y += shift
                    break
                shift -= 1
        return tuple(full)

    # -- spawn -------------------------------------------------------------

    def _spawn_columns(self, kind: str) -> list[int]:
        """spawn 候補 x を「望ましい順」に並べる。

        active_pieces == 1 なら古典テトリスどおり中央固定。
        同時落下モードでは投入位置を盤面幅に均等分散させて巡回させる。
        全部中央から出すと、エージェントの判断とは無関係な渋滞が中央に生まれ、
        干渉の測定が spawn 位置のアーティファクトに汚染されてしまうため。
        """
        W = self.board.shape[1]
        _, _, min_dx, max_dx = cell_span(kind, 0)
        lo = -min_dx
        hi = W - 1 - max_dx
        if hi < lo:
            return []
        n = max(1, self.cfg.active_pieces)
        if n == 1:
            anchor = (W - (max_dx - min_dx + 1)) // 2 - min_dx
        else:
            slot = self._spawn_cursor % n
            anchor = lo + round((hi - lo) * slot / (n - 1))
        cands = [anchor]
        for d in range(1, W + 1):
            cands.extend([anchor - d, anchor + d])
        return [x for x in cands if lo <= x <= hi]

    def _maybe_spawn(self) -> int | None:
        c = self.cfg
        if len(self.active) >= c.active_pieces:
            return None
        if self.tick - self._last_spawn_tick < c.spawn_interval:
            return None
        kind = self._peek_kind()
        y = -cell_span(kind, 0)[0]
        for x in self._spawn_columns(kind):
            if self.can_place(kind, 0, x, y, None):
                self._pop_kind()
                jitter = c.fall_period_jitter
                period = c.fall_period + (self.rng.randint(-jitter, jitter) if jitter else 0)
                p = ActivePiece(
                    pid=self._next_pid,
                    kind=kind,
                    rot=0,
                    x=x,
                    y=y,
                    fall_period=max(1, period),
                    spawn_tick=self.tick,
                )
                self._next_pid += 1
                self.active.append(p)
                self._last_spawn_tick = self.tick
                self._spawn_cursor += 1
                return p.pid
        return None

    def can_spawn(self) -> bool:
        kind = self._peek_kind()
        y = -cell_span(kind, 0)[0]
        return any(self.can_place(kind, 0, x, y, None) for x in self._spawn_columns(kind))

    # -- 1 tick ------------------------------------------------------------

    def step(self, actions: Mapping[int, Action] | None = None) -> TickReport:
        """論理 tick を 1 進める。actions は pid -> Action。

        処理順: (1) アクション適用 (2) 重力 (3) lock & ライン消去 (4) spawn (5) 終了判定
        """
        actions = actions or {}
        rep = TickReport(tick=self.tick)
        if self.game_over:
            rep.game_over = True
            return rep

        illegal: list[int] = []
        to_lock: list[ActivePiece] = []
        forced: list[int] = []
        stalls = 0

        # (1) アクション適用 — pid 昇順で決定的に処理する
        for p in sorted(self.active, key=lambda q: q.pid):
            a = Action(actions.get(p.pid, Action.NONE))
            if not self._apply_action(p, a) and a != Action.NONE:
                illegal.append(p.pid)

        # (2) 重力
        for p in sorted(self.active, key=lambda q: q.pid):
            if p in to_lock:
                continue
            p.fall_accum += 1
            if p.fall_accum < p.fall_period:
                continue
            p.fall_accum = 0
            coll = self.check(p.kind, p.rot, p.x, p.y + 1, p.pid)
            if coll == Coll.FREE:
                p.y += 1
                p.stall = 0
            elif coll == Coll.TERRAIN:
                to_lock.append(p)
            else:
                # 他の落下ピースに阻まれているだけ -> lock しない (下が動けば落ちられる)
                p.stall += 1
                stalls += 1
                if p.stall >= self.cfg.stall_limit:
                    to_lock.append(p)
                    forced.append(p.pid)

        # (3) lock & ライン消去
        locked_ids: list[int] = []
        for p in to_lock:
            self._lock(p)
            locked_ids.append(p.pid)
        if to_lock:
            keep = {p.pid for p in to_lock}
            self.active = [p for p in self.active if p.pid not in keep]
            rep.cleared_rows = self._clear_lines()

        self.tick += 1

        # (4) spawn
        spawned: list[int] = []
        while True:
            pid = self._maybe_spawn()
            if pid is None:
                break
            spawned.append(pid)

        # (5) 終了判定: 落下中ピースが尽き、かつ新規 spawn もできない
        if not self.active and not self.can_spawn():
            self.game_over = True

        rep.spawned = tuple(spawned)
        rep.locked = tuple(locked_ids)
        rep.locked_pieces = tuple(replace(p) for p in to_lock)
        rep.force_locked = tuple(forced)
        rep.stalls = stalls
        rep.illegal = tuple(illegal)
        rep.game_over = self.game_over
        return rep

    # -- 観測 --------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        return Snapshot(
            tick=self.tick,
            board=self.board.copy(),
            active=tuple(replace(p) for p in sorted(self.active, key=lambda q: q.pid)),
            lines_cleared=self.lines_cleared,
            game_over=self.game_over,
        )

    def piece(self, pid: int) -> ActivePiece | None:
        for p in self.active:
            if p.pid == pid:
                return p
        return None


# --------------------------------------------------------------------------
# 盤面特徴量 (ヒューリスティック評価と指標の両方で使う)
# --------------------------------------------------------------------------


def column_heights(board: np.ndarray) -> np.ndarray:
    """各列の高さ (床からの、一番高い占有セルまで)。"""
    H, W = board.shape
    filled = board > 0
    out = np.zeros(W, dtype=np.int32)
    for c in range(W):
        rows = np.nonzero(filled[:, c])[0]
        out[c] = 0 if rows.size == 0 else H - int(rows[0])
    return out


def count_holes(board: np.ndarray) -> int:
    """上に埋まったセルがある空セルの数。"""
    H, W = board.shape
    filled = board > 0
    total = 0
    for c in range(W):
        seen = False
        for r in range(H):
            if filled[r, c]:
                seen = True
            elif seen:
                total += 1
    return total


def bumpiness(board: np.ndarray) -> int:
    h = column_heights(board)
    return int(np.abs(np.diff(h)).sum())


def aggregate_height(board: np.ndarray) -> int:
    return int(column_heights(board).sum())


def row_transitions(board: np.ndarray) -> int:
    """行方向の 占有<->空 の切り替わり回数。盤外は「占有」とみなす。"""
    H, W = board.shape
    filled = (board > 0).astype(np.int8)
    padded = np.ones((H, W + 2), dtype=np.int8)
    padded[:, 1:-1] = filled
    return int(np.abs(np.diff(padded, axis=1)).sum())


def col_transitions(board: np.ndarray) -> int:
    """列方向の切り替わり回数。天井側は「空」、床側は「占有」とみなす。"""
    H, W = board.shape
    filled = (board > 0).astype(np.int8)
    padded = np.zeros((H + 2, W), dtype=np.int8)
    padded[1:-1, :] = filled
    padded[-1, :] = 1
    return int(np.abs(np.diff(padded, axis=0)).sum())


def wells(board: np.ndarray) -> int:
    """井戸の累積深さ sum(d*(d+1)/2)。左右が埋まっている縦の窪み。"""
    H, W = board.shape
    filled = board > 0
    total = 0
    for c in range(W):
        depth = 0
        for r in range(H):
            left = True if c == 0 else bool(filled[r, c - 1])
            right = True if c == W - 1 else bool(filled[r, c + 1])
            if not filled[r, c] and left and right:
                depth += 1
                total += depth
            else:
                depth = 0
    return total


def drop_row(board: np.ndarray, kind: str, rot: int, x: int) -> int | None:
    """確定盤面のみを相手にハードドロップした時の着地 y。置けなければ None。"""
    H, W = board.shape
    cells = ROT_CELLS[kind][rot % 4]
    min_dy, _, min_dx, max_dx = cell_span(kind, rot)
    if x + min_dx < 0 or x + max_dx >= W:
        return None
    y = -min_dy
    # 1 つ下が置けなくなるまで落とす
    while True:
        ny = y + 1
        ok = True
        for dy, dx in cells:
            r, c = ny + dy, x + dx
            if r >= H or (r >= 0 and board[r, c]):
                ok = False
                break
        if not ok:
            break
        y = ny
    # 最終位置が有効か検証 (天井付近で詰まっている場合など)
    for dy, dx in cells:
        r, c = y + dy, x + dx
        if r >= H:
            return None
        if r >= 0 and board[r, c]:
            return None
    return y


def place(board: np.ndarray, kind: str, rot: int, x: int, y: int) -> np.ndarray:
    out = board.copy()
    v = KIND_ID[kind]
    for dy, dx in ROT_CELLS[kind][rot % 4]:
        r, c = y + dy, x + dx
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = v
    return out
