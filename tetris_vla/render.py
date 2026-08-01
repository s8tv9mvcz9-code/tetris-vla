"""tetris_vla.render — 盤面スナップショットの描画と、そこからの厳密デコード。

このモジュールが PoC の「公平性」を担保する。すべてのエージェント (VLA も
ヒューリスティックも) は **まったく同じ PNG バイト列** だけを入力として受け取る。
盤面配列への直接アクセスは runtime 側で遮断されている。

ヒューリスティック baseline はここの `decode()` でピクセルから盤面を復元してから
探索する。つまり「同じピクセル、違う頭脳」という比較になる。
`decode(render(x)) == x` は tests/test_render.py でランダム盤面に対して検証している。

配色設計 (デコードを一意にするための制約):
    空セル        -> BG
    確定ブロック  -> SETTLED[kind]   (BASE を暗くした色)
    落下中ブロック -> BASE[kind]
    自機マーカー  -> セル内側に白リング (中心ピクセルは塗り潰さない)
    霧オプション  -> 他の落下ピースを FOG 単色に (種類が見えなくなる)
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import ID_KIND, KIND_ID, Snapshot

BG = (18, 18, 22)
GRID = (40, 40, 48)
EGO_RING = (255, 255, 255)
FOG = (128, 128, 138)
HEADER_BG = (10, 10, 12)
HEADER_FG = (190, 190, 200)
#: A2A モードで他ピースが「ここに収まる」と宣言した予定地 (ゴースト)。
#: 空セルにしか描かないので、確定/落下ブロックのデコードとは干渉しない。
GHOST = (72, 72, 96)

#: 落下中ブロックの色 (kind id -> RGB)
BASE: dict[int, tuple[int, int, int]] = {
    1: (0, 240, 240),    # I
    2: (240, 240, 0),    # O
    3: (170, 0, 240),    # T
    4: (0, 220, 60),     # S
    5: (240, 40, 40),    # Z
    6: (40, 80, 240),    # J
    7: (240, 150, 0),    # L
}
#: 確定ブロックの色 = BASE を暗くしたもの
SETTLED: dict[int, tuple[int, int, int]] = {
    k: (int(r * 0.45), int(g * 0.45), int(b * 0.45)) for k, (r, g, b) in BASE.items()
}

#: 中心ピクセル -> (kind_id, is_active) の逆引き表
_COLOR_TO_STATE: dict[tuple[int, int, int], tuple[int, bool]] = {}
for _k, _c in BASE.items():
    _COLOR_TO_STATE[_c] = (_k, True)
for _k, _c in SETTLED.items():
    _COLOR_TO_STATE[_c] = (_k, False)
_COLOR_TO_STATE[FOG] = (255, True)  # 255 = 種類不明の落下ブロック
_COLOR_TO_STATE[BG] = (0, False)

#: 「上に落下ピースがある」ことだけ分かる不明種別
UNKNOWN_KIND = 255


@dataclass(frozen=True)
class RenderConfig:
    cell_px: int = 12          # 10x20 盤面 -> 120x240px。VLM の prefill を抑える既定値。
    header_px: int = 14        # 列番号と tick を出すヘッダ帯。デコーダは無視する。
    grid_lines: bool = True
    fog_other: bool = False    # True で他の落下ピースを単色化 (種類が見えなくなる)
    labels: bool = True
    #: True で他ピースの宣言済み予定地をゴーストとして描く (A2A 協調モード)。
    #: 意図の共有を「別チャネル」ではなくスナップショット内に閉じ込めるための仕掛け。
    show_ghosts: bool = False

    def __post_init__(self) -> None:
        if self.cell_px < 8:
            raise ValueError("cell_px は 8 以上が必要 (自機リングと中心ピクセルが衝突する)")

    def image_size(self, width: int, height: int) -> tuple[int, int]:
        return width * self.cell_px, self.header_px + height * self.cell_px


_FONT = ImageFont.load_default()


def render(
    snap: Snapshot,
    ego_pid: int | None = None,
    cfg: RenderConfig | None = None,
    ghost_cells: Iterable[tuple[int, int]] | None = None,
) -> Image.Image:
    """スナップショットを 1 枚の RGB 画像にする。ego_pid の駒に白リングを付ける。

    ghost_cells は他ピースが宣言した予定地。cfg.show_ghosts が True のときだけ描く。
    """
    cfg = cfg or RenderConfig()
    H, W = snap.board.shape
    cw = cfg.cell_px
    img = Image.new("RGB", cfg.image_size(W, H), BG)
    d = ImageDraw.Draw(img)
    top = cfg.header_px

    # ヘッダ (デコード対象外の帯)
    if top:
        d.rectangle([0, 0, img.width, top - 1], fill=HEADER_BG)
        if cfg.labels:
            d.text((2, 2), f"t={snap.tick}", fill=HEADER_FG, font=_FONT)
            for c in range(W):
                d.text((c * cw + cw // 2 - 2, 2), str(c % 10), fill=HEADER_FG, font=_FONT)

    # グリッド線
    if cfg.grid_lines:
        for c in range(W + 1):
            x = min(c * cw, img.width - 1)
            d.line([(x, top), (x, img.height - 1)], fill=GRID)
        for r in range(H + 1):
            y = min(top + r * cw, img.height - 1)
            d.line([(0, y), (img.width - 1, y)], fill=GRID)

    def fill_cell(r: int, c: int, color: tuple[int, int, int]) -> None:
        x0, y0 = c * cw + 1, top + r * cw + 1
        d.rectangle([x0, y0, x0 + cw - 2, y0 + cw - 2], fill=color)

    # ゴースト (他ピースの宣言済み予定地) — 空セルにだけ描く
    if cfg.show_ghosts and ghost_cells:
        occupied = {(r, c) for p in snap.active for r, c in p.cells()}
        for r, c in ghost_cells:
            if 0 <= r < H and 0 <= c < W and not snap.board[r, c] and (r, c) not in occupied:
                fill_cell(r, c, GHOST)

    # 確定ブロック
    for r in range(H):
        for c in range(W):
            v = int(snap.board[r, c])
            if v:
                fill_cell(r, c, SETTLED[v])

    # 落下中ブロック
    for p in snap.active:
        is_ego = ego_pid is not None and p.pid == ego_pid
        if cfg.fog_other and not is_ego:
            color = FOG
        else:
            color = BASE[KIND_ID[p.kind]]
        for r, c in p.cells():
            if 0 <= r < H and 0 <= c < W:
                fill_cell(r, c, color)
        if is_ego:
            for r, c in p.cells():
                if 0 <= r < H and 0 <= c < W:
                    x0, y0 = c * cw + 1, top + r * cw + 1
                    d.rectangle([x0, y0, x0 + cw - 2, y0 + cw - 2], outline=EGO_RING, width=2)
    return img


@dataclass
class Decoded:
    """ピクセルから復元した盤面。エージェントが「見えている分」そのもの。"""

    settled: np.ndarray  # (H, W) uint8 kind id
    active: np.ndarray   # (H, W) uint8 kind id (0=なし, 255=種類不明)
    ego: np.ndarray      # (H, W) bool
    ghost: np.ndarray | None = None  # (H, W) bool  他ピースの宣言済み予定地

    def __post_init__(self) -> None:
        if self.ghost is None:
            self.ghost = np.zeros(self.settled.shape, dtype=bool)

    @property
    def occupancy(self) -> np.ndarray:
        """確定 + 落下中 をまとめた占有マップ (bool)。"""
        return (self.settled > 0) | (self.active > 0)

    def ego_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(r), int(c)) for r, c in zip(*np.nonzero(self.ego)))


def decode(img: Image.Image, width: int, height: int, cfg: RenderConfig | None = None) -> Decoded:
    """render() が作った画像を厳密に盤面へ戻す。

    各セルの中心ピクセルで種別と状態を、内側 2px で自機リングを判定する。
    render/decode の往復同一性は tests で保証される (= 公平性の担保)。
    """
    cfg = cfg or RenderConfig()
    cw = cfg.cell_px
    top = cfg.header_px
    px = np.asarray(img.convert("RGB"), dtype=np.uint8)

    settled = np.zeros((height, width), dtype=np.uint8)
    active = np.zeros((height, width), dtype=np.uint8)
    ego = np.zeros((height, width), dtype=bool)
    ghost = np.zeros((height, width), dtype=bool)

    for r in range(height):
        cy = top + r * cw + cw // 2
        for c in range(width):
            cx = c * cw + cw // 2
            color = tuple(int(v) for v in px[cy, cx])
            if color == GHOST:
                ghost[r, c] = True
                continue
            kind, is_active = _COLOR_TO_STATE.get(color, (0, False))
            if kind:
                if is_active:
                    active[r, c] = kind
                else:
                    settled[r, c] = kind
                probe = tuple(int(v) for v in px[top + r * cw + 2, c * cw + 2])
                if is_active and probe == EGO_RING:
                    ego[r, c] = True
    return Decoded(settled=settled, active=active, ego=ego, ghost=ghost)


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sha1(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:12]


def to_ascii(snap: Snapshot, ego_pid: int | None = None) -> str:
    """人間用の軽量ビュー。確定=大文字、落下中=小文字、自機=[]で囲う。"""
    H, W = snap.board.shape
    grid = [["." for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            v = int(snap.board[r, c])
            if v:
                grid[r][c] = ID_KIND[v]
    marks: dict[tuple[int, int], bool] = {}
    for p in snap.active:
        for r, c in p.cells():
            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = p.kind.lower()
                marks[(r, c)] = ego_pid is not None and p.pid == ego_pid
    lines = []
    who = " ".join(
        f"#{p.pid}:{p.kind}@c{p.left_col}v{p.fall_period}" for p in snap.active
    )
    lines.append(f"t={snap.tick} lines={snap.lines_cleared} active[{who}]")
    lines.append("+" + "-" * (W * 2) + "+")
    for r in range(H):
        row = "|"
        for c in range(W):
            ch = grid[r][c]
            row += f"[{ch}" if marks.get((r, c)) else f" {ch}"
        lines.append(row + " |")
    lines.append("+" + "-" * (W * 2) + "+")
    lines.append(" " + " ".join(str(c % 10) for c in range(W)))
    return "\n".join(lines)


def to_text_board(dec: Decoded) -> str:
    """デコード済み盤面のテキスト表現 (VLM への text-only ablation 用)。

    @=自分 / o=他の落下ピース / #=確定地形 / *=他ピースの宣言済み予定地 / .=空
    """
    H, W = dec.settled.shape
    rows = []
    for r in range(H):
        row = ""
        for c in range(W):
            if dec.ego[r, c]:
                row += "@"
            elif dec.active[r, c]:
                row += "o"
            elif dec.settled[r, c]:
                row += "#"
            elif dec.ghost is not None and dec.ghost[r, c]:
                row += "*"
            else:
                row += "."
        rows.append(f"{r:2d}|{row}|")
    rows.append("   " + "".join(str(c % 10) for c in range(W)))
    return "\n".join(rows)
