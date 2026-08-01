"""tetris_vla.flock — 複数機の同時パラシュート降下を、qwen-VL + SmolVLA だけで飛ばす。

## 役割分担 (PD ヒューリスティックは一切使わない)

    qwen2.5vl:3b (VLM)  「どこへ」  遅い (1〜3秒) が地形を読める。
                                   全機ぶんの目標列を 1 回のリクエストでまとめて出す。
    SmolVLA      (VLA)  「どうやって」速い (0.4秒で50手ぶん)。
                                   指令された列へ、慣性と風を見越して寄せる。

参照解 (Dellacherie) も PD 制御も推論ループには入らない。オラクルは
**採点のときだけ** 使う (どれだけ良い場所に降りたかの物差し)。

## いい塩梅にするための 3 つの仕掛け

1. **VLM は全機まとめて 1 回**。機数ぶん呼ぶと 8GB では待ち行列が破綻する。
2. **VLM の呼び出し間隔を固定** (`vlm_every`)。その間 VLA は直前の指令を持ち越す。
3. **スローモーション倍率**。実レイテンシは `latency / (dt * slowmo)` tick に換算される。
   slowmo=1 が等倍。大きくすると「世界がゆっくり進む」= 遅いモデルでも操縦できる。

## 記録

どちらのモデルが / いつ / 何を見せられて / 何を返して / 何秒かかったか を全部残す。
`ModelCall` がその 1 件で、HTML はこれを時系列に並べたもの。
"""

from __future__ import annotations

import base64
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .engine import KIND_ID, KINDS, ROT_CELLS, cell_span, count_holes, place, unique_rotations
from .parachute import (
    Lander,
    ParachuteConfig,
    Thrust,
    landing_regret,
    make_terrain,
    target_column,
)
from .render import BASE, FLAME, RenderConfig, SETTLED, TRAIL, render_terrain, to_png

PARACHUTE_COLOR = (225, 225, 235)
CORD_COLOR = (140, 140, 150)
EGO_RING = (255, 255, 255)


# --------------------------------------------------------------------------
# 記録
# --------------------------------------------------------------------------


@dataclass
class ModelCall:
    """モデル 1 回の呼び出し。これを時系列に並べたものがデモの中身。"""

    seq: int
    model: str              # "qwen2.5vl:3b" / "smolvla"
    role: str               # "どこへ" / "どうやって"
    tick: int
    t: float                # 論理秒
    pids: list[int]
    latency_s: float
    drift_ticks: int        # この推論の間に世界が進んだ tick
    prompt: str | None = None
    raw: str | None = None
    parsed: dict = field(default_factory=dict)
    error: str | None = None
    png_b64: str | None = None
    inputs: dict = field(default_factory=dict)   # センサ値など画像以外の入力
    note: str = ""


@dataclass
class FlockFrame:
    tick: int
    t: float
    #: pid -> (x, y, vx, vy, fuel, action, landed)
    landers: dict[int, tuple]
    call_seq: int | None = None


# --------------------------------------------------------------------------
# 複数機ワールド
# --------------------------------------------------------------------------


@dataclass
class FlockConfig:
    n_landers: int = 3
    spawn_gap_ticks: int = 10      # 投入間隔
    slowmo: float = 3.0            # スローモーション倍率 (1.0 が等倍)
    vlm_every: int = 24            # VLM に問い合わせる間隔 (tick)
    max_ticks: int = 400
    #: 1 回の推論で流される tick の上限。1 回のハズレで面が終わるのを防ぐ
    max_drift_ticks: int = 24
    warmup: bool = True            # 本番前にモデルを暖機する


class FlockWorld:
    """地形 + N 機。機体同士も衝突する (先に着いたものが地形になる)。"""

    def __init__(self, cfg: ParachuteConfig, fcfg: FlockConfig, board: np.ndarray,
                 seed: int = 0) -> None:
        self.cfg, self.fcfg = cfg, fcfg
        self.board = board
        self.seed = seed
        self.rng = random.Random(seed * 31 + 7)
        self.tick = 0
        self.t = 0.0
        self.landers: dict[int, Lander] = {}
        self.landed: dict[int, dict] = {}
        self.next_pid = 0
        self._last_spawn = -10**9
        self.frames: list[FlockFrame] = []
        self.actions: dict[int, Thrust] = {}
        self.holes_before = count_holes(board)
        self.spawn_order: list[tuple[int, str, int]] = []

    # -- 判定 -------------------------------------------------------------

    def _cells(self, L: Lander, x: float, y: float):
        ix, iy = int(round(x)), int(round(y))
        return [(iy + dy, ix + dx) for dy, dx in ROT_CELLS[L.kind][L.rot % 4]]

    def blocked(self, L: Lander, x: float, y: float, ignore: int) -> bool:
        k = self.collide(L, x, y, ignore)
        return k != "free"

    def collide(self, L: Lander, x: float, y: float, ignore: int) -> str:
        """"free" / "lander" / "terrain" を返す。

        **この区別が要る。** 他機に阻まれただけで着地させると、下の機が退いた時に
        空中固定のブロックが残り、大量の穴として計上されてしまう
        (テトリス側の engine.Coll と同じ理由)。
        """
        H, W = self.board.shape
        others = {c for pid, o in self.landers.items() if pid != ignore
                  for c in self._cells(o, o.x, o.y)}
        hit_lander = False
        for r, c in self._cells(L, x, y):
            if c < 0 or c >= W or r >= H:
                return "terrain"
            if r >= 0 and self.board[r, c]:
                return "terrain"
            if (r, c) in others:
                hit_lander = True
        return "lander" if hit_lander else "free"

    def wind_at(self, t: float) -> float:
        c = self.cfg
        if not c.gust:
            return c.wind
        return c.wind + c.gust * math.sin(2 * math.pi * t / max(1e-6, c.gust_period))

    # -- 投入 -------------------------------------------------------------

    def _maybe_spawn(self) -> int | None:
        f = self.fcfg
        if len(self.landers) + len(self.landed) >= f.n_landers:
            return None
        if self.tick - self._last_spawn < f.spawn_gap_ticks:
            return None
        kind = self.rng.choice(KINDS)
        rot = self.rng.choice(unique_rotations(kind))
        min_dy, _, min_dx, max_dx = cell_span(kind, rot)
        lo, hi = -min_dx, self.cfg.width - 1 - max_dx
        if hi < lo:
            return None
        y = float(-min_dy)
        # 投入位置は盤面幅に均等分散させる。左右 2 箇所だけだと 3 機目が入らないうえ、
        # 「複数機が別々の場所から降りてくる」という設定が成立しない
        k = max(2, f.n_landers)
        slot = self.next_pid % k
        anchor = lo + round((hi - lo) * slot / (k - 1))
        cands = [anchor] + [anchor + d * s for d in range(1, self.cfg.width + 1) for s in (-1, 1)]
        L = None
        for cx in cands:
            if not (lo <= cx <= hi):
                continue
            probe = Lander(kind=kind, rot=rot, x=float(cx), y=y, fuel=self.cfg.fuel)
            if not self.blocked(probe, float(cx), y, -1):
                L = probe
                break
        if L is None:
            return None
        pid = self.next_pid
        self.next_pid += 1
        self.landers[pid] = L
        self._last_spawn = self.tick
        self.spawn_order.append((pid, kind, rot))
        return pid

    # -- 1 tick -----------------------------------------------------------

    def step(self, actions: dict[int, Thrust] | None = None) -> FlockFrame:
        c = self.cfg
        actions = actions or {}
        self.actions = dict(actions)

        for pid in sorted(self.landers):
            L = self.landers[pid]
            act = Thrust(actions.get(pid, Thrust.COAST))
            if act in (Thrust.LEFT, Thrust.RIGHT) and L.fuel <= 0:
                act = Thrust.COAST
            u = {Thrust.LEFT: -1.0, Thrust.RIGHT: 1.0}.get(act, 0.0)
            if u:
                L.fuel -= 1

            wind = self.wind_at(self.t)
            drag_v = c.drag_v * (c.vent_drag_scale if act == Thrust.VENT else 1.0)
            L.vx += (c.thrust * u + wind - c.drag_h * L.vx) * c.dt
            L.vy += (c.gravity - drag_v * L.vy) * c.dt

            nx, ny = L.x + L.vx * c.dt, L.y + L.vy * c.dt
            _, _, min_dx, max_dx = cell_span(L.kind, L.rot)
            lo, hi = -min_dx, c.width - 1 - max_dx
            if nx < lo:
                nx, L.vx = float(lo), 0.0
            elif nx > hi:
                nx, L.vx = float(hi), 0.0

            kind_c = self.collide(L, nx, ny, pid)
            if kind_c == "terrain":
                if self.collide(L, nx, L.y, pid) == "free":
                    L.x = nx
                self._land(pid, L)
            elif kind_c == "lander":
                # 他機に阻まれているだけ。下が退けば落ちられるので lock しない。
                # 長く詰まったままなら強制着地させてデッドロックを断つ。
                if self.collide(L, nx, L.y, pid) == "free":
                    L.x = nx
                L.stall = getattr(L, "stall", 0) + 1
                L.vy = min(L.vy, 0.4)
                if L.stall >= 40:
                    self._land(pid, L)
            else:
                L.x, L.y = nx, ny
                L.stall = 0
            L.trail.append((L.x, L.y))
            if len(L.trail) > c.trail_len:
                L.trail.pop(0)

        self.tick += 1
        self.t += c.dt
        self._maybe_spawn()

        frame = FlockFrame(
            tick=self.tick, t=round(self.t, 3),
            landers={pid: (round(L.x, 3), round(L.y, 3), round(L.vx, 3), round(L.vy, 3),
                           L.fuel, int(actions.get(pid, Thrust.COAST)), False)
                     for pid, L in self.landers.items()},
        )
        for pid, info in self.landed.items():
            frame.landers.setdefault(pid, (info["x"], info["y"], 0.0, 0.0,
                                           info["fuel_left"], 0, True))
        self.frames.append(frame)
        return frame

    def _land(self, pid: int, L: Lander) -> None:
        """接地。**その場で地形になる** ので、後続機の着地面が変わる。"""
        before = count_holes(self.board)
        self.board = place(self.board, L.kind, L.rot, int(round(L.x)), int(round(L.y)))
        self.landed[pid] = {
            "kind": L.kind, "rot": L.rot, "x": round(L.x, 3), "y": round(L.y, 3),
            "col": L.col, "impact_vy": round(L.vy, 3), "impact_vx": round(L.vx, 3),
            "fuel_left": L.fuel, "tick": self.tick,
            "holes_created": max(0, count_holes(self.board) - before),
            "soft": abs(L.vy) <= self.cfg.soft_vy and abs(L.vx) <= self.cfg.soft_vx,
        }
        del self.landers[pid]

    @property
    def done(self) -> bool:
        return (len(self.landed) >= self.fcfg.n_landers) or self.tick >= self.fcfg.max_ticks


# --------------------------------------------------------------------------
# 描画 (モデルに渡る画像)
# --------------------------------------------------------------------------


def render_flock(world: FlockWorld, cfg: ParachuteConfig, rcfg: RenderConfig,
                 ego_pid: int | None = None, labels: bool = True) -> Image.Image:
    """全機が写った 1 枚。ego_pid を指定するとその機に白リングが付く。

    VLM には ego なし (全機の配置を見て割り当てを決めるので) 、
    VLA には ego 付き (自分がどれかを知る必要があるので) を渡す。
    """
    img = render_terrain(world.board, rcfg)
    d = ImageDraw.Draw(img)
    cw, top = rcfg.cell_px, rcfg.header_px

    def px(cx: float, cy: float) -> tuple[float, float]:
        return cx * cw, top + cy * cw

    for pid in sorted(world.landers):
        L = world.landers[pid]
        n = len(L.trail)
        for i, (tx, ty) in enumerate(L.trail[:-1]):
            f = (i + 1) / max(1, n)
            col = tuple(int(TRAIL[k] * (0.3 + 0.7 * f)) for k in range(3))
            cxc, cyc = px(tx + 0.5, ty + 0.5)
            d.ellipse([cxc - 1, cyc - 1, cxc + 1, cyc + 1], fill=col)

        _, _, min_dx, max_dx = cell_span(L.kind, L.rot)
        cx0, _ = px(L.x + (min_dx + max_dx + 1) / 2.0, 0)
        canopy_y = top + (L.y - 1.6) * cw
        if canopy_y > top - cw:
            rx = cw * 1.4
            d.arc([cx0 - rx, canopy_y - cw * 0.7, cx0 + rx, canopy_y + cw * 0.9],
                  180, 360, fill=PARACHUTE_COLOR, width=2)
            d.line([cx0, canopy_y + cw * 0.1, cx0, top + (L.y + 0.2) * cw], fill=CORD_COLOR)

        base = BASE[KIND_ID[L.kind]]
        for dy, dx in ROT_CELLS[L.kind][L.rot % 4]:
            x0, y0 = px(L.x + dx, L.y + dy)
            d.rectangle([x0 + 1, y0 + 1, x0 + cw - 2, y0 + cw - 2], fill=base)
        if ego_pid is not None and pid == ego_pid:
            for dy, dx in ROT_CELLS[L.kind][L.rot % 4]:
                x0, y0 = px(L.x + dx, L.y + dy)
                d.rectangle([x0 + 1, y0 + 1, x0 + cw - 2, y0 + cw - 2],
                            outline=EGO_RING, width=2)

        act = world.actions.get(pid)
        if act in (Thrust.LEFT, Thrust.RIGHT, Thrust.VENT):
            cxm, cym = px(L.x + (min_dx + max_dx + 1) / 2.0, L.y + 0.5)
            if act == Thrust.RIGHT:
                pts = [(cxm - cw * .6, cym - 3), (cxm - cw * .6, cym + 3), (cxm - cw * 1.3, cym)]
            elif act == Thrust.LEFT:
                pts = [(cxm + cw * .6, cym - 3), (cxm + cw * .6, cym + 3), (cxm + cw * 1.3, cym)]
            else:
                pts = [(cxm - 3, cym - cw * .8), (cxm + 3, cym - cw * .8), (cxm, cym - cw * 1.5)]
            d.polygon(pts, fill=FLAME)

    if top and labels:
        d.rectangle([0, 0, img.width, top - 1], fill=(10, 10, 12))
        d.text((2, 3), f"t={world.t:.1f}s 機数{len(world.landers)}", fill=(190, 190, 200))
    return img


def flock_ascii(world: FlockWorld, targets: dict[int, int] | None = None) -> str:
    H, W = world.board.shape
    grid = [["." for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if world.board[r, c]:
                grid[r][c] = "#"
    for pid in sorted(world.landers):
        L = world.landers[pid]
        for tx, ty in L.trail[:-1]:
            r, c = int(round(ty)), int(round(tx))
            if 0 <= r < H and 0 <= c < W and grid[r][c] == ".":
                grid[r][c] = "·"
        for r, c in world._cells(L, L.x, L.y):
            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = str(pid % 10)
    lines = ["   " + "".join(str(c % 10) for c in range(W))]
    lines += [f"{r:2d}|" + "".join(grid[r]) + "|" for r in range(H)]
    foot = [" "] * W
    for pid, t in (targets or {}).items():
        if t is not None and 0 <= t < W and pid in world.landers:
            foot[t] = str(pid % 10)
    lines.append("   " + "".join(foot) + "  ← VLM が指令した目標列")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 上位: qwen-VL が全機の目標列を決める
# --------------------------------------------------------------------------

#: 出力は **位置で対応させる極小スキーマ**。pid を書かせると
#: トークンが増えるうえ、存在しない pid を作られる (実際に起きた)。
FLOCK_PROMPT = """Tetromino pieces are descending by parachute onto the terrain shown.
Dark blocks = ground. Bright blocks = falling pieces. Small dots = their trails.
Board has {width} columns, 0 (left) to {maxcol} (right).

Pieces in the air, in order:
{roster}

For each piece IN THE SAME ORDER, choose the landing column where NO empty cavity
would be left underneath it. Prefer a flat surface or a notch matching its shape.
Do not give two pieces the same column.

Reply with JSON only, exactly {n} numbers in that order:
{{"cols":[{example}]}}"""


class QwenCommander:
    """上位 VLM。全機ぶんの目標列を 1 回のリクエストでまとめて出す。

    機数ぶん呼ぶと 8GB では待ち行列が破綻するので、必ず 1 回にまとめる。
    """

    def __init__(self, model: str = "qwen2.5vl:3b", host: str = "http://localhost:11434",
                 num_predict: int = 40, temperature: float = 0.0, timeout: float = 600.0,
                 keep_alive: str = "30m", unload_after: bool = True) -> None:
        import httpx

        self.model, self.host = model, host.rstrip("/")
        self.num_predict, self.temperature, self.keep_alive = num_predict, temperature, keep_alive
        #: 応答後に VLM をメモリから降ろす。8GB では VLM(3.7GB) と VLA(1.8GB) が
        #: 互いを追い出し合い、毎回リロードが走って 40〜50 秒かかっていた。
        #: 明示的に時分割すると「VLM はロード込みで遅い / VLA は速い」と切り分けられる。
        self.unload_after = unload_after
        self.load_seconds = 0.0
        self._client = httpx.Client(timeout=timeout)

    def health(self) -> str:
        r = self._client.get(f"{self.host}/api/tags")
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        if self.model not in names:
            raise RuntimeError(f"{self.model} が Ollama にありません: {names}")
        return f"ollama ok ({self.model})"

    def build_prompt(self, world: FlockWorld) -> str:
        pids = sorted(world.landers)
        roster = "\n".join(
            f"  {i+1}. shape {world.landers[pid].kind}, now at column "
            f"{world.landers[pid].col}, row {world.landers[pid].y:.0f}, "
            f"drifting {world.landers[pid].vx:+.1f} col/s"
            for i, pid in enumerate(pids)
        )
        return FLOCK_PROMPT.format(
            width=world.cfg.width, maxcol=world.cfg.width - 1, roster=roster,
            n=len(pids), example=",".join(["0"] * len(pids)))

    def __call__(self, world: FlockWorld, png: bytes) -> tuple[dict[int, int], str, str, str | None]:
        prompt = self.build_prompt(world)
        try:
            r = self._client.post(f"{self.host}/api/generate", json={
                "model": self.model, "prompt": prompt,
                "images": [base64.b64encode(png).decode("ascii")],
                "stream": False, "format": "json", "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            })
            r.raise_for_status()
            raw = r.json().get("response", "")
        except Exception as exc:
            return {}, prompt, "", f"backend_error:{type(exc).__name__}"
        finally:
            if self.unload_after:
                self.unload()

        from .agents import _extract_json

        pids = sorted(world.landers)
        obj = _extract_json(raw) or {}
        cols = obj.get("cols") or obj.get("columns") or obj.get("target_cols")
        out: dict[int, int] = {}
        err = None
        if isinstance(cols, list) and cols:
            for pid, v in zip(pids, cols):          # 位置で対応。余りは捨てる
                try:
                    out[pid] = max(0, min(world.cfg.width - 1, int(v)))
                except (TypeError, ValueError):
                    continue
        if not out:
            # 崩れていても数字だけ拾えれば飛ばせる
            import re as _re

            nums = [int(x) for x in _re.findall(r"-?\d+", raw or "")]
            if nums:
                for pid, v in zip(pids, nums):
                    out[pid] = max(0, min(world.cfg.width - 1, v))
                err = "salvaged"
            else:
                err = "parse_fail"
        return out, prompt, raw, err

    def unload(self) -> None:
        """VLM をメモリから降ろして VLA に場所を空ける。"""
        try:
            self._client.post(f"{self.host}/api/generate", json={
                "model": self.model, "keep_alive": 0, "prompt": "", "stream": False})
        except Exception:
            pass

    def warmup(self, world: FlockWorld, png: bytes) -> float:
        """初回はモデルのロードで極端に遅い (実測 80 秒超)。本番から外すため空打ちする。"""
        t0 = time.perf_counter()
        try:
            self(world, png)
        except Exception:
            pass
        return time.perf_counter() - t0


# --------------------------------------------------------------------------
# デモ本体
# --------------------------------------------------------------------------


@dataclass
class FlockResult:
    seeds: int
    frames: list[FlockFrame]
    calls: list[ModelCall]
    landed: dict[int, dict]
    board0: np.ndarray
    board1: np.ndarray
    spawn_order: list
    targets_history: list[dict]
    summary: dict


def run_flock_demo(
    vla_pilot,                 # SmolVLAPilot 相当 (policy/pre/post を持つ)
    commander: QwenCommander,
    cfg: ParachuteConfig | None = None,
    fcfg: FlockConfig | None = None,
    seed: int = 0,
    rcfg: RenderConfig | None = None,
    verbose: bool = True,
    stream=None,
) -> FlockResult:
    """qwen-VL と SmolVLA だけで N 機を降ろす。PD もオラクルも操縦には使わない。"""
    import sys

    import torch

    out = stream or sys.stdout
    cfg = cfg or ParachuteConfig()
    fcfg = fcfg or FlockConfig()
    rcfg = rcfg or RenderConfig()

    board0 = make_terrain(cfg, seed)
    world = FlockWorld(cfg, fcfg, board0.copy(), seed=seed)
    # 最初に数機まとめて投入する。1 機しか飛んでいない状態で VLM に訊いても
    # 「複数機を捌く」という設定が試せないため
    for _ in range(fcfg.n_landers):
        world._maybe_spawn()
        world._last_spawn = -10**9

    from .smolvla_pilot import encode_image, encode_state, task_for

    policy, pre, post = vla_pilot.policy, vla_pilot.pre, vla_pilot.post
    queues: dict[int, deque] = {}
    targets: dict[int, int] = {}
    targets_history: list[dict] = []
    calls: list[ModelCall] = []
    seq = 0
    next_vlm_tick = 0

    def drift(latency: float) -> int:
        """実レイテンシを tick に換算する。slowmo で世界の進みを緩められる。"""
        return min(fcfg.max_drift_ticks,
                   int(round(latency / max(1e-6, cfg.dt * fcfg.slowmo))))

    def advance(n: int) -> None:
        for _ in range(n):
            if world.done:
                break
            world.step({pid: _pop(pid) for pid in list(world.landers)})

    def _pop(pid: int) -> Thrust:
        q = queues.get(pid)
        if q:
            from .smolvla_pilot import decode_action

            return decode_action(q.popleft())
        return Thrust.COAST

    def obs_for(pid: int):
        from .parachute import FlightObservation

        L = world.landers[pid]
        img = render_flock(world, cfg, rcfg, ego_pid=pid)
        return FlightObservation(
            png=to_png(img), image=img, kind=L.kind, rot=L.rot, x=L.x, y=L.y,
            vx=L.vx, vy=L.vy, fuel=L.fuel, tick=world.tick, t=world.t,
            width=cfg.width, height=cfg.height, cfg=cfg, render_cfg=rcfg)

    def log(msg: str) -> None:
        if verbose:
            print(msg, file=out, flush=True)

    log(f"■ 複数機デモ  機数={fcfg.n_landers}  スローモーション×{fcfg.slowmo}  "
        f"VLM間隔={fcfg.vlm_every}tick  seed={seed}")
    log(f"  上位 {commander.model} (どこへ) / 下位 SmolVLA (どうやって)  — PD もオラクルも不使用")

    if fcfg.warmup:
        # 初回はモデルのロードで極端に遅い。本番の計測から外す
        img0 = render_flock(world, cfg, rcfg, ego_pid=None)
        w_vlm = commander.warmup(world, to_png(img0))
        t0 = time.perf_counter()
        pid0 = next(iter(world.landers))
        o0 = obs_for(pid0)
        with torch.inference_mode():
            policy.predict_action_chunk(pre({
                "observation.state": torch.from_numpy(encode_state(o0, 0)),
                "observation.images.camera1": encode_image(o0),
                "task": task_for(0)}))
        w_vla = time.perf_counter() - t0
        log(f"  暖機: VLM {w_vlm:.1f}s / VLA {w_vla:.1f}s (以降の計測から除外)")

    while not world.done:
        # --- 上位: qwen が全機の目標列を決める ---------------------------
        if world.landers and world.tick >= next_vlm_tick:
            img = render_flock(world, cfg, rcfg, ego_pid=None)
            png = to_png(img)
            roster = {pid: (L.kind, L.col, round(L.y, 1), round(L.vx, 2), L.fuel)
                      for pid, L in sorted(world.landers.items())}
            t0 = time.perf_counter()
            assign, prompt, raw, err = commander(world, png)
            latency = time.perf_counter() - t0
            d = drift(latency)
            seq += 1
            calls.append(ModelCall(
                seq=seq, model=commander.model, role="どこへ", tick=world.tick,
                t=round(world.t, 2), pids=sorted(world.landers), latency_s=round(latency, 3),
                drift_ticks=d, prompt=prompt, raw=raw, parsed={"assign": assign},
                error=err, png_b64=base64.b64encode(png).decode("ascii"),
                inputs={"roster": {str(k): v for k, v in roster.items()}},
                note="全機ぶんを 1 リクエストにまとめている",
            ))
            log(f"\n{'─'*66}\n[{seq}] {commander.model} ← 全機の配置画像  t={world.t:.2f}s")
            log(flock_ascii(world, targets))
            log(f"  入力(画像以外): " + ", ".join(
                f"#{k}:{v[0]} 列{v[1]} 行{v[2]} vx{v[3]:+.2f} 燃{v[4]}" for k, v in roster.items()))
            log(f"  生応答: {str(raw).strip()[:150]}")
            log(f"  解釈: 目標列 {assign or '(なし)'}{' ⚠'+str(err) if err else ''}")
            log(f"  コスト: {latency:.2f}s → {d} tick 漂流 (等倍なら {drift(latency)*fcfg.slowmo:.0f} tick)")
            if assign:
                for pid, col in assign.items():
                    if targets.get(pid) != col:
                        queues.pop(pid, None)   # 指令が変わったら行動列を捨てて出し直す
                targets.update(assign)
                targets_history.append({"tick": world.tick, "targets": dict(targets)})
            next_vlm_tick = world.tick + fcfg.vlm_every
            advance(d)

        if world.done:
            break

        # --- 下位: SmolVLA が行動列を作る --------------------------------
        for pid in list(world.landers):
            if queues.get(pid):
                continue
            obs = obs_for(pid)
            tgt = targets.get(pid)
            batch = {
                "observation.state": torch.from_numpy(encode_state(obs, tgt)),
                "observation.images.camera1": encode_image(obs),
                "task": task_for(tgt),
            }
            t0 = time.perf_counter()
            with torch.inference_mode():
                chunk = post(policy.predict_action_chunk(pre(batch)))
            latency = time.perf_counter() - t0
            arr = chunk.detach().float().cpu().numpy()
            if arr.ndim == 3:
                arr = arr[0]
            queues[pid] = deque(arr[: policy.config.n_action_steps])
            d = drift(latency)
            seq += 1
            from .smolvla_pilot import decode_action

            plan = [decode_action(v) for v in list(queues[pid])[:12]]
            calls.append(ModelCall(
                seq=seq, model="smolvla", role="どうやって", tick=world.tick,
                t=round(world.t, 2), pids=[pid], latency_s=round(latency, 3), drift_ticks=d,
                prompt=task_for(tgt), raw=json.dumps({"chunk_len": int(arr.shape[0]),
                    "first12": [[round(float(x), 3) for x in v[:2]] for v in arr[:12]]}),
                parsed={"target_col": tgt, "plan12": [int(a) for a in plan]},
                png_b64=base64.b64encode(obs.png).decode("ascii"),
                inputs={"state6": [round(float(x), 3) for x in encode_state(obs, tgt)],
                        "sensors": {"col": round(obs.col, 2), "y": round(obs.y, 2),
                                    "vx": round(obs.vx, 3), "vy": round(obs.vy, 3),
                                    "fuel": obs.fuel}},
                note=f"1 回で {int(arr.shape[0])} 手ぶん (= {arr.shape[0]*cfg.dt:.1f} 論理秒) を出力",
            ))
            log(f"\n[{seq}] smolvla ← 機 #{pid} 視点の画像 (白枠が自機)  t={world.t:.2f}s")
            log(f"  入力(画像以外): 指令=列{tgt} / state6={[round(float(x),2) for x in encode_state(obs,tgt)]}")
            log(f"  出力: {int(arr.shape[0])} 手ぶんの行動列。先頭12手 = "
                + " ".join({0:'-',1:'◀',2:'▶',3:'▼'}[int(a)] for a in plan))
            log(f"  コスト: {latency:.2f}s → {d} tick 漂流")
            if calls:
                world.frames and setattr(world.frames[-1], "call_seq", seq)
            advance(d)
            break   # 1 tick に 1 機だけ推論する (8GB で同時に走らせない)

        advance(1)

    # --- 採点 (ここで初めてオラクルを使う) --------------------------------
    per = {}
    for pid, info in world.landed.items():
        base = board0
        reg = landing_regret(base, info["kind"], info["rot"], info["col"])
        orc = target_column(base, info["kind"], info["rot"])
        per[pid] = {**info, "regret": None if reg is None else round(reg, 3),
                    "oracle_col": orc, "commanded_col": targets.get(pid),
                    "col_error": None if orc is None else abs(info["col"] - orc)}
    n = max(1, len(per))
    summary = {
        "landed": len(world.landed), "n_landers": fcfg.n_landers, "ticks": world.tick,
        "sim_seconds": round(world.t, 2), "slowmo": fcfg.slowmo,
        "holes_created": sum(v["holes_created"] for v in per.values()),
        "soft_landings": sum(1 for v in per.values() if v["soft"]),
        "mean_regret": round(sum(v["regret"] or 1.0 for v in per.values()) / n, 3),
        "mean_col_error": round(sum(v["col_error"] or 9 for v in per.values()) / n, 2),
        "vlm_calls": sum(1 for c in calls if c.role == "どこへ"),
        "vla_calls": sum(1 for c in calls if c.role == "どうやって"),
        "vlm_latency_mean": round(sum(c.latency_s for c in calls if c.role == "どこへ")
                                  / max(1, sum(1 for c in calls if c.role == "どこへ")), 3),
        "vla_latency_mean": round(sum(c.latency_s for c in calls if c.role == "どうやって")
                                  / max(1, sum(1 for c in calls if c.role == "どうやって")), 3),
        "wall_seconds": round(sum(c.latency_s for c in calls), 1),
        "per_lander": per,
    }
    log(f"\n{'═'*66}")
    log(flock_ascii(world, targets))
    log(f"\n着地 {summary['landed']}/{fcfg.n_landers}  軟着陸 {summary['soft_landings']}  "
        f"作った穴 {summary['holes_created']}  平均regret {summary['mean_regret']}  "
        f"平均列誤差 {summary['mean_col_error']}")
    log(f"推論: VLM {summary['vlm_calls']}回 (平均{summary['vlm_latency_mean']}s) / "
        f"VLA {summary['vla_calls']}回 (平均{summary['vla_latency_mean']}s) / "
        f"実時間 合計{summary['wall_seconds']}s")
    for pid, v in sorted(per.items()):
        log(f"  #{pid} {v['kind']}: 指令列{v['commanded_col']} → 着地列{v['col']} "
            f"(参照解{v['oracle_col']}, 誤差{v['col_error']}, regret{v['regret']}) "
            f"接地vy{v['impact_vy']} {'軟着陸' if v['soft'] else '硬着陸'} 穴{v['holes_created']}")

    return FlockResult(seeds=seed, frames=world.frames, calls=calls, landed=world.landed,
                       board0=board0, board1=world.board, spawn_order=world.spawn_order,
                       targets_history=targets_history, summary=summary)


def save_flock(res: FlockResult, path: str, cfg: ParachuteConfig, fcfg: FlockConfig) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "flock_start", "seed": res.seeds,
                            "config": dict(cfg.__dict__), "flock": asdict(fcfg),
                            "board": res.board0.tolist(),
                            "spawn_order": res.spawn_order}, ensure_ascii=False) + "\n")
        for c in res.calls:
            f.write(json.dumps({"event": "call", **asdict(c)}, ensure_ascii=False) + "\n")
        for fr in res.frames:
            f.write(json.dumps({"event": "frame", **asdict(fr)}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"event": "flock_end", **res.summary,
                            "final_board": res.board1.tolist()}, ensure_ascii=False) + "\n")


def load_flock(path: str) -> dict:
    out = {"calls": [], "frames": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ev = d.pop("event")
            if ev == "flock_start":
                out["start"] = d
                out["board"] = np.array(d["board"], dtype=np.uint8)
                out["config"] = d.get("config", {})
                out["flock"] = d.get("flock", {})
            elif ev == "call":
                out["calls"].append(d)
            elif ev == "frame":
                d["landers"] = {int(k): v for k, v in d["landers"].items()}
                out["frames"].append(d)
            elif ev == "flock_end":
                out["end"] = d
                out["final_board"] = np.array(d["final_board"], dtype=np.uint8)
    return out
