"""tetris_vla.parachute — パラシュート降下と推進ジェットによる着地制御。

テトリスの格子落下では、エージェントの判断はほぼ「どの列か」に潰れてしまう。
そこで同じ盤面・同じ目的 (穴を作らない場所に収まる) のまま、物理だけを差し替える:

    ピースはパラシュートで降りてくる。落下は抗力で終端速度に落ち着き、止められない。
    横方向は推進ジェットで加速できるが、**燃料は有限で、慣性は消えない**。

こうすると VLA の性格がはっきり出る。格子落下なら「今どの列に居るか」だけ見て
1 tick 遅れても取り返せるが、こちらは慣性があるので **判断が遅れた分だけ確実に流される**。
1 回 2 秒の推論は、そのまま 2 秒ぶんの漂流になる。

物理は意図的に最小限 (「シミュレーションはやりすぎない」):
  鉛直   vy += (g - k_v*vy) * dt        -> 終端速度 g/k_v に漸近するだけ
  水平   vx += (T*u + wind - k_h*vx) * dt
  ベント パラシュートを絞る = k_v を下げて速く落ちる (取り返しのつかない加速)
回転も空気力学も弾性衝突もない。

観測の切り分け:
  * **自己受容感覚 (位置・速度・燃料) は数値で与える。** 着陸機に IMU が載っているのと同じ。
  * **地形と目標は画像からしか読めない。** どこに降りるべきかは自分で見て決める。
"""

from __future__ import annotations

import json
import os
import math
import random
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .agents import Persona, PERSONAS, plan_placements
from .engine import (
    ROT_CELLS,
    KIND_ID,
    KINDS,
    cell_span,
    count_holes,
    place,
    unique_rotations,
)
from .render import (
    BASE,
    FLAME,
    RenderConfig,
    SETTLED,
    TRAIL,
    decode,
    render_terrain,
    to_png,
)


class Thrust(IntEnum):
    """パラシュート降下中に取れる行動。"""

    COAST = 0      # 何もしない (燃料を使わない)
    LEFT = 1       # 左へ噴射
    RIGHT = 2      # 右へ噴射
    VENT = 3       # パラシュートを絞って落下を速める (横制御の猶予を捨てる)


THRUST_WORD = {
    Thrust.COAST: "none", Thrust.LEFT: "left", Thrust.RIGHT: "right", Thrust.VENT: "vent",
}
WORD_THRUST = {
    "none": Thrust.COAST, "coast": Thrust.COAST, "hold": Thrust.COAST, "": Thrust.COAST,
    "left": Thrust.LEFT, "l": Thrust.LEFT,
    "right": Thrust.RIGHT, "r": Thrust.RIGHT,
    "vent": Thrust.VENT, "down": Thrust.VENT, "drop": Thrust.VENT,
}
THRUST_JA = {
    Thrust.COAST: "噴射なし", Thrust.LEFT: "左へ噴射",
    Thrust.RIGHT: "右へ噴射", Thrust.VENT: "ベント(加速降下)",
}


@dataclass(frozen=True)
class ParachuteConfig:
    width: int = 10
    height: int = 20
    dt: float = 0.12            # 1 tick の論理秒
    gravity: float = 9.0        # セル/s^2
    drag_v: float = 3.2         # 鉛直抗力。終端速度 = gravity / drag_v
    drag_h: float = 1.4         # 水平抗力。時定数 = 1/drag_h
    thrust: float = 2.8         # 横ジェットの加速度 (セル/s^2)
    vent_drag_scale: float = 0.35   # ベント中の抗力倍率 (小さいほど速く落ちる)
    #: 一定横風。既定で 0 でないのが重要 — 無風だと「何もしない」が強すぎて
    #: 制御の良し悪しが score に出ない (random がほぼ PD に並んでしまう)
    wind: float = 0.8
    gust: float = 1.6           # 風のゆらぎ振幅
    gust_period: float = 3.4
    fuel: int = 60              # 噴射できる tick 数
    max_ticks: int = 260
    soft_vy: float = 3.0        # これ以下の接地速度なら軟着陸 (終端速度で降りれば満たす)
    soft_vx: float = 1.2
    trail_len: int = 14         # 画像に描く軌跡の長さ
    seed: int = 0

    @property
    def terminal_vy(self) -> float:
        return self.gravity / self.drag_v

    @property
    def terminal_vx(self) -> float:
        return self.thrust / self.drag_h


@dataclass
class Lander:
    kind: str
    rot: int
    x: float          # 回転ボックス左上の列座標 (小数)
    y: float          # 同・行座標 (小数)
    vx: float = 0.0
    vy: float = 0.0
    fuel: int = 0
    trail: list[tuple[float, float]] = field(default_factory=list)

    @property
    def col(self) -> int:
        """占有セルの最左列 (格子に丸めた値)。"""
        return int(round(self.x)) + cell_span(self.kind, self.rot)[2]

    def cells(self, x: float | None = None, y: float | None = None) -> tuple[tuple[int, int], ...]:
        ix = int(round(self.x if x is None else x))
        iy = int(round(self.y if y is None else y))
        return tuple((iy + dy, ix + dx) for dy, dx in ROT_CELLS[self.kind][self.rot % 4])


@dataclass
class FlightFrame:
    """1 tick 分の記録。SVG アニメと再生はこれだけあれば作れる。"""

    tick: int
    t: float
    x: float
    y: float
    vx: float
    vy: float
    fuel: int
    action: int
    wind: float
    landed: bool = False
    decision_idx: int | None = None   # この tick に届いた判断


@dataclass
class FlightDecision:
    action: Thrust
    target_col: int | None = None
    why: str = ""
    latency_s: float = 0.0
    raw: str | None = None
    prompt: str | None = None
    error: str | None = None
    submit_tick: int = 0
    deliver_tick: int = 0
    png: bytes | None = None      # そのとき実際にモデルへ渡した画像
    state: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# ワールド
# --------------------------------------------------------------------------


class ParachuteWorld:
    """地形 + 1 機の降下体。純粋・決定的。"""

    def __init__(self, cfg: ParachuteConfig, board: np.ndarray, lander: Lander) -> None:
        self.cfg = cfg
        self.board = board
        self.lander = lander
        self.tick = 0
        self.t = 0.0
        self.landed = False
        self.timeout = False
        self.impact_vy = 0.0
        self.impact_vx = 0.0
        self.fuel_used = 0
        self.wall_hits = 0
        self.frames: list[FlightFrame] = []

    # -- 判定 -------------------------------------------------------------

    def blocked(self, x: float, y: float) -> bool:
        H, W = self.board.shape
        for r, c in self.lander.cells(x, y):
            if c < 0 or c >= W or r >= H:
                return True
            if r >= 0 and self.board[r, c]:
                return True
        return False

    def wind_at(self, t: float) -> float:
        c = self.cfg
        if not c.gust:
            return c.wind
        return c.wind + c.gust * math.sin(2 * math.pi * t / max(1e-6, c.gust_period))

    # -- 1 tick -----------------------------------------------------------

    def step(self, action: Thrust) -> FlightFrame:
        c = self.cfg
        L = self.lander
        if self.landed or self.timeout:
            return self.frames[-1]

        act = Thrust(action)
        if act in (Thrust.LEFT, Thrust.RIGHT) and L.fuel <= 0:
            act = Thrust.COAST  # 燃料切れ。噴射しようとしても何も起きない

        u = {Thrust.LEFT: -1.0, Thrust.RIGHT: 1.0}.get(act, 0.0)
        if u:
            L.fuel -= 1
            self.fuel_used += 1

        wind = self.wind_at(self.t)
        drag_v = c.drag_v * (c.vent_drag_scale if act == Thrust.VENT else 1.0)

        # 明示的オイラー法。これ以上凝る必要はない
        L.vx += (c.thrust * u + wind - c.drag_h * L.vx) * c.dt
        L.vy += (c.gravity - drag_v * L.vy) * c.dt

        nx = L.x + L.vx * c.dt
        ny = L.y + L.vy * c.dt

        # 壁: 水平方向にはめり込まず、速度を失う
        _, _, min_dx, max_dx = cell_span(L.kind, L.rot)
        lo, hi = -min_dx, c.width - 1 - max_dx
        if nx < lo:
            nx, L.vx, self.wall_hits = float(lo), 0.0, self.wall_hits + 1
        elif nx > hi:
            nx, L.vx, self.wall_hits = float(hi), 0.0, self.wall_hits + 1

        if self.blocked(nx, ny):
            # 接地。横だけ動けるなら滑り込ませてから止める
            if not self.blocked(nx, L.y):
                L.x = nx
            self.impact_vy, self.impact_vx = L.vy, L.vx
            self.landed = True
            L.vx = L.vy = 0.0
        else:
            L.x, L.y = nx, ny

        L.trail.append((L.x, L.y))
        if len(L.trail) > c.trail_len:
            L.trail.pop(0)

        self.tick += 1
        self.t += c.dt
        if self.tick >= c.max_ticks and not self.landed:
            self.timeout = True

        frame = FlightFrame(
            tick=self.tick, t=round(self.t, 3), x=round(L.x, 4), y=round(L.y, 4),
            vx=round(L.vx, 4), vy=round(L.vy, 4), fuel=L.fuel, action=int(act),
            wind=round(wind, 4), landed=self.landed,
        )
        self.frames.append(frame)
        return frame

    def settle(self) -> np.ndarray:
        """接地位置で地形に固定した盤面を返す。"""
        L = self.lander
        return place(self.board, L.kind, L.rot, int(round(L.x)), int(round(L.y)))


# --------------------------------------------------------------------------
# 地形の生成
# --------------------------------------------------------------------------


def make_terrain(cfg: ParachuteConfig, seed: int, max_height: int = 7) -> np.ndarray:
    """テトリスらしい起伏のある地形を作る。満杯行は作らない。

    ランダムノイズではなく「柱を積む」やり方にしてあるので、
    平らな着地適地と、穴を作ってしまう窪みの両方が自然に生まれる。
    """
    rng = random.Random(seed * 104729 + 7)
    board = np.zeros((cfg.height, cfg.width), dtype=np.uint8)
    h = rng.randint(1, 3)
    for c in range(cfg.width):
        h = max(0, min(max_height, h + rng.randint(-2, 2)))
        for k in range(h):
            board[cfg.height - 1 - k, c] = rng.randint(1, 7)
    # 満杯行があると消えてしまうので必ず 1 マス空ける
    for r in range(cfg.height):
        if board[r].all():
            board[r, rng.randrange(cfg.width)] = 0
    return board


def spawn_lander(cfg: ParachuteConfig, kind: str, seed: int,
                 board: np.ndarray | None = None) -> Lander:
    """投入位置を決める。

    board を渡すと、**目標列から遠い側** に出す。真上に落とすと「何もしない」が
    正解になってしまい、推進制御の良し悪しが測れなくなるため。
    """
    rng = random.Random(seed * 7919 + 13)
    rot = rng.choice(unique_rotations(kind))
    _, _, min_dx, max_dx = cell_span(kind, rot)
    lo, hi = -min_dx, cfg.width - 1 - max_dx
    x = float(rng.randint(lo, hi))
    if board is not None:
        tgt = target_column(board, kind, rot)
        if tgt is not None:
            # 目標の反対側へ 4〜6 列ずらす。壁に貼り付けると風上横断が
            # 燃料的に不可能になり、制御の巧拙ではなく詰みで決まってしまう
            offset = rng.randint(4, 6)
            side = -1 if tgt > (lo + hi) / 2 else 1
            x = float(max(lo, min(hi, tgt - min_dx + side * offset)))
    y = float(-cell_span(kind, rot)[0])
    return Lander(kind=kind, rot=rot, x=x, y=y, vx=0.0, vy=0.0, fuel=cfg.fuel)


def target_column(board: np.ndarray, kind: str, rot: int, persona: Persona | None = None) -> int | None:
    """その回転で穴を作らずに済む最良の列 (Dellacherie 参照解)。"""
    plans = [p for p in plan_placements(board, kind, persona or PERSONAS["neutral"])
             if p.rot == rot % 4]
    return plans[0].col if plans else None


# --------------------------------------------------------------------------
# 観測とパイロット
# --------------------------------------------------------------------------


@dataclass
class FlightObservation:
    """降下体に渡る情報のすべて。

    地形は画像からしか読めない。位置・速度・燃料は機体センサの値として数値で渡す
    (着陸機に IMU が載っているのと同じ扱い)。
    """

    png: bytes
    image: Image.Image
    kind: str
    rot: int
    x: float
    y: float
    vx: float
    vy: float
    fuel: int
    tick: int
    t: float
    width: int
    height: int
    cfg: ParachuteConfig
    render_cfg: RenderConfig
    persona: Persona = PERSONAS["neutral"]

    @property
    def col(self) -> float:
        return self.x + cell_span(self.kind, self.rot)[2]


class Pilot(Protocol):
    name: str

    def decide(self, obs: FlightObservation) -> FlightDecision: ...


class PDPilot:
    """参照実装。画像から地形を読み、目標列へ PD 制御で寄せる。

    「どこへ降りるか」は VLA と同じくピクセルから決める。違うのは決め方だけ。
    """

    name = "pd"

    def __init__(self, kp: float = 1.6, vmax: float = 2.6, deadband: float = 0.25,
                 vent_margin: float = 0.6, vent_min_clearance: float = 8.0) -> None:
        self.kp, self.vmax, self.deadband = kp, vmax, deadband
        self.vent_margin = vent_margin
        #: 地面までこれだけ余裕がないとベントしない。直前に絞ると硬着陸になる
        self.vent_min_clearance = vent_min_clearance

    def decide(self, obs: FlightObservation) -> FlightDecision:
        t0 = time.perf_counter()
        terrain = decode(obs.image, obs.width, obs.height, obs.render_cfg).settled
        target = target_column(terrain, obs.kind, obs.rot, obs.persona)
        dt = time.perf_counter() - t0
        if target is None:
            return FlightDecision(Thrust.COAST, None, "着地可能列なし",
                                  latency_s=dt, error="no_placement")
        err = target - obs.col
        want_vx = max(-self.vmax, min(self.vmax, self.kp * err))
        dv = want_vx - obs.vx
        if dv > self.deadband:
            act = Thrust.RIGHT
        elif dv < -self.deadband:
            act = Thrust.LEFT
        elif (abs(err) < self.vent_margin and abs(obs.vx) < self.deadband
              and self._clearance(terrain, obs) > self.vent_min_clearance):
            act = Thrust.VENT   # 揃っていて高度も十分あるなら、降下を早めて風に晒される時間を減らす
        else:
            act = Thrust.COAST
        return FlightDecision(act, target, f"err={err:+.1f} vx={obs.vx:+.1f}", latency_s=dt)

    @staticmethod
    def _clearance(terrain: np.ndarray, obs: FlightObservation) -> float:
        """真下の地形までの余裕 (行)。画像から読んだ地形だけで計算する。"""
        H, W = terrain.shape
        _, max_dy, min_dx, max_dx = cell_span(obs.kind, obs.rot)
        ix = int(round(obs.x))
        ground = H
        for dx in range(min_dx, max_dx + 1):
            c = ix + dx
            if not (0 <= c < W):
                continue
            rows = np.nonzero(terrain[:, c])[0]
            ground = min(ground, int(rows[0]) if rows.size else H)
        return ground - (obs.y + max_dy)


class RandomPilot:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def decide(self, obs: FlightObservation) -> FlightDecision:
        return FlightDecision(self.rng.choice(list(Thrust)), None, "random")


# --------------------------------------------------------------------------
# VLA パイロット
# --------------------------------------------------------------------------

#: 日本語版。読みやすいが、qwen2.5vl:3b では prefill 1579tok / 出力 29tok となり
#: 英語版の約 5 倍遅かった (M2 8GB 実測)。既定は en。
FLIGHT_PROMPT_JA = """あなたはパラシュートで降下中のテトリスのピース ({kind}) です。推進ジェットで着地点を狙います。

画像: 暗いブロック=地面の地形、明るいブロック=あなた、小さな点=直前の軌跡 (流れる向き)。
盤面は 横 {width} 列 (左端 0 〜 右端 {maxcol})、縦 {height} 行。

センサ値: 左端が列 {col:.1f}、高さ 行 {y:.1f}、横速度 {vx:+.2f} 列/秒、落下速度 {vy:.2f} 行/秒、燃料 {fuel}
目的: 下に空洞ができない場所 — 平らな面か、自分の形に合う窪み — に着地する。
慣性が残るので目標の手前で逆噴射すること。

JSON のみで答えてください:
{{"thrust":"left"|"right"|"none"|"vent","target_col":<0〜{maxcol}>,"why":"<15字以内>"}}"""

#: 英語版 (既定)。同じ情報量で prefill 1217tok / 出力 19tok。
FLIGHT_PROMPT_EN = """You are a tetromino ({kind}) descending by parachute onto the terrain below.
Image: dark blocks = ground terrain, bright block = you, small dots = your recent trail (which way you drift).
Board is {width} columns (0=left .. {maxcol}=right) and {height} rows.

Sensors: your left edge is at column {col:.1f}, height row {y:.1f}, sideways speed {vx:+.2f} col/s, fall speed {vy:.2f} row/s, fuel {fuel}.
Goal: land where NO cavity forms underneath you - on a flat surface, or in a notch that matches your shape.
Inertia persists, so start braking before you reach the target.

Reply with JSON only:
{{"thrust":"left"|"right"|"none"|"vent","target_col":<0-{maxcol}>,"why":"<=6 words"}}"""

FLIGHT_PROMPTS = {"ja": FLIGHT_PROMPT_JA, "en": FLIGHT_PROMPT_EN}


def build_flight_prompt(obs: FlightObservation, lang: str = "en") -> str:
    return FLIGHT_PROMPTS[lang].format(
        kind=obs.kind, width=obs.width, maxcol=obs.width - 1, height=obs.height,
        col=obs.col, y=obs.y, vx=obs.vx, vy=obs.vy, fuel=obs.fuel,
    )


_REFUSAL = re.compile(r"(cannot|can't|unable|sorry|申し訳|できません|わかりません)", re.I)


def parse_flight(raw: str, width: int) -> tuple[FlightDecision | None, str | None]:
    """生応答を行動に変換する。thrust さえ読めれば target_col が無くても飛べる。"""
    if not raw or not raw.strip():
        return None, "backend_error"
    from .agents import _extract_json

    obj = _extract_json(raw)
    if obj is None:
        m = re.search(r'"?thrust"?\s*[:=]\s*"?(left|right|none|vent|coast)"?', raw, re.I)
        if m:
            obj = {"thrust": m.group(1).lower(), "why": "salvaged"}
        elif _REFUSAL.search(raw):
            return None, "refusal"
        else:
            return None, "parse_fail"

    word = str(obj.get("thrust", obj.get("action", ""))).strip().lower()
    if word not in WORD_THRUST:
        return None, "parse_fail"
    act = WORD_THRUST[word]

    target = obj.get("target_col", obj.get("col"))
    err = None
    try:
        target = int(target) if target is not None else None
    except (TypeError, ValueError):
        target, err = None, "bad_target"
    if target is not None and not (0 <= target < width):
        target = max(0, min(width - 1, target))
        err = "out_of_range"
    return FlightDecision(act, target, str(obj.get("why", ""))[:40], raw=raw), err


class VLAPilot:
    """ローカル VLA を操縦席に座らせる。"""

    def __init__(self, backend, persona: Persona | None = None, lang: str = "en") -> None:
        self.backend = backend
        self.persona = persona or PERSONAS["neutral"]
        self.lang = lang
        self.name = f"vla:{backend.name}"

    def decide(self, obs: FlightObservation) -> FlightDecision:
        prompt = build_flight_prompt(obs, self.lang)
        try:
            raw, latency = self.backend.generate(prompt, obs)
        except Exception as exc:
            return FlightDecision(Thrust.COAST, None, "", latency_s=0.0, prompt=prompt,
                                  error=f"backend_error:{type(exc).__name__}")
        d, err = parse_flight(raw, obs.width)
        if d is None:
            # 読めなかったら噴射しない。燃料を無駄にしないのが最も安全な既定値
            return FlightDecision(Thrust.COAST, None, "", latency_s=latency, raw=raw,
                                  prompt=prompt, error=err)
        d.latency_s, d.prompt = latency, prompt
        d.error = err
        return d


@dataclass
class MockFlightConfig:
    skill: float = 0.7          # PD 参照解と同じ行動を選ぶ確率
    latency_s: float = 1.4
    latency_sigma: float = 0.3
    p_malformed: float = 0.05
    p_refuse: float = 0.02
    sleep: bool = False


class MockFlightBackend:
    """モデル無しで飛ばすためのスタブ。PD 参照解にノイズを載せて返す。"""

    name = "mock"

    def __init__(self, cfg: MockFlightConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or MockFlightConfig()
        self.rng = random.Random(seed)
        self._pd = PDPilot()

    def generate(self, prompt: str, obs: FlightObservation) -> tuple[str, float]:
        c = self.cfg
        latency = max(0.05, self.rng.lognormvariate(math.log(c.latency_s), c.latency_sigma))
        if c.sleep:
            time.sleep(latency)
        r = self.rng.random()
        if r < c.p_malformed:
            return "I think we should drift a bit to the left maybe?", latency
        if r < c.p_malformed + c.p_refuse:
            return "I cannot determine the terrain from this image.", latency
        ref = self._pd.decide(obs)
        if self.rng.random() < c.skill:
            act, target = ref.action, ref.target_col
        else:
            act = self.rng.choice(list(Thrust))
            target = ref.target_col
        return json.dumps(
            {"thrust": THRUST_WORD[act], "target_col": target, "why": "mock"},
            ensure_ascii=False,
        ), latency


# --------------------------------------------------------------------------
# 飛行の実行
# --------------------------------------------------------------------------


@dataclass
class FlightResult:
    pilot: str
    kind: str
    seed: int
    landed: bool
    timeout: bool
    ticks: int
    sim_seconds: float
    landing_col: int | None
    oracle_col: int | None
    col_error: int | None
    regret: float | None
    impact_vy: float
    impact_vx: float
    soft: bool
    holes_before: int
    holes_after: int
    holes_created: int
    fuel_used: int
    fuel_left: int
    wall_hits: int
    decisions: int
    mean_latency_s: float
    errors: dict[str, int]
    score: float
    breakdown: dict[str, float]
    frames: list[FlightFrame] = field(default_factory=list)
    decision_log: list[FlightDecision] = field(default_factory=list)
    board: np.ndarray | None = None
    final_board: np.ndarray | None = None

    def summary(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("frames", "decision_log", "board", "final_board")}
        return d


def landing_regret(board: np.ndarray, kind: str, rot: int, col: int,
                   persona: Persona | None = None) -> float | None:
    """着地位置が Dellacherie 参照解からどれだけ劣るか [0,1]。0 が最良。

    列誤差だけだと「隣の列でも同じくらい良い」場合に不当に減点されるので、
    実際の設置品質そのものを見る。テトリス側の regret と同じ定義。
    """
    plans = plan_placements(board, kind, persona or PERSONAS["neutral"])
    if not plans:
        return None
    best, worst = plans[0].score, plans[-1].score
    span = max(1e-9, best - worst)
    for p in plans:
        if p.col == col and p.rot == rot % 4:
            return float((best - p.score) / span)
    return 1.0


def score_flight(res_kwargs: dict, cfg: ParachuteConfig) -> tuple[float, dict[str, float]]:
    """0〜1000 点。着地品質を主軸に、命中・軟着陸・燃費を従える。

    重み付けの根拠: 「何もしない」でも軟着陸と燃料温存は達成できてしまうので、
    そこに大きな配点をすると無為が強くなる。主目的 (良い場所に降りる) を
    品質 0.50 + 命中 0.20 で 7 割に置く。
    """
    if res_kwargs["timeout"] or not res_kwargs["landed"]:
        return 0.0, {"quality": 0.0, "accuracy": 0.0, "softness": 0.0, "economy": 0.0}
    regret = res_kwargs.get("regret")
    quality = 1.0 if regret is None else max(0.0, 1.0 - regret)
    err = res_kwargs["col_error"]
    accuracy = 0.0 if err is None else max(0.0, 1.0 - err / 4.0)
    vy, vx = abs(res_kwargs["impact_vy"]), abs(res_kwargs["impact_vx"])
    softness = min(1.0, max(0.0, 1.0 - max(0.0, vy - cfg.soft_vy) / 4.0)) * \
        min(1.0, max(0.0, 1.0 - max(0.0, vx - cfg.soft_vx) / 3.0))
    economy = res_kwargs["fuel_left"] / max(1, cfg.fuel)
    parts = {"quality": round(quality, 3), "accuracy": round(accuracy, 3),
             "softness": round(softness, 3), "economy": round(economy, 3)}
    total = 1000.0 * (0.50 * quality + 0.20 * accuracy + 0.20 * softness + 0.10 * economy)
    return round(total, 1), parts


def fly(
    pilot: Pilot,
    cfg: ParachuteConfig | None = None,
    seed: int = 0,
    kind: str | None = None,
    persona: Persona | None = None,
    render_cfg: RenderConfig | None = None,
    decision_period: int = 4,
    wall_clock_latency: bool = True,
    on_frame=None,
    on_decision=None,
    keep_png: bool = True,
) -> FlightResult:
    """1 機を降下させる。

    decision_period tick ごとに判断を求めるが、**推論にかかった実時間は
    そのまま tick に換算されて機体を流す** (wall_clock_latency=True)。
    2 秒考えれば 2 秒ぶん漂流したあとに、その古い判断が効き始める。
    """
    cfg = cfg or ParachuteConfig()
    render_cfg = render_cfg or RenderConfig()
    persona = persona or PERSONAS["neutral"]
    rng = random.Random(seed)
    kind = kind or rng.choice(KINDS)

    board = make_terrain(cfg, seed)
    lander = spawn_lander(cfg, kind, seed, board)
    world = ParachuteWorld(cfg, board, lander)
    holes_before = count_holes(board)
    oracle = target_column(board, kind, lander.rot, persona)

    decisions: list[FlightDecision] = []
    errors: dict[str, int] = {}
    action = Thrust.COAST
    next_decision_tick = 0

    while not world.landed and not world.timeout:
        if world.tick >= next_decision_tick:
            img = render_flight(world, cfg, render_cfg)
            png = to_png(img)
            obs = FlightObservation(
                png=png, image=img, kind=kind, rot=lander.rot,
                x=lander.x, y=lander.y, vx=lander.vx, vy=lander.vy, fuel=lander.fuel,
                tick=world.tick, t=world.t, width=cfg.width, height=cfg.height,
                cfg=cfg, render_cfg=render_cfg, persona=persona,
            )
            submit = world.tick
            d = pilot.decide(obs)
            d.submit_tick = submit
            d.state = {"x": round(lander.x, 3), "y": round(lander.y, 3),
                       "vx": round(lander.vx, 3), "vy": round(lander.vy, 3),
                       "fuel": lander.fuel, "col": round(obs.col, 2)}
            if keep_png:
                d.png = png
            if d.error:
                key = d.error.split(":")[0]
                errors[key] = errors.get(key, 0) + 1

            # 推論にかかった実時間を tick に換算し、その間は前の行動を続ける
            drift = 0
            if wall_clock_latency:
                drift = int(round(d.latency_s / cfg.dt))
            for _ in range(drift):
                if world.landed or world.timeout:
                    break
                f = world.step(action)
                if on_frame:
                    on_frame(world, f, None)

            d.deliver_tick = world.tick
            action = d.action
            decisions.append(d)
            d_idx = len(decisions) - 1
            if on_decision:
                on_decision(world, d, d_idx)
            next_decision_tick = world.tick + decision_period
            if world.landed or world.timeout:
                if world.frames:
                    world.frames[-1].decision_idx = d_idx
                break

        f = world.step(action)
        if decisions and f.decision_idx is None and f.tick == decisions[-1].deliver_tick + 1:
            f.decision_idx = len(decisions) - 1
        if on_frame:
            on_frame(world, f, action)

    final = world.settle() if world.landed else board
    landing_col = lander.col if world.landed else None
    regret = (None if landing_col is None
              else landing_regret(board, kind, lander.rot, landing_col, persona))
    kw = dict(
        landed=world.landed, timeout=world.timeout,
        col_error=None if (landing_col is None or oracle is None) else abs(landing_col - oracle),
        impact_vy=world.impact_vy, impact_vx=world.impact_vx,
        holes_created=max(0, count_holes(final) - holes_before),
        fuel_left=lander.fuel, regret=regret,
    )
    score, parts = score_flight(kw, cfg)
    lat = [d.latency_s for d in decisions] or [0.0]
    return FlightResult(
        pilot=getattr(pilot, "name", "?"), kind=kind, seed=seed,
        landed=world.landed, timeout=world.timeout, ticks=world.tick,
        sim_seconds=round(world.t, 2), landing_col=landing_col, oracle_col=oracle,
        col_error=kw["col_error"], regret=None if regret is None else round(regret, 4),
        impact_vy=round(world.impact_vy, 3),
        impact_vx=round(world.impact_vx, 3),
        soft=abs(world.impact_vy) <= cfg.soft_vy and abs(world.impact_vx) <= cfg.soft_vx,
        holes_before=holes_before, holes_after=count_holes(final),
        holes_created=kw["holes_created"], fuel_used=world.fuel_used,
        fuel_left=lander.fuel, wall_hits=world.wall_hits, decisions=len(decisions),
        mean_latency_s=round(sum(lat) / len(lat), 4), errors=errors,
        score=score, breakdown=parts, frames=world.frames, decision_log=decisions,
        board=board, final_board=final,
    )


# --------------------------------------------------------------------------
# 描画 (エージェントに渡る画像)
# --------------------------------------------------------------------------

PARACHUTE_COLOR = (225, 225, 235)
CORD_COLOR = (140, 140, 150)


def render_flight(
    world: ParachuteWorld, cfg: ParachuteConfig, render_cfg: RenderConfig | None = None,
    action: Thrust | None = None,
) -> Image.Image:
    """降下体から見える 1 枚。

    地形は格子に揃えて描くので `render.decode` でそのまま読める。
    降下体は小数座標のまま描かれ、直前の軌跡が点で残る (流れが画像から見える)。
    """
    render_cfg = render_cfg or RenderConfig()
    img = render_terrain(world.board, render_cfg)
    d = ImageDraw.Draw(img)
    cw = render_cfg.cell_px
    top = render_cfg.header_px
    L = world.lander

    def px(cx: float, cy: float) -> tuple[float, float]:
        return cx * cw, top + cy * cw

    # 軌跡 (古いほど暗い点)
    n = len(L.trail)
    for i, (tx, ty) in enumerate(L.trail[:-1]):
        f = (i + 1) / max(1, n)
        col = tuple(int(TRAIL[k] * (0.3 + 0.7 * f)) for k in range(3))
        cxc, cyc = px(tx + 0.5, ty + 0.5)
        d.ellipse([cxc - 1, cyc - 1, cxc + 1, cyc + 1], fill=col)

    # パラシュート (見た目だけ。物理は抗力係数がすべて)
    _, _, min_dx, max_dx = cell_span(L.kind, L.rot)
    cx0, _ = px(L.x + (min_dx + max_dx + 1) / 2.0, 0)
    canopy_y = top + (L.y - 1.6) * cw
    if canopy_y > top - cw:
        rx = cw * 1.5
        d.arc([cx0 - rx, canopy_y - cw * 0.7, cx0 + rx, canopy_y + cw * 0.9],
              180, 360, fill=PARACHUTE_COLOR, width=2)
        for s in (-1, 0, 1):
            d.line([cx0 + s * rx * 0.9, canopy_y + cw * 0.1,
                    cx0, top + (L.y + 0.2) * cw], fill=CORD_COLOR)

    # 機体 (小数座標のまま)
    base = BASE[KIND_ID[L.kind]]
    for dy, dx in ROT_CELLS[L.kind][L.rot % 4]:
        x0, y0 = px(L.x + dx, L.y + dy)
        d.rectangle([x0 + 1, y0 + 1, x0 + cw - 2, y0 + cw - 2], fill=base)

    # 噴射炎 (噴射方向と反対側に出る)
    if action in (Thrust.LEFT, Thrust.RIGHT, Thrust.VENT):
        cxm, cym = px(L.x + (min_dx + max_dx + 1) / 2.0, L.y + 0.5)
        if action == Thrust.RIGHT:
            pts = [(cxm - cw * 0.6, cym - 3), (cxm - cw * 0.6, cym + 3), (cxm - cw * 1.3, cym)]
        elif action == Thrust.LEFT:
            pts = [(cxm + cw * 0.6, cym - 3), (cxm + cw * 0.6, cym + 3), (cxm + cw * 1.3, cym)]
        else:
            pts = [(cxm - 3, cym - cw * 0.8), (cxm + 3, cym - cw * 0.8), (cxm, cym - cw * 1.5)]
        d.polygon(pts, fill=FLAME)

    # ヘッダ: 燃料バーと速度
    if top:
        d.rectangle([0, 0, img.width, top - 1], fill=(10, 10, 12))
        w = int((img.width - 4) * L.fuel / max(1, cfg.fuel))
        d.rectangle([2, 2, 2 + w, 5], fill=(90, 210, 120) if L.fuel > 8 else (230, 90, 70))
        d.text((2, 6), f"vx{L.vx:+.1f} vy{L.vy:.1f} f{L.fuel}", fill=(190, 190, 200))
    return img


class OllamaFlightBackend:
    """実機バックエンド。ローカル Ollama の VLM を操縦席に座らせる。

    降下は数秒で終わるので、レイテンシがそのまま成績になる。実測 (M2 8GB / qwen2.5vl:3b):
      日本語プロンプト + num_predict 96  ->  約 10〜16 秒
      英語プロンプト   + num_predict 28  ->  約 2〜3 秒 (ウォーム時)
    出力トークン数が支配的なので、`why` を短く保つことが最大の高速化になる。
    """

    name = "ollama"

    def __init__(self, model: str = "qwen2.5vl:3b", host: str = "http://localhost:11434",
                 temperature: float = 0.1, num_predict: int = 28, timeout: float = 300.0,
                 keep_alive: str = "30m") -> None:
        import httpx

        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.num_predict, self.keep_alive = temperature, num_predict, keep_alive
        self._client = httpx.Client(timeout=timeout)
        self.name = f"ollama:{model}"
        self.last_stats: dict = {}

    def generate(self, prompt: str, obs: FlightObservation) -> tuple[str, float]:
        import base64

        t0 = time.perf_counter()
        r = self._client.post(f"{self.host}/api/generate", json={
            "model": self.model, "prompt": prompt,
            "images": [base64.b64encode(obs.png).decode("ascii")],
            "stream": False, "format": "json", "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        })
        r.raise_for_status()
        latency = time.perf_counter() - t0
        j = r.json()
        self.last_stats = {
            "prompt_tokens": j.get("prompt_eval_count"),
            "output_tokens": j.get("eval_count"),
            "prefill_s": round(j.get("prompt_eval_duration", 0) / 1e9, 3),
            "decode_s": round(j.get("eval_duration", 0) / 1e9, 3),
        }
        return j.get("response", ""), latency

    def warmup(self, obs: FlightObservation, lang: str = "en") -> float:
        """モデルをメモリに載せておく。1 回目だけ極端に遅いのを本番から外すため。"""
        t0 = time.perf_counter()
        try:
            self.generate(build_flight_prompt(obs, lang), obs)
        except Exception:
            pass
        return time.perf_counter() - t0

    def health(self) -> str:
        r = self._client.get(f"{self.host}/api/tags")
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if self.model not in names:
            raise RuntimeError(
                f"モデル {self.model} が Ollama にありません。導入済み: {names or '(なし)'}\n"
                f"  ollama pull {self.model}"
            )
        return f"ollama ok: {names}"


# --------------------------------------------------------------------------
# 記録と再生
# --------------------------------------------------------------------------


def save_flight(res: FlightResult, path: str, cfg: ParachuteConfig,
                keep_png: bool = True) -> None:
    """1 回の降下をまるごと JSONL に残す。

    入力画像そのもの (base64) と生応答を含めるので、あとから
    「何を見せて何を答えたか」を完全に再構成できる。
    """
    import base64
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "flight_start", "pilot": res.pilot, "kind": res.kind, "seed": res.seed,
            "config": {k: v for k, v in cfg.__dict__.items()},
            "board": res.board.tolist() if res.board is not None else None,
        }, ensure_ascii=False) + "\n")
        for d in res.decision_log:
            rec = {
                "event": "decision", "submit_tick": d.submit_tick,
                "deliver_tick": d.deliver_tick, "action": int(d.action),
                "action_ja": THRUST_JA[d.action], "target_col": d.target_col,
                "why": d.why, "latency_s": round(d.latency_s, 3), "error": d.error,
                "raw": d.raw, "prompt": d.prompt, "state": d.state,
            }
            if keep_png and d.png:
                rec["png_b64"] = base64.b64encode(d.png).decode("ascii")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for fr in res.frames:
            f.write(json.dumps({"event": "frame", **fr.__dict__}, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "event": "flight_end", **res.summary(),
            "final_board": res.final_board.tolist() if res.final_board is not None else None,
        }, ensure_ascii=False) + "\n")


def load_flight(path: str) -> dict:
    """save_flight が書いた JSONL を読み戻す。"""
    import base64

    out = {"decisions": [], "frames": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ev = d.pop("event")
            if ev == "flight_start":
                out["start"] = d
                out["board"] = np.array(d["board"], dtype=np.uint8) if d.get("board") else None
                out["config"] = d.get("config", {})
            elif ev == "decision":
                if d.get("png_b64"):
                    d["png"] = base64.b64decode(d["png_b64"])
                out["decisions"].append(d)
            elif ev == "frame":
                out["frames"].append(d)
            elif ev == "flight_end":
                out["end"] = d
                if d.get("final_board"):
                    out["final_board"] = np.array(d["final_board"], dtype=np.uint8)
    return out


# --------------------------------------------------------------------------
# 実時間デモ (端末表示)
# --------------------------------------------------------------------------


def flight_ascii(world: "ParachuteWorld", cfg: ParachuteConfig,
                 target: int | None = None) -> str:
    """降下中の盤面を端末に出す。小数座標は最寄りのセルに丸めて表示する。"""
    H, W = world.board.shape
    grid = [["." for _ in range(W)] for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if world.board[r, c]:
                grid[r][c] = "#"
    trail = {(int(round(ty)), int(round(tx))) for tx, ty in world.lander.trail[:-1]}
    for r, c in trail:
        if 0 <= r < H and 0 <= c < W and grid[r][c] == ".":
            grid[r][c] = "·"
    for r, c in world.lander.cells():
        if 0 <= r < H and 0 <= c < W:
            grid[r][c] = "@"
    lines = ["   " + "".join(str(c % 10) for c in range(W))]
    for r in range(H):
        lines.append(f"{r:2d}|" + "".join(grid[r]) + "|")
    foot = [" " for _ in range(W)]
    if target is not None and 0 <= target < W:
        foot[target] = "^"
    lines.append("   " + "".join(foot) + "  ^=目標")
    return "\n".join(lines)


def live_flight(
    pilot,
    cfg: ParachuteConfig | None = None,
    seed: int = 0,
    kind: str | None = None,
    decision_period: int = 4,
    render_cfg: RenderConfig | None = None,
    realtime: bool = False,
    stream=None,
    dump_dir: str | None = None,
) -> "FlightResult":
    """1 機の降下を、判断のたびに端末へ出しながら走らせる。

    「与えた入力・判断内容込みで表示」— 各判断で
    センサ値 / 生の応答 / 解釈結果 / レイテンシ / その間の漂流 を出す。
    """
    import sys

    out = stream or sys.stdout
    cfg = cfg or ParachuteConfig()
    render_cfg = render_cfg or RenderConfig()
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    if hasattr(pilot, "reset"):
        pilot.reset()

    state = {"n": 0}

    def on_decision(world, d: FlightDecision, idx: int) -> None:
        state["n"] += 1
        drift = d.deliver_tick - d.submit_tick
        st = d.state
        print(f"\n{'─'*64}", file=out)
        print(f"判断 #{idx+1}  t={d.submit_tick*cfg.dt:.2f}s (tick {d.submit_tick})", file=out)
        print(flight_ascii(world, cfg, d.target_col), file=out)
        print(f"  [入力] 画像 {render_cfg.image_size(cfg.width, cfg.height)}px"
              f" + センサ: 左端列{st.get('col')} 高さ行{st.get('y')} "
              f"横速度{st.get('vx'):+.2f} 落下{st.get('vy'):.2f} 燃料{st.get('fuel')}", file=out)
        if d.raw is not None:
            print(f"  [生応答] {str(d.raw).strip()[:150]}", file=out)
        print(f"  [解釈] {THRUST_JA[d.action]}"
              + (f" / 目標列 {d.target_col}" if d.target_col is not None else "")
              + (f" / {d.why}" if d.why else ""), file=out)
        tag = f" ⚠ {d.error}" if d.error else ""
        print(f"  [コスト] レイテンシ {d.latency_s:.2f}s → この間に {drift} tick"
              f" ({drift*cfg.dt:.2f}論理秒) 漂流{tag}", file=out)
        if dump_dir and d.png:
            with open(f"{dump_dir}/decision_{idx:03d}.png", "wb") as f:
                f.write(d.png)

    def on_frame(world, frame, action) -> None:
        if realtime:
            time.sleep(cfg.dt)

    res = fly(pilot, cfg, seed=seed, kind=kind, render_cfg=render_cfg,
              decision_period=decision_period, on_decision=on_decision, on_frame=on_frame)
    print(f"\n{'═'*64}", file=out)
    print(flight_ascii(ParachuteWorld(cfg, res.final_board, Lander(res.kind, 0, 0, 0)),
                       cfg, res.oracle_col), file=out)
    print(f"\n結果: {res.score:.0f}点  内訳 {res.breakdown}", file=out)
    print(f"  着地列 {res.landing_col} / 参照解 {res.oracle_col} (誤差 {res.col_error}) "
          f"regret {res.regret}", file=out)
    print(f"  接地速度 下{res.impact_vy:.2f} 横{res.impact_vx:.2f} "
          f"({'軟着陸' if res.soft else '硬着陸'}) / 作った穴 {res.holes_created}", file=out)
    print(f"  燃料 {res.fuel_used}使用/{res.fuel_left}残  滞空 {res.sim_seconds}秒  "
          f"判断 {res.decisions}回 平均 {res.mean_latency_s:.2f}s", file=out)
    if res.errors:
        print(f"  エラー: {res.errors}", file=out)
    return res
