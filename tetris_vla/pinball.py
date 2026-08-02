"""tetris_vla.pinball — 産業制御 (ラダー + シーケンサ) と AI (VLM + VLA) の同居デモ。

## 何が見たいのか

現場の制御は「全部 AI」にはならない。速くて決定的でなければならない部分と、
柔軟でなければならない部分が、**別々のクロックで同居する**。この題材はその形を
そのまま盤面に出す。

    層          周期            役割                        技術
    ─────────────────────────────────────────────────────────────────
    ラダー      毎 tick (50Hz)  ゲートの開閉、インタロック  PLC ラダーロジック
    シーケンサ  毎 tick         事故対応・修理の段取り      逐次ステートマシン
    VLA         25 tick (0.5s)  パドルの連続制御            SmolVLA (行動列)
    VLM         150 tick (3s)   狙う区画・修理の可否判断    qwen2.5vl (言語)

**下の層ほど速く、上の層ほど賢い。** そして上の層は必ず古い情報で喋る。
その食い違いをどう吸収するかが Physical AI のリアルタイム運用そのものになる。

## 盤面

  * **ジグ** (中段): 6 個。**ジグ j にボールが当たると列 j のブロックが 1 個崩れる。**
    工程を 1 回こなす、というイメージ。満遍なくこなすほど高得点 (均等性ボーナス)。
    ただし **叩くたびにジグ自身が摩耗する**。耐久度は外から見えず、打数だけが観測できる。
    耐久 0 で素通りになり、**その工程はもう回せなくなる** (修理するまで)。
    「使う道具が減っていく」という緊張がゲームの中心。
  * **ブロック** (上段): 6 列 x 4 段。ジグ j に対応。直接は触れない。
  * **ゲート**: ラダーロジックが開閉する。開いていればボールが落ちる。
  * **パドル** (下段): 1 自由度 (左右のみ)。VLA が動かす。自由度を削って検証を簡単にしてある。
    当たった位置で反射角が変わるので、**どのジグへ返すかを狙える**。
  * **ボール**: 既定 2 個。カオス要素であり、同時に「並行して回る工程」でもある。
  * **救済機**: シーケンサが故障ジグへ飛ばす修理ドローン。
    ボール・パドル・健全なジグに触れたら修理失敗。

## 隠れ状態

耐久度が見えないので、「そろそろ壊れる」は打数から**推測するしかない**。
これは Physical AI の中心的な難しさ (観測できない物理状態の推定) の最小模型になっている。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np

# --------------------------------------------------------------------------
# 盤面定数
# --------------------------------------------------------------------------

FIELD_W = 24.0
FIELD_H = 32.0
N_SEG = 6                 # ジグの数 = ブロックの列数
BLOCK_ROWS = 4
BLOCK_TOP = 2.0
BLOCK_H = 1.4
PADDLE_Y = 29.0
PADDLE_W = 7.2
BALL_R = 0.45
AIM_GAIN = 7.0            # パドル端で当てたときに横速度へ乗る量 (狙いの強さ)
DOCK = (1.5, 30.5)        # 救済機の待機位置


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    alive: bool = True

    def pos(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class Jig:
    """ピンボールの固定ジグ。**耐久度は外から見えない。**"""

    jid: int
    x: float
    y: float
    r: float
    durability: int          # 隠れ状態。外部からは読めない (残り何回打てるか)
    max_durability: int      # 設定の上限。**全ジグ共通の公開値**であって真値ではない
    hits: int = 0            # 観測できるテレメトリ (打数カウンタ。修理で 0 に戻る)
    broken: bool = False     # 耐久 0。素通りになるので結果として観測できる
    repairing: bool = False
    #: この個体が「今の寿命で」打てる回数の真値。隠れ状態で、エージェントには渡さない。
    #: 答え合わせ (jig_truth) 専用。max_durability を答えとして出すと全ジグ同じ定数に
    #: なってしまい、「壊れて初めて答えが分かる」という主題そのものが消える。
    initial_durability: int = 0
    #: 嵌合クリアランス。**隠れ状態**。打つたびに wear_per_hit ずつ広がる。
    #: 公差 (cfg.fit_tolerance) を超えると嵌合不良になり、挿入はできても工程が進まない。
    clearance: float = 0.0
    wear_per_hit: float = 0.0    # 個体差。隠れ状態
    #: 嵌合不良の回数。**これは観測できる** (検査で分かる) ので、
    #: 素通り (完全故障) より手前で劣化に気づくための唯一の手掛かりになる。
    fit_ng: int = 0

    def telemetry(self) -> dict:
        """外部 (シーケンサ / VLM / VLA) に見せてよい情報だけ。

        クリアランスと摩耗率は隠したまま、**その結果である嵌合不良の回数だけ**を
        見せる。現場で言えば「公差は測っていないが、不良は数えている」状態。
        """
        return {"jid": self.jid, "x": round(self.x, 1), "y": round(self.y, 1),
                "hits": self.hits, "fit_ng": self.fit_ng,
                "broken": self.broken, "repairing": self.repairing}


@dataclass
class Gate:
    """ラダーロジックが開閉する落とし戸。開いていればボールが素通りする。"""

    gid: int
    x: float
    y: float
    w: float
    open: bool = False


@dataclass
class Paddle:
    """1 自由度の稼働部。VLA が動かす。"""

    x: float = FIELD_W / 2      # 中心
    vx: float = 0.0
    w: float = PADDLE_W

    @property
    def left(self) -> float:
        return self.x - self.w / 2

    def segment_of(self, hit_x: float) -> int:
        """当たった位置がどのセグメントか。これがブロックの列に対応する。"""
        t = (hit_x - self.left) / self.w
        return max(0, min(N_SEG - 1, int(t * N_SEG)))

    def segment_center(self, seg: int) -> float:
        """セグメント中心の絶対 x。目標セグメントへ導くときに使う。"""
        return self.left + (seg + 0.5) * self.w / N_SEG


@dataclass
class Drone:
    """救済機。故障ジグまで飛んで修理する。何かに触れたら失敗。"""

    x: float = DOCK[0]
    y: float = DOCK[1]
    active: bool = False
    target: int | None = None
    speed: float = 9.0
    failed_reason: str | None = None


# --------------------------------------------------------------------------
# ラダーロジック — 本物のスキャン意味論を小さく実装する
# --------------------------------------------------------------------------


@dataclass
class Rung:
    """1 本のラング。直列 (AND) の中に並列 (OR) の枝を持つ、という PLC の基本形。

        branches: [[接点, 接点], [接点]]  ->  (A and B) or (C)
    出力コイルは 1 つ。ラッチ (自己保持) は `seal` で表現する。
    """

    name: str
    branches: list[list[str]]      # 接点名。先頭 "!" で b 接点 (否定)
    coil: str
    seal: bool = False             # True で自己保持 (コイル自身を OR に足す)
    comment: str = ""

    def evaluate(self, bits: dict[str, bool]) -> bool:
        def contact(n: str) -> bool:
            return (not bits.get(n[1:], False)) if n.startswith("!") else bits.get(n, False)

        result = any(all(contact(c) for c in br) for br in self.branches) if self.branches else False
        if self.seal:
            result = result or bits.get(self.coil, False)
        return result


class LadderPLC:
    """毎スキャン「入力を読む → 上のラングから順に評価 → 出力を書く」を回すだけ。

    実機の PLC と同じく **評価順が意味を持つ**。上のラングが書いたコイルを
    下のラングが接点として読める。これが「決定的で読み切れる」ことの根拠になる。
    """

    def __init__(self, rungs: Sequence[Rung]) -> None:
        self.rungs = list(rungs)
        self.bits: dict[str, bool] = {}
        self.scan_count = 0
        self.trace: list[dict] = []

    def scan(self, inputs: dict[str, bool], keep_trace: bool = False) -> dict[str, bool]:
        self.bits.update(inputs)
        fired = []
        for r in self.rungs:
            v = r.evaluate(self.bits)
            if v != self.bits.get(r.coil, False):
                fired.append(r.name)
            self.bits[r.coil] = v
        self.scan_count += 1
        if keep_trace:
            # ドラレコ用。入力接点も一緒に残さないと「なぜコイルが立ったか」を
            # あとから追えない
            self.trace.append({"scan": self.scan_count, "changed": fired,
                               "coils": dict(self.bits)})
        return dict(self.bits)


def build_ladder() -> list[Rung]:
    """このデモのラダー。**ボールを落とす** ためのゲート制御と安全インタロック。

    実務のラダーらしく、意味が 1 行で読めるように書いてある。
    """
    return [
        # ボールが上部にいる間はゲートを閉じておく (落として即死させない)
        Rung("R1 上部滞留でゲート閉", [["ball_upper"]], "hold_gates",
             comment="ボールが上段にある間はゲートを開けない"),
        # 下段に来て、かつ滞留しすぎ (タイマ) ならゲートを開けて落とす
        Rung("R2 滞留タイマでゲート開", [["ball_lower", "stall_timer", "!hold_gates"]],
             "gate0_open", seal=False,
             comment="下段で停滞したらゲート 0 を開けてボールを落とす"),
        Rung("R3 交互ゲート", [["gate0_open", "alt_flag"]], "gate1_open",
             comment="ゲート 0 と交互に動かして偏りを防ぐ"),
        # 安全インタロック: 救済機が飛んでいる間はゲートを動かさない
        Rung("R4 救済機作動中はゲート禁止", [["drone_active"]], "gate_inhibit",
             comment="ドローン飛行中はゲート操作を禁止 (巻き込み防止)"),
        # 修理許可: 故障があり、かつボールが危険域にいないこと (自己保持)
        Rung("R5 修理許可ラッチ", [["jig_broken", "!ball_in_danger", "!drone_active"]],
             "repair_permit", seal=True,
             comment="故障検知 かつ ボールが危険域外 なら修理許可を保持"),
        # 危険域にボールが入ったら許可を落とす (下位ラングで打ち消す)
        Rung("R6 危険域で許可解除", [["ball_in_danger"], ["drone_abort"]], "repair_cancel",
             comment="危険域侵入または中断要求で許可を取り消す"),
    ]


# --------------------------------------------------------------------------
# 逐次シーケンサ — 事故対応の段取り
# --------------------------------------------------------------------------


class SeqState(Enum):
    IDLE = "待機"
    RUN = "通常運転"
    FAULT = "故障検知"
    PLAN = "修理計画"
    DISPATCH = "救済機発進"
    REPAIR = "修理中"
    RECOVER = "復帰"
    ABORT = "中断"


@dataclass
class SeqEvent:
    tick: int
    frm: str
    to: str
    reason: str


class Sequencer:
    """産業用の逐次 SM。**事故対応はここが持つ**。AI には任せない。

    AI (VLM) は「今修理してよいか」の *助言* を出すだけで、
    実際にドローンを出す/止めるの判断は必ずこのステートマシンを通る。
    これが「AI 連動でも安全側の権限は組み込みに残す」という形。
    """

    def __init__(self) -> None:
        self.state = SeqState.IDLE
        self.events: list[SeqEvent] = []
        self.repairs_ok = 0
        self.repairs_failed = 0
        self.vlm_advice: dict = {}

    def _to(self, tick: int, s: SeqState, reason: str) -> None:
        if s is self.state:
            return
        self.events.append(SeqEvent(tick, self.state.value, s.value, reason))
        self.state = s

    def step(self, tick: int, world: "PinballWorld", bits: dict[str, bool]) -> None:
        broken = [j for j in world.jigs if j.broken and not j.repairing]
        d = world.drone

        if self.state is SeqState.IDLE:
            self._to(tick, SeqState.RUN, "起動")

        elif self.state is SeqState.RUN:
            if broken:
                self._to(tick, SeqState.FAULT, f"ジグ#{broken[0].jid} 素通りを検知")

        elif self.state is SeqState.FAULT:
            # 修理許可はラダー側のインタロックが出す。SM は待つだけ
            if bits.get("repair_permit") and not bits.get("repair_cancel"):
                self._to(tick, SeqState.PLAN, "ラダーが修理許可を出した")
            elif not broken:
                self._to(tick, SeqState.RUN, "故障が解消した")

        elif self.state is SeqState.PLAN:
            # VLM の助言があれば尊重するが、拒否権はこちらが持つ
            advice = self.vlm_advice.get("repair_ok")
            if advice is False:
                self._to(tick, SeqState.RUN, "VLM が今は危険と助言 → 見送り")
            elif broken:
                d.target = broken[0].jid
                self._to(tick, SeqState.DISPATCH, f"目標ジグ#{d.target} を決定")
            else:
                self._to(tick, SeqState.RUN, "対象なし")

        elif self.state is SeqState.DISPATCH:
            d.active = True
            d.failed_reason = None
            for j in world.jigs:
                if j.jid == d.target:
                    j.repairing = True
            self._to(tick, SeqState.REPAIR, "救済機発進")

        elif self.state is SeqState.REPAIR:
            if d.failed_reason:
                self.repairs_failed += 1
                self._to(tick, SeqState.ABORT, f"修理失敗: {d.failed_reason}")
            elif not d.active:
                self.repairs_ok += 1
                self._to(tick, SeqState.RECOVER, "修理完了")

        elif self.state in (SeqState.RECOVER, SeqState.ABORT):
            d.active = False
            d.target = None
            d.x, d.y = DOCK
            for j in world.jigs:
                j.repairing = False
            self._to(tick, SeqState.RUN, "定常へ復帰")


# --------------------------------------------------------------------------
# ワールド
# --------------------------------------------------------------------------


@dataclass
class PinballConfig:
    dt: float = 0.02              # 50 Hz
    gravity: float = 4.6
    ball_speed0: float = 10.0
    paddle_speed: float = 19.0    # パドルの最大速度 (1 自由度)
    n_balls: int = 2              # カオス要素であり、並行して回る工程でもある
    n_lives: int = 4              # 落としても復帰する回数。面を長く保って修理劇を見せる
    jig_durability: tuple[int, int] = (4, 9)   # 隠れ耐久度の範囲
    #: 嵌合公差。クリアランスがこれを超えると嵌合不良 (挿入はできるが工程が進まない)。
    #: クリアランスは寿命いっぱいで 1.0 に達するよう正規化してあるので、
    #: 0.65 なら「寿命の終盤 35% は不良を出しながら回る」という意味になる。
    fit_tolerance: float = 0.8
    #: 滞留タイマ。**「この tick 数だけ工程が進まなければ滞留」** と読む。
    #:
    #: 以前は「全球が y>=14 に居続けた tick 数」を数えて 220 (4.4秒) と比べていたが、
    #: 全球が下段に居続ける最長は実測 108 スキャン (2.16秒) しかなく、しきい値が
    #: 達成可能な最大値の 2 倍だった。結果 R2/R3 (ゲート開閉) は全走行を通じて
    #: 一度も通電せず、6 本のラングのうち 2 本が死んでいた。
    #:
    #: しきい値ではなく述語の方が誤りだった。位置ではなく進捗 (ブロックが崩れたか)
    #: で計るよう直したところ、ほぼ同じ 200 (4.0秒) で素直に動くようになった
    #: — 3 seed 平均で 712 点 / ゲート開閉 8.3 回 (直す前は 0 回)。
    stall_ticks: int = 200
    danger_y: float = 22.0        # これより下にボールがあると危険域
    max_ticks: int = 6000
    seed: int = 0


class PinballWorld:
    def __init__(self, cfg: PinballConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.tick = 0
        self.t = 0.0
        self.paddle = Paddle()
        self.drone = Drone()
        self.blocks = np.full((BLOCK_ROWS, N_SEG), True)
        self.broken_per_col = [0] * N_SEG
        self.balls: list[Ball] = []
        for i in range(cfg.n_balls):
            ang = self.rng.uniform(-0.5, 0.5)
            self.balls.append(Ball(
                x=FIELD_W / 2 + self.rng.uniform(-4, 4), y=12.0,
                vx=cfg.ball_speed0 * math.sin(ang), vy=abs(cfg.ball_speed0 * math.cos(ang))))
        self.jigs = self._make_jigs()
        self.gates = [Gate(0, 7.0, 21.0, 4.0), Gate(1, 17.0, 21.0, 4.0)]
        self.stall = 0
        #: 最後に工程が進んだ (ブロックが崩れた) tick。滞留判定の基準
        self.last_progress_tick = 0
        self.alt = False
        self.lives = cfg.n_lives
        self.lost_balls = 0
        self.paddle_hits = 0
        self.events: list[dict] = []

    #: ジグの配置。**ジグ j が列 j に対応する** ので、数は N_SEG と一致させる。
    #: ジグの配置。**パドル中心から等距離の扇形**に並べる。
    #:
    #: 以前は 2 段配置 ((4,13) (9,10.5) (14,10.5) (20,13) / (7,17.5) (17,17.5)) で、
    #: 下段の 2 個が上段の 4 個を遮蔽していた。パドル中心から射出角を振って
    #: 「最初に当たるジグ」を測ると、**どの角度でも 4 か 5 にしか当たらない**:
    #:
    #:     vx  -8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7   8
    #:     jig  4   4   4   1   4   4   4   4   .   5   5   5   5   1   5   5   5
    #:
    #: つまり 6 工程のうち 4 つが狙って到達できず、狙ったジグに当たる確率は
    #: 15.3% (偶然 16.7%) だった。腕前がスコアに出ないのも、教師を模倣した VLA が
    #: 符号一致率 37% (偶然 33%) にしかならないのも、制御則ではなく**配置**が原因。
    #: 到達できない目標を狙う教師ラベルからは、学習しようがない。
    #:
    #: 球が到達できる高さは初速で決まっていて、実測すると頂点は常に y=17.2。
    #: 横速度は上昇に影響しないので、**到達可能な線は y=17.2 の 1 本**しかない。
    #: その線上で頂点 x は vx に線形対応する (vx -4〜+4 で x 3.4〜20.6)。
    #: それを超えると壁で折り返して対応が崩れるので、単調な区間だけを使う。
    #:
    #: そこで 6 基を y=18.0 の 1 列に、その区間へ等間隔で置く。
    #: 隣接間隔 3.44 (直径 2.2 なので重ならず、隙間 1.24 は球径 0.9 より広いので抜けられる)。
    JIG_SPOTS = [(3.4, 18.0), (6.8, 18.0), (10.3, 18.0),
                 (13.7, 18.0), (17.2, 18.0), (20.6, 18.0)]

    def _make_jigs(self) -> list[Jig]:
        lo, hi = self.cfg.jig_durability
        jigs = []
        for i, (x, y) in enumerate(self.JIG_SPOTS[:N_SEG]):
            d = self.rng.randint(lo, hi)      # 個体ごとの真の寿命。これが隠れ状態
            # 摩耗率も個体差。寿命 d 回でクリアランスが 1.0 に達する速さを基準に、
            # ±20% ばらつかせる。同じ打数でも先に不良を出す個体が現れる
            wear = (1.0 / d) * self.rng.uniform(0.8, 1.2)
            jigs.append(Jig(jid=i, x=x, y=y, r=1.1, durability=d,
                            max_durability=hi, initial_durability=d,
                            wear_per_hit=wear))
        return jigs

    # -- 観測 (外部に見せてよいもの) --------------------------------------

    def telemetry(self) -> dict:
        b = self.balls[0] if self.balls else None
        return {
            "tick": self.tick, "t": round(self.t, 2),
            "paddle_x": round(self.paddle.x, 2),
            "balls": [{"x": round(x.x, 2), "y": round(x.y, 2),
                       "vx": round(x.vx, 2), "vy": round(x.vy, 2)}
                      for x in self.balls if x.alive],
            "jigs": [j.telemetry() for j in self.jigs],
            "gates": [{"gid": g.gid, "open": g.open} for g in self.gates],
            "blocks_left_per_col": [int(self.blocks[:, c].sum()) for c in range(N_SEG)],
            "broken_per_col": list(self.broken_per_col),
            "drone": {"active": self.drone.active, "target": self.drone.target,
                      "x": round(self.drone.x, 2), "y": round(self.drone.y, 2)},
        }

    def ladder_inputs(self) -> dict[str, bool]:
        alive = [b for b in self.balls if b.alive]
        upper = any(b.y < 14.0 for b in alive)
        lower = any(b.y >= 14.0 for b in alive)
        return {
            "ball_upper": upper,
            "ball_lower": lower,
            "stall_timer": self.stall > 0,
            "alt_flag": self.alt,
            "drone_active": self.drone.active,
            "jig_broken": any(j.broken and not j.repairing for j in self.jigs),
            "ball_in_danger": any(b.y > self.cfg.danger_y for b in alive),
            "drone_abort": self.drone.failed_reason is not None,
        }

    # -- 物理 -------------------------------------------------------------

    def _bounce_walls(self, b: Ball) -> None:
        if b.x < BALL_R:
            b.x, b.vx = BALL_R, abs(b.vx)
        elif b.x > FIELD_W - BALL_R:
            b.x, b.vx = FIELD_W - BALL_R, -abs(b.vx)
        if b.y < BALL_R:
            b.y, b.vy = BALL_R, abs(b.vy)

    def _bounce_jigs(self, b: Ball) -> None:
        for j in self.jigs:
            if j.broken:
                continue     # 耐久 0 は素通り。ここで初めて故障が「見える」
            dx, dy = b.x - j.x, b.y - j.y
            d = math.hypot(dx, dy)
            if d < j.r + BALL_R and d > 1e-6:
                nx, ny = dx / d, dy / d
                dot = b.vx * nx + b.vy * ny
                if dot < 0:
                    b.vx -= 2 * dot * nx
                    b.vy -= 2 * dot * ny
                    b.x, b.y = j.x + nx * (j.r + BALL_R + 1e-3), j.y + ny * (j.r + BALL_R + 1e-3)
                    j.hits += 1
                    j.durability -= 1
                    j.clearance += j.wear_per_hit      # 打つたびに嵌合が緩む
                    if j.clearance > self.cfg.fit_tolerance:
                        # **嵌合不良**。挿入はしたが公差を外れているので工程は進まない。
                        # 素通り (完全故障) の手前で、外から観測できる劣化の兆候になる。
                        j.fit_ng += 1
                        self.events.append({"tick": self.tick, "kind": "fit_ng",
                                            "jid": j.jid, "hits": j.hits})
                    else:
                        # **工程を 1 回こなす**: ジグ j に当たったら列 j が 1 個崩れる
                        self._break_block(j.jid)
                    if j.durability <= 0 and not j.broken:
                        j.broken = True
                        self.events.append({"tick": self.tick, "kind": "jig_broken",
                                            "jid": j.jid, "hits": j.hits})

    def _paddle_and_blocks(self, b: Ball) -> None:
        p = self.paddle
        if b.vy > 0 and PADDLE_Y - BALL_R <= b.y <= PADDLE_Y + 0.8:
            if p.left <= b.x <= p.left + p.w:
                b.y = PADDLE_Y - BALL_R
                b.vy = -abs(b.vy)
                # 当たった位置で反射角が変わる = **どのジグへ返すかを狙える**。
                # パドル自体はブロックを崩さない (崩すのはジグ)
                rel = (b.x - p.x) / (p.w / 2)
                b.vx += rel * AIM_GAIN
                self.paddle_hits += 1

    def _break_block(self, seg: int) -> None:
        """ジグ seg を 1 回叩いた = 工程を 1 回こなした → 列 seg が 1 個崩れる。"""
        col = self.blocks[:, seg]
        idx = np.nonzero(col)[0]
        if idx.size:
            self.blocks[idx[-1], seg] = False
            self.broken_per_col[seg] += 1
            self.last_progress_tick = self.tick
            self.events.append({"tick": self.tick, "kind": "block", "seg": seg,
                                "left": int(self.blocks[:, seg].sum())})

    def _gates(self, b: Ball) -> None:
        for g in self.gates:
            if g.open:
                continue
            if abs(b.y - g.y) < 0.5 and abs(b.x - g.x) < g.w / 2 and b.vy > 0:
                b.y, b.vy = g.y - 0.5, -abs(b.vy) * 0.9   # 閉ゲートは弾く

    def _drone(self) -> None:
        d = self.drone
        if not d.active:
            return
        tgt = next((j for j in self.jigs if j.jid == d.target), None)
        if tgt is None:
            d.active = False
            return
        dx, dy = tgt.x - d.x, tgt.y - d.y
        dist = math.hypot(dx, dy)
        step = d.speed * self.cfg.dt
        if dist <= step:
            d.x, d.y = tgt.x, tgt.y
            # 整備は「元の寿命に戻す」。設定上限まで回復させると、耐久 6 の個体が
            # 修理後 9 になり、直すたびに新品より強くなってしまう
            tgt.durability = tgt.initial_durability
            tgt.clearance = 0.0        # 整備で嵌合も出荷時に戻る
            tgt.fit_ng = 0
            tgt.broken = False
            tgt.hits = 0
            d.active = False
            self.events.append({"tick": self.tick, "kind": "repair_ok", "jid": tgt.jid})
            return
        d.x += dx / dist * step
        d.y += dy / dist * step
        # 衝突判定: ボール / パドル / 健全なジグ に触れたら失敗
        for b in self.balls:
            if b.alive and math.hypot(b.x - d.x, b.y - d.y) < BALL_R + 0.5:
                d.failed_reason = "ボールに接触"
        p = self.paddle
        if abs(d.y - PADDLE_Y) < 0.7 and p.left - 0.4 <= d.x <= p.left + p.w + 0.4:
            d.failed_reason = "稼働部に接触"
        for j in self.jigs:
            if j.jid == d.target or j.broken:
                continue
            if math.hypot(j.x - d.x, j.y - d.y) < j.r + 0.5:
                d.failed_reason = f"ジグ#{j.jid} に接触"
        if d.failed_reason:
            d.active = False
            self.events.append({"tick": self.tick, "kind": "repair_fail",
                                "jid": d.target, "reason": d.failed_reason})

    # -- 1 tick -----------------------------------------------------------

    def step(self, paddle_cmd: float, bits: dict[str, bool]) -> None:
        """paddle_cmd は -1..+1 の速度指令 (1 自由度)。"""
        c = self.cfg
        for g in self.gates:
            g.open = bits.get(f"gate{g.gid}_open", False) and not bits.get("gate_inhibit", False)

        p = self.paddle
        p.vx = max(-1.0, min(1.0, paddle_cmd)) * c.paddle_speed
        p.x = max(p.w / 2, min(FIELD_W - p.w / 2, p.x + p.vx * c.dt))

        for b in self.balls:
            if not b.alive:
                continue
            b.vy += c.gravity * c.dt
            sp = math.hypot(b.vx, b.vy)
            if sp > 22.0:
                b.vx, b.vy = b.vx / sp * 22.0, b.vy / sp * 22.0
            b.x += b.vx * c.dt
            b.y += b.vy * c.dt
            self._bounce_walls(b)
            self._gates(b)
            self._bounce_jigs(b)
            self._paddle_and_blocks(b)
            if b.y > FIELD_H:
                self.lost_balls += 1
                self.lives -= 1
                self.events.append({"tick": self.tick, "kind": "ball_lost",
                                    "lives_left": self.lives})
                if self.lives > 0:
                    ang = self.rng.uniform(-0.5, 0.5)
                    b.x, b.y = FIELD_W / 2 + self.rng.uniform(-4, 4), 12.0
                    b.vx = c.ball_speed0 * math.sin(ang)
                    b.vy = abs(c.ball_speed0 * math.cos(ang))
                else:
                    b.alive = False

        self._drone()
        alive = [b for b in self.balls if b.alive]
        # 滞留タイマ。**位置ではなく「工程が進んでいないこと」で計る**。
        # 以前は「全球が y>=14」という位置だけを見ていたので、下段で激しく跳ねて
        # ジグを叩き続けていても滞留扱いになり、ゲートが工程中の球を落としていた。
        # 産業ラインの滞留検知と同じく、見るべきは進捗が止まっていることの方。
        stalled = bool(alive) and (self.tick - self.last_progress_tick) >= self.cfg.stall_ticks
        self.stall = self.stall + 1 if stalled else 0
        if self.stall == 1:
            self.alt = not self.alt
        self.tick += 1
        self.t += c.dt

    @property
    def done(self) -> bool:
        return (not any(b.alive for b in self.balls)
                or not self.blocks.any()
                or self.tick >= self.cfg.max_ticks)

    # -- 採点 -------------------------------------------------------------

    def score(self) -> dict:
        broken = int(BLOCK_ROWS * N_SEG - self.blocks.sum())
        per = self.broken_per_col
        evenness = (min(per) / max(per)) if max(per) > 0 else 0.0
        jig_hits = [j.hits for j in self.jigs]
        jig_even = (min(jig_hits) / max(jig_hits)) if max(jig_hits) > 0 else 0.0
        return {
            "lives_left": self.lives,
            "blocks_broken": broken,
            "blocks_total": BLOCK_ROWS * N_SEG,
            "per_col": list(per),
            "evenness": round(evenness, 3),
            "jig_hits": jig_hits,
            "jig_evenness": round(jig_even, 3),
            "broken_jigs": sum(1 for j in self.jigs if j.broken),
            "paddle_hits": self.paddle_hits,
            "lost_balls": self.lost_balls,
            "score": round(1000 * (0.55 * broken / (BLOCK_ROWS * N_SEG)
                                   + 0.30 * evenness + 0.15 * jig_even), 1),
        }
