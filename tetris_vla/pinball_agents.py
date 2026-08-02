"""tetris_vla.pinball_agents — 4 つの制御層を結線して 1 面走らせる。

    ラダー (毎tick) → シーケンサ (毎tick) → VLA (25tick) → VLM (150tick)

**下ほど速く、上ほど賢い。** 上位の判断は必ず古い盤面に対する答えになるので、
下位はそれを前提に動く。ここが Physical AI のリアルタイム運用の核心。

各層の呼び出しは `LayerCall` として全部記録され、HTML でクロックの食い違いが見える。
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from .pinball import (
    BALL_R,
    BLOCK_H,
    BLOCK_ROWS,
    BLOCK_TOP,
    DOCK,
    FIELD_H,
    FIELD_W,
    N_SEG,
    PADDLE_Y,
    LadderPLC,
    PinballConfig,
    PinballWorld,
    SeqState,
    Sequencer,
    build_ladder,
)

# --------------------------------------------------------------------------
# 描画
# --------------------------------------------------------------------------

BG = (16, 16, 22)
WALL = (54, 54, 66)
BLOCK_COLS = [(255, 105, 97), (255, 179, 71), (253, 253, 150), (119, 221, 119),
              (119, 158, 203), (200, 146, 232)]
JIG_OK = (120, 200, 255)
JIG_HURT = (255, 190, 90)
JIG_BROKEN = (90, 90, 100)
BALL_C = (255, 255, 255)
PADDLE_C = (100, 240, 180)
SEG_LINE = (30, 90, 70)
GATE_OPEN = (60, 200, 120)
GATE_SHUT = (200, 80, 80)
DRONE_C = (255, 120, 220)
TARGET_C = (255, 240, 120)


def render_pinball(w: PinballWorld, px_per_unit: int = 9, target_seg: int | None = None,
                   show_hidden: bool = False) -> Image.Image:
    """盤面 1 枚。**耐久度は既定で描かない** (隠れ状態なので)。

    show_hidden=True は人間が答え合わせをするときだけ使う。
    モデルに渡す画像では必ず False にすること。
    """
    W, H = int(FIELD_W * px_per_unit), int(FIELD_H * px_per_unit)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    u = px_per_unit

    def P(x: float, y: float) -> tuple[float, float]:
        return x * u, y * u

    d.rectangle([0, 0, W - 1, H - 1], outline=WALL, width=2)

    # ブロック (6 列 x 4 段)
    cw = FIELD_W / N_SEG
    for r in range(BLOCK_ROWS):
        for c in range(N_SEG):
            if not w.blocks[r, c]:
                continue
            x0, y0 = P(c * cw + 0.35, BLOCK_TOP + r * BLOCK_H)
            x1, y1 = P((c + 1) * cw - 0.35, BLOCK_TOP + (r + 1) * BLOCK_H - 0.3)
            d.rectangle([x0, y0, x1, y1], fill=BLOCK_COLS[c % len(BLOCK_COLS)])

    # ゲート
    for g in w.gates:
        x0, y0 = P(g.x - g.w / 2, g.y - 0.18)
        x1, y1 = P(g.x + g.w / 2, g.y + 0.18)
        d.rectangle([x0, y0, x1, y1], fill=GATE_OPEN if g.open else GATE_SHUT)

    # ジグ
    for j in w.jigs:
        x0, y0 = P(j.x - j.r, j.y - j.r)
        x1, y1 = P(j.x + j.r, j.y + j.r)
        if j.broken:
            d.ellipse([x0, y0, x1, y1], outline=JIG_BROKEN, width=2)   # 素通り = 輪郭だけ
        else:
            # 打数は観測できるので色に出す。耐久度そのものは出さない
            frac = min(1.0, j.hits / max(1, j.max_durability))
            col = tuple(int(JIG_OK[i] + (JIG_HURT[i] - JIG_OK[i]) * frac) for i in range(3))
            d.ellipse([x0, y0, x1, y1], fill=col)
        if show_hidden:
            d.text(P(j.x - 0.3, j.y - 0.4), f"{j.durability}", fill=(0, 0, 0))
        if j.repairing:
            d.ellipse([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=TARGET_C, width=2)

    # パドル (セグメント境界を描く = どこに当てるかが意味を持つ)
    p = w.paddle
    x0, y0 = P(p.left, PADDLE_Y - 0.3)
    x1, y1 = P(p.left + p.w, PADDLE_Y + 0.3)
    d.rectangle([x0, y0, x1, y1], fill=PADDLE_C)
    for s in range(1, N_SEG):
        sx, _ = P(p.left + s * p.w / N_SEG, 0)
        d.line([sx, y0, sx, y1], fill=SEG_LINE, width=1)
    if target_seg is not None and 0 <= target_seg < len(w.jigs):
        # 狙うのはパドルの区画ではなく **ジグ** なので、印はジグの上に出す
        tj = w.jigs[target_seg]
        tx, ty = P(tj.x, tj.y - tj.r - 0.5)
        d.polygon([(tx - 5, ty - 7), (tx + 5, ty - 7), (tx, ty)], fill=TARGET_C)

    # ボール
    for b in w.balls:
        if not b.alive:
            continue
        x0, y0 = P(b.x - BALL_R, b.y - BALL_R)
        x1, y1 = P(b.x + BALL_R, b.y + BALL_R)
        d.ellipse([x0, y0, x1, y1], fill=BALL_C)

    # 救済機
    if w.drone.active:
        x0, y0 = P(w.drone.x - 0.5, w.drone.y - 0.5)
        x1, y1 = P(w.drone.x + 0.5, w.drone.y + 0.5)
        d.rectangle([x0, y0, x1, y1], fill=DRONE_C)
    return img


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# 記録
# --------------------------------------------------------------------------


@dataclass
class LayerCall:
    seq: int
    layer: str        # ladder / sequencer / vla / vlm
    tick: int
    t: float
    latency_s: float
    inputs: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    raw: str | None = None
    prompt: str | None = None
    error: str | None = None
    png_b64: str | None = None
    note: str = ""


# --------------------------------------------------------------------------
# パドル制御 (VLA が担当する層)
# --------------------------------------------------------------------------


def predict_landing_x(w: PinballWorld) -> float | None:
    """ボールがパドル高さに来るときの x を、壁反射込みで素朴に外挿する。

    これは **教師データ生成と mock のため** のもので、実運用の VLA には渡さない。
    """
    alive = [b for b in w.balls if b.alive]
    if not alive:
        return None
    b = min(alive, key=lambda z: abs(PADDLE_Y - z.y) if z.vy > 0 else 1e9)
    if b.vy <= 0:
        return b.x
    x, y, vx, vy = b.x, b.y, b.vx, b.vy
    dt = w.cfg.dt
    for _ in range(400):
        vy += w.cfg.gravity * dt
        x += vx * dt
        y += vy * dt
        if x < BALL_R:
            x, vx = BALL_R, abs(vx)
        elif x > FIELD_W - BALL_R:
            x, vx = FIELD_W - BALL_R, -abs(vx)
        if y >= PADDLE_Y - BALL_R:
            return x
    return x


def aim_offset(w: PinballWorld, land_x: float, vy_at_hit: float,
               target_jig: int | None) -> float:
    """目標ジグへ返すには、パドルのどこに当てればよいか (-1..+1 の相対位置)。

    反射後 vx += rel * AIM_GAIN なので、必要な vx から rel を逆算する。
    重力込みで「上昇してジグ高さに達するまでの時間」を解いてから横速度を決める。
    壁反射は無視する近似 (それでも十分狙える)。
    """
    if target_jig is None or not (0 <= target_jig < len(w.jigs)):
        return 0.0
    j = w.jigs[target_jig]
    g = w.cfg.gravity
    v0 = abs(vy_at_hit)                    # 反射直後の上向き速度
    rise = PADDLE_Y - j.y
    if rise <= 0:
        return 0.0
    disc = v0 * v0 - 2 * g * rise
    if disc < 0:                            # そもそも届かない高さ
        t = v0 / max(1e-6, g)
    else:
        t = (v0 - math.sqrt(disc)) / max(1e-6, g)
    t = max(0.08, t)
    vx_need = (j.x - land_x) / t
    cur_vx = next((b.vx for b in w.balls if b.alive), 0.0)
    from tetris_vla.pinball import AIM_GAIN

    return max(-1.0, min(1.0, (vx_need - cur_vx) / AIM_GAIN))


class ScriptedPaddle:
    """教師。**返球しつつ、目標ジグへ向くようにパドルの当て所を作る**。

    ジグを叩くと工程が 1 回進む (対応する列が崩れる) ので、
    「どのジグへ返すか」がそのまま生産計画の実行になっている。
    """

    name = "scripted"
    layer = "vla"

    def __init__(self, kp: float = 0.75, bang_bang: bool = True,
                 deadband: float = 0.25) -> None:
        self.kp = kp
        #: 指令を {-1, 0, +1} に量子化する。実測で連続版は 55% が飽和しており、
        #: 実質バンバン制御だった。中間値を残すと BC が平均へ回帰して振幅が半減し
        #: (実測 0.636 -> 0.312)、パドルが届かなくなる。分布を素直にしておく。
        self.bang_bang = bang_bang
        self.deadband = deadband

    def __call__(self, w: PinballWorld, target_seg: int | None) -> tuple[float, dict]:
        land = predict_landing_x(w)
        if land is None:
            return 0.0, {"note": "ボールなし"}
        alive = [b for b in w.balls if b.alive and b.vy > 0]
        vy = max((b.vy for b in alive), default=8.0)
        rel = aim_offset(w, land, vy, target_seg)
        # 着弾点がパドル上の rel の位置に来るようなパドル中心
        want_center = land - rel * (w.paddle.w / 2)
        # パドルから外れるくらいなら狙いを捨てて返球を優先する
        want_center = max(land - w.paddle.w / 2 * 0.92,
                          min(land + w.paddle.w / 2 * 0.92, want_center))
        err = want_center - w.paddle.x
        cmd = max(-1.0, min(1.0, self.kp * err))
        if self.bang_bang:
            cmd = 0.0 if abs(err) < self.deadband else (1.0 if err > 0 else -1.0)
        return cmd, {
            "landing_x": round(land, 2), "aim_rel": round(rel, 2),
            "want_center": round(want_center, 2), "err": round(err, 2),
            "target_jig": target_seg, "bang_bang": self.bang_bang}


class MockVLAPaddle:
    """モデル無しで動かすための VLA スタブ。行動列とレイテンシを模す。"""

    name = "mock-vla"
    layer = "vla"

    def __init__(self, chunk: int = 25, skill: float = 0.85, latency_s: float = 0.35,
                 seed: int = 0) -> None:
        self.chunk, self.skill, self.latency_s = chunk, skill, latency_s
        self.rng = random.Random(seed)
        self._teacher = ScriptedPaddle()

    def plan(self, w: PinballWorld, target_seg: int | None) -> tuple[list[float], float, dict]:
        lat = max(0.02, self.rng.lognormvariate(math.log(self.latency_s), 0.25))
        cmd, info = self._teacher(w, target_seg)
        seq = []
        for i in range(self.chunk):
            v = cmd if self.rng.random() < self.skill else self.rng.uniform(-1, 1)
            seq.append(v * (1.0 - 0.02 * i))     # 先に行くほど自信を落とす
        return seq, lat, {**info, "chunk": self.chunk}


class SmolVLAPaddle:
    """本物の VLA。1 回の推論で chunk 手ぶんのパドル指令を返す。

    state 6 = [パドル位置, ボール x, ボール y, ボール vx, ボール vy, 目標セグメント誤差]
    action  = 先頭 1 次元だけ使う (1 自由度)。自由度を削ってあるので検証が簡単。
    """

    name = "smolvla"
    layer = "vla"

    def __init__(self, checkpoint: str | None = None, device: str = "auto",
                 chunk: int = 25) -> None:
        import torch

        from .smolvla_pilot import build_policy, pick_device

        self.policy, self.pre, self.post, self.cfg, dev = build_policy(
            device, chunk_size=chunk)
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu")
            self.policy.load_state_dict(sd["model"])
            self.policy.to(torch.device(pick_device(dev)))
        self.policy.eval()
        self.chunk = chunk
        self._torch = torch
        self.name = "smolvla" + ("-bc" if checkpoint else "-zeroshot")

    @staticmethod
    def encode_state(w: PinballWorld, target_seg: int | None) -> np.ndarray:
        """6 次元。最後が **目標ジグの位置** (上位からの指令)。

        「そこへ返すにはパドルのどこに当てればよいか」の弾道計算は渡さない。
        そこを学習させたいので、教師の中間結果は入力に入れない。
        降りてくるボールは **最初に着弾するもの** を基準にする (2 個あるため)。
        """
        alive = [b for b in w.balls if b.alive]
        # 先に降りてくる (= 先に捌かねばならない) ボールを主対象にする
        b = min(alive, key=lambda z: (PADDLE_Y - z.y) / max(0.1, z.vy) if z.vy > 0 else 1e9,
                default=None) if alive else None
        tgt_x = w.jigs[target_seg].x if target_seg is not None and 0 <= target_seg < len(w.jigs) \
            else w.paddle.x
        return np.array([
            w.paddle.x / FIELD_W * 2 - 1,
            (b.x / FIELD_W * 2 - 1) if b else 0.0,
            (b.y / FIELD_H * 2 - 1) if b else 1.0,
            (b.vx / 20.0) if b else 0.0,
            (b.vy / 20.0) if b else 0.0,
            tgt_x / FIELD_W * 2 - 1,
        ], dtype=np.float32)

    def plan(self, w: PinballWorld, target_seg: int | None) -> tuple[list[float], float, dict]:
        from .smolvla_pilot import image_to_tensor

        torch = self._torch
        img = render_pinball(w, target_seg=target_seg)
        task = pinball_task(target_seg)   # 学習時と同じ文にすること
        batch = {"observation.state": torch.from_numpy(self.encode_state(w, target_seg)),
                 "observation.images.camera1": image_to_tensor(img),
                 "task": task}
        t0 = time.perf_counter()
        with torch.inference_mode():
            chunk = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        lat = time.perf_counter() - t0
        arr = chunk.detach().float().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        seq = [float(max(-1.0, min(1.0, v[0]))) for v in arr[: self.chunk]]
        return seq, lat, {"task": task, "chunk": len(seq),
                          "first6": [round(v, 3) for v in seq[:6]]}


# --------------------------------------------------------------------------
# VLM 戦略層
# --------------------------------------------------------------------------

STRATEGY_PROMPT = """You are the strategy layer of an industrial control stack for the game shown.

Rules:
- The paddle at the bottom has {nseg} segments, numbered 0 (left) to {maxseg} (right).
- When the ball hits paddle segment k, one block is removed from column k.
- Goal: remove blocks EVENLY across all columns, and remove as many as possible.
- Round bumpers in the middle wear out after an unknown number of hits.
  A bumper drawn as an empty outline is already broken (the ball passes through).
  A repair drone can fix it, but the repair FAILS if the drone touches the ball,
  the paddle, or a healthy bumper.

Telemetry:
- blocks remaining per column (0..{maxseg}): {blocks}
- blocks already removed per column: {removed}
- bumper hit counters: {jigs}
- ball: {ball}
- broken bumpers: {broken}

Choose ONE segment to prioritise next, and decide whether a repair is safe right now.
Reply with JSON only:
{{"target_seg":<0-{maxseg}>,"repair_ok":true|false,"why":"<=8 words"}}"""


class VLMStrategist:
    """上位の戦略層。**狙う区画** と **修理の可否助言** を返す。

    助言であって命令ではない。ドローンを出す/止めるの最終判断はシーケンサが持つ。
    """

    layer = "vlm"

    def __init__(self, model: str = "qwen2.5vl:3b", host: str = "http://localhost:11434",
                 num_predict: int = 44, timeout: float = 600.0,
                 unload_after: bool = True) -> None:
        import httpx

        self.model, self.host = model, host.rstrip("/")
        self.num_predict, self.unload_after = num_predict, unload_after
        self._client = httpx.Client(timeout=timeout)
        self.name = model

    def health(self) -> str:
        r = self._client.get(f"{self.host}/api/tags")
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        if self.model not in names:
            raise RuntimeError(f"{self.model} が Ollama にありません: {names}")
        return f"ollama ok ({self.model})"

    def unload(self) -> None:
        try:
            self._client.post(f"{self.host}/api/generate",
                              json={"model": self.model, "keep_alive": 0,
                                    "prompt": "", "stream": False})
        except Exception:
            pass

    def build_prompt(self, tel: dict) -> str:
        ball = tel["balls"][0] if tel["balls"] else {}
        return STRATEGY_PROMPT.format(
            nseg=N_SEG, maxseg=N_SEG - 1,
            blocks=tel["blocks_left_per_col"], removed=tel["broken_per_col"],
            jigs=[{"id": j["jid"], "hits": j["hits"]} for j in tel["jigs"]],
            ball={k: ball.get(k) for k in ("x", "y", "vy")},
            broken=[j["jid"] for j in tel["jigs"] if j["broken"]] or "none")

    def __call__(self, w: PinballWorld) -> tuple[dict, float, str, str, str | None]:
        tel = w.telemetry()
        prompt = self.build_prompt(tel)
        png = to_png(render_pinball(w))
        t0 = time.perf_counter()
        try:
            r = self._client.post(f"{self.host}/api/generate", json={
                "model": self.model, "prompt": prompt,
                "images": [base64.b64encode(png).decode("ascii")],
                "stream": False, "format": "json", "keep_alive": "30m",
                "options": {"temperature": 0.0, "num_predict": self.num_predict}})
            r.raise_for_status()
            raw = r.json().get("response", "")
        except Exception as exc:
            return {}, time.perf_counter() - t0, prompt, "", f"backend_error:{type(exc).__name__}"
        finally:
            if self.unload_after:
                self.unload()
        lat = time.perf_counter() - t0

        from .agents import _extract_json

        obj = _extract_json(raw) or {}
        out: dict = {}
        err = None
        try:
            out["target_seg"] = max(0, min(N_SEG - 1, int(obj["target_seg"])))
        except (KeyError, TypeError, ValueError):
            err = "parse_fail"
        if "repair_ok" in obj:
            out["repair_ok"] = bool(obj["repair_ok"])
        out["why"] = str(obj.get("why", ""))[:40]
        return out, lat, prompt, raw, err


class HeuristicStrategist:
    """VLM が無い環境でも動かすための戦略層。残ブロックが最多の列を狙う。"""

    layer = "vlm"
    name = "heuristic-strategy"

    def __call__(self, w: PinballWorld) -> tuple[dict, float, str, str, str | None]:
        t0 = time.perf_counter()
        # 均等にこなすのが目的なので「まだ残っていて、健全で、最もこなしていない工程」を狙う。
        # 壊れたジグはもう回せないので候補から外す (修理はシーケンサの仕事)
        left = [int(w.blocks[:, c].sum()) for c in range(N_SEG)]
        cand = [c for c in range(N_SEG) if left[c] > 0 and not w.jigs[c].broken]
        if cand:
            seg = min(cand, key=lambda c: (w.broken_per_col[c], -left[c]))
            why = "未消化かつ健全な工程"
        else:
            seg = next((c for c in range(N_SEG) if left[c] > 0), 0)
            why = "健全な工程がない"
        alive = [b for b in w.balls if b.alive]
        safe = not any(b.y > w.cfg.danger_y for b in alive)
        return ({"target_seg": seg, "repair_ok": safe, "why": why},
                time.perf_counter() - t0, "(内蔵ヒューリスティック)",
                json.dumps({"target_seg": seg, "repair_ok": safe}), None)


# --------------------------------------------------------------------------
# 実行ループ — 4 層を別クロックで回す
# --------------------------------------------------------------------------


@dataclass
class StackConfig:
    vla_every: int = 25       # VLA の行動列の長さ = 再計画周期
    vlm_every: int = 150      # VLM の問い合わせ周期 (tick)
    slowmo: float = 1.0       # 推論レイテンシを tick に換算するときの緩和倍率
    max_drift_ticks: int = 30
    keep_png_every: int = 1   # 記録に画像を残す間隔 (1 で全部)
    verbose: bool = True


def run_stack(
    world: PinballWorld,
    paddle_ctl,
    strategist,
    scfg: StackConfig | None = None,
    stream=None,
) -> dict:
    """1 面を走らせて、全層の呼び出しを記録して返す。"""
    import sys

    out = stream or sys.stdout
    scfg = scfg or StackConfig()
    plc = LadderPLC(build_ladder())
    seq = Sequencer()
    calls: list[LayerCall] = []
    frames: list[dict] = []
    cmd_queue: list[float] = []
    target_seg: int | None = None
    next_vlm = 0
    n = 0

    def log(m: str) -> None:
        if scfg.verbose:
            print(m, file=out, flush=True)

    def drift(latency: float) -> int:
        return min(scfg.max_drift_ticks,
                   int(round(latency / max(1e-6, world.cfg.dt * scfg.slowmo))))

    log(f"■ ピンボール×ブロック崩し  ラダー/シーケンサ=毎tick  "
        f"VLA={scfg.vla_every}tick  VLM={scfg.vlm_every}tick")
    log(f"  パドル制御: {getattr(paddle_ctl,'name','?')} / 戦略: {getattr(strategist,'name','?')}")
    log(f"  ジグ耐久度は隠れ状態 (打数だけ観測できる)")

    while not world.done:
        tick = world.tick

        # --- 層 4: VLM (最も遅く、最も賢い) -----------------------------
        if tick >= next_vlm:
            advice, lat, prompt, raw, err = strategist(world)
            n += 1
            d = drift(lat)
            calls.append(LayerCall(
                seq=n, layer="vlm", tick=tick, t=round(world.t, 2), latency_s=round(lat, 3),
                inputs={"telemetry": world.telemetry()}, output=advice, raw=raw,
                prompt=prompt, error=err,
                png_b64=base64.b64encode(to_png(render_pinball(world))).decode("ascii"),
                note=f"この推論の間に {d} tick 進む = 助言は {d*world.cfg.dt:.2f}秒 古い盤面のもの"))
            if "target_seg" in advice:
                target_seg = advice["target_seg"]
                cmd_queue.clear()      # 目標が変わったら行動列を捨てる
            seq.vlm_advice = advice
            log(f"\n[{n}] VLM t={world.t:.1f}s lat={lat:.2f}s → 狙い=セグ{advice.get('target_seg')} "
                f"修理可否={advice.get('repair_ok')} 「{advice.get('why','')}」"
                f"{' ⚠'+str(err) if err else ''}  (この間 {d} tick 進行)")
            next_vlm = tick + scfg.vlm_every
            _advance(world, plc, seq, cmd_queue, d, frames)

        if world.done:
            break

        # --- 層 3: VLA (行動列をまとめて作る) ---------------------------
        if not cmd_queue:
            cmds, lat, info = paddle_ctl.plan(world, target_seg)
            n += 1
            d = drift(lat)
            keep = (n % max(1, scfg.keep_png_every) == 0)
            calls.append(LayerCall(
                seq=n, layer="vla", tick=tick, t=round(world.t, 2), latency_s=round(lat, 3),
                inputs={"target_seg": target_seg,
                        "state6": [round(float(v), 3) for v in
                                   SmolVLAPaddle.encode_state(world, target_seg)]},
                output={"chunk_len": len(cmds), "first6": [round(v, 3) for v in cmds[:6]]},
                raw=json.dumps({"cmds": [round(v, 3) for v in cmds[:12]]}),
                png_b64=(base64.b64encode(
                    to_png(render_pinball(world, target_seg=target_seg))).decode("ascii")
                    if keep else None),
                note=f"1 回で {len(cmds)} 手ぶん (= {len(cmds)*world.cfg.dt:.2f}秒) の指令列"))
            cmd_queue.extend(cmds)
            log(f"[{n}] VLA t={world.t:.1f}s lat={lat:.2f}s → {len(cmds)}手の指令列 "
                f"先頭 {[round(v,2) for v in cmds[:5]]} (目標セグ{target_seg})")
            _advance(world, plc, seq, cmd_queue, d, frames)

        _advance(world, plc, seq, cmd_queue, 1, frames)

    sc = world.score()
    log(f"\n{'═'*64}")
    log(f"崩したブロック {sc['blocks_broken']}/{sc['blocks_total']}  列ごと {sc['per_col']}  "
        f"均等性 {sc['evenness']}  スコア {sc['score']}")
    log(f"パドル命中 {sc['paddle_hits']}  ロスト {sc['lost_balls']}")
    log(f"修理 成功{seq.repairs_ok} / 失敗{seq.repairs_failed}  "
        f"シーケンサ遷移 {len(seq.events)} 回  ラダースキャン {plc.scan_count} 回")
    for e in seq.events:
        log(f"  t={e.tick*world.cfg.dt:5.1f}s  {e.frm} → {e.to}   ({e.reason})")

    return {
        "score": sc,
        "ladder_trace": _compact_trace(plc, world.cfg.dt),
        "calls": [asdict(c) for c in calls],
        "frames": frames,
        "seq_events": [asdict(e) for e in seq.events],
        "ladder_scans": plc.scan_count,
        "repairs": {"ok": seq.repairs_ok, "failed": seq.repairs_failed},
        "world_events": world.events,
        "jig_truth": [{"jid": j.jid, "max_durability": j.max_durability,
                       "hits": j.hits, "broken": j.broken} for j in world.jigs],
        "config": {**asdict(world.cfg), **asdict(scfg)},
    }


def _compact_trace(plc: LadderPLC, dt: float) -> dict:
    """ドラレコ用にラダーのビット履歴を圧縮する。

    信号ごとに "0101..." の 1 文字/スキャンで持つ。2000 スキャン x 十数信号でも数十 KB。
    タイムチャート (PLC 屋が見慣れた縦=信号 / 横=時間 の図) をそのまま描ける。
    """
    if not plc.trace:
        return {"signals": {}, "scans": 0, "dt": dt}
    coils = [r.coil for r in plc.rungs]                       # ラングが書く出力
    allbits = sorted({k for fr in plc.trace for k in fr["coils"]})
    inputs = [k for k in allbits if k not in coils]            # 外から入る接点
    names = [k for k in coils if k in allbits]
    sig: dict[str, str] = {}
    for n in inputs + names:
        sig[n] = "".join("1" if fr["coils"].get(n) else "0" for fr in plc.trace)
    return {"signals": sig, "scans": len(plc.trace), "dt": dt,
            "coils": names, "inputs": inputs}


def _advance(world: PinballWorld, plc: LadderPLC, seq: Sequencer,
             queue: list[float], n_ticks: int, frames: list[dict]) -> None:
    """n tick 進める。ラダーとシーケンサは **毎 tick** 回る (ここが速い層)。"""
    for _ in range(max(0, n_ticks)):
        if world.done:
            return
        bits = plc.scan(world.ladder_inputs(), keep_trace=True)
        seq.step(world.tick, world, bits)
        cmd = queue.pop(0) if queue else 0.0
        world.step(cmd, bits)
        frames.append({
            "tick": world.tick, "t": round(world.t, 3),
            "paddle_x": round(world.paddle.x, 3),
            "balls": [[round(b.x, 3), round(b.y, 3)] for b in world.balls if b.alive],
            "blocks": int(world.blocks.sum()),
            "block_mask": int(sum(int(v) << i for i, v in enumerate(world.blocks.flatten()))),
            "jigs": [[j.hits, int(j.broken)] for j in world.jigs],
            "gates": [int(g.open) for g in world.gates],
            "drone": [round(world.drone.x, 2), round(world.drone.y, 2),
                      int(world.drone.active)],
            "state": seq.state.value,
        })


# --------------------------------------------------------------------------
# ピンボール用の behavior cloning
# --------------------------------------------------------------------------


def pinball_task(target_seg) -> str:
    """VLA に渡す指示文。上位 (VLM) の判断がここに言語として入る。"""
    if target_seg is None:
        return "return the balls with the paddle and keep them in play"
    return f"bounce the ball off the paddle toward bumper {int(target_seg)}"


def collect_pinball_expert(n_episodes: int = 24, chunk: int = 25, stride: int = 2,
                           seed0: int = 500, max_ticks: int = 2200, progress=None):
    """ScriptedPaddle を教師に (画像, state6, 行動列) を集める。

    上位の目標セグメントは HeuristicStrategist が与える。**学習後にその指令元を
    VLM へ差し替えられる** ようにするため、目標は必ず state と指示文の両方に入れる。
    """
    from .smolvla_pilot import BCSample

    teacher = ScriptedPaddle()
    strat = HeuristicStrategist()
    out = []
    for ep in range(n_episodes):
        w = PinballWorld(PinballConfig(seed=seed0 + ep, max_ticks=max_ticks))
        plc = LadderPLC(build_ladder())
        seq = Sequencer()
        obs_seq, act_seq = [], []
        target = None
        while not w.done:
            if w.tick % 150 == 0:
                target = strat(w)[0].get("target_seg")
            bits = plc.scan(w.ladder_inputs())
            seq.step(w.tick, w, bits)
            cmd, _ = teacher(w, target)
            obs_seq.append((SmolVLAPaddle.encode_state(w, target),
                            np.asarray(render_pinball(w, target_seg=target).convert("RGB"),
                                       dtype=np.uint8), target))
            v = np.zeros(6, dtype=np.float32)
            v[0] = cmd
            act_seq.append(v)
            w.step(cmd, bits)
        if not act_seq:
            continue
        pad = act_seq[-1]
        for i in range(0, len(obs_seq), stride):
            fut = act_seq[i:i + chunk]
            if len(fut) < chunk:
                fut = fut + [pad] * (chunk - len(fut))
            st, im, tg = obs_seq[i]
            out.append(BCSample(state=st, image=im, actions=np.stack(fut), target=tg))
        if progress:
            progress(f"  収集 ep{ep+1}/{n_episodes} 累計 {len(out)} サンプル")
    return out
