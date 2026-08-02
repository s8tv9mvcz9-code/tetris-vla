"""tetris_vla.smolvla_pilot — 真の VLA (SmolVLA) を操縦席に座らせる。

## なぜ VLM ではなく VLA なのか

qwen2.5vl のような VLM は「画像を見て JSON を喋る」。1 回の推論で **1 アクション** しか出ず、
M2 8GB では 1 回に 2〜10 秒かかる。降下が 6 秒で終わるこの課題では、判断が届く前に着地する。

SmolVLA (0.45B) は行動ヘッドを持つ本来の VLA で、決定的に違うのは **action chunking**:

    1 回の推論 (実測 0.43 秒) で 50 ステップ分の行動列がまとめて返る。
    50 ステップ = 6.0 論理秒 = 降下まるごと。

つまり「推論は遅いが 1 回で足りる」。VLM の「速くできないし 1 手しか出ない」とは
レイテンシの効き方が構造的に違う。これがこの PoC で見たかった VLA の性格そのもの。

## 事前学習重みはそのままでは使えない

`lerobot/smolvla_base` はロボット操作 (関節角) で訓練されている。この課題の行動空間
(横推進 / ベント) とは意味が違うので、ゼロショットの出力はほぼランダムになる。
そこで **PD パイロットの軌跡を教師にして behavior cloning** する。
視覚エンコーダと VLM 本体は凍結し、action expert だけを訓練する (8GB で回る範囲)。

## 契約 (実物から確認した値)

    observation.state            (6,)       [列, 行, vx, vy, 燃料, 風]  を正規化
    observation.images.camera1   (3,256,256) 降下画面
    action                       (50, 6)    先頭 2 次元だけ使う
                                            [0] 横推進 -1..+1  [1] ベント 0/1
    task                         自然言語の指示 (1 本)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .parachute import (
    FlightDecision,
    FlightObservation,
    ParachuteConfig,
    ParachuteWorld,
    PDPilot,
    Thrust,
    make_terrain,
    render_flight,
    spawn_lander,
)
from .render import RenderConfig

#: SmolVLA に与える自然言語の指示。VLA の "L" の部分。
#: **上位の判断はここに文章として入る。** 階層型 VLA (VLM が目標を言語で渡し、
#: VLA が軌道を実行する) を成立させている接点。
TASK_INSTRUCTION = "land the falling block on a flat surface without creating a cavity underneath"


def task_for(target_col: int | None) -> str:
    """上位から与えられた目標列を自然言語の指示にする。"""
    if target_col is None:
        return TASK_INSTRUCTION
    return f"land the falling block in column {int(target_col)}"

STATE_DIM = 6
ACTION_DIM = 6
IMG_SIZE = 256

#: 行動のしきい値。連続値ヘッドの出力を離散の噴射コマンドに戻す
THRUST_THRESHOLD = 0.33
VENT_THRESHOLD = 0.5


def encode_action(act: Thrust) -> np.ndarray:
    """離散の噴射コマンドを、行動ヘッドが吐く連続ベクトルに直す。"""
    v = np.zeros(ACTION_DIM, dtype=np.float32)
    if act == Thrust.LEFT:
        v[0] = -1.0
    elif act == Thrust.RIGHT:
        v[0] = 1.0
    elif act == Thrust.VENT:
        v[1] = 1.0
    return v


def decode_action(vec) -> Thrust:
    """行動ヘッドの連続出力を噴射コマンドに戻す。"""
    v = np.asarray(vec, dtype=np.float32).flatten()
    if v.shape[0] < 2:
        return Thrust.COAST
    if v[1] > VENT_THRESHOLD and abs(v[0]) <= THRUST_THRESHOLD:
        return Thrust.VENT
    if v[0] > THRUST_THRESHOLD:
        return Thrust.RIGHT
    if v[0] < -THRUST_THRESHOLD:
        return Thrust.LEFT
    return Thrust.COAST


def encode_state(obs: FlightObservation, target_col: int | None = None) -> np.ndarray:
    """センサ値を 6 次元に畳む。おおむね [-1, 1] に収まるようスケールする。

    最後の 1 次元が **上位から指令された目標までの誤差**。ここに何を入れるかで
    同じ重みのまま「誰が目標を決めるか」を差し替えられる:
        オラクル -> 上限性能の測定
        VLM      -> 階層型 (qwen が見て決め、SmolVLA が飛ばす)
        None     -> 目標なし (自分の現在位置を目標とみなす = 何もしない指令)
    """
    c = obs.cfg
    err = 0.0 if target_col is None else (target_col - obs.col) / max(1, c.width - 1)
    return np.array([
        obs.col / max(1, c.width - 1) * 2.0 - 1.0,     # 横位置
        obs.y / max(1, c.height) * 2.0 - 1.0,          # 高度
        obs.vx / max(1e-6, c.terminal_vx),             # 横速度
        obs.vy / max(1e-6, c.terminal_vy),             # 落下速度
        obs.fuel / max(1, c.fuel) * 2.0 - 1.0,         # 燃料
        max(-1.0, min(1.0, err * 2.0)),                # 目標までの誤差 (上位からの指令)
    ], dtype=np.float32)


def encode_image(obs: FlightObservation):
    """降下画面を (3, 256, 256) の float テンソルに直す。"""
    return image_to_tensor(obs.image)


def image_to_tensor(src):
    """PIL 画像 (120x254) をアスペクト保持で 256x256 に載せ、float テンソルにする。

    余白は黒。教師データは元サイズの uint8 で持ち、バッチを組む時にここを通す
    (float32 の 256x256 を数千枚メモリに置くと 8GB では足りないため)。
    """
    import torch
    from PIL import Image

    src = src.convert("RGB")
    scale = min(IMG_SIZE / src.width, IMG_SIZE / src.height)
    new = src.resize((max(1, int(src.width * scale)), max(1, int(src.height * scale))),
                     Image.NEAREST)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    canvas.paste(new, ((IMG_SIZE - new.width) // 2, (IMG_SIZE - new.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def image_from_uint8(arr: np.ndarray):
    """collect_expert が貯めた uint8 (H,W,3) をバッチ用テンソルに戻す。"""
    from PIL import Image

    return image_to_tensor(Image.fromarray(arr))


# --------------------------------------------------------------------------
# ポリシーの生成
# --------------------------------------------------------------------------


def pick_device(prefer: str = "auto") -> str:
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_policy(device: str = "auto", pretrained: str | None = "lerobot/smolvla_base",
                 chunk_size: int = 50):
    """この課題の入出力次元に合わせた SmolVLA を作る。

    state 6 / action 6 / カメラ 1 本。事前学習重みの入出力射影は 32 次元にパディング
    されているので、次元を変えてもそのまま読み込める。
    """
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    dev = pick_device(device)
    cfg = SmolVLAConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.images.camera1": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, IMG_SIZE, IMG_SIZE)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        device=dev,
    )
    # 状態も行動もこちらで [-1,1] に収めてあるので、正規化は恒等にする
    stats = {
        "observation.state": {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        "action": {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
    }
    if pretrained:
        policy = SmolVLAPolicy.from_pretrained(pretrained, config=cfg, dataset_stats=stats)
    else:
        policy = SmolVLAPolicy(cfg, dataset_stats=stats)
    policy.to(torch.device(dev))
    pre, post = make_smolvla_pre_post_processors(cfg, stats)
    return policy, pre, post, cfg, dev


# --------------------------------------------------------------------------
# パイロット
# --------------------------------------------------------------------------


class SmolVLAPilot:
    """SmolVLA を操縦席に。

    `select_action` は内部キューから 1 手ずつ返し、キューが空になった時だけ
    実際に推論する。したがって報告されるレイテンシは
    「推論した tick だけ 0.4 秒、それ以外は 3 ミリ秒」になる。
    fly() はこのレイテンシをそのまま漂流 tick に換算するので、
    action chunking の効果が成績にそのまま出る。
    """

    def __init__(self, policy=None, pre=None, post=None, device: str = "auto",
                 checkpoint: str | None = None, chunk_size: int = 50,
                 target_source=None, label: str | None = None) -> None:
        import torch

        if policy is None:
            policy, pre, post, _, device = build_policy(device, chunk_size=chunk_size)
            if checkpoint:
                sd = torch.load(checkpoint, map_location="cpu")
                policy.load_state_dict(sd["model"] if "model" in sd else sd)
                policy.to(torch.device(pick_device(device)))
        self.policy = policy.eval()
        self.pre, self.post = pre, post
        #: 目標列の供給元。obs -> (target_col | None, 追加レイテンシ秒)
        self.target_source = target_source
        self.device = pick_device(device)
        self.name = label or ("smolvla" + ("-bc" if checkpoint else "-zeroshot"))
        self.inference_calls = 0
        self.last_target: int | None = None
        self._torch = torch

    def reset(self) -> None:
        self.policy.reset()
        self.last_target = None

    def decide(self, obs: FlightObservation) -> FlightDecision:
        torch = self._torch
        queued = len(self.policy._queues.get("action", ()))
        extra_latency = 0.0
        note = ""
        if self.target_source is not None:
            tgt, extra_latency, note = self.target_source(obs)
            if tgt is not None:
                self.last_target = tgt

        t0 = time.perf_counter()
        batch = {
            "observation.state": torch.from_numpy(encode_state(obs, self.last_target)),
            "observation.images.camera1": encode_image(obs),
            "task": task_for(self.last_target),
        }
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        latency = time.perf_counter() - t0 + extra_latency
        vec = action.detach().float().cpu().numpy().flatten()
        act = decode_action(vec)
        if queued == 0:
            self.inference_calls += 1
        why = f"chunk残{len(self.policy._queues.get('action', ()))} u={vec[0]:+.2f}"
        if note:
            why = f"{note} / {why}"
        return FlightDecision(
            act, self.last_target, why[:40], latency_s=latency,
            raw=json.dumps({"action": [round(float(v), 4) for v in vec[:2]],
                            "target_col": self.last_target}),
        )


# --------------------------------------------------------------------------
# 教師データ (PD パイロットの軌跡)
# --------------------------------------------------------------------------


@dataclass
class BCSample:
    state: np.ndarray          # (6,)
    image: np.ndarray          # uint8 (H, W, 3)  元解像度のまま保持してメモリを節約
    actions: np.ndarray        # (chunk, 6)
    target: int | None = None  # 上位からの指令 (指示文の生成に使う)


def collect_expert(
    n_episodes: int = 40,
    cfg: ParachuteConfig | None = None,
    chunk: int = 50,
    seed0: int = 1000,
    render_cfg: RenderConfig | None = None,
    stride: int = 1,
    progress=None,
) -> list[BCSample]:
    """PD パイロットを走らせて (画像, 状態, 行動列) を集める。

    PD は 3 ミリ秒で決まるので、教師データはいくらでも安く作れる。
    行動は **毎 tick** 記録する (chunk 学習のため密である必要がある)。
    """
    cfg = cfg or ParachuteConfig()
    render_cfg = render_cfg or RenderConfig()
    expert = PDPilot()
    samples: list[BCSample] = []

    for ep in range(n_episodes):
        seed = seed0 + ep
        board = make_terrain(cfg, seed)
        kind = random.Random(seed).choice(list("IOTSZJL"))
        lander = spawn_lander(cfg, kind, seed, board)
        world = ParachuteWorld(cfg, board, lander)

        obs_seq: list[tuple[np.ndarray, object]] = []
        act_seq: list[np.ndarray] = []
        while not world.landed and not world.timeout:
            img = render_flight(world, cfg, render_cfg)
            obs = FlightObservation(
                png=b"", image=img, kind=kind, rot=lander.rot, x=lander.x, y=lander.y,
                vx=lander.vx, vy=lander.vy, fuel=lander.fuel, tick=world.tick, t=world.t,
                width=cfg.width, height=cfg.height, cfg=cfg, render_cfg=render_cfg,
            )
            d = expert.decide(obs)          # 教師は毎 tick 最適行動を出す
            # 教師の目標列を「上位からの指令」として state と指示文に入れる。
            # こうすると学習後に指令元を VLM に差し替えられる
            obs_seq.append((encode_state(obs, d.target_col),
                            np.asarray(img.convert("RGB"), dtype=np.uint8), d.target_col))
            act_seq.append(encode_action(d.action))
            world.step(d.action)

        if not act_seq:
            continue
        pad = act_seq[-1]
        for i in range(0, len(obs_seq), stride):
            fut = act_seq[i:i + chunk]
            if len(fut) < chunk:
                fut = fut + [pad] * (chunk - len(fut))
            st, im, tgt = obs_seq[i]
            samples.append(BCSample(state=st, image=im, actions=np.stack(fut), target=tgt))
        if progress:
            progress(f"  収集 ep{ep+1}/{n_episodes} 累計 {len(samples)} サンプル")
    return samples


def train_bc(
    samples: Sequence[BCSample],
    task_fn=None,
    steps: int = 300,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "auto",
    chunk: int = 50,
    out: str | None = None,
    max_seconds: float | None = None,
    log_every: int = 20,
    seed: int = 0,
    progress=None,
):
    """action expert だけを behavior cloning で訓練する。

    視覚エンコーダと VLM 本体は凍結 (SmolVLA の既定)。8GB でも回る範囲に収めている。
    """
    import torch

    policy, pre, post, cfg, dev = build_policy(device, chunk_size=chunk)
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    dev_t = torch.device(dev)

    history: list[dict] = []
    t_start = time.perf_counter()
    if progress:
        progress(f"  訓練対象パラメータ {n_train/1e6:.1f}M / 全体 "
                 f"{sum(p.numel() for p in policy.parameters())/1e6:.1f}M  device={dev}")

    for step in range(1, steps + 1):
        idx = [rng.randrange(len(samples)) for _ in range(batch_size)]
        batch = {
            "observation.state": torch.stack(
                [torch.from_numpy(samples[i].state) for i in idx]).to(dev_t),
            "observation.images.camera1": torch.stack(
                [image_from_uint8(samples[i].image) for i in idx]).to(dev_t),
            "action": torch.stack(
                [torch.from_numpy(samples[i].actions) for i in idx]).to(dev_t),
            "task": [(task_fn or task_for)(samples[i].target) for i in idx],
        }
        loss, _ = policy.forward(pre(batch))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        elapsed = time.perf_counter() - t_start
        if step % log_every == 0 or step == 1:
            history.append({"step": step, "loss": round(float(loss.item()), 5),
                            "elapsed_s": round(elapsed, 1)})
            if progress:
                progress(f"  step {step:>4}/{steps}  loss {float(loss.item()):.4f}  "
                         f"{elapsed:.0f}s ({elapsed/step:.2f}s/step)")
        if max_seconds and elapsed > max_seconds:
            if progress:
                progress(f"  時間上限 {max_seconds:.0f}s に到達したので step {step} で打ち切り")
            break

    policy.eval()
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        torch.save({"model": policy.state_dict(), "history": history,
                    "chunk": chunk, "steps": step}, out)
        if progress:
            progress(f"  保存: {out} ({os.path.getsize(out)/1e6:.0f} MB)")
    return policy, pre, post, history


# --------------------------------------------------------------------------
# 目標列の供給元 — 「誰が どこへ を決めるか」を差し替える口
# --------------------------------------------------------------------------


class OracleTarget:
    """画像から Dellacherie 参照解を計算して目標にする。上限性能の測定用。"""

    def __init__(self) -> None:
        self._pd = PDPilot()

    def __call__(self, obs: FlightObservation):
        from .render import decode
        from .parachute import target_column

        t0 = time.perf_counter()
        terrain = decode(obs.image, obs.width, obs.height, obs.render_cfg).settled
        tgt = target_column(terrain, obs.kind, obs.rot, obs.persona)
        return tgt, time.perf_counter() - t0, f"参照解 列{tgt}"


class NoTarget:
    """指令なし。VLA は自力で判断するしかない (BC 時の条件から外れる ablation)。"""

    def __call__(self, obs: FlightObservation):
        return None, 0.0, "指令なし"


#: 上位 VLM に投げる質問。**列番号だけ**を求めるので出力が極端に短く、
#: 完全な操縦を任せる場合より桁違いに速い (実測 9.3s -> 1〜2s)。
TARGET_PROMPT = """Look at the terrain in this image.
Dark blocks are the ground. The bright block is a piece descending by parachute.
The board has {width} columns, numbered 0 (left) to {maxcol} (right).

Which column should the piece land in so that NO empty cavity is left underneath it?
Prefer a flat surface, or a notch that matches the piece shape.

Reply with JSON only: {{"target_col": <0-{maxcol}>}}"""


class VLMTarget:
    """上位 VLM (Ollama) が目標列を決める。階層型の "どこへ" 側。

    every_ticks ごとにだけ問い合わせる。それ以外の tick では (None, 0, "") を返し、
    VLA は直前の指令を持ち越す。**VLM の遅さは、問い合わせた tick にだけ課金される。**
    """

    def __init__(self, model: str = "qwen2.5vl:3b", host: str = "http://localhost:11434",
                 every_ticks: int = 20, num_predict: int = 12, temperature: float = 0.0,
                 timeout: float = 300.0, keep_alive: str = "30m") -> None:
        import httpx

        self.model, self.host = model, host.rstrip("/")
        self.every_ticks, self.num_predict = every_ticks, num_predict
        self.temperature, self.keep_alive = temperature, keep_alive
        self._client = httpx.Client(timeout=timeout)
        self._next_tick = 0
        self.calls = 0
        self.latencies: list[float] = []
        self.raw_log: list[dict] = []

    def reset(self) -> None:
        self._next_tick = 0
        self.calls = 0
        self.latencies.clear()
        self.raw_log.clear()

    def __call__(self, obs: FlightObservation):
        if obs.tick < self._next_tick:
            return None, 0.0, ""
        self._next_tick = obs.tick + self.every_ticks
        import base64

        from .parachute import to_png

        png = obs.png or to_png(obs.image)
        prompt = TARGET_PROMPT.format(width=obs.width, maxcol=obs.width - 1)
        t0 = time.perf_counter()
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
            return None, time.perf_counter() - t0, f"VLM異常:{type(exc).__name__}"
        latency = time.perf_counter() - t0
        self.calls += 1
        self.latencies.append(latency)

        from .agents import _extract_json

        obj = _extract_json(raw) or {}
        try:
            tgt = int(obj.get("target_col"))
        except (TypeError, ValueError):
            self.raw_log.append({"tick": obs.tick, "raw": raw, "target": None, "latency": latency})
            return None, latency, "VLM解釈不能"
        tgt = max(0, min(obs.width - 1, tgt))
        self.raw_log.append({"tick": obs.tick, "raw": raw, "target": tgt, "latency": latency})
        return tgt, latency, f"VLM指令 列{tgt}"


def make_pilot(mode: str, checkpoint: str | None = None, device: str = "auto",
               chunk_size: int = 50, vlm_model: str = "qwen2.5vl:3b",
               vlm_every: int = 20):
    """比較したい 4 構成をまとめて作る。

        vla-oracle : 参照解を指令 -> VLA が飛ばす (上限)
        vla-vlm    : qwen が指令 -> VLA が飛ばす (階層型)  ← 協力構成
        vla-solo   : 指令なし
        vla-zeroshot: BC なしの事前学習重みそのまま
    """
    src = {"vla-oracle": OracleTarget(), "vla-solo": NoTarget(),
           "vla-zeroshot": OracleTarget()}.get(mode)
    if mode == "vla-vlm":
        src = VLMTarget(model=vlm_model, every_ticks=vlm_every)
    if src is None:
        raise ValueError(f"未知の mode: {mode}")
    return SmolVLAPilot(
        device=device, chunk_size=chunk_size, target_source=src, label=mode,
        checkpoint=None if mode == "vla-zeroshot" else checkpoint,
    )
