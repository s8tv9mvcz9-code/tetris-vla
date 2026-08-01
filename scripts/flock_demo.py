#!/usr/bin/env python
"""qwen-VL (どこへ) + SmolVLA (どうやって) だけで複数機を降ろし、HTML にする。

    .venv/bin/python scripts/flock_demo.py --landers 3 --slowmo 8 --out results/flock

PD ヒューリスティックも参照解も操縦には使わない (採点にのみ使用)。
"""
from __future__ import annotations

import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tetris_vla.flock import (FlockConfig, QwenCommander, load_flock, run_flock_demo,
                              save_flock)
from tetris_vla.flockviz import flock_html, flock_svg
from tetris_vla.parachute import ParachuteConfig
from tetris_vla.render import RenderConfig
from tetris_vla.smolvla_pilot import SmolVLAPilot, build_policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landers", type=int, default=3)
    ap.add_argument("--slowmo", type=float, default=8.0,
                    help="世界の進みを何倍ゆっくりにするか (1.0 が等倍)")
    ap.add_argument("--vlm-every", type=int, default=22, help="VLM 問い合わせ間隔 (tick)")
    ap.add_argument("--max-drift", type=int, default=20)
    ap.add_argument("--max-ticks", type=int, default=260)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--checkpoint", default="checkpoints/smolvla_bc.pt")
    ap.add_argument("--cell-px", type=int, default=12)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", default="results/flock")
    a = ap.parse_args()

    cmd = QwenCommander(model=a.model)
    print(cmd.health(), file=sys.stderr)
    policy, pre, post, _, dev = build_policy()
    if a.checkpoint and os.path.exists(a.checkpoint):
        sd = torch.load(a.checkpoint, map_location="cpu")
        policy.load_state_dict(sd["model"])
        policy.to(torch.device(dev)).eval()
        print(f"SmolVLA (BC後) 読み込み: {a.checkpoint}  device={dev}", file=sys.stderr)
    pilot = SmolVLAPilot(policy=policy, pre=pre, post=post, label="smolvla-bc")

    cfg = ParachuteConfig()
    fcfg = FlockConfig(n_landers=a.landers, slowmo=a.slowmo, vlm_every=a.vlm_every,
                       max_ticks=a.max_ticks, max_drift_ticks=a.max_drift,
                       warmup=not a.no_warmup, spawn_gap_ticks=6)
    t0 = time.perf_counter()
    res = run_flock_demo(pilot, cmd, cfg, fcfg, seed=a.seed,
                         rcfg=RenderConfig(cell_px=a.cell_px))
    print(f"\n実時間 {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    jsonl = f"{a.out}.jsonl"
    save_flock(res, jsonl, cfg, fcfg)
    data = load_flock(jsonl)
    svg = flock_svg(data)
    with open(f"{a.out}.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    with open(f"{a.out}.html", "w", encoding="utf-8") as f:
        f.write(flock_html(data, svg))
    print(f"出力: {a.out}.html / {a.out}.svg / {jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
