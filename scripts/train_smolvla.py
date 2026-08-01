#!/usr/bin/env python
"""PD パイロットの軌跡で SmolVLA を behavior cloning する。

    .venv/bin/python scripts/train_smolvla.py --episodes 40 --steps 400 --max-seconds 1500

視覚エンコーダと VLM 本体は凍結され、action expert だけが訓練される。
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetris_vla.parachute import ParachuteConfig
from tetris_vla.smolvla_pilot import collect_expert, train_bc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=1800)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="checkpoints/smolvla_bc.pt")
    a = ap.parse_args()

    log = lambda m: (print(m, flush=True))
    t0 = time.perf_counter()
    log(f"[1/2] 教師データ収集 ({a.episodes} エピソード, stride={a.stride}) ...")
    samples = collect_expert(n_episodes=a.episodes, cfg=ParachuteConfig(),
                             chunk=a.chunk, stride=a.stride, progress=log)
    mb = sum(s.image.nbytes for s in samples) / 1e6
    log(f"  {len(samples)} サンプル / 画像 {mb:.0f} MB / {time.perf_counter()-t0:.0f}s")

    log(f"[2/2] behavior cloning ({a.steps} step, batch {a.batch_size}) ...")
    _, _, _, hist = train_bc(samples, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                             chunk=a.chunk, device=a.device, out=a.out,
                             max_seconds=a.max_seconds, progress=log)
    os.makedirs("results", exist_ok=True)
    with open("results/smolvla_train.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "samples": len(samples), "history": hist},
                  f, ensure_ascii=False, indent=2)
    log(f"完了 {time.perf_counter()-t0:.0f}s  最終 loss {hist[-1]['loss'] if hist else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
