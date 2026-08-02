#!/usr/bin/env python
"""ピンボールのパドル制御を SmolVLA に behavior cloning する。

    .venv/bin/python scripts/train_pinball_vla.py --episodes 6 --steps 500

教師は ScriptedPaddle (着弾点を目標セグメントへ寄せる制御則)。
視覚エンコーダと VLM 本体は凍結、action expert だけを訓練する。
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetris_vla.pinball_agents import collect_pinball_expert, pinball_task
from tetris_vla.smolvla_pilot import train_bc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seconds", type=float, default=1200)
    ap.add_argument("--out", default="checkpoints/smolvla_pinball.pt")
    ap.add_argument("--bang-bang", type=int, default=1)
    a = ap.parse_args()

    log = lambda m: print(m, flush=True)
    t0 = time.perf_counter()
    log(f"[1/2] 教師データ収集 ({a.episodes} エピソード, stride={a.stride}) ...")
    samples = collect_pinball_expert(n_episodes=a.episodes, chunk=a.chunk,
                                     stride=a.stride, progress=log)
    mb = sum(s.image.nbytes for s in samples) / 1e6
    log(f"  {len(samples)} サンプル / 画像 {mb:.0f} MB / {time.perf_counter()-t0:.0f}s")

    log(f"[2/2] behavior cloning ({a.steps} step, batch {a.batch_size}) ...")
    _, _, _, hist = train_bc(samples, task_fn=pinball_task, steps=a.steps,
                             batch_size=a.batch_size, lr=a.lr, chunk=a.chunk,
                             out=a.out, max_seconds=a.max_seconds, progress=log)
    os.makedirs("results", exist_ok=True)
    with open("results/pinball_train.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "samples": len(samples), "history": hist},
                  f, ensure_ascii=False, indent=2)
    log(f"完了 {time.perf_counter()-t0:.0f}s  最終 loss {hist[-1]['loss'] if hist else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
