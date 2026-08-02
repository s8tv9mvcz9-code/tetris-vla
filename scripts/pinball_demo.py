#!/usr/bin/env python
"""ラダー + シーケンサ + VLA + VLM を 1 面で同居させ、HTML にする。

    .venv/bin/python scripts/pinball_demo.py --paddle mock --strategy heuristic
    .venv/bin/python scripts/pinball_demo.py --paddle mock --strategy vlm
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetris_vla.pinball import PinballConfig, PinballWorld
from tetris_vla.pinball_agents import (HeuristicStrategist, MockVLAPaddle, ScriptedPaddle,
                                       StackConfig, VLMStrategist, run_stack)
from tetris_vla.pinballviz import pinball_html, pinball_svg


class _ChunkedScripted:
    """教師をそのまま行動列にした対照 (VLA の上限性能の目安)。"""
    name = "scripted-chunk"
    layer = "vla"

    def __init__(self, chunk: int = 25):
        self.chunk, self._t = chunk, ScriptedPaddle()

    def plan(self, w, target_seg):
        c, info = self._t(w, target_seg)
        return [c] * self.chunk, 0.002, {**info, "chunk": self.chunk}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paddle", default="mock", choices=["mock", "scripted", "smolvla"])
    ap.add_argument("--strategy", default="heuristic", choices=["heuristic", "vlm"])
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--vlm-every", type=int, default=60)
    ap.add_argument("--slowmo", type=float, default=1.0)
    ap.add_argument("--max-ticks", type=int, default=5000)
    ap.add_argument("--skill", type=float, default=0.9)
    ap.add_argument("--out", default="results/pinball")
    # 文脈長。既定 (None) は ollama 任せで従来どおり。常駐メモリが文脈長で
    # 決まっている仮説を検証するための口 — docs/physical-ai-realtime.md 参照
    ap.add_argument("--num-ctx", type=int, default=None)
    a = ap.parse_args()

    world = PinballWorld(PinballConfig(seed=a.seed, max_ticks=a.max_ticks))
    print("隠れ耐久度(真値):", [(j.jid, j.durability) for j in world.jigs], file=sys.stderr)

    if a.paddle == "mock":
        paddle = MockVLAPaddle(chunk=a.chunk, skill=a.skill)
    elif a.paddle == "scripted":
        paddle = _ChunkedScripted(a.chunk)   # 解析解 (弾道外挿 + 狙いの逆算)
    else:
        from tetris_vla.pinball_agents import SmolVLAPaddle
        paddle = SmolVLAPaddle(checkpoint=a.checkpoint, chunk=a.chunk)

    if a.strategy == "vlm":
        strat = VLMStrategist(model=a.model, num_ctx=a.num_ctx)
        print(strat.health(), file=sys.stderr)
    else:
        strat = HeuristicStrategist()

    t0 = time.perf_counter()
    res = run_stack(world, paddle, strat,
                    StackConfig(vla_every=a.chunk, vlm_every=a.vlm_every, slowmo=a.slowmo))
    print(f"\n実時間 {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(f"{a.out}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    svg = pinball_svg(res)
    with open(f"{a.out}.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    with open(f"{a.out}.html", "w", encoding="utf-8") as f:
        f.write(pinball_html(res, svg))
    print(f"出力: {a.out}.html / {a.out}.svg / {a.out}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
