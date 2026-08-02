#!/usr/bin/env python
"""各層を「学習モデル / ヒューリスティック」で差し替えて比較する。

    .venv/bin/python scripts/pinball_matrix.py --seeds 6

層のインタフェースだけ守れば中身は差し替え可能、という主張を数字で裏付けるための表。
"""
from __future__ import annotations

import argparse, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetris_vla.pinball import PinballConfig, PinballWorld
from tetris_vla.pinball_agents import (HeuristicStrategist, MockVLAPaddle, ScriptedPaddle,
                                       StackConfig, run_stack)


class ChunkedScripted:
    """教師をそのまま行動列にした対照 (= 解析解が使える場合の上限)。"""
    name = "scripted(解析解)"
    layer = "vla"

    def __init__(self, chunk: int = 25):
        self.chunk, self._t = chunk, ScriptedPaddle()

    def plan(self, w, seg):
        c, info = self._t(w, seg)
        return [c] * self.chunk, 0.002, {**info, "chunk": self.chunk}


def bench(make_paddle, make_strat, seeds, ticks, label, every=60):
    rows = []
    for s in seeds:
        w = PinballWorld(PinballConfig(seed=s, max_ticks=ticks))
        r = run_stack(w, make_paddle(), make_strat(),
                      StackConfig(vlm_every=every, verbose=False))
        rows.append((r["score"], r["repairs"]))
    sc = [x[0] for x in rows]
    import statistics as st
    return {
        "label": label,
        "sd": st.stdev(x["score"] for x in sc) if len(sc) > 1 else 0.0,
        "score": statistics.fmean(x["score"] for x in sc),
        "broken": statistics.fmean(x["blocks_broken"] for x in sc),
        "even": statistics.fmean(x["evenness"] for x in sc),
        "hits": statistics.fmean(x["paddle_hits"] for x in sc),
        "lost": statistics.fmean(x["lost_balls"] for x in sc),
        "rep_ok": sum(x[1]["ok"] for x in rows),
        "rep_ng": sum(x[1]["failed"] for x in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--vlm-every", type=int, default=60,
                    help="上位の指令周期。長いと腕前の差が消えるので既定は新鮮側")
    ap.add_argument("--checkpoint", default="checkpoints/smolvla_pinball.pt")
    ap.add_argument("--out", default="results/pinball_matrix.md")
    a = ap.parse_args()
    seeds = list(range(a.seeds))

    variants = [
        (lambda: ChunkedScripted(), "scripted(解析解)"),
        (lambda: MockVLAPaddle(skill=0.9), "mock-VLA skill=0.9"),
        (lambda: MockVLAPaddle(skill=0.5), "mock-VLA skill=0.5"),
        (lambda: MockVLAPaddle(skill=0.0), "mock-VLA skill=0.0 (無作為)"),
    ]
    if a.checkpoint and os.path.exists(a.checkpoint):
        from tetris_vla.pinball_agents import SmolVLAPaddle
        shared = SmolVLAPaddle(checkpoint=a.checkpoint, chunk=25)
        variants.insert(1, (lambda: shared, "SmolVLA (BC後)"))
        zero = SmolVLAPaddle(checkpoint=None, chunk=25)
        variants.append((lambda: zero, "SmolVLA (学習前)"))

    out = []
    for mk, label in variants:
        print(f"  {label} ...", file=sys.stderr)
        out.append(bench(mk, HeuristicStrategist, seeds, a.ticks, label, a.vlm_every))

    hdr = ("| パドル制御層 | 点↑ | ±σ | 崩し↑ | 均等性↑ | 命中 | ロスト↓ | 修理成功/失敗 |\n"
           "|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r['label']} | {r['score']:.0f} | ±{r['sd']:.0f} | {r['broken']:.1f}/24 | "
        f"{r['even']:.2f} | {r['hits']:.1f} | {r['lost']:.1f} | "
        f"{r['rep_ok']}/{r['rep_ng']} |\n" for r in out)
    md = (f"# パドル制御層の差し替え比較\n\n"
          f"- seeds: {seeds} / 1面 {a.ticks} tick / 戦略層=ヒューリスティック固定\n"
          f"- 上位の指令周期: {a.vlm_every} tick "
          f"(これを伸ばすと腕前の差は消える — 別表参照)\n\n" + hdr + body)
    print()
    print(md)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(a.out.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"出力: {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
