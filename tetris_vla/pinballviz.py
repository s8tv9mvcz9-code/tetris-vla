"""tetris_vla.pinballviz — 4 層スタックの動きを 1 枚の HTML に落とす。

見せたいのは点数ではなく **層ごとのクロックの違い**:

    ラダー/シーケンサ  毎 tick   決定的・即応
    VLA                25 tick   1 回で 25 手ぶんの指令列
    VLM                150 tick  賢いが、返ってくる頃には盤面が古い

したがって画面は「盤面アニメ + 層タイムライン + 各層の入出力カード」。
外部参照ゼロの自己完結 HTML。
"""

from __future__ import annotations

import html
import json
from typing import Sequence

from .pinball import (
    BALL_R,
    BLOCK_H,
    BLOCK_ROWS,
    BLOCK_TOP,
    FIELD_H,
    FIELD_W,
    N_SEG,
    PADDLE_Y,
    PADDLE_W,
    build_ladder,
)

BG = "#101016"
WALLC = "#36364a"
INK = "#e8e8ef"
DIM = "#9a9aa8"
LAYER_C = {"ladder": "#7ee787", "sequencer": "#ffd24d", "vla": "#5ac8fa", "vlm": "#c98bff"}
LAYER_JA = {"ladder": "ラダー", "sequencer": "シーケンサ", "vla": "VLA", "vlm": "VLM"}
BLOCK_COLS = ["#ff6961", "#ffb347", "#fdfd96", "#77dd77", "#779ecb", "#c892e8"]


def _fmt(vals, nd=2) -> str:
    return ";".join(f"{v:.{nd}f}" for v in vals)


def pinball_svg(res: dict, px: int = 11, speed: float = 1.0, stride: int = 2) -> str:
    """盤面アニメ。フレームが多いので stride で間引く。"""
    frames = res["frames"][::max(1, stride)]
    if not frames:
        raise ValueError("フレームがありません")
    cfg = res.get("config", {})
    dt = float(cfg.get("dt", 0.02)) * max(1, stride)
    n = len(frames)
    dur = max(1.0, n * dt / max(0.05, speed))
    kt = _fmt([i / max(1, n - 1) for i in range(n)], 4)
    W, H = int(FIELD_W * px), int(FIELD_H * px)
    band = 30
    o: list[str] = []
    a = o.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H+band}" width="{W}" '
      f'height="{H+band}" font-family="ui-monospace,Menlo,monospace">')
    a(f'<rect width="{W}" height="{H+band}" fill="{BG}"/>')
    a(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" fill="none" stroke="{WALLC}" stroke-width="2"/>')

    # ブロック: 各セルの残存を discrete で明滅させる
    cw = FIELD_W / N_SEG
    for r in range(BLOCK_ROWS):
        for c in range(N_SEG):
            idx = r * N_SEG + c
            vis = [(f["block_mask"] >> idx) & 1 for f in frames]
            if not any(vis):
                continue
            x0 = (c * cw + 0.35) * px
            y0 = (BLOCK_TOP + r * BLOCK_H) * px
            w_ = (cw - 0.7) * px
            h_ = (BLOCK_H - 0.3) * px
            a(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w_:.1f}" height="{h_:.1f}" '
              f'fill="{BLOCK_COLS[c % len(BLOCK_COLS)]}" opacity="0">'
              f'<animate attributeName="opacity" values="{_fmt(vis,0)}" keyTimes="{kt}" '
              f'dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></rect>')

    # ゲート (開=緑 / 閉=赤)
    for gi, gx in enumerate((7.0, 17.0)):
        op = [f["gates"][gi] for f in frames]
        for state, color in ((1, "#3cc87a"), (0, "#c85050")):
            vals = [1.0 if v == state else 0.0 for v in op]
            a(f'<rect x="{(gx-2.0)*px:.1f}" y="{(21-0.18)*px:.1f}" width="{4.0*px:.1f}" '
              f'height="{0.36*px:.1f}" fill="{color}" opacity="0">'
              f'<animate attributeName="opacity" values="{_fmt(vals,0)}" keyTimes="{kt}" '
              f'dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></rect>')

    # ジグ (壊れると輪郭だけになる)。**ジグ j に当たると列 j が崩れる** ので、
    # ブロックと同じ色で縁取って対応関係を見えるようにする
    from .pinball import PinballWorld

    spots = PinballWorld.JIG_SPOTS[:N_SEG]
    truth = {t["jid"]: t for t in res.get("jig_truth", [])}
    for jid, (jx, jy) in enumerate(spots):
        broken = [f["jigs"][jid][1] for f in frames]
        col = BLOCK_COLS[jid % len(BLOCK_COLS)]
        a(f'<circle cx="{jx*px:.1f}" cy="{jy*px:.1f}" r="{1.1*px:.1f}" fill="#78c8ff" '
          f'stroke="{col}" stroke-width="3" opacity="0"><animate attributeName="opacity" '
          f'values="{_fmt([0.0 if b else 1.0 for b in broken],0)}" keyTimes="{kt}" '
          f'dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        a(f'<circle cx="{jx*px:.1f}" cy="{jy*px:.1f}" r="{1.1*px:.1f}" fill="none" '
          f'stroke="#6a6a78" stroke-width="2" stroke-dasharray="3 3" opacity="0">'
          f'<animate attributeName="opacity" values="{_fmt([1.0 if b else 0.0 for b in broken],0)}" '
          f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        a(f'<text x="{jx*px:.1f}" y="{(jy+0.3)*px:.1f}" fill="#0b0b11" font-size="11" '
          f'text-anchor="middle" font-weight="bold">{jid}</text>')

    # パドル (セグメント線つき)
    pw = PADDLE_W * px
    pad_vals = ";".join(f"{(f['paddle_x'] - PADDLE_W / 2) * px:.1f},0" for f in frames)
    a(f'<g><animateTransform attributeName="transform" type="translate" '
      f'values="{pad_vals}" '
      f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite"/>')
    a(f'<rect x="0" y="{(PADDLE_Y-0.3)*px:.1f}" width="{pw:.1f}" height="{0.6*px:.1f}" '
      f'fill="#64f0b4" rx="2"/>')
    a('</g>')

    # ボール (複数)
    nb = max((len(f["balls"]) for f in frames), default=0)
    for k in range(nb):
        bx = [(f["balls"][k][0] * px if len(f["balls"]) > k else -99) for f in frames]
        by = [(f["balls"][k][1] * px if len(f["balls"]) > k else -99) for f in frames]
        a(f'<circle r="{BALL_R*px:.1f}" fill="#fff" stroke="#9ad" stroke-width="1">'
          f'<animate attributeName="cx" values="{_fmt(bx)}" keyTimes="{kt}" dur="{dur:.2f}s" '
          f'repeatCount="indefinite"/><animate attributeName="cy" values="{_fmt(by)}" '
          f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite"/></circle>')

    # 救済機
    dx = [f["drone"][0] * px for f in frames]
    dy = [f["drone"][1] * px for f in frames]
    da = [float(f["drone"][2]) for f in frames]
    a(f'<rect width="{1.0*px:.1f}" height="{1.0*px:.1f}" fill="#ff78dc" opacity="0" rx="2">'
      f'<animate attributeName="x" values="{_fmt([v-0.5*px for v in dx])}" keyTimes="{kt}" '
      f'dur="{dur:.2f}s" repeatCount="indefinite"/>'
      f'<animate attributeName="y" values="{_fmt([v-0.5*px for v in dy])}" keyTimes="{kt}" '
      f'dur="{dur:.2f}s" repeatCount="indefinite"/>'
      f'<animate attributeName="opacity" values="{_fmt(da,0)}" keyTimes="{kt}" '
      f'dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></rect>')

    # 下帯: シーケンサの状態を文字で
    states = sorted({f["state"] for f in frames})
    a(f'<g transform="translate(4,{H+12})">')
    a(f'<text x="0" y="0" fill="{DIM}" font-size="9">シーケンサ:</text>')
    for i, st in enumerate(states):
        vals = [1.0 if f["state"] == st else 0.0 for f in frames]
        a(f'<text x="{62+i*54}" y="0" fill="{LAYER_C["sequencer"]}" font-size="9" opacity="0.18">'
          f'{html.escape(st)}<animate attributeName="opacity" values="{_fmt(vals,2)}" '
          f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></text>')
    a('</g>')

    # 推論帯 (VLA / VLM の呼び出し位置)
    total = max(1, res["frames"][-1]["tick"])
    a(f'<g transform="translate(0,{H+18})">')
    for c in res["calls"]:
        x0 = c["tick"] / total * W
        w_ = max(2.0, c["latency_s"] / max(1e-6, float(cfg.get("dt", 0.02))) / total * W)
        y = 4 if c["layer"] == "vlm" else 9
        a(f'<rect x="{x0:.1f}" y="{y}" width="{w_:.1f}" height="4" rx="1" '
          f'fill="{LAYER_C.get(c["layer"], "#888")}" opacity="0.9"><title>'
          f'[{c["seq"]}] {LAYER_JA.get(c["layer"], c["layer"])} {c["latency_s"]:.2f}s</title></rect>')
    a('</g></svg>')
    return "\n".join(o)


# --------------------------------------------------------------------------

_CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;padding:24px;line-height:1.7;
 font-family:ui-sans-serif,system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
h1{font-size:1.4rem;margin:0 0 .2rem}
h2{font-size:1.05rem;margin:1.9rem 0 .5rem;padding-top:.8rem;
 border-top:1px solid color-mix(in srgb,currentColor 18%,transparent)}
p.lede{opacity:.78;max-width:78ch;margin:.2rem 0 1rem}
.wrap{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}
svg{max-width:100%;height:auto;border-radius:6px}
table{border-collapse:collapse;font-size:.79rem}
th,td{border:1px solid color-mix(in srgb,currentColor 16%,transparent);padding:.24rem .5rem;text-align:left}
th{background:color-mix(in srgb,currentColor 7%,transparent)}
.full{width:100%}
.layers{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:.6rem 0}
.lay{border:1px solid color-mix(in srgb,currentColor 18%,transparent);border-radius:8px;padding:10px}
.lay h3{margin:0 0 .2rem;font-size:.9rem}
.lay .clk{font-size:.72rem;opacity:.7}
.card{display:flex;gap:13px;align-items:flex-start;padding:11px;border-radius:8px;margin-bottom:11px;
 border:1px solid color-mix(in srgb,currentColor 16%,transparent)}
.card img{image-rendering:pixelated;border-radius:4px;flex:0 0 auto;
 border:1px solid color-mix(in srgb,currentColor 22%,transparent)}
.card .body{min-width:0;flex:1}
.meta{font-size:.75rem;opacity:.76;margin:.12rem 0}
pre{margin:.3rem 0 0;padding:.4rem .55rem;border-radius:5px;overflow-x:auto;font-size:.71rem;
 white-space:pre-wrap;word-break:break-word;background:color-mix(in srgb,currentColor 8%,transparent)}
.tag{display:inline-block;font-size:.68rem;padding:.02rem .36rem;border-radius:3px;margin-right:.26rem;
 border:1px solid color-mix(in srgb,currentColor 30%,transparent)}
.bad{color:#e0483c;border-color:#e0483c}
.vlm{color:#a855f7;border-color:#a855f7}
.vla{color:#0ea5e9;border-color:#0ea5e9}
.seqt{color:#d19a00;border-color:#d19a00}
details summary{cursor:pointer;font-size:.76rem;opacity:.75}
code{font-size:.85em}
.rung{font-family:ui-monospace,Menlo,monospace;font-size:.74rem}
"""


def ladder_timing_chart(trace: dict, width: int = 940, row_h: int = 17,
                        max_scans: int = 1400) -> str:
    """PLC 屋が見慣れたタイムチャート (縦=信号 / 横=スキャン) を SVG で描く。

    ドラレコ用途。「なぜあの瞬間にコイルが立ったのか」を後から追えるように、
    入力接点と出力コイルを同じ時間軸に並べる。
    """
    sig = trace.get("signals") or {}
    if not sig:
        return "<p class='lede'>ラダートレースがありません</p>"
    n = trace["scans"]
    step = max(1, n // max_scans)
    inputs = trace.get("inputs", [])
    coils = trace.get("coils", [])
    order = [x for x in inputs if x in sig] + [x for x in coils if x in sig]
    label_w = 150
    plot_w = width - label_w - 10
    cols = len(range(0, n, step))
    px_per = plot_w / max(1, cols)
    H = row_h * len(order) + 26
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {H}" width="{width}" '
         f'height="{H}" font-family="ui-monospace,Menlo,monospace">',
         f'<rect width="{width}" height="{H}" fill="#0e0e14"/>']
    dt = trace.get("dt", 0.02)
    for k in range(0, cols, max(1, cols // 8)):
        x = label_w + k * px_per
        o.append(f'<line x1="{x:.1f}" y1="16" x2="{x:.1f}" y2="{H-6}" stroke="#25252f"/>')
        o.append(f'<text x="{x+2:.1f}" y="12" fill="#7a7a88" font-size="9">'
                 f'{k*step*dt:.0f}s</text>')
    for i, name in enumerate(order):
        y = 22 + i * row_h
        is_coil = name in coils
        col = "#7ee787" if is_coil else "#5ac8fa"
        o.append(f'<text x="4" y="{y+10}" fill="{col}" font-size="10">'
                 f'{"◆" if is_coil else "○"} {html.escape(name)}</text>')
        o.append(f'<line x1="{label_w}" y1="{y+row_h-2}" x2="{width-6}" y2="{y+row_h-2}" '
                 f'stroke="#1c1c25"/>')
        bits = sig[name]
        run_start = None
        for k, idx in enumerate(range(0, n, step)):
            on = bits[idx] == "1"
            if on and run_start is None:
                run_start = k
            elif not on and run_start is not None:
                x0 = label_w + run_start * px_per
                o.append(f'<rect x="{x0:.1f}" y="{y+2}" width="{max(0.6,(k-run_start)*px_per):.1f}" '
                         f'height="{row_h-6}" fill="{col}" opacity="0.85"/>')
                run_start = None
        if run_start is not None:
            x0 = label_w + run_start * px_per
            o.append(f'<rect x="{x0:.1f}" y="{y+2}" width="{max(0.6,(cols-run_start)*px_per):.1f}" '
                     f'height="{row_h-6}" fill="{col}" opacity="0.85"/>')
    o.append("</svg>")
    return "\n".join(o)


# --------------------------------------------------------------------------
# ラダー図 — 条件式ではなく、母線・接点・コイルをそのまま描く
# --------------------------------------------------------------------------

_WIRE = "#43435a"        # 非通電の配線
_HOT = "#7ee787"         # 通電している配線・接点・コイル
_HOTINK = "#ccffd8"      # 通電中のラベル


def _contact_on(name: str, bits: dict[str, bool] | None) -> bool | None:
    """接点が導通しているか。bits が無ければ None (= 状態を塗らない静的図)。

    `!` 接頭辞が b 接点。判定は Rung.evaluate と同じ規則にそろえてある。
    """
    if bits is None:
        return None
    return (not bits.get(name[1:], False)) if name.startswith("!") else bits.get(name, False)


def trace_bits_at(trace: dict, scan: int) -> dict[str, bool]:
    """タイムチャートのビット列から、あるスキャン時点の信号状態を切り出す。"""
    out: dict[str, bool] = {}
    for name, s in (trace.get("signals") or {}).items():
        if 0 <= scan < len(s):
            out[name] = s[scan] == "1"
    return out


def find_scan(trace: dict, signal: str, value: str = "1", start: int = 0) -> int | None:
    """信号が指定の値になる最初のスキャンを探す (見どころの自動選択用)。"""
    s = (trace.get("signals") or {}).get(signal)
    if not s:
        return None
    i = s.find(value, start)
    return None if i < 0 else i


def _contact(cx: float, cy: float, name: str, on: bool | None) -> list[str]:
    """a 接点 ─┤├─ / b 接点 ─┤/├─ を 1 個描く。"""
    nc = name.startswith("!")
    col = _WIRE if on is False else (_HOT if on else "#8f8fa6")
    ink = _HOTINK if on else DIM
    w = 2.0 if on else 1.4
    g = [f'<line x1="{cx-8:.1f}" y1="{cy-11}" x2="{cx-8:.1f}" y2="{cy+11}" '
         f'stroke="{col}" stroke-width="{w}"/>',
         f'<line x1="{cx+8:.1f}" y1="{cy-11}" x2="{cx+8:.1f}" y2="{cy+11}" '
         f'stroke="{col}" stroke-width="{w}"/>']
    if nc:                                   # b 接点は斜線を入れる
        g.append(f'<line x1="{cx-9:.1f}" y1="{cy+11}" x2="{cx+9:.1f}" y2="{cy-11}" '
                 f'stroke="{col}" stroke-width="{w}"/>')
    g.append(f'<text x="{cx:.1f}" y="{cy-16}" fill="{ink}" font-size="9.5" '
             f'text-anchor="middle">{html.escape(name.lstrip("!"))}</text>')
    if nc:
        g.append(f'<text x="{cx:.1f}" y="{cy+23}" fill="{DIM}" font-size="8" '
                 f'text-anchor="middle">b接点</text>')
    return g


def _coil(cx: float, cy: float, name: str, on: bool | None) -> list[str]:
    """出力コイル ─( )─ を 1 個描く。"""
    col = _WIRE if on is False else (_HOT if on else "#8f8fa6")
    ink = _HOTINK if on else INK
    w = 2.2 if on else 1.5
    g = [f'<path d="M {cx-10:.1f},{cy-12} A 15,15 0 0 0 {cx-10:.1f},{cy+12}" fill="none" '
         f'stroke="{col}" stroke-width="{w}"/>',
         f'<path d="M {cx+10:.1f},{cy-12} A 15,15 0 0 1 {cx+10:.1f},{cy+12}" fill="none" '
         f'stroke="{col}" stroke-width="{w}"/>']
    if on:                                   # 通電中は淡く光らせる
        g.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="7" fill="{_HOT}" opacity="0.22"/>')
    g.append(f'<text x="{cx:.1f}" y="{cy-18}" fill="{ink}" font-size="9.5" '
             f'text-anchor="middle">{html.escape(name)}</text>')
    return g


def ladder_diagram(rungs: Sequence, bits: dict[str, bool] | None = None,
                   width: int = 940) -> str:
    """ラダー図そのものを SVG で描く。

    `bits` を渡すと、そのスキャン時点で導通している接点・配線・コイルが緑に光る
    (実機 PLC のモニタ画面と同じ見え方)。`bits=None` なら状態なしの静的図。

    自己保持 (`seal`) は「コイル自身を並列枝に足す」という Rung.evaluate の
    実装そのままに、最下段の並列枝として描く。
    """
    BR_H, HEAD, PAD, CW = 40, 24, 16, 92
    rail_l, rail_r = 46, width - 40
    coil_x = rail_r - 62

    plans = []
    for r in rungs:
        brs = [list(b) for b in (r.branches or [])] or [[]]
        if r.seal:
            brs.append([r.coil])             # 自己保持 = コイルの a 接点を OR
        plans.append((r, brs))

    H = PAD + sum(HEAD + len(b) * BR_H + PAD for _, b in plans)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {H}" width="{width}" '
         f'height="{H}" font-family="ui-monospace,Menlo,monospace">',
         f'<rect width="{width}" height="{H}" fill="#0e0e14"/>',
         # 母線 (左=電源 / 右=帰線)
         f'<line x1="{rail_l}" y1="6" x2="{rail_l}" y2="{H-6}" stroke="#7ee787" '
         f'stroke-width="3" opacity="0.85"/>',
         f'<line x1="{rail_r}" y1="6" x2="{rail_r}" y2="{H-6}" stroke="#7ee787" '
         f'stroke-width="3" opacity="0.85"/>']

    y = PAD
    for r, brs in plans:
        n = len(brs)
        # ラング見出し
        o.append(f'<text x="{rail_l+4}" y="{y+12}" fill="{DIM}" font-size="10">'
                 f'{html.escape(r.name)}</text>')
        top = y + HEAD + BR_H / 2
        node_r = coil_x - 26
        hot_any = False
        for i, br in enumerate(brs):
            by = top + i * BR_H
            seal_row = r.seal and i == n - 1
            states = [_contact_on(c, bits) for c in br]
            # 枝の導通 = 直列 AND (接点ゼロなら素通し)
            hot = None if bits is None else all(s for s in states)
            hot_any = hot_any or bool(hot)
            wc = _HOT if hot else _WIRE
            ww = 2.0 if hot else 1.2
            xs = [rail_l + 52 + k * CW for k in range(len(br))]
            # 母線から最初の接点まで / 接点間 / 最後の接点から結合点まで
            pts = [rail_l] + [x for x in xs] + [node_r]
            for k in range(len(pts) - 1):
                x0 = pts[k] + (8 if 0 < k <= len(xs) else 0)
                x1 = pts[k + 1] - (8 if k + 1 <= len(xs) else 0)
                o.append(f'<line x1="{x0:.1f}" y1="{by}" x2="{x1:.1f}" y2="{by}" '
                         f'stroke="{wc}" stroke-width="{ww}"/>')
            for x, c, st in zip(xs, br, states):
                o.extend(_contact(x, by, c, st))
            if seal_row:
                o.append(f'<text x="{rail_l+52+len(br)*CW-40:.1f}" y="{by+4}" fill="{DIM}" '
                         f'font-size="8.5">← 自己保持</text>')
        if n > 1:                            # 並列枝を縦線で束ねる
            y0, y1 = top, top + (n - 1) * BR_H
            for x in (rail_l, node_r):
                o.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" '
                         f'stroke="{_HOT if hot_any else _WIRE}" '
                         f'stroke-width="{2.0 if hot_any else 1.2}"/>')
        # 結合点 → コイル → 右母線
        wc = _HOT if hot_any else _WIRE
        ww = 2.0 if hot_any else 1.2
        o.append(f'<line x1="{node_r}" y1="{top}" x2="{coil_x-10:.1f}" y2="{top}" '
                 f'stroke="{wc}" stroke-width="{ww}"/>')
        o.append(f'<line x1="{coil_x+10:.1f}" y1="{top}" x2="{rail_r}" y2="{top}" '
                 f'stroke="{wc}" stroke-width="{ww}"/>')
        o.extend(_coil(coil_x, top, r.coil, None if bits is None else hot_any))
        y += HEAD + n * BR_H + PAD

    o.append("</svg>")
    return "\n".join(o)


def layer_stack_diagram(cfg: dict, width: int = 940) -> str:
    """4 層が別々のクロックで回っていることを 1 枚で見せる。

    横軸を時間 (tick) にとって、各層が「いつ発火するか」を実寸で刻む。
    ラダーが埋まって見えるのに VLM がスカスカなのが、この構成の本質。
    """
    dt = float(cfg.get("dt", 0.02))
    span = 300                                # 描く tick 数
    rows = [("ladder", "ラダー", 1, "ゲート開閉・安全インタロック"),
            ("sequencer", "シーケンサ", 1, "故障検知 → 修理段取り → 復帰"),
            ("vla", "VLA", int(cfg.get("vla_every", 25) or 25), "パドル連続制御 (1 回で数十手)"),
            ("vlm", "VLM", int(cfg.get("vlm_every", 150) or 150), "狙う区画の助言 (命令権なし)")]
    lab_w, row_h, top = 128, 62, 26
    plot_w = width - lab_w - 20
    H = top + row_h * len(rows) + 10
    px = plot_w / span
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {H}" width="{width}" '
         f'height="{H}" font-family="ui-monospace,Menlo,monospace">',
         f'<rect width="{width}" height="{H}" fill="#0e0e14"/>',
         f'<text x="6" y="15" fill="{DIM}" font-size="10">制御層ごとの発火タイミング '
         f'(横軸 {span} tick = {span*dt:.0f} 秒)</text>']
    for k in range(0, span + 1, 50):          # 時間目盛
        x = lab_w + k * px
        o.append(f'<line x1="{x:.1f}" y1="{top-6}" x2="{x:.1f}" y2="{H-8}" stroke="#22222d"/>')
        o.append(f'<text x="{x+3:.1f}" y="{top-10}" fill="#6d6d7d" font-size="9">'
                 f'{k*dt:.0f}s</text>')
    for i, (key, name, every, role) in enumerate(rows):
        y = top + i * row_h
        c = LAYER_C[key]
        o.append(f'<text x="6" y="{y+18}" fill="{c}" font-size="11">{html.escape(name)}</text>')
        o.append(f'<text x="6" y="{y+32}" fill="{DIM}" font-size="8.5">'
                 f'{"毎 tick" if every <= 1 else f"{every} tick 毎"}</text>')
        o.append(f'<text x="6" y="{y+45}" fill="#5f5f70" font-size="8">'
                 f'{html.escape(role[:22])}</text>')
        o.append(f'<line x1="{lab_w}" y1="{y+26}" x2="{width-20}" y2="{y+26}" '
                 f'stroke="#1e1e28"/>')
        # 発火マーク。毎 tick の層は帯、間欠の層は縦棒 + 有効区間
        if every <= 1:
            o.append(f'<rect x="{lab_w}" y="{y+16}" width="{plot_w:.1f}" height="20" '
                     f'fill="{c}" opacity="0.5"/>')
            o.append(f'<text x="{lab_w+8}" y="{y+30}" fill="#0e0e14" font-size="9">'
                     f'決定的・毎スキャン評価</text>')
        else:
            for k in range(0, span, every):
                x = lab_w + k * px
                o.append(f'<rect x="{x:.1f}" y="{y+16}" width="{max(1.5,every*px-2):.1f}" '
                         f'height="20" fill="{c}" opacity="0.18"/>')
                o.append(f'<line x1="{x:.1f}" y1="{y+12}" x2="{x:.1f}" y2="{y+40}" '
                         f'stroke="{c}" stroke-width="2"/>')
            o.append(f'<text x="{lab_w+4}" y="{y+52}" fill="#5f5f70" font-size="8">'
                     f'縦棒＝指令が降りてくる瞬間。その間、下位は古い指令で走り続ける</text>')
    o.append("</svg>")
    return "\n".join(o)


def pinball_html(res: dict, svg: str,
                 title: str = "産業制御 × Physical AI — ピンボール／ブロック崩し") -> str:
    sc = res["score"]
    cfg = res.get("config", {})
    calls = res["calls"]
    dt = float(cfg.get("dt", 0.02))
    p: list[str] = [f"<title>{html.escape(title)}</title>", f"<style>{_CSS}</style>",
                    f"<h1>{html.escape(title)}</h1>"]
    p.append(
        "<p class='lede'>同じ盤面の上で <b>4 つの制御層が別々のクロックで</b> 動いています。"
        "下ほど速く決定的、上ほど遅く賢い。上位の判断は必ず<b>古い盤面に対する答え</b>として"
        "降りてくるので、下位はそれを前提に動きます。"
        "<b>ジグ（バンパー）に当てると対応する列が 1 個崩れます</b>（工程を 1 回こなす）。"
        "ただし叩くたびにジグ自身が摩耗し、<b>耐久度は隠れ状態</b>で打数しか観測できません。"
        "耐久が尽きると素通りになり<b>その工程はもう回せなくなる</b>ので、"
        "シーケンサが救済機を出して直します。使う道具が減っていく、という緊張が中心です。</p>")

    p.append("<div class='layers'>")
    for key, name, clk, role in (
        ("ladder", "ラダーロジック", "毎 tick (50Hz)", "ゲート開閉・安全インタロック。決定的で読み切れる"),
        ("sequencer", "逐次シーケンサ", "毎 tick", "故障検知→修理段取り→復帰。事故対応の権限はここ"),
        ("vla", "VLA (行動列)", f"{cfg.get('vla_every','?')} tick", "パドル 1 自由度の連続制御。1 回で数十手ぶん"),
        ("vlm", "VLM (言語)", f"{cfg.get('vlm_every','?')} tick", "狙う区画と修理可否の助言。命令権は持たない"),
    ):
        p.append(f"<div class='lay' style='border-left:4px solid {LAYER_C[key]}'>"
                 f"<h3>{name}</h3><div class='clk'>周期: {clk}</div>"
                 f"<div class='clk'>{role}</div></div>")
    p.append("</div>")

    p.append("<p class='lede'>カードで並べても伝わらないので、<b>実際の発火タイミングを実寸で</b>"
             "刻んだのが下の図です。ラダーは塗りつぶしに見えるほど回り、VLM は数えるほどしか"
             "発火しない。この密度差がそのまま「上位の判断が古い」ことの正体です。</p>")
    p.append("<div style='overflow-x:auto'>" + layer_stack_diagram(cfg) + "</div>")

    rows = [
        ("崩したブロック", f"{sc['blocks_broken']}/{sc['blocks_total']}"),
        ("列ごとの内訳", str(sc["per_col"])),
        ("均等性 (工程の消化)", f"{sc['evenness']}  (1.0 が理想)"),
        ("ジグ打数 / 均等性", f"{sc.get('jig_hits')} / {sc.get('jig_evenness')}"),
        ("摩耗して素通りになったジグ", sc.get("broken_jigs")),
        ("スコア", f"{sc['score']} / 1000"),
        ("パドル命中 / ロスト / 残機", f"{sc['paddle_hits']} / {sc['lost_balls']} / {sc['lives_left']}"),
        ("修理", f"成功 {res['repairs']['ok']} / 失敗 {res['repairs']['failed']}"),
        ("ラダースキャン", f"{res['ladder_scans']} 回"),
        ("シーケンサ遷移", f"{len(res['seq_events'])} 回"),
    ]
    p.append("<div class='wrap'><div>" + svg + "</div><div>")
    p.append("<table>" + "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows) + "</table>")

    truth = res.get("jig_truth", [])
    if truth:
        p.append("<table style='margin-top:.7rem'><tr><th>ジグ</th><th>観測できた打数</th>"
                 "<th>隠れ耐久度(答え)</th><th>状態</th></tr>")
        for t in truth:
            p.append(f"<tr><td>#{t['jid']}</td><td>{t['hits']}</td>"
                     f"<td>{t['max_durability']} 以下</td>"
                     f"<td>{'素通り' if t['broken'] else '健全'}</td></tr>")
        p.append("</table>")
        p.append("<p class='lede' style='font-size:.76rem'>打数は見えるが耐久度は見えない。"
                 "「そろそろ壊れる」は推測するしかない。</p>")
    p.append("</div></div>")

    # ラダー
    p.append("<h2>ラダーロジック (毎スキャン評価される)</h2>")
    p.append("<p class='lede'>上から順に評価され、上のラングが書いたコイルを下のラングが読めます。"
             "だから挙動が完全に読み切れる — ここが AI に置き換えられない理由です。</p>")
    rungs = build_ladder()
    p.append("<p class='lede'>左右の縦線が母線。<code>─┤├─</code> が a 接点、"
             "斜線入りの <code>─┤/├─</code> が b 接点 (否定)、右端の <code>─( )─</code> が"
             "出力コイルです。横に並ぶ接点が直列 (AND)、縦に積まれた枝が並列 (OR)。</p>")
    p.append("<div style='overflow-x:auto'>" + ladder_diagram(rungs) + "</div>")

    p.append("<table class='full'><tr><th>#</th><th>ラング</th><th>条件</th>"
             "<th>コイル</th><th>意味</th></tr>")
    for i, r in enumerate(rungs, 1):
        cond = " OR ".join("(" + " AND ".join(b) + ")" for b in r.branches)
        if r.seal:
            cond += f" OR <b>{r.coil}</b>(自己保持)"
        p.append(f"<tr><td>{i}</td><td>{html.escape(r.name)}</td>"
                 f"<td class='rung'>{cond}</td><td class='rung'>{html.escape(r.coil)}</td>"
                 f"<td>{html.escape(r.comment)}</td></tr>")
    p.append("</table>")

    # ラダーのタイムチャート (ドラレコ)
    tr = res.get("ladder_trace") or {}
    if tr.get("signals"):
        p.append("<h2>ラダーのタイムチャート (ドラレコ)</h2>")
        p.append(f"<p class='lede'>全 {tr['scans']} スキャンぶんのビット履歴。"
                 "<span style='color:#5ac8fa'>○ 青 = 入力接点</span>、"
                 "<span style='color:#7ee787'>◆ 緑 = 出力コイル</span>。"
                 "帯が立っている区間がその信号 ON。"
                 "「なぜあの瞬間に修理許可が出た/落ちたのか」を後から追えます。</p>")
        p.append("<div style='overflow-x:auto'>" + ladder_timing_chart(tr) + "</div>")

        # タイムチャートで見つけた「見どころ」のスキャンを、ラダー図に通電表示で焼き直す
        marks = []
        for sig_name, why in (("repair_permit", "修理許可が立った瞬間"),
                              ("drone_active", "救済機が飛んでいる最中"),
                              ("ball_in_danger", "ボールが危険域に入った瞬間")):
            s = find_scan(tr, sig_name)
            if s is not None:
                marks.append((s, sig_name, why))
        if marks:
            p.append("<h2>そのときラダーはどう通電していたか</h2>")
            p.append("<p class='lede'>タイムチャートは「いつ」しか分かりません。"
                     "同じ瞬間をラダー図に焼き直すと <b>どの経路を電気が通ったか</b> が見えます。"
                     "<span style='color:#7ee787'>緑＝通電</span>、"
                     "暗い線＝遮断。b 接点が効いて落ちている経路に注目してください。</p>")
            for s, sig_name, why in marks:
                p.append(f"<h3 style='font-size:.86rem;margin:.9rem 0 .2rem'>"
                         f"スキャン {s} ({s*dt:.1f}s) — {html.escape(why)} "
                         f"<code style='opacity:.7'>{html.escape(sig_name)}</code></h3>")
                p.append("<div style='overflow-x:auto'>"
                         + ladder_diagram(rungs, trace_bits_at(tr, s)) + "</div>")

    # シーケンサ
    p.append("<h2>シーケンサの遷移 (事故対応)</h2>")
    p.append("<p class='lede'>VLM は「今なら修理してよい」と<b>助言</b>するだけで、"
             "実際に救済機を出す/止めるの判断は必ずこのステートマシンを通ります。"
             "AI 連動でも安全側の権限は組み込みに残す、という形です。</p>")
    p.append("<table class='full'><tr><th>t</th><th>遷移</th><th>理由</th></tr>")
    dt = float(cfg.get("dt", 0.02))
    for e in res["seq_events"]:
        p.append(f"<tr><td>{e['tick']*dt:.1f}s</td>"
                 f"<td><span class='tag seqt'>{html.escape(e['frm'])} → "
                 f"{html.escape(e['to'])}</span></td><td>{html.escape(e['reason'])}</td></tr>")
    p.append("</table>")

    # 呼び出し
    p.append(f"<h2>AI 層の入出力と依頼タイミング ({len(calls)} 件)</h2>")
    p.append("<p class='lede'>VLM は数秒に 1 回しか呼べません。その助言が届く頃には"
             "ボールは既に動いています。だから VLA は「区画」という<b>粗い目標</b>だけを受け取り、"
             "細かい追従は自分の行動列で埋めます。この粒度の切り分けが実運用の勘所です。</p>")
    for c in calls:
        cls = c["layer"]
        img = (f"<img src='data:image/png;base64,{c['png_b64']}' width='120' alt='入力'>"
               if c.get("png_b64") else "")
        tags = [f"<span class='tag {cls}'>[{c['seq']}] {LAYER_JA.get(cls, cls)}</span>",
                f"<span class='tag'>t={c['t']}s</span>",
                f"<span class='tag'>{c['latency_s']:.2f}s</span>"]
        if c.get("error"):
            tags.append(f"<span class='tag bad'>{html.escape(str(c['error']))}</span>")
        p.append(
            f"<div class='card' style='border-left:4px solid {LAYER_C.get(cls,'#888')}'>{img}"
            f"<div class='body'><div class='meta'>{''.join(tags)}</div>"
            f"<div class='meta'>入力: <code>"
            f"{html.escape(json.dumps(c.get('inputs',{}), ensure_ascii=False)[:340])}</code></div>"
            f"<div class='meta'>出力: <code>"
            f"{html.escape(json.dumps(c.get('output',{}), ensure_ascii=False)[:240])}</code></div>"
            f"<div class='meta'>{html.escape(c.get('note',''))}</div>"
            f"<pre>{html.escape(str(c.get('raw') or '')[:400])}</pre>"
            + (f"<details><summary>プロンプト全文</summary>"
               f"<pre>{html.escape(str(c.get('prompt')))}</pre></details>"
               if c.get("prompt") else "")
            + "</div></div>")
    return "\n".join(p)
