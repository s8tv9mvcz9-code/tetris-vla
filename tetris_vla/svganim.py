"""tetris_vla.svganim — 降下の記録を、自己完結の SVG アニメと HTML に起こす。

SVG は SMIL (`<animate>` / `<animateTransform>`) だけで動く。スクリプトも外部参照も
使わないので、GitHub のファイルビューでもそのまま再生される。

HTML のほうは「何を見せて、何を答えたか」を並べる:
  * 判断のたびに **モデルへ実際に渡した PNG** をそのまま貼る
  * その時のセンサ値、生の応答文字列、パース結果、レイテンシを添える
"""

from __future__ import annotations

import base64
import html
import json
from typing import Sequence

import numpy as np

from .render import BASE, SETTLED

BG = "#121218"
GRID = "#2a2a33"
INK = "#e8e8ef"
DIM = "#9a9aa8"
ACCENT = "#5ac8fa"
FLAME_C = "#ffaa3c"
TRAIL_C = "#78aad2"
TARGET_C = "#5ad07a"

ACTION_LABEL = {0: "噴射なし", 1: "左へ噴射", 2: "右へ噴射", 3: "ベント(加速降下)"}
ACTION_SHORT = {0: "—", 1: "◀", 2: "▶", 3: "▼"}


def _rgb(t: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % t


def _fmt(vals: Sequence[float], nd: int = 2) -> str:
    return ";".join(f"{v:.{nd}f}" for v in vals)


def flight_svg(data: dict, cell: int = 16, speed: float = 1.0,
               panel_w: int = 300, loop: bool = True) -> str:
    """飛行記録 1 本を SVG アニメにする。

    speed は再生倍率。1.0 なら論理時間どおりの速さで再生される。
    """
    board: np.ndarray = data["board"]
    frames = data["frames"]
    decisions = data["decisions"]
    end = data.get("end", {})
    cfg = data.get("config", {})
    H, W = board.shape
    if not frames:
        raise ValueError("フレームが空です")

    dt = float(cfg.get("dt", 0.12))
    dur = max(0.5, len(frames) * dt / max(0.05, speed))
    head = 26                      # 上部の情報帯
    bw, bh = W * cell, H * cell
    width, height = bw + panel_w + 24, head + bh + 8
    repeat = "indefinite" if loop else "1"

    n = len(frames)
    kt = _fmt([i / max(1, n - 1) for i in range(n)], 4)

    o: list[str] = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
      f'width="{width}" height="{height}" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">')
    a(f'<rect width="{width}" height="{height}" fill="{BG}"/>')

    # --- 見出し -----------------------------------------------------------
    pilot = html.escape(str(data.get("start", {}).get("pilot", "?")))
    kind = html.escape(str(data.get("start", {}).get("kind", "?")))
    score = end.get("score", 0)
    a(f'<text x="8" y="17" fill="{INK}" font-size="12">{pilot} / {kind} ピース '
      f'/ {score:.0f}点 / {end.get("sim_seconds", 0)}秒 / 判断{end.get("decisions", 0)}回'
      f' / 平均{end.get("mean_latency_s", 0):.2f}s</text>')

    # --- 地形 -------------------------------------------------------------
    a(f'<g transform="translate(0,{head})">')
    a(f'<rect width="{bw}" height="{bh}" fill="#0e0e13"/>')
    for r in range(H + 1):
        a(f'<line x1="0" y1="{r*cell}" x2="{bw}" y2="{r*cell}" stroke="{GRID}" stroke-width="1"/>')
    for c in range(W + 1):
        a(f'<line x1="{c*cell}" y1="0" x2="{c*cell}" y2="{bh}" stroke="{GRID}" stroke-width="1"/>')
    for r in range(H):
        for c in range(W):
            v = int(board[r, c])
            if v:
                a(f'<rect x="{c*cell+1}" y="{r*cell+1}" width="{cell-2}" height="{cell-2}" '
                  f'fill="{_rgb(SETTLED[v])}"/>')

    # 目標列 (Dellacherie 参照解) を足元に印として出す
    tgt = end.get("oracle_col")
    if tgt is not None:
        a(f'<rect x="{tgt*cell}" y="{bh-3}" width="{cell}" height="3" fill="{TARGET_C}"/>')
        a(f'<text x="{tgt*cell+cell/2}" y="{bh-6}" fill="{TARGET_C}" font-size="9" '
          f'text-anchor="middle">目標</text>')

    # --- 軌跡 (静的に全経路を描く) ----------------------------------------
    pts = " ".join(f"{(f['x']+0.5)*cell:.1f},{(f['y']+0.5)*cell:.1f}" for f in frames)
    a(f'<polyline points="{pts}" fill="none" stroke="{TRAIL_C}" stroke-width="1" '
      f'stroke-opacity="0.45" stroke-dasharray="2 2"/>')

    # --- 機体 -------------------------------------------------------------
    kind_id = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}.get(kind, 2)
    from .engine import ROT_CELLS, cell_span

    rot = int(data.get("start", {}).get("rot", 0))
    cells = ROT_CELLS[kind][rot % 4] if kind in ROT_CELLS else ((0, 0),)
    _, _, min_dx, max_dx = cell_span(kind, rot) if kind in ROT_CELLS else (0, 0, 0, 0)
    cx = (min_dx + max_dx + 1) / 2.0

    xs = [f["x"] * cell for f in frames]
    ys = [f["y"] * cell for f in frames]
    a('<g>')
    a(f'<animateTransform attributeName="transform" type="translate" '
      f'values="{";".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))}" '
      f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="{repeat}" calcMode="linear"/>')
    # パラシュート
    a(f'<path d="M {cx*cell-cell*1.4} {-cell*0.5} A {cell*1.4} {cell*1.1} 0 0 1 '
      f'{cx*cell+cell*1.4} {-cell*0.5}" fill="none" stroke="#dcdce6" stroke-width="2"/>')
    for sgn in (-1, 0, 1):
        a(f'<line x1="{cx*cell+sgn*cell*1.25}" y1="{-cell*0.45}" x2="{cx*cell}" y2="{cell*0.2}" '
          f'stroke="#8a8a95" stroke-width="0.8"/>')
    for dy, dx in cells:
        a(f'<rect x="{dx*cell+1}" y="{dy*cell+1}" width="{cell-2}" height="{cell-2}" '
          f'fill="{_rgb(BASE[kind_id])}" rx="1"/>')
    # 噴射炎: 行動に応じて明滅させる (噴射方向と反対側に出る)
    acts = [int(f["action"]) for f in frames]
    for act, pts_d in (
        (2, f"M {cx*cell-cell*0.5} {cell*0.1} L {cx*cell-cell*0.5} {cell*0.9} "
            f"L {cx*cell-cell*1.5} {cell*0.5} Z"),
        (1, f"M {cx*cell+cell*0.5} {cell*0.1} L {cx*cell+cell*0.5} {cell*0.9} "
            f"L {cx*cell+cell*1.5} {cell*0.5} Z"),
        (3, f"M {cx*cell-cell*0.35} {-cell*0.1} L {cx*cell+cell*0.35} {-cell*0.1} "
            f"L {cx*cell} {-cell*1.0} Z"),
    ):
        vals = _fmt([1.0 if x == act else 0.0 for x in acts], 0)
        a(f'<path d="{pts_d}" fill="{FLAME_C}" opacity="0">'
          f'<animate attributeName="opacity" values="{vals}" keyTimes="{kt}" '
          f'dur="{dur:.2f}s" repeatCount="{repeat}" calcMode="discrete"/></path>')
    a('</g>')
    a('</g>')

    # --- 右パネル: センサ + 判断ログ --------------------------------------
    px = bw + 16
    a(f'<g transform="translate({px},{head})">')
    a(f'<text x="0" y="10" fill="{DIM}" font-size="10">センサ (機体が知っている値)</text>')

    fuel_max = float(cfg.get("fuel", 60)) or 1.0
    a(f'<rect x="0" y="18" width="{panel_w-32}" height="6" fill="#22222c" rx="3"/>')
    a(f'<rect x="0" y="18" width="{panel_w-32}" height="6" fill="#5ad07a" rx="3">'
      f'<animate attributeName="width" values="'
      f'{_fmt([(panel_w-32)*f["fuel"]/fuel_max for f in frames], 1)}" '
      f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="{repeat}"/></rect>')
    a(f'<text x="{panel_w-28}" y="24" fill="{DIM}" font-size="9">燃料</text>')

    # 速度バー (中央から左右に伸びる)
    mid = (panel_w - 32) / 2
    a(f'<line x1="{mid}" y1="34" x2="{mid}" y2="46" stroke="{GRID}"/>')
    vscale = 22.0
    a(f'<rect y="37" height="6" fill="{ACCENT}" rx="2">'
      f'<animate attributeName="x" values="'
      f'{_fmt([mid + min(0, f["vx"]) * vscale for f in frames], 1)}" '
      f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="{repeat}"/>'
      f'<animate attributeName="width" values="'
      f'{_fmt([max(0.5, abs(f["vx"]) * vscale) for f in frames], 1)}" '
      f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="{repeat}"/></rect>')
    a(f'<text x="0" y="56" fill="{DIM}" font-size="9">← 横速度 vx →</text>')

    # 判断ログ
    a(f'<text x="0" y="76" fill="{DIM}" font-size="10">判断 ({len(decisions)} 回)</text>')
    row_h = 30
    top0 = 84
    for i, d in enumerate(decisions):
        y = top0 + i * row_h
        if y > bh - 20:
            a(f'<text x="0" y="{y}" fill="{DIM}" font-size="9">… 他 {len(decisions)-i} 件</text>')
            break
        act = int(d["action"])
        st = d.get("state", {})
        why = html.escape(str(d.get("why", ""))[:26])
        err = d.get("error")
        col = "#ff6b6b" if err else INK
        a(f'<g opacity="0.32"><set attributeName="opacity" to="1" '
          f'begin="{d["deliver_tick"]*dt/max(0.05,speed):.2f}s" '
          f'dur="{max(0.4, row_h*0.0+2.2):.2f}s" repeatCount="{repeat}"/>')
        a(f'<text x="0" y="{y}" fill="{col}" font-size="10">'
          f'{ACTION_SHORT.get(act,"?")} {html.escape(ACTION_LABEL.get(act,"?"))}'
          f'{"  目標列" + str(d["target_col"]) if d.get("target_col") is not None else ""}</text>')
        a(f'<text x="0" y="{y+11}" fill="{DIM}" font-size="8.5">'
          f't={d["submit_tick"]*dt:.1f}s 列{st.get("col","?")} vx{st.get("vx",0):+.1f} '
          f'燃料{st.get("fuel","?")} lat{d.get("latency_s",0):.1f}s</text>')
        a(f'<text x="0" y="{y+21}" fill="{DIM}" font-size="8.5">'
          f'{"⚠ " + html.escape(str(err)) if err else why}</text>')
        a('</g>')
    a('</g>')
    a('</svg>')
    return "\n".join(o)


# --------------------------------------------------------------------------
# HTML: 与えた入力と、返ってきた判断を並べる
# --------------------------------------------------------------------------

_CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;padding:24px;line-height:1.7;
 font-family:ui-sans-serif,system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
h1{font-size:1.45rem;margin:0 0 .2rem}
h2{font-size:1.1rem;margin:2.2rem 0 .6rem;padding-top:.9rem;
 border-top:1px solid color-mix(in srgb,currentColor 18%,transparent)}
p.lede{opacity:.75;max-width:74ch;margin:.2rem 0 1.2rem}
.anim{overflow-x:auto;padding:8px 0}
.anim svg{max-width:100%;height:auto}
.cards{display:flex;flex-direction:column;gap:14px}
.card{display:flex;gap:14px;align-items:flex-start;padding:12px;border-radius:8px;
 border:1px solid color-mix(in srgb,currentColor 16%,transparent)}
.card.err{border-color:#e0483c}
.card img{image-rendering:pixelated;border-radius:4px;flex:0 0 auto;
 border:1px solid color-mix(in srgb,currentColor 22%,transparent)}
.card .body{min-width:0;flex:1}
.act{font-weight:600;font-size:1rem;margin:0 0 .2rem}
.meta{font-size:.78rem;opacity:.72;margin:.1rem 0}
pre{margin:.4rem 0 0;padding:.5rem .6rem;border-radius:5px;overflow-x:auto;font-size:.75rem;
 background:color-mix(in srgb,currentColor 8%,transparent)}
table{border-collapse:collapse;font-size:.82rem;width:100%;margin-top:.6rem}
th,td{border:1px solid color-mix(in srgb,currentColor 16%,transparent);padding:.28rem .5rem;text-align:left}
th{background:color-mix(in srgb,currentColor 7%,transparent)}
.tag{display:inline-block;font-size:.7rem;padding:.05rem .4rem;border-radius:3px;margin-right:.3rem;
 border:1px solid color-mix(in srgb,currentColor 30%,transparent)}
.bad{color:#e0483c;border-color:#e0483c}
details summary{cursor:pointer;font-size:.8rem;opacity:.75}
"""


def flight_html(data: dict, svg: str, title: str = "パラシュート降下 — 入力と判断") -> str:
    end = data.get("end", {})
    start = data.get("start", {})
    cfg = data.get("config", {})
    decisions = data.get("decisions", [])
    dt = float(cfg.get("dt", 0.12))

    p: list[str] = [f"<title>{html.escape(title)}</title>", f"<style>{_CSS}</style>",
                    f"<h1>{html.escape(title)}</h1>"]
    p.append(
        f"<p class='lede'>操縦者 <b>{html.escape(str(start.get('pilot','?')))}</b> が "
        f"{html.escape(str(start.get('kind','?')))} ピースをパラシュートで降ろした記録です。"
        f"落下は止められず、横方向だけ推進ジェットで操れます (燃料 {cfg.get('fuel','?')})。"
        f"下のカードは <b>実際にモデルへ渡した画像そのもの</b> と、返ってきた生の応答です。</p>"
    )

    rows = [
        ("得点", f"{end.get('score', 0):.0f} / 1000"),
        ("内訳", " ".join(f"{k}={v}" for k, v in (end.get("breakdown") or {}).items())),
        ("着地列 / 参照解", f"{end.get('landing_col')} / {end.get('oracle_col')} "
                            f"(誤差 {end.get('col_error')})"),
        ("配置 regret", end.get("regret")),
        ("接地速度", f"下 {end.get('impact_vy')} / 横 {end.get('impact_vx')} "
                     f"({'軟着陸' if end.get('soft') else '硬着陸'})"),
        ("作った穴", end.get("holes_created")),
        ("燃料 使用 / 残", f"{end.get('fuel_used')} / {end.get('fuel_left')}"),
        ("滞空 / 判断回数", f"{end.get('sim_seconds')}秒 / {end.get('decisions')}回"),
        ("平均レイテンシ", f"{end.get('mean_latency_s', 0):.2f}s "
                           f"(= {end.get('mean_latency_s', 0)/dt:.0f} tick 分の漂流)"),
        ("エラー内訳", end.get("errors") or "なし"),
    ]
    p.append("<table>" + "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    ) + "</table>")

    p.append("<h2>降下アニメーション</h2>")
    p.append("<p class='lede'>破線が実際の軌跡、緑の印が参照解の着地列。"
             "右のログは、その時刻に届いた判断が光ります。</p>")
    p.append(f"<div class='anim'>{svg}</div>")

    p.append(f"<h2>与えた入力と、返ってきた判断 ({len(decisions)} 回)</h2>")
    p.append("<p class='lede'>画像は縦 20 行 × 横 10 列の盤面を 12px/セルで描いたもの (実寸)。"
             "モデルはこの PNG と、下に並ぶセンサ値だけを受け取ります。</p>")
    p.append("<div class='cards'>")
    for i, d in enumerate(decisions):
        st = d.get("state", {})
        err = d.get("error")
        img_tag = ""
        if d.get("png"):
            uri = "data:image/png;base64," + base64.b64encode(d["png"]).decode("ascii")
            img_tag = f"<img src='{uri}' width='120' alt='判断{i+1}の入力'>"
        tags = [f"<span class='tag'>#{i+1}</span>",
                f"<span class='tag'>t={d['submit_tick']*dt:.1f}s</span>",
                f"<span class='tag'>レイテンシ {d.get('latency_s',0):.2f}s</span>"]
        drift = d["deliver_tick"] - d["submit_tick"]
        if drift:
            tags.append(f"<span class='tag'>この間に {drift} tick 漂流</span>")
        if err:
            tags.append(f"<span class='tag bad'>{html.escape(str(err))}</span>")
        raw = html.escape(str(d.get("raw") or "(バックエンドが生応答を返さない種類)"))
        prompt = html.escape(str(d.get("prompt") or ""))
        p.append(
            f"<div class='card{' err' if err else ''}'>{img_tag}<div class='body'>"
            f"<p class='act'>{html.escape(ACTION_LABEL.get(int(d['action']),'?'))}"
            + (f" → 目標列 {d['target_col']}" if d.get("target_col") is not None else "")
            + f"</p><div class='meta'>{''.join(tags)}</div>"
            f"<div class='meta'>センサ: 左端列 {st.get('col','?')} / 高さ行 {st.get('y','?')} / "
            f"横速度 {st.get('vx','?')} / 落下速度 {st.get('vy','?')} / 燃料 {st.get('fuel','?')}</div>"
            f"<pre>{raw}</pre>"
            + (f"<details><summary>渡したプロンプト全文</summary><pre>{prompt}</pre></details>"
               if prompt else "")
            + "</div></div>"
        )
    p.append("</div>")
    return "\n".join(p)
