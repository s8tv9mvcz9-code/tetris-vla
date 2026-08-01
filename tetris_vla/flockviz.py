"""tetris_vla.flockviz — 複数機デモを「2 つのモデルの往復」として見せる HTML。

見せたいのは軌道そのものより **どちらのモデルが / いつ呼ばれ / 何を見せられ /
何を返し / その間に世界がどれだけ進んだか**。したがって画面は 3 層:

  1. SVG アニメ  … 全機の降下。VLM が指令を出した瞬間に目標マーカーが切り替わる。
  2. タイムライン … 上段=VLM、下段=VLA の帯。幅がそのまま推論時間 (= 漂流) になる。
  3. 呼び出しカード … 実際に渡した PNG、画像以外の入力、生応答、解釈、コスト。

外部参照ゼロの自己完結 HTML。
"""

from __future__ import annotations

import html
import json
from typing import Sequence

import numpy as np

from .render import BASE, SETTLED

BG = "#121218"
GRID = "#2a2a33"
INK = "#e8e8ef"
DIM = "#9a9aa8"
VLM_C = "#c98bff"
VLA_C = "#5ac8fa"
TARGET_C = "#5ad07a"
FLAME_C = "#ffaa3c"
TRAIL_C = "#78aad2"
PIECE_TINT = ["#ff6b6b", "#5ac8fa", "#ffd24d", "#8ef08a", "#d79bff"]
ACTION_MARK = {0: "—", 1: "◀", 2: "▶", 3: "▼"}
ACTION_JA = {0: "噴射なし", 1: "左へ噴射", 2: "右へ噴射", 3: "ベント"}


def _rgb(t) -> str:
    return "#%02x%02x%02x" % tuple(t)


def _fmt(vals, nd=1) -> str:
    return ";".join(f"{v:.{nd}f}" for v in vals)


def flock_svg(data: dict, cell: int = 18, speed: float = 1.0) -> str:
    """全機の降下を SMIL で動かす。VLM の指令切替も目標マーカーの移動として出る。"""
    board = data["board"]
    frames = data["frames"]
    calls = data["calls"]
    end = data.get("end", {})
    cfg = data.get("config", {})
    H, W = board.shape
    dt = float(cfg.get("dt", 0.12))
    n = len(frames)
    if n == 0:
        raise ValueError("フレームがありません")
    dur = max(1.0, n * dt / max(0.05, speed))
    kt = _fmt([i / max(1, n - 1) for i in range(n)], 4)
    head, foot = 22, 26
    bw, bh = W * cell, H * cell
    o: list[str] = []
    a = o.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {bw} {head+bh+foot}" '
      f'width="{bw}" height="{head+bh+foot}" '
      f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace">')
    a(f'<rect width="{bw}" height="{head+bh+foot}" fill="{BG}"/>')
    a(f'<text x="6" y="15" fill="{DIM}" font-size="11">'
      f'{html.escape(str(end.get("landed","?")))}/{end.get("n_landers","?")} 機着地 · '
      f'{end.get("sim_seconds","?")}論理秒 · スローモーション×{end.get("slowmo","?")}</text>')

    a(f'<g transform="translate(0,{head})">')
    a(f'<rect width="{bw}" height="{bh}" fill="#0e0e13"/>')
    for r in range(H + 1):
        a(f'<line x1="0" y1="{r*cell}" x2="{bw}" y2="{r*cell}" stroke="{GRID}"/>')
    for c in range(W + 1):
        a(f'<line x1="{c*cell}" y1="0" x2="{c*cell}" y2="{bh}" stroke="{GRID}"/>')
    for r in range(H):
        for c in range(W):
            v = int(board[r, c])
            if v:
                a(f'<rect x="{c*cell+1}" y="{r*cell+1}" width="{cell-2}" height="{cell-2}" '
                  f'fill="{_rgb(SETTLED[v])}"/>')

    # --- 各機 --------------------------------------------------------------
    from .engine import ROT_CELLS, KIND_ID, cell_span

    shapes = {pid: (kind, rot) for pid, kind, rot in data["start"]["spawn_order"]}
    per = (end.get("per_lander") or {})
    for i, (pid, (kind, rot)) in enumerate(sorted(shapes.items())):
        tint = PIECE_TINT[i % len(PIECE_TINT)]
        xs, ys, acts, vis = [], [], [], []
        last = (0.0, 0.0)
        for f in frames:
            st = f["landers"].get(pid)
            if st is None:
                xs.append(last[0]); ys.append(last[1]); acts.append(0); vis.append(0)
            else:
                xs.append(st[0] * cell); ys.append(st[1] * cell)
                acts.append(int(st[5])); vis.append(1)
                last = (st[0] * cell, st[1] * cell)
        # 軌跡
        pts = " ".join(f"{x+cell/2:.1f},{y+cell/2:.1f}"
                       for x, y, v in zip(xs, ys, vis) if v)
        if pts:
            a(f'<polyline points="{pts}" fill="none" stroke="{tint}" stroke-width="1" '
              f'stroke-opacity="0.35" stroke-dasharray="2 3"/>')
        a(f'<g opacity="0"><animate attributeName="opacity" values="{_fmt(vis,0)}" '
          f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/>')
        a(f'<animateTransform attributeName="transform" type="translate" '
          f'values="{";".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))}" '
          f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite"/>')
        _, _, min_dx, max_dx = cell_span(kind, rot)
        cx = (min_dx + max_dx + 1) / 2.0
        a(f'<path d="M {cx*cell-cell*1.2} {-cell*0.5} A {cell*1.2} {cell} 0 0 1 '
          f'{cx*cell+cell*1.2} {-cell*0.5}" fill="none" stroke="#dcdce6" stroke-width="1.6"/>')
        a(f'<line x1="{cx*cell}" y1="{-cell*0.45}" x2="{cx*cell}" y2="{cell*0.2}" '
          f'stroke="#8a8a95" stroke-width="0.8"/>')
        for dy, dx in ROT_CELLS[kind][rot % 4]:
            a(f'<rect x="{dx*cell+1}" y="{dy*cell+1}" width="{cell-2}" height="{cell-2}" '
              f'fill="{_rgb(BASE[KIND_ID[kind]])}" stroke="{tint}" stroke-width="1.5" rx="1"/>')
        a(f'<text x="{cx*cell}" y="{-cell*0.8}" fill="{tint}" font-size="10" '
          f'text-anchor="middle">#{pid}</text>')
        for act, d_ in (
            (2, f"M {cx*cell-cell*.5} {cell*.1} L {cx*cell-cell*.5} {cell*.9} "
                f"L {cx*cell-cell*1.4} {cell*.5} Z"),
            (1, f"M {cx*cell+cell*.5} {cell*.1} L {cx*cell+cell*.5} {cell*.9} "
                f"L {cx*cell+cell*1.4} {cell*.5} Z"),
            (3, f"M {cx*cell-cell*.3} {-cell*.1} L {cx*cell+cell*.3} {-cell*.1} "
                f"L {cx*cell} {-cell*.9} Z"),
        ):
            a(f'<path d="{d_}" fill="{FLAME_C}" opacity="0"><animate attributeName="opacity" '
              f'values="{_fmt([1.0 if x == act else 0.0 for x in acts],0)}" keyTimes="{kt}" '
              f'dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></path>')
        a('</g>')

    # --- VLM の指令 (目標列マーカー) --------------------------------------
    hist = []
    for c in calls:
        if c["role"] == "どこへ" and c.get("parsed", {}).get("assign"):
            hist.append((c["tick"], {int(k): v for k, v in c["parsed"]["assign"].items()}))
    for i, (pid, _) in enumerate(sorted(shapes.items())):
        tint = PIECE_TINT[i % len(PIECE_TINT)]
        xs, op = [], []
        cur = None
        hi = 0
        for f in frames:
            while hi < len(hist) and hist[hi][0] <= f["tick"]:
                cur = hist[hi][1].get(pid, cur)
                hi += 1
            xs.append((cur or 0) * cell)
            op.append(1 if cur is not None else 0)
        a(f'<g opacity="0"><animate attributeName="opacity" values="{_fmt(op,0)}" '
          f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/>')
        a(f'<rect y="{bh-4-i*3}" width="{cell}" height="2.5" fill="{tint}">'
          f'<animate attributeName="x" values="{_fmt(xs)}" keyTimes="{kt}" '
          f'dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></rect>')
        a('</g>')
    a('</g>')

    # --- 下部: 推論帯 ------------------------------------------------------
    a(f'<g transform="translate(0,{head+bh+4})">')
    a(f'<text x="2" y="8" fill="{DIM}" font-size="8">推論帯 上=VLM 下=VLA (幅=漂流tick)</text>')
    total_ticks = max(1, frames[-1]["tick"])
    for c in calls:
        x0 = c["tick"] / total_ticks * bw
        w = max(2.0, c["drift_ticks"] / total_ticks * bw)
        y = 11 if c["role"] == "どこへ" else 17
        col = VLM_C if c["role"] == "どこへ" else VLA_C
        a(f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="5" fill="{col}" '
          f'opacity="0.85" rx="1"><title>[{c["seq"]}] {c["model"]} '
          f'{c["latency_s"]:.1f}s</title></rect>')
    a('</g></svg>')
    return "\n".join(o)


# --------------------------------------------------------------------------

_CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;padding:24px;line-height:1.7;
 font-family:ui-sans-serif,system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
h1{font-size:1.4rem;margin:0 0 .2rem}
h2{font-size:1.05rem;margin:2rem 0 .5rem;padding-top:.8rem;
 border-top:1px solid color-mix(in srgb,currentColor 18%,transparent)}
p.lede{opacity:.75;max-width:76ch;margin:.2rem 0 1rem}
.wrap{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}
.anim svg{max-width:100%;height:auto;border-radius:6px}
table{border-collapse:collapse;font-size:.8rem}
th,td{border:1px solid color-mix(in srgb,currentColor 16%,transparent);padding:.25rem .5rem;text-align:left}
th{background:color-mix(in srgb,currentColor 7%,transparent)}
.tl{width:100%;overflow-x:auto;margin:.6rem 0 1rem}
.tl table{width:100%;font-size:.75rem}
.card{display:flex;gap:14px;align-items:flex-start;padding:12px;border-radius:8px;margin-bottom:12px;
 border:1px solid color-mix(in srgb,currentColor 16%,transparent)}
.card.vlm{border-left:4px solid #c98bff}
.card.vla{border-left:4px solid #5ac8fa}
.card.err{border-color:#e0483c}
.card img{image-rendering:pixelated;border-radius:4px;flex:0 0 auto;
 border:1px solid color-mix(in srgb,currentColor 22%,transparent)}
.card .body{min-width:0;flex:1}
.hd{font-weight:600;margin:0 0 .25rem}
.meta{font-size:.76rem;opacity:.75;margin:.15rem 0}
pre{margin:.35rem 0 0;padding:.45rem .6rem;border-radius:5px;overflow-x:auto;font-size:.72rem;
 white-space:pre-wrap;word-break:break-word;
 background:color-mix(in srgb,currentColor 8%,transparent)}
.tag{display:inline-block;font-size:.68rem;padding:.03rem .38rem;border-radius:3px;margin-right:.28rem;
 border:1px solid color-mix(in srgb,currentColor 30%,transparent)}
.bad{color:#e0483c;border-color:#e0483c}
.vlmt{color:#a855f7;border-color:#a855f7}
.vlat{color:#0ea5e9;border-color:#0ea5e9}
details summary{cursor:pointer;font-size:.76rem;opacity:.75}
.plan{font-family:ui-monospace,Menlo,monospace;letter-spacing:.15em;font-size:.9rem}
"""


def flock_html(data: dict, svg: str, title: str = "qwen-VL × SmolVLA — 複数機同時降下") -> str:
    end = data.get("end", {})
    start = data.get("start", {})
    cfg = data.get("config", {})
    fk = data.get("flock", {})
    calls = data.get("calls", [])
    dt = float(cfg.get("dt", 0.12))
    p: list[str] = [f"<title>{html.escape(title)}</title>", f"<style>{_CSS}</style>",
                    f"<h1>{html.escape(title)}</h1>"]
    p.append(
        "<p class='lede'>上位 <b>qwen2.5vl:3b</b> が全機ぶんの<b>着地列</b>を 1 リクエストで決め、"
        "下位 <b>SmolVLA</b> が指令された列へ<b>推進ジェットを操って</b>寄せます。"
        "PD 制御も参照解も操縦には使っていません（参照解は採点にだけ使用）。"
        f"世界はスローモーション×{fk.get('slowmo','?')}で進むので、"
        "実推論 1 秒あたりの漂流はその分の 1 になります。</p>")

    rows = [
        ("着地", f"{end.get('landed')}/{end.get('n_landers')}  "
                 f"(軟着陸 {end.get('soft_landings')})"),
        ("作った穴", end.get("holes_created")),
        ("平均 regret / 列誤差", f"{end.get('mean_regret')} / {end.get('mean_col_error')}"),
        ("VLM 呼び出し", f"{end.get('vlm_calls')} 回 · 平均 {end.get('vlm_latency_mean')}s"),
        ("VLA 呼び出し", f"{end.get('vla_calls')} 回 · 平均 {end.get('vla_latency_mean')}s"),
        ("論理時間 / 実時間", f"{end.get('sim_seconds')}秒 / 推論だけで {end.get('wall_seconds')}秒"),
    ]
    p.append("<div class='wrap'>")
    p.append(f"<div class='anim'>{svg}</div>")
    p.append("<div><table>" + "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows) + "</table>")

    per = end.get("per_lander") or {}
    if per:
        p.append("<table style='margin-top:.8rem'><tr><th>機</th><th>VLM指令</th><th>着地</th>"
                 "<th>参照解</th><th>regret</th><th>穴</th><th>接地</th></tr>")
        for pid, v in sorted(per.items(), key=lambda kv: int(kv[0])):
            p.append(f"<tr><td>#{pid} {html.escape(str(v['kind']))}</td>"
                     f"<td>列{v.get('commanded_col')}</td><td>列{v.get('col')}</td>"
                     f"<td>列{v.get('oracle_col')}</td><td>{v.get('regret')}</td>"
                     f"<td>{v.get('holes_created')}</td>"
                     f"<td>{'軟' if v.get('soft') else '硬'} vy{v.get('impact_vy')}</td></tr>")
        p.append("</table>")
    p.append("</div></div>")

    # --- タイムライン ------------------------------------------------------
    p.append("<h2>依頼タイミング</h2>")
    p.append("<p class='lede'>「いつ誰に訊いたか」と「返ってくるまでに世界がどれだけ進んだか」。"
             "漂流 tick が大きいほど、その判断は古い盤面に対する答えになっています。</p>")
    p.append("<div class='tl'><table><tr><th>#</th><th>モデル</th><th>役割</th><th>論理時刻</th>"
             "<th>対象機</th><th>実レイテンシ</th><th>漂流</th><th>結果</th></tr>")
    for c in calls:
        col = "vlmt" if c["role"] == "どこへ" else "vlat"
        if c["role"] == "どこへ":
            got = ", ".join(f"#{k}→列{v}" for k, v in (c["parsed"].get("assign") or {}).items()) or "—"
        else:
            plan = "".join(ACTION_MARK.get(x, "?") for x in (c["parsed"].get("plan12") or []))
            got = f"列{c['parsed'].get('target_col')} へ  <span class='plan'>{plan}</span>"
        p.append(f"<tr><td>{c['seq']}</td><td><span class='tag {col}'>"
                 f"{html.escape(c['model'])}</span></td><td>{html.escape(c['role'])}</td>"
                 f"<td>{c['t']}s</td><td>{','.join('#'+str(x) for x in c['pids'])}</td>"
                 f"<td>{c['latency_s']:.2f}s</td><td>{c['drift_ticks']} tick "
                 f"({c['drift_ticks']*dt:.2f}論理秒)</td><td>{got}"
                 + (f" <span class='tag bad'>{html.escape(str(c['error']))}</span>"
                    if c.get("error") else "") + "</td></tr>")
    p.append("</table></div>")

    # --- カード ------------------------------------------------------------
    p.append(f"<h2>各呼び出しの入出力 ({len(calls)} 件)</h2>")
    p.append("<p class='lede'>画像は<b>実際にモデルへ渡した PNG そのもの</b>です。"
             "VLM には全機が写った 1 枚（自機マーカーなし）、"
             "VLA には自機を白枠で囲った 1 枚を渡しています。</p>")
    for c in calls:
        cls = "vlm" if c["role"] == "どこへ" else "vla"
        img = ""
        if c.get("png_b64"):
            img = (f"<img src='data:image/png;base64,{c['png_b64']}' width='130' "
                   f"alt='呼び出し{c['seq']}の入力'>")
        tags = [f"<span class='tag {'vlmt' if cls=='vlm' else 'vlat'}'>[{c['seq']}] "
                f"{html.escape(c['model'])}</span>",
                f"<span class='tag'>t={c['t']}s</span>",
                f"<span class='tag'>{c['latency_s']:.2f}s</span>",
                f"<span class='tag'>{c['drift_ticks']} tick 漂流</span>"]
        if c.get("error"):
            tags.append(f"<span class='tag bad'>{html.escape(str(c['error']))}</span>")
        if c["role"] == "どこへ":
            head = "「どこへ」— 全機の着地列を決める"
            got = ", ".join(f"#{k} → 列{v}"
                            for k, v in (c["parsed"].get("assign") or {}).items()) or "(取得できず)"
        else:
            head = f"「どうやって」— 機 #{c['pids'][0]} の行動列を出す"
            plan = "".join(ACTION_MARK.get(x, "?") for x in (c["parsed"].get("plan12") or []))
            got = (f"目標 列{c['parsed'].get('target_col')} / "
                   f"先頭12手 <span class='plan'>{plan}</span>")
        p.append(
            f"<div class='card {cls}{' err' if c.get('error') else ''}'>{img}<div class='body'>"
            f"<p class='hd'>{html.escape(head)}</p>"
            f"<div class='meta'>{''.join(tags)}</div>"
            f"<div class='meta'>画像以外の入力: <code>"
            f"{html.escape(json.dumps(c.get('inputs', {}), ensure_ascii=False)[:400])}</code></div>"
            f"<div class='meta'>解釈: {got}</div>"
            f"<div class='meta'>{html.escape(c.get('note',''))}</div>"
            f"<pre>{html.escape(str(c.get('raw') or '')[:700])}</pre>"
            + (f"<details><summary>渡したプロンプト全文</summary>"
               f"<pre>{html.escape(str(c.get('prompt')))}</pre></details>"
               if c.get("prompt") else "")
            + "</div></div>")
    return "\n".join(p)
