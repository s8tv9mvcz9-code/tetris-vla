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

    # ジグ (打数で色が変わり、壊れると輪郭だけになる)
    spots = [(6.0, 12.0), (12.0, 9.5), (18.0, 12.0), (9.0, 16.5), (15.0, 16.5)]
    truth = {t["jid"]: t for t in res.get("jig_truth", [])}
    for jid, (jx, jy) in enumerate(spots):
        broken = [f["jigs"][jid][1] for f in frames]
        a(f'<circle cx="{jx*px:.1f}" cy="{jy*px:.1f}" r="{1.15*px:.1f}" fill="#78c8ff" '
          f'opacity="0"><animate attributeName="opacity" '
          f'values="{_fmt([0.0 if b else 1.0 for b in broken],0)}" keyTimes="{kt}" '
          f'dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        a(f'<circle cx="{jx*px:.1f}" cy="{jy*px:.1f}" r="{1.15*px:.1f}" fill="none" '
          f'stroke="#6a6a78" stroke-width="2" stroke-dasharray="3 3" opacity="0">'
          f'<animate attributeName="opacity" values="{_fmt([1.0 if b else 0.0 for b in broken],0)}" '
          f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        mx = truth.get(jid, {}).get("max_durability", "?")
        a(f'<text x="{jx*px:.1f}" y="{(jy+2.1)*px:.1f}" fill="{DIM}" font-size="9" '
          f'text-anchor="middle">#{jid}</text>')

    # パドル (セグメント線つき)
    pw = PADDLE_W * px
    pad_vals = ";".join(f"{(f['paddle_x'] - PADDLE_W / 2) * px:.1f},0" for f in frames)
    a(f'<g><animateTransform attributeName="transform" type="translate" '
      f'values="{pad_vals}" '
      f'keyTimes="{kt}" dur="{dur:.2f}s" repeatCount="indefinite"/>')
    a(f'<rect x="0" y="{(PADDLE_Y-0.3)*px:.1f}" width="{pw:.1f}" height="{0.6*px:.1f}" '
      f'fill="#64f0b4" rx="2"/>')
    for s in range(1, N_SEG):
        a(f'<line x1="{s*pw/N_SEG:.1f}" y1="{(PADDLE_Y-0.3)*px:.1f}" x2="{s*pw/N_SEG:.1f}" '
          f'y2="{(PADDLE_Y+0.3)*px:.1f}" stroke="#1e5a46"/>')
    a('</g>')

    # ボール
    bx = [(f["balls"][0][0] * px if f["balls"] else -99) for f in frames]
    by = [(f["balls"][0][1] * px if f["balls"] else -99) for f in frames]
    a(f'<circle r="{BALL_R*px:.1f}" fill="#fff">'
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


def pinball_html(res: dict, svg: str,
                 title: str = "産業制御 × Physical AI — ピンボール／ブロック崩し") -> str:
    sc = res["score"]
    cfg = res.get("config", {})
    calls = res["calls"]
    p: list[str] = [f"<title>{html.escape(title)}</title>", f"<style>{_CSS}</style>",
                    f"<h1>{html.escape(title)}</h1>"]
    p.append(
        "<p class='lede'>同じ盤面の上で <b>4 つの制御層が別々のクロックで</b> 動いています。"
        "下ほど速く決定的、上ほど遅く賢い。上位の判断は必ず<b>古い盤面に対する答え</b>として"
        "降りてくるので、下位はそれを前提に動きます。"
        "ジグ（バンパー）の<b>耐久度は隠れ状態</b>で、打数しか観測できません。"
        "壊れて素通りになって初めて故障が分かり、シーケンサが救済機を出します。</p>")

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

    rows = [
        ("崩したブロック", f"{sc['blocks_broken']}/{sc['blocks_total']}"),
        ("列ごとの内訳", str(sc["per_col"])),
        ("均等性", f"{sc['evenness']}  (1.0 が理想)"),
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
    p.append("<table class='full'><tr><th>#</th><th>ラング</th><th>条件</th>"
             "<th>コイル</th><th>意味</th></tr>")
    for i, r in enumerate(build_ladder(), 1):
        cond = " OR ".join("(" + " AND ".join(b) + ")" for b in r.branches)
        if r.seal:
            cond += f" OR <b>{r.coil}</b>(自己保持)"
        p.append(f"<tr><td>{i}</td><td>{html.escape(r.name)}</td>"
                 f"<td class='rung'>{cond}</td><td class='rung'>{html.escape(r.coil)}</td>"
                 f"<td>{html.escape(r.comment)}</td></tr>")
    p.append("</table>")

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
