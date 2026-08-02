"""GitHub Pages 用の静的サイトを組む。

Markdown のドキュメントを HTML に変換し、results/ の成果物 (自己完結 HTML と SVG)
を集めて、目次から辿れる形にする。**実装コードと学習データは載せない。**

    python3 scripts/build_site.py [出力先]     # 既定は _site

依存は標準ライブラリだけ。Markdown ライブラリを入れずに済ませているのは、
CI で余計な依存を持ち込まないため。対応しているのはこのリポジトリの docs が
実際に使っている記法 (見出し / 表 / コードブロック / リンク / 強調 / 箇条書き /
引用 / 水平線) に限る。
"""

from __future__ import annotations

import html
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

CSS = """
:root{color-scheme:dark}
body{background:#101016;color:#e8e8ef;font-family:-apple-system,BlinkMacSystemFont,
"Hiragino Sans","Noto Sans JP",sans-serif;line-height:1.75;margin:0;
padding:2rem 1.4rem 5rem;max-width:900px;margin-inline:auto;font-size:15px}
h1{font-size:1.5rem;line-height:1.4;border-bottom:2px solid #2a2a38;padding-bottom:.5rem;
margin-top:2.4rem}
h2{font-size:1.18rem;margin-top:2.2rem;border-left:4px solid #7ee787;padding-left:.6rem}
h3{font-size:1rem;margin-top:1.6rem;color:#c8c8d8}
a{color:#78c8ff}
code{font-family:ui-monospace,Menlo,monospace;font-size:.87em;background:#1b1b24;
padding:.12em .38em;border-radius:3px;color:#ffd9a0}
pre{background:#0b0b11;border:1px solid #24242f;border-radius:6px;padding:.85rem 1rem;
overflow-x:auto}
pre code{background:none;padding:0;color:#cfe8ff}
table{border-collapse:collapse;margin:1rem 0;font-size:.9rem;display:block;overflow-x:auto}
th,td{border:1px solid #2a2a38;padding:.4rem .7rem;text-align:left;white-space:nowrap}
th{background:#1a1a24;color:#9fd8ff}
blockquote{border-left:3px solid #55556a;margin:1rem 0;padding:.2rem 0 .2rem 1rem;
color:#a8a8ba}
hr{border:none;border-top:1px solid #2a2a38;margin:2.2rem 0}
strong{color:#fff}
ul,ol{padding-left:1.4rem}
.nav{position:sticky;top:0;background:#101016ee;backdrop-filter:blur(6px);
border-bottom:1px solid #24242f;margin:-2rem -1.4rem 1.6rem;padding:.7rem 1.4rem;
font-size:.85rem}
.nav a{margin-right:1rem;text-decoration:none}
.card{border:1px solid #2a2a38;border-radius:8px;padding:.9rem 1.1rem;margin:.7rem 0;
background:#14141c}
.card h3{margin:0 0 .3rem;font-size:.98rem}
.card p{margin:.2rem 0;color:#a8a8ba;font-size:.87rem}
.tag{display:inline-block;font-size:.72rem;border:1px solid;border-radius:3px;
padding:.05rem .4rem;margin-right:.4rem;vertical-align:.08em}
.lede{color:#a8a8ba}
"""

NAV = ('<div class="nav"><a href="index.html">← 目次</a>'
       '<a href="README.html">README</a>'
       '<a href="physical-ai-realtime.html">リアルタイム制約</a>'
       '<a href="design-qa.html">設計 Q&amp;A</a>'
       '<a href="results/pinball.html">走行の記録</a></div>')

#: (元ファイル, 出力名, 説明)。存在しないものは黙って飛ばす
DOCS = [
    ("README.md", "README.html", "README — PoC 全体と中心的な発見"),
    ("docs/physical-ai-realtime.md", "physical-ai-realtime.html",
     "Physical AI のリアルタイム運用課題（実測）"),
    ("docs/design-qa.md", "design-qa.html", "設計 Q&A — なぜこの題材にしたか"),
    ("results/pinball_matrix.md", "pinball_matrix.html", "パドル制御層の差し替え比較"),
    ("results/benchmark.md", "benchmark.html", "マシンベンチマーク"),
    ("results/flight_bench.md", "flight_bench.html", "パラシュート降下ベンチ"),
    ("results/coordination.md", "coordination.html", "協調モードの比較"),
]

#: 成果物の一言説明。無いものは説明なしで並べる
NOTES = {
    "pinball.html": "産業制御 × Physical AI。タクト図・ラダー図・状態遷移・"
                    "ST コード・4 段タイムライン",
    "pinball.svg": "同じ走行の盤面アニメーション単体",
    "flock.html": "群れ制御",
    "flock.svg": "同じ走行の盤面アニメーション単体",
    "flight_pd.svg": "同じ走行の軌跡アニメーション単体",
    "flight_pd.html": "パラシュート降下 + 推進ジェット",
    "decisions.html": "テトリス — 同じ盤面に対する各方式の判断の突き合わせ",
}


#: 変換元 .md -> 出力 .html の対応。出力はフラットなので、
#: リンク中の `docs/xxx.md` から docs/ を落として解決する必要がある。
#: DOCS から機械的に作るので、片方だけ直して食い違うことがない。
_LINK_MAP: dict[str, str] = {}


def _rewrite_md_link(target: str) -> str | None:
    """`.md` へのリンクを、出力側のファイル名に写像する。対象外なら None。"""
    t = target.lstrip("./")
    if t in _LINK_MAP:
        return _LINK_MAP[t]
    base = t.rsplit("/", 1)[-1]
    for src, dst in _LINK_MAP.items():
        if src.rsplit("/", 1)[-1] == base:
            return dst
    return None


def _inline(s: str) -> str:
    """行内記法。エスケープしてから記法を戻す。

    コード span は先にプレースホルダへ退避する。バッククォートで分割してから
    強調を処理すると、`**冷えたマシンで `--num-ctx 2048` を測った**` のように
    **強調がコードをまたぐ場合に開始と終了が別の断片へ落ちて変換されない**。
    """
    s = html.escape(s)
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    # .md へのリンクは出力側のファイル名へ張り替える。
    # 対応が無ければリポジトリの該当ファイルへ逃がす (サイト内で切れさせない)
    def _md(m: re.Match) -> str:
        label, target = m.group(1), m.group(2) + ".md"
        dst = _rewrite_md_link(target)
        if dst:
            return f'<a href="{dst}">{label}</a>'
        return (f'<a href="https://github.com/s8tv9mvcz9-code/tetris-vla/blob/main/'
                f'{target.lstrip("./")}">{label}</a>')

    s = re.sub(r"\[([^\]]+)\]\(([^)]+?)\.md\)", _md, s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{spans[int(m.group(1))]}</code>", s)


def md2html(md: str) -> str:
    lines, out, i = md.split("\n"), [], 0
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                        # コードブロック
            flush()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            i += 1
            continue
        if re.match(r"^\|.*\|\s*$", ln) and i + 1 < len(lines) \
                and re.match(r"^\|[\s:\-|]+\|\s*$", lines[i + 1]):
            flush()                                     # GFM テーブル

            def cells(r: str) -> list[str]:
                return [c.strip() for c in r.strip().strip("|").split("|")]

            head = cells(ln)
            i += 2
            rows = []
            while i < len(lines) and re.match(r"^\|.*\|\s*$", lines[i]):
                rows.append(cells(lines[i]))
                i += 1
            t = ["<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("".join(t) + "</table>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush()
            lv = len(m.group(1))
            out.append(f"<h{lv}>{_inline(m.group(2))}</h{lv}>")
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})\s*$", ln):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if re.match(r"^\s*[-*+]\s+", ln) or re.match(r"^\s*\d+[.)]\s+", ln):
            flush()
            ordered = bool(re.match(r"^\s*\d+[.)]\s+", ln))
            items: list[str] = []
            while i < len(lines) and (re.match(r"^\s*[-*+]\s+", lines[i])
                                      or re.match(r"^\s*\d+[.)]\s+", lines[i])
                                      or (items and lines[i].startswith("  ")
                                          and lines[i].strip())):
                mm = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$", lines[i])
                if mm:
                    items.append(mm.group(1))
                elif items:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items)
                       + f"</{tag}>")
            continue
        if ln.startswith(">"):
            flush()
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(buf)) + "</blockquote>")
            continue
        if not ln.strip():
            flush()
            i += 1
            continue
        para.append(ln.strip())
        i += 1
    flush()
    return "\n".join(out)


def page(title: str, body: str, nav: bool = True) -> str:
    return (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body>'
            f'{NAV if nav else ""}{body}</body></html>')


def head_commit() -> str:
    """CI でも壊れないように git 情報の取得は失敗を許す。"""
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%h %ad %s",
             "--date=format:%Y-%m-%d %H:%M"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def check_links(out: pathlib.Path) -> list[tuple[str, str]]:
    """出力サイト内のリンクが全部解決するか確かめる。

    手で点検すると必ず抜けるので、ビルドの一部にして落とす。
    成果物を消したり増やしたりしたときに、目次や本文の参照が置き去りになるのを防ぐ。
    """
    pat = re.compile(r"""(?:href|src)\s*=\s*["\']([^"\'#]+)""")
    bad: list[tuple[str, str]] = []
    for f in sorted(out.rglob("*.html")):
        for m in pat.finditer(f.read_text(errors="ignore")):
            h = m.group(1)
            if h.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            if not (f.parent / h).resolve().exists():
                bad.append((str(f.relative_to(out)), h))
    return bad


def main(out_dir: str = "_site") -> None:
    out = pathlib.Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "results").mkdir(parents=True)

    _LINK_MAP.clear()
    _LINK_MAP.update({src: dst for src, dst, _ in DOCS})
    made = []
    for src, dst, desc in DOCS:
        p = ROOT / src
        if not p.exists():
            continue
        (out / dst).write_text(page(desc, md2html(p.read_text(encoding="utf-8"))),
                               encoding="utf-8")
        made.append((dst, desc, src))

    arts = []
    res = ROOT / "results"
    if res.is_dir():
        for f in sorted(res.glob("*.html")) + sorted(res.glob("*.svg")):
            shutil.copy2(f, out / "results" / f.name)
            arts.append(f.name)

    b = ["<h1>tetris-vla — 産業制御 × Physical AI</h1>",
         "<p class='lede'>ラダー / シーケンサ / VLA / VLM が別々のクロックで同居する"
         "デモの、ドキュメントと実測成果物です。実装コードと学習データは載せていません"
         "（<a href='https://github.com/s8tv9mvcz9-code/tetris-vla'>リポジトリ</a>を"
         "参照してください）。</p>"]
    hc = head_commit()
    if hc:
        b.append(f"<p class='lede'>基準コミット: <code>{html.escape(hc)}</code></p>")
    b += ["<h2>まず読むもの</h2>",
          "<div class='card'><h3><a href='results/pinball.html'>"
          "ピンボール — 4 層が別クロックで同居する走行</a></h3>"
          "<p>工程のタクト、ラダー図と通電表示、状態遷移、同じ論理の ST コード、"
          "そして VLM の助言・シーケンサ・盤面の事故・工程消化を重ねた"
          "4 段タイムライン。</p></div>",
          "<div class='card'><h3><a href='README.html'>README</a></h3>"
          "<p>3 つの中心的な発見。3 つめが上の実機の話。</p></div>",
          "<h2>ドキュメント</h2>"]
    for dst, desc, src in made:
        b.append(f"<div class='card'><h3><a href='{dst}'>{html.escape(desc)}</a></h3>"
                 f"<p><code>{html.escape(src)}</code></p></div>")
    b.append("<h2>成果物（HTML / SVG）</h2>")
    for n in arts:
        b.append(f"<div class='card'><h3><a href='results/{n}'>{n}</a></h3>"
                 f"<p>{NOTES.get(n, '')}</p></div>")
    (out / "index.html").write_text(page("tetris-vla — 産業制御 × Physical AI",
                                         "\n".join(b), nav=False), encoding="utf-8")
    # Jekyll に処理させない (results/ の自己完結 HTML をそのまま出すため)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    broken = check_links(out)
    if broken:
        for f, h in broken:
            print(f"リンク切れ: {f} -> {h}", file=sys.stderr)
        raise SystemExit(f"リンク切れ {len(broken)} 件。サイトを出さない")
    print(f"docs {len(made)} / artifacts {len(arts)} -> {out}  (リンク切れなし)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site")
