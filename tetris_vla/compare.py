"""tetris_vla.compare — *途中判断* の突き合わせ。

結果のグラフは「どちらが勝ったか」しか教えてくれない。この PoC で本当に見たいのは
**まったく同じ盤面・同じピクセルを見せたときに、各方式が何を選んだか** である。

そこで:

1. 1 回の走行から「判断が割れやすい盤面」をいくつか抜き出す
2. その 1 枚を、自律 / A2A / 独裁 / オラクル にそのまま見せる
3. 各方式が宣言した着地点を、同じ盤面の上に重ねて描く

同じ入力に対する出力の差分だけが残るので、スコアの上下ではなく
「何を見落としたのか」「誰と誰がぶつかったのか」が直接読める。

出力は自己完結の HTML 1 枚 (PNG は data URI で埋め込み)。
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .agents import (
    PERSONAS,
    HeuristicAgent,
    HeuristicDictator,
    Intent,
    JointObservation,
    MockBackend,
    MockConfig,
    Observation,
    Persona,
    PieceInfo,
    VLAAgent,
    VLADictator,
    plan_placements,
)
from .engine import (
    ROT_CELLS,
    Engine,
    EngineConfig,
    cell_span,
    drop_row,
)
from .render import BASE, EGO_RING, RenderConfig, render, to_png
from .runtime import RuntimeConfig, Session

#: 判断を重ね描きするときの色 (ピース ID ごとに巡回)
OVERLAY_COLORS = [
    (255, 96, 96),
    (96, 200, 255),
    (255, 210, 80),
    (150, 255, 140),
    (220, 140, 255),
]


# --------------------------------------------------------------------------
# 盤面の切り出し
# --------------------------------------------------------------------------


@dataclass
class Scene:
    """比較用に凍結した 1 場面。"""

    tick: int
    engine: Engine
    personas: dict[int, Persona]

    @property
    def pieces(self) -> list:
        return sorted(self.engine.active, key=lambda p: p.pid)


def capture_scenes(
    seed: int = 1,
    active: int = 3,
    n_scenes: int = 4,
    first_at: int = 120,
    every: int = 90,
    fall_period: int = 4,
) -> list[Scene]:
    """1 本の走行を進めながら、複数ピースが同時に飛んでいる瞬間を凍結する。

    ヒューリスティック自律で走らせた「ごく普通の進行」から拾うので、
    比較のために都合よく作った盤面ではない。
    """
    from copy import deepcopy

    ecfg = EngineConfig(active_pieces=active, fall_period=fall_period, seed=seed)
    rcfg = RuntimeConfig(max_ticks=first_at + every * n_scenes + 5,
                         render=RenderConfig(), compute_oracle=False)
    sess = Session(ecfg, rcfg, HeuristicAgent())

    scenes: list[Scene] = []
    targets = {first_at + every * i for i in range(n_scenes)}
    # Session.run() を使わず手回しして、狙った tick で内部状態を複製する
    while sess.engine.tick < rcfg.max_ticks and not sess.engine.game_over:
        tick = sess.engine.tick
        if tick in targets and len(sess.engine.active) >= min(2, active):
            scenes.append(
                Scene(
                    tick=tick,
                    engine=deepcopy(sess.engine),
                    personas={pid: st.persona for pid, st in sess.states.items()},
                )
            )
        sess._submit(tick)
        sess._start_services(tick)
        sess._deliver(tick)
        actions = {}
        for piece in sess.engine.active:
            st = sess.states.get(piece.pid)
            from .runtime import reflex

            a, _ = reflex(piece, st.intent if st else None,
                          lambda act, p=piece: sess.engine.action_is_legal(p, act))
            actions[piece.pid] = a
        sess.engine.step(actions)
    return scenes


# --------------------------------------------------------------------------
# 各方式に同じ盤面を見せて判断だけ取り出す
# --------------------------------------------------------------------------


@dataclass
class Verdict:
    """1 方式・1 場面ぶんの判断。"""

    method: str
    intents: dict[int, Intent]
    notes: dict[int, str] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    model_calls: int = 0
    latency_s: float = 0.0

    def landing(self, scene: Scene, pid: int) -> tuple[tuple[int, int], ...]:
        """宣言した (col, rot) から確定地形への着地セルを求める。"""
        intent = self.intents.get(pid)
        p = scene.engine.piece(pid)
        if intent is None or p is None:
            return ()
        rot = intent.rot % 4
        x = intent.col - cell_span(p.kind, rot)[2]
        y = drop_row(scene.engine.board, p.kind, rot, x)
        if y is None:
            return ()
        return tuple((y + dy, x + dx) for dy, dx in ROT_CELLS[p.kind][rot])


def _observation(scene: Scene, pid: int, cfg: RenderConfig,
                 ghost_cells=None, coordination: str = "autonomous") -> Observation:
    p = scene.engine.piece(pid)
    snap = scene.engine.snapshot()
    img = render(snap, ego_pid=pid, cfg=cfg, ghost_cells=ghost_cells)
    return Observation(
        png=to_png(img), image=img, ego_pid=pid, ego_kind=p.kind,
        persona=scene.personas.get(pid, PERSONAS["neutral"]),
        tick=snap.tick, width=snap.width, height=snap.height,
        fall_period=p.fall_period, render_cfg=cfg, coordination=coordination,
    )


def verdict_autonomous(scene: Scene, agent=None, label: str = "自律 (通信なし)") -> Verdict:
    """各ピースが独立に、他人の位置だけを見て決める。"""
    agent = agent or HeuristicAgent()
    cfg = RenderConfig()
    intents, notes, errors = {}, {}, {}
    lat = 0.0
    for p in scene.pieces:
        d = agent.decide(_observation(scene, p.pid, cfg))
        lat += d.latency_s
        if d.intent:
            intents[p.pid] = d.intent
            notes[p.pid] = d.intent.why
        if d.error:
            errors[p.pid] = d.error
    return Verdict(label, intents, notes, errors, model_calls=len(scene.pieces), latency_s=lat)


def verdict_a2a(scene: Scene, agent=None, label: str = "A2A (予定地を共有)",
                rounds: int = 2) -> Verdict:
    """宣言 -> ゴースト描画 -> 再考、を数ラウンド回す。

    最初のラウンドは自律と同じ。2 ラウンド目以降で、他人の宣言が
    スナップショットにゴーストとして写り込む。通信は画像の中だけで完結している。
    """
    agent = agent or HeuristicAgent()
    cfg = RenderConfig(show_ghosts=True)
    declared: dict[int, Intent] = {}
    notes, errors = {}, {}
    calls = 0
    lat = 0.0
    for _ in range(max(1, rounds)):
        for p in scene.pieces:
            ghosts: list[tuple[int, int]] = []
            for other_pid, intent in declared.items():
                if other_pid == p.pid:
                    continue
                q = scene.engine.piece(other_pid)
                if q is None:
                    continue
                rot = intent.rot % 4
                x = intent.col - cell_span(q.kind, rot)[2]
                y = drop_row(scene.engine.board, q.kind, rot, x)
                if y is not None:
                    ghosts.extend((y + dy, x + dx) for dy, dx in ROT_CELLS[q.kind][rot])
            d = agent.decide(_observation(scene, p.pid, cfg, ghosts, "a2a"))
            calls += 1
            lat += d.latency_s
            if d.intent:
                declared[p.pid] = d.intent
                notes[p.pid] = d.intent.why
            if d.error:
                errors[p.pid] = d.error
    return Verdict(label, dict(declared), notes, errors, model_calls=calls, latency_s=lat)


def verdict_dictator(scene: Scene, agent=None, label: str = "独裁 (一括指揮)") -> Verdict:
    """1 エージェントが全ピースをまとめて 1 回で決める。"""
    agent = agent or HeuristicDictator()
    cfg = RenderConfig()
    snap = scene.engine.snapshot()
    img = render(snap, ego_pid=None, cfg=cfg)
    infos = tuple(
        PieceInfo(
            pid=p.pid, kind=p.kind,
            persona=scene.personas.get(p.pid, PERSONAS["neutral"]),
            fall_period=p.fall_period, left_col=p.left_col,
            top_row=min(r for r, _ in p.cells()),
        )
        for p in scene.pieces
    )
    jobs = JointObservation(
        png=to_png(img), image=img, pieces=infos, tick=snap.tick,
        width=snap.width, height=snap.height, render_cfg=cfg,
    )
    jd = agent.decide_joint(jobs)
    notes = {pid: i.why for pid, i in jd.intents.items()}
    return Verdict(label, dict(jd.intents), notes, dict(jd.errors),
                   model_calls=1, latency_s=jd.latency_s)


def verdict_oracle(scene: Scene, label: str = "参照 (地形のみの最良手)") -> Verdict:
    """他ピースを一切考えず、確定地形に対してだけ最適な手。判断のものさし。"""
    intents, notes = {}, {}
    for p in scene.pieces:
        persona = scene.personas.get(p.pid, PERSONAS["neutral"])
        plans = plan_placements(scene.engine.board, p.kind, persona)
        if plans:
            intents[p.pid] = plans[0].intent()
            notes[p.pid] = f"score={plans[0].score:.1f}"
    return Verdict(label, intents, notes, model_calls=0)


def default_verdicts(scene: Scene, vla_skill: float | None = None) -> list[Verdict]:
    out = [
        verdict_oracle(scene),
        verdict_autonomous(scene),
        verdict_a2a(scene),
        verdict_dictator(scene),
    ]
    if vla_skill is not None:
        mock = VLAAgent(MockBackend(MockConfig(skill=vla_skill, p_malformed=0.0,
                                               p_refuse=0.0, p_out_of_range=0.0), seed=7))
        out.append(verdict_autonomous(scene, mock, f"VLA 自律 (skill={vla_skill})"))
        dict_mock = VLADictator(MockBackend(MockConfig(skill=vla_skill, p_malformed=0.0,
                                                       p_refuse=0.0, p_out_of_range=0.0), seed=7))
        out.append(verdict_dictator(scene, dict_mock, f"VLA 独裁 (skill={vla_skill})"))
    return out


# --------------------------------------------------------------------------
# 判断の重ね描き (人間向け。エージェント用レンダラとは別物)
# --------------------------------------------------------------------------


CONFLICT_COLOR = (255, 50, 50)


def render_verdict(scene: Scene, verdict: Verdict, scale: int = 3) -> Image.Image:
    """盤面の上に「各ピースがどこへ行くと言ったか」を重ねる。

    これは人間が読むための図。厳密デコードの制約は課さないので自由に描いてよい
    (エージェントに渡る画像は render.render のほうであり、こちらとは別物)。

      細い枠   = そのピースの現在位置
      塗り+太枠 = 宣言した着地点
      赤       = 複数ピースが同じセルを狙っている (意図の衝突)
    """
    cfg = RenderConfig(labels=False)
    snap = scene.engine.snapshot()
    base = render(snap, ego_pid=None, cfg=cfg).convert("RGBA")
    base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
    cw = cfg.cell_px * scale
    top = cfg.header_px * scale

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    conflict = _conflict_cells(scene, verdict)

    for i, p in enumerate(scene.pieces):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        for r, c in p.cells():
            if 0 <= r < snap.height and 0 <= c < snap.width:
                x0, y0 = c * cw, top + r * cw
                od.rectangle([x0 + 1, y0 + 1, x0 + cw - 2, y0 + cw - 2],
                             outline=(*color, 255), width=1)
        for r, c in verdict.landing(scene, p.pid):
            if not (0 <= r < snap.height and 0 <= c < snap.width):
                continue
            x0, y0 = c * cw, top + r * cw
            hit = (r, c) in conflict
            edge = CONFLICT_COLOR if hit else color
            od.rectangle([x0 + 1, y0 + 1, x0 + cw - 2, y0 + cw - 2],
                         fill=(*edge, 90 if not hit else 130), outline=(*edge, 255), width=2)

    img = Image.alpha_composite(base, overlay).convert("RGB")
    # 列番号の帯を自前で描く (エージェント用ヘッダと違い、読みやすさ優先)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, top - 1], fill=(10, 10, 12))
    for c in range(snap.width):
        d.text((c * cw + cw // 2 - 3, top // 2 - 4), str(c % 10), fill=(170, 170, 180))
    return img


def _conflict_cells(scene: Scene, verdict: Verdict) -> set[tuple[int, int]]:
    """複数ピースが同じセルを狙っている箇所 = 意図の衝突。"""
    seen: dict[tuple[int, int], int] = {}
    clash: set[tuple[int, int]] = set()
    for p in scene.pieces:
        for cell in verdict.landing(scene, p.pid):
            if cell in seen and seen[cell] != p.pid:
                clash.add(cell)
            seen[cell] = p.pid
    return clash


def conflict_count(scene: Scene, verdict: Verdict) -> int:
    return len(_conflict_cells(scene, verdict))


def disagreement(reference: Verdict, other: Verdict) -> list[int]:
    """参照と判断が食い違った pid の一覧。"""
    out = []
    for pid, ref in reference.intents.items():
        got = other.intents.get(pid)
        if got is None or (got.col, got.rot) != (ref.col, ref.rot):
            out.append(pid)
    return out


# --------------------------------------------------------------------------
# HTML 出力
# --------------------------------------------------------------------------


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin:0; padding:24px; font-family: ui-sans-serif, system-ui, -apple-system,
  "Hiragino Sans", "Noto Sans JP", sans-serif; line-height:1.7; }
h1 { font-size:1.5rem; margin:0 0 .3rem; }
h2 { font-size:1.15rem; margin:2.5rem 0 .5rem; padding-top:1rem;
     border-top:1px solid color-mix(in srgb, currentColor 18%, transparent); }
p.lede { opacity:.75; margin:.2rem 0 1.4rem; max-width:70ch; }
.row { display:flex; gap:18px; overflow-x:auto; padding-bottom:8px; }
.card { flex:0 0 auto; min-width:160px; }
.card h3 { font-size:.85rem; margin:0 0 .4rem; font-weight:600; }
.card img { display:block; image-rendering: pixelated;
  border:1px solid color-mix(in srgb, currentColor 22%, transparent); border-radius:4px; }
.meta { font-size:.72rem; opacity:.75; margin-top:.4rem; max-width:200px; }
.tag { display:inline-block; font-size:.68rem; padding:.05rem .4rem; border-radius:3px;
  border:1px solid color-mix(in srgb, currentColor 30%, transparent); margin-right:.25rem; }
.bad { color:#e0483c; border-color:#e0483c; }
.good { color:#1a9b52; border-color:#1a9b52; }
table { border-collapse:collapse; font-size:.8rem; margin-top:.8rem; width:100%; }
th, td { border:1px solid color-mix(in srgb, currentColor 18%, transparent);
  padding:.28rem .5rem; text-align:left; }
th { background: color-mix(in srgb, currentColor 7%, transparent); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.85em; }
.legend { font-size:.78rem; opacity:.8; margin-top:1rem; }
"""


def to_html(
    scenes: Sequence[Scene],
    verdicts_per_scene: Sequence[Sequence[Verdict]],
    title: str = "途中判断の比較",
) -> str:
    parts = [f"<title>{html.escape(title)}</title>", f"<style>{_CSS}</style>",
             f"<h1>{html.escape(title)}</h1>",
             "<p class='lede'>同じ盤面・同じピクセルを各方式に見せ、"
             "<b>その場で何を選んだか</b>だけを並べています。細枠=現在位置、"
             "太枠+斜線=宣言した着地点、<span class='tag bad'>赤</span>=複数ピースが"
             "同じセルを狙っている衝突。</p>"]

    for scene, verdicts in zip(scenes, verdicts_per_scene):
        ref = verdicts[0]
        roster = "、".join(
            f"#{p.pid} {p.kind} (信条 {scene.personas.get(p.pid, PERSONAS['neutral']).name})"
            for p in scene.pieces
        )
        parts.append(f"<h2>t={scene.tick} — 落下中 {len(scene.pieces)} 体</h2>")
        parts.append(f"<p class='lede'>{html.escape(roster)}</p>")
        parts.append("<div class='row'>")
        for v in verdicts:
            img = render_verdict(scene, v)
            clash = conflict_count(scene, v)
            diff = disagreement(ref, v) if v is not ref else []
            tags = []
            if clash:
                tags.append(f"<span class='tag bad'>衝突 {clash}セル</span>")
            else:
                tags.append("<span class='tag good'>衝突なし</span>")
            if v is not ref:
                tags.append(f"<span class='tag'>参照と相違 {len(diff)}体</span>")
            tags.append(f"<span class='tag'>推論 {v.model_calls}回</span>")
            rows = "".join(
                f"<tr><td>#{p.pid} {p.kind}</td>"
                f"<td>{scene.personas.get(p.pid, PERSONAS['neutral']).name}</td>"
                f"<td>{_fmt_intent(v.intents.get(p.pid))}</td>"
                f"<td>{html.escape(v.errors.get(p.pid, '') or '')}</td></tr>"
                for p in scene.pieces
            )
            parts.append(
                f"<div class='card'><h3>{html.escape(v.method)}</h3>"
                f"<img src='{_data_uri(img)}' alt='{html.escape(v.method)}'>"
                f"<div class='meta'>{''.join(tags)}</div>"
                f"<table><tr><th>駒</th><th>信条</th><th>宣言</th><th>異常</th></tr>"
                f"{rows}</table></div>"
            )
        parts.append("</div>")

    parts.append(
        "<p class='legend'>「参照」は他ピースを一切考えず確定地形にだけ最適化した手です。"
        "自律モードは参照と一致しがちですが、その一致こそが衝突の原因になります"
        "— 全員が同じ地形を見て同じ最良手に集中するためです。</p>"
    )
    return "\n".join(parts)


def _fmt_intent(intent: Intent | None) -> str:
    if intent is None:
        return "<i>なし</i>"
    return f"<code>列{intent.col} 回転{intent.rot}</code>"
