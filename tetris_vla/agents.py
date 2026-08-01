"""tetris_vla.agents — 「意思を持つピース」の頭脳。

エージェントは盤面スナップショット PNG を 1 枚受け取り、`Intent` (置きたい列と回転)
を返す。盤面配列そのものは決して渡らない。

  * RandomAgent    : 下限ベースライン
  * HeuristicAgent : ピクセルをデコードして Dellacherie 探索。上限ベースライン兼オラクル
  * VLAAgent       : ローカル VLA (Vision-Language-Action) に画像を投げて JSON を貰う

VLAAgent のバックエンドは差し替え式:
  MockBackend   — モデル無しで動く既定。遅延ゆらぎ / JSON 崩れ / 拒否 / 範囲外を実際に混ぜる
  OllamaBackend — 実機。ローカル Ollama の VLM を叩く
  ReplayBackend — 記録済み生応答を再生 (実モデル実行の決定的リプレイ)
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .engine import (
    ROT_CELLS,
    KIND_ID,
    cell_span,
    col_transitions,
    count_holes,
    bumpiness as _bumpiness,
    drop_row,
    place,
    row_transitions,
    unique_rotations,
    wells,
)
from .render import UNKNOWN_KIND, Decoded, RenderConfig, decode, to_text_board

# --------------------------------------------------------------------------
# 意図 (Intent) と観測 (Observation)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Intent:
    """エージェントの「ここに収まりたい」という意思。

    col は *占有セルの最左列*。回転に依らず意味が変わらないので、VLM に説明しやすく
    reflex コントローラ側でも扱いやすい。
    """

    col: int
    rot: int
    why: str = ""

    def key(self) -> tuple[int, int]:
        return (self.col, self.rot)


@dataclass
class Observation:
    """エージェントに渡る情報のすべて。ここに盤面配列は含まれない。"""

    png: bytes
    image: Image.Image
    ego_pid: int
    ego_kind: str          # 自分が何のピースかは「自己知識」として与える
    persona: "Persona"
    tick: int              # スナップショットを撮った tick (受け取る時には古くなっている)
    width: int
    height: int
    fall_period: int
    render_cfg: RenderConfig
    #: 協調モード。プロンプトの説明文だけが変わる (autonomous / a2a)
    coordination: str = "autonomous"
    #: A2A テキストチャネルを使う場合の他ピースの宣言 (ゴースト方式なら空)
    declarations: tuple[str, ...] = ()


@dataclass
class PieceInfo:
    """独裁モードで「自分の体」を識別するための最小情報。

    自律モードでは自機リング + ego_kind が同じ役割を果たしている。
    独裁者は全ピースを所有するので、その全部について同じ自己知識を持つ。
    """

    pid: int
    kind: str
    persona: "Persona"
    fall_period: int
    left_col: int
    top_row: int


@dataclass
class JointObservation:
    """独裁モードの観測。スナップショットは 1 枚、意思決定は全ピース分まとめて 1 回。"""

    png: bytes
    image: Image.Image
    pieces: tuple[PieceInfo, ...]
    tick: int
    width: int
    height: int
    render_cfg: RenderConfig
    coordination: str = "dictator"


@dataclass
class Decision:
    intent: Intent | None
    latency_s: float = 0.0
    raw: str | None = None
    prompt: str | None = None
    error: str | None = None  # parse_fail / refusal / out_of_range / backend_error / no_placement


@dataclass
class JointDecision:
    """独裁者の 1 回の推論で得られる全ピース分の意図。"""

    intents: dict[int, Intent]
    latency_s: float = 0.0
    raw: str | None = None
    prompt: str | None = None
    errors: dict[int, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# ペルソナ = 各ピースに持たせる「意思」
# --------------------------------------------------------------------------

#: Dellacherie / El-Tetris の標準重み
BASE_WEIGHTS: dict[str, float] = {
    "landing_height": -4.500158825082766,
    "eroded": 3.4181268101392694,
    "row_trans": -3.2178882868487753,
    "col_trans": -9.348695305445199,
    "holes": -7.899265427351652,
    "wells": -3.3855972247263626,
    "bumpiness": 0.0,
    "col_dist": 0.0,
}

FEATURE_ORDER = tuple(BASE_WEIGHTS.keys())


@dataclass(frozen=True)
class Persona:
    """ピース個体の性格。ヒューリスティックの重みと VLA プロンプトの両方を同時に動かす。

    同じ目的関数を「探索で解く頭」と「言語で解く頭」に与えることで、
    両者の比較が意味を持つようにしてある。
    """

    name: str
    directive: str
    weight_overrides: dict[str, float] = field(default_factory=dict)
    fav_col: int | None = None

    def weights(self) -> dict[str, float]:
        w = dict(BASE_WEIGHTS)
        w.update(self.weight_overrides)
        return w


PERSONAS: dict[str, Persona] = {
    "neutral": Persona(
        "neutral",
        "とにかく穴を作らず、平均的に良い場所へ収まれ。",
    ),
    "flat": Persona(
        "flat",
        "地形をできるだけ平らに保て。段差を作る置き方は避けろ。",
        {"bumpiness": -2.2},
    ),
    "nohole": Persona(
        "nohole",
        "自分の下に空洞を作ることを何よりも嫌え。多少高くなっても穴だけは作るな。",
        {"holes": -20.0},
    ),
    "left": Persona(
        "left",
        "盤面の左寄りに収まりたい。左側の列を優先して埋めろ。",
        {"col_dist": -1.2},
        fav_col=0,
    ),
    "right": Persona(
        "right",
        "盤面の右寄りに収まりたい。右側の列を優先して埋めろ。",
        {"col_dist": -1.2},
        fav_col=99,  # クランプされて右端になる
    ),
    "wellkeeper": Persona(
        "wellkeeper",
        "右端の列は I ピースのために空けておきたい。そこを塞ぐ置き方は避けろ。",
        {"col_dist": -0.9},
        fav_col=0,
    ),
}

DEFAULT_PERSONA_CYCLE = ("neutral", "flat", "nohole", "left", "right")


def persona_for(index: int, cycle: Sequence[str] = DEFAULT_PERSONA_CYCLE) -> Persona:
    return PERSONAS[cycle[index % len(cycle)]]


# --------------------------------------------------------------------------
# 配置探索 (Dellacherie) — ヒューリスティックとオラクルの共通実装
# --------------------------------------------------------------------------


@dataclass
class Placement:
    rot: int
    x: int
    y: int
    col: int  # 占有セルの最左列 = Intent.col
    score: float
    features: dict[str, float]

    def intent(self, why: str = "") -> Intent:
        return Intent(col=self.col, rot=self.rot, why=why)


def _clear_full_rows(board: np.ndarray) -> tuple[np.ndarray, list[int]]:
    full = [r for r in range(board.shape[0]) if bool(board[r].all())]
    if not full:
        return board, []
    keep = [r for r in range(board.shape[0]) if r not in set(full)]
    out = np.zeros_like(board)
    out[len(full):] = board[keep]
    return out, full


def evaluate_placement(
    settled: np.ndarray, kind: str, rot: int, x: int, y: int, weights: dict[str, float],
    fav_col: int | None,
) -> tuple[float, dict[str, float]]:
    H, W = settled.shape
    cells = [(y + dy, x + dx) for dy, dx in ROT_CELLS[kind][rot % 4]]
    after = place(settled, kind, rot, x, y)
    after_cleared, full = _clear_full_rows(after)

    rows_from_bottom = [H - 1 - r for r, _ in cells]
    landing_height = (min(rows_from_bottom) + max(rows_from_bottom)) / 2.0
    piece_cells_in_cleared = sum(1 for r, _ in cells if r in set(full))
    eroded = len(full) * piece_cells_in_cleared
    left_col = x + cell_span(kind, rot)[2]

    feats = {
        "landing_height": float(landing_height),
        "eroded": float(eroded),
        "row_trans": float(row_transitions(after_cleared)),
        "col_trans": float(col_transitions(after_cleared)),
        "holes": float(count_holes(after_cleared)),
        "wells": float(wells(after_cleared)),
        "bumpiness": float(_bumpiness(after_cleared)),
        "col_dist": float(abs(left_col - min(fav_col, W - 1))) if fav_col is not None else 0.0,
    }
    score = sum(weights.get(k, 0.0) * v for k, v in feats.items())
    return score, feats


def plan_placements(
    settled: np.ndarray,
    kind: str,
    persona: Persona | None = None,
    obstacles: np.ndarray | None = None,
) -> list[Placement]:
    """確定盤面に対する全 (回転, 列) をスコア順に並べて返す。

    obstacles を渡すと、他の落下ピースも「今そこにある障害物」として着地計算に含める。
    既定では含めない: 落下中ピースは動くので、確定地形だけを相手に計画するのが素直。
    """
    persona = persona or PERSONAS["neutral"]
    weights = persona.weights()
    plan_board = settled if obstacles is None else np.maximum(settled, obstacles)
    H, W = settled.shape
    out: list[Placement] = []
    for rot in unique_rotations(kind):
        min_dy, _, min_dx, max_dx = cell_span(kind, rot)
        for x in range(-min_dx, W - max_dx):
            y = drop_row(plan_board, kind, rot, x)
            if y is None:
                continue
            score, feats = evaluate_placement(settled, kind, rot, x, y, weights, persona.fav_col)
            out.append(Placement(rot=rot, x=x, y=y, col=x + min_dx, score=score, features=feats))
    out.sort(key=lambda p: -p.score)
    return out


def col_range(kind: str, rot: int, width: int) -> tuple[int, int]:
    """その回転で Intent.col が取りうる [min, max]。"""
    _, _, min_dx, max_dx = cell_span(kind, rot)
    return 0, width - 1 - (max_dx - min_dx)


# --------------------------------------------------------------------------
# エージェント実装
# --------------------------------------------------------------------------


class Agent(Protocol):
    name: str

    def decide(self, obs: Observation) -> Decision: ...


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def decide(self, obs: Observation) -> Decision:
        rots = unique_rotations(obs.ego_kind)
        rot = self.rng.choice(rots)
        lo, hi = col_range(obs.ego_kind, rot, obs.width)
        return Decision(
            intent=Intent(col=self.rng.randint(lo, hi), rot=rot, why="random"),
            latency_s=0.0,
        )


class HeuristicAgent:
    """ピクセルをデコードして Dellacherie 探索する上限ベースライン。

    VLA と *完全に同じ画像* しか見ない。違うのは頭脳だけ。
    """

    name = "heuristic"

    def __init__(self, treat_active_as_obstacle: bool = False, top_k: int = 1, seed: int = 0) -> None:
        self.treat_active_as_obstacle = treat_active_as_obstacle
        self.top_k = top_k
        self.rng = random.Random(seed)

    def decide(self, obs: Observation) -> Decision:
        t0 = time.perf_counter()
        dec = decode(obs.image, obs.width, obs.height, obs.render_cfg)
        plans = plan_from_decoded(
            dec, obs.ego_kind, obs.persona, self.treat_active_as_obstacle
        )
        dt = time.perf_counter() - t0
        if not plans:
            return Decision(intent=None, latency_s=dt, error="no_placement")
        pick = plans[0] if self.top_k <= 1 else self.rng.choice(plans[: self.top_k])
        return Decision(intent=pick.intent(why=f"dellacherie={pick.score:.1f}"), latency_s=dt)


def plan_from_decoded(
    dec: Decoded,
    kind: str,
    persona: Persona,
    treat_active_as_obstacle: bool = False,
    avoid_ghosts: bool = True,
) -> list[Placement]:
    """デコード済み観測から配置候補を作る。自機セルは障害物から除外する。

    avoid_ghosts=True なら、他ピースが宣言した予定地 (A2A のゴースト) も避ける。
    ゴーストは画像の中にしか存在しないので、これは「見えている分だけ」の範囲に収まる。
    """
    obstacles = None
    if treat_active_as_obstacle:
        obstacles = np.where(dec.ego, 0, dec.active).astype(np.uint8)
    if avoid_ghosts and dec.ghost is not None and dec.ghost.any():
        g = (dec.ghost & ~dec.ego).astype(np.uint8) * UNKNOWN_KIND
        obstacles = g if obstacles is None else np.maximum(obstacles, g)
    return plan_placements(dec.settled, kind, persona, obstacles)


def plan_joint(
    settled: np.ndarray, pieces: Sequence[PieceInfo]
) -> tuple[dict[int, Intent], dict[int, Placement]]:
    """独裁者の同時計画: 先に着地する順に、互いの決定を織り込みながら割り当てる。

    自律モードとの唯一の差は「他ピースの *意図* を知っていること」。
    仮想盤面に前の割り当てを書き込んでから次を計画するので、
    同じ列を二人が同時に狙う事故が構造的に起きない。
    """
    virtual = settled.copy()
    intents: dict[int, Intent] = {}
    chosen: dict[int, Placement] = {}
    # 下にいる (=先に確定する) ピースから決める
    for info in sorted(pieces, key=lambda p: (-p.top_row, p.pid)):
        plans = plan_placements(virtual, info.kind, info.persona)
        if not plans:
            continue
        best = plans[0]
        intents[info.pid] = best.intent(why=f"joint={best.score:.1f}")
        chosen[info.pid] = best
        virtual = place(virtual, info.kind, best.rot, best.x, best.y)
    return intents, chosen


# --------------------------------------------------------------------------
# VLA
# --------------------------------------------------------------------------

PROMPT_TEMPLATE = """あなたは落下中のテトリスのピース本体です。自分の意思で置き場所を決めます。

添付画像は現在の盤面スナップショットです。
- 白い枠で囲われたブロックが「あなた自身」({kind} ピース) です。
- 暗い色のブロックは既に固定された地形です。
- 白枠のない明るいブロックは、あなたとは別の意思を持つ落下中のピースです。
  彼らが次に何をするかは分かりません。見えている位置だけが手がかりです。
- 盤面は 横 {width} 列 (左端 0 〜 右端 {maxcol})、縦 {height} 行です。
- あなたは 1 tick ごとに「左移動・右移動・回転」を 1 回だけ実行でき、
  {fall_period} tick ごとに必ず 1 行落ちます。落下は止められません。

あなたの信条: {directive}

自分が最終的に収まるべき場所を決めてください。
出力は次の JSON のみ。説明文やコードフェンスは付けないでください。
{{"col": <0〜{maxcol} の整数、あなたの左端ブロックが来る列>, "rot": <0〜3 の整数、回転回数>, "why": "<20字以内の理由>"}}"""


#: 協調モードごとにプロンプトへ差し込む説明
COORDINATION_NOTES: dict[str, str] = {
    "autonomous": (
        "他のピースとは一切通信できません。彼らが今どこにいるかしか分かりません。"
    ),
    "a2a": (
        "灰青色の半透明ブロックは、他のピースが『自分はここに収まる』と宣言した予定地です。"
        "彼らの意思表明なので尊重し、そこを避けて自分の場所を決めてください。"
        "ただし宣言は変更されることがあります。"
    ),
}


def build_prompt(obs: Observation, text_board: str | None = None) -> str:
    p = PROMPT_TEMPLATE.format(
        kind=obs.ego_kind,
        width=obs.width,
        maxcol=obs.width - 1,
        height=obs.height,
        fall_period=obs.fall_period,
        directive=obs.persona.directive,
    )
    note = COORDINATION_NOTES.get(obs.coordination)
    if note:
        p += f"\n\n【協調】{note}"
    if obs.declarations:
        p += "\n他ピースの宣言:\n" + "\n".join(f"  - {d}" for d in obs.declarations)
    if text_board is not None:
        p += ("\n\n盤面のテキスト表現 (@=あなた, o=他のピース, #=固定地形, "
              "*=他ピースの予定地, .=空):\n" + text_board + "\n")
    return p


DICTATOR_PROMPT_TEMPLATE = """あなたは盤面上の落下中ピースを **すべて同時に指揮する司令塔** です。

添付画像は現在の盤面スナップショットです。
- 暗い色のブロックが既に固定された地形、明るい色のブロックが落下中のピースです。
- 盤面は 横 {width} 列 (左端 0 〜 右端 {maxcol})、縦 {height} 行です。
- 各ピースは 1 tick に「左移動・右移動・回転」を 1 回だけ実行でき、
  自分の速度に従って必ず落下します。落下は止められません。

指揮下のピース:
{roster}

あなたは全員の行き先を同時に決められます。互いに邪魔をしない割り当てを作ってください。
先に着地するピース (行番号が大きいもの) の決定が優先されます。

出力は次の JSON のみ。説明文やコードフェンスは付けないでください。
{{"moves": [{{"pid": <ピースID>, "col": <0〜{maxcol}>, "rot": <0〜3>}}, ...]}}"""


def build_joint_prompt(jobs: JointObservation, text_board: str | None = None) -> str:
    roster = "\n".join(
        f"  - pid={p.pid} 種類={p.kind} 現在の左端列={p.left_col} 上端行={p.top_row} "
        f"速度={p.fall_period}tick/行 信条=「{p.persona.directive}」"
        for p in jobs.pieces
    )
    p = DICTATOR_PROMPT_TEMPLATE.format(
        width=jobs.width, maxcol=jobs.width - 1, height=jobs.height, roster=roster
    )
    if text_board is not None:
        p += f"\n\n盤面のテキスト表現 (o=落下中, #=固定地形, .=空):\n{text_board}\n"
    return p


_REFUSAL_PAT = re.compile(
    r"(cannot|can't|unable|sorry|申し訳|できません|わかりません|不明です)", re.I
)


def _extract_json(raw: str) -> dict | None:
    """最初の平衡した {...} を取り出して JSON として読む。"""
    start = raw.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = raw.find("{", start + 1)
    return None


def parse_intent(raw: str, kind: str, width: int) -> tuple[Intent | None, str | None]:
    """生応答を Intent に変換する。失敗理由をタグで返す (指標として集計される)。"""
    if not raw or not raw.strip():
        return None, "backend_error"
    obj = _extract_json(raw)
    if obj is None:
        # JSON が崩れていても数値だけ拾えることがある
        mc = re.search(r'"?col"?\s*[:=]\s*(-?\d+)', raw)
        mr = re.search(r'"?rot"?\s*[:=]\s*(-?\d+)', raw)
        if mc and mr:
            obj = {"col": int(mc.group(1)), "rot": int(mr.group(1)), "why": "salvaged"}
        elif _REFUSAL_PAT.search(raw):
            return None, "refusal"
        else:
            return None, "parse_fail"
    try:
        col = int(obj["col"])
        rot = int(obj["rot"]) % 4
    except (KeyError, TypeError, ValueError):
        return None, "parse_fail"
    why = str(obj.get("why", ""))[:40]

    if rot not in unique_rotations(kind):
        # 見た目が同じ回転に丸める (O の rot=3 など)
        rot = unique_rotations(kind)[rot % len(unique_rotations(kind))]
    lo, hi = col_range(kind, rot, width)
    if col < lo or col > hi:
        clamped = max(lo, min(hi, col))
        return Intent(col=clamped, rot=rot, why=why), "out_of_range"
    return Intent(col=col, rot=rot, why=why), None


def parse_joint_intents(
    raw: str, pieces: Sequence[PieceInfo], width: int
) -> tuple[dict[int, Intent], dict[int, str]]:
    """独裁者の応答を pid -> Intent に変換する。1 体分だけ壊れても他は活かす。"""
    by_pid = {p.pid: p for p in pieces}
    intents: dict[int, Intent] = {}
    errors: dict[int, str] = {}
    if not raw or not raw.strip():
        return intents, {p.pid: "backend_error" for p in pieces}

    obj = _extract_json(raw)
    moves = None
    if isinstance(obj, dict):
        moves = obj.get("moves") or obj.get("assignments") or obj.get("pieces")
    if not isinstance(moves, list):
        if _REFUSAL_PAT.search(raw):
            return intents, {p.pid: "refusal" for p in pieces}
        return intents, {p.pid: "parse_fail" for p in pieces}

    for m in moves:
        if not isinstance(m, dict):
            continue
        try:
            pid = int(m["pid"])
        except (KeyError, TypeError, ValueError):
            continue
        info = by_pid.get(pid)
        if info is None:
            continue
        intent, err = parse_intent(json.dumps(m), info.kind, width)
        if intent is not None:
            intents[pid] = intent
        if err:
            errors[pid] = err
    for p in pieces:
        if p.pid not in intents and p.pid not in errors:
            errors[p.pid] = "missing_assignment"
    return intents, errors


class HeuristicDictator:
    """全ピースを同時に最適化する上限ベースライン (独裁モード)。

    自律モードとの唯一の差は「他ピースの意図を知っていること」なので、
    両者のスコア差がそのまま『意図の可視性』の価値になる。
    """

    name = "dictator-heuristic"

    def decide_joint(self, jobs: JointObservation) -> JointDecision:
        t0 = time.perf_counter()
        dec = decode(jobs.image, jobs.width, jobs.height, jobs.render_cfg)
        intents, _ = plan_joint(dec.settled, jobs.pieces)
        errors = {p.pid: "no_placement" for p in jobs.pieces if p.pid not in intents}
        return JointDecision(
            intents=intents, latency_s=time.perf_counter() - t0, errors=errors
        )


class VLADictator:
    """ローカル VLA に全ピース分の割り当てを 1 回の推論で出させる。"""

    def __init__(self, backend: "Backend", include_text_board: bool = False) -> None:
        self.backend = backend
        self.include_text_board = include_text_board
        self.name = f"dictator-vla:{backend.name}"

    def decide_joint(self, jobs: JointObservation) -> JointDecision:
        text_board = None
        if self.include_text_board:
            text_board = to_text_board(decode(jobs.image, jobs.width, jobs.height, jobs.render_cfg))
        prompt = build_joint_prompt(jobs, text_board)
        try:
            raw, latency = self.backend.generate(prompt, jobs)
        except Exception as exc:
            return JointDecision(
                intents={}, latency_s=0.0, prompt=prompt,
                errors={p.pid: f"backend_error:{type(exc).__name__}" for p in jobs.pieces},
            )
        intents, errors = parse_joint_intents(raw, jobs.pieces, jobs.width)
        return JointDecision(
            intents=intents, latency_s=latency, raw=raw, prompt=prompt, errors=errors
        )


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, obs: Observation) -> tuple[str, float]: ...


class VLAAgent:
    """Vision-Language-Action ループ: 画像 -> 言語モデル -> 離散アクション意図。"""

    def __init__(self, backend: Backend, include_text_board: bool = False) -> None:
        self.backend = backend
        self.include_text_board = include_text_board
        self.name = f"vla:{backend.name}"

    def decide(self, obs: Observation) -> Decision:
        text_board = None
        if self.include_text_board:
            text_board = to_text_board(decode(obs.image, obs.width, obs.height, obs.render_cfg))
        prompt = build_prompt(obs, text_board)
        try:
            raw, latency = self.backend.generate(prompt, obs)
        except Exception as exc:  # バックエンド障害でゲームを止めない
            return Decision(intent=None, latency_s=0.0, prompt=prompt,
                            raw=None, error=f"backend_error:{type(exc).__name__}")
        intent, err = parse_intent(raw, obs.ego_kind, obs.width)
        return Decision(intent=intent, latency_s=latency, raw=raw, prompt=prompt, error=err)


# --------------------------------------------------------------------------
# バックエンド
# --------------------------------------------------------------------------


@dataclass
class MockConfig:
    """モデル無しで PoC を回すためのシミュレータ設定。

    「良い手だけ返すダミー」ではなく、実際の小型 VLM で起きる失敗モードを混ぜる。
    これが無いと fallback 経路が一度もテストされない。
    """

    skill: float = 0.55        # オラクル最良手を選ぶ確率
    rank_decay: float = 0.45   # 外した時に何位を選ぶかの幾何分布パラメータ
    p_malformed: float = 0.06  # JSON が壊れる
    p_refuse: float = 0.03     # 拒否 / 「分かりません」
    p_out_of_range: float = 0.05
    latency_s: float = 1.2     # 中央値レイテンシ (3B VLM on M2 の実測レンジ想定)
    latency_sigma: float = 0.35
    sleep: bool = False        # True にすると実時間でも待つ (既定は論理時間のみ)


class MockBackend:
    """ローカル VLA のスタブ。ピクセルから盤面を復元し、ノイズを載せて答える。

    「ピクセル経由のノイズ付きオラクル」という位置づけなので、
    skill を振ることで「モデルが賢くなったら何が起きるか」を先に測れる。
    """

    name = "mock"

    def __init__(self, cfg: MockConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or MockConfig()
        self.rng = random.Random(seed)

    def generate(self, prompt: str, obs: Observation | JointObservation) -> tuple[str, float]:
        if isinstance(obs, JointObservation):
            return self._generate_joint(obs)
        c = self.cfg
        latency = max(0.05, self.rng.lognormvariate(math.log(c.latency_s), c.latency_sigma))
        if c.sleep:
            time.sleep(latency)

        r = self.rng.random()
        if r < c.p_malformed:
            return ("Looking at the board, I think the piece should go around "
                    "the middle-left area. col: maybe 3?"), latency
        if r < c.p_malformed + c.p_refuse:
            return "I cannot reliably determine the board state from this image.", latency

        dec = decode(obs.image, obs.width, obs.height, obs.render_cfg)
        plans = plan_from_decoded(dec, obs.ego_kind, obs.persona)
        if not plans:
            return '{"col": 0, "rot": 0, "why": "no space"}', latency

        if self.rng.random() < c.skill:
            pick = plans[0]
        else:
            # 幾何分布で下位の手を選ぶ = 「惜しいが最良ではない」挙動
            idx = min(len(plans) - 1, int(self.rng.expovariate(c.rank_decay)))
            pick = plans[idx]

        col = pick.col
        if self.rng.random() < c.p_out_of_range:
            col = obs.width + self.rng.randint(0, 3)
        return json.dumps(
            {"col": col, "rot": pick.rot, "why": obs.persona.name}, ensure_ascii=False
        ), latency

    def _generate_joint(self, jobs: JointObservation) -> tuple[str, float]:
        """独裁モードのモック。1 回の推論で全員分を返す分、レイテンシは長めにする。"""
        c = self.cfg
        n = max(1, len(jobs.pieces))
        # 出力トークンが増えるぶんの伸びを素朴にモデル化 (プレフィルは共通なので線形未満)
        scale = 1.0 + 0.35 * (n - 1)
        latency = max(0.05, self.rng.lognormvariate(
            math.log(c.latency_s * scale), c.latency_sigma))
        if c.sleep:
            time.sleep(latency)

        r = self.rng.random()
        if r < c.p_malformed:
            return "The pieces should spread out across the board. moves: unclear", latency
        if r < c.p_malformed + c.p_refuse:
            return "I cannot assign all pieces reliably from this image.", latency

        dec = decode(jobs.image, jobs.width, jobs.height, jobs.render_cfg)
        ideal, _ = plan_joint(dec.settled, jobs.pieces)
        moves = []
        for info in jobs.pieces:
            best = ideal.get(info.pid)
            if best is None:
                continue
            if self.rng.random() < c.skill:
                col, rot = best.col, best.rot
            else:
                plans = plan_placements(dec.settled, info.kind, info.persona)
                if not plans:
                    continue
                idx = min(len(plans) - 1, int(self.rng.expovariate(c.rank_decay)))
                col, rot = plans[idx].col, plans[idx].rot
            if self.rng.random() < c.p_out_of_range:
                col = jobs.width + self.rng.randint(0, 3)
            moves.append({"pid": info.pid, "col": col, "rot": rot})
        return json.dumps({"moves": moves}, ensure_ascii=False), latency


class OllamaBackend:
    """実機バックエンド。ローカル Ollama の VLM を叩く。

    事前に:  ollama pull qwen2.5vl:3b
    (8GB 機なら 3B 級 / moondream 程度が現実的。7B 以上はスワップして遅くなる)
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: float = 120.0,
        num_predict: int = 96,
    ) -> None:
        import httpx  # 遅延 import: mock 実行時に依存させない

        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self._client = httpx.Client(timeout=timeout)
        self.name = f"ollama:{model}"

    def generate(self, prompt: str, obs: Observation) -> tuple[str, float]:
        import base64

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(obs.png).decode("ascii")],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        t0 = time.perf_counter()
        resp = self._client.post(f"{self.host}/api/generate", json=payload)
        resp.raise_for_status()
        latency = time.perf_counter() - t0
        return resp.json().get("response", ""), latency

    def health(self) -> str:
        r = self._client.get(f"{self.host}/api/tags")
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
            raise RuntimeError(
                f"モデル {self.model} が Ollama に見つかりません。"
                f"インストール済み: {names or '(なし)'}\n"
                f"  ollama pull {self.model}"
            )
        return f"ollama ok: {names}"


class ReplayBackend:
    """記録済み JSONL の生応答を再生する。実モデル実行を決定的に再現するため。"""

    name = "replay"

    def __init__(self, records: dict[str, tuple[str, float]]) -> None:
        self.records = records
        self.misses = 0

    @classmethod
    def from_jsonl(cls, path: str) -> "ReplayBackend":
        recs: dict[str, tuple[str, float]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("event") != "decision":
                    continue
                recs[f'{d["pid"]}:{d["submit_tick"]}'] = (d.get("raw") or "", d.get("latency_s", 0.0))
        return cls(recs)

    def generate(self, prompt: str, obs: Observation) -> tuple[str, float]:
        key = f"{obs.ego_pid}:{obs.tick}"
        if key not in self.records:
            self.misses += 1
            raise KeyError(f"replay に記録がありません: {key}")
        return self.records[key]
