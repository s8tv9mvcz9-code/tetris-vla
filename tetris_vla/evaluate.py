"""tetris_vla.evaluate — エージェント構成の比較実験。

PoC の主張は「小型ローカル VLA でテトリスが上手い」ではない (3B 級では消去数はほぼ 0 になる)。
測りたいのは **同じピクセルを与えたとき、VLA は探索ソルバの判断にどれだけ近づけるか**。
そこで主指標を `agreement_top1` / `mean_regret` に置き、ゲーム成績は副指標として併記する。

同時に、マルチエージェント特有の量 (stall / force_lock / staleness / queue_wait) を
必ず出力する。これが「お互いの挙動はスナップショットからしか分からない」ことの代償にあたる。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Iterable, Sequence

from .agents import (
    Agent,
    HeuristicAgent,
    HeuristicDictator,
    MockBackend,
    MockConfig,
    RandomAgent,
    VLAAgent,
    VLADictator,
)
from .engine import EngineConfig
from .runtime import EpisodeResult, RuntimeConfig, Session


@dataclass
class AgentSpec:
    """エージェント 1 構成の宣言。seed だけ差し替えて複数エピソード回す。"""

    label: str
    kind: str  # random | heuristic | vla-mock | vla-ollama | vla-replay
    options: dict = field(default_factory=dict)

    def build(self, seed: int) -> Agent:
        k = self.kind
        o = dict(self.options)
        if k == "random":
            return RandomAgent(seed=seed)
        if k == "heuristic":
            return HeuristicAgent(
                treat_active_as_obstacle=o.get("treat_active_as_obstacle", False),
                top_k=o.get("top_k", 1),
                seed=seed,
            )
        if k == "vla-mock":
            return VLAAgent(
                MockBackend(_mock_cfg(o), seed=seed),
                include_text_board=o.get("text_board", False),
            )
        if k == "vla-ollama":
            from .agents import OllamaBackend

            be = OllamaBackend(
                model=o.get("model", "qwen2.5vl:3b"),
                host=o.get("host", "http://localhost:11434"),
                temperature=o.get("temperature", 0.1),
            )
            return VLAAgent(be, include_text_board=o.get("text_board", False))
        if k == "vla-replay":
            from .agents import ReplayBackend

            return VLAAgent(
                ReplayBackend.from_jsonl(o["log"]), include_text_board=o.get("text_board", False)
            )
        if k == "dictator-heuristic":
            return HeuristicDictator()
        if k == "dictator-vla-mock":
            return VLADictator(
                MockBackend(_mock_cfg(o), seed=seed), include_text_board=o.get("text_board", False)
            )
        if k == "dictator-vla-ollama":
            from .agents import OllamaBackend

            return VLADictator(
                OllamaBackend(model=o.get("model", "qwen2.5vl:3b"),
                              host=o.get("host", "http://localhost:11434"),
                              temperature=o.get("temperature", 0.1)),
                include_text_board=o.get("text_board", False),
            )
        raise ValueError(f"未知の agent kind: {k}")


#: 独裁モードを持つ agent kind (Session が自動判別するが、CLI の検証にも使う)
DICTATOR_KINDS = ("dictator-heuristic", "dictator-vla-mock", "dictator-vla-ollama")
AGENT_KINDS = (
    "random", "heuristic", "vla-mock", "vla-ollama", "vla-replay", *DICTATOR_KINDS,
)


def _mock_cfg(o: dict) -> MockConfig:
    return MockConfig(
        skill=o.get("skill", 0.55),
        rank_decay=o.get("rank_decay", 0.45),
        p_malformed=o.get("p_malformed", 0.06),
        p_refuse=o.get("p_refuse", 0.03),
        p_out_of_range=o.get("p_out_of_range", 0.05),
        latency_s=o.get("latency_s", 1.2),
        latency_sigma=o.get("latency_sigma", 0.35),
        sleep=o.get("sleep", False),
    )


@dataclass
class Variant:
    """比較する 1 条件。エージェントだけでなく物理/知覚の設定も差し替えられる。"""

    label: str
    spec: AgentSpec
    engine: dict = field(default_factory=dict)   # EngineConfig の上書き
    runtime: dict = field(default_factory=dict)  # RuntimeConfig の上書き
    render: dict = field(default_factory=dict)   # RenderConfig の上書き


#: 既定の比較セット。VLA の腕前 (skill) を振って感度も見る。
DEFAULT_SUITE: tuple[Variant, ...] = (
    Variant("random", AgentSpec("random", "random")),
    Variant("heuristic(pixels)", AgentSpec("heuristic", "heuristic")),
    Variant("vla-mock skill=0.3", AgentSpec("vla", "vla-mock", {"skill": 0.3})),
    Variant("vla-mock skill=0.6", AgentSpec("vla", "vla-mock", {"skill": 0.6})),
    Variant("vla-mock skill=0.9", AgentSpec("vla", "vla-mock", {"skill": 0.9})),
)


def run_episode(
    variant: Variant,
    engine_cfg: EngineConfig,
    runtime_cfg: RuntimeConfig,
    seed: int,
) -> EpisodeResult:
    ecfg = replace(engine_cfg, seed=seed, **variant.engine)
    rcfg = replace(runtime_cfg, seed=seed, **variant.runtime)
    if variant.render:
        rcfg = replace(rcfg, render=replace(rcfg.render, **variant.render))
    res = Session(ecfg, rcfg, variant.spec.build(seed)).run()
    res.agent_name = variant.label
    return res


#: 集計対象の指標 (高い方が良い / 低い方が良い は report 側で注記)
METRIC_KEYS = (
    "lines_cleared",
    "lines_per_piece",
    "holes_per_piece",
    "settle_rate",
    "pieces_locked",
    "ticks",
    "final_holes",
    "final_max_height",
    "agreement_top1",
    "mean_regret",
    "mean_rank",
    "infeasible_rate",
    "mean_staleness",
    "mean_queue_wait",
    "mean_latency_s",
    "stall_ticks",
    "blocked_ticks",
    "simultaneity_conflicts",
    "force_locks",
    "decisions",
    "late_decisions",
    "undelivered_decisions",
    "dropped_requests",
    "locked_without_intent",
    "model_calls",
    "calls_per_piece",
)


@dataclass
class Aggregate:
    label: str
    n: int
    values: dict[str, tuple[float, float]]  # key -> (mean, stdev)
    errors: dict[str, int]
    reach_rate: float

    def get(self, key: str) -> tuple[float, float] | None:
        return self.values.get(key)


def aggregate(label: str, results: Sequence[EpisodeResult]) -> Aggregate:
    vals: dict[str, tuple[float, float]] = {}
    for key in METRIC_KEYS:
        nums = [getattr(r, key) for r in results if getattr(r, key) is not None]
        if not nums:
            continue
        m = statistics.fmean(nums)
        s = statistics.stdev(nums) if len(nums) > 1 else 0.0
        vals[key] = (m, s)
    errs: dict[str, int] = {}
    for r in results:
        for k, v in r.errors.items():
            errs[k] = errs.get(k, 0) + v
    locked = sum(r.pieces_locked for r in results) or 1
    reach = sum(r.settled_on_target for r in results) / locked
    return Aggregate(label=label, n=len(results), values=vals, errors=errs, reach_rate=reach)


def run_matrix(
    variants: Sequence[Variant],
    engine_cfg: EngineConfig,
    runtime_cfg: RuntimeConfig,
    seeds: Sequence[int],
    progress: Callable[[str], None] | None = None,
) -> list[Aggregate]:
    out: list[Aggregate] = []
    for v in variants:
        results = []
        for seed in seeds:
            if progress:
                progress(f"  {v.label} seed={seed} ...")
            results.append(run_episode(v, engine_cfg, runtime_cfg, seed))
        out.append(aggregate(v.label, results))
    return out


# --------------------------------------------------------------------------
# アブレーション研究 — PoC で本当に知りたい問いを 1 コマンドにしたもの
# --------------------------------------------------------------------------


def _heur() -> AgentSpec:
    return AgentSpec("heuristic", "heuristic")


def study_coordination() -> list[Variant]:
    """自律 / A2A / 独裁 の強弱。頭脳の質は同じで、意図の共有度だけが違う。

    - autonomous: 他人の位置しか見えない
    - a2a       : 他人の宣言した予定地がスナップショットにゴーストとして写る
    - dictator  : 1 エージェントが全ピースを一括で決める (共有コストゼロ)
    """
    return [
        Variant("自律 (通信なし)", _heur(), runtime={"coordination": "autonomous"}),
        Variant("A2A (予定地ゴースト)", _heur(),
                runtime={"coordination": "a2a"}, render={"show_ghosts": True}),
        Variant("A2A (ゴースト+テキスト)", _heur(),
                runtime={"coordination": "a2a", "a2a_text_channel": True},
                render={"show_ghosts": True}),
        Variant("独裁 (一括指揮)", AgentSpec("dictator", "dictator-heuristic")),
    ]


def study_coordination_vla() -> list[Variant]:
    """同じ比較を VLA (mock) で行う。推論コストの違いも同時に見える。"""
    opts = {"skill": 0.8}
    return [
        Variant("VLA 自律", AgentSpec("vla", "vla-mock", dict(opts)),
                runtime={"coordination": "autonomous"}),
        Variant("VLA A2A", AgentSpec("vla", "vla-mock", dict(opts)),
                runtime={"coordination": "a2a"}, render={"show_ghosts": True}),
        Variant("VLA 独裁", AgentSpec("vla", "dictator-vla-mock", dict(opts))),
    ]


def study_concurrency() -> list[Variant]:
    """同じ最適ソルバのまま同時落下数だけを振る。

    「意図が見えないこと」の純粋なコストを測る。エージェントの賢さは一定なので、
    劣化はすべて『他人の意図が盤面スナップショットに写らない』ことに由来する。
    """
    return [
        Variant(f"active={n}", _heur(), engine={"active_pieces": n})
        for n in (1, 2, 3, 4)
    ]


def study_perception() -> list[Variant]:
    """何が見えると得なのか。他ピースを障害物として扱う/種類を隠す。"""
    return [
        Variant("他ピースを無視して計画", _heur()),
        Variant("他ピースを障害物として計画",
                AgentSpec("heuristic", "heuristic", {"treat_active_as_obstacle": True})),
        Variant("霧: 種類が見えない", _heur(), render={"fog_other": True}),
        Variant("霧 + 障害物として計画",
                AgentSpec("heuristic", "heuristic", {"treat_active_as_obstacle": True}),
                render={"fog_other": True}),
    ]


def study_skill() -> list[Variant]:
    """VLA の腕前を振る。モデルが賢くなったら何が改善するかを先に見る。"""
    return [
        Variant(f"skill={s:.1f}", AgentSpec("vla", "vla-mock", {"skill": s}))
        for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]


def study_latency() -> list[Variant]:
    """推論レイテンシを振る。遅さは staleness と『間に合わない判断』に化ける。"""
    return [
        Variant(f"latency={l}s",
                AgentSpec("vla", "vla-mock", {"skill": 0.8, "latency_s": l}))
        for l in (0.1, 0.5, 1.2, 2.5, 5.0)
    ]


def study_robustness() -> list[Variant]:
    """出力の壊れ方を振る。JSON 崩れ/拒否がどれだけ効くか。"""
    base = {"skill": 0.8, "p_malformed": 0.0, "p_refuse": 0.0, "p_out_of_range": 0.0}
    return [
        Variant("崩れなし", AgentSpec("vla", "vla-mock", dict(base))),
        Variant("JSON崩れ 20%", AgentSpec("vla", "vla-mock", {**base, "p_malformed": 0.2})),
        Variant("拒否 20%", AgentSpec("vla", "vla-mock", {**base, "p_refuse": 0.2})),
        Variant("範囲外 20%", AgentSpec("vla", "vla-mock", {**base, "p_out_of_range": 0.2})),
        Variant("全部 10%ずつ",
                AgentSpec("vla", "vla-mock",
                          {**base, "p_malformed": 0.1, "p_refuse": 0.1, "p_out_of_range": 0.1})),
    ]


STUDIES: dict[str, tuple[str, Callable[[], list[Variant]]]] = {
    "coordination": ("協調モードの強弱 (自律 / A2A / 独裁)", study_coordination),
    "coordination-vla": ("協調モードの強弱 (VLA 版)", study_coordination_vla),
    "concurrency": ("同時落下数の影響 (意図が見えないことのコスト)", study_concurrency),
    "perception": ("知覚条件の影響 (他ピースをどう見るか)", study_perception),
    "skill": ("VLA の腕前の影響", study_skill),
    "latency": ("推論レイテンシの影響", study_latency),
    "robustness": ("出力破損への頑健性", study_robustness),
}


# --------------------------------------------------------------------------
# レポート
# --------------------------------------------------------------------------

_COLUMNS = [
    ("agreement_top1", "一致率↑", "{:.3f}"),
    ("mean_regret", "regret↓", "{:.3f}"),
    ("mean_rank", "順位↓", "{:.1f}"),
    ("infeasible_rate", "不能↓", "{:.3f}"),
    ("lines_cleared", "消去↑", "{:.1f}"),
    ("holes_per_piece", "穴/個↓", "{:.2f}"),
    ("pieces_locked", "設置", "{:.0f}"),
    ("final_max_height", "高さ↓", "{:.1f}"),
    ("stall_ticks", "停滞", "{:.0f}"),
    ("blocked_ticks", "進路阻害", "{:.0f}"),
    ("simultaneity_conflicts", "同時競合", "{:.0f}"),
    ("force_locks", "強制", "{:.1f}"),
    ("mean_staleness", "遅延t", "{:.1f}"),
    ("mean_queue_wait", "待ちt", "{:.1f}"),
    ("late_decisions", "遅着", "{:.1f}"),
    ("locked_without_intent", "無策", "{:.1f}"),
    ("calls_per_piece", "呼出/個", "{:.2f}"),
]


def to_markdown(aggs: Sequence[Aggregate]) -> str:
    head = "| agent | n | 収まり率↑ | " + " | ".join(c[1] for c in _COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 3)
    lines = [head, sep]
    for a in aggs:
        cells = [a.label, str(a.n), f"{a.reach_rate:.2f}"]
        for key, _, fmt in _COLUMNS:
            v = a.get(key)
            cells.append("-" if v is None else fmt.format(v[0]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("凡例: 一致率=Dellacherie最良手との top-1 一致 / regret=正規化スコア差(0が最良) / "
                 "順位=候補中の順位(0が最良) / 不能=置けない場所を指定した割合 / "
                 "収まり率=lock 時点で宣言どおりの(列,回転)だったピースの割合 / "
                 "遅延t=見た盤面の古さ(tick) / 待ちt=推論待ち行列の待機(tick) / "
                 "遅着=届いた時にはlock済みだった判断 / 無策=意図が一度も届かないままlockされたピース / "
                 "停滞=他ピースに阻まれて降りられなかったtick / 進路阻害=横移動を待たされたtick")
    err_lines = []
    for a in aggs:
        if a.errors:
            err_lines.append(f"- {a.label}: " + ", ".join(f"{k}={v}" for k, v in sorted(a.errors.items())))
    if err_lines:
        lines.append("")
        lines.append("パース/バックエンドのエラー内訳:")
        lines.extend(err_lines)
    return "\n".join(lines)


def to_json(aggs: Sequence[Aggregate]) -> str:
    return json.dumps([asdict(a) for a in aggs], ensure_ascii=False, indent=2, default=str)
