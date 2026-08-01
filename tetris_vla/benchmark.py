"""tetris_vla.benchmark — マシン性能ベンチマークとしてのテトリス。

## なぜこれがマシンのベンチマークになるのか

ゲームは **論理 tick** で進み、落下速度は設定値で固定されている (決められた時速)。
一方、エージェントの推論にかかる時間は **実測の wall-clock** で、
`service_ticks = round(latency_s / tick_seconds)` で tick に換算される。

つまり:

    速いマシン -> レイテンシが短い -> 消費する tick が少ない
              -> 判断が新鮮なうちに届く -> ピースが意図どおり収まる -> 高スコア

    遅いマシン -> 答えが届く頃には盤面が変わっている (staleness)
              -> 待ち行列が伸びる -> 間に合わない判断が増える -> 低スコア

ゲーム側は一切の忖度をしない。1 秒でも遅ければ、その分だけ盤面は先に進む。
`tick_seconds` が「このマシンに許される 1 手あたりの時間」を定義する。

## ハードル (tier)

tier が上がるほど、同時落下数が増え、落下が速くなり、1 tick の実時間が短くなる。
T2 なら 1 判断あたり 1.2 秒、T4 なら 0.4 秒しか猶予がない。
8GB の M2 で 3B 級 VLM を動かすと 1〜3 秒/推論なので、実機の合格ラインは T0〜T1 あたり。

## スコア

各 tier で「同じピース数を置き終えるまで」走らせ、4 つの軸を合成して 0〜1000 点にする。
tier を 1 つでも落とすと、そこから上は挑戦しない (逐次的なハードル)。
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Sequence

from .engine import EngineConfig
from .evaluate import AgentSpec
from .render import RenderConfig
from .runtime import EpisodeResult, RuntimeConfig, Session

#: 1 ピース (4 セル) が理論上稼げる最大ライン数。10 幅なら 4/10 = 0.4
MAX_LINES_PER_PIECE = 0.4
#: これ以上の穴/個は「壊滅」とみなす正規化の分母
HOLE_SATURATION = 1.5
#: この点数以上でその tier は合格
PASS_SCORE = 500.0


@dataclass(frozen=True)
class Tier:
    name: str
    label: str
    active_pieces: int
    fall_period: int
    tick_seconds: float
    replan_period: int
    spawn_interval: int
    target_pieces: int
    max_ticks: int

    @property
    def latency_budget_s(self) -> float:
        """1 判断に許される実時間。これを超えると判断が確実に陳腐化する。"""
        return self.replan_period * self.tick_seconds

    @property
    def piece_lifetime_s(self) -> float:
        """1 ピースが天井から床まで落ちるのに相当する実時間。"""
        return 20 * self.fall_period * self.tick_seconds


#: max_ticks は「目標ピース数を置き切るのに必要な tick」に余裕を持たせた値。
#: ここで打ち切られると survival が下がるので、能力ではなく設定不足で減点されないようにする。
TIERS: tuple[Tier, ...] = (
    Tier("T0", "warmup   単独・低速", 1, 8, 0.30, 8, 6, 15, 2600),
    Tier("T1", "easy     2体・やや低速", 2, 6, 0.25, 8, 5, 20, 1600),
    Tier("T2", "standard 3体・標準", 3, 4, 0.20, 6, 4, 25, 1200),
    Tier("T3", "hard     4体・高速", 4, 3, 0.15, 5, 3, 30, 900),
    Tier("T4", "extreme  5体・超高速", 5, 2, 0.10, 4, 2, 35, 700),
)

TIER_BY_NAME = {t.name: t for t in TIERS}


@dataclass
class Score:
    """0〜1000 点の内訳。どの軸で落としたかが分かるように分解して保持する。"""

    total: float
    survival: float     # 目標ピース数まで生き延びたか
    efficiency: float   # ライン効率 (理論上限比)
    cleanliness: float  # 穴の少なさ
    control: float      # 宣言どおりに収まれたか

    WEIGHTS = {"survival": 0.35, "efficiency": 0.30, "cleanliness": 0.20, "control": 0.15}

    @classmethod
    def of(cls, res: EpisodeResult, target_pieces: int) -> "Score":
        survival = min(1.0, res.pieces_locked / max(1, target_pieces))
        efficiency = min(1.0, res.lines_per_piece / MAX_LINES_PER_PIECE)
        cleanliness = max(0.0, 1.0 - res.holes_per_piece / HOLE_SATURATION)
        control = min(1.0, res.settle_rate)
        total = 1000.0 * (
            cls.WEIGHTS["survival"] * survival
            + cls.WEIGHTS["efficiency"] * efficiency
            + cls.WEIGHTS["cleanliness"] * cleanliness
            + cls.WEIGHTS["control"] * control
        )
        return cls(round(total, 1), round(survival, 3), round(efficiency, 3),
                   round(cleanliness, 3), round(control, 3))


@dataclass
class TierResult:
    tier: str
    label: str
    n_seeds: int
    score: float
    score_stdev: float
    breakdown: dict[str, float]
    passed: bool
    # マシン性能に直結する量
    mean_latency_s: float
    latency_budget_s: float
    latency_ratio: float      # 1.0 未満なら推論が落下に追いつけている
    mean_staleness: float
    mean_queue_wait: float
    wasted_call_rate: float   # 間に合わなかった / 届かなかった推論の割合
    # 参考
    model_calls: float
    calls_per_piece: float
    pieces_locked: float
    lines_cleared: float
    holes_per_piece: float
    game_over_rate: float


@dataclass
class BenchmarkResult:
    agent: str
    coordination: str
    tiers: list[TierResult]
    total_score: float
    highest_pass: str
    notes: list[str] = field(default_factory=list)


def _episode(
    tier: Tier,
    spec: AgentSpec,
    coordination: str,
    seed: int,
    render_cfg: RenderConfig,
    max_model_calls: int | None,
) -> EpisodeResult:
    ecfg = EngineConfig(
        active_pieces=tier.active_pieces,
        fall_period=tier.fall_period,
        fall_period_jitter=1,
        spawn_interval=tier.spawn_interval,
        seed=seed,
    )
    rcfg = RuntimeConfig(
        tick_seconds=tier.tick_seconds,
        replan_period=tier.replan_period,
        decision_budget_per_piece=8,
        inference_workers=1,
        max_ticks=tier.max_ticks,
        stop_after_pieces=tier.target_pieces,
        compute_oracle=False,   # ベンチマークは結果指標のみ。計測オーバーヘッドを避ける
        coordination=coordination,
        max_model_calls=max_model_calls,
        render=render_cfg,
        seed=seed,
    )
    return Session(ecfg, rcfg, spec.build(seed)).run()


def run_tier(
    tier: Tier,
    spec: AgentSpec,
    coordination: str = "autonomous",
    seeds: Sequence[int] = (0, 1, 2),
    max_model_calls: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> TierResult:
    render_cfg = RenderConfig(show_ghosts=(coordination == "a2a"))
    results: list[EpisodeResult] = []
    scores: list[Score] = []
    for seed in seeds:
        if progress:
            progress(f"  {tier.name} seed={seed} ...")
        res = _episode(tier, spec, coordination, seed, render_cfg, max_model_calls)
        results.append(res)
        scores.append(Score.of(res, tier.target_pieces))

    def m(key: str) -> float:
        return statistics.fmean(getattr(r, key) for r in results)

    totals = [s.total for s in scores]
    mean_lat = m("mean_latency_s")
    wasted = sum(r.late_decisions + r.undelivered_decisions for r in results)
    calls = sum(r.model_calls for r in results) or 1
    return TierResult(
        tier=tier.name,
        label=tier.label,
        n_seeds=len(seeds),
        score=round(statistics.fmean(totals), 1),
        score_stdev=round(statistics.stdev(totals), 1) if len(totals) > 1 else 0.0,
        breakdown={
            "survival": round(statistics.fmean(s.survival for s in scores), 3),
            "efficiency": round(statistics.fmean(s.efficiency for s in scores), 3),
            "cleanliness": round(statistics.fmean(s.cleanliness for s in scores), 3),
            "control": round(statistics.fmean(s.control for s in scores), 3),
        },
        passed=statistics.fmean(totals) >= PASS_SCORE,
        mean_latency_s=round(mean_lat, 4),
        latency_budget_s=round(tier.latency_budget_s, 3),
        latency_ratio=round(mean_lat / tier.latency_budget_s, 3),
        mean_staleness=round(m("mean_staleness"), 2),
        mean_queue_wait=round(m("mean_queue_wait"), 2),
        wasted_call_rate=round(wasted / calls, 3),
        model_calls=round(m("model_calls"), 1),
        calls_per_piece=round(m("calls_per_piece"), 2),
        pieces_locked=round(m("pieces_locked"), 1),
        lines_cleared=round(m("lines_cleared"), 1),
        holes_per_piece=round(m("holes_per_piece"), 2),
        game_over_rate=round(sum(1 for r in results if r.game_over) / len(results), 2),
    )


def run_benchmark(
    spec: AgentSpec,
    coordination: str = "autonomous",
    seeds: Sequence[int] = (0, 1, 2),
    tiers: Sequence[Tier] = TIERS,
    stop_on_fail: bool = True,
    max_model_calls: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    out: list[TierResult] = []
    notes: list[str] = []
    highest = "-"
    for tier in tiers:
        if progress:
            progress(f"{tier.name} {tier.label} (許容 {tier.latency_budget_s:.2f}s/判断)")
        tr = run_tier(tier, spec, coordination, seeds, max_model_calls, progress)
        out.append(tr)
        if tr.passed:
            highest = tier.name
        elif stop_on_fail:
            notes.append(f"{tier.name} で不合格 ({tr.score:.0f} < {PASS_SCORE:.0f}) のため以降は未挑戦")
            break
    return BenchmarkResult(
        agent=spec.label,
        coordination=coordination,
        tiers=out,
        total_score=round(sum(t.score for t in out), 1),
        highest_pass=highest,
        notes=notes,
    )


# --------------------------------------------------------------------------
# レポート
# --------------------------------------------------------------------------


def to_markdown(bench: BenchmarkResult) -> str:
    lines = [
        f"## ベンチマーク結果: {bench.agent} / 協調={bench.coordination}",
        "",
        f"- **合格した最高 tier: {bench.highest_pass}**",
        f"- 合計スコア: {bench.total_score:.0f}",
        "",
        "| tier | 条件 | スコア | ±σ | 判定 | 生存 | 効率 | 無穴 | 制御 | "
        "実測lat | 許容lat | 比率 | 陳腐化t | 待ちt | 無駄率 | 呼出/個 |",
        "|" + "---|" * 16,
    ]
    for t in bench.tiers:
        b = t.breakdown
        lines.append(
            f"| {t.tier} | {t.label} | {t.score:.0f} | {t.score_stdev:.0f} | "
            f"{'合格' if t.passed else '不合格'} | "
            f"{b['survival']:.2f} | {b['efficiency']:.2f} | {b['cleanliness']:.2f} | "
            f"{b['control']:.2f} | {t.mean_latency_s:.3f}s | {t.latency_budget_s:.2f}s | "
            f"{t.latency_ratio:.2f} | {t.mean_staleness:.1f} | {t.mean_queue_wait:.1f} | "
            f"{t.wasted_call_rate:.2f} | {t.calls_per_piece:.2f} |"
        )
    lines += [
        "",
        "凡例:",
        "- **比率** = 実測レイテンシ / 許容レイテンシ。**1.0 未満なら推論が落下に追いつけている**。"
        "ここがマシン性能を最も直接に表す。",
        "- **陳腐化t** = 判断が届いた時点でスナップショットが何 tick 古かったか。",
        "- **無駄率** = 間に合わなかった + 届かなかった推論 / 全推論。遅いマシンほど上がる。",
        "- スコア内訳: 生存(0.35) 効率(0.30) 無穴(0.20) 制御(0.15)。",
        f"- 合格ライン {PASS_SCORE:.0f} 点。1 つ落とすとそれ以上の tier は未挑戦。",
    ]
    lines.append("")
    lines.append(f"**律速要因の診断**: {diagnose(bench)}")
    if bench.notes:
        lines.append("")
        lines.extend(f"- {n}" for n in bench.notes)
    return "\n".join(lines)


def diagnose(bench: BenchmarkResult) -> str:
    """スコアを落としている原因が「マシンの遅さ」か「協調の難しさ」かを判定する。

    このベンチは 2 つのレジームを持つ。どちらにいるかを明示しないと、
    速いエージェントの低スコアを「マシンが遅い」と誤読してしまう。
    """
    if not bench.tiers:
        return "データなし"
    worst = min(bench.tiers, key=lambda t: t.score)
    ratio = max(t.latency_ratio for t in bench.tiers)
    if ratio >= 1.0:
        return (
            f"**推論律速**。{worst.tier} でレイテンシ比 {worst.latency_ratio:.2f} "
            f"(実測 {worst.mean_latency_s:.2f}s / 許容 {worst.latency_budget_s:.2f}s)。"
            f"判断が届く頃には盤面が {worst.mean_staleness:.1f} tick 進んでいる。"
            "モデルを小さくするか tick_seconds を伸ばすとスコアが動く = マシン性能が効いている。"
        )
    if ratio >= 0.3:
        return (
            f"**推論と協調が拮抗**。最大レイテンシ比 {ratio:.2f}。"
            "マシンを速くしても、協調不全ぶんの減点は残る。"
        )
    clean = min(t.breakdown["cleanliness"] for t in bench.tiers)
    cause = {
        "autonomous": "各ピースの判断が個別には最適でも互いに衝突しているため",
        "a2a": "宣言の共有だけでは、宣言が古くなる速度に追いつけないため",
        "dictator": "一括指揮でも、計画時点では他ピースの最終着地が未確定なため",
    }.get(bench.coordination, "同時落下そのものの難しさによる")
    return (
        f"**協調律速**。最大レイテンシ比 {ratio:.2f} で推論は落下に十分追いついている。"
        f"それでも無穴スコアが {clean:.2f} まで落ちるのは、{cause}。"
        "マシンを速くしてもここは改善しない — 協調の設計を変える必要がある。"
    )


def compare_modes_markdown(results: Sequence[BenchmarkResult]) -> str:
    """協調モード間の強弱比較表。"""
    lines = [
        "## 協調モード比較",
        "",
        "| 協調 | エージェント | 最高tier | 合計 | " + " | ".join(t.name for t in TIERS) + " |",
        "|" + "---|" * (4 + len(TIERS)),
    ]
    for b in results:
        by_tier = {t.tier: t for t in b.tiers}
        cells = [
            f"{by_tier[t.name].score:.0f}" if t.name in by_tier else "-" for t in TIERS
        ]
        lines.append(
            f"| {b.coordination} | {b.agent} | {b.highest_pass} | {b.total_score:.0f} | "
            + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "- **autonomous**: 各ピースが独立に判断。他人の *位置* しか見えない。",
        "- **a2a**: 各ピースが宣言した予定地がスナップショットにゴーストとして描かれる。"
        "他人の *意図* が見えるようになる。通信は画像の中だけで完結する。",
        "- **dictator**: 1 エージェントが全ピースを 1 回の推論で一括指揮。"
        "意図の共有コストがゼロになる代わりに、推論 1 回の出力が重くなる。",
    ]
    return "\n".join(lines)


def to_json(bench: BenchmarkResult) -> dict:
    return asdict(bench)
