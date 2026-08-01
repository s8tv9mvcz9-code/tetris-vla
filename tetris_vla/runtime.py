"""tetris_vla.runtime — 論理 tick クロック、推論キュー、反射コントローラ、ロガー。

設計の中心にある 3 つの決定:

1. **時間は論理 tick で進む。** 落下速度は「決められた時速」= 設定値であって、
   VLM の返答が遅いかどうかとは無関係。実レイテンシは
   `service_ticks = round(latency_s / tick_seconds)` で tick に換算され、
   「答えが返ってくる頃には盤面が古い」という遅延として *ゲーム内で* 表現される。

2. **推論器は 1 台。** 8GB / 1 モデルインスタンスという現実に合わせて、
   推論は単一サーバ待ち行列。3 個のピースが同時に判断を求めたら 2 個は待つ。
   待ち時間も staleness としてそのまま計測される。

3. **エージェントは操舵しかできない。** 重力は交渉不能。
   意図 (col, rot) を反射コントローラが 1 tick 1 アクションに落とす。

観測の遮断: エージェントに渡すのは `Observation` (PNG + 自己情報) だけ。
Engine への参照は一切渡らない。
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

import numpy as np

from .agents import (
    Agent,
    Decision,
    Intent,
    JointDecision,
    JointObservation,
    Persona,
    PERSONAS,
    DEFAULT_PERSONA_CYCLE,
    Observation,
    PieceInfo,
    Placement,
    plan_placements,
    persona_for,
)
from .engine import (
    ROT_CELLS,
    Action,
    ActivePiece,
    Engine,
    EngineConfig,
    aggregate_height,
    bumpiness,
    cell_span,
    column_heights,
    count_holes,
    drop_row,
)
from .render import RenderConfig, decode, render, sha1, to_ascii, to_png

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    tick_seconds: float = 0.15        # 1 論理 tick の実時間換算 (レイテンシ→tick 変換に使う)
    replan_period: int = 6            # 同じピースが再計画するまでの最小 tick 数
    decision_budget_per_piece: int = 6
    inference_workers: int = 1        # ローカル VLA は基本 1
    max_service_ticks: int = 40       # 1 回の推論が消費する tick の上限 (暴走防止)
    blocked_replan_threshold: int = 3 # 目標に到達できない時に再計画するまでの失敗回数
    max_ticks: int = 600
    #: N 個設置したら打ち切る。ベンチマークで条件間の「試行回数」を揃えるために使う
    stop_after_pieces: int | None = None
    compute_oracle: bool = True       # 意図一致率メトリクスを取るか
    #: 協調モード: autonomous (通信なし) / a2a (予定地をスナップショットに書き込む) /
    #: dictator (1 エージェントが全ピースを一括指揮; agent が decide_joint を持てば自動)
    coordination: str = "autonomous"
    #: A2A でゴースト描画に加えてテキストの宣言も渡すか (視覚 vs 言語の切り分け用)
    a2a_text_channel: bool = False
    #: エピソード全体の推論呼び出し上限。モード間で推論コストを揃える時に使う
    max_model_calls: int | None = None
    realtime: bool = False
    ascii_every: int = 0              # >0 で N tick ごとに ASCII を stdout へ
    dump_frames: str | None = None
    log_path: str | None = None
    personas: tuple[str, ...] = DEFAULT_PERSONA_CYCLE
    render: RenderConfig = field(default_factory=RenderConfig)
    seed: int = 0


# --------------------------------------------------------------------------
# 反射コントローラ: 意図 -> 1 tick 分のアクション
# --------------------------------------------------------------------------


def desired_action(piece: ActivePiece, intent: Intent | None) -> Action:
    """意図と現在姿勢の差から「やりたいこと」を 1 つ選ぶ。回転を先に済ませる。

    落下そのものは操作できない (= 決められた時速で降り続ける)。
    """
    if intent is None:
        return Action.NONE
    diff = (intent.rot - piece.rot) % 4
    if diff == 3:
        return Action.ROT_CCW
    if diff != 0:
        return Action.ROT_CW
    if piece.left_col < intent.col:
        return Action.RIGHT
    if piece.left_col > intent.col:
        return Action.LEFT
    return Action.NONE


def reflex(
    piece: ActivePiece,
    intent: Intent | None,
    legal: Callable[[Action], bool] | None = None,
) -> tuple[Action, bool]:
    """(実行するアクション, 進路を塞がれたか) を返す。

    `legal` は「今その方向へ動けるか」を答える身体感覚。塞がれている時は
    無駄に突っ込まず待つ。待たされ続けたら上位に再計画を要求する。
    """
    want = desired_action(piece, intent)
    if want == Action.NONE or legal is None:
        return want, False
    if legal(want):
        return want, False
    # 180 度回したいだけなら逆回転で回り込めることがある
    if intent is not None and (intent.rot - piece.rot) % 4 == 2:
        alt = Action.ROT_CCW if want == Action.ROT_CW else Action.ROT_CW
        if legal(alt):
            return alt, False
    return Action.NONE, True


def on_target(piece: ActivePiece, intent: Intent | None) -> bool:  # noqa: D401
    return intent is not None and piece.rot == intent.rot and piece.left_col == intent.col


# --------------------------------------------------------------------------
# 内部状態
# --------------------------------------------------------------------------


@dataclass
class Request:
    """推論 1 回分。自律モードは 1 ピース、独裁モードは全ピースをまとめて 1 件。"""

    pids: tuple[int, ...]
    submit_tick: int
    snapshot_sha: str
    obs: Observation | None = None
    jobs: JointObservation | None = None
    oracle: dict[int, list[Placement]] = field(default_factory=dict)
    start_tick: int | None = None
    deliver_tick: int | None = None
    decision: Decision | JointDecision | None = None

    @property
    def is_joint(self) -> bool:
        return self.jobs is not None


@dataclass
class PieceState:
    pid: int
    kind: str
    persona: Persona
    intent: Intent | None = None
    intent_submit_tick: int = -1
    intent_deliver_tick: int = -1
    decisions: int = 0
    pending: bool = False
    blocked: int = 0
    ever_on_target: bool = False
    spawn_tick: int = 0


@dataclass
class DecisionRecord:
    tick: int
    pid: int
    kind: str
    persona: str
    submit_tick: int
    start_tick: int
    deliver_tick: int
    staleness: int          # deliver - submit (エージェントが見た盤面の古さ)
    queue_wait: int         # start - submit (推論器の混雑による待ち)
    latency_s: float
    service_ticks: int
    snapshot_sha: str
    intent_col: int | None
    intent_rot: int | None
    why: str
    error: str | None
    oracle_col: int | None = None
    oracle_rot: int | None = None
    agree: bool | None = None       # persona オラクルの最良手と一致したか
    rank: int | None = None         # 何番目に良い手だったか (0 が最良)
    regret: float | None = None     # 正規化スコア差 [0,1]、0 が最良
    infeasible: bool | None = None  # そもそも置けない場所を指定した
    #: 推論は走ったが、答えがエピソード終了までに届かなかった
    delivered: bool = True
    #: 答えは届いたが、その時点でピースは既に lock 済みだった (間に合わなかった)
    too_late: bool = False
    #: 独裁モードで 1 回の推論から生まれたレコードか
    joint: bool = False
    raw: str | None = None


@dataclass
class EpisodeResult:
    # ゲーム成績
    ticks: int = 0
    lines_cleared: int = 0
    line_histogram: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    pieces_locked: int = 0
    game_over: bool = False
    final_holes: int = 0
    final_bumpiness: int = 0
    final_max_height: int = 0
    final_aggregate_height: int = 0
    # マルチエージェント干渉
    stall_ticks: int = 0        # 他の落下ピースに阻まれて降りられなかった tick
    force_locks: int = 0        # デッドロック回避のため強制 lock された回数
    #: 進みたい方向が他ピースに塞がれて待たされた tick (水平方向の干渉の強さ)
    blocked_ticks: int = 0
    #: 発行時点では合法だったのに、同 tick で先に動いた他ピースに割り込まれて
    #: 無効化されたアクション数 (同時手番の競合。同時落下モード特有)
    simultaneity_conflicts: int = 0
    actions_issued: int = 0
    # 意思の実行率
    #: lock した瞬間に、自分が宣言した (列, 回転) と一致していたピース数
    settled_on_target: int = 0
    #: 生涯で一度でも目標姿勢に到達したピース数 (到達したが押し戻された分も含む)
    ever_on_target: int = 0
    #: 意図を一度も受け取れないまま lock されたピース数
    locked_without_intent: int = 0
    # 推論
    #: 実際にモデルを呼んだ回数。独裁モードでは 1 回で全ピース分を賄うので少なくなる
    model_calls: int = 0
    calls_per_piece: float = 0.0
    coordination: str = "autonomous"
    decisions: int = 0            # 答えが届いて実際に使われた判断
    late_decisions: int = 0       # 届いた時にはピースが lock 済みだった判断
    undelivered_decisions: int = 0  # 推論は走ったが終了までに届かなかった
    dropped_requests: int = 0     # 推論開始前にピースが lock され破棄された要求
    mean_latency_s: float = 0.0
    mean_staleness: float = 0.0
    mean_queue_wait: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)
    # VLA の質
    agreement_top1: float | None = None
    mean_rank: float | None = None
    mean_regret: float | None = None
    infeasible_rate: float | None = None
    # ピース数で正規化した量 (active_pieces を振ると設置数自体が変わるため)
    lines_per_piece: float = 0.0
    holes_per_piece: float = 0.0
    settle_rate: float = 0.0
    # メタ
    agent_name: str = ""
    config: dict = field(default_factory=dict)
    records: list[DecisionRecord] = field(default_factory=list)

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "records"}
        return d


# --------------------------------------------------------------------------
# セッション
# --------------------------------------------------------------------------


class Session:
    """1 エピソードを回す。エンジン / スケジューラ / ロガーの結線。"""

    def __init__(
        self,
        engine_cfg: EngineConfig,
        runtime_cfg: RuntimeConfig,
        agent: Agent,
    ) -> None:
        self.ecfg = engine_cfg
        self.rcfg = runtime_cfg
        self.agent = agent
        self.engine = Engine(engine_cfg)
        self.rcfg.render = runtime_cfg.render

        self.states: dict[int, PieceState] = {}
        self.queue: deque[Request] = deque()
        self.in_flight: list[Request] = []
        self.worker_free_at = [0] * max(1, runtime_cfg.inference_workers)
        self.records: list[DecisionRecord] = []
        self.dropped_requests = 0
        self.model_calls = 0
        self._oracle_cache: dict[tuple[bytes, str, str], list[Placement]] = {}
        self._log_fh = None
        self._persona_idx = 0
        self._frame_no = 0

        # 協調モード: agent が decide_joint を持っていれば独裁モードとみなす
        self.is_dictator = hasattr(agent, "decide_joint")
        self.mode = "dictator" if self.is_dictator else runtime_cfg.coordination
        if self.mode not in ("autonomous", "a2a", "dictator"):
            raise ValueError(f"未知の coordination: {self.mode}")
        if self.mode == "dictator" and not self.is_dictator:
            raise ValueError("coordination=dictator には decide_joint を持つエージェントが必要です")
        #: A2A で各ピースが公開している「自分はここに収まる」宣言
        self.declared: dict[int, Intent] = {}
        self._last_joint_submit = -10**9

    # -- ロギング ----------------------------------------------------------

    def _open_log(self) -> None:
        if self.rcfg.log_path:
            self._log_fh = open(self.rcfg.log_path, "w", encoding="utf-8")

    def _log(self, obj: dict) -> None:
        if self._log_fh is not None:
            self._log_fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def _close_log(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    # -- オラクル ----------------------------------------------------------

    def _oracle(self, settled: np.ndarray, kind: str, persona: Persona) -> list[Placement]:
        key = (settled.tobytes(), kind, persona.name)
        hit = self._oracle_cache.get(key)
        if hit is None:
            hit = plan_placements(settled, kind, persona)
            if len(self._oracle_cache) > 4096:
                self._oracle_cache.clear()
            self._oracle_cache[key] = hit
        return hit

    @staticmethod
    def _score_intent(plans: list[Placement], intent: Intent | None) -> tuple[int | None, float | None, bool]:
        """意図が候補リストの何番目か、正規化 regret、実行不能かを返す。"""
        if intent is None or not plans:
            return None, None, True
        best = plans[0].score
        worst = plans[-1].score
        span = max(1e-9, best - worst)
        for i, p in enumerate(plans):
            if p.col == intent.col and p.rot == intent.rot:
                return i, float((best - p.score) / span), False
        return None, 1.0, True

    # -- 観測の生成 --------------------------------------------------------

    def _ghost_cells(self, exclude_pid: int | None) -> list[tuple[int, int]]:
        """A2A で公開されている宣言を「予定地セル」に変換する。

        宣言 (col, rot) から確定地形へのハードドロップ位置を計算して描く。
        これはランタイム側 (=通信基盤) の仕事であり、エージェントは
        あくまで描かれた結果をピクセルとして見るだけ。
        """
        if self.mode != "a2a":
            return []
        cells: list[tuple[int, int]] = []
        for pid, intent in self.declared.items():
            if pid == exclude_pid:
                continue
            p = self.engine.piece(pid)
            if p is None:
                continue
            rot = intent.rot % 4
            x = intent.col - cell_span(p.kind, rot)[2]
            y = drop_row(self.engine.board, p.kind, rot, x)
            if y is None:
                continue
            cells.extend((y + dy, x + dx) for dy, dx in ROT_CELLS[p.kind][rot])
        return cells

    def _declarations(self, exclude_pid: int | None) -> tuple[str, ...]:
        if self.mode != "a2a" or not self.rcfg.a2a_text_channel:
            return ()
        out = []
        for pid, intent in sorted(self.declared.items()):
            if pid == exclude_pid:
                continue
            p = self.engine.piece(pid)
            if p is None:
                continue
            out.append(f"#{pid} ({p.kind}) は 列{intent.col} 回転{intent.rot} に収まると宣言")
        return tuple(out)

    def _dump(self, img, tick: int, tag: str) -> None:
        if not self.rcfg.dump_frames:
            return
        import os

        os.makedirs(self.rcfg.dump_frames, exist_ok=True)
        img.save(f"{self.rcfg.dump_frames}/t{tick:05d}_{tag}.png")
        self._frame_no += 1

    def _observe(self, piece: ActivePiece, persona: Persona) -> tuple[Observation, str]:
        snap = self.engine.snapshot()
        img = render(
            snap, ego_pid=piece.pid, cfg=self.rcfg.render,
            ghost_cells=self._ghost_cells(piece.pid),
        )
        png = to_png(img)
        obs = Observation(
            png=png,
            image=img,
            ego_pid=piece.pid,
            ego_kind=piece.kind,
            persona=persona,
            tick=snap.tick,
            width=snap.width,
            height=snap.height,
            fall_period=piece.fall_period,
            render_cfg=self.rcfg.render,
            coordination=self.mode,
            declarations=self._declarations(piece.pid),
        )
        self._dump(img, snap.tick, f"p{piece.pid}")
        return obs, sha1(png)

    def _observe_joint(self) -> tuple[JointObservation, str]:
        """独裁モードの観測: 自機リング無しの 1 枚 + 全ピースの自己情報。

        自律モードでは各ピースが「自分がどれか」を白リングで知る。独裁者は全ピースを
        所有するので、同じ自己知識を名簿として一度に受け取る。盤面そのものは
        あくまでピクセルからしか読めない。
        """
        snap = self.engine.snapshot()
        img = render(snap, ego_pid=None, cfg=self.rcfg.render)
        png = to_png(img)
        infos = []
        for p in sorted(self.engine.active, key=lambda q: q.pid):
            st = self.states.get(p.pid)
            infos.append(
                PieceInfo(
                    pid=p.pid,
                    kind=p.kind,
                    persona=st.persona if st else PERSONAS["neutral"],
                    fall_period=p.fall_period,
                    left_col=p.left_col,
                    top_row=min(r for r, _ in p.cells()),
                )
            )
        jobs = JointObservation(
            png=png,
            image=img,
            pieces=tuple(infos),
            tick=snap.tick,
            width=snap.width,
            height=snap.height,
            render_cfg=self.rcfg.render,
        )
        self._dump(img, snap.tick, "joint")
        return jobs, sha1(png)

    # -- スケジューラ ------------------------------------------------------

    def _ensure_state(self, piece: ActivePiece, tick: int) -> PieceState:
        st = self.states.get(piece.pid)
        if st is None:
            st = PieceState(
                pid=piece.pid,
                kind=piece.kind,
                persona=self._assign_persona(),
                spawn_tick=tick,
            )
            self.states[piece.pid] = st
            piece.persona = st.persona.name
        return st

    def _budget_left(self) -> bool:
        cap = self.rcfg.max_model_calls
        pending = len(self.queue) + len(self.in_flight)
        return cap is None or (self.model_calls + pending) < cap

    def _submit(self, tick: int) -> None:
        if self.mode == "dictator":
            self._submit_joint(tick)
            return
        r = self.rcfg
        for piece in sorted(self.engine.active, key=lambda p: p.pid):
            st = self._ensure_state(piece, tick)
            if st.pending or st.decisions >= r.decision_budget_per_piece:
                continue
            if not self._budget_left():
                continue
            need = (
                st.intent is None
                or st.blocked >= r.blocked_replan_threshold
                or (tick - st.intent_submit_tick) >= r.replan_period
            )
            if not need:
                continue
            obs, sha = self._observe(piece, st.persona)
            oracle: dict[int, list[Placement]] = {}
            if r.compute_oracle:
                dec = decode(obs.image, obs.width, obs.height, r.render)
                oracle[piece.pid] = self._oracle(dec.settled, piece.kind, st.persona)
            self.queue.append(
                Request(pids=(piece.pid,), submit_tick=tick, obs=obs,
                        snapshot_sha=sha, oracle=oracle)
            )
            st.pending = True
            st.blocked = 0

    def _submit_joint(self, tick: int) -> None:
        """独裁モード: 全ピース分をまとめて 1 件だけ投入する。"""
        r = self.rcfg
        if not self.engine.active or not self._budget_left():
            return
        if any(req.is_joint for req in list(self.queue) + self.in_flight):
            return
        for piece in sorted(self.engine.active, key=lambda p: p.pid):
            self._ensure_state(piece, tick)
        need = any(
            self.states[p.pid].intent is None
            or self.states[p.pid].blocked >= r.blocked_replan_threshold
            for p in self.engine.active
        ) or (tick - self._last_joint_submit) >= r.replan_period
        if not need:
            return
        jobs, sha = self._observe_joint()
        oracle: dict[int, list[Placement]] = {}
        if r.compute_oracle:
            dec = decode(jobs.image, jobs.width, jobs.height, r.render)
            for info in jobs.pieces:
                oracle[info.pid] = self._oracle(dec.settled, info.kind, info.persona)
        self.queue.append(
            Request(pids=tuple(i.pid for i in jobs.pieces), submit_tick=tick,
                    jobs=jobs, snapshot_sha=sha, oracle=oracle)
        )
        self._last_joint_submit = tick
        for info in jobs.pieces:
            st = self.states[info.pid]
            st.pending = True
            st.blocked = 0

    def _assign_persona(self) -> Persona:
        names = self.rcfg.personas
        p = PERSONAS[names[self._persona_idx % len(names)]]
        self._persona_idx += 1
        return p

    def _start_services(self, tick: int) -> None:
        """空いている推論器に仕事を渡す。ここで実際にモデルを呼ぶ (ブロッキング)。"""
        for w in range(len(self.worker_free_at)):
            while self.worker_free_at[w] <= tick and self.queue:
                req = self.queue.popleft()
                # ピースが既に lock されていたら破棄 (推論コストは払っていない)
                if all(self.engine.piece(pid) is None for pid in req.pids):
                    for pid in req.pids:
                        st = self.states.get(pid)
                        if st:
                            st.pending = False
                    self.dropped_requests += 1
                    continue
                if req.is_joint:
                    decision = self.agent.decide_joint(req.jobs)
                else:
                    decision = self.agent.decide(req.obs)
                self.model_calls += 1
                service = int(round(decision.latency_s / max(1e-6, self.rcfg.tick_seconds)))
                service = max(0, min(self.rcfg.max_service_ticks, service))
                req.start_tick = tick
                req.deliver_tick = tick + service
                req.decision = decision
                self.worker_free_at[w] = tick + max(1, service)
                self.in_flight.append(req)
                break  # このワーカーは埋まった

    def _deliver(self, tick: int) -> None:
        still: list[Request] = []
        for req in self.in_flight:
            if req.deliver_tick is None or req.deliver_tick > tick:
                still.append(req)
                continue
            self._apply(req, tick)
        self.in_flight = still

    def _apply(self, req: Request, tick: int, delivered: bool = True) -> None:
        """判断を反映して記録する。自律モードは 1 件、独裁モードは全ピース分。

        delivered=False は「推論は走ったがエピソード終了までに答えが届かなかった」。
        遅い VLA ではこれが無視できない割合になるので、指標として必ず残す。
        """
        d = req.decision
        assert d is not None
        joint = req.is_joint
        for pid in req.pids:
            if joint:
                intent = d.intents.get(pid)
                error = d.errors.get(pid)
                kind = next((i.kind for i in req.jobs.pieces if i.pid == pid), "?")
                persona = next(
                    (i.persona.name for i in req.jobs.pieces if i.pid == pid), "?"
                )
            else:
                intent, error = d.intent, d.error
                kind, persona = req.obs.ego_kind, req.obs.persona.name
            self._apply_one(req, pid, intent, error, kind, persona, tick, delivered, joint, d)

    def _apply_one(self, req, pid, intent, error, kind, persona, tick,
                   delivered, joint, d) -> None:
        st = self.states.get(pid)
        too_late = self.engine.piece(pid) is None
        if st is not None:
            st.pending = False
            if delivered and not too_late:
                st.decisions += 1
                if intent is not None:
                    st.intent = intent
                    st.intent_submit_tick = req.submit_tick
                    st.intent_deliver_tick = tick
                    # A2A: 受け取った意図をそのまま「宣言」として公開する
                    if self.mode == "a2a":
                        self.declared[pid] = intent

        rank = regret = None
        infeasible = None
        agree = None
        ocol = orot = None
        plans = req.oracle.get(pid)
        if plans:
            ocol, orot = plans[0].col, plans[0].rot
            rank, regret, infeasible = self._score_intent(plans, intent)
            agree = intent is not None and (intent.col, intent.rot) == (ocol, orot)

        rec = DecisionRecord(
            tick=tick,
            pid=pid,
            kind=kind,
            persona=persona,
            submit_tick=req.submit_tick,
            start_tick=int(req.start_tick if req.start_tick is not None else tick),
            deliver_tick=tick if delivered else -1,
            staleness=(tick - req.submit_tick) if delivered else -1,
            queue_wait=int((req.start_tick if req.start_tick is not None else tick) - req.submit_tick),
            latency_s=d.latency_s,
            service_ticks=int((req.deliver_tick or tick) - (req.start_tick or tick)),
            snapshot_sha=req.snapshot_sha,
            intent_col=intent.col if intent else None,
            intent_rot=intent.rot if intent else None,
            why=intent.why if intent else "",
            error=error,
            oracle_col=ocol,
            oracle_rot=orot,
            agree=agree,
            rank=rank,
            regret=regret,
            infeasible=infeasible,
            delivered=delivered,
            too_late=too_late,
            joint=joint,
            raw=d.raw,
        )
        self.records.append(rec)
        self._log({"event": "decision", **asdict(rec)})

    # -- メインループ ------------------------------------------------------

    def run(self) -> EpisodeResult:
        self._open_log()
        r = self.rcfg
        self._log({
            "event": "run_start",
            "agent": getattr(self.agent, "name", "?"),
            "engine": asdict(self.ecfg),
            "runtime": {k: v for k, v in asdict(r).items() if k != "render"},
            "render": asdict(r.render),
        })

        stall_ticks = 0
        force_locks = 0
        conflicts = 0
        blocked_ticks = 0
        actions_issued = 0
        ever_reached = 0
        settled_on_target = 0
        locked_without_intent = 0
        t_wall = time.perf_counter()

        while self.engine.tick < r.max_ticks and not self.engine.game_over:
            if r.stop_after_pieces and self.engine.pieces_locked >= r.stop_after_pieces:
                break
            tick = self.engine.tick

            self._submit(tick)
            self._start_services(tick)
            self._deliver(tick)

            actions: dict[int, Action] = {}
            for piece in self.engine.active:
                st = self.states.get(piece.pid)
                intent = st.intent if st else None
                if st is not None and on_target(piece, intent):
                    if not st.ever_on_target:
                        st.ever_on_target = True
                        ever_reached += 1
                a, was_blocked = reflex(
                    piece, intent, lambda act, p=piece: self.engine.action_is_legal(p, act)
                )
                actions[piece.pid] = a
                if a != Action.NONE:
                    actions_issued += 1
                if was_blocked:
                    blocked_ticks += 1
                    if st is not None:
                        st.blocked += 1

            rep = self.engine.step(actions)
            stall_ticks += rep.stalls
            force_locks += len(rep.force_locked)
            # 発行時は合法だったが、同 tick に先着した他ピースに潰された分
            conflicts += len(rep.illegal)
            for pid in rep.illegal:
                st = self.states.get(pid)
                if st:
                    st.blocked += 1

            # 「宣言した場所に本当に収まれたか」は lock の瞬間にしか測れない
            for lp in rep.locked_pieces:
                self.declared.pop(lp.pid, None)
                st = self.states.get(lp.pid)
                if st is None or st.intent is None:
                    locked_without_intent += 1
                    continue
                if on_target(lp, st.intent):
                    settled_on_target += 1

            if r.ascii_every and tick % r.ascii_every == 0:
                ego = self.engine.active[0].pid if self.engine.active else None
                print(to_ascii(self.engine.snapshot(), ego))
                print()

            if r.realtime:
                time.sleep(r.tick_seconds)

        # 終了時にまだ推論中だった要求も必ず記録する。
        # これを落とすと (a) 「間に合わなかった推論」が指標から消え
        #                (b) replay 時にログが欠けて再生が破綻する
        for req in self.in_flight:
            self._apply(req, self.engine.tick, delivered=False)
        self.in_flight = []

        board = self.engine.board
        res = EpisodeResult(
            ticks=self.engine.tick,
            lines_cleared=self.engine.lines_cleared,
            line_histogram=list(self.engine.line_clear_histogram),
            pieces_locked=self.engine.pieces_locked,
            game_over=self.engine.game_over,
            final_holes=count_holes(board),
            final_bumpiness=bumpiness(board),
            final_max_height=int(column_heights(board).max()),
            final_aggregate_height=aggregate_height(board),
            stall_ticks=stall_ticks,
            force_locks=force_locks,
            simultaneity_conflicts=conflicts,
            blocked_ticks=blocked_ticks,
            actions_issued=actions_issued,
            settled_on_target=settled_on_target,
            ever_on_target=ever_reached,
            locked_without_intent=locked_without_intent,
            agent_name=getattr(self.agent, "name", "?"),
            dropped_requests=self.dropped_requests,
            records=self.records,
        )
        n = max(1, res.pieces_locked)
        res.lines_per_piece = round(res.lines_cleared / n, 4)
        res.holes_per_piece = round(res.final_holes / n, 4)
        res.settle_rate = round(res.settled_on_target / n, 4)
        res.model_calls = self.model_calls
        res.calls_per_piece = round(self.model_calls / n, 3)
        res.coordination = self.mode
        _fill_decision_metrics(res, self.records)
        res.config = {
            "engine": asdict(self.ecfg),
            "runtime": {k: v for k, v in asdict(r).items() if k != "render"},
            "render": asdict(r.render),
            "wall_seconds": round(time.perf_counter() - t_wall, 2),
        }
        self._log({"event": "run_end", **res.summary()})
        self._close_log()
        return res


def _fill_decision_metrics(res: EpisodeResult, records: list[DecisionRecord]) -> None:
    """時間系の指標は「実際に届いて使われた判断」だけを対象にする。

    出力の質 (一致率 / regret) は届いたかどうかと独立なので、
    推論が走った全レコードを対象にする。
    """
    used = [r for r in records if r.delivered and not r.too_late]
    res.decisions = len(used)
    res.late_decisions = sum(1 for r in records if r.delivered and r.too_late)
    res.undelivered_decisions = sum(1 for r in records if not r.delivered)
    if not records:
        return
    if used:
        res.mean_latency_s = round(float(np.mean([r.latency_s for r in used])), 4)
        res.mean_staleness = round(float(np.mean([r.staleness for r in used])), 3)
        res.mean_queue_wait = round(float(np.mean([r.queue_wait for r in used])), 3)
    errs: dict[str, int] = {}
    for rec in records:
        if rec.error:
            key = rec.error.split(":")[0]
            errs[key] = errs.get(key, 0) + 1
    res.errors = errs

    scored = [r for r in records if r.agree is not None]
    if scored:
        res.agreement_top1 = round(sum(1 for r in scored if r.agree) / len(scored), 4)
        ranks = [r.rank for r in scored if r.rank is not None]
        res.mean_rank = round(float(np.mean(ranks)), 3) if ranks else None
        regrets = [r.regret for r in scored if r.regret is not None]
        res.mean_regret = round(float(np.mean(regrets)), 4) if regrets else None
        res.infeasible_rate = round(
            sum(1 for r in scored if r.infeasible) / len(scored), 4
        )
