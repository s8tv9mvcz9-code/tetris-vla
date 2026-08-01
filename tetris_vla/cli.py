"""tetris_vla.cli — コマンドラインエントリ。

  python -m tetris_vla run      --agent vla-mock --active 3 --ascii-every 20
  python -m tetris_vla eval     --seeds 5 --out results
  python -m tetris_vla snapshot --out board.png --warmup 40
  python -m tetris_vla replay   --log runs/a.jsonl
  python -m tetris_vla doctor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace

from .agents import DEFAULT_PERSONA_CYCLE, PERSONAS
from .engine import Action, EngineConfig
from .benchmark import (
    TIERS,
    TIER_BY_NAME,
    compare_modes_markdown,
    run_benchmark,
)
from .benchmark import to_markdown as bench_markdown
from .evaluate import (
    AGENT_KINDS,
    DEFAULT_SUITE,
    DICTATOR_KINDS,
    STUDIES,
    AgentSpec,
    Variant,
    run_episode,
    run_matrix,
    to_json,
    to_markdown,
)
from .render import RenderConfig, render, to_ascii, to_png
from .runtime import RuntimeConfig, Session


# --------------------------------------------------------------------------


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("盤面 / 物理")
    g.add_argument("--width", type=int, default=10)
    g.add_argument("--height", type=int, default=20)
    g.add_argument("--active", type=int, default=3,
                   help="同時に落下するピース数。1 で古典テトリス相当")
    g.add_argument("--fall-period", type=int, default=4,
                   help="何 tick に 1 セル落ちるか (決められた時速)")
    g.add_argument("--fall-jitter", type=int, default=1,
                   help="ピースごとの速度ばらつき")
    g.add_argument("--spawn-interval", type=int, default=5)
    g.add_argument("--stall-limit", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--ticks", type=int, default=600, help="最大 tick 数")

    g = p.add_argument_group("知覚")
    g.add_argument("--cell-px", type=int, default=12)
    g.add_argument("--fog", action="store_true",
                   help="他の落下ピースの種類を隠す (位置だけ見える)")
    g.add_argument("--no-labels", action="store_true")

    g = p.add_argument_group("推論スケジューリング")
    g.add_argument("--tick-seconds", type=float, default=0.15,
                   help="1 tick の実時間換算。実レイテンシを tick に直す係数")
    g.add_argument("--replan", type=int, default=6, help="再計画の最小間隔 (tick)")
    g.add_argument("--budget", type=int, default=6, help="1 ピースあたりの最大判断回数")
    g.add_argument("--workers", type=int, default=1, help="同時推論数 (ローカルは 1 推奨)")
    g.add_argument("--personas", type=str, default=",".join(DEFAULT_PERSONA_CYCLE),
                   help=f"割り当てるペルソナ (候補: {','.join(PERSONAS)})")
    g.add_argument("--no-oracle", action="store_true", help="一致率メトリクスを取らない (高速)")


def _cfgs(a: argparse.Namespace) -> tuple[EngineConfig, RuntimeConfig]:
    ecfg = EngineConfig(
        width=a.width,
        height=a.height,
        active_pieces=a.active,
        fall_period=a.fall_period,
        fall_period_jitter=a.fall_jitter,
        spawn_interval=a.spawn_interval,
        stall_limit=a.stall_limit,
        seed=a.seed,
    )
    rcfg = RuntimeConfig(
        tick_seconds=a.tick_seconds,
        replan_period=a.replan,
        decision_budget_per_piece=a.budget,
        inference_workers=a.workers,
        max_ticks=a.ticks,
        compute_oracle=not a.no_oracle,
        personas=tuple(x for x in a.personas.split(",") if x),
        coordination=getattr(a, "mode", "autonomous"),
        render=RenderConfig(cell_px=a.cell_px, fog_other=a.fog, labels=not a.no_labels,
                            show_ghosts=getattr(a, "mode", "autonomous") == "a2a"),
        seed=a.seed,
    )
    return ecfg, rcfg


def _spec(a: argparse.Namespace) -> AgentSpec:
    opts: dict = {}
    if a.agent == "vla-mock":
        opts = {
            "skill": a.skill,
            "latency_s": a.latency,
            "p_malformed": a.p_malformed,
            "p_refuse": a.p_refuse,
            "p_out_of_range": a.p_oob,
            # --realtime は run サブコマンドにしかない
            "sleep": getattr(a, "realtime", False),
            "text_board": a.text_board,
        }
    elif a.agent == "vla-ollama":
        opts = {"model": a.model, "host": a.host, "text_board": a.text_board}
    elif a.agent == "vla-replay":
        opts = {"log": a.replay_log, "text_board": a.text_board}
    elif a.agent == "heuristic":
        opts = {"treat_active_as_obstacle": a.avoid_active}
    elif a.agent == "dictator-vla-mock":
        opts = {"skill": a.skill, "latency_s": a.latency, "p_malformed": a.p_malformed,
                "p_refuse": a.p_refuse, "p_out_of_range": a.p_oob,
                "sleep": getattr(a, "realtime", False), "text_board": a.text_board}
    elif a.agent == "dictator-vla-ollama":
        opts = {"model": a.model, "host": a.host, "text_board": a.text_board}
    return AgentSpec(label=a.agent, kind=a.agent, options=opts)


def _agent_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("エージェント")
    g.add_argument("--agent", default="vla-mock", choices=list(AGENT_KINDS))
    g.add_argument("--mode", default="autonomous", choices=["autonomous", "a2a", "dictator"],
                   help="協調モード。dictator-* エージェントを選ぶと自動で独裁になる")
    g.add_argument("--skill", type=float, default=0.55, help="mock: 最良手を選ぶ確率")
    g.add_argument("--latency", type=float, default=1.2, help="mock: 中央値レイテンシ(秒)")
    g.add_argument("--p-malformed", type=float, default=0.06)
    g.add_argument("--p-refuse", type=float, default=0.03)
    g.add_argument("--p-oob", type=float, default=0.05)
    g.add_argument("--model", default="qwen2.5vl:3b", help="ollama: モデル名")
    g.add_argument("--host", default="http://localhost:11434")
    g.add_argument("--replay-log", default=None)
    g.add_argument("--text-board", action="store_true",
                   help="画像に加えてテキスト盤面も渡す (視覚と推論の切り分け ablation)")
    g.add_argument("--avoid-active", action="store_true",
                   help="heuristic: 他の落下ピースも障害物として計画に含める")


# --------------------------------------------------------------------------


def cmd_run(a: argparse.Namespace) -> int:
    if a.agent in DICTATOR_KINDS:
        a.mode = "dictator"
    ecfg, rcfg = _cfgs(a)
    rcfg = replace(
        rcfg,
        ascii_every=a.ascii_every,
        log_path=a.log,
        dump_frames=a.dump_frames,
        realtime=a.realtime,
    )
    spec = _spec(a)
    if a.agent == "vla-ollama":
        from .agents import OllamaBackend

        print(OllamaBackend(model=a.model, host=a.host).health(), file=sys.stderr)

    agent = spec.build(a.seed)
    sess = Session(ecfg, rcfg, agent)
    res = sess.run()

    print(to_ascii(sess.engine.snapshot()))
    print()
    _print_summary(res)
    if a.log:
        print(f"\nログ: {a.log}  ({len(res.records)} decisions)")
    return 0


def _print_summary(res) -> None:
    s = res.summary()
    print(f"=== {res.agent_name} ===")
    rows = [
        ("ticks / game_over", f"{s['ticks']} / {s['game_over']}"),
        ("消去ライン (内訳)", f"{s['lines_cleared']}  {s['line_histogram'][1:]}"),
        ("設置ピース数", s["pieces_locked"]),
        ("最終 穴 / 凸凹 / 最高列", f"{s['final_holes']} / {s['final_bumpiness']} / {s['final_max_height']}"),
        ("宣言どおり収まった数", f"{s['settled_on_target']} / {s['pieces_locked']}"),
        ("意図なしでlockされた数", s["locked_without_intent"]),
        ("停滞 tick / 強制lock", f"{s['stall_ticks']} / {s['force_locks']}"),
        ("進路を塞がれた tick", s["blocked_ticks"]),
        ("発行アクション", s["actions_issued"]),
        ("同時手番の競合", s["simultaneity_conflicts"]),
        ("協調モード", s["coordination"]),
        ("推論呼び出し回数", f"{s['model_calls']} ({s['calls_per_piece']:.2f}/個)"),
        ("判断回数 (使われた)", s["decisions"]),
        ("遅着 / 未達 / 破棄", f"{s['late_decisions']} / {s['undelivered_decisions']} / {s['dropped_requests']}"),
        ("平均レイテンシ(s)", s["mean_latency_s"]),
        ("平均 staleness(tick)", s["mean_staleness"]),
        ("平均 queue wait(tick)", s["mean_queue_wait"]),
        ("オラクル一致率(top1)", s["agreement_top1"]),
        ("平均 regret", s["mean_regret"]),
        ("平均 順位", s["mean_rank"]),
        ("実行不能な意図の割合", s["infeasible_rate"]),
        ("エラー内訳", s["errors"] or "-"),
        ("実時間(s)", s["config"].get("wall_seconds")),
    ]
    w = max(len(str(k)) for k, _ in rows)
    for k, v in rows:
        print(f"  {str(k).ljust(w)} : {v}")


def _write_report(out: str, name: str, title: str, ecfg, rcfg, seeds, md: str, aggs) -> None:
    os.makedirs(out, exist_ok=True)
    header = (
        f"# {title}\n\n"
        f"- active_pieces: {ecfg.active_pieces}\n"
        f"- fall_period: {ecfg.fall_period} (±{ecfg.fall_period_jitter}), "
        f"spawn_interval: {ecfg.spawn_interval}\n"
        f"- max_ticks: {rcfg.max_ticks}, seeds: {seeds}\n"
        f"- tick_seconds: {rcfg.tick_seconds}, workers: {rcfg.inference_workers}, "
        f"replan: {rcfg.replan_period}, budget: {rcfg.decision_budget_per_piece}\n"
        f"- 画像: {rcfg.render.cell_px}px/cell, fog={rcfg.render.fog_other}\n\n"
    )
    with open(os.path.join(out, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(header + md + "\n")
    with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as f:
        f.write(to_json(aggs))
    print(f"\n出力: {out}/{name}.md, {out}/{name}.json", file=sys.stderr)


def cmd_eval(a: argparse.Namespace) -> int:
    ecfg, rcfg = _cfgs(a)
    rcfg = replace(rcfg, ascii_every=0, log_path=None)
    seeds = list(range(a.seeds))
    variants = list(DEFAULT_SUITE)
    if a.with_ollama:
        variants.append(Variant(f"vla-ollama {a.model}",
                                AgentSpec("vla", "vla-ollama", {"model": a.model})))
    print(f"実験: {len(variants)} 構成 x {len(seeds)} seed, active={ecfg.active_pieces}, "
          f"max_ticks={rcfg.max_ticks}", file=sys.stderr)
    aggs = run_matrix(variants, ecfg, rcfg, seeds, progress=lambda m: print(m, file=sys.stderr))
    md = to_markdown(aggs)
    print()
    print(md)
    if a.out:
        _write_report(a.out, "report", "tetris-vla 比較実験", ecfg, rcfg, seeds, md, aggs)
    return 0


def cmd_ablate(a: argparse.Namespace) -> int:
    names = list(STUDIES) if a.study == "all" else [a.study]
    ecfg, rcfg = _cfgs(a)
    rcfg = replace(rcfg, ascii_every=0, log_path=None)
    seeds = list(range(a.seeds))
    for name in names:
        title, factory = STUDIES[name]
        variants = factory()
        print(f"\n## {name} — {title}", file=sys.stderr)
        aggs = run_matrix(variants, ecfg, rcfg, seeds,
                          progress=lambda m: print(m, file=sys.stderr))
        md = to_markdown(aggs)
        print(f"\n## {name} — {title}\n")
        print(md)
        if a.out:
            _write_report(a.out, name, f"tetris-vla ablation: {title}",
                          ecfg, rcfg, seeds, md, aggs)
    return 0


def cmd_snapshot(a: argparse.Namespace) -> int:
    """エージェントが実際に見ている 1 枚を書き出す。"""
    ecfg, rcfg = _cfgs(a)
    rcfg = replace(rcfg, max_ticks=a.warmup, log_path=None)
    sess = Session(ecfg, rcfg, AgentSpec("heuristic", "heuristic").build(a.seed))
    sess.run()
    snap = sess.engine.snapshot()
    ego = snap.active[0].pid if snap.active else None
    img = render(snap, ego_pid=ego, cfg=rcfg.render)
    img.save(a.out)
    print(to_ascii(snap, ego))
    print(f"\n{a.out} ({img.width}x{img.height}px, {len(to_png(img))} bytes) "
          f"ego=#{ego} — これがエージェントに渡る全情報です")
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    with open(a.log, "r", encoding="utf-8") as f:
        events = [json.loads(l) for l in f if l.strip()]
    start = next((e for e in events if e.get("event") == "run_start"), None)
    end = next((e for e in events if e.get("event") == "run_end"), None)
    decisions = [e for e in events if e.get("event") == "decision"]
    print(f"agent: {start.get('agent') if start else '?'}  decisions: {len(decisions)}")
    if start:
        print(f"engine: {start['engine']}")
    print()
    print(f"{'tick':>5} {'pid':>4} {'kind':>4} {'persona':>10} {'stale':>5} {'wait':>4} "
          f"{'lat(s)':>7} {'intent':>10} {'oracle':>8} {'rank':>4} {'err':<12} why")
    for d in decisions[: a.limit]:
        it = f"c{d['intent_col']}r{d['intent_rot']}" if d["intent_col"] is not None else "-"
        orc = f"c{d['oracle_col']}r{d['oracle_rot']}" if d.get("oracle_col") is not None else "-"
        print(f"{d['tick']:>5} {d['pid']:>4} {d['kind']:>4} {d['persona']:>10} "
              f"{d['staleness']:>5} {d['queue_wait']:>4} {d['latency_s']:>7.2f} "
              f"{it:>10} {orc:>8} {str(d.get('rank')):>4} {str(d.get('error') or ''):<12} {d.get('why','')}")
    if end:
        print()
        print(f"結果: lines={end['lines_cleared']} pieces={end['pieces_locked']} "
              f"agreement={end.get('agreement_top1')} regret={end.get('mean_regret')}")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    """マシン性能ベンチマーク。実測レイテンシが tick に換算されてスコアに効く。"""
    tiers = TIERS if a.tier == "all" else (TIER_BY_NAME[a.tier],)
    seeds = tuple(range(a.seeds))
    modes = a.modes.split(",") if a.modes else [a.mode]
    results = []
    for mode in modes:
        agent = a.agent
        if mode == "dictator" and agent not in DICTATOR_KINDS:
            agent = {"heuristic": "dictator-heuristic",
                     "vla-mock": "dictator-vla-mock",
                     "vla-ollama": "dictator-vla-ollama"}.get(agent, "dictator-heuristic")
            print(f"[info] mode=dictator のため agent を {agent} に切り替えました", file=sys.stderr)
        elif agent in DICTATOR_KINDS:
            mode = "dictator"
        holder = argparse.Namespace(**{**vars(a), "agent": agent})
        spec = _spec(holder)
        if agent == "vla-ollama" or agent == "dictator-vla-ollama":
            from .agents import OllamaBackend

            print(OllamaBackend(model=a.model, host=a.host).health(), file=sys.stderr)
        print(f"\n=== {agent} / 協調={mode} / seeds={list(seeds)} ===", file=sys.stderr)
        bench = run_benchmark(
            spec, coordination=mode, seeds=seeds, tiers=tiers,
            stop_on_fail=not a.all_tiers, max_model_calls=a.max_calls,
            progress=lambda m: print(m, file=sys.stderr),
        )
        results.append(bench)
        print()
        print(bench_markdown(bench))
    if len(results) > 1:
        print()
        print(compare_modes_markdown(results))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        md = "\n\n".join(bench_markdown(b) for b in results)
        if len(results) > 1:
            md += "\n\n" + compare_modes_markdown(results)
        with open(os.path.join(a.out, "benchmark.md"), "w", encoding="utf-8") as f:
            f.write("# tetris-vla マシンベンチマーク\n\n" + md + "\n")
        with open(os.path.join(a.out, "benchmark.json"), "w", encoding="utf-8") as f:
            json.dump([asdict(b) for b in results], f, ensure_ascii=False, indent=2)
        print(f"\n出力: {a.out}/benchmark.md", file=sys.stderr)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    """途中判断の比較。同じ盤面を各方式に見せて、選んだ着地点を並べる。"""
    from .compare import capture_scenes, default_verdicts, to_html

    scenes = capture_scenes(seed=a.seed, active=a.active, n_scenes=a.scenes,
                            first_at=a.first_at, every=a.every, fall_period=a.fall_period)
    if not scenes:
        print("比較できる場面が取れませんでした (--first-at / --every を調整してください)",
              file=sys.stderr)
        return 1
    verdicts = [default_verdicts(s, vla_skill=a.vla_skill) for s in scenes]

    for scene, vs in zip(scenes, verdicts):
        from .compare import conflict_count, disagreement

        print(f"\n=== t={scene.tick} 落下中 {len(scene.pieces)} 体 ===")
        ref = vs[0]
        for v in vs:
            picks = " ".join(
                f"#{p.pid}{p.kind}->"
                + (f"c{v.intents[p.pid].col}r{v.intents[p.pid].rot}"
                   if p.pid in v.intents else "--")
                for p in scene.pieces
            )
            print(f"  {v.method:<26} {picks}   衝突{conflict_count(scene, v)}セル "
                  f"参照相違{len(disagreement(ref, v))}体 推論{v.model_calls}回")

    html_doc = to_html(scenes, verdicts)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"\n出力: {a.out} ({len(html_doc)} bytes, {len(scenes)} 場面)")
    return 0


PILOT_KINDS = ("pd", "random", "coast", "vla-mock", "vla-ollama",
               "vla-oracle", "vla-vlm", "vla-solo", "vla-zeroshot")


def _build_pilot(a: argparse.Namespace):
    """パイロットを 1 つ作る。VLA 系は lerobot が必要なので遅延 import。"""
    from .parachute import (MockFlightBackend, MockFlightConfig, OllamaFlightBackend,
                            PDPilot, RandomPilot, VLAPilot)

    k = a.pilot
    if k == "pd":
        return PDPilot()
    if k == "random":
        return RandomPilot(seed=a.seed)
    if k == "coast":
        from .parachute import FlightDecision, Thrust

        class _Coast:
            name = "coast"

            def decide(self, obs):
                return FlightDecision(Thrust.COAST, None, "何もしない")

        return _Coast()
    if k == "vla-mock":
        return VLAPilot(MockFlightBackend(
            MockFlightConfig(skill=a.skill, latency_s=a.latency), seed=a.seed), lang=a.lang)
    if k == "vla-ollama":
        be = OllamaFlightBackend(model=a.model, host=a.host, num_predict=a.num_predict)
        print(be.health(), file=sys.stderr)
        return VLAPilot(be, lang=a.lang)
    # ここから SmolVLA 系
    from .smolvla_pilot import make_pilot

    return make_pilot(k, checkpoint=a.checkpoint, device=a.device,
                      vlm_model=a.model, vlm_every=a.vlm_every)


def cmd_fly(a: argparse.Namespace) -> int:
    """パラシュート降下デモ。入力と判断を出しながら 1 機降ろす。"""
    from .parachute import ParachuteConfig, live_flight, save_flight

    cfg = ParachuteConfig(width=a.width, height=a.height, wind=a.wind, gust=a.gust,
                          fuel=a.fuel, seed=a.seed)
    rc = RenderConfig(cell_px=a.cell_px)
    pilot = _build_pilot(a)
    period = 1 if a.pilot.startswith("vla-") and a.pilot not in ("vla-mock", "vla-ollama") \
        else a.decision_period
    res = live_flight(pilot, cfg, seed=a.seed, kind=a.kind, decision_period=period,
                      render_cfg=rc, realtime=a.realtime, dump_dir=a.dump_frames)
    if a.log:
        save_flight(res, a.log, cfg)
        print(f"\n記録: {a.log}", file=sys.stderr)
    if a.svg or a.html:
        from .parachute import load_flight
        from .svganim import flight_html, flight_svg

        tmp = a.log or os.path.join(os.path.dirname(a.svg or a.html) or ".", "_flight.jsonl")
        if not a.log:
            save_flight(res, tmp, cfg)
        data = load_flight(tmp)
        svg = flight_svg(data, speed=a.svg_speed)
        if a.svg:
            os.makedirs(os.path.dirname(os.path.abspath(a.svg)) or ".", exist_ok=True)
            with open(a.svg, "w", encoding="utf-8") as f:
                f.write(svg)
            print(f"SVG アニメ: {a.svg}", file=sys.stderr)
        if a.html:
            os.makedirs(os.path.dirname(os.path.abspath(a.html)) or ".", exist_ok=True)
            with open(a.html, "w", encoding="utf-8") as f:
                f.write(flight_html(data, svg))
            print(f"HTML (入力+判断): {a.html}", file=sys.stderr)
    return 0


def cmd_flybench(a: argparse.Namespace) -> int:
    """複数のパイロットを同じ seed 群で走らせて比較する。"""
    import statistics

    from .parachute import ParachuteConfig, fly

    cfg = ParachuteConfig(width=a.width, height=a.height, wind=a.wind, gust=a.gust, fuel=a.fuel)
    rc = RenderConfig(cell_px=a.cell_px)
    seeds = list(range(a.seeds))
    rows = []
    for kind in a.pilots.split(","):
        holder = argparse.Namespace(**{**vars(a), "pilot": kind})
        pilot = _build_pilot(holder)
        sc, rg, er, soft, lat, dec, land = [], [], [], 0, [], [], 0
        for seed in seeds:
            if hasattr(pilot, "reset"):
                pilot.reset()
            period = 1 if kind.startswith("vla-") and kind not in ("vla-mock", "vla-ollama") \
                else a.decision_period
            r = fly(pilot, cfg, seed=seed, render_cfg=rc, decision_period=period, keep_png=False)
            sc.append(r.score); soft += r.soft; land += r.landed
            lat.append(r.mean_latency_s); dec.append(r.decisions)
            if r.regret is not None: rg.append(r.regret)
            if r.col_error is not None: er.append(r.col_error)
            print(f"  {kind} seed={seed} {r.score:.0f}点", file=sys.stderr)
        rows.append((kind, statistics.fmean(sc), statistics.fmean(rg) if rg else None,
                     statistics.fmean(er) if er else None, soft, land,
                     statistics.fmean(lat), statistics.fmean(dec)))
    print()
    print("| 操縦者 | 点↑ | regret↓ | 列誤差↓ | 軟着陸 | 着地 | 平均lat(s) | 判断回数 |")
    print("|---|---|---|---|---|---|---|---|")
    for k, s_, g, e, so, la, l_, d in rows:
        print(f"| {k} | {s_:.0f} | {g if g is None else f'{g:.3f}'} | "
              f"{e if e is None else f'{e:.2f}'} | {so}/{len(seeds)} | {la}/{len(seeds)} | "
              f"{l_:.3f} | {d:.1f} |")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "flight_bench.md"), "w", encoding="utf-8") as f:
            f.write(f"# パラシュート降下 比較 (seeds={seeds}, 風={cfg.wind}±{cfg.gust})\n\n")
            f.write("| 操縦者 | 点↑ | regret↓ | 列誤差↓ | 軟着陸 | 着地 | 平均lat(s) | 判断回数 |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for k, s_, g, e, so, la, l_, d in rows:
                f.write(f"| {k} | {s_:.0f} | {g if g is None else f'{g:.3f}'} | "
                        f"{e if e is None else f'{e:.2f}'} | {so}/{len(seeds)} | "
                        f"{la}/{len(seeds)} | {l_:.3f} | {d:.1f} |\n")
        print(f"\n出力: {a.out}/flight_bench.md", file=sys.stderr)
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    print("--- 依存 ---")
    for m in ("numpy", "PIL", "httpx"):
        try:
            __import__(m)
            print(f"  {m:8s} OK")
        except ImportError:
            print(f"  {m:8s} 見つかりません")
    print("--- Ollama ---")
    try:
        import httpx

        r = httpx.get(f"{a.host}/api/tags", timeout=3.0)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        print(f"  接続 OK: {a.host}")
        print(f"  モデル: {names or '(なし)'}")
        if not names:
            print("  -> ollama pull qwen2.5vl:3b")
    except Exception as exc:
        print(f"  接続不可 ({type(exc).__name__}). mock バックエンドのみ利用可能です。")
        print("  実機で試すなら:")
        print("    brew install ollama && ollama serve")
        print("    ollama pull qwen2.5vl:3b     # 8GB 機なら 3B 級が上限の目安")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tetris_vla",
        description="localVLA マルチエージェント・テトリス PoC",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="1 エピソード実行")
    _common(r)
    _agent_args(r)
    r.add_argument("--ascii-every", type=int, default=0, help="N tick ごとに盤面を表示")
    r.add_argument("--log", default=None, help="JSONL ログ出力先")
    r.add_argument("--dump-frames", default=None, help="エージェント視点 PNG の出力ディレクトリ")
    r.add_argument("--realtime", action="store_true", help="実時間で待つ (観賞用)")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="比較実験マトリクス")
    _common(e)
    e.add_argument("--seeds", type=int, default=5)
    e.add_argument("--out", default=None, help="レポート出力ディレクトリ")
    e.add_argument("--with-ollama", action="store_true")
    e.add_argument("--model", default="qwen2.5vl:3b")
    e.set_defaults(func=cmd_eval)

    ab = sub.add_parser("ablate", help="アブレーション研究")
    _common(ab)
    ab.add_argument("--study", default="all", choices=["all", *STUDIES],
                    help="; ".join(f"{k}: {v[0]}" for k, v in STUDIES.items()))
    ab.add_argument("--seeds", type=int, default=5)
    ab.add_argument("--out", default=None)
    ab.set_defaults(func=cmd_ablate)

    s = sub.add_parser("snapshot", help="エージェント視点の 1 枚を書き出す")
    _common(s)
    s.add_argument("--out", default="snapshot.png")
    s.add_argument("--warmup", type=int, default=60, help="何 tick 進めてから撮るか")
    s.set_defaults(func=cmd_snapshot)

    rp = sub.add_parser("replay", help="JSONL ログを読む")
    rp.add_argument("--log", required=True)
    rp.add_argument("--limit", type=int, default=60)
    rp.set_defaults(func=cmd_replay)

    b = sub.add_parser("bench", help="マシン性能ベンチマーク (tier 制ハードル + スコア)")
    _agent_args(b)
    b.add_argument("--tier", default="all", choices=["all", *[t.name for t in TIERS]])
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--modes", default=None,
                   help="カンマ区切りで複数モードを比較 (例: autonomous,a2a,dictator)")
    b.add_argument("--all-tiers", action="store_true",
                   help="不合格でも打ち切らず全 tier を走らせる")
    b.add_argument("--max-calls", type=int, default=None,
                   help="エピソードあたりの推論回数上限 (モード間で推論コストを揃える)")
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_bench)

    cp = sub.add_parser("compare", help="途中判断の比較 (HTML 出力)")
    cp.add_argument("--seed", type=int, default=1)
    cp.add_argument("--active", type=int, default=3)
    cp.add_argument("--fall-period", type=int, default=4)
    cp.add_argument("--scenes", type=int, default=4)
    cp.add_argument("--first-at", type=int, default=120)
    cp.add_argument("--every", type=int, default=90)
    cp.add_argument("--vla-skill", type=float, default=0.6,
                   help="VLA (mock) の腕前も並べる。None にしたい場合は -1")
    cp.add_argument("--out", default="decisions.html")
    cp.set_defaults(func=cmd_compare)

    def _fly_args(q):
        q.add_argument("--width", type=int, default=10)
        q.add_argument("--height", type=int, default=20)
        q.add_argument("--wind", type=float, default=0.8)
        q.add_argument("--gust", type=float, default=1.6)
        q.add_argument("--fuel", type=int, default=60)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--kind", default=None, help="ピース種類 (既定は seed で決定)")
        q.add_argument("--cell-px", type=int, default=12)
        q.add_argument("--decision-period", type=int, default=4,
                       help="何 tick ごとに判断を求めるか (VLA 系は毎 tick)")
        q.add_argument("--skill", type=float, default=0.8, help="vla-mock の腕前")
        q.add_argument("--latency", type=float, default=1.4, help="vla-mock の中央値レイテンシ")
        q.add_argument("--model", default="qwen2.5vl:3b")
        q.add_argument("--host", default="http://localhost:11434")
        q.add_argument("--num-predict", type=int, default=28)
        q.add_argument("--lang", default="en", choices=["en", "ja"],
                       help="VLM プロンプトの言語 (en が約5倍速い)")
        q.add_argument("--checkpoint", default="checkpoints/smolvla_bc.pt")
        q.add_argument("--device", default="auto")
        q.add_argument("--vlm-every", type=int, default=20,
                       help="階層型で上位 VLM に問い合わせる間隔 (tick)")

    fl = sub.add_parser("fly", help="パラシュート降下デモ (入力と判断を表示)")
    _fly_args(fl)
    fl.add_argument("--pilot", default="pd", choices=list(PILOT_KINDS))
    fl.add_argument("--realtime", action="store_true", help="実時間で待つ (観賞用)")
    fl.add_argument("--log", default=None, help="JSONL 記録の出力先")
    fl.add_argument("--svg", default=None, help="SVG アニメの出力先")
    fl.add_argument("--html", default=None, help="入力+判断の HTML 出力先")
    fl.add_argument("--svg-speed", type=float, default=1.0)
    fl.add_argument("--dump-frames", default=None)
    fl.set_defaults(func=cmd_fly)

    fb = sub.add_parser("flybench", help="パラシュート降下の操縦者比較")
    _fly_args(fb)
    fb.add_argument("--pilots", default="pd,coast,random,vla-mock")
    fb.add_argument("--seeds", type=int, default=8)
    fb.add_argument("--out", default=None)
    fb.set_defaults(func=cmd_flybench)

    d = sub.add_parser("doctor", help="環境チェック")
    d.add_argument("--host", default="http://localhost:11434")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
