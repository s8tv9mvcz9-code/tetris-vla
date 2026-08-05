//! TDD のサイクルそのもの。上から順に読むと、赤から緑までの道筋になる。
//!
//! ```text
//! cargo test -p sim-harness --test tdd_cycle
//! cargo test -p sim-harness -- --ignored     # 生成された規則テストの赤を見る
//! ```

use sim_harness::runners::{self, PickCycle};
use sim_harness::scenes::cell01;
use sim_harness::{run_legacy_raw, run_legacy_staggered, run_typed_sdk};
use vtime::Prog;

// ---------------------------------------------------------------------------
// 段 0: 走らせる前に分かること
// ---------------------------------------------------------------------------

/// 合成した手順の WCET は型から取れる。1 ms 周期の制御タスクに載る。
const _: () = vtime::Budget::<PickCycle, 1_000>::FITS;

#[test]
fn wcet_is_known_before_running() {
    assert_eq!(PickCycle::WCET_US, 76);
    assert_eq!(vtime::Budget::<PickCycle, 1_000>::SLACK_US, 924);
}

// ---------------------------------------------------------------------------
// 段 1 (赤): レガシーの手順をそのまま走らせる
// ---------------------------------------------------------------------------

#[test]
fn red_the_legacy_sequence_puts_both_arms_in_the_same_zone() {
    let r = run_legacy_raw(&cell01::SCENE);

    // 宣言した規則のうち 2 本が破れる。
    r.assert_rule_violated(&cell01::SCENE, cell01::SEPARATION);
    r.assert_rule_violated(&cell01::SCENE, cell01::EXCLUSIVE_X);

    // 破れ方まで固定しておく。ここが動いたら前提が変わったということ。
    let sep = r.violation_of(cell01::SEPARATION).unwrap();
    assert_eq!(sep.t_us, 15_000);
    assert_eq!(
        sep.measured, 48_000,
        "表面間 48 mm — 下限 50 mm を切っている"
    );

    let exc = r.violation_of(cell01::EXCLUSIVE_X).unwrap();
    assert_eq!(exc.t_us, 15_800);

    // C 側は異常だと思っていない。手順は最後まで「正常に」流れる。
    assert!(r.violation_of(cell01::TACT).is_none());
}

/// レガシーの排他フラグは、干渉より **後** に立つわけではない。
/// ちゃんと立っているのに干渉する — つまりフラグは嘘をついていない。嘘なのは設計のほう。
#[test]
fn red_the_interference_happens_while_the_legacy_flag_says_busy() {
    let r = run_legacy_raw(&cell01::SCENE);
    let fault = r.violation_of(cell01::SEPARATION).unwrap();
    // 干渉時刻より前に、両方のタスクが「手順開始」を打っている。
    let rises: Vec<_> = r
        .events
        .iter()
        .filter(|e| e.kind == pulse_trace::Kind::TaskRise && e.t_us <= fault.t_us)
        .collect();
    assert_eq!(rises.len(), 2, "2 本とも手順に入った状態で干渉している");
}

// ---------------------------------------------------------------------------
// 段 2: 現場の回避策 —「時間差でかわす」は緑になる。だが証明ではない
// ---------------------------------------------------------------------------

#[test]
fn the_field_workaround_is_green_but_only_by_timing() {
    let ok = run_legacy_staggered(&cell01::SCENE);
    assert!(
        ok.is_clean(),
        "40 ms ずらせば当たらない: {:?}",
        ok.violations
    );

    // ずらし量を削っていくと、どこかで当たり始める。
    // 「当たらない」は設計の性質ではなく、たまたま成立している時間差でしかない。
    let mut boundary = None;
    for stagger_us in [0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000] {
        let r = runners::run_legacy(&cell01::SCENE, "legacy-sweep", stagger_us);
        if r.is_clean() && boundary.is_none() {
            boundary = Some(stagger_us);
        }
        if boundary.is_some() {
            assert!(
                r.is_clean(),
                "ずらし {} µs で違反が復活した (境界の外側のはずが揺れている)",
                stagger_us
            );
        }
    }
    let boundary = boundary.expect("どれだけずらしても当たるなら、話が別");
    assert!(boundary > 0, "ずらし 0 で緑なら、そもそも競合していない");
    // 境界はサーボ速度と保持時間で動く。定数を 1 つ変えれば破れる類のもの。
    assert_eq!(boundary, 15_000, "この設備定数での境界は 15 ms");
}

// ---------------------------------------------------------------------------
// 段 3 (緑): 手順を型安全 SDK へ移す。送りループはレガシーのまま
// ---------------------------------------------------------------------------

#[test]
fn green_the_typed_cycle_satisfies_every_declared_rule() {
    let r = run_typed_sdk(&cell01::SCENE);
    assert!(r.is_clean(), "違反が残っている: {:?}", r.violations);
    for id in 0..cell01::SCENE.rules.len() {
        r.assert_rule_holds(&cell01::SCENE, id as u16);
    }
}

/// 型状態の意図 (許可証の取得 / 返却) も痕跡に残る。
/// 取得と返却が厳密に交互 = ゾーンに 2 者が居た瞬間が無い。
#[test]
fn green_permits_are_strictly_alternating() {
    use pulse_trace::Kind;
    let r = run_typed_sdk(&cell01::SCENE);
    let mut held = 0i32;
    for e in r.events.iter() {
        match e.kind {
            Kind::PermitTake => held += 1,
            Kind::PermitDrop => held -= 1,
            _ => continue,
        }
        assert!(
            (0..=1).contains(&held),
            "許可証の同時保有数が {} になった @ t={} µs",
            held,
            e.t_us
        );
    }
    assert_eq!(held, 0, "許可証が返っていない");
    assert_eq!(
        r.events
            .iter()
            .filter(|e| e.kind == Kind::PermitTake)
            .count(),
        2
    );
}

/// 申告した WCET と、合成が実際に計上した時間が一致する。
///
/// 型の上の足し算が絵に描いた餅でないことの確認。
#[test]
fn declared_wcet_matches_what_the_composition_charges() {
    let r = run_typed_sdk(&cell01::SCENE);
    assert_eq!(r.cpu_us, PickCycle::WCET_US);
}

// ---------------------------------------------------------------------------
// 器そのものの性質
// ---------------------------------------------------------------------------

/// 同じシナリオは同じ痕跡を返す。競合バグが「たまに出る」ものではなくなる。
#[test]
fn the_same_scenario_reproduces_bit_for_bit() {
    let a = run_legacy_raw(&cell01::SCENE);
    let b = run_legacy_raw(&cell01::SCENE);
    assert_eq!(a.events, b.events);
    assert_eq!(a.violations, b.violations);
    assert_eq!(a.t_end_us, b.t_end_us);
}

/// 仮想時間は実時間を食わない。20 回まわしても実時計は数十 ms。
#[test]
fn virtual_time_does_not_cost_wall_time() {
    let wall = std::time::Instant::now();
    let mut virt = 0u64;
    for _ in 0..20 {
        virt += run_typed_sdk(&cell01::SCENE).t_end_us;
    }
    assert_eq!(virt, 20 * 103_000, "仮想では 2 秒ぶん回した");
    assert!(
        wall.elapsed().as_secs() < 5,
        "実時間を食っている: {:?}",
        wall.elapsed()
    );
}

/// タクトを詰めると、干渉ではなくデッドラインで落ちる。
/// 「緑にする」は 1 種類ではない — どの規則で落ちているかが設計判断につながる。
#[test]
fn tightening_the_tact_moves_the_failure_from_space_to_time() {
    use sim_harness::scenes::cell01_tight;
    let r = run_typed_sdk(&cell01_tight::SCENE);
    r.assert_rule_holds(&cell01_tight::SCENE, cell01_tight::SEPARATION);
    r.assert_rule_holds(&cell01_tight::SCENE, cell01_tight::EXCLUSIVE_X);
    let miss = r
        .violation_of(cell01_tight::TACT)
        .expect("80 ms のタクトには載らないはず");
    assert_eq!(miss.measured, 23_000, "23 ms の超過");
}

/// 干渉の因果は痕跡の上でたどれる (UI の赤い矢印はこの 2 本を結ぶ)。
#[test]
fn the_fault_can_be_traced_back_to_two_zone_entries() {
    use pulse_trace::Kind;
    use sim_harness::world::LANE_GEOM_BASE;

    let r = run_legacy_raw(&cell01::SCENE);
    let fault = r
        .events
        .iter()
        .find(|e| e.kind == Kind::Violation && e.a == cell01::EXCLUSIVE_X as i32)
        .expect("排他違反の事象");

    let entries: Vec<_> = r
        .events
        .iter()
        .filter(|e| e.kind == Kind::ZoneEnter && e.lane >= LANE_GEOM_BASE && e.t_us <= fault.t_us)
        .collect();
    assert_eq!(entries.len(), 2, "2 本ぶんの進入が原因として残っている");
    assert_ne!(entries[0].lane, entries[1].lane);
}
