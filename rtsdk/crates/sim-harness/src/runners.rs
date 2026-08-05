//! TDD の 3 段階を、そのまま走らせられる形にしたもの。
//!
//! | ランナー | 何を走らせるか | 期待 |
//! |---|---|---|
//! | [`run_legacy_raw`] | レガシーの外側手順をそのまま 2 軸同時に | **赤** (干渉する) |
//! | [`run_legacy_staggered`] | 時間差でかわす現場の運用 | 条件次第で緑 (脆い) |
//! | [`run_typed_sdk`] | 手順だけ型安全 SDK に移す。送りループはレガシーのまま | **緑** |
//!
//! 3 つとも同じシミュレータ・同じ規則で採点する。違うのは「手順を誰が書いているか」だけ。

use legacy_bridge::{acquire, LegacyAxis};
use pulse_trace::{Kind, Sink};
use vtime::{delay_us, emit, Ctx, Micros, Prog};
use zoneguard::{Arm, Enter, Interlock, Leave, Outside, Permit, Um, Work, ZoneX, CMD_WCET_US};

use crate::scene::SceneDef;
use crate::world::{RunReport, SharedWorld};

/// ゾーン内の進入点 [µm] (軸座標)。2 本のアームは左右からここを狙う。
pub const APPROACH_UM: Um = 200_000;
/// ゾーン内の作業点 [µm]。
pub const PICK_UM: Um = 240_000;
/// 作業点での保持時間 [µs]。
pub const HOLD_US: Micros = 3_000;
/// アームの可達長 [µm]。
pub const REACH_UM: i32 = 300_000;
/// 制御周期 [µs]。この刻みの上でしか事象は起きない。
pub const TICK_US: Micros = 100;
/// 仮想時間の打ち切り [µs]。
pub const SIM_BUDGET_US: Micros = 10_000_000;

type ArmA<S> = Arm<0, REACH_UM, LegacyAxis, S>;
type ArmB<S> = Arm<1, REACH_UM, LegacyAxis, S>;

/// 型安全 SDK で書き直したピックサイクル。
///
/// 中身は `Enter -> Work -> (待ち) -> Leave` を 2 本ぶん `bind` で繋いだだけ。
/// 繋いだ結果の WCET は型から取れる ([`PickCycle::WCET_US`])。
///
/// 許可証は A が退出したときに戻り、それを B が受け取る。
/// **B の `Enter` は A の `Leave` の出力からしか作れない** ので、
/// 「A が出る前に B が入る」順序は書こうとしても型が通らない。
pub struct PickCycle {
    pub arm_a: ArmA<Outside>,
    pub arm_b: ArmB<Outside>,
    pub permit: Permit<ZoneX>,
}

impl Prog for PickCycle {
    type Out = (ArmA<Outside>, ArmB<Outside>, Permit<ZoneX>);

    /// 動作指令 6 回 (2 本 × 入る / 作業 / 出る) と、ライフサイクルの縁 4 本。
    /// 待ちは CPU を使わないので 0。
    ///
    /// この値が合成の実績と一致することは
    /// `tests/tdd_cycle.rs::declared_wcet_matches_what_the_composition_charges` で見ている。
    const WCET_US: Micros = 6 * CMD_WCET_US + 4 * vtime::Emit::WCET_US;

    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> Self::Out {
        let PickCycle {
            arm_a,
            arm_b,
            permit,
        } = self;

        emit(0, Kind::TaskRise, 0, 0)
            .then(Enter::new(arm_a, permit, APPROACH_UM))
            .bind(|a| Work::new(a, PICK_UM))
            .bind(|a| delay_us(HOLD_US).map(move |_| a))
            .bind(Leave::new)
            .bind(|r| emit(0, Kind::TaskFall, 0, 0).map(move |_| r))
            .bind(move |(a, permit)| {
                emit(1, Kind::TaskRise, 0, 0)
                    .then(Enter::new(arm_b, permit, APPROACH_UM))
                    .bind(|b| Work::new(b, PICK_UM))
                    .bind(|b| delay_us(HOLD_US).map(move |_| b))
                    .bind(Leave::new)
                    .bind(|r| emit(1, Kind::TaskFall, 0, 0).map(move |_| r))
                    .map(move |(b, permit)| (a, b, permit))
            })
            .run(ctx)
    }
}

/// 段 1+2: レガシーの外側手順をそのまま走らせる。
///
/// `stagger_us` は 2 本目の起動遅れ。現場で「時間差でかわしている」あれで、
/// 0 にすると素の競合が出る。
pub fn run_legacy(scene: &'static SceneDef, label: &'static str, stagger_us: u64) -> RunReport {
    let plant = acquire();
    let world = SharedWorld::new(scene);
    plant.set_io(Box::new(world.clone()));

    let sched = legacy_bridge::run_legacy_pick_pair(
        &plant,
        APPROACH_UM,
        HOLD_US as u32,
        stagger_us,
        SIM_BUDGET_US,
    );

    let t_end = sched.now_us();
    let mut w = world.lock();
    w.finish(t_end);
    w.report(label, 0, sched.overran())
}

/// 赤: 時間差なし。2 本が同じタイミングでゾーンを狙う。
pub fn run_legacy_raw(scene: &'static SceneDef) -> RunReport {
    run_legacy(scene, "legacy-raw", 0)
}

/// 現場の運用: 2 本目を 40 ms 遅らせて「かわす」。
///
/// これは修正ではない。たまたま当たっている時間差で、
/// サーボを速い型に替えれば当たらなくなる。
pub fn run_legacy_staggered(scene: &'static SceneDef) -> RunReport {
    run_legacy(scene, "legacy-staggered", 40_000)
}

/// 緑: 手順だけ型安全 SDK へ移す。**軸を動かすのは同じレガシー関数**。
pub fn run_typed_sdk(scene: &'static SceneDef) -> RunReport {
    let plant = acquire();
    let world = SharedWorld::new(scene);
    plant.set_io(Box::new(world.clone()));
    plant.use_direct_clock(0);

    let il = Interlock::take().expect("許可証は設備起動時に 1 度だけ組み立てる");
    let cycle = PickCycle {
        arm_a: Arm::new(LegacyAxis::at::<0>(), 0),
        arm_b: Arm::new(LegacyAxis::at::<1>(), 0),
        permit: il.zone_x,
    };

    let mut ctx = Ctx::new(0, TICK_US, world.clone());
    let _ = cycle.run(&mut ctx);

    let t_end = ctx.now();
    let cpu = ctx.cpu_us();
    let mut w = world.lock();
    w.finish(t_end);
    w.report("typed-sdk", cpu, false)
}
