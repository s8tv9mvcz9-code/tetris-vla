//! 空間の意味を型で持つ機構モデル。
//!
//! ここには実行時の排他フラグが**無い**。ゾーンへの立ち入りは
//! [`Permit<Z>`](Permit) という複製できない値の所有で表し、機構の状態は
//! [`Arm<ID, REACH, A, S>`](Arm) の型引数 `S` に出す。結果として、
//!
//! * 同じゾーンに 2 つの機構が入るコードは**書けない** (渡すトークンが無い)、
//! * ゾーンに入っている間に別の機構を動かす関数は**呼べない** (借りる先が無い)、
//! * 届かないアームにゾーン作業を割り付けると**ビルドが落ちる** (可達長の静的検査)。
//!
//! 干渉検知は、走らせて初めて分かるものから、走らせる前に分かるものに変わる。
//!
//! 時間は [`vtime`] から注入する。この層も `no_std` で、確保も実時計も持たない。
#![no_std]
#![forbid(unsafe_code)]

pub mod arm;
pub mod geom;
pub mod prog;
pub mod zone;

pub use arm::{Arm, Axis, Inside, NullAxis, Outside, CMD_WCET_US};
pub use geom::{dist2_um, dist_um, Aabb, Um, P3};
pub use prog::{Enter, Leave, MoveOutside, Work};
pub use zone::{conveyor_advance_while_free, Interlock, Permit, Zone, ZoneX, ZoneY};

/// アームがゾーンに入っている間、そのゾーンが空いている証拠は誰も出せない。
///
/// ```compile_fail
/// use zoneguard::{conveyor_advance_while_free, Arm, Interlock, NullAxis, ZoneX};
/// use vtime::Ctx;
///
/// let il = Interlock::take().unwrap();
/// let mut ctx = Ctx::new(0, 0, ());
/// let a: Arm<0, 300_000, _, _> = Arm::new(NullAxis::new(4), 0);
/// let a = a.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
/// // アームが中に居る = 許可証は型状態の中。コンベアは動かせない。
/// conveyor_advance_while_free(&il.zone_x, 10_000);
/// ```
///
/// 退出して許可証が戻れば、同じ行が通る。
///
/// ```
/// use zoneguard::{conveyor_advance_while_free, Arm, Interlock, NullAxis, ZoneX};
/// use vtime::Ctx;
///
/// let il = Interlock::take().unwrap();
/// let mut ctx = Ctx::new(0, 0, ());
/// let a: Arm<0, 300_000, _, _> = Arm::new(NullAxis::new(4), 0);
/// let a = a.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
/// let (_a, permit) = a.leave(&mut ctx);
/// assert_eq!(conveyor_advance_while_free(&permit, 10_000), 10_000);
/// # Interlock::rearm_for_harness();
/// ```
pub mod _compile_time_proofs {}
