//! 型状態を仮想時間モナドに載せるための手続き部品。
//!
//! [`crate::Arm`] のメソッドは `&mut Ctx` を取る素の関数だが、それだけだと
//! 「合成した手順全体の WCET」が型から取れない。そこで 1 動作 = 1 つの [`Prog`] にして、
//! `bind` で繋げるようにする。
//!
//! こうすると 2 つが同時に成り立つ。
//!
//! * **型状態が bind を通って流れる。** `Enter` の出力は `Arm<.., Inside<Z>>` で、
//!   次に繋げられるのは `Inside` を受ける手続きだけ。順序違いは型エラーになる。
//! * **WCET が合成で足される。** 手順全体の `WCET_US` がコンパイル時に決まり、
//!   [`vtime::Budget`] で制御周期と突き合わせられる。

use pulse_trace::Sink;
use vtime::{Ctx, Micros, Prog};

use crate::arm::{Arm, Axis, Inside, Outside, CMD_WCET_US};
use crate::geom::Um;
use crate::zone::{Permit, Zone};

/// ゾーンへ入る 1 動作。許可証を消費する。
pub struct Enter<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> {
    arm: Arm<ID, REACH_UM, A, Outside>,
    permit: Permit<Z>,
    target_um: Um,
}

impl<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> Enter<ID, REACH_UM, A, Z> {
    pub const fn new(arm: Arm<ID, REACH_UM, A, Outside>, permit: Permit<Z>, target_um: Um) -> Self {
        Enter {
            arm,
            permit,
            target_um,
        }
    }
}

impl<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> Prog for Enter<ID, REACH_UM, A, Z> {
    type Out = Arm<ID, REACH_UM, A, Inside<Z>>;
    const WCET_US: Micros = CMD_WCET_US;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> Self::Out {
        self.arm.enter::<Z, S>(self.permit, self.target_um, ctx)
    }
}

/// ゾーン内での 1 動作。
pub struct Work<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> {
    arm: Arm<ID, REACH_UM, A, Inside<Z>>,
    target_um: Um,
}

impl<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> Work<ID, REACH_UM, A, Z> {
    pub const fn new(arm: Arm<ID, REACH_UM, A, Inside<Z>>, target_um: Um) -> Self {
        Work { arm, target_um }
    }
}

impl<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> Prog for Work<ID, REACH_UM, A, Z> {
    type Out = Arm<ID, REACH_UM, A, Inside<Z>>;
    const WCET_US: Micros = CMD_WCET_US;
    fn run<S: Sink>(mut self, ctx: &mut Ctx<S>) -> Self::Out {
        self.arm.work_at(self.target_um, ctx);
        self.arm
    }
}

/// ゾーンからの退出。許可証が戻る。
pub struct Leave<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> {
    arm: Arm<ID, REACH_UM, A, Inside<Z>>,
}

impl<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> Leave<ID, REACH_UM, A, Z> {
    pub const fn new(arm: Arm<ID, REACH_UM, A, Inside<Z>>) -> Self {
        Leave { arm }
    }
}

impl<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> Prog for Leave<ID, REACH_UM, A, Z> {
    type Out = (Arm<ID, REACH_UM, A, Outside>, Permit<Z>);
    const WCET_US: Micros = CMD_WCET_US;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> Self::Out {
        self.arm.leave(ctx)
    }
}

/// ゾーンの外での 1 動作。許可証は要らない。
pub struct MoveOutside<const ID: u16, const REACH_UM: i32, A: Axis> {
    arm: Arm<ID, REACH_UM, A, Outside>,
    target_um: Um,
}

impl<const ID: u16, const REACH_UM: i32, A: Axis> MoveOutside<ID, REACH_UM, A> {
    pub const fn new(arm: Arm<ID, REACH_UM, A, Outside>, target_um: Um) -> Self {
        MoveOutside { arm, target_um }
    }
}

impl<const ID: u16, const REACH_UM: i32, A: Axis> Prog for MoveOutside<ID, REACH_UM, A> {
    type Out = Arm<ID, REACH_UM, A, Outside>;
    const WCET_US: Micros = CMD_WCET_US;
    fn run<S: Sink>(mut self, ctx: &mut Ctx<S>) -> Self::Out {
        self.arm.move_outside(self.target_um, ctx);
        self.arm
    }
}
