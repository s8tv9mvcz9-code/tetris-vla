//! アームの型状態 (Typestate)。
//!
//! `Arm<ID, REACH, A, Outside>` と `Arm<ID, REACH, A, Inside<Z>>` は**別の型**で、
//! 呼べる操作が違う。状態遷移は値を消費して別の型を返すので、
//! 「入ったまま入る」「出ていないのに出る」は書けない。
//!
//! さらに `Inside<Z>` は許可証 `Permit<Z>` を飲み込んで持つ。だから
//! 「もう 1 本のアームを同じゾーンへ入れる」コードは、渡すトークンが無くて
//! コンパイルが通らない。

use core::marker::PhantomData;

use pulse_trace::{Kind, Sink};
use vtime::{Ctx, Micros};

use crate::geom::Um;
use crate::zone::{Permit, Zone};

/// 実際に軸を動かす先。レガシー C++ でも、実機の EtherCAT スレーブでもよい。
///
/// `now_us` を受けて「消費した仮想時間 [µs]」を返す — 下位が勝手に実時間で
/// 眠らないための約束。実装側は自分の中の時計をこの `now_us` に合わせてから動く。
pub trait Axis {
    fn move_to(&mut self, target_um: Um, now_us: Micros) -> Micros;
    fn pos_um(&self) -> Um;
}

/// 何もしない軸。ドキュメントとユニットテスト用。
#[derive(Default, Clone, Copy, Debug)]
pub struct NullAxis {
    pos: Um,
    /// 1 µm あたりの所要時間 [ns 相当の整数]。0 なら瞬時。
    pub us_per_mm: Micros,
}

impl NullAxis {
    pub const fn new(us_per_mm: Micros) -> Self {
        NullAxis { pos: 0, us_per_mm }
    }
}

impl Axis for NullAxis {
    fn move_to(&mut self, target_um: Um, _now_us: Micros) -> Micros {
        let d = (target_um - self.pos).unsigned_abs() as u64;
        self.pos = target_um;
        d * self.us_per_mm / 1000
    }

    fn pos_um(&self) -> Um {
        self.pos
    }
}

/// 動作指令 1 回あたりの最悪計算時間 [µs]。
///
/// 指令の組み立て、下位への書き込み、トレースへの記録まで含む。
/// 軸が実際に動いている時間 (設備側の時間) はここには入らない —
/// 制御タスクが CPU を握っている時間だけを数える。
pub const CMD_WCET_US: Micros = 12;

/// 型状態: ゾーンの外。
#[derive(Debug)]
pub struct Outside;

/// 型状態: ゾーン `Z` の中。許可証はここに封じ込められている。
///
/// フィールドは私有。`Inside` から `&Permit<Z>` を取り出す口はわざと用意していない —
/// 出せてしまうと「中に居るのにゾーンが空いている証拠」を配れてしまう。
#[derive(Debug)]
pub struct Inside<Z: Zone> {
    permit: Permit<Z>,
}

/// 1 本のアーム。
///
/// * `ID` … トレース上のレーン番号。
/// * `REACH_UM` … 可達長 [µm]。ゾーンへ届くかの静的検査に使う (依存型の代わり)。
pub struct Arm<const ID: u16, const REACH_UM: i32, A: Axis, S> {
    axis: A,
    home_um: Um,
    state: S,
}

/// 「このアームはこのゾーンへ物理的に届くか」を単相化時に評価する検査。
///
/// 値としては `()` しか作らないが、評価された時点で `assert!` が走るので、
/// 届かない組み合わせはビルドが落ちる。const generics で依存型の真似をする常套手段。
struct Reach<const REACH_UM: i32, Z: Zone>(PhantomData<fn() -> Z>);

impl<const REACH_UM: i32, Z: Zone> Reach<REACH_UM, Z> {
    const COVERS_ZONE: () = assert!(
        REACH_UM >= Z::BOX.far_x(),
        "アームの可達長がゾーンの遠端に届いていない: この組み合わせは機械的に成立しない"
    );
}

impl<const ID: u16, const REACH_UM: i32, A: Axis> Arm<ID, REACH_UM, A, Outside> {
    /// 動作指令 1 回あたりの最悪計算時間 [µs]。
    pub const CMD_WCET_US: Micros = CMD_WCET_US;

    pub const fn new(axis: A, home_um: Um) -> Self {
        Arm {
            axis,
            home_um,
            state: Outside,
        }
    }

    pub fn pos_um(&self) -> Um {
        self.axis.pos_um()
    }

    /// ゾーンの外での移動。許可証は要らないが、ゾーンへは入れない。
    pub fn move_outside<S: Sink>(&mut self, target_um: Um, ctx: &mut Ctx<S>) {
        let el = self.axis.move_to(target_um, ctx.now());
        ctx.charge_us(Self::CMD_WCET_US);
        ctx.advance(el);
        ctx.emit(ID, Kind::AxisPos, self.axis.pos_um(), 0);
    }

    /// ゾーン `Z` へ入る。**許可証を消費する** ので、
    /// これ以降その許可証は他の誰にも渡せない。
    ///
    /// ```
    /// use zoneguard::{Arm, Interlock, NullAxis, ZoneX};
    /// use vtime::Ctx;
    ///
    /// let il = Interlock::take().unwrap();
    /// let mut ctx = Ctx::new(0, 0, ());
    /// let arm_a: Arm<0, 300_000, _, _> = Arm::new(NullAxis::new(4), 0);
    /// let arm_a = arm_a.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
    /// assert_eq!(arm_a.pos_um(), 200_000);
    /// # Interlock::rearm_for_harness();
    /// ```
    ///
    /// 2 本目のアームを同じゾーンへ入れようとすると、渡す許可証が無い:
    ///
    /// ```compile_fail
    /// use zoneguard::{Arm, Interlock, NullAxis, ZoneX};
    /// use vtime::Ctx;
    ///
    /// let il = Interlock::take().unwrap();
    /// let mut ctx = Ctx::new(0, 0, ());
    /// let arm_a: Arm<0, 300_000, _, _> = Arm::new(NullAxis::new(4), 0);
    /// let arm_b: Arm<1, 300_000, _, _> = Arm::new(NullAxis::new(4), 400_000);
    /// let arm_a = arm_a.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
    /// // il.zone_x は arm_a の型状態へ移動済み。ここで借りることも動かすこともできない。
    /// let arm_b = arm_b.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
    /// ```
    ///
    /// 可達長が足りない組み合わせも、走らせる前に落ちる:
    ///
    /// ```compile_fail
    /// use zoneguard::{Arm, Interlock, NullAxis, ZoneX};
    /// use vtime::Ctx;
    ///
    /// let il = Interlock::take().unwrap();
    /// let mut ctx = Ctx::new(0, 0, ());
    /// // ZoneX の遠端は 250 mm。100 mm しか伸びないアームでは届かない。
    /// let short: Arm<2, 100_000, _, _> = Arm::new(NullAxis::new(4), 0);
    /// let short = short.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
    /// ```
    pub fn enter<Z: Zone, S: Sink>(
        mut self,
        permit: Permit<Z>,
        target_um: Um,
        ctx: &mut Ctx<S>,
    ) -> Arm<ID, REACH_UM, A, Inside<Z>> {
        // 機械的な成立性の検査。ここで型引数の組み合わせが評価される。
        let () = Reach::<REACH_UM, Z>::COVERS_ZONE;

        ctx.charge_us(Self::CMD_WCET_US);
        ctx.emit(ID, Kind::PermitTake, Z::ID as i32, 0);
        ctx.emit(ID, Kind::ZoneEnter, Z::ID as i32, target_um);
        let el = self.axis.move_to(target_um, ctx.now());
        ctx.advance(el);
        ctx.emit(ID, Kind::AxisPos, self.axis.pos_um(), 0);

        Arm {
            axis: self.axis,
            home_um: self.home_um,
            state: Inside { permit },
        }
    }
}

impl<const ID: u16, const REACH_UM: i32, A: Axis, Z: Zone> Arm<ID, REACH_UM, A, Inside<Z>> {
    pub const CMD_WCET_US: Micros = CMD_WCET_US;

    pub fn pos_um(&self) -> Um {
        self.axis.pos_um()
    }

    pub const fn zone_id(&self) -> u16 {
        Z::ID
    }

    /// ゾーンの中での作業。ゾーンの外へ出る指令はここからは出せない
    /// (境界の外を指定したら、その場で止める)。
    pub fn work_at<S: Sink>(&mut self, target_um: Um, ctx: &mut Ctx<S>) {
        let clamped = clamp_into_zone::<Z>(target_um);
        let el = self.axis.move_to(clamped, ctx.now());
        ctx.charge_us(Self::CMD_WCET_US);
        ctx.advance(el);
        ctx.emit(ID, Kind::AxisPos, self.axis.pos_um(), 0);
    }

    /// 退出。許可証が戻ってくる — 戻して初めて、他の機構が入れるようになる。
    pub fn leave<S: Sink>(
        mut self,
        ctx: &mut Ctx<S>,
    ) -> (Arm<ID, REACH_UM, A, Outside>, Permit<Z>) {
        ctx.charge_us(Self::CMD_WCET_US);
        let el = self.axis.move_to(self.home_um, ctx.now());
        ctx.advance(el);
        ctx.emit(ID, Kind::AxisPos, self.axis.pos_um(), 0);
        ctx.emit(ID, Kind::ZoneExit, Z::ID as i32, 0);
        ctx.emit(ID, Kind::PermitDrop, Z::ID as i32, 0);
        (
            Arm {
                axis: self.axis,
                home_um: self.home_um,
                state: Outside,
            },
            self.state.permit,
        )
    }
}

const fn clamp_into_zone<Z: Zone>(x: Um) -> Um {
    if x < Z::BOX.min.x {
        Z::BOX.min.x
    } else if x > Z::BOX.max.x {
        Z::BOX.max.x
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zone::{Interlock, ZoneX};
    use pulse_trace::Recorder;

    #[test]
    fn entering_and_leaving_returns_the_permit() {
        Interlock::rearm_for_harness();
        let il = Interlock::take().unwrap();
        let mut rec = Recorder::<32>::new();
        let mut ctx = Ctx::new(0, 0, &mut rec);

        let a: Arm<0, 300_000, _, _> = Arm::new(NullAxis::new(4), 0);
        let a = a.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
        assert_eq!(a.zone_id(), ZoneX::ID);
        let (a, permit) = a.leave(&mut ctx);
        assert_eq!(a.pos_um(), 0);
        assert_eq!(permit.zone_id(), ZoneX::ID);

        assert_eq!(rec.count(Kind::ZoneEnter), 1);
        assert_eq!(rec.count(Kind::ZoneExit), 1);
        Interlock::rearm_for_harness();
    }

    #[test]
    fn work_stays_inside_the_zone_boundary() {
        Interlock::rearm_for_harness();
        let il = Interlock::take().unwrap();
        let mut ctx = Ctx::new(0, 0, ());
        let a: Arm<0, 300_000, _, _> = Arm::new(NullAxis::new(4), 0);
        let mut a = a.enter::<ZoneX, _>(il.zone_x, 200_000, &mut ctx);
        a.work_at(900_000, &mut ctx); // ゾーンの外を指しても
        assert_eq!(a.pos_um(), ZoneX::BOX.max.x); // 境界で止まる
        Interlock::rearm_for_harness();
    }
}
