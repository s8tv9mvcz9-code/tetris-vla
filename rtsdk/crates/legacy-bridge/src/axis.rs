//! レガシーの送りループを、型安全 SDK の軸として使えるようにする層。
//!
//! 包み方の要点は 2 つ。
//!
//! * **内側は捨てない。** 実際に動かすのは `legacy_arm_move` のまま。
//!   加減速もサーボの癖もそこに入っている。
//! * **外側の時間だけ奪う。** 呼ぶ前に仮想時計を SDK の現在時刻に合わせ、
//!   呼び終わったら「何 µs 消費したか」を返す。レガシー側の待ちは
//!   SDK のモナドの時計にそのまま合流する。

use zoneguard::{Axis, Um};

use crate::ffi;
use crate::plant;

/// レガシー資産で駆動される 1 軸。
#[derive(Clone, Copy, Debug)]
pub struct LegacyAxis {
    axis: i32,
}

impl LegacyAxis {
    /// 軸番号は 0 か 1。C 側の配列長がそうなっている (`ffi::CONTRACT` 参照)。
    pub const fn new(axis: u8) -> Option<LegacyAxis> {
        if (axis as usize) < 2 {
            Some(LegacyAxis { axis: axis as i32 })
        } else {
            None
        }
    }

    /// 軸番号が定数で分かっているとき用。範囲外はビルドが落ちる。
    pub const fn at<const AXIS: u8>() -> LegacyAxis {
        assert!((AXIS as usize) < 2, "レガシー側の軸は 0 と 1 だけ");
        LegacyAxis { axis: AXIS as i32 }
    }
}

impl Axis for LegacyAxis {
    fn move_to(&mut self, target_um: Um, now_us: u64) -> u64 {
        plant::sync_direct_clock(now_us);
        // SAFETY: 設備ロック (plant::acquire) の下でのみ呼ばれ、軸番号は 0..2 に
        // 絞ってある。再入は仮想時間スケジューラが実行権を 1 枚しか配らないことで
        // 防いでいる。C 側は待ちも書き込みもフック経由で、実時計には触れない。
        unsafe { ffi::legacy_arm_move(self.axis, target_um) };
        plant::now_us().saturating_sub(now_us)
    }

    fn pos_um(&self) -> Um {
        // SAFETY: 読み出しのみ。
        unsafe { ffi::legacy_axis_pos(self.axis) }
    }
}
