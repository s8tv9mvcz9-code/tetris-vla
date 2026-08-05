//! C++ 資産の生の入口。**このモジュールの外に `unsafe` を漏らさない**。
//!
//! ここに書いてあるのは、資産が守っている前提の一覧でもある。
//! 破れば壊れる約束を、Rust 側の型と橋渡し層で肩代わりする。

use core::ffi::c_void;

pub type SleepFn = extern "C" fn(u32);
pub type AxisFn = extern "C" fn(i32, i32);
pub type MarkFn = extern "C" fn(i32, i32);

extern "C" {
    /// 待ちと軸書き込みの差し替え。実時計を殺すための唯一の改造点。
    pub fn legacy_install_hooks(s: SleepFn, w: AxisFn, m: MarkFn);
    /// プロセス内グローバルな軸位置とフラグを初期化する。
    pub fn legacy_reset();
    pub fn legacy_axis_pos(axis: i32) -> i32;
    pub fn legacy_zone_busy() -> i32;
    /// 残す資産: 内側の送りループ。
    pub fn legacy_arm_move(axis: i32, target_um: i32);
    /// 捨てる資産: 排他のつもりの外側手順。
    pub fn legacy_pick_sequence(axis: i32, zone_um: i32, hold_us: u32);
}

/// 資産が暗黙に要求している前提。橋渡し層はこれを満たす責任を負う。
///
/// 1. `g_pos` / `g_busy` は **プロセス内グローバル**。同時に 2 つのシナリオを
///    走らせてはならない (→ [`crate::plant::acquire`] がプロセス全体を直列化する)。
/// 2. フックを差す前に `legacy_arm_move` を呼ぶと、待ちも記録も消える
///    (→ `acquire` が必ず先に差す)。
/// 3. `axis` は 0 か 1 のみ。範囲外は C 側で握り潰されるが、
///    Rust 側では [`crate::axis::LegacyAxis::new`] が型で絞る。
/// 4. 再入不可。1 つの軸に対して 2 か所から同時に指令を出してはならない
///    (→ 仮想時間スケジューラが実行権を 1 つしか配らない)。
pub const CONTRACT: &str = "process-global, hooks-first, axis in 0..2, non-reentrant";

/// リンクが生きていることの最小確認 (テストから使う)。
pub(crate) fn _link_probe() -> *const c_void {
    legacy_reset as *const c_void
}
