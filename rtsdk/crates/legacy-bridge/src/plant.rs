//! プロセス内にただ 1 つある「設備」。
//!
//! レガシー C++ は状態をグローバルに持っている。それは資産の性質であって
//! 直す対象ではないので、**その性質を型で囲う**。[`acquire`] が返すガードを
//! 持っている間だけ設備に触れてよく、ガードはプロセス全体で 1 つしか取れない。
//! テストが並列に走っても、シナリオ同士は必ず直列になる。

use std::sync::{Arc, Mutex, MutexGuard};

use crate::ffi;
use crate::vsched::{current_task, Sched};

/// 設備から外へ出る観測。シミュレータ側 (`sim-harness`) がこれを実装して差す。
pub trait PlantIo: Send {
    /// 軸が動いた。`t_us` は仮想時間。
    fn write_axis(&mut self, t_us: u64, axis: i32, pos_um: i32);
    /// レガシー側の手順マーカ (1 = 手順開始、0 = 手順終了)。
    fn mark(&mut self, t_us: u64, axis: i32, code: i32);
}

enum TimeBase {
    /// 単一スレッドで回すとき。待ちはその場で時計を進めるだけ。
    Direct(u64),
    /// 複数のレガシータスクを並べ替えるとき。待ちは離散事象になる。
    Scheduled(Arc<Sched>),
}

static TIME: Mutex<TimeBase> = Mutex::new(TimeBase::Direct(0));
static IO: Mutex<Option<Box<dyn PlantIo>>> = Mutex::new(None);
static PLANT: Mutex<()> = Mutex::new(());

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// 現在の仮想時刻 [µs]。実時計は読まない。
pub fn now_us() -> u64 {
    match &*lock(&TIME) {
        TimeBase::Direct(t) => *t,
        TimeBase::Scheduled(s) => s.now_us(),
    }
}

extern "C" fn hook_sleep(us: u32) {
    // ここが usleep() だった場所。実時間は 1 µs も進めない。
    let mut g = lock(&TIME);
    match &mut *g {
        TimeBase::Direct(t) => {
            *t += us as u64;
        }
        TimeBase::Scheduled(s) => {
            let s = Arc::clone(s);
            let id = current_task().unwrap_or(0);
            let until = s.now_us() + us as u64;
            drop(g); // 待ちに入る前に必ず手放す
            s.park(id, until);
        }
    }
}

extern "C" fn hook_axis(axis: i32, pos_um: i32) {
    let t = now_us();
    if let Some(io) = lock(&IO).as_mut() {
        io.write_axis(t, axis, pos_um);
    }
}

extern "C" fn hook_mark(axis: i32, code: i32) {
    let t = now_us();
    if let Some(io) = lock(&IO).as_mut() {
        io.mark(t, axis, code);
    }
}

/// 設備を握っている証。落とすと設備は初期状態に戻る。
pub struct Plant {
    _g: MutexGuard<'static, ()>,
}

/// 設備を握る。他のシナリオが握っていれば、離すまで待つ。
///
/// 握った時点で
/// * レガシー側のグローバル (軸位置・排他フラグ) を初期化し、
/// * 実時計を仮想時計に差し替え、
/// * ゾーンの許可証を刷り直す。
pub fn acquire() -> Plant {
    let g = lock(&PLANT);
    // SAFETY: フックの差し替えと初期化は、設備ロックを握った状態でしか呼ばない。
    // 3 つの関数ポインタはいずれも 'static で、C 側は保存するだけ。
    unsafe {
        ffi::legacy_install_hooks(hook_sleep, hook_axis, hook_mark);
        ffi::legacy_reset();
    }
    *lock(&TIME) = TimeBase::Direct(0);
    *lock(&IO) = None;
    zoneguard::Interlock::rearm_for_harness();
    Plant { _g: g }
}

impl Plant {
    /// 観測の出口を差す。
    pub fn set_io(&self, io: Box<dyn PlantIo>) {
        *lock(&IO) = Some(io);
    }

    /// 単一スレッドの直接時計に切り替え、時刻を合わせる。
    pub fn use_direct_clock(&self, now_us: u64) {
        *lock(&TIME) = TimeBase::Direct(now_us);
    }

    /// 離散事象スケジューラに切り替える。
    pub fn use_scheduler(&self, s: &Arc<Sched>) {
        *lock(&TIME) = TimeBase::Scheduled(Arc::clone(s));
    }

    pub fn now_us(&self) -> u64 {
        now_us()
    }

    pub fn axis_pos_um(&self, axis: i32) -> i32 {
        // SAFETY: 設備ロックを握っており、C 側は読み出しだけ。
        unsafe { ffi::legacy_axis_pos(axis) }
    }

    /// レガシー側が「ゾーンは使用中」と思っているか。
    /// この値と実際の干渉が食い違うことこそが、この PoC の出発点。
    pub fn legacy_thinks_zone_busy(&self) -> bool {
        // SAFETY: 同上。
        unsafe { ffi::legacy_zone_busy() != 0 }
    }
}

impl Drop for Plant {
    fn drop(&mut self) {
        *lock(&IO) = None;
        *lock(&TIME) = TimeBase::Direct(0);
        zoneguard::Interlock::rearm_for_harness();
    }
}

/// 単一スレッドで直接時計を進める (SDK 側の軸実装から使う)。
pub(crate) fn sync_direct_clock(to_us: u64) {
    let mut g = lock(&TIME);
    if let TimeBase::Direct(t) = &mut *g {
        *t = to_us;
    }
}
