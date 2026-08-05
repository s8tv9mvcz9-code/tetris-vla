//! 既存 C/C++ 資産を、書き換えずに「型安全なテストの器」へ入れるための橋渡し。
//!
//! 包む順序は 3 段階で、**どの段階でも資産のソースは 1 行も変えない**
//! (唯一の改造は待ちと軸書き込みを関数ポインタにしたこと)。
//!
//! | 段 | やること | 得られるもの |
//! |---|---|---|
//! | 1 | `unsafe extern` で生のまま呼ぶ ([`ffi`]) | 資産が守っている前提が文章になる |
//! | 2 | 実時計を仮想時間に差し替える ([`plant`], [`vsched`]) | 決定的に再現する。実時間を食わない |
//! | 3 | 手順の側だけ型安全 SDK に移す ([`axis`]) | 干渉がコンパイル時に落ちる |
//!
//! 段 2 まで来た時点で、レガシーのバグは「たまに出る」から
//! 「毎回同じ時刻に出る反例」に変わる。段 3 でそれが消える。

pub mod axis;
pub mod ffi;
pub mod plant;
pub mod vsched;

pub use axis::LegacyAxis;
pub use plant::{acquire, Plant, PlantIo};
pub use vsched::Sched;

use std::sync::Arc;

/// 段 1 + 2: レガシーの外側手順 (`legacy_pick_sequence`) を 2 軸ぶん、
/// 仮想時間の上で同時に走らせる。
///
/// これが「今そこにある設備」の再現。2 本のアームは同じゾーンを狙い、
/// 排他は `g_busy` 1 個。何が起きるかは走らせれば分かる。
///
/// * `zone_um` … ゾーン内の作業点 (軸座標)
/// * `hold_us` … ゾーン内での保持時間
/// * `stagger_us` … 2 本目の起動遅れ。現場で「時間差でかわしている」あれ
/// * `budget_us` … 仮想時間の打ち切り
pub fn run_legacy_pick_pair(
    plant: &Plant,
    zone_um: i32,
    hold_us: u32,
    stagger_us: u64,
    budget_us: u64,
) -> Arc<Sched> {
    let sched = Sched::new(&[0, stagger_us], budget_us);
    plant.use_scheduler(&sched);

    let mut specs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
    for axis in 0..2i32 {
        specs.push(Box::new(move || {
            // SAFETY: 設備ロックの下で、仮想時間スケジューラが実行権を 1 枚しか
            // 配らない状態で呼んでいる。軸番号は 0/1。
            // ここで守られていないのは C 側の排他だけ — それを見にきている。
            unsafe { ffi::legacy_pick_sequence(axis, zone_um, hold_us) };
        }));
    }
    vsched::run_tasks(&sched, specs);
    sched
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Log {
        axis: Vec<(u64, i32, i32)>,
    }

    struct LogIo(Arc<Mutex<Log>>);

    impl PlantIo for LogIo {
        fn write_axis(&mut self, t_us: u64, axis: i32, pos_um: i32) {
            self.0.lock().unwrap().axis.push((t_us, axis, pos_um));
        }
        fn mark(&mut self, _t_us: u64, _axis: i32, _code: i32) {}
    }

    #[test]
    fn legacy_moves_without_touching_the_wall_clock() {
        let plant = acquire();
        let log = Arc::new(Mutex::new(Log::default()));
        plant.set_io(Box::new(LogIo(Arc::clone(&log))));
        plant.use_direct_clock(0);

        let wall = std::time::Instant::now();
        // SAFETY: 設備ロック下、単一スレッド、軸 0。
        unsafe { ffi::legacy_arm_move(0, 200_000) };

        assert_eq!(plant.axis_pos_um(0), 200_000);
        // 2 mm/送り × 200 µs = 100 送り、仮想 20 ms。
        assert_eq!(plant.now_us(), 20_000);
        assert_eq!(log.lock().unwrap().axis.len(), 100);
        assert!(wall.elapsed().as_millis() < 500, "実時間で眠っている");
    }

    #[test]
    fn the_legacy_interlock_lets_both_arms_in() {
        let plant = acquire();
        plant.use_direct_clock(0);
        let sched = run_legacy_pick_pair(&plant, 200_000, 3_000, 0, 10_000_000);
        assert!(!sched.overran());
        // 手順としては両方とも「完了」する。C 側から見れば異常は無い。
        assert_eq!(plant.axis_pos_um(0), 0);
        assert_eq!(plant.axis_pos_um(1), 0);
        assert!(!plant.legacy_thinks_zone_busy());
    }
}
