//! ブロックするレガシーコードを、決定的な離散事象で回すための仮想時間スケジューラ。
//!
//! レガシー側は「待つ」と書いてある。書き換えずに待たせるには、待ちを
//! **実時間の睡眠ではなく事象の並べ替え**に変換するしかない。
//!
//! やり方は素朴で、実行権 (バトン) をプロセス内に 1 枚しか置かない。
//!
//! * タスクは待ちに入るとき「いつ起きたいか」を置いてバトンを返す。
//! * ドライバは *起床時刻の小さい順、同着ならタスク番号順* に 1 つだけ選び、
//!   仮想時計をその時刻へ進めてバトンを渡す。
//!
//! 並行に走るスレッドは常に 1 つなので、
//!
//! * 実時間は 1 マイクロ秒も消費しない (20 秒のタクトが数ミリ秒で回る)、
//! * 同じ入力なら**必ず同じ順序**で再現する (競合を「たまに再現するバグ」から
//!   「毎回再現する反例」に変えられる)。
//!
//! 産業用 IPC の PREEMPT_RT 上では、この並べ替えを行うのは Linux のスケジューラで、
//! そこは決定的ではない。だからここでは**最悪の並び**を人為的に作って先に踏む。

use std::cell::Cell;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum St {
    Parked,
    Running,
    Done,
}

struct Slot {
    wake_at: u64,
    st: St,
}

struct Inner {
    now_us: u64,
    budget_us: u64,
    overrun: bool,
    baton: Option<usize>,
    tasks: Vec<Slot>,
}

pub struct Sched {
    m: Mutex<Inner>,
    cv: Condvar,
}

thread_local! {
    static CURRENT_TASK: Cell<Option<usize>> = const { Cell::new(None) };
}

/// いま走っているタスク番号。フックはこれで自分が誰かを知る。
pub fn current_task() -> Option<usize> {
    CURRENT_TASK.with(|c| c.get())
}

fn lock(m: &Mutex<Inner>) -> MutexGuard<'_, Inner> {
    // シナリオが panic した後も、ハーネスとしては続きを診たい。
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Sched {
    /// `start_at[i]` はタスク `i` の起動時刻 [µs]。`budget_us` は仮想時間の打ち切り。
    pub fn new(start_at: &[u64], budget_us: u64) -> Arc<Sched> {
        Arc::new(Sched {
            m: Mutex::new(Inner {
                now_us: 0,
                budget_us,
                overrun: false,
                baton: None,
                tasks: start_at
                    .iter()
                    .map(|&t| Slot {
                        wake_at: t,
                        st: St::Parked,
                    })
                    .collect(),
            }),
            cv: Condvar::new(),
        })
    }

    pub fn now_us(&self) -> u64 {
        lock(&self.m).now_us
    }

    /// 仮想時間の予算を使い切ったか。使い切った後は時計が止まったまま走り切る。
    pub fn overran(&self) -> bool {
        lock(&self.m).overrun
    }

    /// 待ちに入る。バトンを返し、`until_us` に起こされるまで戻らない。
    pub fn park(&self, id: usize, until_us: u64) {
        let mut g = lock(&self.m);
        g.tasks[id].wake_at = until_us;
        g.tasks[id].st = St::Parked;
        g.baton = None;
        self.cv.notify_all();
        while g.baton != Some(id) {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// 最初のバトンを待つ (タスクスレッドの入口)。
    fn wait_for_start(&self, id: usize) {
        let mut g = lock(&self.m);
        while g.baton != Some(id) {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn finish(&self, id: usize) {
        let mut g = lock(&self.m);
        g.tasks[id].st = St::Done;
        g.baton = None;
        self.cv.notify_all();
    }

    /// ドライバ。全タスクが終わるまで、バトンを 1 枚ずつ配り続ける。
    fn drive(&self) {
        let mut g = lock(&self.m);
        loop {
            while g.baton.is_some() {
                g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            let next = g
                .tasks
                .iter()
                .enumerate()
                .filter(|(_, s)| s.st == St::Parked)
                .min_by_key(|(i, s)| (s.wake_at, *i))
                .map(|(i, _)| i);
            let Some(i) = next else { return };

            let w = g.tasks[i].wake_at;
            if w > g.budget_us {
                // 予算超過。時計は止めるが、C++ の呼び出し途中で投げ出すわけには
                // いかないので、走っているものは走り切らせて事実として記録する。
                g.overrun = true;
                g.now_us = g.budget_us;
            } else if w > g.now_us {
                g.now_us = w;
            }
            g.tasks[i].st = St::Running;
            g.baton = Some(i);
            self.cv.notify_all();
        }
    }
}

/// タスク群を仮想時間の上で走らせ、終わるまで戻らない。
///
/// `specs` は (起動時刻 [µs], 本体)。本体の中でレガシー C++ を呼んでよい —
/// その中の待ちはこのスケジューラが吸収する。
pub fn run_tasks(sched: &Arc<Sched>, specs: Vec<Box<dyn FnOnce() + Send>>) {
    std::thread::scope(|scope| {
        for (id, body) in specs.into_iter().enumerate() {
            let s = Arc::clone(sched);
            scope.spawn(move || {
                CURRENT_TASK.with(|c| c.set(Some(id)));
                s.wait_for_start(id);
                // タスクが落ちてもバトンは必ず返す。返さないとドライバが止まる。
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
                s.finish(id);
                if let Err(e) = r {
                    std::panic::resume_unwind(e);
                }
            });
        }
        sched.drive();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// 起床時刻順に、同着ならタスク番号順に配られる。
    #[test]
    fn interleaving_is_deterministic() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let sched = Sched::new(&[0, 0], 1_000_000);

        let mut specs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
        for id in 0..2usize {
            let o = Arc::clone(&order);
            let s = Arc::clone(&sched);
            // タスク 0 は 300 µs 刻み、タスク 1 は 200 µs 刻み。
            let step = if id == 0 { 300 } else { 200 };
            specs.push(Box::new(move || {
                for _ in 0..3 {
                    o.lock().unwrap().push((s.now_us(), id));
                    let t = s.now_us() + step;
                    s.park(id, t);
                }
            }));
        }
        run_tasks(&sched, specs);

        let got = order.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![(0, 0), (0, 1), (200, 1), (300, 0), (400, 1), (600, 0)]
        );
    }

    #[test]
    fn no_real_time_is_consumed() {
        let sched = Sched::new(&[0], 10_000_000);
        let s = Arc::clone(&sched);
        let wall = std::time::Instant::now();
        run_tasks(
            &sched,
            vec![Box::new(move || {
                for _ in 0..1000 {
                    let t = s.now_us() + 5_000; // 5 ms の待ちを 1000 回 = 仮想 5 秒
                    s.park(0, t);
                }
            })],
        );
        assert_eq!(sched.now_us(), 5_000_000);
        assert!(
            wall.elapsed().as_millis() < 2_000,
            "仮想 5 秒が実時間を食っている"
        );
    }

    #[test]
    fn budget_overrun_is_reported_not_hung() {
        let sched = Sched::new(&[0], 1_000);
        let s = Arc::clone(&sched);
        let n = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&n);
        run_tasks(
            &sched,
            vec![Box::new(move || {
                for _ in 0..5 {
                    c.fetch_add(1, Ordering::SeqCst);
                    let t = s.now_us() + 400;
                    s.park(0, t);
                }
            })],
        );
        assert!(sched.overran());
        assert_eq!(n.load(Ordering::SeqCst), 5); // 走り切ってはいる
        assert_eq!(sched.now_us(), 1_000); // 時計は予算で止まる
    }
}
