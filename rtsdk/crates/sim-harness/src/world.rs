//! シミュレータ本体。軸が動くたびに空間セマンティクスを評価し、
//! 破れた瞬間を仮想時刻つきで記録する。
//!
//! 評価は**書き込みの都度**行う。周期サンプリングにすると、
//! 「サンプルとサンプルの間だけ 50 mm を切って戻った」を取り逃がす。
//! レガシー側の 1 送りごとにフックが来るので、そこが自然な評価点になる。

use std::sync::{Arc, Mutex, MutexGuard};

use legacy_bridge::PlantIo;
use pulse_trace::{Ev, Kind, Recorder, Sink};
use zoneguard::{dist_um, P3};

use crate::scene::{RuleDef, SceneDef};

pub const MAX_AXES: usize = 4;
pub const MAX_RULES: usize = 8;
pub const TRACE_CAP: usize = 4096;

/// レーン割り当て。UI の縦軸はこの規約で並ぶ。
pub const LANE_TASK_BASE: u16 = 0; // 0,1 … 制御タスク / レガシー手順
pub const LANE_GEOM_BASE: u16 = 10; // 10,11 … 幾何から見たゾーン占有
pub const LANE_FAULT: u16 = 20; // 20 … 違反

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub rule: u16,
    pub name: &'static str,
    pub t_us: u64,
    /// 規則ごとの実測値。Separation なら表面間距離 [µm]、Deadline なら超過 [µs]。
    pub measured: i64,
    pub msg: String,
}

pub struct World {
    scene: &'static SceneDef,
    pos: [i32; MAX_AXES],
    occupied: Vec<u32>,
    active: [bool; MAX_RULES],
    rec: Recorder<TRACE_CAP>,
    viol: Vec<Violation>,
    t_last_us: u64,
}

impl World {
    pub fn new(scene: &'static SceneDef) -> World {
        assert!(scene.rules.len() <= MAX_RULES, "規則が多すぎる");
        World {
            scene,
            pos: [0; MAX_AXES],
            occupied: vec![0; scene.zones.len()],
            active: [false; MAX_RULES],
            rec: Recorder::new(),
            viol: Vec::new(),
            t_last_us: 0,
        }
    }

    pub fn scene(&self) -> &'static SceneDef {
        self.scene
    }

    fn tip(&self, body: usize) -> P3 {
        let axis = self.scene.bodies[body].axis as usize;
        self.scene.tip(body, self.pos[axis])
    }

    fn fault(&mut self, rule: u16, name: &'static str, t_us: u64, measured: i64, msg: String) {
        self.rec.emit(Ev::new(
            t_us,
            LANE_FAULT,
            Kind::Violation,
            rule as i32,
            measured as i32,
        ));
        self.viol.push(Violation {
            rule,
            name,
            t_us,
            measured,
            msg,
        });
    }

    /// 空間セマンティクスの評価。ここが「図面の約束」と「実際の座標」の突き合わせ。
    fn evaluate(&mut self, t_us: u64) {
        self.t_last_us = self.t_last_us.max(t_us);

        for id in 0..self.scene.rules.len() {
            match self.scene.rules[id] {
                RuleDef::Separation { name, a, b, min_um } => {
                    let gap = dist_um(self.tip(a), self.tip(b))
                        - self.scene.bodies[a].radius_um as i64
                        - self.scene.bodies[b].radius_um as i64;
                    let bad = gap < min_um as i64;
                    if bad && !self.active[id] {
                        let msg = format!(
                            "{} と {} の表面間が {} µm まで詰まった (下限 {} µm) @ t={} µs",
                            self.scene.bodies[a].name, self.scene.bodies[b].name, gap, min_um, t_us
                        );
                        self.fault(id as u16, name, t_us, gap, msg);
                    }
                    self.active[id] = bad;
                }
                RuleDef::Exclusive { name, zone } => {
                    let aabb = self.scene.zones[zone].aabb;
                    let mut mask = 0u32;
                    for b in 0..self.scene.bodies.len() {
                        if aabb.contains(self.tip(b)) {
                            mask |= 1 << b;
                        }
                    }
                    let prev = self.occupied[zone];
                    if mask != prev {
                        for b in 0..self.scene.bodies.len() {
                            let was = prev & (1 << b) != 0;
                            let now = mask & (1 << b) != 0;
                            if was != now {
                                self.rec.emit(Ev::new(
                                    t_us,
                                    LANE_GEOM_BASE + b as u16,
                                    if now { Kind::ZoneEnter } else { Kind::ZoneExit },
                                    self.scene.zones[zone].id as i32,
                                    0,
                                ));
                            }
                        }
                        self.occupied[zone] = mask;
                    }
                    let n = mask.count_ones();
                    let bad = n > 1;
                    if bad && !self.active[id] {
                        let names: Vec<&str> = (0..self.scene.bodies.len())
                            .filter(|b| mask & (1 << b) != 0)
                            .map(|b| self.scene.bodies[b].name)
                            .collect();
                        let msg = format!(
                            "排他ゾーン {} に {} が同時に入った @ t={} µs",
                            self.scene.zones[zone].name,
                            names.join(" と "),
                            t_us
                        );
                        self.fault(id as u16, name, t_us, n as i64, msg);
                    }
                    self.active[id] = bad;
                }
                RuleDef::Deadline { .. } => {}
            }
        }
    }

    /// サイクル終了。デッドラインはここで判定する。
    pub fn finish(&mut self, t_end_us: u64) {
        self.t_last_us = self.t_last_us.max(t_end_us);
        for id in 0..self.scene.rules.len() {
            if let RuleDef::Deadline { name, budget_us } = self.scene.rules[id] {
                if t_end_us > budget_us {
                    self.rec.emit(Ev::new(
                        t_end_us,
                        LANE_FAULT,
                        Kind::DeadlineMiss,
                        budget_us as i32,
                        t_end_us as i32,
                    ));
                    let msg = format!(
                        "1 サイクルが {} µs かかった (予算 {} µs、超過 {} µs)",
                        t_end_us,
                        budget_us,
                        t_end_us - budget_us
                    );
                    self.viol.push(Violation {
                        rule: id as u16,
                        name,
                        t_us: t_end_us,
                        measured: (t_end_us - budget_us) as i64,
                        msg,
                    });
                }
            }
        }
    }

    pub fn report(&self, label: &'static str, cpu_us: u64, overran: bool) -> RunReport {
        RunReport {
            scene: self.scene.name,
            label,
            events: {
                let mut e = self.rec.events().to_vec();
                e.sort_by_key(|x| x.t_us);
                e
            },
            violations: self.viol.clone(),
            t_end_us: self.t_last_us,
            cpu_us,
            dropped: self.rec.dropped(),
            overran,
        }
    }
}

impl Sink for World {
    fn emit(&mut self, ev: Ev) {
        self.rec.emit(ev);
    }
}

impl PlantIo for World {
    fn write_axis(&mut self, t_us: u64, axis: i32, pos_um: i32) {
        let a = axis as usize;
        if a < MAX_AXES {
            self.pos[a] = pos_um;
        }
        self.rec.emit(Ev::new(
            t_us,
            LANE_TASK_BASE + axis as u16,
            Kind::AxisPos,
            pos_um,
            0,
        ));
        self.evaluate(t_us);
    }

    fn mark(&mut self, t_us: u64, axis: i32, code: i32) {
        let kind = if code != 0 {
            Kind::TaskRise
        } else {
            Kind::TaskFall
        };
        self.rec
            .emit(Ev::new(t_us, LANE_TASK_BASE + axis as u16, kind, code, 0));
    }
}

/// 設備と共有する世界。フック (別スレッドから来る) と、
/// ハーネス (結果を読む) の両方が触るのでロックの下に置く。
#[derive(Clone)]
pub struct SharedWorld(Arc<Mutex<World>>);

impl SharedWorld {
    pub fn new(scene: &'static SceneDef) -> SharedWorld {
        SharedWorld(Arc::new(Mutex::new(World::new(scene))))
    }

    pub fn lock(&self) -> MutexGuard<'_, World> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl PlantIo for SharedWorld {
    fn write_axis(&mut self, t_us: u64, axis: i32, pos_um: i32) {
        self.lock().write_axis(t_us, axis, pos_um);
    }
    fn mark(&mut self, t_us: u64, axis: i32, code: i32) {
        self.lock().mark(t_us, axis, code);
    }
}

/// SDK 側 (`vtime::Ctx`) のイベントを同じトレースへ流す出口。
impl Sink for SharedWorld {
    fn emit(&mut self, ev: Ev) {
        self.lock().emit(ev);
    }
}

/// 1 回の走行の結果。テストの合否も、UI の表示も、これだけを見る。
#[derive(Clone, Debug)]
pub struct RunReport {
    pub scene: &'static str,
    pub label: &'static str,
    pub events: Vec<Ev>,
    pub violations: Vec<Violation>,
    pub t_end_us: u64,
    /// 制御タスクが申告した計算時間の累計 [µs] (レガシー経路では 0)。
    pub cpu_us: u64,
    pub dropped: u32,
    /// 仮想時間の予算を使い切ったか。
    pub overran: bool,
}

impl RunReport {
    pub fn violation_of(&self, rule: u16) -> Option<&Violation> {
        self.violations.iter().find(|v| v.rule == rule)
    }

    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    /// 規則 1 本の検査。落ちたときに、いつ・何が・どれだけ破れたかを出す。
    pub fn assert_rule_holds(&self, scene: &'static SceneDef, rule: u16) {
        if let Some(v) = self.violation_of(rule) {
            panic!(
                "\n[{}] 規則 {} ({}) が破れている\n  {}\n  走行: {}\n  最初の違反時刻: {} µs\n",
                scene.name, rule, v.name, v.msg, self.label, v.t_us
            );
        }
    }

    /// 規則が「まだ破れていること」の確認 (赤を凍結するためのテスト)。
    pub fn assert_rule_violated(&self, scene: &'static SceneDef, rule: u16) {
        assert!(
            self.violation_of(rule).is_some(),
            "\n[{}] 規則 {} ({}) は破れなかった。\n  走行 {} は「型で包む前は破れる」ことを示すための前提だった。\n  レガシー側かシナリオが変わっている。前提を確認しなおすこと。\n",
            scene.name,
            rule,
            scene.rule(rule).name(),
            self.label
        );
    }

    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "{{\"scene\":\"{}\",\"label\":\"{}\",\"t_end_us\":{},\"cpu_us\":{},\"dropped\":{},\"overran\":{},\"violations\":[",
            self.scene, self.label, self.t_end_us, self.cpu_us, self.dropped, self.overran
        ));
        for (i, v) in self.violations.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"rule\":{},\"name\":\"{}\",\"t_us\":{},\"measured\":{},\"msg\":\"{}\"}}",
                v.rule,
                v.name,
                v.t_us,
                v.measured,
                v.msg.replace('"', "'")
            ));
        }
        s.push_str("],\"events\":[");
        for (i, e) in self.events.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"t_us\":{},\"lane\":{},\"kind\":\"{}\",\"a\":{},\"b\":{}}}",
                e.t_us,
                e.lane,
                e.kind.as_str(),
                e.a,
                e.b
            ));
        }
        s.push_str("]}");
        s
    }
}
