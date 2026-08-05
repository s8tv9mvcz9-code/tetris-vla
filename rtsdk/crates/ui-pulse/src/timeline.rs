//! パルスグラフの中身 — 描画に依らない側。
//!
//! ここには egui も Wasm も出てこない。トレース (事象の列) から
//!
//! * レーンごとのパルス (立ち上がり / H レベル / 立ち下がり)、
//! * 設計タイムラインと実測の**重ね合わせ**と、そのずれ、
//! * 落ちた瞬間の**因果**(どの進入とどの進入が干渉を作ったか)、
//! * 任意時刻の**状態変数**(時間を巻き戻して見る)
//!
//! を作る。描画側はこの結果を絵にするだけなので、UI 無しで全部テストできる。

use pulse_trace::{Ev, Kind};

/// 縦軸 1 本の役割。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// 制御タスク / レガシー手順のライフサイクル。
    Task,
    /// 幾何から見たゾーン占有。
    Zone,
    /// 違反。
    Fault,
}

/// パルス 1 本 (H レベルの区間)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pulse {
    pub t0_us: u64,
    /// 立ち下がりが無いまま終わったら `None` (走行が途中で終わっている印)。
    pub t1_us: Option<u64>,
    pub kind: PulseKind,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PulseKind {
    /// タスクが立ち上がってから消えるまで。
    TaskAlive,
    /// 排他トークンを握っている間 (SDK の意図)。
    PermitHeld,
    /// 実際にゾーンの中に居た間 (幾何の実測)。
    ZoneOccupied,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub lane: u16,
    pub name: String,
    pub role: Role,
    pub pulses: Vec<Pulse>,
}

/// 事象の列をレーンごとのパルスに畳む。
pub fn tracks(events: &[Ev]) -> Vec<Track> {
    let mut out: Vec<Track> = Vec::new();

    let idx = |out: &mut Vec<Track>, lane: u16| -> usize {
        if let Some(i) = out.iter().position(|t| t.lane == lane) {
            return i;
        }
        let (role, name) = match lane {
            0..=9 => (Role::Task, format!("task/axis {}", lane)),
            10..=19 => (Role::Zone, format!("zone occupancy body {}", lane - 10)),
            _ => (Role::Fault, "faults".to_string()),
        };
        out.push(Track {
            lane,
            name,
            role,
            pulses: Vec::new(),
        });
        out.len() - 1
    };

    for e in events {
        let i = idx(&mut out, e.lane);
        match e.kind {
            Kind::TaskRise => out[i].pulses.push(Pulse {
                t0_us: e.t_us,
                t1_us: None,
                kind: PulseKind::TaskAlive,
            }),
            Kind::PermitTake => out[i].pulses.push(Pulse {
                t0_us: e.t_us,
                t1_us: None,
                kind: PulseKind::PermitHeld,
            }),
            Kind::ZoneEnter => out[i].pulses.push(Pulse {
                t0_us: e.t_us,
                t1_us: None,
                kind: PulseKind::ZoneOccupied,
            }),
            Kind::TaskFall => close(&mut out[i], PulseKind::TaskAlive, e.t_us),
            Kind::PermitDrop => close(&mut out[i], PulseKind::PermitHeld, e.t_us),
            Kind::ZoneExit => close(&mut out[i], PulseKind::ZoneOccupied, e.t_us),
            _ => {}
        }
    }
    out.sort_by_key(|t| t.lane);
    out
}

fn close(track: &mut Track, kind: PulseKind, t_us: u64) {
    if let Some(p) = track
        .pulses
        .iter_mut()
        .rev()
        .find(|p| p.kind == kind && p.t1_us.is_none())
    {
        p.t1_us = Some(t_us);
    }
}

/// 因果の矢印。落ちた瞬間から、原因になった縁へ向けて引く。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arrow {
    pub from_lane: u16,
    pub from_t_us: u64,
    pub to_lane: u16,
    pub to_t_us: u64,
    pub label: String,
}

/// 違反ごとに、直前の「ゾーン進入」を原因として結ぶ。
///
/// 同時に中に居た 2 本を指すので、矢印は基本 2 本出る。
/// デッドライン超過なら、そのサイクルを立ち上げたタスクを指す。
pub fn causal_arrows(events: &[Ev]) -> Vec<Arrow> {
    let mut arrows = Vec::new();
    for (i, f) in events.iter().enumerate() {
        if !f.kind.is_fault() {
            continue;
        }
        let past = &events[..i];
        match f.kind {
            Kind::Violation => {
                // 直前に「入ったまま」になっているレーンをすべて拾う。
                let mut open: Vec<(u16, u64)> = Vec::new();
                for e in past {
                    match e.kind {
                        Kind::ZoneEnter => open.push((e.lane, e.t_us)),
                        Kind::ZoneExit => open.retain(|(l, _)| *l != e.lane),
                        _ => {}
                    }
                }
                for (lane, t) in open {
                    arrows.push(Arrow {
                        from_lane: lane,
                        from_t_us: t,
                        to_lane: f.lane,
                        to_t_us: f.t_us,
                        label: format!("進入 @ {} µs → 違反 {}", t, f.a),
                    });
                }
            }
            Kind::DeadlineMiss => {
                if let Some(rise) = past.iter().find(|e| e.kind == Kind::TaskRise) {
                    arrows.push(Arrow {
                        from_lane: rise.lane,
                        from_t_us: rise.t_us,
                        to_lane: f.lane,
                        to_t_us: f.t_us,
                        label: format!("サイクル開始 → 予算 {} µs 超過", f.a),
                    });
                }
            }
            _ => {}
        }
    }
    arrows
}

/// ある時刻の状態変数。時間を巻き戻して覗く先。
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Snapshot {
    pub t_us: u64,
    /// 軸位置 [µm] (レーン = 軸番号)。
    pub axis_pos_um: Vec<(u16, i32)>,
    /// 生きているタスクのレーン。
    pub alive: Vec<u16>,
    /// 排他トークンを握っているレーンとゾーン id。
    pub permits: Vec<(u16, i32)>,
    /// 実際にゾーンの中に居るレーン。
    pub occupying: Vec<u16>,
    /// この時刻までに発生した違反。
    pub faults: Vec<Ev>,
}

/// 事象を `t_us` まで再生して状態を作る。
///
/// 実機のダンプではなく再生なので、**どこまでも巻き戻せる**。
/// 「落ちた 200 µs 前に何を握っていたか」がその場で出る。
pub fn state_at(events: &[Ev], t_us: u64) -> Snapshot {
    let mut s = Snapshot {
        t_us,
        ..Default::default()
    };
    for e in events.iter().filter(|e| e.t_us <= t_us) {
        match e.kind {
            Kind::AxisPos => {
                if let Some(p) = s.axis_pos_um.iter_mut().find(|(l, _)| *l == e.lane) {
                    p.1 = e.a;
                } else {
                    s.axis_pos_um.push((e.lane, e.a));
                }
            }
            Kind::TaskRise => s.alive.push(e.lane),
            Kind::TaskFall => s.alive.retain(|l| *l != e.lane),
            Kind::PermitTake => s.permits.push((e.lane, e.a)),
            Kind::PermitDrop => s.permits.retain(|(l, _)| *l != e.lane),
            Kind::ZoneEnter => s.occupying.push(e.lane),
            Kind::ZoneExit => s.occupying.retain(|l| *l != e.lane),
            Kind::Violation | Kind::DeadlineMiss => s.faults.push(*e),
            Kind::Mark => {}
        }
    }
    s.axis_pos_um.sort_by_key(|(l, _)| *l);
    s
}

/// 設計 (型安全 SDK が意図した並び) と実測 (実際に走った並び) のずれ。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Divergence {
    pub lane: u16,
    pub kind: Kind,
    pub designed_t_us: Option<u64>,
    pub measured_t_us: Option<u64>,
}

impl Divergence {
    /// ずれ [µs]。片方にしか無い事象なら `None`。
    pub fn skew_us(&self) -> Option<i64> {
        match (self.designed_t_us, self.measured_t_us) {
            (Some(d), Some(m)) => Some(m as i64 - d as i64),
            _ => None,
        }
    }
}

/// 設計と実測を重ねる。
///
/// レーンと種別が同じ事象を出た順に突き合わせ、余ったほうを「片側だけ」として残す。
/// 落ちた走行では、設計側に無い `Violation` がここに出てくる。
pub fn overlay(designed: &[Ev], measured: &[Ev]) -> Vec<Divergence> {
    let interesting = |k: Kind| {
        matches!(
            k,
            Kind::TaskRise
                | Kind::TaskFall
                | Kind::ZoneEnter
                | Kind::ZoneExit
                | Kind::Violation
                | Kind::DeadlineMiss
        )
    };

    let mut out = Vec::new();
    let mut m_used = vec![false; measured.len()];

    for d in designed.iter().filter(|e| interesting(e.kind)) {
        let hit = measured
            .iter()
            .enumerate()
            .find(|(i, m)| !m_used[*i] && m.lane == d.lane && m.kind == d.kind);
        match hit {
            Some((i, m)) => {
                m_used[i] = true;
                out.push(Divergence {
                    lane: d.lane,
                    kind: d.kind,
                    designed_t_us: Some(d.t_us),
                    measured_t_us: Some(m.t_us),
                });
            }
            None => out.push(Divergence {
                lane: d.lane,
                kind: d.kind,
                designed_t_us: Some(d.t_us),
                measured_t_us: None,
            }),
        }
    }

    for (i, m) in measured.iter().enumerate() {
        if !m_used[i] && interesting(m.kind) {
            out.push(Divergence {
                lane: m.lane,
                kind: m.kind,
                designed_t_us: None,
                measured_t_us: Some(m.t_us),
            });
        }
    }
    out.sort_by_key(|d| d.measured_t_us.or(d.designed_t_us).unwrap_or(0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(t: u64, lane: u16, kind: Kind, a: i32) -> Ev {
        Ev::new(t, lane, kind, a, 0)
    }

    fn legacy_like() -> Vec<Ev> {
        vec![
            ev(0, 0, Kind::TaskRise, 1),
            ev(0, 1, Kind::TaskRise, 1),
            ev(10_000, 10, Kind::ZoneEnter, 1),
            ev(11_000, 11, Kind::ZoneEnter, 1),
            ev(11_000, 20, Kind::Violation, 1),
            ev(20_000, 10, Kind::ZoneExit, 1),
            ev(21_000, 11, Kind::ZoneExit, 1),
            ev(30_000, 0, Kind::TaskFall, 0),
            ev(31_000, 1, Kind::TaskFall, 0),
        ]
    }

    #[test]
    fn pulses_have_rising_and_falling_edges() {
        let t = tracks(&legacy_like());
        let task0 = t.iter().find(|t| t.lane == 0).unwrap();
        assert_eq!(task0.role, Role::Task);
        assert_eq!(task0.pulses.len(), 1);
        assert_eq!(task0.pulses[0].t0_us, 0);
        assert_eq!(task0.pulses[0].t1_us, Some(30_000));
    }

    #[test]
    fn an_unclosed_pulse_stays_open() {
        let evs = vec![ev(5, 0, Kind::TaskRise, 1)];
        let t = tracks(&evs);
        assert_eq!(t[0].pulses[0].t1_us, None);
    }

    #[test]
    fn the_fault_points_back_at_both_entries() {
        let a = causal_arrows(&legacy_like());
        assert_eq!(a.len(), 2, "同時に中に居た 2 本が原因");
        assert_eq!(a[0].from_t_us, 10_000);
        assert_eq!(a[1].from_t_us, 11_000);
        assert!(a.iter().all(|x| x.to_t_us == 11_000 && x.to_lane == 20));
    }

    #[test]
    fn time_travel_reconstructs_the_state_before_the_fault() {
        let evs = vec![
            ev(0, 0, Kind::TaskRise, 1),
            ev(1_000, 0, Kind::AxisPos, 2_000),
            ev(2_000, 0, Kind::PermitTake, 1),
            ev(3_000, 0, Kind::AxisPos, 6_000),
            ev(4_000, 0, Kind::PermitDrop, 1),
        ];
        let before = state_at(&evs, 2_500);
        assert_eq!(before.axis_pos_um, vec![(0, 2_000)]);
        assert_eq!(before.permits, vec![(0, 1)]);
        assert_eq!(before.alive, vec![0]);

        let after = state_at(&evs, 3_500);
        assert_eq!(after.axis_pos_um, vec![(0, 6_000)]);
        assert!(after.permits.is_empty() || after.permits == vec![(0, 1)]);

        let end = state_at(&evs, 10_000);
        assert!(end.permits.is_empty(), "返した後は握っていない");
    }

    #[test]
    fn overlay_reports_skew_and_events_that_only_one_side_has() {
        let designed = vec![
            ev(0, 0, Kind::TaskRise, 1),
            ev(50_000, 0, Kind::TaskFall, 0),
        ];
        let measured = vec![
            ev(0, 0, Kind::TaskRise, 1),
            ev(52_000, 0, Kind::TaskFall, 0),
            ev(15_000, 20, Kind::Violation, 1),
        ];
        let d = overlay(&designed, &measured);
        let fall = d.iter().find(|x| x.kind == Kind::TaskFall).unwrap();
        assert_eq!(fall.skew_us(), Some(2_000), "設計より 2 ms 遅れて消えた");

        let v = d.iter().find(|x| x.kind == Kind::Violation).unwrap();
        assert_eq!(v.designed_t_us, None, "設計側にこの事象は無い");
        assert_eq!(v.measured_t_us, Some(15_000));
    }
}
