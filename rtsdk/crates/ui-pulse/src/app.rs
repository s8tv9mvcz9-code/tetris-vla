//! 描画側。egui の `Painter` に直接矩形と線を置くだけで、外部の描画資産は使わない。
//!
//! 画面は 3 段。
//!
//! 1. **パルスグラフ** — 横軸 µs、縦軸レーン。設計は細い枠、実測は塗り。
//! 2. **因果** — 違反へ向かう赤い矢印。時刻カーソルより手前のものだけ光らせる。
//! 3. **状態変数** — カーソル位置の軸位置・保持トークン・占有。巻き戻して読む。

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use pulse_trace::Ev;

use crate::timeline::{causal_arrows, overlay, state_at, tracks, Arrow, PulseKind, Role, Track};

const ROW_H: f32 = 26.0;
const LABEL_W: f32 = 190.0;

const C_TASK: Color32 = Color32::from_rgb(80, 160, 220);
const C_PERMIT: Color32 = Color32::from_rgb(120, 200, 140);
const C_ZONE: Color32 = Color32::from_rgb(200, 170, 90);
const C_FAULT: Color32 = Color32::from_rgb(220, 70, 70);
const C_DESIGN: Color32 = Color32::from_rgb(130, 130, 150);

/// 1 画面ぶんの状態。
pub struct PulseApp {
    /// 設計 = 型安全 SDK が意図した並び。
    pub designed: Vec<Ev>,
    /// 実測 = 実際に走った並び (TDD の走行ログ)。
    pub measured: Vec<Ev>,
    /// 時間カーソル [µs]。ここまで巻き戻して状態を見る。
    pub cursor_us: u64,
    pub show_design: bool,
    pub show_arrows: bool,
    /// 表示する時間窓 [µs]。
    pub view_us: (u64, u64),
    measured_tracks: Vec<Track>,
    designed_tracks: Vec<Track>,
    arrows: Vec<Arrow>,
}

impl PulseApp {
    pub fn new(designed: Vec<Ev>, measured: Vec<Ev>) -> Self {
        let t_max = measured
            .iter()
            .chain(designed.iter())
            .map(|e| e.t_us)
            .max()
            .unwrap_or(1);
        let arrows = causal_arrows(&measured);
        // 最初の違反があれば、そこにカーソルを置いて開く。
        let cursor_us = measured
            .iter()
            .find(|e| e.kind.is_fault())
            .map(|e| e.t_us)
            .unwrap_or(t_max / 2);
        PulseApp {
            measured_tracks: tracks(&measured),
            designed_tracks: tracks(&designed),
            arrows,
            designed,
            measured,
            cursor_us,
            show_design: true,
            show_arrows: true,
            view_us: (0, t_max.max(1)),
        }
    }

    /// トレースを差し替える (走行を切り替えたとき)。
    pub fn set_traces(&mut self, designed: Vec<Ev>, measured: Vec<Ev>) {
        *self = PulseApp::new(designed, measured);
    }

    fn lanes(&self) -> Vec<u16> {
        let mut v: Vec<u16> = self
            .measured_tracks
            .iter()
            .chain(self.designed_tracks.iter())
            .map(|t| t.lane)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.controls(ui);
        ui.separator();
        self.graph(ui);
        ui.separator();
        self.inspector(ui);
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_design, "設計を重ねる");
            ui.checkbox(&mut self.show_arrows, "因果を出す");
            ui.separator();
            let (lo, hi) = self.view_us;
            let mut c = self.cursor_us as f64;
            ui.label("時刻 [µs]");
            if ui
                .add(egui::Slider::new(&mut c, lo as f64..=hi as f64).step_by(100.0))
                .changed()
            {
                self.cursor_us = c as u64;
            }
            if ui.button("最初の違反へ").clicked() {
                if let Some(f) = self.measured.iter().find(|e| e.kind.is_fault()) {
                    self.cursor_us = f.t_us;
                }
            }
            if ui.button("200 µs 戻す").clicked() {
                self.cursor_us = self.cursor_us.saturating_sub(200);
            }
        });
    }

    fn graph(&mut self, ui: &mut egui::Ui) {
        let lanes = self.lanes();
        let h = ROW_H * lanes.len() as f32 + 12.0;
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), h), Sense::click_and_drag());
        let p = ui.painter_at(rect);

        let (t0, t1) = self.view_us;
        let span = (t1.saturating_sub(t0)).max(1) as f32;
        let plot_x0 = rect.left() + LABEL_W;
        let plot_w = (rect.width() - LABEL_W).max(1.0);
        let x_of = |t: u64| plot_x0 + (t.saturating_sub(t0) as f32 / span) * plot_w;
        let y_of = |lane: u16| {
            rect.top() + 6.0 + lanes.iter().position(|l| *l == lane).unwrap_or(0) as f32 * ROW_H
        };

        // クリック / ドラッグで時刻カーソルを動かす = 巻き戻し。
        if let Some(pos) = resp.interact_pointer_pos() {
            let f = ((pos.x - plot_x0) / plot_w).clamp(0.0, 1.0);
            self.cursor_us = t0 + (f * span) as u64;
        }

        for lane in &lanes {
            let y = y_of(*lane);
            let name = self
                .measured_tracks
                .iter()
                .chain(self.designed_tracks.iter())
                .find(|t| t.lane == *lane)
                .map(|t| t.name.clone())
                .unwrap_or_default();
            p.text(
                Pos2::new(rect.left() + 4.0, y + ROW_H * 0.5),
                Align2::LEFT_CENTER,
                name,
                FontId::monospace(12.0),
                ui.visuals().text_color(),
            );
            p.hline(
                plot_x0..=rect.right(),
                y + ROW_H - 2.0,
                Stroke::new(0.5, ui.visuals().weak_text_color()),
            );
        }

        // 設計: 枠だけ。実測: 塗り。重なって見えることが目的。
        if self.show_design {
            for tr in &self.designed_tracks {
                for pu in &tr.pulses {
                    let r = pulse_rect(y_of(tr.lane), x_of(pu.t0_us), x_of(pu.t1_us.unwrap_or(t1)));
                    p.rect_stroke(r, 2.0, Stroke::new(1.0, C_DESIGN));
                }
            }
        }
        for tr in &self.measured_tracks {
            for pu in &tr.pulses {
                let color = match (tr.role, pu.kind) {
                    (_, PulseKind::PermitHeld) => C_PERMIT,
                    (Role::Zone, _) => C_ZONE,
                    _ => C_TASK,
                };
                let x0 = x_of(pu.t0_us);
                let x1 = x_of(pu.t1_us.unwrap_or(t1));
                let r = pulse_rect(y_of(tr.lane), x0, x1);
                p.rect_filled(r, 2.0, color.gamma_multiply(0.55));
                // 立ち上がり / 立ち下がりの縁を強調する (DAW のブロックの端)。
                p.vline(x0, r.y_range(), Stroke::new(2.0, color));
                if pu.t1_us.is_some() {
                    p.vline(x1, r.y_range(), Stroke::new(2.0, color));
                }
            }
        }

        // 違反の点。
        for e in self.measured.iter().filter(|e| e.kind.is_fault()) {
            let x = x_of(e.t_us);
            let y = y_of(e.lane) + ROW_H * 0.5;
            p.circle_filled(Pos2::new(x, y), 5.0, C_FAULT);
        }

        // 因果の矢印。カーソルより手前のものだけ赤く光らせる。
        if self.show_arrows {
            for a in &self.arrows {
                let from = Pos2::new(x_of(a.from_t_us), y_of(a.from_lane) + ROW_H * 0.5);
                let to = Pos2::new(x_of(a.to_t_us), y_of(a.to_lane) + ROW_H * 0.5);
                let hot = a.to_t_us <= self.cursor_us;
                let col = if hot {
                    C_FAULT
                } else {
                    C_FAULT.gamma_multiply(0.35)
                };
                p.line_segment([from, to], Stroke::new(if hot { 2.0 } else { 1.0 }, col));
                arrow_head(&p, from, to, col);
            }
        }

        // 時刻カーソル。
        let cx = x_of(self.cursor_us);
        p.vline(
            cx,
            rect.y_range(),
            Stroke::new(1.5, Color32::from_rgb(240, 240, 120)),
        );
        p.text(
            Pos2::new(cx + 4.0, rect.top() + 2.0),
            Align2::LEFT_TOP,
            format!("{} µs", self.cursor_us),
            FontId::monospace(11.0),
            Color32::from_rgb(240, 240, 120),
        );
    }

    /// カーソル時刻の状態変数と、設計とのずれ。
    fn inspector(&mut self, ui: &mut egui::Ui) {
        let s = state_at(&self.measured, self.cursor_us);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.strong(format!("t = {} µs の状態", s.t_us));
                for (lane, pos) in &s.axis_pos_um {
                    ui.monospace(format!("axis {} : {:>8} µm", lane, pos));
                }
                ui.monospace(format!("生存タスク : {:?}", s.alive));
                ui.monospace(format!("保持トークン: {:?}", s.permits));
                ui.monospace(format!("ゾーン占有 : {:?}", s.occupying));
                if s.occupying.len() > 1 {
                    ui.colored_label(C_FAULT, "この時刻、2 者が同じゾーンに居る");
                }
            });
            ui.separator();
            ui.vertical(|ui| {
                ui.strong("設計とのずれ");
                let d = overlay(&self.designed, &self.measured);
                let mut shown = 0;
                for x in d.iter() {
                    if x.designed_t_us.is_none() {
                        ui.colored_label(
                            C_FAULT,
                            format!(
                                "実測にのみ {:?} @ {} µs (lane {})",
                                x.kind,
                                x.measured_t_us.unwrap_or(0),
                                x.lane
                            ),
                        );
                    } else if let Some(sk) = x.skew_us() {
                        if sk.abs() >= 1_000 {
                            ui.monospace(format!(
                                "lane {} {:?} : 設計より {:+} µs",
                                x.lane, x.kind, sk
                            ));
                        } else {
                            continue;
                        }
                    } else {
                        ui.monospace(format!("設計にのみ {:?} (lane {})", x.kind, x.lane));
                    }
                    shown += 1;
                    if shown > 12 {
                        ui.label("…");
                        break;
                    }
                }
                if shown == 0 {
                    ui.label("設計どおり");
                }
            });
        });
    }
}

fn pulse_rect(y: f32, x0: f32, x1: f32) -> Rect {
    Rect::from_min_max(
        Pos2::new(x0, y + 4.0),
        Pos2::new((x1).max(x0 + 1.5), y + ROW_H - 6.0),
    )
}

fn arrow_head(p: &egui::Painter, from: Pos2, to: Pos2, col: Color32) {
    let d = to - from;
    let len = d.length().max(1.0);
    let u = d / len;
    let n = Vec2::new(-u.y, u.x);
    let tip = to;
    let a = tip - u * 8.0 + n * 4.0;
    let b = tip - u * 8.0 - n * 4.0;
    p.line_segment([tip, a], Stroke::new(1.5, col));
    p.line_segment([tip, b], Stroke::new(1.5, col));
}

/// 走行の切り替え (焼き込んだ 4 本)。
pub const RUNS: [(&str, &str); 4] = [
    ("赤: レガシー手順そのまま", crate::samples::LEGACY_RAW),
    ("現場: 40 ms ずらして回避", crate::samples::LEGACY_STAGGERED),
    ("緑: 型安全 SDK", crate::samples::TYPED_SDK),
    ("緑だがタクト 80 ms", crate::samples::TYPED_SDK_TIGHT),
];

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web {
    use super::*;

    /// eframe から呼ばれる画面。左に走行の一覧、右にパルスグラフ。
    pub struct App {
        pulse: PulseApp,
        selected: usize,
    }

    impl Default for App {
        fn default() -> Self {
            let designed = crate::samples::parse(crate::samples::TYPED_SDK);
            let measured = crate::samples::parse(RUNS[0].1);
            App {
                pulse: PulseApp::new(designed, measured),
                selected: 0,
            }
        }
    }

    impl eframe::App for App {
        fn update(&mut self, ctx: &egui::Context, _f: &mut eframe::Frame) {
            egui::TopBottomPanel::top("runs").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("rtsdk pulse analyzer");
                    for (i, (name, json)) in RUNS.iter().enumerate() {
                        if ui.selectable_label(self.selected == i, *name).clicked() {
                            self.selected = i;
                            let designed = crate::samples::parse(crate::samples::TYPED_SDK);
                            self.pulse.set_traces(designed, crate::samples::parse(json));
                        }
                    }
                });
            });
            egui::CentralPanel::default().show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.pulse.ui(ui));
            });
        }
    }
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use web::App;
