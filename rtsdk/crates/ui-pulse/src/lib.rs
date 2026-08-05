//! デジタルツイン・パルスグラフ。
//!
//! 横軸が µs、縦軸がレーン (制御タスク / レガシー手順 / ゾーン占有 / 違反)。
//! DAW のシーケンサと同じ見え方で、Wasm インスタンスの
//! **起動 (立ち上がり) → 実行中 (H レベル) → 消滅 (立ち下がり)** をブロックにする。
//!
//! 単なる波形ビューアと違うのは 3 点。
//!
//! * **重ね合わせ。** 型安全 SDK が意図した並び (設計) の上に、実際に走った並び (実測) を重ねる。
//!   ずれた事象と、片側にしか無い事象が [`timeline::overlay`] で出る。
//! * **因果。** 落ちた瞬間から、原因になった進入へ矢印を引く ([`timeline::causal_arrows`])。
//! * **タイムトラベル。** 任意の時刻へ巻き戻して状態変数を見る ([`timeline::state_at`])。
//!   実機のダンプではなく事象の再生なので、何度でもどこへでも戻れる。
//!
//! 描画に依らない側 ([`timeline`]) と描画側 ([`app`]) を分けてあるので、
//! 画面を出さずに全部テストできる。

pub mod timeline;

#[cfg(feature = "gui")]
pub mod app;

#[cfg(feature = "gui")]
pub use app::PulseApp;

// ブラウザ向けの入口。wasm 以外の的では web-sys が無いので出さない。
#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web;

/// `cargo run -p sim-harness --bin dump-traces` が出したトレース。
///
/// UI をブラウザで開いた直後から何か映っているように、既定の題材を焼き込んである。
pub mod samples {
    /// 型で包む前 (レガシー手順、時間差なし) — 干渉する。
    pub const LEGACY_RAW: &str = include_str!("../../../artifacts/cell01__legacy-raw.json");
    /// 現場の回避策 (40 ms ずらす) — たまたま当たらない。
    pub const LEGACY_STAGGERED: &str =
        include_str!("../../../artifacts/cell01__legacy-staggered.json");
    /// 型安全 SDK に手順を移した後 — 規則をすべて満たす。
    pub const TYPED_SDK: &str = include_str!("../../../artifacts/cell01__typed-sdk.json");
    /// 同じ手順を 80 ms タクトに載せようとしたとき — デッドラインで落ちる。
    pub const TYPED_SDK_TIGHT: &str =
        include_str!("../../../artifacts/cell01_tight__typed-sdk.json");

    /// 焼き込んだトレースを事象の列に戻す。
    pub fn parse(s: &str) -> Vec<pulse_trace::Ev> {
        pulse_trace::parse_events(s).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulse_trace::Kind;

    #[test]
    fn the_baked_in_traces_load() {
        let red = samples::parse(samples::LEGACY_RAW);
        let green = samples::parse(samples::TYPED_SDK);
        assert!(red.len() > 100);
        assert!(green.len() > 100);
        assert_eq!(red.iter().filter(|e| e.kind.is_fault()).count(), 2);
        assert_eq!(green.iter().filter(|e| e.kind.is_fault()).count(), 0);
    }

    /// 実データで、干渉の因果が 2 本の進入に戻ること。
    #[test]
    fn the_real_red_trace_yields_causal_arrows() {
        let red = samples::parse(samples::LEGACY_RAW);
        let arrows = timeline::causal_arrows(&red);
        assert!(!arrows.is_empty());
        assert!(arrows.iter().any(|a| a.from_lane == 10));
        assert!(arrows.iter().any(|a| a.from_lane == 11));
    }

    /// 実データで、干渉の直前へ巻き戻すと「2 本とも中に居る」が見える。
    #[test]
    fn time_travel_shows_both_arms_inside_at_the_fault() {
        let red = samples::parse(samples::LEGACY_RAW);
        let fault = red.iter().find(|e| e.kind == Kind::Violation).unwrap();

        let at = timeline::state_at(&red, fault.t_us);
        assert!(!at.axis_pos_um.is_empty());

        let exclusive = red
            .iter()
            .filter(|e| e.kind == Kind::Violation)
            .find(|e| e.a == 1)
            .unwrap();
        let at_exc = timeline::state_at(&red, exclusive.t_us);
        assert_eq!(
            at_exc.occupying.len(),
            2,
            "その瞬間、2 本ともゾーンの中に居る"
        );
    }

    /// 設計 (緑) と実測 (赤) を重ねると、赤にしか無い違反が残る。
    #[test]
    fn overlaying_the_green_design_on_the_red_run_surfaces_the_faults() {
        let designed = samples::parse(samples::TYPED_SDK);
        let measured = samples::parse(samples::LEGACY_RAW);
        let d = timeline::overlay(&designed, &measured);
        let only_measured: Vec<_> = d
            .iter()
            .filter(|x| x.designed_t_us.is_none() && x.kind.is_fault())
            .collect();
        assert_eq!(only_measured.len(), 2, "設計側に無い違反が 2 件");
    }
}
