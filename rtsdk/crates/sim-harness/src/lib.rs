//! 3D 空間セマンティクスを `#[test]` にするシミュレータ。
//!
//! ## この層がやっていること
//!
//! 1. 図面の約束を [`semantics!`] で宣言する (剛体・ゾーン・規則)。
//! 2. 走行のたびに、軸が動いた**その瞬間に**規則を評価する ([`world::World`])。
//! 3. 規則 1 本ごとに `#[test]` を生やす。落ちたテストは
//!    「いつ・何と何が・何 µm まで詰まったか」を出す。
//!
//! ## TDD の回し方
//!
//! ```text
//! cargo test -p sim-harness -- --ignored     # 赤: 包む前に何が破れるかを見る
//! cargo test -p sim-harness                  # 緑: 型で包んだ後
//! ```
//!
//! 赤のテストを `--ignored` に伏せてあるのは、CI を常時赤にしないため。
//! 代わりに `premise_is_still_red` が「出発点がまだ赤であること」を毎回確かめている —
//! レガシー側が誰かに直されたら、TDD の前提が変わったこととして気づける。

pub mod runners;
pub mod scene;
pub mod scenes;
pub mod world;

pub use runners::{run_legacy_raw, run_legacy_staggered, run_typed_sdk, PickCycle};
pub use scene::{BodyDef, RuleDef, SceneDef, ZoneDef};
pub use world::{RunReport, SharedWorld, Violation, World};

/// 走行結果をまとめて出す (UI が読む JSON を作るときに使う)。
pub fn run_all() -> Vec<RunReport> {
    vec![
        run_legacy_raw(&scenes::cell01::SCENE),
        run_legacy_staggered(&scenes::cell01::SCENE),
        run_typed_sdk(&scenes::cell01::SCENE),
        run_typed_sdk(&scenes::cell01_tight::SCENE),
    ]
}
