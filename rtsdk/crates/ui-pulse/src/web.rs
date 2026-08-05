//! ブラウザの Canvas に載せる入口。`--features web` のときだけ有効。
//!
//! ```bash
//! cargo build -p ui-pulse --features web --target wasm32-unknown-unknown --release
//! wasm-bindgen --target web --out-dir crates/ui-pulse/web \
//!     target/wasm32-unknown-unknown/release/ui_pulse.wasm
//! # crates/ui-pulse/web/index.html を任意の静的サーバで開く
//! ```

use eframe::wasm_bindgen::{self, prelude::*};

/// `index.html` から呼ばれる。指定した canvas に画面を張る。
#[wasm_bindgen]
pub async fn start(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    let document = eframe::web_sys::window()
        .ok_or_else(|| JsValue::from_str("window が無い"))?
        .document()
        .ok_or_else(|| JsValue::from_str("document が無い"))?;
    let canvas = document
        .get_element_by_id(&canvas_id)
        .ok_or_else(|| JsValue::from_str("canvas が見つからない"))?
        .dyn_into::<eframe::web_sys::HtmlCanvasElement>()?;

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(crate::app::App::default()))),
        )
        .await
}
