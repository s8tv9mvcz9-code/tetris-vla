//! 各走行のトレースを JSON に出す。UI (ui-pulse) が読む入力になる。
//!
//! ```text
//! cargo run -p sim-harness --bin dump-traces -- artifacts/
//! ```

fn main() -> std::io::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "artifacts".to_string());
    std::fs::create_dir_all(&dir)?;

    for r in sim_harness::run_all() {
        let path = format!("{}/{}__{}.json", dir, r.scene, r.label);
        std::fs::write(&path, r.to_json())?;
        println!(
            "{:<40} 事象 {:>5} 件  違反 {} 件  終了 {:>8} µs  CPU {:>5} µs",
            path,
            r.events.len(),
            r.violations.len(),
            r.t_end_us,
            r.cpu_us
        );
        for v in &r.violations {
            println!("    ! {} ({}) {}", v.rule, v.name, v.msg);
        }
    }
    Ok(())
}
