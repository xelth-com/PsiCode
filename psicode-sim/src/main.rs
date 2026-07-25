//! psicode-sim: симулятор канала «монитор -> камера» и Monte Carlo-развёртки SER.
//!
//! Физика L0/§9.1 SPEC: гамма, гомография, дефокус-блюр, кросстолк, усиление/
//! смещение, шум. Плюс развёртки, заполняющие BENCHMARKS.md §1–2. Зависимостей,
//! кроме psicode-core, нет — весь ГПСЧ, геометрия и PPM свои.
//!
//! Подкоманды:
//!   sweep        — три развёртки (blur σ; px/клетку; шум) эталонным профилем
//!   dump <dir>   — P6 PPM: чистый кадр + канал при σ=1 и σ=4 (глазами посмотреть)

mod channel;
mod image;
mod pipeline;
mod report;
mod rng;

use channel::ChannelParams;
use pipeline::{apply_channel, run_trial};
use psicode_core::{CalibProfile, ChromaMode};
use rng::{seed_for, Rng};
use std::path::Path;
use std::time::Instant;

/// Эталонный профиль §7.4 SPEC — конфигурация симулятора по умолчанию.
fn reference_profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        luma_bits: 3,
        chroma_mode: ChromaMode::Chroma2,
        gamma_g_q: 28, // γ_G = 2.200
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15, // 100%
        black_level_q: 2,
        noise_sigma_q: 12, // σ ≈ 2.0 градации
        mtf_limit_px: 6,
        torn_frames_q: 5,
        crosstalk_rg_q: 3, // 6%
        crosstalk_gb_q: 4, // 8%
        quiet_zone: 1,     // 4 клетки
        fec_overhead: 2,
    }
}

const TRIALS: usize = 15;

/// Средний SER по TRIALS попыткам в точке развёртки `point_idx`.
fn mean_ser(p: &CalibProfile, ch: &ChannelParams, point_idx: usize) -> f64 {
    let mut acc = 0.0;
    for t in 0..TRIALS {
        acc += run_trial(p, ch, point_idx, t).ser();
    }
    acc / TRIALS as f64
}

fn cmd_sweep() {
    let p = reference_profile();
    let base = ChannelParams::from_profile(&p);
    println!("# psicode-sim sweeps");
    println!(
        "профиль §7.4: cell {} px, luma {} бит, chroma {} бит, γ_G {:.3}; \
         канал: кросстолк {}%/{}%, шум σ {:.2} град/255",
        p.cell_size_px,
        p.luma_bits,
        p.chroma_bits(),
        p.gamma_g(),
        p.crosstalk_rg_pct(),
        p.crosstalk_gb_pct(),
        p.noise_sigma(),
    );
    println!("{TRIALS} попыток на точку\n");

    let t0 = Instant::now();
    let mut point = 0usize;

    // (a) SER vs blur σ, px_per_cell = 8  (BENCHMARKS §1)
    let sigmas = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0];
    println!("## 1. SER vs blur σ (px/cell = 8)");
    println!("| source | blur σ (px) → | 0.5 | 1 | 2 | 4 | 6 | 8 |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &s in &sigmas {
        let mut ch = base.clone();
        ch.px_per_cell = 8.0;
        ch.blur_sigma_px = s;
        cells.push(report::sig4(mean_ser(&p, &ch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    // (b) SER vs px/cell, σ = 1  (BENCHMARKS §2)
    let ppcs = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5];
    println!("\n## 2. SER vs px/cell (blur σ = 1)");
    println!("| source | px/cell → | 8 | 6 | 4 | 3 | 2 | 1.5 |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &ppc in &ppcs {
        let mut ch = base.clone();
        ch.px_per_cell = ppc;
        ch.blur_sigma_px = 1.0;
        cells.push(report::sig4(mean_ser(&p, &ch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    // (c) бонус: SER vs множитель шума, σ = 1, px/cell = 8
    let base_noise = base.noise_sigma;
    let mults = [1.0, 2.0, 4.0, 8.0];
    println!("\n## 3. (бонус) SER vs множитель шума (σ_blur = 1, px/cell = 8)");
    println!("| source | noise ×base → | 1 | 2 | 4 | 8 |");
    println!("|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &m in &mults {
        let mut ch = base.clone();
        ch.px_per_cell = 8.0;
        ch.blur_sigma_px = 1.0;
        ch.noise_sigma = base_noise * m;
        cells.push(report::sig4(mean_ser(&p, &ch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

fn cmd_dump(dir: &str) {
    let p = reference_profile();
    let dir = Path::new(dir);
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("не создать {}: {e}", dir.display());
        std::process::exit(1);
    }

    // фиксированный сид для воспроизводимого кадра
    let mut rng = Rng::new(seed_for(0, 0));
    let bpc = psicode_core::symbol::bits_per_cell(&p);
    let n_levels = 1u32 << bpc;
    let n_cells = psicode_core::symbol::PAYLOAD_COLS * psicode_core::symbol::PAYLOAD_ROWS;
    let cells: Vec<u8> = (0..n_cells)
        .map(|_| rng.next_u32_below(n_levels) as u8)
        .collect();
    let frame = psicode_core::symbol::render_symbol(&p, &cells);

    // 1. чистый отрендеренный кадр (drive-байты как есть)
    let clean_path = dir.join("clean.ppm");
    report::write_ppm(&clean_path, frame.size_px, frame.size_px, &frame.rgb).unwrap();
    println!("записан {}", clean_path.display());

    // 2–3. после канала при σ = 1 и σ = 4 (px/cell = 8), обратная гамма для глаза
    for &s in &[1.0f64, 4.0] {
        let mut ch = ChannelParams::from_profile(&p);
        ch.px_per_cell = 8.0;
        ch.blur_sigma_px = s;
        let (img, _geom) = apply_channel(&frame, &ch, &mut rng);
        let drive = report::image_to_drive(&img, ch.gammas);
        let path = dir.join(format!("channel_s{}.ppm", s as u32));
        report::write_ppm(&path, img.w, img.h, &drive).unwrap();
        println!("записан {}", path.display());
    }

    // 4. пресет мягкой перспективы (σ = 1) — увидеть геометрическое искажение
    let mut ch = ChannelParams::from_profile(&p);
    ch.px_per_cell = 8.0;
    ch.blur_sigma_px = 1.0;
    ch.homography = channel::mild_perspective(channel::symbol_size_px(&p));
    let (img, _geom) = apply_channel(&frame, &ch, &mut rng);
    let drive = report::image_to_drive(&img, ch.gammas);
    let path = dir.join("channel_perspective.ppm");
    report::write_ppm(&path, img.w, img.h, &drive).unwrap();
    println!("записан {}", path.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("sweep") => cmd_sweep(),
        Some("dump") => {
            let dir = args.get(2).map(String::as_str).unwrap_or("psicode-sim-dump");
            cmd_dump(dir);
        }
        _ => {
            eprintln!("usage: psicode-sim <sweep | dump [dir]>");
            std::process::exit(2);
        }
    }
}
