//! Живой снимок: декод фотографии экрана (телефон -> JPEG -> PPM) готовым
//! трактом: детекция ЗЧ-рамки -> демодуляция -> SER против известной нагрузки
//! `psicode-tx single` (тот же сид splitmix64, что и в psicode-tx/frames.rs).
//!
//! Гамма-оговорка v0: JPEG телефона ~sRGB, дисплей ~2.2 — точную сквозную
//! кривую мы не знаем. Для детекции это не важно (монотонное преобразование,
//! пороги перцентильные); для демодуляции якоря референсной строки снимают
//! глобальные gain/offset на канал, а остаточная ошибка кривой честно уходит
//! в SER. Это первый живой замер, не финальная точность.

use std::path::Path;

use psicode_core::detect::{detect_symbol, frame_map};
use psicode_core::symbol::{self, demod_symbol, read_counters};
use psicode_core::tone;
use psicode_core::{CalibProfile, ChromaMode};

use crate::report;

/// Приближение обратной гаммы камеры/дисплея для v0.
const CAMERA_GAMMA_DEFAULT: f64 = 2.2;

// Гамму линеаризации по умолчанию можно переопределить переменной окружения
// PSICODE_CAMERA_GAMMA (форс на все каналы, для развёрток тон-кривой); иначе она
// оценивается пер-канально по референсной строке (см. estimate_channel_gammas).

/// Профиль, которым рендерит `psicode-tx single` (reference_profile из tx).
fn tx_profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        luma_bits: 2,
        chroma_mode: ChromaMode::Chroma1,
        gamma_g_q: 28,
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15,
        black_level_q: 2,
        noise_sigma_q: 12,
        mtf_limit_px: 6,
        torn_frames_q: 5,
        crosstalk_rg_q: 3,
        crosstalk_gb_q: 4,
        quiet_zone: 1,
        fec_overhead: 2,
        border: psicode_core::profile::BorderMode::LegacyInverted,
    }
}

/// Нагрузка клеток `psicode-tx single`: тот же splitmix64 с тем же сидом.
fn tx_single_cells(p: &CalibProfile) -> Vec<u8> {
    let bpc = symbol::bits_per_cell(p);
    let mask: u16 = if bpc >= 16 { u16::MAX } else { (1u16 << bpc) - 1 };
    let n = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
    let mut state = 0x0D15_EA5E_5EED_1234u64;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (((z ^ (z >> 31)) >> 24) as u16 & mask) as u8
        })
        .collect()
}

/// Гомография детекции: клеточные координаты (u,v) -> пиксели снимка.
fn apply_h(h: &[[f64; 3]; 3], u: f64, v: f64) -> (f64, f64) {
    let d = h[2][0] * u + h[2][1] * v + h[2][2];
    (
        (h[0][0] * u + h[0][1] * v + h[0][2]) / d,
        (h[1][0] * u + h[1][1] * v + h[1][2]) / d,
    )
}

/// Декод одного PPM-снимка экрана; `cell_override` — если tx рендерил
/// с --cell, отличным от профиля.
pub fn cmd_live(path: &str, cell_override: Option<u8>) {
    let (w, h, rgb) = match report::read_ppm(Path::new(path)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("не прочитать {path}: {e}");
            std::process::exit(1);
        }
    };
    println!("# live: {path} ({w}x{h})");

    // Плоскость яркости для ДЕТЕКЦИИ: детекция перцентильно-корреляционная и
    // монотонно-инвариантна к гамме, поэтому берём фиксированную (2.2) — тон
    // подберём позже по референсной строке уже для демодуляции.
    let luma: Vec<f32> = rgb
        .iter()
        .map(|p| (p[1] as f64 / 255.0).powf(CAMERA_GAMMA_DEFAULT) as f32)
        .collect();

    let det = match detect_symbol(w, h, &luma) {
        Ok(d) => d,
        Err(e) => {
            println!("детекция: НЕ НАЙДЕНО ({e:?})");
            std::process::exit(2);
        }
    };
    let (x0, y0) = apply_h(&det.homography, 0.0, 0.0);
    let (x1, y1) = apply_h(&det.homography, 61.0, 0.0);
    let (x2, y2) = apply_h(&det.homography, 61.0, 61.0);
    let (x3, y3) = apply_h(&det.homography, 0.0, 61.0);
    let (cx0, cy0) = apply_h(&det.homography, 30.0, 30.0);
    let (cx1, cy1) = apply_h(&det.homography, 31.0, 30.0);
    let px_per_cell = ((cx1 - cx0).powi(2) + (cy1 - cy0).powi(2)).sqrt();
    println!("детекция: НАЙДЕНО");
    println!("  score            : {:.4}", det.score);
    println!("  поворот          : {} x 90°", det.rotation_quadrants);
    println!("  px/клетку (центр): {px_per_cell:.2}");
    println!(
        "  углы             : ({x0:.0},{y0:.0}) ({x1:.0},{y1:.0}) ({x2:.0},{y2:.0}) ({x3:.0},{y3:.0})"
    );

    // демодуляция против известной нагрузки tx single
    let mut p = tx_profile();
    if let Some(c) = cell_override {
        p.cell_size_px = c;
    }
    let map = frame_map(&p, &det);

    // --- оценка тон-кривой по референсной строке (§3.4) ---
    // Приёмник имеет право на самокалибровку по референсной лесенке кадра. Если
    // PSICODE_CAMERA_GAMMA задана — форсируем её на все каналы (для развёрток);
    // иначе оцениваем пер-канальную гамму линеаризации по нейтральным клеткам
    // строки (K, W и 6 серых ступеней). Демодулятор дальше сам снимет
    // gain/offset по K/W — мы отдаём ему верную ФОРМУ кривой.
    let forced = std::env::var("PSICODE_CAMERA_GAMMA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());
    // СЫРОЙ (до гаммы) билинейный сэмпл rgb[0,1] с зажимом к краю — тот же
    // стиль map/sample, что у демодулятора; ядро (§3.4) оценивает по нему форму
    // тон-кривой на канал.
    let raw_sample = |x: f64, y: f64| -> [f32; 3] {
        let xc = x.clamp(0.0, (w - 1) as f64);
        let yc = y.clamp(0.0, (h - 1) as f64);
        let x0 = xc.floor() as usize;
        let y0 = yc.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = (xc - x0 as f64) as f32;
        let fy = (yc - y0 as f64) as f32;
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            let a = rgb[y0 * w + x0][c] as f32 * (1.0 - fx) + rgb[y0 * w + x1][c] as f32 * fx;
            let b = rgb[y1 * w + x0][c] as f32 * (1.0 - fx) + rgb[y1 * w + x1][c] as f32 * fx;
            o[c] = (a * (1.0 - fy) + b * fy) / 255.0;
        }
        o
    };
    let cg = match forced {
        Some(g) => [g; 3],
        None => tone::estimate_channel_gammas(&p, &map, &raw_sample),
    };
    println!(
        "  тон γ (R,G,B)    : {:.2}, {:.2}, {:.2}{}",
        cg[0],
        cg[1],
        cg[2],
        if forced.is_some() { " (форс)" } else { " (по реф-строке)" }
    );

    // сенсорно-линейные плоскости с ПЕР-КАНАЛЬНОЙ гаммой тона.
    let lin: Vec<[f32; 3]> = rgb
        .iter()
        .map(|q| {
            [
                (q[0] as f64 / 255.0).powf(cg[0]) as f32,
                (q[1] as f64 / 255.0).powf(cg[1]) as f32,
                (q[2] as f64 / 255.0).powf(cg[2]) as f32,
            ]
        })
        .collect();

    let sample = |x: f64, y: f64| -> [f32; 3] {
        // билинейная выборка с зажимом к краю
        let xc = x.clamp(0.0, (w - 1) as f64);
        let yc = y.clamp(0.0, (h - 1) as f64);
        let x0 = xc.floor() as usize;
        let y0 = yc.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fx = (xc - x0 as f64) as f32;
        let fy = (yc - y0 as f64) as f32;
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let a = lin[y0 * w + x0][c] * (1.0 - fx) + lin[y0 * w + x1][c] * fx;
            let b = lin[y1 * w + x0][c] * (1.0 - fx) + lin[y1 * w + x1][c] * fx;
            out[c] = a * (1.0 - fy) + b * fy;
        }
        out
    };
    // ректифицированный вид: что «видит» демодулятор (канонические координаты,
    // 8 px/клетку, лёгкая гамма для глаза) — рядом с входным файлом.
    {
        let out_cell = 8usize;
        let cells_span = symbol::GRID + 2 * p.quiet_zone_cells() as usize;
        let out_px = cells_span * out_cell;
        let scale = p.cell_size_px as f64 / out_cell as f64;
        let mut rect = vec![[0u8; 3]; out_px * out_px];
        for oy in 0..out_px {
            for ox in 0..out_px {
                let (x, y) = map(ox as f64 * scale, oy as f64 * scale);
                let s = sample(x, y);
                for c in 0..3 {
                    rect[oy * out_px + ox][c] =
                        ((s[c] as f64).max(0.0).powf(1.0 / cg[c]) * 255.0)
                            .clamp(0.0, 255.0) as u8;
                }
            }
        }
        let rect_path = format!("{}.rect.ppm", path);
        if report::write_ppm(Path::new(&rect_path), out_px, out_px, &rect).is_ok() {
            println!("  ректификация     : {rect_path}");
        }
    }

    let got = demod_symbol(&p, &map, &sample);
    let truth = tx_single_cells(&p);
    let bpc = symbol::bits_per_cell(&p);
    let wrong = got.iter().zip(&truth).filter(|(a, b)| a != b).count();
    let bit_err: u32 = got
        .iter()
        .zip(&truth)
        .map(|(a, b)| ((a ^ b) as u32).count_ones())
        .sum();
    let n = truth.len();
    let (c_start, c_end) = read_counters(&p, &map, &sample);
    println!("демодуляция ({} бит/клетку):", bpc);
    println!(
        "  клеток верно     : {}/{}  (SER {:.4})",
        n - wrong,
        n,
        wrong as f64 / n as f64
    );
    println!(
        "  бит верно        : {:.4} BER",
        bit_err as f64 / (n as f64 * bpc as f64)
    );
    println!("  счётчик кадра    : старт {c_start}, конец {c_end} (ожидается 0, 0)");
}
