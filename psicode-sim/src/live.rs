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

/// Оценивает пер-канальную гамму линеаризации по референсной строке (§3.4).
/// На канал подбираем γ так, чтобы (raw/255)^γ была максимально ЛИНЕЙНА
/// (d/255)^γ_профиля по НЕЙТРАЛЬНЫМ клеткам строки (K, W и 6 серых ступеней) —
/// т.е. форма кривой совпала с той, что ждёт демодулятор (он затем сам снимет
/// gain/offset по K/W). Заменяет жёсткую CAMERA_GAMMA самокалибровкой на кадр.
/// Возвращает [γr, γg, γb]; при нехватке точек — дефолт на все каналы.
fn estimate_channel_gammas(
    p: &CalibProfile,
    w: usize,
    h: usize,
    rgb: &[[u8; 3]],
    map: &dyn Fn(f64, f64) -> (f64, f64),
) -> [f64; 3] {
    const REF_PERIOD: usize = 16; // §3.4: K W R G B C M Y K W + 6 серых
    const REF_GRAY_STEPS: usize = 6;
    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    let white = (255.0 * p.white_level_pct() as f64 / 100.0).round();
    let black = (255.0 * p.black_level_pct() as f64 / 100.0).round();
    // нейтральный drive клетки строки по индексу периода (None — цветная).
    let neutral = |idx: usize| -> Option<f64> {
        match idx {
            0 | 8 => Some(black),
            1 | 9 => Some(white),
            10..=15 => {
                let step = (idx - 10) as f64;
                Some(black + (white - black) * step / (REF_GRAY_STEPS - 1) as f64)
            }
            _ => None,
        }
    };
    // сырой билинейный сэмпл rgb (0..1 на канал), зажим к краю.
    let raw = |x: f64, y: f64| -> [f64; 3] {
        let xc = x.clamp(0.0, (w - 1) as f64);
        let yc = y.clamp(0.0, (h - 1) as f64);
        let x0 = xc.floor() as usize;
        let y0 = yc.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let (fx, fy) = (xc - x0 as f64, yc - y0 as f64);
        let mut o = [0.0f64; 3];
        for c in 0..3 {
            let a = rgb[y0 * w + x0][c] as f64 * (1.0 - fx) + rgb[y0 * w + x1][c] as f64 * fx;
            let b = rgb[y1 * w + x0][c] as f64 * (1.0 - fx) + rgb[y1 * w + x1][c] as f64 * fx;
            o[c] = (a * (1.0 - fy) + b * fy) / 255.0;
        }
        o
    };
    // сэмпл клетки (cx,cy) канона: 4 точки ±cell/4 от центра (как sample_cell).
    let cell_raw = |cx: usize, cy: usize| -> [f64; 3] {
        let u = ((quiet + cx) * cell) as f64 + cell as f64 / 2.0;
        let v = ((quiet + cy) * cell) as f64 + cell as f64 / 2.0;
        let d = cell as f64 / 4.0;
        let mut acc = [0.0f64; 3];
        for &(sx, sy) in &[(-d, -d), (-d, d), (d, -d), (d, d)] {
            let (x, y) = map(u + sx, v + sy);
            let s = raw(x, y);
            for c in 0..3 {
                acc[c] += s[c];
            }
        }
        for c in 0..3 {
            acc[c] /= 4.0;
        }
        acc
    };
    // собираем (drive01, raw[c]) по нейтральным клеткам, ГРУППИРУЯ по уровню
    // drive (усредняем повторы периода -> чистая лесенка из ~6 точек).
    let mut levels: Vec<(f64, [f64; 3], usize)> = Vec::new();
    for ic in 0..symbol::INTERIOR {
        if let Some(d) = neutral(ic % REF_PERIOD) {
            let d01 = d / 255.0;
            let s = cell_raw(symbol::RING + ic, symbol::RING);
            match levels.iter_mut().find(|e| (e.0 - d01).abs() < 1e-6) {
                Some(e) => {
                    for c in 0..3 {
                        e.1[c] += s[c];
                    }
                    e.2 += 1;
                }
                None => levels.push((d01, s, 1)),
            }
        }
    }
    if levels.len() < 3 {
        return [CAMERA_GAMMA_DEFAULT; 3];
    }
    levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for e in &mut levels {
        for c in 0..3 {
            e.1[c] /= e.2 as f64;
        }
    }
    let gprof = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    let mut out = [CAMERA_GAMMA_DEFAULT; 3];
    // Крайние уровни (чёрный/белый) — якоря a,b, как у демодулятора (§3.4).
    let (dk, sk_all) = (levels[0].0, levels[0].1);
    let (dw, sw_all) = (levels[levels.len() - 1].0, levels[levels.len() - 1].1);
    for c in 0..3 {
        let gp = gprof[c];
        let (xk, xw) = (dk.powf(gp), dw.powf(gp));
        // подбираем γ ∈ [1,6] по МИНИМУМУ невязки восстановления drive: строим
        // сенсорную модель s=(raw)^γ, снимаем a,b по чёрн/бел, инвертируем как
        // демодулятор d̂=((s−b)/a)^(1/γ_проф) и штрафуем |d̂−d| по всем ступеням.
        let mut best_g = CAMERA_GAMMA_DEFAULT;
        let mut best_res = f64::MAX;
        let mut g = 1.0;
        while g <= 6.0 + 1e-9 {
            let sk = sk_all[c].max(1e-6).powf(g);
            let sw = sw_all[c].max(1e-6).powf(g);
            let denom = xw - xk;
            if denom.abs() > 1e-12 {
                let a = (sw - sk) / denom;
                let b = sk - a * xk;
                if a.abs() > 1e-12 {
                    let mut res = 0.0;
                    for e in &levels {
                        let s = e.1[c].max(1e-6).powf(g);
                        let base = ((s - b) / a).max(0.0);
                        let dhat = base.powf(1.0 / gp);
                        res += (dhat - e.0).powi(2);
                    }
                    if res < best_res {
                        best_res = res;
                        best_g = g;
                    }
                }
            }
            g += 0.05;
        }
        out[c] = best_g;
    }
    out
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
    let cg = match forced {
        Some(g) => [g; 3],
        None => estimate_channel_gammas(&p, w, h, &rgb, &map),
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
