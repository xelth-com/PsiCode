//! Экспериментальные развёртки для решений по спеке (подкоманда `exp`).
//!
//! Две изолированные студии, обе поверх уже существующей машинерии render/
//! channel/demod (psicode-core + pipeline.rs). Ничего в core/rx/android/docs не
//! трогает — только читает публичный API и рендерит варианты локально.
//!
//! # Студия 1 — ISP temporal mixing
//! Живой замер на Samsung A22: ISP камеры усредняет соседние кадры дисплея даже
//! с выключенным шумоподавлением. Модель: captured = (1−α)·current + α·prev, где
//! prev — предыдущий (независимый) tx-кадр. Смешиваем ДВА отрендеренных кадра в
//! drive-домене ДО оптического канала (тот же приём, что `framed::compose_torn`),
//! затем один композит гоняем сквозь `apply_channel`. Меряем SER, выживаемость
//! страйпа (399 клеток) и построчную выживаемость (57 клеток) против α.
//!
//! # Студия 2 — детектируемость рамки/маяков под блюром
//! Критика владельца: §3.2-рамка (двойное кольцо, внутреннее = инверсия внешнего)
//! под блюром σ≳1 клетки локально усредняется в серое и «самостирается». Четыре
//! варианта рамки на одной геометрии рендерим локально (яркостный шаблон) и меряем
//! matched-filter SNR: корреляцию размытой области рамки с её же неразмытым
//! шаблоном, нормированную на фоновый шум. Отвечаем: при каком σ каждый вариант
//! падает ниже общего порога (5× шумового пола).

use crate::channel::ChannelParams;
use crate::pipeline::{apply_channel, gaussian_kernel, STRIPE_ROWS};
use crate::report;
use crate::rng::{seed_for, Rng};
use psicode_core::detect::detect_symbol;
use psicode_core::symbol::{self, zc_binary, Frame, GRID};
use psicode_core::{CalibProfile, ChromaMode};
use std::time::Instant;

/// Эталонный профиль §7.4 (та же телеметрия, что `main::reference_profile`) —
/// «правда канала». Локальная копия, чтобы не тянуть private main.
fn base_profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        luma_bits: 3,
        chroma_mode: ChromaMode::Chroma2,
        gamma_g_q: 28,
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15,
        black_level_q: 2,
        noise_sigma_q: 12, // σ ≈ 2.0 градации
        mtf_limit_px: 6,
        torn_frames_q: 5,
        crosstalk_rg_q: 3,
        crosstalk_gb_q: 4,
        quiet_zone: 1,
        fec_overhead: 2,
        border: psicode_core::profile::BorderMode::LegacyInverted,
    }
}

/// Профиль эталона с подменёнными luma_bits и chroma_mode (одна точка конфига).
fn config_profile(base: &CalibProfile, luma: u8, chroma: ChromaMode) -> CalibProfile {
    let mut p = *base;
    p.luma_bits = luma;
    p.chroma_mode = chroma;
    p
}

pub fn cmd_exp() {
    let t0 = Instant::now();
    study1_mixing();
    println!();
    study2_border();
    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

// ===========================================================================
// СТУДИЯ 1 — ISP temporal mixing
// ===========================================================================

const MIX_TRIALS: usize = 20;
/// Значения α (доля предыдущего кадра в смеси).
const ALPHAS: [f64; 6] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
/// Display-px на клетку для студии 1: 8 при px/клетку 8 даёт масштаб 1:1 (без
/// супердискретизации), вчетверо дешевле рендера, чем эталонные 16 px; α-тренды
/// (нас интересуют именно они) от этого не меняются.
const MIX_CELL_PX: u8 = 8;

/// Розыгрыш случайных клеточных символов, равномерно по 2^bits_per_cell
/// (реплика private `pipeline::draw_random_cells`; порядок расхода ГПСЧ важен).
fn draw_cells(p: &CalibProfile, rng: &mut Rng) -> Vec<u8> {
    let bpc = symbol::bits_per_cell(p);
    let levels = 1u32 << bpc;
    let n = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
    (0..n).map(|_| rng.next_u32_below(levels) as u8).collect()
}

/// Смешивание двух отрендеренных кадров в drive-домене: mixed = (1−α)·cur + α·prev
/// поканально, округление в u8 (аналог `framed::compose_torn`, но линейный блендинг).
fn blend_frames(cur: &Frame, prev: &Frame, alpha: f64) -> Frame {
    assert_eq!(cur.size_px, prev.size_px, "смесь: кадры разной геометрии");
    let rgb: Vec<[u8; 3]> = cur
        .rgb
        .iter()
        .zip(prev.rgb.iter())
        .map(|(a, b)| {
            let mut px = [0u8; 3];
            for c in 0..3 {
                let v = (1.0 - alpha) * a[c] as f64 + alpha * b[c] as f64;
                px[c] = v.round().clamp(0.0, 255.0) as u8;
            }
            px
        })
        .collect();
    Frame {
        size_px: cur.size_px,
        quiet_cells: cur.quiet_cells,
        rgb,
    }
}

/// Одна попытка со смешиванием: (клетки ТЕКУЩЕГО кадра — истина, демодулированные).
/// Предыдущий кадр — независимая случайная нагрузка (помеха). Всё из одного ГПСЧ
/// по (point, trial): при фиксированном point α — единственный меняющийся рычаг
/// (нагрузки и шум идентичны, порядок расхода ГПСЧ не зависит от α).
fn run_mix_trial(
    p: &CalibProfile,
    ch: &ChannelParams,
    alpha: f64,
    point: usize,
    trial: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng::new(seed_for(point, trial));
    let cur = draw_cells(p, &mut rng);
    let prev = draw_cells(p, &mut rng);
    let f_cur = symbol::render_symbol(p, &cur);
    let f_prev = symbol::render_symbol(p, &prev);
    let mixed = blend_frames(&f_cur, &f_prev, alpha);
    let (img, geom) = apply_channel(&mixed, ch, &mut rng);
    let map = |u: f64, v: f64| geom.forward(u, v);
    let sample = |x: f64, y: f64| img.sample(x, y);
    let got = symbol::demod_symbol(p, &map, &sample);
    (cur, got)
}

/// Доля выживших страйпов (399/342 клетки): страйп жив ⇔ все его клетки верны (§6.2).
fn stripe_survival(sent: &[u8], got: &[u8]) -> (usize, usize) {
    let cols = symbol::PAYLOAD_COLS;
    let mut surv = 0usize;
    let mut r0 = 0usize;
    for &rows in &STRIPE_ROWS {
        let (s, e) = (r0 * cols, (r0 + rows) * cols);
        if sent[s..e].iter().zip(&got[s..e]).all(|(a, b)| a == b) {
            surv += 1;
        }
        r0 += rows;
    }
    (surv, STRIPE_ROWS.len())
}

/// Доля выживших payload-строк (57 клеток каждая): строка жива ⇔ все 57 клеток верны.
/// Это гипотетическая L3-гранулярность «CRC на строку» вместо «CRC на страйп».
fn row_survival(sent: &[u8], got: &[u8]) -> (usize, usize) {
    let cols = symbol::PAYLOAD_COLS;
    let rows = symbol::PAYLOAD_ROWS;
    let mut surv = 0usize;
    for r in 0..rows {
        let (s, e) = (r * cols, (r + 1) * cols);
        if sent[s..e].iter().zip(&got[s..e]).all(|(a, b)| a == b) {
            surv += 1;
        }
    }
    (surv, rows)
}

/// Средние (SER, доля живых страйпов, доля живых строк) по MIX_TRIALS попыткам.
fn mix_point(p: &CalibProfile, ch: &ChannelParams, alpha: f64, point: usize) -> (f64, f64, f64) {
    let (mut wrong, mut total) = (0usize, 0usize);
    let (mut ss, mut st) = (0usize, 0usize);
    let (mut rs, mut rt) = (0usize, 0usize);
    for t in 0..MIX_TRIALS {
        let (sent, got) = run_mix_trial(p, ch, alpha, point, t);
        wrong += sent.iter().zip(&got).filter(|(a, b)| a != b).count();
        total += sent.len();
        let (a, b) = stripe_survival(&sent, &got);
        ss += a;
        st += b;
        let (c, d) = row_survival(&sent, &got);
        rs += c;
        rt += d;
    }
    (
        wrong as f64 / total as f64,
        ss as f64 / st as f64,
        rs as f64 / rt as f64,
    )
}

/// Конфиги студии 1: (метка, luma_bits, chroma_mode) — bpc = 1, 2, 3.
const MIX_CONFIGS: [(&str, u8, ChromaMode); 3] = [
    ("1bpc luma1+Mono", 1, ChromaMode::Mono),
    ("2bpc luma2+Mono", 2, ChromaMode::Mono),
    ("3bpc luma2+Chroma1", 2, ChromaMode::Chroma1),
];

fn study1_mixing() {
    let base = base_profile();
    println!("# Студия 1 — ISP temporal mixing (captured = (1−α)·cur + α·prev)");
    println!(
        "канал: телеметрия §7.4 (кросстолк 6/8%, шум σ≈2 град, matched γ), σ_blur=1, px/клетку 8, \
         display-cell {MIX_CELL_PX}px (масштаб 1:1), {MIX_TRIALS} попыток/точку. \
         Истина — ТЕКУЩИЙ кадр; предыдущий — независимая помеха."
    );

    // предрасчёт всех точек: [config][alpha] -> (ser, stripe, row)
    let mut data = [[(0.0f64, 0.0f64, 0.0f64); ALPHAS.len()]; MIX_CONFIGS.len()];
    for (ci, &(_, luma, chroma)) in MIX_CONFIGS.iter().enumerate() {
        let mut p = config_profile(&base, luma, chroma);
        p.cell_size_px = MIX_CELL_PX;
        let mut ch = ChannelParams::from_profile(&p);
        ch.px_per_cell = 8.0; // масштаб 1:1 (cell_size_px = 8)
        ch.set_blur(1.0);
        for (ai, &alpha) in ALPHAS.iter().enumerate() {
            let point = 7000 + ci * 10 + ai;
            data[ci][ai] = mix_point(&p, &ch, alpha, point);
        }
    }

    let header = "| config \\ α → | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |";
    let sep = "|---|---|---|---|---|---|---|";

    println!("\n## 1.1 SER vs α");
    println!("{header}");
    println!("{sep}");
    for (ci, &(label, _, _)) in MIX_CONFIGS.iter().enumerate() {
        let cells: Vec<String> = (0..ALPHAS.len())
            .map(|ai| report::sig4(data[ci][ai].0))
            .collect();
        println!("{}", report::table_row(label, &cells));
    }

    println!("\n## 1.2 Выживаемость СТРАЙПА (399-cell, доля 8 страйпов)");
    println!("{header}");
    println!("{sep}");
    for (ci, &(label, _, _)) in MIX_CONFIGS.iter().enumerate() {
        let cells: Vec<String> = (0..ALPHAS.len())
            .map(|ai| format!("{:.3}", data[ci][ai].1))
            .collect();
        println!("{}", report::table_row(label, &cells));
    }

    println!("\n## 1.3 Выживаемость СТРОКИ (57-cell, доля 55 строк)");
    println!("{header}");
    println!("{sep}");
    for (ci, &(label, _, _)) in MIX_CONFIGS.iter().enumerate() {
        let cells: Vec<String> = (0..ALPHAS.len())
            .map(|ai| format!("{:.3}", data[ci][ai].2))
            .collect();
        println!("{}", report::table_row(label, &cells));
    }

    // производные факты для вердикта.
    println!("\n### производные факты");
    for (ci, &(label, _, _)) in MIX_CONFIGS.iter().enumerate() {
        // максимальная α, где страйп ещё жив (доля > 0), и где строка ещё жива.
        let last_stripe = ALPHAS
            .iter()
            .zip(data[ci].iter())
            .filter(|(_, d)| d.1 > 0.0)
            .map(|(a, _)| *a)
            .fold(-1.0f64, f64::max);
        let last_row = ALPHAS
            .iter()
            .zip(data[ci].iter())
            .filter(|(_, d)| d.2 > 0.0)
            .map(|(a, _)| *a)
            .fold(-1.0f64, f64::max);
        let stripe_at_02 = data[ci][2].1; // α=0.2
        let row_at_02 = data[ci][2].2;
        println!(
            "- {label}: страйп жив до α={:.1}, строка жива до α={:.1}; при α=0.2 страйп {:.3}, строка {:.3}",
            last_stripe.max(0.0),
            last_row.max(0.0),
            stripe_at_02,
            row_at_02
        );
    }
}

// ===========================================================================
// СТУДИЯ 2 — детектируемость рамки/маяков под блюром
// ===========================================================================

/// px на клетку при рендере вариантов рамки (совпадает с px/клетку 8 канала).
const CELL2: usize = 8;
/// Клеток фона вокруг кольца: тихая зона (4) + запас под блюр (для σ до 16 px
/// радиус ядра ~48 px = 6 клеток; итого 10 держит края чистыми под clamp-to-edge).
const PAD2: usize = 10;
/// σ маяка ψ00 (Гаусс) в клетках (§3.2 идея владельца).
const BEACON_SIGMA_CELLS: f64 = 1.5;
/// Порог детекции: matched-filter SNR = 5× фонового шумового пола.
const SNR_THRESHOLD: f64 = 5.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// A: текущее двойное кольцо, внутреннее = инверсия внешнего (baseline §3.2).
    RingInverted,
    /// B: двойное кольцо, внутреннее = ПОВТОР внешнего (без инверсии).
    RingRepeated,
    /// C: сплошные кольца — внешнее белое, внутреннее чёрное (грубый концентрический край).
    SolidEdge,
    /// D: текущее кольцо + 4 гауссовых угловых маяка в тихой зоне (ψ00).
    RingBeacons,
    /// E: ТОЛЬКО 4 гауссовых маяка (изоляция вклада ψ00, без кольца).
    BeaconsOnly,
}

/// Яркость клетки по белизне.
#[inline]
fn wk(white: bool) -> f64 {
    if white {
        1.0
    } else {
        0.0
    }
}

/// Рендер яркостного шаблона рамки варианта: фон 0.5 (средне-серый), кольцо в
/// [0,1], опционально маяки. Возврат (буфер f64 построчно, w, h). Кольцо строится
/// той же раскраской, что `symbol::build_symbol_cells` (корни ЗЧ по сторонам,
/// приоритет углов верх>право>низ>лево).
fn render_border_luma(variant: Variant) -> (Vec<f64>, usize, usize) {
    let g = GRID;
    let last = g - 1;

    // карта белизны ВНЕШНЕГО кольца (как в build_symbol_cells).
    let mut owhite = vec![false; g * g];
    for n in 0..g {
        owhite[(last - n) * g] = zc_binary(4, n); // лево, корень 4
    }
    for n in 0..g {
        owhite[last * g + (last - n)] = zc_binary(3, n); // низ, корень 3
    }
    for n in 0..g {
        owhite[n * g + last] = zc_binary(2, n); // право, корень 2
    }
    for n in 0..g {
        owhite[n] = zc_binary(1, n); // верх, корень 1 (красится последним)
    }

    // клеточная сетка яркости; интерьер/фон = 0.5.
    let mut cell = vec![0.5f64; g * g];
    match variant {
        // только маяки — кольцо не рисуем (сетка остаётся серой).
        Variant::BeaconsOnly => {}
        Variant::SolidEdge => {
            // внешнее кольцо сплошь белое, внутреннее сплошь чёрное.
            for x in 0..g {
                cell[x] = 1.0;
                cell[last * g + x] = 1.0;
            }
            for y in 0..g {
                cell[y * g] = 1.0;
                cell[y * g + last] = 1.0;
            }
            for y in 1..last {
                cell[y * g + 1] = 0.0;
            }
            for x in 1..last {
                cell[(last - 1) * g + x] = 0.0;
            }
            for y in 1..last {
                cell[y * g + (last - 1)] = 0.0;
            }
            for x in 1..last {
                cell[g + x] = 0.0;
            }
        }
        _ => {
            // внешнее кольцо по ЗЧ.
            for x in 0..g {
                cell[x] = wk(owhite[x]);
                cell[last * g + x] = wk(owhite[last * g + x]);
            }
            for y in 0..g {
                cell[y * g] = wk(owhite[y * g]);
                cell[y * g + last] = wk(owhite[y * g + last]);
            }
            // внутреннее кольцо: инверсия (A/D) либо повтор (B) примыкающего внешнего.
            let inv = !matches!(variant, Variant::RingRepeated);
            for y in 1..last {
                cell[y * g + 1] = wk(owhite[y * g] ^ inv);
            }
            for x in 1..last {
                cell[(last - 1) * g + x] = wk(owhite[last * g + x] ^ inv);
            }
            for y in 1..last {
                cell[y * g + (last - 1)] = wk(owhite[y * g + last] ^ inv);
            }
            for x in 1..last {
                cell[g + x] = wk(owhite[x] ^ inv);
            }
        }
    }

    // раздутие клеток в пиксели с полем PAD2 фона вокруг.
    let total = g + 2 * PAD2;
    let w = total * CELL2;
    let h = w;
    let mut img = vec![0.5f64; w * h];
    for cy in 0..g {
        for cx in 0..g {
            let v = cell[cy * g + cx];
            let px0 = (PAD2 + cx) * CELL2;
            let py0 = (PAD2 + cy) * CELL2;
            for dy in 0..CELL2 {
                let row = (py0 + dy) * w + px0;
                for dx in 0..CELL2 {
                    img[row + dx] = v;
                }
            }
        }
    }

    // маяки D/E: 4 гауссовых пятна в тихой зоне, диагонально на 2 клетки наружу
    // от внешних углов кольца, амплитуда до белого (пик 1.0 над фоном 0.5).
    if matches!(variant, Variant::RingBeacons | Variant::BeaconsOnly) {
        let sb = BEACON_SIGMA_CELLS * CELL2 as f64;
        let lo = PAD2 as isize - 2; // клетка центра со стороны «наружу»
        let hi = (PAD2 + g) as isize + 1;
        let centers = [(lo, lo), (hi, lo), (lo, hi), (hi, hi)];
        for &(ccx, ccy) in &centers {
            let cxp = (ccx as f64 + 0.5) * CELL2 as f64;
            let cyp = (ccy as f64 + 0.5) * CELL2 as f64;
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f64 - cxp;
                    let dy = y as f64 - cyp;
                    let gauss = (-(dx * dx + dy * dy) / (2.0 * sb * sb)).exp();
                    let cur = img[y * w + x];
                    img[y * w + x] = (cur + 0.5 * gauss).min(1.0);
                }
            }
        }
    }

    (img, w, h)
}

/// Скалярный сепарабельный гауссов блюр σ (px), clamp-to-edge — та же физика, что
/// `pipeline::blur`, но по одному каналу (яркость).
fn blur_scalar(src: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as isize;
    let clampi = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;

    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &wt) in k.iter().enumerate() {
                let sx = clampi(x as isize + ki as isize - r, w);
                acc += wt * src[y * w + sx];
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &wt) in k.iter().enumerate() {
                let sy = clampi(y as isize + ki as isize - r, h);
                acc += wt * tmp[sy * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Matched-filter SNR: корреляция размытого шаблона с неразмытым (фон-вычтенным)
/// шаблоном, нормированная на шумовой пол σ_bg·‖t‖. t = template − 0.5 (отклонение
/// от известной серой тихой зоны), signal = Σ t·(blurred − 0.5). При σ=0 даёт
/// пиковый SNR = ‖t‖/σ_bg; растёт блюр -> сигнал коллапсирует -> SNR падает.
fn matched_snr(template: &[f64], blurred: &[f64], sigma_bg: f64) -> f64 {
    let mut sig = 0.0f64;
    let mut nrm = 0.0f64;
    for i in 0..template.len() {
        let t = template[i] - 0.5;
        sig += t * (blurred[i] - 0.5);
        nrm += t * t;
    }
    let nrm = nrm.sqrt();
    if nrm < 1e-12 {
        return 0.0;
    }
    sig / (sigma_bg * nrm)
}

/// Потолок поиска σ-пересечения (px). ~8 клеток — за пределом любого реального
/// дефокуса; вариант, не упавший здесь, помечается «>CAP».
const CROSS_CAP: f64 = 64.0;

/// σ (px), при котором SNR пересекает `thresh` (бинпоиск; SNR монотонно убывает
/// по σ). Возвращает `CROSS_CAP`, если пересечения нет в диапазоне.
fn crossing_sigma(template: &[f64], w: usize, h: usize, sigma_bg: f64, thresh: f64) -> f64 {
    let snr = |s: f64| matched_snr(template, &blur_scalar(template, w, h, s), sigma_bg);
    let mut hi = 1.0;
    while snr(hi) >= thresh && hi < CROSS_CAP {
        hi *= 2.0;
    }
    if snr(hi) >= thresh {
        return CROSS_CAP; // не пересекает в пределах диапазона
    }
    let mut lo = hi / 2.0; // snr(lo) >= thresh при hi>1; при hi=1 lo=0.5 — ок
    if snr(lo) < thresh {
        lo = 0.0;
    }
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        if snr(mid) >= thresh {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

const VARIANTS: [(&str, Variant); 5] = [
    ("A ring-inverted (baseline)", Variant::RingInverted),
    ("B ring-repeated", Variant::RingRepeated),
    ("C solid-edge", Variant::SolidEdge),
    ("D ring+beacons (ψ00)", Variant::RingBeacons),
    ("E beacons-only (ψ00)", Variant::BeaconsOnly),
];

/// σ (camera px) для таблицы SNR: до 4 клеток (32 px) блюра.
const SIGMAS2: [f64; 8] = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0];
/// σ (camera px) для дорогой кросс-проверки настоящим детектором (до 1 клетки —
/// дальше §3.2-детектор всё равно теряет ЗЧ-лок, а провал запускает медленный фолбэк).
const DETECT_SIGMAS: [f64; 4] = [1.0, 2.0, 4.0, 8.0];

fn study2_border() {
    let p = base_profile();
    let sigma_bg = p.noise_sigma() as f64 / 255.0;
    println!("# Студия 2 — детектируемость рамки/маяков под блюром");
    println!(
        "px/клетку {CELL2}; фоновый шум σ_bg = {:.5} (телеметрия §7.4, {:.2} град/255); \
         маяк σ={BEACON_SIGMA_CELLS} клетки. Порог = {SNR_THRESHOLD}× шумового пола.",
        sigma_bg,
        p.noise_sigma()
    );
    println!("метрика = matched-filter SNR (размытая рамка × собственный неразмытый шаблон / σ_bg·‖t‖).\n");

    // шаблоны.
    let templates: Vec<(&str, Vec<f64>, usize, usize)> = VARIANTS
        .iter()
        .map(|&(label, v)| {
            let (img, w, h) = render_border_luma(v);
            (label, img, w, h)
        })
        .collect();

    println!("## 2.1 matched-filter SNR vs blur σ (camera px; 8 px = 1 клетка)");
    println!("| variant \\ σ px → | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | σ@SNR=5 (px / клетки) |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for (label, img, w, h) in &templates {
        let cells: Vec<String> = SIGMAS2
            .iter()
            .map(|&s| {
                let snr = matched_snr(img, &blur_scalar(img, *w, *h, s), sigma_bg);
                report::sig4(snr)
            })
            .collect();
        let cross = crossing_sigma(img, *w, *h, sigma_bg, SNR_THRESHOLD);
        let cross_str = if cross >= CROSS_CAP {
            format!(">{:.0} / >{:.1}", CROSS_CAP, CROSS_CAP / CELL2 as f64)
        } else {
            format!("{:.1} / {:.1}", cross, cross / CELL2 as f64)
        };
        let mut row = report::table_row(label, &cells);
        row.push_str(&format!(" {cross_str} |"));
        println!("{row}");
    }

    // кросс-проверка настоящим детектором §3.2 (только A/B — это ЗЧ-кольца).
    // score, если detect_symbol залочился, иначе «—» (лок потерян). До 1 клетки σ.
    println!("\n## 2.2 кросс-проверка настоящим §3.2-детектором (detect_symbol score; A/B — ЗЧ-кольца)");
    println!("| variant \\ σ px → | 1 | 2 | 4 | 8 |");
    println!("|---|---|---|---|---|");
    for &(label, v) in VARIANTS.iter().take(2) {
        let (img, w, h) = render_border_luma(v);
        let cells: Vec<String> = DETECT_SIGMAS
            .iter()
            .map(|&s| {
                let blurred = blur_scalar(&img, w, h, s);
                let luma: Vec<f32> = blurred.iter().map(|&x| x as f32).collect();
                match detect_symbol(w, h, &luma) {
                    Ok(d) => format!("{:.2}", d.score),
                    Err(_) => "—".to_string(),
                }
            })
            .collect();
        println!("{}", report::table_row(&format!("{label} score"), &cells));
    }

    // производные факты для вердикта: скорость самостирания (не упирается в CAP).
    // полу-энергия — σ, где matched-SNR падает вдвое от пика; удержание — доля
    // пиковой SNR при σ = 1 и 2 клетки (порог самостирания владельца ~1 клетка).
    println!("\n### производные факты (скорость самостирания)");
    for (label, img, w, h) in templates.iter() {
        let peak = matched_snr(img, &blur_scalar(img, *w, *h, 0.01), sigma_bg);
        let half = crossing_sigma(img, *w, *h, sigma_bg, 0.5 * peak);
        let ret_1c = matched_snr(img, &blur_scalar(img, *w, *h, 8.0), sigma_bg) / peak;
        let ret_2c = matched_snr(img, &blur_scalar(img, *w, *h, 16.0), sigma_bg) / peak;
        let half_str = if half >= CROSS_CAP {
            format!(">{:.0} px (>{:.1} кл)", CROSS_CAP, CROSS_CAP / CELL2 as f64)
        } else {
            format!("{:.1} px ({:.1} кл)", half, half / CELL2 as f64)
        };
        println!(
            "- {label}: полу-энергия при σ≈{half_str}; удержано @1клетку {:.0}%, @2клетки {:.0}%",
            100.0 * ret_1c,
            100.0 * ret_2c
        );
    }
}
