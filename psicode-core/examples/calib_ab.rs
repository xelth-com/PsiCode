//! [ДИАГНОСТИКА] Head-to-head: РЕФЕРЕНСНАЯ СТРОКА §3.4 против ВНУТРИПОЛОСНОГО
//! КАЛИБРОВОЧНОГО КАДРА §4-IB на ОДНОМ И ТОМ ЖЕ синтетическом канале.
//!
//! ```text
//! cargo run --release -p psicode-core --example calib_ab
//! ```
//!
//! # Что сравнивается
//!
//! Оба источника дают одну и ту же величину — матрицу развязки каналов §3.4
//! `t̂ = N·s + q` — и обрабатываются ОДНИМИ И ТЕМИ ЖЕ оценщиками
//! ([`psicode_core::symbol::solve4x3`] внутри, гамма — через
//! [`psicode_core::tone::fit_linearisation_gamma`]). Отличается ТОЛЬКО источник
//! точек:
//!
//! * **реф.строка** — 57 клеток ВЫСОТОЙ В ОДНУ КЛЕТКУ (12 px), 16 цветов с
//!   периодом 16 клеток, то есть цвета меняются каждые 12 px;
//! * **калиб.кадр** — 24 плитки по 120×72 px, 12 классов, каждый в двух
//!   точечно-симметричных копиях; измеряется внутренняя часть плитки.
//!
//! # Метрика
//!
//! `min_α ‖α·N̂ − N‖_F / ‖N‖_F` — ошибка ФОРМЫ смешивания. Общий скаляр из
//! метрики убран намеренно: поле освещённости входит во все три канала
//! одинаково, поэтому любая оценка по патчам несёт множитель `1/f̄`, а
//! демодулятор §5.1-CL делит на ИЗМЕРЕННУЮ сумму каналов и этот множитель
//! сокращает точно. Вредна только СМЕСЬ каналов — её и меряем. Абсолютная
//! норма печатается рядом для контроля.
//!
//! # Модель канала
//!
//! `гамма дисплея -> Y/Cb/Cr, блюр яркости σl и НЧ-фильтр ХРОМЫ σc -> поле
//! освещённости -> спектральное перекрытие сенсора (3×3) -> вуаль -> шум ->
//! обратная гамма камеры`. Смешивание НЕОТРИЦАТЕЛЬНО (перекрытие спектральных
//! чувствительностей); отрицательные коэффициенты, которыми меряют «кросстолк
//! −0.26», живут в ОБРАТНОЙ матрице — то есть в CCM, которую приёмник и должен
//! восстановить.
//!
//! `σc` — единственный параметр, ради которого всё затевалось. Его реальное
//! значение оценивается из лестницы масштабов: усиление оси Im падает 0.900
//! (блок 120 px) -> 0.757 (блок 12 px), то есть 12-px патч сохраняет ≈ 0.84
//! хромы 120-px патча, что для гауссова НЧ даёт **σc ≈ 3 px**.

use psicode_core::calframe;
use psicode_core::symbol::{self, ChannelMatrix};
use psicode_core::tone;
use psicode_core::{CalibProfile, ChromaMode};

const CELL: u8 = 12;

fn profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: CELL,
        frame_hold_periods: 6,
        luma_bits: 1,
        chroma_mode: ChromaMode::ConstLuma1,
        gamma_g_q: 28,
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15,
        black_level_q: 2,
        noise_sigma_q: 0,
        mtf_limit_px: 6,
        torn_frames_q: 0,
        crosstalk_rg_q: 0,
        crosstalk_gb_q: 0,
        quiet_zone: 1,
        fec_overhead: 2,
        border: psicode_core::profile::BorderMode::LegacyInverted,
    }
}

// ---------------------------------------------------------------------------
// Канал
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn gauss(&mut self) -> f64 {
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy)]
struct Chan {
    gamma: [f64; 3],
    sigma_luma: f64,
    sigma_chroma: f64,
    mix: [[f64; 3]; 3],
    off: [f64; 3],
    field: (f64, f64),
    noise: f64,
    seed: u64,
}

/// Замеренный канал (Galaxy A22 / Note 10 Lite): кросстолк до 0.26 в прямой
/// (спектральной) форме, поле 0.86 -> 0.62 по кадру, шум 1.79 кода на клетку.
fn measured(sigma_luma: f64, sigma_chroma: f64, noise: f64) -> Chan {
    Chan {
        gamma: [2.2, 2.2, 2.25],
        sigma_luma,
        sigma_chroma,
        mix: [[0.95, 0.20, 0.02], [0.05, 1.00, 0.26], [0.01, 0.08, 0.90]],
        off: [0.012, 0.010, 0.015],
        field: (0.86, 0.62),
        noise,
        seed: 0x51C0_DE00_1122_3344,
    }
}

/// `s = A·t + off` => `t = A⁻¹s − A⁻¹off`: истинная матрица развязки.
fn truth(ch: &Chan) -> ChannelMatrix {
    let a = ch.mix;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let mut n = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            n[i][j] = (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) / det;
        }
    }
    let mut q = [0.0f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            q[i] -= n[i][j] * ch.off[j];
        }
    }
    ChannelMatrix { n, q }
}

fn blur(src: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let r = (3.0 * sigma).ceil().min(24.0) as isize;
    let mut kern = vec![0.0f64; (2 * r + 1) as usize];
    let mut ks = 0.0;
    for (j, kv) in kern.iter_mut().enumerate() {
        let d = j as isize - r;
        *kv = (-(d * d) as f64 / (2.0 * sigma * sigma)).exp();
        ks += *kv;
    }
    for kv in kern.iter_mut() {
        *kv /= ks;
    }
    let cx = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, &kv) in kern.iter().enumerate() {
                acc += kv * src[y * w + cx(x as isize + j as isize - r, w)];
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut dst = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, &kv) in kern.iter().enumerate() {
                acc += kv * tmp[cx(y as isize + j as isize - r, h) * w + x];
            }
            dst[y * w + x] = acc;
        }
    }
    dst
}

fn apply(rgb: &[[u8; 3]], size: usize, ch: &Chan) -> Vec<[f32; 3]> {
    let n = size * size;
    let mut lin = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
    for i in 0..n {
        for c in 0..3 {
            lin[c][i] = (rgb[i][c] as f64 / 255.0).powf(ch.gamma[c]);
        }
    }
    let mut y = vec![0.0f64; n];
    let mut cb = vec![0.0f64; n];
    let mut cr = vec![0.0f64; n];
    for i in 0..n {
        y[i] = 0.299 * lin[0][i] + 0.587 * lin[1][i] + 0.114 * lin[2][i];
        cb[i] = lin[2][i] - y[i];
        cr[i] = lin[0][i] - y[i];
    }
    let y = blur(&y, size, size, ch.sigma_luma);
    let cb = blur(&cb, size, size, ch.sigma_chroma);
    let cr = blur(&cr, size, size, ch.sigma_chroma);
    for i in 0..n {
        let r = y[i] + cr[i];
        let b = y[i] + cb[i];
        lin[1][i] = (y[i] - 0.299 * r - 0.114 * b) / 0.587;
        lin[0][i] = r;
        lin[2][i] = b;
    }
    let (f0, f1) = ch.field;
    let c = (size as f64 - 1.0) / 2.0;
    let rmax = 2.0f64.sqrt() * c;
    let mut rng = Rng(ch.seed | 1);
    let noise = ch.noise / 255.0;
    let mut out = vec![[0.0f32; 3]; n];
    for i in 0..n {
        let (px, py) = ((i % size) as f64, (i / size) as f64);
        let rr = ((px - c).powi(2) + (py - c).powi(2)).sqrt() / rmax;
        let f = f0 + (f1 - f0) * rr;
        let t = [lin[0][i] * f, lin[1][i] * f, lin[2][i] * f];
        for cc in 0..3 {
            let mut v = ch.off[cc];
            for j in 0..3 {
                v += ch.mix[cc][j] * t[j];
            }
            v += noise * rng.gauss();
            out[i][cc] = v.max(0.0).powf(1.0 / 2.2) as f32;
        }
    }
    out
}

fn sampler(buf: Vec<[f32; 3]>, size: usize) -> impl Fn(f64, f64) -> [f32; 3] {
    move |x: f64, y: f64| {
        let xi = (x.round() as isize).clamp(0, size as isize - 1) as usize;
        let yi = (y.round() as isize).clamp(0, size as isize - 1) as usize;
        buf[yi * size + xi]
    }
}

fn random_cells(bpc: u32, seed: u64) -> Vec<u8> {
    let mask: u16 = if bpc >= 16 { u16::MAX } else { (1u16 << bpc) - 1 };
    let mut rng = Rng(seed | 1);
    (0..symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS)
        .map(|_| ((rng.next_u64() >> 24) as u16 & mask) as u8)
        .collect()
}

/// Матрица + гамма по РЕФЕРЕНСНОЙ СТРОКЕ §3.4 обычного payload-кадра.
fn from_reference_row(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    s: &dyn Fn(f64, f64) -> [f32; 3],
) -> Option<(ChannelMatrix, [f64; 3])> {
    let g = tone::estimate_channel_gammas(p, map, s);
    let lin = |x: f64, y: f64| -> [f32; 3] {
        let r = s(x, y);
        [
            (r[0].max(0.0) as f64).powf(g[0]) as f32,
            (r[1].max(0.0) as f64).powf(g[1]) as f32,
            (r[2].max(0.0) as f64).powf(g[2]) as f32,
        ]
    };
    let gp = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    symbol::estimate_matrix_reference_row(p, &gp, map, &lin).map(|m| (m, g))
}

fn main() {
    let p = profile();
    let map = |u: f64, v: f64| (u, v);
    let (bw, bh, rw, rh, gx, gy) = calframe::layout_metrics(CELL as usize);

    println!("PsiCode §4-IB — head-to-head: референсная строка §3.4 vs калибровочный кадр\n");
    println!(
        "раскладка при cell {CELL} px: тело плитки ≥ {bw}×{bh} px, измеряемая часть ≥ {rw}×{rh} px, \
         до чужого цвета ≥ {gx} px по X и ≥ {gy} px по Y"
    );
    println!(
        "  {} плиток = {} классов × 2 точечно-симметричные копии; маркер {} строк; \
         реф.строка §3.4 для сравнения: патч {}×{} px",
        calframe::TILES,
        calframe::CLASSES,
        calframe::MARKER_ROWS,
        CELL,
        CELL
    );
    println!(
        "расписание §4-IB.3: накладные расходы {:.3} % @1k, {:.3} % @10k, {:.3} % @100k кадров; \
         худшее ожидание позднего приёмника {} кадров",
        100.0 * calframe::schedule_overhead(1_000),
        100.0 * calframe::schedule_overhead(10_000),
        100.0 * calframe::schedule_overhead(100_000),
        calframe::worst_join_wait(4096),
    );

    println!("\nσ_luma  σ_chroma  шум |   реф.строка       калиб.кадр      выигрыш");
    println!("  px       px    код/255 | форма   (абс)   форма   (абс)      ×");
    println!("{}", "-".repeat(78));

    let mut best_gain: f64 = f64::INFINITY;
    for &sl in &[1.0f64, 2.0, 3.0] {
        for &sc in &[1.0f64, 2.0, 3.0, 4.0, 6.0, 10.0] {
            for &noise in &[1.79f64, 6.0] {
                let ch = measured(sl, sc, noise);
                let tr = truth(&ch);

                // (a) референсная строка обычного payload-кадра
                let cells = random_cells(symbol::bits_per_cell(&p), 0xC0FF_EE00);
                let pf = symbol::render_symbol_counter(&p, &cells, 3);
                let ps = sampler(apply(&pf.rgb, pf.size_px, &ch), pf.size_px);
                let (rf, ra) = match from_reference_row(&p, &map, &ps) {
                    Some((m, _)) => (m.shape_rel_error(&tr), m.frobenius_rel_error(&tr)),
                    None => (f64::INFINITY, f64::INFINITY),
                };

                // (b) внутриполосный калибровочный кадр
                let cf = calframe::render_calibration_frame(&p, 0);
                let cs = sampler(apply(&cf.rgb, cf.size_px, &ch), cf.size_px);
                let est = calframe::estimate_from_frame(&p, &map, &cs);
                let (kf, ka) = if est.ok {
                    (
                        est.matrix.shape_rel_error(&tr),
                        est.matrix.frobenius_rel_error(&tr),
                    )
                } else {
                    (f64::INFINITY, f64::INFINITY)
                };
                let gain = rf / kf;
                best_gain = best_gain.min(gain);
                println!(
                    "{sl:>5.1}  {sc:>7.1}  {noise:>6.2} | {rf:>6.4} ({ra:>6.4}) {kf:>6.4} ({ka:>6.4})  {gain:>7.1}"
                );
            }
        }
    }
    println!("{}", "-".repeat(78));
    println!("минимальный выигрыш по развёртке: ×{best_gain:.2}");

    // --- восстановление гаммы и σ блюра ---
    println!("\nвосстановление γ линеаризации и σ блюра калибровочным кадром:");
    println!("σ_true  σ̂x     σ̂y     γ̂r    γ̂g    γ̂b    невязка  разброс половин  поле");
    for &sl in &[0.5f64, 1.0, 1.5, 2.0, 3.0, 4.0] {
        let ch = measured(sl, 3.0, 1.79);
        let cf = calframe::render_calibration_frame(&p, 0);
        let cs = sampler(apply(&cf.rgb, cf.size_px, &ch), cf.size_px);
        let e = calframe::estimate_from_frame(&p, &map, &cs);
        println!(
            "{sl:>5.1}  {:>5.2}  {:>5.2}  {:>4.2}  {:>4.2}  {:>4.2}  {:>7.4}  {:>14.4}  {:.3}..{:.3}",
            e.sigma_x_px,
            e.sigma_y_px,
            e.gammas[0],
            e.gammas[1],
            e.gammas[2],
            e.residual,
            e.spatial_spread,
            e.field_lo,
            e.field_hi,
        );
    }

    // --- запас детекции маркера ---
    println!("\nзапас детекции маркера §4-IB.2 (порог ρ ≥ {:.2}, U ≥ {:.2}):", calframe::MARKER_RHO_MIN, calframe::UNIFORMITY_MIN);
    println!("σ_luma  шум |  калиб.кадр ρ / U     |  payload-кадр ρ / U");
    for &sl in &[0.5f64, 1.0, 2.0, 4.0, 6.0] {
        for &noise in &[1.79f64, 8.0] {
            let ch = measured(sl, 3.0, noise);
            let cf = calframe::render_calibration_frame(&p, 0);
            let cs = sampler(apply(&cf.rgb, cf.size_px, &ch), cf.size_px);
            let (crho, cu) = (
                calframe::marker_score(&p, &map, &cs),
                calframe::tile_uniformity(&p, &map, &cs),
            );
            let mut prho: f64 = 0.0;
            let mut pu = f64::NEG_INFINITY;
            for seed in 0..8u64 {
                let cells = random_cells(symbol::bits_per_cell(&p), 0x1234 + seed);
                let pf = symbol::render_symbol_counter(&p, &cells, 3);
                let ps = sampler(apply(&pf.rgb, pf.size_px, &ch), pf.size_px);
                prho = prho.max(calframe::marker_score(&p, &map, &ps).abs());
                pu = pu.max(calframe::tile_uniformity(&p, &map, &ps));
            }
            println!("{sl:>5.1}  {noise:>4.2} |  {crho:>6.4} / {cu:>7.4}  |  {prho:>6.4} / {pu:>8.3}");
        }
    }
}
