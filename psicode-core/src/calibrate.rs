//! Калибровка (§4): рендер тестового паттерна и приёмные измерители канала,
//! дающие телеметрию [`CalibProfile`]. Замыкает ядро формата:
//! истина канала -> паттерн через канал -> измеренная телеметрия -> код профиля.
//!
//! Модуль под фичей `std` (нужна вещественная математика `powf`/`log2`).
//!
//! # Раскладка тестового паттерна (амендмент к SPEC §4)
//!
//! Полотно квадратное, `size_px × size_px`, растровое, drive-значения [R,G,B].
//! Логически это сетка `61 × 61` клеток, `cell = size_px / 61` display-px на
//! клетку (номинал: `cell = 12`, `size = 732`). Хвостовые px за `61·cell`
//! (при неделимом size_px) заполняются средне-серым (128).
//!
//! Слои (в клеточных координатах `cx,cy ∈ 0..=60`):
//!
//! ```text
//! ┌───────────────────────────────────────────────┐  cx,cy ∈ {0,1,59,60}
//! │ двойное ЗЧ-кольцо (§3.2), 2 клетки, чёрн/бел   │  — рамка детекции
//! │ ┌───────────────────────────────────────────┐ │  внутренняя область
//! │ │ клин частот: 11 полос по 3 клетки высотой  │ │  cy 2..=34  (33 клетки)
//! │ │   pitch 64,48,32,24,16,12,8,6,4,3,2 px      │ │  вертикальные штрихи
//! │ ├───────────────────────────────────────────┤ │
//! │ │ серая лесенка 16 ступеней (4 клетки)        │ │  cy 35..=38
//! │ ├───────────────────────────────────────────┤ │
//! │ │ R-лесенка 16 ступеней (4 клетки)            │ │  cy 39..=42
//! │ ├───────────────────────────────────────────┤ │
//! │ │ B-лесенка 16 ступеней (4 клетки)            │ │  cy 43..=46
//! │ ├───────────────────────────────────────────┤ │
//! │ │ белое поле (12 клеток) ≈ 18 % площади       │ │  cy 47..=58
//! │ └───────────────────────────────────────────┘ │
//! └───────────────────────────────────────────────┘
//! ```
//!
//! Внутренняя область по x охватывает px `[2·cell, 59·cell)` (57 клеток).
//!
//! * **Клин частот** (§4.2): 11 горизонтальных полос, снизу вверх по частоте.
//!   Полоса i несёт вертикальные чёрно-белые штрихи с шагом `WEDGE_PITCHES[i]`
//!   display-px (шаг задан в АБСОЛЮТНЫХ px полотна и не масштабируется с
//!   size_px, поэтому `mtf_limit_px` прямо сравним между размерами). Приёмник
//!   ищет мельчайший шаг с контрастом Мишельсона ≥ 0.4.
//! * **Лесенки** (§4.3): 16 ступеней по ширине, drive `round(255·i/15)`,
//!   i = 0..15. Серая `[d,d,d]`, R-только `[d,0,0]`, B-только `[0,0,d]`.
//! * **Белое поле** (§4.4): сплошной full-drive `[255,255,255]`, ≥ 10 % площади
//!   (здесь ≈ 18 %) — детекция клиппинга/блюминга и опорный белый.
//!
//! # Вне области действия
//!
//! Анимированный угловой счётчик (§4.5, измерение рвущихся кадров, tearing)
//! в статический паттерн НЕ входит: `torn_frames_q` измеряется отдельно по
//! серии снимков и пробрасывается в [`Telemetry::torn_frames_q`] снаружи.
//! Геометрия (гомография) в v0 считается известной (genie): `measure_channel`
//! принимает `map` из плоскости паттерна (display-px) в координаты снимка,
//! как в [`crate::symbol::demod_symbol`]. Интеграция с детекцией — позже.

use crate::profile::{CalibProfile, ChromaMode};
use crate::symbol::zc_binary;
use alloc::vec;
use alloc::vec::Vec;

/// Длина стороны полотна в клетках (= длина ЗЧ-последовательности N, §3.2).
pub const GRID: usize = 61;
/// Толщина ЗЧ-кольца в клетках (двойное кольцо, §3.2).
pub const RING: usize = 2;
/// Шаги клина частот в display-px полотна, от грубого к мелкому (§4.2).
pub const WEDGE_PITCHES: [usize; 11] = [64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2];
/// Число ступеней в каждой лесенке (§4.3).
pub const STAIR_STEPS: usize = 16;
/// Порог контраста Мишельсона для разрешимости полосы клина (§4.2).
pub const MTF_CONTRAST_THRESHOLD: f64 = 0.4;

const WHITE: [u8; 3] = [255, 255, 255];
const BLACK: [u8; 3] = [0, 0, 0];
const GRAY: [u8; 3] = [128, 128, 128];

// ---------------------------------------------------------------------------
// Геометрия: единый источник координат для рендера и измерителя
// ---------------------------------------------------------------------------

/// Раскладка паттерна в px, вычисляется из `size_px`. И рендер, и измеритель
/// пользуются ОДНИМ экземпляром, поэтому расходиться в геометрии не могут.
struct Layout {
    cell: usize,
    in_lo: usize, // начало внутренней области, px
    in_hi: usize, // конец внутренней области, px (исключительно) = 59·cell
    in_w: usize,  // ширина внутренней области, px = 57·cell
    // вертикальные границы полос (в px)
    wedge_y0: usize,
    wedge_band_h: usize,
    gray_y: (usize, usize),
    r_y: (usize, usize),
    b_y: (usize, usize),
    white_y: (usize, usize),
}

impl Layout {
    fn new(size_px: usize) -> Self {
        assert!(size_px >= GRID, "size_px должен быть >= {GRID}");
        let cell = size_px / GRID;
        assert!(cell >= 1, "size_px слишком мал для сетки {GRID}");
        let in_lo = RING * cell;
        let in_hi = (GRID - RING) * cell; // 59·cell
        let in_w = in_hi - in_lo; // 57·cell
        let wedge_band_h = 3 * cell;
        let wedge_y0 = in_lo;
        let stair_h = 4 * cell;
        let g0 = in_lo + 33 * cell;
        let r0 = g0 + stair_h;
        let b0 = r0 + stair_h;
        let w0 = b0 + stair_h;
        Layout {
            cell,
            in_lo,
            in_hi,
            in_w,
            wedge_y0,
            wedge_band_h,
            gray_y: (g0, r0),
            r_y: (r0, b0),
            b_y: (b0, w0),
            white_y: (w0, in_hi),
        }
    }

    /// Границы x центральной трети ступени i лесенки (без краёв — под блюр).
    fn step_center_x(&self, i: usize) -> (f64, f64) {
        let sx0 = self.in_lo + i * self.in_w / STAIR_STEPS;
        let sx1 = self.in_lo + (i + 1) * self.in_w / STAIR_STEPS;
        let m = (sx1 - sx0) / 4;
        ((sx0 + m) as f64, (sx1 - m) as f64)
    }

    /// Центральная треть по вертикали для полосы `(y0, y1)`.
    fn band_center_y(&self, band: (usize, usize)) -> (f64, f64) {
        let (y0, y1) = band;
        let m = (y1 - y0) / 4;
        ((y0 + m) as f64, (y1 - m) as f64)
    }

    /// drive серой ступени i: линейные шаги 0..255.
    fn stair_drive(i: usize) -> u8 {
        (255 * i / (STAIR_STEPS - 1)) as u8
    }
}

/// true = белый штрих клина в позиции x_rel внутри полосы с шагом `pitch`.
fn wedge_white(x_rel: usize, pitch: usize) -> bool {
    (x_rel % pitch) >= pitch / 2
}

/// Белизна внешнего ЗЧ-кольца в клетке (cx, cy) (§3.2). Приоритет углов
/// верх > право > низ > лево (как в [`crate::symbol`]: верх красится последним).
fn outer_white(cx: usize, cy: usize) -> Option<bool> {
    let last = GRID - 1;
    let mut w = None;
    if cx == 0 {
        w = Some(zc_binary(4, last - cy)); // лево: (0, 60−n)
    }
    if cy == last {
        w = Some(zc_binary(3, last - cx)); // низ: (60−n, 60)
    }
    if cx == last {
        w = Some(zc_binary(2, cy)); // право: (60, n)
    }
    if cy == 0 {
        w = Some(zc_binary(1, cx)); // верх: (n, 0)
    }
    w
}

/// Цвет клетки ЗЧ-кольца (внешнего или внутреннего) либо None для внутренней
/// области. Внутреннее кольцо — инверсия примыкающей внешней клетки (§3.2).
fn ring_color(cx: usize, cy: usize) -> Option<[u8; 3]> {
    let last = GRID - 1;
    // внешнее кольцо
    if let Some(w) = outer_white(cx, cy) {
        return Some(if w { WHITE } else { BLACK });
    }
    // внутреннее кольцо: тот же приоритет (лево, низ, право, верх — верх последним)
    let mut inv: Option<bool> = None;
    if cx == 1 {
        inv = outer_white(0, cy).map(|w| !w);
    }
    if cy == last - 1 {
        inv = outer_white(cx, last).map(|w| !w);
    }
    if cx == last - 1 {
        inv = outer_white(last, cy).map(|w| !w);
    }
    if cy == 1 {
        inv = outer_white(cx, 0).map(|w| !w);
    }
    inv.map(|w| if w { WHITE } else { BLACK })
}

// ---------------------------------------------------------------------------
// Передатчик: рендер тестового паттерна (§4)
// ---------------------------------------------------------------------------

/// Рендерит статический калибровочный паттерн §4 в drive-значениях [R,G,B],
/// растрово, `size_px × size_px`. Требует `size_px >= 61`; номинал 732.
pub fn render_calibration_pattern(size_px: usize) -> Vec<[u8; 3]> {
    let lay = Layout::new(size_px);
    let cell = lay.cell;
    let mut out = vec![GRAY; size_px * size_px];

    for py in 0..size_px {
        let cy = py / cell;
        for px in 0..size_px {
            let cx = px / cell;
            let color = if cx > GRID - 1 || cy > GRID - 1 {
                GRAY // хвостовой паддинг при неделимом size_px
            } else if let Some(c) = ring_color(cx, cy) {
                c
            } else {
                interior_pixel(&lay, px, py)
            };
            out[py * size_px + px] = color;
        }
    }
    out
}

/// Цвет пикселя внутренней области (клин / лесенки / белое поле).
fn interior_pixel(lay: &Layout, px: usize, py: usize) -> [u8; 3] {
    let x_rel = px - lay.in_lo;
    if py < lay.wedge_y0 + 33 * lay.cell {
        // клин частот: полоса по вертикали задаёт шаг
        let idx = (py - lay.wedge_y0) / lay.wedge_band_h;
        let pitch = WEDGE_PITCHES[idx.min(WEDGE_PITCHES.len() - 1)];
        return if wedge_white(x_rel, pitch) { WHITE } else { BLACK };
    }
    let step = (x_rel * STAIR_STEPS / lay.in_w).min(STAIR_STEPS - 1);
    let d = Layout::stair_drive(step);
    if py < lay.gray_y.1 {
        [d, d, d]
    } else if py < lay.r_y.1 {
        [d, 0, 0]
    } else if py < lay.b_y.1 {
        [0, 0, d]
    } else {
        WHITE // белое поле
    }
}

// ---------------------------------------------------------------------------
// Приёмник: измерение канала (§4)
// ---------------------------------------------------------------------------

/// Телеметрия канала: сырые q-поля §7.3 плюс физические величины.
///
/// `torn_frames_q` — проброс из внешнего измерения tearing (§4.5), в
/// `measure_channel` не вычисляется (там всегда 0), задаётся вызывающим.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Telemetry {
    // сырые квантованные поля (§7.3)
    pub gamma_g_q: u8,
    pub gamma_r_delta_q: u8,
    pub gamma_b_delta_q: u8,
    pub white_level_q: u8,
    pub black_level_q: u8,
    pub noise_sigma_q: u8,
    pub mtf_limit_px: u8,
    pub crosstalk_rg_q: u8,
    pub crosstalk_gb_q: u8,
    pub torn_frames_q: u8,
    // физические величины (для диагностики и рекомендаций)
    pub gamma_g: f64,
    pub gamma_r: f64,
    pub gamma_b: f64,
    pub white_pct: f64,
    pub black_pct: f64,
    pub noise_sigma_gray: f64,
    pub crosstalk_rg_pct: f64,
    pub crosstalk_gb_pct: f64,
}

impl Telemetry {
    /// Переносит измеренные q-поля телеметрии в профиль (§7.3, поля 6–15).
    /// Prescriptions (cell/hold/luma/chroma/quiet/fec) не трогает.
    pub fn apply_to_profile(&self, p: &mut CalibProfile) {
        p.gamma_g_q = self.gamma_g_q;
        p.gamma_r_delta_q = self.gamma_r_delta_q;
        p.gamma_b_delta_q = self.gamma_b_delta_q;
        p.white_level_q = self.white_level_q;
        p.black_level_q = self.black_level_q;
        p.noise_sigma_q = self.noise_sigma_q;
        p.mtf_limit_px = self.mtf_limit_px;
        p.crosstalk_rg_q = self.crosstalk_rg_q;
        p.crosstalk_gb_q = self.crosstalk_gb_q;
        p.torn_frames_q = self.torn_frames_q;
    }
}

/// Измеряет канал по снимку калибровочного паттерна. Контракт `map`/`sample`
/// как в [`crate::symbol::demod_symbol`]: `map(u,v)` переводит координаты
/// плоскости паттерна (display-px) в координаты снимка; `sample(x,y)` даёт
/// сенсорно-линейный RGB [0,1]. Геометрия v0 — genie (см. док. модуля).
///
/// Никогда не паникует: вырожденный вход (всё чёрное/серое) даёт зажатую
/// телеметрию с нулями/дефолтами, а не панику.
pub fn measure_channel(
    size_px: usize,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> Telemetry {
    let lay = Layout::new(size_px);

    // --- статистика по ступеням трёх лесенок ---
    let gray = stair_stats(&lay, lay.gray_y, map, sample);
    let rs = stair_stats(&lay, lay.r_y, map, sample);
    let bs = stair_stats(&lay, lay.b_y, map, sample);

    let drives: Vec<f64> = (0..STAIR_STEPS)
        .map(|i| Layout::stair_drive(i) as f64)
        .collect();

    // --- белый уровень и граница клиппинга по серой лесенке (канал G) ---
    let gray_g: Vec<f64> = gray.iter().map(|s| s.mean[1]).collect();
    let (white_pct, last_valid) = detect_white(&drives, &gray_g);
    let n_valid = last_valid + 1;

    // --- кросстолк: наклон утечки в G по неклиппированным ступеням ---
    let x_rg = leak_slope(&rs, 0, 1, n_valid);
    let x_gb = leak_slope(&bs, 2, 1, n_valid);
    let crosstalk_rg_pct = (100.0 * x_rg).max(0.0);
    let crosstalk_gb_pct = (100.0 * x_gb).max(0.0);
    let crosstalk_rg_q = xtalk_to_q(crosstalk_rg_pct);
    let crosstalk_gb_q = xtalk_to_q(crosstalk_gb_pct);

    // --- гамма-фит по неклиппированным ступеням ---
    // R и B чисты (утечка только В G, §4.3), фитятся напрямую.
    let (_, gamma_r) = fit_gamma(&drives[..n_valid], &channel(&rs, 0)[..n_valid], &[]);
    let (_, gamma_b) = fit_gamma(&drives[..n_valid], &channel(&bs, 2)[..n_valid], &[]);
    // G загрязнён утечкой R и B: фитим с поправочным базисом
    // u^γ_G + x_rg·u^γ_R + x_gb·u^γ_B (иначе LSQ смещает γ_G вниз).
    let corr = [(x_rg, gamma_r), (x_gb, gamma_b)];
    let (_, gamma_g) = fit_gamma(&drives[..n_valid], &channel(&gray, 1)[..n_valid], &corr);
    let gamma_g_q = gamma_to_q(gamma_g);
    let gamma_r_delta_q = delta_to_q(gamma_r - gamma_g);
    let gamma_b_delta_q = delta_to_q(gamma_b - gamma_g);

    // --- чёрный подъём: s(drive0) / s(белое поле) ---
    let white_patch = white_patch_mean(&lay, map, sample);
    let s_white = white_patch[1].max(1e-9);
    let black_pct = (100.0 * gray[0].mean[1] / s_white).clamp(0.0, 100.0);
    let white_level_q = white_pct_to_q(white_pct);
    let black_level_q = (black_pct.round() as i32).clamp(0, 15) as u8;

    // --- шум: пул дисперсий канала G по неклиппированным ступеням ---
    let mut var_sum = 0.0;
    for s in &gray[..n_valid] {
        var_sum += s.var[1];
    }
    let sigma_sensor = (var_sum / n_valid as f64).max(0.0).sqrt();
    let noise_sigma_gray = sigma_sensor * 255.0;
    let noise_sigma_q = noise_to_q(noise_sigma_gray);

    // --- MTF: мельчайший шаг клина с контрастом Мишельсона >= порога ---
    let mtf_limit_px = measure_mtf(&lay, map, sample);

    Telemetry {
        gamma_g_q,
        gamma_r_delta_q,
        gamma_b_delta_q,
        white_level_q,
        black_level_q,
        noise_sigma_q,
        mtf_limit_px,
        crosstalk_rg_q,
        crosstalk_gb_q,
        torn_frames_q: 0,
        gamma_g,
        gamma_r,
        gamma_b,
        white_pct,
        black_pct,
        noise_sigma_gray,
        crosstalk_rg_pct,
        crosstalk_gb_pct,
    }
}

/// [EXPERIMENTAL] Референсная эвристика PRESCRIPTION (§7.3, «v1 передатчики
/// MAY пересчитывать»). Возвращает `(luma_bits, chroma_mode, cell_size_px,
/// frame_hold_periods)`. Правило v0, будет настроено sweeps'ами BENCHMARKS §6:
///
/// * `luma_bits`: число уровней при шаге ≥ 4σ на полезном диапазоне
///   белый−чёрный (в градациях серого) -> `floor(log2(levels))`, зажато 1..4.
/// * `chroma_mode`: хрома надёжна лишь при малом кросстолке — `Chroma2`, если
///   luma≥2 и max-кросстолк < 10 %; `GreenOnly` при ≥ 20 %; иначе `Mono`.
/// * `cell_size_px`: ≥ 2·`mtf_limit_px`, зажато 2..65.
/// * `frame_hold_periods`: из tearing (проброс) — `torn_frames_q + 1`, 1..16.
pub fn recommend(t: &Telemetry) -> (u8, ChromaMode, u8, u8) {
    let range_gray = ((t.white_pct - t.black_pct).max(0.0)) / 100.0 * 255.0;
    let sigma = t.noise_sigma_gray.max(0.25);
    let levels = (range_gray / (4.0 * sigma)).max(1.0);
    let luma_bits = (levels.log2().floor() as i32).clamp(1, 4) as u8;

    let max_x = t.crosstalk_rg_pct.max(t.crosstalk_gb_pct);
    let chroma_mode = if max_x >= 20.0 {
        ChromaMode::GreenOnly
    } else if luma_bits >= 2 && max_x < 10.0 {
        ChromaMode::Chroma2
    } else {
        ChromaMode::Mono
    };

    let cell = (2 * t.mtf_limit_px as u32).clamp(2, 65) as u8;
    let hold = ((t.torn_frames_q as i32) + 1).clamp(1, 16) as u8;
    (luma_bits, chroma_mode, cell, hold)
}

// ---------------------------------------------------------------------------
// Измерительные помощники
// ---------------------------------------------------------------------------

/// Среднее и дисперсия по трём каналам в прямоугольнике (сетка nx×ny точек).
#[derive(Clone, Copy)]
struct Stats {
    mean: [f64; 3],
    var: [f64; 3],
}

/// Собирает Stats по регулярной сетке 16×16 в прямоугольнике плоскости паттерна.
fn rect_stats(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> Stats {
    const N: usize = 16;
    let mut sum = [0.0f64; 3];
    let mut sumsq = [0.0f64; 3];
    let mut cnt = 0.0f64;
    for iy in 0..N {
        let v = y0 + (iy as f64 + 0.5) * (y1 - y0) / N as f64;
        for ix in 0..N {
            let u = x0 + (ix as f64 + 0.5) * (x1 - x0) / N as f64;
            let (x, y) = map(u, v);
            let s = sample(x, y);
            for c in 0..3 {
                let val = s[c] as f64;
                sum[c] += val;
                sumsq[c] += val * val;
            }
            cnt += 1.0;
        }
    }
    let mut mean = [0.0f64; 3];
    let mut var = [0.0f64; 3];
    for c in 0..3 {
        mean[c] = sum[c] / cnt;
        var[c] = (sumsq[c] / cnt - mean[c] * mean[c]).max(0.0);
    }
    Stats { mean, var }
}

/// Статистика по 16 ступеням лесенки в полосе `band`.
fn stair_stats(
    lay: &Layout,
    band: (usize, usize),
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> Vec<Stats> {
    let (y0, y1) = lay.band_center_y(band);
    (0..STAIR_STEPS)
        .map(|i| {
            let (x0, x1) = lay.step_center_x(i);
            rect_stats(x0, x1, y0, y1, map, sample)
        })
        .collect()
}

/// Среднее канала белого поля.
fn white_patch_mean(
    lay: &Layout,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> [f64; 3] {
    let (y0, y1) = lay.band_center_y(lay.white_y);
    let x0 = (lay.in_lo + lay.in_w / 4) as f64;
    let x1 = (lay.in_hi - lay.in_w / 4) as f64;
    rect_stats(x0, x1, y0, y1, map, sample).mean
}

/// Извлекает средние одного канала по ступеням лесенки.
fn channel(steps: &[Stats], c: usize) -> Vec<f64> {
    steps.iter().map(|s| s.mean[c]).collect()
}

/// Наклон утечки канала `src` в канал `dst` по ступеням `[0..n)` (регрессия
/// dst на src). Возвращает slope = доля утечки (усиление/смещение сокращаются).
fn leak_slope(steps: &[Stats], src: usize, dst: usize, n: usize) -> f64 {
    let n = n.min(steps.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for s in &steps[..n] {
        let x = s.mean[src];
        let y = s.mean[dst];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        0.0
    } else {
        (nf * sxy - sx * sy) / denom
    }
}

/// Детекция белого уровня по серой лесенке (канал G). Возвращает
/// `(white_pct, last_valid_step)`: процент full-drive для белого и индекс
/// последней НЕклиппированной ступени (для гамма-фита/шума/кросстолка).
fn detect_white(drives: &[f64], means: &[f64]) -> (f64, usize) {
    let n = means.len();
    if n < 2 {
        return (0.0, 0);
    }
    let mn = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = mx - mn;
    if span <= 1e-9 {
        return (0.0, 0); // вырожденно (всё одинаково): белого не видно
    }
    let eps = 0.02 * span;
    // начало верхнего плато: наименьший i, от которого все ступени в пределах eps от max
    let mut fp = n - 1;
    for i in (0..n).rev() {
        if means[i] >= mx - eps {
            fp = i;
        } else {
            break;
        }
    }
    if fp >= n - 1 {
        // строго растёт до верха — клиппинга нет
        (100.0, n - 1)
    } else {
        // клип: белый уровень — между последней растущей и первой плоской ступенью
        let wd = 0.5 * (drives[fp - 1] + drives[fp]);
        (100.0 * wd / 255.0, fp - 1)
    }
}

/// LSQ-фит s = a·[(d/255)^γ + Σ coef·(d/255)^γ_c] + b: скан γ по сетке §7.3
/// (1.5+0.025·q, 64 кандидата), a,b в закрытой форме на кандидата, выбор по
/// min SSE. `corr` — поправочные слагаемые кросстолка (coef, γ_c); для чистых
/// каналов пусто. Возвращает `(q, γ)`.
fn fit_gamma(drives: &[f64], s: &[f64], corr: &[(f64, f64)]) -> (u8, f64) {
    let n = drives.len();
    let mut best_q = 0u8;
    let mut best_sse = f64::INFINITY;
    let mut best_g = 1.5;
    if n == 0 {
        return (0, 1.5);
    }
    let basis = |u: f64, g: f64| -> f64 {
        let mut x = u.powf(g);
        for &(coef, gc) in corr {
            x += coef * u.powf(gc);
        }
        x
    };
    for q in 0..=63u8 {
        let g = 1.5 + 0.025 * q as f64;
        let nf = n as f64;
        let mut sx = 0.0;
        let mut sxx = 0.0;
        let mut ss = 0.0;
        let mut sxs = 0.0;
        for i in 0..n {
            let x = basis(drives[i] / 255.0, g);
            sx += x;
            sxx += x * x;
            ss += s[i];
            sxs += x * s[i];
        }
        let denom = nf * sxx - sx * sx;
        let (a, b) = if denom.abs() < 1e-15 {
            (0.0, ss / nf)
        } else {
            let a = (nf * sxs - sx * ss) / denom;
            (a, (ss - a * sx) / nf)
        };
        let mut sse = 0.0;
        for i in 0..n {
            let x = basis(drives[i] / 255.0, g);
            let e = s[i] - (a * x + b);
            sse += e * e;
        }
        if sse < best_sse {
            best_sse = sse;
            best_q = q;
            best_g = g;
        }
    }
    (best_q, best_g)
}

/// MTF: мельчайший шаг клина с контрастом Мишельсона ≥ порога (§4.2).
/// Кодируется как `mtf_limit_px` (1..32); шаги > 32 зажимаются к 32.
fn measure_mtf(
    lay: &Layout,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> u8 {
    let mut contrast = [0.0f64; WEDGE_PITCHES.len()];
    for (idx, &pitch) in WEDGE_PITCHES.iter().enumerate() {
        let cy = (lay.wedge_y0 + idx * lay.wedge_band_h + lay.wedge_band_h / 2) as f64;
        let half = pitch / 2;
        let black_rem = half / 2;
        let white_rem = half + (pitch - half) / 2;
        let n_per = lay.in_w / pitch;
        let mut sum_w = 0.0;
        let mut sum_b = 0.0;
        let mut cnt = 0.0;
        for k in 0..n_per {
            let x0 = lay.in_lo + k * pitch;
            let (xw, yw) = map((x0 + white_rem) as f64, cy);
            let (xb, yb) = map((x0 + black_rem) as f64, cy);
            sum_w += sample(xw, yw)[1] as f64;
            sum_b += sample(xb, yb)[1] as f64;
            cnt += 1.0;
        }
        if cnt > 0.0 {
            let imax = sum_w / cnt;
            let imin = sum_b / cnt;
            let denom = imax + imin;
            contrast[idx] = if denom.abs() < 1e-12 {
                0.0
            } else {
                (imax - imin) / denom
            };
        }
    }
    // мельчайший разрешимый шаг = наименьший pitch с контрастом ≥ порога
    let mut best = WEDGE_PITCHES[0];
    for idx in (0..WEDGE_PITCHES.len()).rev() {
        if contrast[idx] >= MTF_CONTRAST_THRESHOLD {
            best = WEDGE_PITCHES[idx];
            break;
        }
    }
    best.clamp(1, 32) as u8
}

// ---------------------------------------------------------------------------
// Квантование физических величин в q-поля §7.3
// ---------------------------------------------------------------------------

/// γ = 1.5 + 0.025·q -> q, зажато 0..63.
fn gamma_to_q(g: f64) -> u8 {
    (((g - 1.5) / 0.025).round() as i32).clamp(0, 63) as u8
}
/// Δγ = 0.025·(q−8) -> q, зажато 0..15 (центр 8).
fn delta_to_q(d: f64) -> u8 {
    ((d / 0.025).round() as i32 + 8).clamp(0, 15) as u8
}
/// white = (55 + 3q) % -> q, зажато 0..15.
fn white_pct_to_q(pct: f64) -> u8 {
    (((pct - 55.0) / 3.0).round() as i32).clamp(0, 15) as u8
}
/// σ = 0.25·2^(q/4) градаций -> q, зажато 0..31.
fn noise_to_q(sigma_gray: f64) -> u8 {
    if sigma_gray <= 0.0 {
        return 0;
    }
    ((4.0 * (sigma_gray / 0.25).log2()).round() as i32).clamp(0, 31) as u8
}
/// crosstalk = 2q % -> q, зажато 0..15.
fn xtalk_to_q(pct: f64) -> u8 {
    ((pct / 2.0).round() as i32).clamp(0, 15) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::CalibProfile;

    // -----------------------------------------------------------------------
    // Тестовый канал (НЕ зависит от psicode-sim): gamma -> blur -> crosstalk
    // -> gain/offset(+white clip) -> gaussian noise. Детерминированный.
    // -----------------------------------------------------------------------

    struct XorShift64(u64);
    impl XorShift64 {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        /// равномерно (0,1]
        fn next_f64(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        }
        /// N(0,1) через Бокса–Мюллера
        fn next_gauss(&mut self) -> f64 {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
        }
    }

    struct Channel {
        gamma: [f64; 3],      // R, G, B
        blur_sigma: f64,      // display-px, 0 = без блюра
        x_rg: f64,            // утечка R в G
        x_gb: f64,            // утечка B в G
        gain: f64,
        offset: f64,
        white_clip_drive: f64, // drive, выше которого выход насыщается (255 = нет)
        noise_gray: f64,       // σ шума в градациях серого
        seed: u64,
    }

    /// Прогоняет drive-паттерн через канал, возвращает сенсорный буфер [0,1].
    /// Порядок §4: клип drive по white_level (дисплей не тянет выше белого)
    /// -> гамма -> блюр -> кросстолк -> gain/offset -> гаусс-шум.
    fn apply_channel(pattern: &[[u8; 3]], size: usize, ch: &Channel) -> Vec<[f32; 3]> {
        // 1. клип drive по белому уровню (drive-домен) + линеаризация (гамма)
        let mut lin = vec![[0.0f64; 3]; size * size];
        for (i, px) in pattern.iter().enumerate() {
            for c in 0..3 {
                let d = (px[c] as f64).min(ch.white_clip_drive);
                lin[i][c] = (d / 255.0).powf(ch.gamma[c]);
            }
        }
        // 2. блюр (сепарабельный гаусс) на линейном свете
        if ch.blur_sigma > 0.0 {
            lin = gaussian_blur(&lin, size, ch.blur_sigma);
        }
        // 3. кросстолк: R и B утекают в G; 4. gain/offset; 5. шум
        let mut out = vec![[0.0f32; 3]; size * size];
        let mut rng = XorShift64(ch.seed | 1);
        let noise = ch.noise_gray / 255.0;
        for i in 0..size * size {
            let lr = lin[i][0];
            let lg = lin[i][1] + ch.x_rg * lin[i][0] + ch.x_gb * lin[i][2];
            let lb = lin[i][2];
            let l = [lr, lg, lb];
            for c in 0..3 {
                let mut v = ch.gain * l[c] + ch.offset;
                v += noise * rng.next_gauss();
                out[i][c] = v as f32;
            }
        }
        out
    }

    /// Сепарабельный гаусс с зажатием краёв.
    fn gaussian_blur(src: &[[f64; 3]], size: usize, sigma: f64) -> Vec<[f64; 3]> {
        let radius = (3.0 * sigma).ceil() as isize;
        let mut kernel = vec![0.0f64; (2 * radius + 1) as usize];
        let mut ksum = 0.0;
        for (j, k) in kernel.iter_mut().enumerate() {
            let d = j as isize - radius;
            *k = (-(d * d) as f64 / (2.0 * sigma * sigma)).exp();
            ksum += *k;
        }
        for k in kernel.iter_mut() {
            *k /= ksum;
        }
        let clamp = |v: isize| v.clamp(0, size as isize - 1) as usize;
        // горизонтальный проход
        let mut tmp = vec![[0.0f64; 3]; size * size];
        for y in 0..size {
            for x in 0..size {
                let mut acc = [0.0f64; 3];
                for (j, &k) in kernel.iter().enumerate() {
                    let xx = clamp(x as isize + j as isize - radius);
                    for c in 0..3 {
                        acc[c] += k * src[y * size + xx][c];
                    }
                }
                tmp[y * size + x] = acc;
            }
        }
        // вертикальный проход
        let mut dst = vec![[0.0f64; 3]; size * size];
        for y in 0..size {
            for x in 0..size {
                let mut acc = [0.0f64; 3];
                for (j, &k) in kernel.iter().enumerate() {
                    let yy = clamp(y as isize + j as isize - radius);
                    for c in 0..3 {
                        acc[c] += k * tmp[yy * size + x][c];
                    }
                }
                dst[y * size + x] = acc;
            }
        }
        dst
    }

    /// Замыкание sample поверх сенсорного буфера (ближайший пиксель, map=id).
    fn buf_sampler(buf: Vec<[f32; 3]>, size: usize) -> impl Fn(f64, f64) -> [f32; 3] {
        move |x: f64, y: f64| {
            let xi = (x.round() as isize).clamp(0, size as isize - 1) as usize;
            let yi = (y.round() as isize).clamp(0, size as isize - 1) as usize;
            buf[yi * size + xi]
        }
    }

    const SIZE: usize = 732; // номинал cell=12

    /// «Истинные» q-поля из физических параметров канала (для сверки ±1).
    struct Truth {
        gamma_g_q: u8,
        gamma_r_delta_q: u8,
        gamma_b_delta_q: u8,
        white_level_q: u8,
        black_level_q: u8,
        noise_sigma_q: u8,
        crosstalk_rg_q: u8,
        crosstalk_gb_q: u8,
    }

    fn truth_of(ch: &Channel) -> Truth {
        let gg = ch.gamma[1];
        let white_pct = 100.0 * ch.white_clip_drive / 255.0;
        // белое поле: full-drive зажат до white_clip_drive, с учётом кросстолка в G
        let cl = |g: f64| (ch.white_clip_drive / 255.0).powf(g);
        let lg = cl(gg) + ch.x_rg * cl(ch.gamma[0]) + ch.x_gb * cl(ch.gamma[2]);
        let s_white = ch.gain * lg + ch.offset;
        let black_pct = 100.0 * ch.offset / s_white;
        Truth {
            gamma_g_q: gamma_to_q(gg),
            gamma_r_delta_q: delta_to_q(ch.gamma[0] - gg),
            gamma_b_delta_q: delta_to_q(ch.gamma[2] - gg),
            white_level_q: white_pct_to_q(white_pct),
            black_level_q: (black_pct.round() as i32).clamp(0, 15) as u8,
            noise_sigma_q: noise_to_q(ch.noise_gray),
            crosstalk_rg_q: xtalk_to_q(100.0 * ch.x_rg),
            crosstalk_gb_q: xtalk_to_q(100.0 * ch.x_gb),
        }
    }

    fn near(got: u8, want: u8, field: &str, tag: &str) {
        let d = (got as i32 - want as i32).abs();
        assert!(d <= 1, "{tag}: {field} got {got} want {want} (Δ={d} > 1)");
    }

    /// Измеряет канал и сверяет все q-поля телеметрии с истиной (±1 шаг).
    /// MTF сверяется с чистым (без шума) измерением того же канала.
    fn check_config(ch: &Channel, tag: &str) -> Telemetry {
        let pattern = render_calibration_pattern(SIZE);
        let buf = apply_channel(&pattern, SIZE, ch);
        let sampler = buf_sampler(buf, SIZE);
        let map = |u: f64, v: f64| (u, v);
        let tele = measure_channel(SIZE, &map, &sampler);
        let t = truth_of(ch);

        near(tele.gamma_g_q, t.gamma_g_q, "gamma_g_q", tag);
        near(tele.gamma_r_delta_q, t.gamma_r_delta_q, "gamma_r_delta_q", tag);
        near(tele.gamma_b_delta_q, t.gamma_b_delta_q, "gamma_b_delta_q", tag);
        near(tele.white_level_q, t.white_level_q, "white_level_q", tag);
        near(tele.black_level_q, t.black_level_q, "black_level_q", tag);
        near(tele.noise_sigma_q, t.noise_sigma_q, "noise_sigma_q", tag);
        near(tele.crosstalk_rg_q, t.crosstalk_rg_q, "crosstalk_rg_q", tag);
        near(tele.crosstalk_gb_q, t.crosstalk_gb_q, "crosstalk_gb_q", tag);

        // MTF: чистый (noise=0) прогон того же канала — эталон стабильности
        let clean = Channel {
            noise_gray: 0.0,
            seed: ch.seed ^ 0xABCD,
            ..*ch
        };
        let pattern2 = render_calibration_pattern(SIZE);
        let buf2 = apply_channel(&pattern2, SIZE, &clean);
        let sampler2 = buf_sampler(buf2, SIZE);
        let clean_tele = measure_channel(SIZE, &map, &sampler2);
        assert_eq!(
            tele.mtf_limit_px, clean_tele.mtf_limit_px,
            "{tag}: mtf_limit_px нестабилен к шуму: {} vs чистый {}",
            tele.mtf_limit_px, clean_tele.mtf_limit_px
        );
        tele
    }

    impl Clone for Channel {
        fn clone(&self) -> Self {
            Channel { ..*self }
        }
    }
    impl Copy for Channel {}

    fn mild() -> Channel {
        Channel {
            gamma: [2.2, 2.4, 2.45],
            blur_sigma: 0.0,
            x_rg: 0.0,
            x_gb: 0.0,
            gain: 0.92,
            offset: 0.02,
            white_clip_drive: 255.0,
            noise_gray: 1.0,
            seed: 0x1111_2222_3333_4444,
        }
    }
    fn medium() -> Channel {
        Channel {
            gamma: [2.2, 2.4, 2.45],
            blur_sigma: 1.0,
            x_rg: 0.06,
            x_gb: 0.08,
            gain: 0.85, // headroom под кросстолк-буст (max ≈ 0.85·1.14+0.02 ≈ 0.99)
            offset: 0.02,
            white_clip_drive: 255.0,
            noise_gray: 1.0,
            seed: 0x5555_6666_7777_8888,
        }
    }
    fn harsher() -> Channel {
        Channel {
            gamma: [2.2, 2.4, 2.45],
            blur_sigma: 2.0,
            x_rg: 0.0,
            x_gb: 0.0,
            gain: 0.88,
            offset: 0.015,
            white_clip_drive: 232.0, // ≈91 %
            noise_gray: 4.0,
            seed: 0x9999_AAAA_BBBB_CCCC,
        }
    }

    #[test]
    fn estimators_within_one_step_mild() {
        // достигнутое: ВСЕ q-поля Δ=0 к истине; клин без блюра разрешает 2 px
        let t = check_config(&mild(), "mild");
        assert_eq!(t.mtf_limit_px, 2, "mild mtf");
    }

    #[test]
    fn estimators_within_one_step_medium() {
        // достигнутое: ВСЕ q-поля Δ=0 (в т.ч. gg=36 при кросстолк-поправке);
        // блюр σ=1 px -> мельчайший разрешимый шаг клина 6 px
        let t = check_config(&medium(), "medium");
        assert_eq!(t.mtf_limit_px, 6, "medium mtf");
    }

    #[test]
    fn estimators_within_one_step_harsher() {
        // достигнутое: ВСЕ q-поля Δ=0 (white_q=12 при клипе 91 %, noise_q=16 при
        // σ=4); блюр σ=2 px -> мельчайший разрешимый шаг клина 12 px
        let t = check_config(&harsher(), "harsher");
        assert_eq!(t.mtf_limit_px, 12, "harsher mtf");
    }

    /// Полный цикл: телеметрия -> CalibProfile (референсные prescriptions)
    /// -> encode_string -> decode_string -> точный roundtrip.
    #[test]
    fn full_loop_telemetry_to_profile_roundtrip() {
        for (ch, torn) in [(mild(), 0u8), (medium(), 5), (harsher(), 3)] {
            let pattern = render_calibration_pattern(SIZE);
            let buf = apply_channel(&pattern, SIZE, &ch);
            let sampler = buf_sampler(buf, SIZE);
            let map = |u: f64, v: f64| (u, v);
            let mut tele = measure_channel(SIZE, &map, &sampler);
            tele.torn_frames_q = torn; // проброс из §4.5 (измеряется отдельно)

            let (luma_bits, chroma_mode, cell, hold) = recommend(&tele);
            let mut p = CalibProfile {
                version: CalibProfile::VERSION,
                cell_size_px: cell,
                frame_hold_periods: hold,
                luma_bits,
                chroma_mode,
                gamma_g_q: 0,
                gamma_r_delta_q: 0,
                gamma_b_delta_q: 0,
                white_level_q: 0,
                black_level_q: 0,
                noise_sigma_q: 0,
                mtf_limit_px: 1,
                torn_frames_q: 0,
                crosstalk_rg_q: 0,
                crosstalk_gb_q: 0,
                quiet_zone: 1,
                fec_overhead: 2,
                border: crate::profile::BorderMode::LegacyInverted,
            };
            tele.apply_to_profile(&mut p);

            let s = p.encode_string().expect("профиль должен кодироваться");
            let (q, fixed) = CalibProfile::decode_string(&s).expect("декод");
            assert_eq!(fixed, 0);
            assert_eq!(p, q, "roundtrip профиля разошёлся");
        }
    }

    /// Геометрия рендера: размер, тихая-зона отсутствует (паттерн — не символ),
    /// угол — ЗЧ, белое поле занимает ≥ 10 % площади.
    #[test]
    fn pattern_geometry_and_white_area() {
        let pat = render_calibration_pattern(SIZE);
        assert_eq!(pat.len(), SIZE * SIZE);
        let white_px = pat.iter().filter(|&&c| c == WHITE).count();
        // белое поле одно даёт ~18 %; штрихи клина добавляют ещё — точно ≥ 10 %
        assert!(
            white_px as f64 / (SIZE * SIZE) as f64 >= 0.10,
            "белого < 10 %: {white_px}"
        );
        // хвостовой паддинг: 732 = 61·12 ровно, значит серого-паддинга нет,
        // но угол полотна — клетка (0,0) ЗЧ-кольца, чёрная или белая
        assert!(pat[0] == WHITE || pat[0] == BLACK);
    }

    /// Вырожденный вход не паникует и даёт зажатую телеметрию.
    #[test]
    fn degenerate_inputs_do_not_panic() {
        let map = |u: f64, v: f64| (u, v);
        // всё чёрное
        let black = |_x: f64, _y: f64| [0.0f32; 3];
        let t = measure_channel(SIZE, &map, &black);
        assert!(t.white_level_q <= 15 && t.black_level_q <= 15);
        assert!(t.mtf_limit_px >= 1 && t.mtf_limit_px <= 32);
        // всё средне-серое
        let gray = |_x: f64, _y: f64| [0.5f32; 3];
        let t2 = measure_channel(SIZE, &map, &gray);
        assert!(t2.gamma_g_q <= 63 && t2.noise_sigma_q <= 31);
        // всё белое (клиппинг всюду)
        let white = |_x: f64, _y: f64| [1.0f32; 3];
        let t3 = measure_channel(SIZE, &map, &white);
        assert!(t3.crosstalk_rg_q <= 15 && t3.crosstalk_gb_q <= 15);
    }
}
