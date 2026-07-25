//! Сборка и демодуляция кадра v0: §3 геометрия (ЗЧ-рамка, референсная строка),
//! §5.1 комплексно-цветовое отображение, §5.2 Mode A (сетка клеток).
//!
//! Модуль доступен только с фичей `std` (нужна вещественная математика с
//! transcendental-функциями для гамма-коррекции).

use crate::profile::{CalibProfile, ChromaMode};
use alloc::vec;
use alloc::vec::Vec;

/// Толщина ЗЧ-кольца в клетках (§3.2: внутреннее кольцо — инверсия внешнего).
pub const RING: usize = 2;
/// Клеток на сторону символа v0 = длина ЗЧ-последовательности N.
pub const GRID: usize = 61;
/// Внутренняя область после двойного кольца.
pub const INTERIOR: usize = GRID - 2 * RING; // 57
/// Ширина payload-сетки в клетках.
pub const PAYLOAD_COLS: usize = INTERIOR; // 57
/// Высота payload-сетки: внутренняя область минус референсная строка (§3.4)
/// и строка счётчика кадров (§3.3; v0 — неактивная, средне-серая).
pub const PAYLOAD_ROWS: usize = INTERIOR - 2; // 55
/// Корни ЗЧ по сторонам: верх, право, низ, лево (§3.2).
pub const ZC_ROOTS: [u32; 4] = [1, 2, 3, 4];

/// Модуль периода бинаризации ЗЧ: exp(−jπ·m/61) имеет период 122 по m (§3.2).
const ZC_MOD: u64 = 2 * GRID as u64; // 122
/// Средняя точка динамического диапазона (§5.1).
const MID: f64 = 128.0;
/// Период референсного паттерна §3.4 в клетках: `K W R G B C M Y K W` + 6 серых.
const REF_PERIOD: usize = 16;
/// Число ступеней серой лесенки в референсном паттерне (§3.4).
const REF_GRAY_STEPS: usize = 6;
/// Бит счётчика кадров в строке счётчика (§3.3/§6.3): младшие 8 бит номера
/// кадра, продублированные в начале и в конце строки счётчика.
pub const COUNTER_BITS: usize = 8;

/// [EXPERIMENTAL] Доля `usable` под яркость в хромо-режимах v0 (§5.1).
/// Заморозится после канальных измерений; см. RESEARCH/BENCHMARKS.
pub const A_L_FRACTION_CHROMA: f64 = 0.7;
/// [EXPERIMENTAL] Доля `usable` под хрому в хромо-режимах v0 (§5.1).
pub const A_C_FRACTION_CHROMA: f64 = 0.3;

/// Битов на клетку payload: luma_bits + биты хромы режима (§5.2).
pub fn bits_per_cell(p: &CalibProfile) -> u32 {
    p.luma_bits as u32 + p.chroma_bits() as u32
}

/// true = белая клетка бинаризованной ЗЧ-последовательности корня `root`
/// в позиции n (§3.2: белая, если arg(z[n]) > 0). Целочисленная арифметика.
///
/// Вывод: z[n] = exp(−jπ·q·n(n+1)/61), значит arg(z[n]) = −π·k/61 (mod 2π),
/// где k = (q·n(n+1)) mod 122. Приведя к (−π, π]: arg > 0 ⇔ k ≥ 61
/// (граница arg=π ⇒ белая, arg=0 ⇒ чёрная). k всегда чётно, поэтому точная
/// граница k=61 недостижима, а k=0 (n=0) даёт arg=0 ⇒ чёрная.
pub fn zc_binary(root: u32, n: usize) -> bool {
    let m = root as u64 * n as u64 * (n as u64 + 1);
    (m % ZC_MOD) >= GRID as u64
}

/// Отрендеренный кадр: RGB построчно, size_px × size_px, тихая зона включена.
pub struct Frame {
    /// Сторона кадра в display-пикселях: (GRID + 2·quiet_cells) · cell_size_px.
    pub size_px: usize,
    /// Ширина тихой зоны в клетках (из профиля, §3.1).
    pub quiet_cells: usize,
    /// Пиксели построчно, [R, G, B] в drive-значениях 0..255.
    pub rgb: Vec<[u8; 3]>,
}

/// Собирает полный символ v0: тихая зона, двойное ЗЧ-кольцо, референсная
/// строка, Mode A payload-сетка, строка счётчика (v0: средне-серая, counter=0).
/// `cells` — ровно PAYLOAD_COLS·PAYLOAD_ROWS символов клеток в растровом
/// порядке (в каждом байте значимы младшие bits_per_cell бит, Грей-код).
///
/// Обёртка над [`render_symbol_counter`] с `counter = 0` — публичный API v0
/// не меняется; счётчик кадров задаётся явным вызовом render_symbol_counter.
pub fn render_symbol(p: &CalibProfile, cells: &[u8]) -> Frame {
    render_symbol_counter(p, cells, 0)
}

/// Как [`render_symbol`], но строка счётчика кадров (§3.3/§6.3) несёт младшие
/// 8 бит `counter`: 8 чёрно-белых клеток в НАЧАЛЕ строки счётчика И их дубль
/// в КОНЦЕ (white = бит 1), середина строки остаётся средне-серой. Порядок бит
/// — старшим вперёд (клетка в позиции 0 = бит 7, позиция 7 = бит 0). Дубль
/// нужен рваному снимку (§6.3): tear выше строки счётчика оставляет копию
/// кадра N+1, ниже — копию кадра N; две копии считываются приёмником отдельно.
pub fn render_symbol_counter(p: &CalibProfile, cells: &[u8], counter: u8) -> Frame {
    assert_eq!(
        cells.len(),
        PAYLOAD_COLS * PAYLOAD_ROWS,
        "cells: ожидалось {}·{} символов",
        PAYLOAD_COLS,
        PAYLOAD_ROWS
    );

    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    let total_cells = GRID + 2 * quiet;
    let size_px = total_cells * cell;

    // 1. клеточная решётка символа GRID×GRID в drive-RGB.
    let sym = build_symbol_cells(p, cells, counter);

    // 2. композиция полотна: тихая зона (средне-серая), затем «раздутие»
    //    каждой клетки символа в сплошной квадрат cell×cell (§5.2).
    let gray = [MID as u8; 3];
    let mut rgb = vec![gray; size_px * size_px];
    for cy in 0..GRID {
        for cx in 0..GRID {
            let color = sym[cy * GRID + cx];
            let px0 = (quiet + cx) * cell;
            let py0 = (quiet + cy) * cell;
            for dy in 0..cell {
                let row = (py0 + dy) * size_px + px0;
                for dx in 0..cell {
                    rgb[row + dx] = color;
                }
            }
        }
    }

    Frame {
        size_px,
        quiet_cells: quiet,
        rgb,
    }
}

/// Демодуляция Mode A по снимку. `map(u, v)` переводит координаты плоскости
/// символа (display-px, той же системы, что Frame, тихая зона включена) в
/// координаты снимка; `sample(x, y)` возвращает линейный (сенсорный) RGB
/// [0,1] снимка в этой точке. Возвращает символы клеток в растровом порядке
/// (та же система, что вход render_symbol).
///
/// v0: цветокоррекция ограничена нормировкой по референсной строке (§3.4) —
/// на канал восстанавливаются gain/offset и снимается гамма. Полноценная
/// матрица 3×3 (§3.4) в v0 НЕ применяется; появится после канальных измерений.
pub fn demod_symbol(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> Vec<u8> {
    let (black_255, white_255) = levels(p);
    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    let (a_l, a_c) = amplitudes(p, black_255, white_255);
    // GreenOnly и Mono несут chroma_bits=0, поэтому Im-ветка ниже отключается
    // сама собой; отдельного флага режима демодулятору не требуется.
    let luma_bits = p.luma_bits as u32;
    let chroma_bits = p.chroma_bits() as u32;
    let l_levels = 1u32 << luma_bits;
    let c_levels = 1u32 << chroma_bits;
    let gammas = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];

    // --- нормировка по референсной строке (§3.4) ---
    // Модель сенсора на канал: s = a·(d/255)^γ + b. Референсная строка даёт
    // два якоря (все K-клетки и все W-клетки), из них решаем a и b, затем
    // инвертируем: d = 255·((s − b)/a)^(1/γ). Это снимает произвольные
    // per-channel gain/offset датчика (белый баланс, чёрный уровень).
    let black = [black_255; 3];
    let white = [white_255; 3];
    let mut s_k = [0.0f64; 3];
    let mut s_w = [0.0f64; 3];
    let mut nk = 0usize;
    let mut nw = 0usize;
    for ic in 0..INTERIOR {
        let pat = ref_pattern(ic % REF_PERIOD, black_255, white_255);
        let cx = RING + ic;
        let cy = RING; // референсная строка — первая строка внутренней области
        if pat == black {
            let s = sample_cell(quiet, cell, cx, cy, map, sample);
            for c in 0..3 {
                s_k[c] += s[c];
            }
            nk += 1;
        } else if pat == white {
            let s = sample_cell(quiet, cell, cx, cy, map, sample);
            for c in 0..3 {
                s_w[c] += s[c];
            }
            nw += 1;
        }
    }
    // nk и nw заведомо > 0 (K и W присутствуют в каждом периоде паттерна);
    // защищаемся на случай вырожденной геометрии.
    let nk = nk.max(1) as f64;
    let nw = nw.max(1) as f64;
    let mut a_gain = [1.0f64; 3];
    let mut b_off = [0.0f64; 3];
    for c in 0..3 {
        s_k[c] /= nk;
        s_w[c] /= nw;
        let dkc = (black_255 as f64 / 255.0).powf(gammas[c]);
        let dwc = (white_255 as f64 / 255.0).powf(gammas[c]);
        let denom = dwc - dkc;
        let a = if denom.abs() < 1e-12 {
            1.0
        } else {
            (s_w[c] - s_k[c]) / denom
        };
        a_gain[c] = if a.abs() < 1e-12 { 1e-12 } else { a };
        b_off[c] = s_k[c] - a_gain[c] * dkc;
    }

    // --- демодуляция payload-клеток ---
    let mut out = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
    for pr in 0..PAYLOAD_ROWS {
        let cy = RING + 1 + pr; // под референсной строкой
        for pc in 0..PAYLOAD_COLS {
            let cx = RING + pc;
            let s = sample_cell(quiet, cell, cx, cy, map, sample);
            // снимаем сенсорную модель на канал -> drive 0..255
            let mut d = [0.0f64; 3];
            for c in 0..3 {
                let base = ((s[c] - b_off[c]) / a_gain[c]).max(0.0);
                d[c] = (255.0 * base.powf(1.0 / gammas[c])).clamp(0.0, 255.0);
            }
            // §5.1 обратное отображение: Re из G, Im из (R − B)/2.
            let re_hat = (d[1] - MID) / a_l;
            let b_l = nearest_level(re_hat, l_levels);
            let luma_gray = binary_to_gray(b_l);
            let sym = if chroma_bits > 0 {
                let im_hat = (d[0] - d[2]) / (2.0 * a_c);
                let b_c = nearest_level(im_hat, c_levels);
                let chroma_gray = binary_to_gray(b_c);
                (luma_gray << chroma_bits) | chroma_gray
            } else {
                luma_gray
            };
            out[pr * PAYLOAD_COLS + pc] = sym as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Внутренние помощники
// ---------------------------------------------------------------------------

/// Drive-значения чёрного и белого уровней (§5.1): (black_255, white_255).
fn levels(p: &CalibProfile) -> (u8, u8) {
    let white_255 = (255.0 * p.white_level_pct() as f64 / 100.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    let black_255 = (255.0 * p.black_level_pct() as f64 / 100.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    (black_255, white_255)
}

/// Амплитуды (A_L, A_C) для §5.1 (v0, экспериментальные доли usable).
fn amplitudes(p: &CalibProfile, black_255: u8, white_255: u8) -> (f64, f64) {
    let usable = ((white_255 as f64 - MID).min(MID - black_255 as f64)).max(0.0);
    match p.chroma_mode {
        ChromaMode::Mono | ChromaMode::GreenOnly => (usable, 0.0),
        ChromaMode::Chroma1 | ChromaMode::Chroma2 | ChromaMode::Chroma3 => {
            (A_L_FRACTION_CHROMA * usable, A_C_FRACTION_CHROMA * usable)
        }
    }
}

/// Клетка референсного паттерна §3.4 по индексу внутри периода (0..16).
/// `K W R G B C M Y K W` затем 6-ступенчатая серая лесенка от чёрного к белому.
fn ref_pattern(idx: usize, k: u8, w: u8) -> [u8; 3] {
    match idx {
        0 => [k, k, k], // K
        1 => [w, w, w], // W
        2 => [w, k, k], // R
        3 => [k, w, k], // G
        4 => [k, k, w], // B
        5 => [k, w, w], // C
        6 => [w, k, w], // M
        7 => [w, w, k], // Y
        8 => [k, k, k], // K
        9 => [w, w, w], // W
        _ => {
            // серая лесенка: ступени 0..=5, линейный drive от k до w включительно
            let step = (idx - 10) as f64; // 0..5
            let d = (k as f64 + (w as f64 - k as f64) * step / (REF_GRAY_STEPS - 1) as f64)
                .round()
                .clamp(0.0, 255.0) as u8;
            [d, d, d]
        }
    }
}

/// §5.1 прямое отображение поля (Re, Im) в drive-RGB [R, G, B].
fn encode_field(re: f64, im: f64, a_l: f64, a_c: f64, greenonly: bool) -> [u8; 3] {
    let g = MID + a_l * re;
    let (r, b) = if greenonly {
        // GreenOnly: R = B = M всегда (§5.1), яркость несёт только G.
        (MID, MID)
    } else {
        (MID + a_l * re + a_c * im, MID + a_l * re - a_c * im)
    };
    [quant(r), quant(g), quant(b)]
}

/// Символ клетки Mode A -> drive-RGB. luma и chroma декодируются из Грей-кода
/// раздельно (§5.2): symbol = (luma_gray << chroma_bits) | chroma_gray.
fn encode_cell(
    s: u32,
    chroma_bits: u32,
    l_levels: u32,
    c_levels: u32,
    a_l: f64,
    a_c: f64,
    greenonly: bool,
) -> [u8; 3] {
    let luma_gray = s >> chroma_bits;
    let chroma_mask = (1u32 << chroma_bits) - 1; // при chroma_bits=0 даёт 0
    let chroma_gray = s & chroma_mask;
    let b_l = gray_to_binary(luma_gray);
    let re = -1.0 + 2.0 * b_l as f64 / (l_levels - 1) as f64;
    let im = if chroma_bits > 0 {
        let b_c = gray_to_binary(chroma_gray);
        -1.0 + 2.0 * b_c as f64 / (c_levels - 1) as f64
    } else {
        0.0
    };
    encode_field(re, im, a_l, a_c, greenonly)
}

/// Сборка клеточной решётки символа GRID×GRID в drive-RGB. `counter` — младшие
/// 8 бит номера кадра для строки счётчика (§3.3/§6.3).
fn build_symbol_cells(p: &CalibProfile, cells: &[u8], counter: u8) -> Vec<[u8; 3]> {
    let (black_255, white_255) = levels(p);
    let white = [white_255; 3];
    let black = [black_255; 3];
    let mut sym = vec![[0u8; 3]; GRID * GRID];

    // --- внешнее ЗЧ-кольцо (§3.2) ---
    // Булева карта белизны кольца. Порядок покраски: лево, низ, право, верх —
    // приоритет углов «верх > право > низ > лево» получается автоматически,
    // так как верх красится последним (§3.2, задача).
    let last = GRID - 1;
    let mut owhite = vec![false; GRID * GRID];
    for n in 0..GRID {
        owhite[(last - n) * GRID] = zc_binary(4, n); // лево: (0, 60−n), корень 4
    }
    for n in 0..GRID {
        owhite[last * GRID + (last - n)] = zc_binary(3, n); // низ: (60−n, 60), корень 3
    }
    for n in 0..GRID {
        owhite[n * GRID + last] = zc_binary(2, n); // право: (60, n), корень 2
    }
    for n in 0..GRID {
        owhite[n] = zc_binary(1, n); // верх: (n, 0), корень 1
    }
    // покраска внешнего кольца из карты
    for x in 0..GRID {
        sym[x] = pick(owhite[x], white, black);
        sym[last * GRID + x] = pick(owhite[last * GRID + x], white, black);
    }
    for y in 0..GRID {
        sym[y * GRID] = pick(owhite[y * GRID], white, black);
        sym[y * GRID + last] = pick(owhite[y * GRID + last], white, black);
    }

    // --- внутреннее кольцо: инверсия примыкающей внешней клетки (§3.2) ---
    // Тот же приоритет углов: лево, низ, право, верх (верх красится последним).
    for y in 1..last {
        sym[y * GRID + 1] = pick(!owhite[y * GRID], white, black); // лево, примыкает (0, y)
    }
    for x in 1..last {
        sym[(last - 1) * GRID + x] = pick(!owhite[last * GRID + x], white, black); // низ, (x, 60)
    }
    for y in 1..last {
        sym[y * GRID + (last - 1)] = pick(!owhite[y * GRID + last], white, black); // право, (60, y)
    }
    for x in 1..last {
        sym[GRID + x] = pick(!owhite[x], white, black); // верх, примыкает (x, 0)
    }

    // --- внутренняя область: смещения [2..58] (§3.3) ---
    let greenonly = matches!(p.chroma_mode, ChromaMode::GreenOnly);
    let chroma_bits = p.chroma_bits() as u32;
    let l_levels = 1u32 << (p.luma_bits as u32);
    let c_levels = 1u32 << chroma_bits;
    let (a_l, a_c) = amplitudes(p, black_255, white_255);

    // строка ir=0 (y=2): референсная строка §3.4
    for ic in 0..INTERIOR {
        let x = RING + ic;
        sym[RING * GRID + x] = ref_pattern(ic % REF_PERIOD, black_255, white_255);
    }
    // строки ir=1..=55 (y=3..=57): payload-сетка 57×55 (§3.3)
    for pr in 0..PAYLOAD_ROWS {
        let y = RING + 1 + pr;
        for pc in 0..PAYLOAD_COLS {
            let x = RING + pc;
            let s = cells[pr * PAYLOAD_COLS + pc] as u32;
            sym[y * GRID + x] = encode_cell(s, chroma_bits, l_levels, c_levels, a_l, a_c, greenonly);
        }
    }
    // строка ir=56 (y=58): строка счётчика кадров (§3.3/§6.3). Середина
    // средне-серая; младшие 8 бит `counter` — 8 клеток в начале строки И их
    // дубль в конце (white = бит 1, MSB-first). Дубль позволяет рваному снимку
    // прочитать номер кадра как выше, так и ниже разрыва.
    let gray = [MID as u8; 3];
    let crow = (RING + INTERIOR - 1) * GRID;
    for ic in 0..INTERIOR {
        sym[crow + RING + ic] = gray;
    }
    for k in 0..COUNTER_BITS {
        let bit = (counter >> (COUNTER_BITS - 1 - k)) & 1 == 1;
        let color = pick(bit, white, black);
        sym[crow + RING + k] = color; // копия в начале строки
        sym[crow + RING + (INTERIOR - COUNTER_BITS + k)] = color; // копия в конце
    }

    sym
}

/// Считывает обе копии счётчика кадров из строки счётчика снимка (§3.3/§6.3):
/// (копия из начала строки, копия из конца строки). `map`/`sample` — те же
/// соглашения, что у [`demod_symbol`]. Порог — яркость (канал G) средне-серой
/// клетки из середины строки счётчика: она строго между чёрным и белым по
/// яркости при монотонной гамме, поэтому white ⇔ G_клетки > G_серого. Обе
/// копии обычно совпадают; в рваном снимке они относятся к разным кадрам.
pub fn read_counters(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> (u8, u8) {
    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    let cy = RING + INTERIOR - 1; // строка счётчика в координатах GRID
    // серый эталон из середины строки счётчика (заведомо gray-регион).
    let g_ref = sample_cell(quiet, cell, RING + INTERIOR / 2, cy, map, sample)[1];
    let mut start = 0u8;
    let mut end = 0u8;
    for k in 0..COUNTER_BITS {
        let cx_s = RING + k;
        let cx_e = RING + (INTERIOR - COUNTER_BITS + k);
        let g_s = sample_cell(quiet, cell, cx_s, cy, map, sample)[1];
        let g_e = sample_cell(quiet, cell, cx_e, cy, map, sample)[1];
        start = (start << 1) | (g_s > g_ref) as u8;
        end = (end << 1) | (g_e > g_ref) as u8;
    }
    (start, end)
}

/// Сэмплирует клетку (cx, cy) символа: центр, либо среднее 2×2 субсэмплов
/// в центр ± cell/4 при cell ≥ 8 (§5.2 MUST). Возвращает сенсорный RGB.
fn sample_cell(
    quiet: usize,
    cell: usize,
    cx: usize,
    cy: usize,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> [f64; 3] {
    let u = ((quiet + cx) * cell) as f64 + cell as f64 / 2.0;
    let v = ((quiet + cy) * cell) as f64 + cell as f64 / 2.0;
    if cell >= 8 {
        let d = cell as f64 / 4.0;
        let mut acc = [0.0f64; 3];
        for &(sx, sy) in &[(-d, -d), (-d, d), (d, -d), (d, d)] {
            let (x, y) = map(u + sx, v + sy);
            let s = sample(x, y);
            for c in 0..3 {
                acc[c] += s[c] as f64;
            }
        }
        for c in 0..3 {
            acc[c] /= 4.0;
        }
        acc
    } else {
        let (x, y) = map(u, v);
        let s = sample(x, y);
        [s[0] as f64, s[1] as f64, s[2] as f64]
    }
}

/// Ближайший индекс уровня для значения в [−1, 1] на `levels` уровнях.
fn nearest_level(val: f64, levels: u32) -> u32 {
    if levels <= 1 {
        return 0;
    }
    let idx = ((val + 1.0) * 0.5 * (levels - 1) as f64).round();
    idx.clamp(0.0, (levels - 1) as f64) as u32
}

/// b -> Грей-код.
fn binary_to_gray(b: u32) -> u32 {
    b ^ (b >> 1)
}

/// Грей-код -> b.
fn gray_to_binary(mut g: u32) -> u32 {
    let mut b = 0u32;
    while g != 0 {
        b ^= g;
        g >>= 1;
    }
    b
}

/// Округление + кламп drive в 0..255.
fn quant(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Выбор цвета клетки кольца по её белизне.
fn pick(white: bool, w: [u8; 3], k: [u8; 3]) -> [u8; 3] {
    if white {
        w
    } else {
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ChromaMode;

    fn prof(luma_bits: u8, chroma_mode: ChromaMode, cell: u8, quiet_zone: u8) -> CalibProfile {
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: cell,
            frame_hold_periods: 6,
            luma_bits,
            chroma_mode,
            gamma_g_q: 28,       // γ_G = 2.2
            gamma_r_delta_q: 6,  // γ_R = 2.15
            gamma_b_delta_q: 10, // γ_B = 2.25
            white_level_q: 15,   // 100%
            black_level_q: 2,    // 2%
            noise_sigma_q: 0,
            mtf_limit_px: 6,
            torn_frames_q: 0,
            crosstalk_rg_q: 0,
            crosstalk_gb_q: 0,
            quiet_zone,
            fec_overhead: 0,
        }
    }

    /// Эталонная бинаризация ЗЧ через f64: белая при arg(z[n]) > 0, где
    /// z[n] = exp(−jπ·q·n(n+1)/61). Фазу точно приводим по целочисленному
    /// модулю периода (122), затем берём НАСТОЯЩИЙ аргумент через atan2 —
    /// независимо от проверяемого порога «k ≥ 61». Приведение по модулю
    /// избегает катастрофической потери точности на больших фазах (иначе
    /// граница arg=0 при k=0, напр. root=1,n=60, считается неустойчиво).
    fn zc_white_ref(root: u32, n: usize) -> bool {
        let k = (root as u64 * n as u64 * (n as u64 + 1)) % (2 * GRID as u64);
        let theta = -std::f64::consts::PI * k as f64 / GRID as f64; // (−2π, 0]
        let arg = theta.sin().atan2(theta.cos()); // (−π, π]
        arg > 0.0
    }

    #[test]
    fn zc_binary_matches_float_reference() {
        for root in ZC_ROOTS {
            for n in 0..GRID {
                assert_eq!(
                    zc_binary(root, n),
                    zc_white_ref(root, n),
                    "root {root}, n {n}"
                );
            }
        }
    }

    #[test]
    fn zc_root_sequences_pairwise_distinct() {
        let seqs: Vec<Vec<bool>> = ZC_ROOTS
            .iter()
            .map(|&r| (0..GRID).map(|n| zc_binary(r, n)).collect())
            .collect();
        for i in 0..seqs.len() {
            for j in (i + 1)..seqs.len() {
                assert_ne!(seqs[i], seqs[j], "roots {} и {}", ZC_ROOTS[i], ZC_ROOTS[j]);
            }
        }
    }

    #[test]
    fn frame_size_math_and_quiet_zone_presets() {
        let cells = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
        for qz in 0..=3u8 {
            let p = prof(2, ChromaMode::Mono, 2, qz);
            let quiet = 2 * (qz as usize + 1); // §3.1: пресеты 2,4,6,8
            let f = render_symbol(&p, &cells);
            assert_eq!(f.quiet_cells, quiet, "qz {qz}");
            let expect = (GRID + 2 * quiet) * p.cell_size_px as usize;
            assert_eq!(f.size_px, expect, "qz {qz}");
            assert_eq!(f.rgb.len(), expect * expect, "qz {qz}");
            // угол тихой зоны — средне-серый
            assert_eq!(f.rgb[0], [128, 128, 128], "qz {qz}");
        }
    }

    #[test]
    fn gray_code_adjacent_levels_differ_in_one_bit() {
        for bits in 1..=4u32 {
            let levels = 1u32 << bits;
            for b in 0..levels {
                // Грей-код обратим
                assert_eq!(gray_to_binary(binary_to_gray(b)), b, "bits {bits}, b {b}");
                if b + 1 < levels {
                    let diff = binary_to_gray(b) ^ binary_to_gray(b + 1);
                    assert_eq!(
                        diff.count_ones(),
                        1,
                        "соседние уровни {b},{} различаются не в одном бите",
                        b + 1
                    );
                }
            }
        }
    }

    /// xorshift64: детерминированный ГПСЧ без внешних зависимостей.
    struct XorShift64(u64);
    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    /// Прогоняет полный roundtrip render -> идеальный дисплей+линейный сенсор
    /// -> demod и требует SER = 0. `gain`/`off` моделируют произвольные
    /// per-channel усиление/смещение сенсора (нормировка §3.4 обязана их снять).
    fn assert_roundtrip(p: &CalibProfile, gain: f64, off: f64, seed: u64) {
        let bits = bits_per_cell(p);
        let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        let mut rng = XorShift64(seed);
        let cells: Vec<u8> = (0..PAYLOAD_COLS * PAYLOAD_ROWS)
            .map(|_| (rng.next() as u32 & mask) as u8)
            .collect();

        let frame = render_symbol(p, &cells);
        let size = frame.size_px;
        let rgb = &frame.rgb;
        let gr = p.gamma_r() as f64;
        let gg = p.gamma_g() as f64;
        let gb = p.gamma_b() as f64;

        // идеальный дисплей (drive^γ) + линейный сенсор (a··+b), map = тождество
        let sample = |x: f64, y: f64| -> [f32; 3] {
            let xi = (x as usize).min(size - 1);
            let yi = (y as usize).min(size - 1);
            let d = rgb[yi * size + xi];
            [
                (gain * (d[0] as f64 / 255.0).powf(gr) + off) as f32,
                (gain * (d[1] as f64 / 255.0).powf(gg) + off) as f32,
                (gain * (d[2] as f64 / 255.0).powf(gb) + off) as f32,
            ]
        };
        let map = |u: f64, v: f64| (u, v);

        let got = demod_symbol(p, &map, &sample);
        assert_eq!(got.len(), cells.len());
        let errors = got.iter().zip(cells.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(
            errors, 0,
            "SER != 0 для luma={} chroma={:?} cell={} gain={} off={}",
            p.luma_bits, p.chroma_mode, p.cell_size_px, gain, off
        );
    }

    #[test]
    fn mode_a_clean_roundtrip_ideal_channel() {
        let cases = [
            (3u8, ChromaMode::Chroma2),
            (2u8, ChromaMode::Mono),
            (1u8, ChromaMode::GreenOnly),
        ];
        for (luma, mode) in cases {
            for cell in [8u8, 16u8] {
                let p = prof(luma, mode, cell, 0);
                assert_roundtrip(&p, 1.0, 0.0, 0x1234_5678_9ABC_DEF0 ^ cell as u64);
            }
        }
    }

    /// Счётчик кадров (§3.3/§6.3): render_symbol_counter -> read_counters через
    /// идеальный дисплей+линейный сенсор возвращает ОБЕ копии для всех 256 значений.
    #[test]
    fn frame_counter_roundtrip_all_values() {
        let p = prof(3, ChromaMode::Chroma2, 16, 1);
        let cells = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
        let gr = p.gamma_r() as f64;
        let gg = p.gamma_g() as f64;
        let gb = p.gamma_b() as f64;
        for counter in 0u8..=255 {
            let frame = render_symbol_counter(&p, &cells, counter);
            let size = frame.size_px;
            let rgb = frame.rgb.clone();
            let sample = move |x: f64, y: f64| -> [f32; 3] {
                let xi = (x as usize).min(size - 1);
                let yi = (y as usize).min(size - 1);
                let d = rgb[yi * size + xi];
                [
                    (d[0] as f64 / 255.0).powf(gr) as f32,
                    (d[1] as f64 / 255.0).powf(gg) as f32,
                    (d[2] as f64 / 255.0).powf(gb) as f32,
                ]
            };
            let map = |u: f64, v: f64| (u, v);
            let (start, end) = read_counters(&p, &map, &sample);
            assert_eq!(start, counter, "start-копия для counter={counter}");
            assert_eq!(end, counter, "end-копия для counter={counter}");
        }
    }

    #[test]
    fn mode_a_roundtrip_absorbs_sensor_gain_offset() {
        let cases = [
            (3u8, ChromaMode::Chroma2),
            (2u8, ChromaMode::Mono),
            (1u8, ChromaMode::GreenOnly),
        ];
        for (luma, mode) in cases {
            for cell in [8u8, 16u8] {
                let p = prof(luma, mode, cell, 0);
                // сенсор с усилением 0.8 и подъёмом чёрного 0.05 на всех каналах
                assert_roundtrip(&p, 0.8, 0.05, 0xC0FF_EE00_1234_5678 ^ cell as u64);
            }
        }
    }
}
