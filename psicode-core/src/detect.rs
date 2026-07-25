//! Обнаружение ЗЧ-рамки символа на снимке (§3.2, нормативный набросок §3.2/§3.3):
//! грубая детекция четырёхугольника -> 4-точечная гомография -> пер-сторонняя
//! 1-D корреляция с корнями ЗЧ -> назначение стороны/ориентации -> итеративное
//! уточнение. Это реальная замена «genie geometry» симулятора: на входе только
//! снимок (линейная яркость), на выходе — гомография и поворот для demod.
//!
//! Модуль под фичей `std`: нужна вещественная математика (`sqrt`/`powf`/`floor`).
//! Внешних зависимостей нет (жёсткое правило воркспейса): гомография решается
//! собственным Гауссом 8×8, ресемплинг — собственной билинейкой.

use crate::profile::CalibProfile;
use crate::symbol::{zc_binary, GRID, ZC_ROOTS};
use alloc::vec::Vec;
use core::fmt;

/// Сторона символа в клетках как f64 (габарит клеточной сетки).
const G: f64 = GRID as f64;
/// Полуклеточное смещение к центру клетки/центровой линии кольца.
const HALF: f64 = 0.5;
/// Окно поиска дробного лага при корреляции стороны, в клетках.
const LAG_MAX: i32 = 8; // ±8 шагов
/// Шаг поиска лага, в клетках.
const LAG_STEP: f64 = 0.25;
/// Минимальная средняя корреляция четырёх сторон для признания кадра (§3.2).
/// Значение приколочено измерениями в тестах (real≈1.0, шум/белый квадрат≈0).
const SCORE_MIN: f64 = 0.45;
/// Минимальный отрыв лучшей ориентации от второй (в единицах score), иначе
/// ориентация неоднозначна. Real-кадр даёт отрыв ~0.6; импосторы — <0.1.
const AMBIG_MARGIN: f64 = 0.15;

/// Результат обнаружения символа на снимке.
pub struct Detection {
    /// Гомография: клеточные координаты сетки 61×61 (u,v ∈ [0,61], (0,0) —
    /// внешний угол верхне-левой клетки кольца) -> координаты снимка (px).
    pub homography: [[f64; 3]; 3],
    /// На сколько четвертей (90°) повёрнут снимок относительно канона.
    pub rotation_quadrants: u8,
    /// Средняя нормированная корреляция четырёх сторон, [0,1].
    pub score: f64,
}

/// Ошибки обнаружения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectError {
    /// Кадр не найден: нет контраста/структуры или корреляция ниже порога.
    NotFound,
    /// Найдено, но ориентация неоднозначна (две четверти близки по score).
    Ambiguous,
    /// Некорректный вход: размеры не бьются с длиной буфера.
    BadInput,
}

impl fmt::Display for DetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectError::NotFound => f.write_str("ZC frame not found"),
            DetectError::Ambiguous => f.write_str("frame orientation ambiguous"),
            DetectError::BadInput => f.write_str("bad input: dimensions do not match buffer"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DetectError {}

/// Обнаружение символа на снимке. `luma` — линейная яркость (напр. G-канал),
/// `w`·`h`, построчно. См. алгоритм v0 в описании модуля.
pub fn detect_symbol(w: usize, h: usize, luma: &[f32]) -> Result<Detection, DetectError> {
    if w == 0 || h == 0 || luma.len() != w * h {
        return Err(DetectError::BadInput);
    }
    // символ v0 — минимум GRID×GRID пикселей (1 px/клетку); меньше не бывает.
    if w < GRID || h < GRID {
        return Err(DetectError::NotFound);
    }

    // 1. грубая детекция: нормировка по робастным перцентилям + маска активных
    //    (не средне-серых) пикселей + экстремальные углы четырёхугольника.
    let corners = coarse_corners(w, h, luma).ok_or(DetectError::NotFound)?;
    let mut h0 = build_h(&corners).ok_or(DetectError::NotFound)?;

    // 2. пер-сторонняя корреляция со всеми четырьмя корнями (forward): матрица
    //    4×4 корреляций и лучших лагов при текущей гомографии.
    let mut cm = [[0.0f64; 4]; 4];
    for side in 0..4 {
        for ri in 0..4 {
            let (c, _l) = side_corr(luma, w, h, &h0, side, ZC_ROOTS[ri]);
            cm[side][ri] = c;
        }
    }

    // 3. ориентация: снимок = канон, повёрнутый на r четвертей CW, тогда сторона
    //    i (по часовой: 0=верх,1=право,2=низ,3=лево) несёт корень с индексом
    //    (i−r) mod 4. Выбираем r с максимальной суммой корреляций.
    let mut totals = [0.0f64; 4];
    for r in 0..4usize {
        let mut t = 0.0;
        for side in 0..4 {
            let ri = (side + 4 - r) % 4;
            t += cm[side][ri];
        }
        totals[r] = t;
    }
    let (best_r, best_total, second_total) = argmax2(&totals);
    let score0 = (best_total / 4.0).clamp(0.0, 1.0);
    if score0 < SCORE_MIN {
        return Err(DetectError::NotFound);
    }
    if (best_total - second_total) / 4.0 < AMBIG_MARGIN {
        return Err(DetectError::Ambiguous);
    }

    // 4. уточнение: пер-сторонний лаг сдвигает углы вдоль своих сторон,
    //    пересобираем гомографию. Две итерации сходятся с запасом.
    let mut c = corners;
    for _ in 0..2 {
        let hc = match build_h(&c) {
            Some(hc) => hc,
            None => break,
        };
        let mut deltas = [0.0f64; 4];
        for side in 0..4 {
            let ri = (side + 4 - best_r) % 4;
            let (_c, l) = side_corr(luma, w, h, &hc, side, ZC_ROOTS[ri]);
            deltas[side] = l;
        }
        // единичные (на клетку) векторы сторон по часовой стрелке.
        let e_t = scale(sub(c[1], c[0]), 1.0 / G);
        let e_r = scale(sub(c[2], c[1]), 1.0 / G);
        let e_b = scale(sub(c[3], c[2]), 1.0 / G);
        let e_l = scale(sub(c[0], c[3]), 1.0 / G);
        // угол смещается на сумму вкладов двух примыкающих сторон (ортогональные
        // направления рёбер -> суммирование, не усреднение).
        let n0 = add(c[0], add(scale(e_l, deltas[3]), scale(e_t, deltas[0])));
        let n1 = add(c[1], add(scale(e_t, deltas[0]), scale(e_r, deltas[1])));
        let n2 = add(c[2], add(scale(e_r, deltas[1]), scale(e_b, deltas[2])));
        let n3 = add(c[3], add(scale(e_b, deltas[2]), scale(e_l, deltas[3])));
        c = [n0, n1, n2, n3];
    }
    if let Some(hr) = build_h(&c) {
        h0 = hr;
    }

    // 4b. тонкое выравнивание по ПОЛНОМУ двойному кольцу (§3.2): 2-D корреляция
    //     всех ~488 известных клеток (внешнее + инвертированное внутреннее,
    //     четыре стороны) вместо 244 одномерных отсчётов. Координатный спуск по
    //     углам ловит субклеточный сдвиг, устойчиво к блюру (центры клеток
    //     дольше краёв сохраняют знак). Уточнение геометрии, не ориентации.
    let cf = refine_ring(luma, w, h, &c, best_r);
    if let Some(hf) = build_h(&cf) {
        h0 = hf;
    }

    // 5. финальный score по уточнённой гомографии.
    let mut fs = 0.0;
    for side in 0..4 {
        let ri = (side + 4 - best_r) % 4;
        let (cc, _l) = side_corr(luma, w, h, &h0, side, ZC_ROOTS[ri]);
        fs += cc;
    }
    let score = (fs / 4.0).clamp(0.0, 1.0);
    if score < SCORE_MIN {
        return Err(DetectError::NotFound);
    }

    Ok(Detection {
        homography: h0,
        rotation_quadrants: best_r as u8,
        score,
    })
}

/// Мост к [`crate::symbol::demod_symbol`]: map из плоскости Frame (display-px, с
/// тихой зоной) в снимок. display-px -> клеточные координаты канона -> учёт
/// поворота (канон -> наблюдаемая сетка) -> гомография, так что demod читает
/// payload в каноническом порядке.
pub fn frame_map(p: &CalibProfile, d: &Detection) -> impl Fn(f64, f64) -> (f64, f64) {
    let quiet = p.quiet_zone_cells() as f64;
    let cell = p.cell_size_px as f64;
    let h = d.homography;
    let r = d.rotation_quadrants % 4;
    move |u: f64, v: f64| {
        // display-px -> клеточные координаты канона.
        let cu = u / cell - quiet;
        let cv = v / cell - quiet;
        // канон -> наблюдаемая (снимочная) сетка: снимок повёрнут на r·90° CW.
        let (ou, ov) = rotate_cell(cu, cv, r);
        apply_h(&h, ou, ov)
    }
}

// ---------------------------------------------------------------------------
// Грубая детекция
// ---------------------------------------------------------------------------

/// Четыре внешних угла кольца [tl, tr, br, bl] в непрерывных координатах снимка.
/// None — если контраста нет (средне-серое поле) или четырёхугольник вырожден.
fn coarse_corners(w: usize, h: usize, luma: &[f32]) -> Option<[(f64, f64); 4]> {
    // робастные перцентили 1%/99% для оценки контрастного размаха (устойчиво
    // к выбросам). Работаем в исходном (линейном) домене яркости.
    let (p1, p99) = percentiles(luma, 0.01, 0.99);
    let span = p99 - p1;
    if span < 1e-6 {
        return None; // равномерное поле — детектировать нечего.
    }
    // Фон — тихая зона (§3.1): равномерное средне-серое, но в ЛИНЕЙНОМ домене
    // оно не 0.5, поэтому опорное значение берём как среднюю яркость по рамке
    // снимка (у реального кадра рамка целиком в тихой зоне). Активны пиксели,
    // далёкие от фона: клетки кольца почти чёрные/белые -> активны при любом
    // бите ЗЧ, а тихая зона и средне-серый payload — нет.
    let bg = border_mean(luma, w, h);
    let thr = 0.15 * span;

    // экстремальные точки по четырём диагональным проекциям (§ алгоритм v0.1).
    let mut tl = (isize::MAX, 0usize, 0usize); // минимизируем i+j
    let mut tr = (isize::MIN, 0usize, 0usize); // максимизируем i−j
    let mut br = (isize::MIN, 0usize, 0usize); // максимизируем i+j
    let mut bl = (isize::MIN, 0usize, 0usize); // максимизируем j−i
    let mut active = 0usize;
    for j in 0..h {
        let row = j * w;
        for i in 0..w {
            if (luma[row + i] as f64 - bg).abs() <= thr {
                continue; // близко к фону — неактивен.
            }
            active += 1;
            let ii = i as isize;
            let jj = j as isize;
            if ii + jj < tl.0 {
                tl = (ii + jj, i, j);
            }
            if ii - jj > tr.0 {
                tr = (ii - jj, i, j);
            }
            if ii + jj > br.0 {
                br = (ii + jj, i, j);
            }
            if jj - ii > bl.0 {
                bl = (jj - ii, i, j);
            }
        }
    }
    if active < 4 * GRID {
        return None; // недостаточно структуры.
    }
    // полупиксельная привязка к ВНЕШНЕЙ границе кольца: угол = внешний край
    // экстремального пикселя (уточнение потом снимет остаточный сдвиг).
    let p_tl = (tl.1 as f64, tl.2 as f64);
    let p_tr = (tr.1 as f64 + 1.0, tr.2 as f64);
    let p_br = (br.1 as f64 + 1.0, br.2 as f64 + 1.0);
    let p_bl = (bl.1 as f64, bl.2 as f64 + 1.0);

    // отбраковка вырожденной геометрии: все стороны заметной длины.
    let quad = [p_tl, p_tr, p_br, p_bl];
    for k in 0..4 {
        let len = norm(sub(quad[(k + 1) % 4], quad[k]));
        if len < 4.0 {
            return None;
        }
    }
    Some(quad)
}

/// Средняя яркость по 1-пиксельной рамке снимка (опора «фон/тихая зона»).
fn border_mean(luma: &[f32], w: usize, h: usize) -> f64 {
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for i in 0..w {
        sum += luma[i] as f64; // верхняя строка
        sum += luma[(h - 1) * w + i] as f64; // нижняя строка
        cnt += 2;
    }
    for j in 1..h - 1 {
        sum += luma[j * w] as f64; // левый столбец
        sum += luma[j * w + w - 1] as f64; // правый столбец
        cnt += 2;
    }
    sum / cnt as f64
}

/// Перцентили `lo`/`hi` (доли 0..1) по копии буфера с частичной сортировкой.
fn percentiles(luma: &[f32], lo: f64, hi: f64) -> (f64, f64) {
    let mut v: Vec<f32> = luma.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = v.len();
    let il = ((lo * (n - 1) as f64).round() as usize).min(n - 1);
    let ih = ((hi * (n - 1) as f64).round() as usize).min(n - 1);
    (v[il] as f64, v[ih] as f64)
}

// ---------------------------------------------------------------------------
// Пер-сторонняя корреляция
// ---------------------------------------------------------------------------

/// Точка центровой линии стороны `side` (0=верх,1=право,2=низ,3=лево) в
/// наблюдаемых клеточных координатах для позиции `t` вдоль обхода по часовой.
/// Центровая линия — 0.5 клетки внутрь от внешнего края (внешнее кольцо).
#[inline]
fn side_point(side: usize, t: f64) -> (f64, f64) {
    match side {
        0 => (HALF + t, HALF),         // верх: слева направо
        1 => (G - HALF, HALF + t),     // право: сверху вниз
        2 => (G - HALF - t, G - HALF), // низ: справа налево
        _ => (HALF, G - HALF - t),     // лево: снизу вверх
    }
}

/// Корреляция стороны `side` с бинаризованным корнем `root`: возвращает лучшую
/// нормированную корреляцию (Пирсон, [−1,1]) и уточнённый дробный лаг (клетки).
fn side_corr(
    luma: &[f32],
    w: usize,
    h: usize,
    hom: &[[f64; 3]; 3],
    side: usize,
    root: u32,
) -> (f64, f64) {
    // эталон ±1 по бинаризации ЗЧ (§3.2).
    let mut refseq = [0.0f64; GRID];
    for n in 0..GRID {
        refseq[n] = if zc_binary(root, n) { 1.0 } else { -1.0 };
    }
    // корреляция на сетке лагов; заодно ищем максимум для параболической
    // интерполяции субклеточного пика.
    let count = (2 * LAG_MAX + 1) as usize;
    let mut vals = Vec::with_capacity(count);
    let mut samp = [0.0f64; GRID];
    for li in -LAG_MAX..=LAG_MAX {
        let lag = li as f64 * LAG_STEP;
        for n in 0..GRID {
            let (cu, cv) = side_point(side, n as f64 + lag);
            let (x, y) = apply_h(hom, cu, cv);
            samp[n] = sample_luma(luma, w, h, x, y);
        }
        vals.push((lag, pearson(&samp, &refseq)));
    }
    // максимум корреляции.
    let best_c = vals.iter().map(|v| v.1).fold(f64::MIN, f64::max);
    // «Блочный» рендер (сплошные клетки) при точечной выборке даёт ПЛОСКИЙ пик:
    // несколько лагов совпадают ровно на максимуме. Наивный argmax выбрал бы
    // край плато и внёс бы ложный сдвиг; берём ЦЕНТР плато — он же истинный лаг
    // (для гладкого одиночного пика плато вырождается в одну точку).
    const EPS: f64 = 1e-9;
    let mut bi = 0usize;
    let mut tie_sum = 0.0;
    let mut tie_n = 0usize;
    for (k, &(lag, c)) in vals.iter().enumerate() {
        if c >= best_c - EPS {
            tie_sum += lag;
            tie_n += 1;
        }
        if c > vals[bi].1 {
            bi = k;
        }
    }
    if tie_n > 1 {
        return (best_c, tie_sum / tie_n as f64);
    }
    // Одиночный пик (гладкое изображение: downscale/keystone/сдвиг): локальный
    // взвешенный центроид в окне ±2 шага вокруг argmax. Устойчивее 3-точечной
    // параболы к шуму (усредняет по 5 отсчётам), при этом отслеживает реальный
    // субклеточный сдвиг пика (в отличие от глобального центроида).
    let k0 = bi.saturating_sub(2);
    let k1 = (bi + 2).min(vals.len() - 1);
    let mut base = f64::MAX;
    for k in k0..=k1 {
        if vals[k].1 < base {
            base = vals[k].1;
        }
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for k in k0..=k1 {
        let wk = (vals[k].1 - base).max(0.0);
        num += wk * vals[k].0;
        den += wk;
    }
    if den > 1e-12 {
        (best_c, num / den)
    } else {
        (best_c, vals[bi].0)
    }
}

/// Корреляция Пирсона (нормированная, [−1,1]). 0, если дисперсия почти нулевая
/// (напр. константная сторона белого квадрата -> корреляция не определена).
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for k in 0..a.len() {
        let da = a[k] - ma;
        let db = b[k] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa < 1e-12 || sbb < 1e-12 {
        return 0.0;
    }
    sab / (saa.sqrt() * sbb.sqrt())
}

// ---------------------------------------------------------------------------
// Тонкое выравнивание по полному двойному кольцу (2-D корреляция)
// ---------------------------------------------------------------------------

/// Центр клетки кольца в наблюдаемых клеточных координатах: `side`
/// (0=верх,1=право,2=низ,3=лево), `depth` — глубина внутрь от внешнего края
/// (0 — внешнее кольцо, 1 — внутреннее), `n` — позиция вдоль стороны по часовой
/// (0..GRID). Согласовано со [`side_point`] при depth=0.
#[inline]
fn ring_point(side: usize, depth: f64, n: f64) -> (f64, f64) {
    let a = HALF + n; // вдоль стороны (по часовой)
    let d = HALF + depth; // внутрь от внешнего края
    match side {
        0 => (a, d),         // верх: слева направо, внутрь = +v
        1 => (G - d, a),     // право: сверху вниз, внутрь = −u
        2 => (G - a, G - d), // низ: справа налево, внутрь = −v
        _ => (d, G - a),     // лево: снизу вверх, внутрь = +u
    }
}

/// Тонкое выравнивание гомографии по ПОЛНОМУ двойному ЗЧ-кольцу (§3.2): и
/// внешнее, и (инвертированное) внутреннее кольцо, все четыре стороны —
/// 2·4·GRID известных клеток ±1. Координатный спуск по четырём углам (по 2-D
/// каждый) максимизирует нормированную корреляцию СЫРЫХ (не бинаризованных)
/// центровых отсчётов с ±1-эталоном. Линейная корреляция мягко деградирует под
/// блюром (центр клетки дольше края держит знак), поэтому даёт субклеточную
/// точность там, где пер-сторонний 1-D лаг смещается под сглаживанием.
/// Возвращает уточнённые углы [tl, tr, br, bl].
fn refine_ring(
    luma: &[f32],
    w: usize,
    h: usize,
    corners: &[(f64, f64); 4],
    best_r: usize,
) -> [(f64, f64); 4] {
    // Канонические координаты отсчётов кольца + эталон ±1 (от гомографии не
    // зависят — считаем один раз). Внутреннее кольцо = инверсия внешнего, так
    // что эталон почти знако-симметричен (точное Σt=0 нарушают лишь угловые
    // пропуски внутреннего кольца — поэтому score считает честный Пирсон).
    // Отсчёты берём НЕ только по центру клетки, а тремя точками вдоль стороны
    // (−0.35, 0, +0.35 клетки): центр устойчив по знаку, а при-краевые точки
    // лежат на скате размытого перехода и делают корреляцию ЧУВСТВИТЕЛЬНОЙ к
    // субклеточному сдвигу (иначе центровая корреляция сатурируется у 1.0 под
    // блюром и не может вести спуск). t — знак клетки n для всех трёх точек.
    const TANG: [f64; 3] = [-0.35, 0.0, 0.35];
    let mut pts: Vec<(f64, f64, f64)> = Vec::with_capacity(2 * 4 * GRID * TANG.len());
    for side in 0..4 {
        let ri = (side + 4 - best_r) % 4;
        let root = ZC_ROOTS[ri];
        for n in 0..GRID {
            let t = if zc_binary(root, n) { 1.0 } else { -1.0 };
            for &off in &TANG {
                // внешнее кольцо: полная сторона (углы принадлежат стороне по
                // приоритету §3.2, корень стороны верен во всех позициях).
                let (ou, ov) = ring_point(side, 0.0, n as f64 + off);
                pts.push((ou, ov, t));
                // внутреннее кольцо: инверсия примыкающей внешней клетки, но
                // КРОМЕ угловых n=0/GRID-1 — там глубина 1 уходит на внешнее
                // кольцо СОСЕДНЕЙ стороны (иной корень), эталон был бы неверен.
                if n > 0 && n < GRID - 1 {
                    let (iu, iv) = ring_point(side, 1.0, n as f64 + off);
                    pts.push((iu, iv, -t));
                }
            }
        }
    }

    // Нормированная корреляция Пирсона кольца при углах `c` (общий вид: эталон
    // почти нулевого среднего, но угловые пропуски делают Σt≠0 — считаем честно).
    let score = |c: &[(f64, f64); 4]| -> f64 {
        let hom = match build_h(c) {
            Some(x) => x,
            None => return f64::MIN,
        };
        let n = pts.len() as f64;
        let mut sa = 0.0; // Σs
        let mut sb = 0.0; // Σt
        let mut saa = 0.0; // Σs²
        let mut sbb = 0.0; // Σt²
        let mut sab = 0.0; // Σs·t
        for &(u, v, t) in &pts {
            let (x, y) = apply_h(&hom, u, v);
            let s = sample_luma(luma, w, h, x, y);
            sa += s;
            sb += t;
            saa += s * s;
            sbb += t * t;
            sab += s * t;
        }
        let nvar_a = saa - sa * sa / n; // N·Var[s]
        let nvar_b = sbb - sb * sb / n; // N·Var[t]
        if nvar_a < 1e-12 || nvar_b < 1e-12 {
            return f64::MIN; // константная выборка — корреляция не определена.
        }
        (sab - sa * sb / n) / (nvar_a.sqrt() * nvar_b.sqrt())
    };

    let mut c = *corners;
    let mut best = score(&c);
    if best < -1.5 {
        return c; // вырожденная стартовая геометрия — не трогаем.
    }
    // масштаб клетки в px из средней длины стороны (шаг спуска задан в клетках).
    let cell_px = (norm(sub(c[1], c[0]))
        + norm(sub(c[2], c[1]))
        + norm(sub(c[3], c[2]))
        + norm(sub(c[0], c[3])))
        / (4.0 * G);
    // Порог принятия шага: корреляция кольца флуктуирует под сенсорным шумом;
    // движение принимаем только при улучшении СВЕРХ этого пола, иначе на
    // чистом-но-шумном кадре (без блюра) спуск гонялся бы за шумом и смещал
    // углы. Под блюром выигрыш субклеточного выравнивания на порядок больше
    // порога (корреляция при-краевых точек растёт с ~0.97 до ~0.99), поэтому
    // порог не мешает сходимости. Значение приколочено развёртками (см. отчёт).
    const ACCEPT_MARGIN: f64 = 5e-4;
    // паттерн-поиск с уменьшающимся шагом: 0.25 → 0.1 → 0.05 клетки, до 3 свипов.
    for &step_cells in &[0.25f64, 0.1, 0.05] {
        let step = step_cells * cell_px;
        for _ in 0..3 {
            let mut improved = false;
            for k in 0..4 {
                for axis in 0..2 {
                    for &dir in &[1.0f64, -1.0] {
                        let mut trial = c;
                        if axis == 0 {
                            trial[k].0 += dir * step;
                        } else {
                            trial[k].1 += dir * step;
                        }
                        let s = score(&trial);
                        if s > best + ACCEPT_MARGIN {
                            best = s;
                            c = trial;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Гомография и геометрия
// ---------------------------------------------------------------------------

/// 4-точечная гомография (DLT): клеточные углы (0,0),(61,0),(61,61),(0,61) ->
/// точки снимка `dst` = [tl, tr, br, bl]. Решается собственным Гауссом 8×8.
fn build_h(dst: &[(f64, f64); 4]) -> Option<[[f64; 3]; 3]> {
    let src = [(0.0, 0.0), (G, 0.0), (G, G), (0.0, G)];
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for k in 0..4 {
        let (sx, sy) = src[k];
        let (dx, dy) = dst[k];
        a[2 * k] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sy * dx];
        b[2 * k] = dx;
        a[2 * k + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -sx * dy, -sy * dy];
        b[2 * k + 1] = dy;
    }
    let x = solve8(a, b)?;
    Some([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]],
        [x[6], x[7], 1.0],
    ])
}

/// Решение системы 8×8 методом Гаусса–Жордана с частичным выбором ведущего.
fn solve8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..8 {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = a[r][col] / d;
            if f != 0.0 {
                for c in col..8 {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = [0.0f64; 8];
    for i in 0..8 {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

/// Применение гомографии: (u,v) -> (x,y).
#[inline]
fn apply_h(h: &[[f64; 3]; 3], u: f64, v: f64) -> (f64, f64) {
    let d = h[2][0] * u + h[2][1] * v + h[2][2];
    let d = if d.abs() < 1e-12 { 1e-12 } else { d };
    (
        (h[0][0] * u + h[0][1] * v + h[0][2]) / d,
        (h[1][0] * u + h[1][1] * v + h[1][2]) / d,
    )
}

/// Поворот клеточной точки канона на `r` четвертей CW в наблюдаемые координаты.
/// r=1 (90° CW): канон-верх уезжает на правую сторону снимка (§3.2 ориентация).
#[inline]
fn rotate_cell(cu: f64, cv: f64, r: u8) -> (f64, f64) {
    match r % 4 {
        0 => (cu, cv),
        1 => (G - cv, cu),
        2 => (G - cu, G - cv),
        _ => (cv, G - cu),
    }
}

/// Билинейная выборка яркости; координаты — центры пикселей (i+0.5, j+0.5).
/// За границей — зажим к краю.
fn sample_luma(luma: &[f32], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let fx = (x - 0.5).clamp(0.0, (w - 1) as f64);
    let fy = (y - 0.5).clamp(0.0, (h - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let a = luma[y0 * w + x0] as f64;
    let b = luma[y0 * w + x1] as f64;
    let c = luma[y1 * w + x0] as f64;
    let d = luma[y1 * w + x1] as f64;
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

/// Индекс и значения максимума и второго максимума по массиву из 4.
fn argmax2(t: &[f64; 4]) -> (usize, f64, f64) {
    let mut best = 0usize;
    for k in 1..4 {
        if t[k] > t[best] {
            best = k;
        }
    }
    let mut second = f64::MIN;
    for k in 0..4 {
        if k != best && t[k] > second {
            second = t[k];
        }
    }
    (best, t[best], second)
}

// --- мелкая 2-D векторная арифметика (кортежи f64) ---
#[inline]
fn sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}
#[inline]
fn add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}
#[inline]
fn scale(a: (f64, f64), s: f64) -> (f64, f64) {
    (a.0 * s, a.1 * s)
}
#[inline]
fn norm(a: (f64, f64)) -> f64 {
    (a.0 * a.0 + a.1 * a.1).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{CalibProfile, ChromaMode};
    use crate::symbol::{
        self, bits_per_cell, render_symbol, PAYLOAD_COLS, PAYLOAD_ROWS,
    };
    use alloc::vec;
    use alloc::vec::Vec;

    // --- детерминированный ГПСЧ (как в остальном репозитории) ---
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
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Стандартный нормальный отсчёт (Box–Muller), для сенсорного шума.
        fn gaussian(&mut self) -> f64 {
            let u1 = self.unit().max(1e-12);
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    /// Референсный §7.4-подобный профиль с варьируемыми cell/quiet/luma/chroma.
    fn prof(luma_bits: u8, mode: ChromaMode, cell: u8, quiet: u8) -> CalibProfile {
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: cell,
            frame_hold_periods: 6,
            luma_bits,
            chroma_mode: mode,
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
            quiet_zone: quiet,
            fec_overhead: 0,
        }
    }

    fn rand_cells(p: &CalibProfile, seed: u64) -> Vec<u8> {
        let bits = bits_per_cell(p);
        let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        let mut rng = XorShift64(seed);
        (0..PAYLOAD_COLS * PAYLOAD_ROWS)
            .map(|_| (rng.next() as u32 & mask) as u8)
            .collect()
    }

    /// Линейная яркость (G-канал в display-гамме) из drive-RGB буфера.
    fn luma_g(rgb: &[[u8; 3]], gg: f64) -> Vec<f32> {
        rgb.iter()
            .map(|c| ((c[1] as f64 / 255.0).powf(gg)) as f32)
            .collect()
    }

    /// Билинейная выборка drive-RGB (центры пикселей), зажим к краю.
    fn bilin_rgb(buf: &[[u8; 3]], w: usize, h: usize, x: f64, y: f64) -> [f64; 3] {
        let fx = (x - 0.5).clamp(0.0, (w - 1) as f64);
        let fy = (y - 0.5).clamp(0.0, (h - 1) as f64);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let mut out = [0.0f64; 3];
        for c in 0..3 {
            let a = buf[y0 * w + x0][c] as f64;
            let b = buf[y0 * w + x1][c] as f64;
            let cc = buf[y1 * w + x0][c] as f64;
            let d = buf[y1 * w + x1][c] as f64;
            let top = a + (b - a) * tx;
            let bot = cc + (d - cc) * tx;
            out[c] = top + (bot - top) * ty;
        }
        out
    }

    /// «Сенсор»: билинейка по drive-буферу + display-гамма на канал -> линейный RGB.
    fn make_sampler<'a>(
        buf: &'a [[u8; 3]],
        w: usize,
        h: usize,
        g: [f64; 3],
    ) -> impl Fn(f64, f64) -> [f32; 3] + 'a {
        move |x: f64, y: f64| {
            let d = bilin_rgb(buf, w, h, x, y);
            [
                (d[0] / 255.0).powf(g[0]) as f32,
                (d[1] / 255.0).powf(g[1]) as f32,
                (d[2] / 255.0).powf(g[2]) as f32,
            ]
        }
    }

    fn ser(got: &[u8], want: &[u8]) -> f64 {
        let e = got.iter().zip(want).filter(|(a, b)| a != b).count();
        e as f64 / want.len() as f64
    }

    fn gammas(p: &CalibProfile) -> [f64; 3] {
        [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64]
    }

    /// Поворот квадратного буфера на 90° CW (без потерь), r раз.
    fn rotate_cw(mut buf: Vec<[u8; 3]>, n: usize, r: u8) -> Vec<[u8; 3]> {
        for _ in 0..r {
            let mut out = vec![[0u8; 3]; n * n];
            for y in 0..n {
                for x in 0..n {
                    out[y * n + x] = buf[(n - 1 - x) * n + y];
                }
            }
            buf = out;
        }
        buf
    }

    /// Ground-truth углы наблюдаемой сетки в display-px для неповёрнутого кадра.
    fn gt_corners(p: &CalibProfile) -> [(f64, f64); 4] {
        let q = p.quiet_zone_cells() as f64;
        let c = p.cell_size_px as f64;
        let lo = q * c;
        let hi = (q + G) * c;
        [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    }

    fn detected_corners(d: &Detection) -> [(f64, f64); 4] {
        [
            apply_h(&d.homography, 0.0, 0.0),
            apply_h(&d.homography, G, 0.0),
            apply_h(&d.homography, G, G),
            apply_h(&d.homography, 0.0, G),
        ]
    }

    fn max_corner_err_cells(det: &[(f64, f64); 4], gt: &[(f64, f64); 4], cell: f64) -> f64 {
        let mut m = 0.0f64;
        for k in 0..4 {
            m = m.max(norm(sub(det[k], gt[k])) / cell);
        }
        m
    }

    // -----------------------------------------------------------------------
    // (a) свойства бинаризованной ЗЧ: автокорреляция и граница кросс-корреляции
    // -----------------------------------------------------------------------
    #[test]
    fn zc_binarized_autocorr_and_crosscorr_bound() {
        let seqs: Vec<[f64; GRID]> = ZC_ROOTS
            .iter()
            .map(|&r| {
                let mut s = [0.0f64; GRID];
                for n in 0..GRID {
                    s[n] = if zc_binary(r, n) { 1.0 } else { -1.0 };
                }
                s
            })
            .collect();

        // автокорреляция (Пирсон) при нулевом сдвиге = 1 для каждого корня.
        for s in &seqs {
            let a = pearson(s, s);
            assert!((a - 1.0).abs() < 1e-9, "автокорреляция != 1: {a}");
        }

        // максимум |кросс-корреляции| по (парам корней × циклич. сдвиги × реверс).
        let mut max_x = 0.0f64;
        for i in 0..seqs.len() {
            for j in 0..seqs.len() {
                if i == j {
                    continue;
                }
                for &rev in &[false, true] {
                    for shift in 0..GRID {
                        let mut b = [0.0f64; GRID];
                        for n in 0..GRID {
                            let m = (n + shift) % GRID;
                            b[n] = if rev { seqs[j][GRID - 1 - m] } else { seqs[j][m] };
                        }
                        let c = pearson(&seqs[i], &b).abs();
                        if c > max_x {
                            max_x = c;
                        }
                    }
                }
            }
        }
        eprintln!("[a] max |cross-corr| (roots×shifts×reversal) = {max_x:.4}");
        // ПРИКОЛОЧЕНО измерением: измеренная граница = 0.3524. Различимые корни
        // дают ограниченную кросс-корреляцию, много ниже автокорреляции 1.0 ->
        // стороны/ориентация различимы (§3.2). Порог с запасом над измеренным.
        assert!(
            max_x < 0.36,
            "кросс-корреляция {max_x:.4} слишком велика — стороны неразличимы"
        );
    }

    // -----------------------------------------------------------------------
    // (b) чистая детекция: углы, поворот, score, SER=0
    // -----------------------------------------------------------------------
    #[test]
    fn clean_detection_and_demod() {
        let mut worst_corner = 0.0f64;
        let mut worst_score = 1.0f64;
        for cell in [8u8, 16u8] {
            for quiet in 0..=3u8 {
                let p = prof(3, ChromaMode::Chroma2, cell, quiet);
                let cells = rand_cells(&p, 0xABCD_0000 ^ (cell as u64) ^ ((quiet as u64) << 8));
                let frame = render_symbol(&p, &cells);
                let sz = frame.size_px;
                let g = gammas(&p);
                let luma = luma_g(&frame.rgb, g[1]);

                let d = detect_symbol(sz, sz, &luma).expect("детекция чистого кадра");
                assert_eq!(d.rotation_quadrants, 0, "cell {cell} quiet {quiet}");
                let ce = max_corner_err_cells(&detected_corners(&d), &gt_corners(&p), cell as f64);
                worst_corner = worst_corner.max(ce);
                worst_score = worst_score.min(d.score);
                assert!(ce < 0.15, "corner err {ce:.4} cells (cell {cell} quiet {quiet})");

                let map = frame_map(&p, &d);
                let sample = make_sampler(&frame.rgb, sz, sz, g);
                let got = symbol::demod_symbol(&p, &map, &sample);
                assert_eq!(ser(&got, &cells), 0.0, "SER!=0 (cell {cell} quiet {quiet})");
            }
        }
        eprintln!("[b] worst corner err = {worst_corner:.4} cells, worst score = {worst_score:.4}");
        // измерено: чистый кадр -> corner err 0.0, score 1.0.
        assert!(worst_score >= 0.99, "score {worst_score:.4} ниже ожидаемого");
    }

    // -----------------------------------------------------------------------
    // (c) повороты 0/90/180/270: корректный rotation_quadrants и SER=0
    // -----------------------------------------------------------------------
    #[test]
    fn rotations_all_quadrants() {
        let p = prof(3, ChromaMode::Chroma2, 8, 1);
        let cells = rand_cells(&p, 0x5151_2727);
        let frame = render_symbol(&p, &cells);
        let sz = frame.size_px;
        let g = gammas(&p);
        for r in 0..4u8 {
            let buf = rotate_cw(frame.rgb.clone(), sz, r);
            let luma = luma_g(&buf, g[1]);
            let d = detect_symbol(sz, sz, &luma).expect("детекция повёрнутого кадра");
            assert_eq!(d.rotation_quadrants, r, "ожидался поворот {r}");
            let map = frame_map(&p, &d);
            let sample = make_sampler(&buf, sz, sz, g);
            let got = symbol::demod_symbol(&p, &map, &sample);
            assert_eq!(ser(&got, &cells), 0.0, "SER!=0 при повороте {r}");
        }
    }

    // -----------------------------------------------------------------------
    // (d) downscale 0.5× + аддитивный шум: углы и SER
    // -----------------------------------------------------------------------
    #[test]
    fn downscale_and_noise() {
        let p = prof(2, ChromaMode::Mono, 16, 1);
        let cells = rand_cells(&p, 0xD0D0_5C1E);
        let frame = render_symbol(&p, &cells);
        let sz = frame.size_px;
        let g = gammas(&p);

        // билинейный downscale 0.5× + равномерный шум ±6 drive.
        let w2 = sz / 2;
        let h2 = sz / 2;
        let mut rng = XorShift64(0x9911_2244);
        let mut small = vec![[0u8; 3]; w2 * h2];
        for j in 0..h2 {
            for i in 0..w2 {
                let s = bilin_rgb(&frame.rgb, sz, sz, (i as f64 + 0.5) * 2.0, (j as f64 + 0.5) * 2.0);
                let mut px = [0u8; 3];
                for c in 0..3 {
                    let noise = (rng.unit() - 0.5) * 12.0;
                    px[c] = (s[c] + noise).round().clamp(0.0, 255.0) as u8;
                }
                small[j * w2 + i] = px;
            }
        }
        let luma = luma_g(&small, g[1]);
        let d = detect_symbol(w2, h2, &luma).expect("детекция downscale+шум");
        assert_eq!(d.rotation_quadrants, 0);

        // ground-truth углы в координатах downscale (× 0.5), клетка = cell/2.
        let mut gt = gt_corners(&p);
        for k in 0..4 {
            gt[k] = scale(gt[k], 0.5);
        }
        let ce = max_corner_err_cells(&detected_corners(&d), &gt, p.cell_size_px as f64 * 0.5);
        eprintln!("[d] corner err = {ce:.4} cells, score = {:.4}", d.score);
        assert!(ce < 0.3, "corner err {ce:.4} cells");

        let map = frame_map(&p, &d);
        let sample = make_sampler(&small, w2, h2, g);
        let got = symbol::demod_symbol(&p, &map, &sample);
        let s = ser(&got, &cells);
        eprintln!("[d] SER = {:.4}%", s * 100.0);
        assert!(s <= 0.01, "SER {:.4}% > 1%", s * 100.0);
    }

    // -----------------------------------------------------------------------
    // (e) перспектива: мягкий keystone; detected SER близок к genie SER
    // -----------------------------------------------------------------------
    /// Общая гомография src->dst (переиспользует внутренний solve8).
    fn homog(src: &[(f64, f64); 4], dst: &[(f64, f64); 4]) -> [[f64; 3]; 3] {
        let mut a = [[0.0f64; 8]; 8];
        let mut b = [0.0f64; 8];
        for k in 0..4 {
            let (sx, sy) = src[k];
            let (dx, dy) = dst[k];
            a[2 * k] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sy * dx];
            b[2 * k] = dx;
            a[2 * k + 1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -sx * dy, -sy * dy];
            b[2 * k + 1] = dy;
        }
        let x = solve8(a, b).expect("невырожденная гомография");
        [[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], 1.0]]
    }

    #[test]
    fn perspective_keystone() {
        let p = prof(3, ChromaMode::Chroma2, 16, 1);
        let cells = rand_cells(&p, 0x0EE5_0A0E_u64);
        let frame = render_symbol(&p, &cells);
        let sz = frame.size_px;
        let g = gammas(&p);
        let sf = sz as f64;

        // мягкий keystone kx≈0.1: верхний край уже нижнего.
        let kx = 0.1 * sf;
        let src = [(0.0, 0.0), (sf, 0.0), (sf, sf), (0.0, sf)];
        let dst = [(kx, 0.0), (sf - kx, 0.0), (sf, sf), (0.0, sf)];
        let fwd = homog(&src, &dst); // display-px -> warped-px (genie map)
        let inv = homog(&dst, &src); // warped-px -> display-px

        // warped-буфер: обратной выборкой.
        let mut warped = vec![[0u8; 3]; sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let (u, v) = apply_h(&inv, x as f64 + 0.5, y as f64 + 0.5);
                warped[y * sz + x] = {
                    let d = bilin_rgb(&frame.rgb, sz, sz, u, v);
                    [d[0].round() as u8, d[1].round() as u8, d[2].round() as u8]
                };
            }
        }
        let sample = make_sampler(&warped, sz, sz, g);

        // genie: точная gie-map = fwd.
        let genie_map = move |u: f64, v: f64| apply_h(&fwd, u, v);
        let genie = symbol::demod_symbol(&p, &genie_map, &sample);
        let genie_ser = ser(&genie, &cells);

        // detected: собственная детекция.
        let luma = luma_g(&warped, g[1]);
        let d = detect_symbol(sz, sz, &luma).expect("детекция keystone");
        let map = frame_map(&p, &d);
        let det = symbol::demod_symbol(&p, &map, &sample);
        let det_ser = ser(&det, &cells);
        eprintln!(
            "[e] genie SER = {:.4}%, detected SER = {:.4}%, score = {:.4}",
            genie_ser * 100.0,
            det_ser * 100.0,
            d.score
        );
        assert!(
            det_ser - genie_ser <= 0.05,
            "detected SER {:.4}% сильно хуже genie {:.4}%",
            det_ser * 100.0,
            genie_ser * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // (f) негативы: серое поле, шум, белый квадрат -> ошибка, без ложного score
    // -----------------------------------------------------------------------
    #[test]
    fn negatives_never_false_positive() {
        let n = 300usize;
        // 1. равномерно серое поле.
        let gray = vec![0.5f32; n * n];
        assert!(matches!(
            detect_symbol(n, n, &gray),
            Err(DetectError::NotFound)
        ));

        // 2. чистый шум.
        let mut rng = XorShift64(0x7777_1111);
        let noise: Vec<f32> = (0..n * n).map(|_| rng.unit() as f32).collect();
        let noise_res = detect_symbol(n, n, &noise);
        let noise_score = noise_res.as_ref().map(|d| d.score).unwrap_or(0.0);
        eprintln!("[f] noise result = {:?}", noise_res.as_ref().map(|d| d.score));
        assert!(noise_res.is_err(), "шум принят за кадр");

        // 3. простой белый квадрат (без ЗЧ-модуляции) на сером фоне.
        let mut sq = vec![0.5f32; n * n];
        let lo = n / 4;
        let hi = 3 * n / 4;
        for y in lo..hi {
            for x in lo..hi {
                sq[y * n + x] = 1.0;
            }
        }
        let sq_res = detect_symbol(n, n, &sq);
        let sq_score = sq_res.as_ref().map(|d| d.score).unwrap_or(0.0);
        eprintln!("[f] white-square result = {:?}", sq_res.as_ref().map(|d| d.score));
        assert!(sq_res.is_err(), "белый квадрат принят за кадр");

        // 4. отрыв реального score от лучшего импостора.
        let p = prof(3, ChromaMode::Chroma2, 8, 1);
        let cells = rand_cells(&p, 0x1234_9999);
        let frame = render_symbol(&p, &cells);
        let real = detect_symbol(frame.size_px, frame.size_px, &luma_g(&frame.rgb, gammas(&p)[1]))
            .unwrap()
            .score;
        let impostor = noise_score.max(sq_score);
        eprintln!("[f] real score = {real:.4}, best impostor = {impostor:.4}, gap = {:.4}", real - impostor);
        // измерено: real ≈ 1.0, лучший импостор = 0.0 -> отрыв ≈ 1.0.
        assert!(real - impostor > 0.9, "недостаточный отрыв real от импостора");
    }

    // -----------------------------------------------------------------------
    // (g) субклеточная точность: восстановление известного дробного сдвига
    // -----------------------------------------------------------------------
    fn shift_buffer(frame: &[[u8; 3]], sz: usize, dx: f64, dy: f64) -> Vec<[u8; 3]> {
        let mut out = vec![[0u8; 3]; sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let d = bilin_rgb(frame, sz, sz, x as f64 + 0.5 - dx, y as f64 + 0.5 - dy);
                out[y * sz + x] = [d[0].round() as u8, d[1].round() as u8, d[2].round() as u8];
            }
        }
        out
    }

    #[test]
    fn subcell_shift_recovery() {
        let p = prof(3, ChromaMode::Chroma2, 16, 1);
        let cells = rand_cells(&p, 0x0FF5_E7E7);
        let frame = render_symbol(&p, &cells);
        let sz = frame.size_px;
        let g = gammas(&p);
        let cell = p.cell_size_px as f64;

        for &(du, dv) in &[(0.5f64, 0.0f64), (1.25, -0.75)] {
            let dx = du * cell;
            let dy = dv * cell;
            let buf = shift_buffer(&frame.rgb, sz, dx, dy);
            let luma = luma_g(&buf, g[1]);
            let d = detect_symbol(sz, sz, &luma).expect("детекция сдвинутого кадра");
            assert_eq!(d.rotation_quadrants, 0);
            // восстановленный сдвиг = смещение детектированных углов относительно
            // неповёрнутого ground-truth, в клетках.
            let det = detected_corners(&d);
            let gt = gt_corners(&p);
            let mut ex = 0.0f64;
            let mut ey = 0.0f64;
            for k in 0..4 {
                ex += (det[k].0 - gt[k].0) / 4.0;
                ey += (det[k].1 - gt[k].1) / 4.0;
            }
            let err = ((ex - dx).powi(2) + (ey - dy).powi(2)).sqrt() / cell;
            eprintln!("[g] shift ({du},{dv}) recovered err = {err:.4} cells");
            assert!(err < 0.15, "сдвиг ({du},{dv}) восстановлен с ошибкой {err:.4} клетки");
        }
    }

    // -----------------------------------------------------------------------
    // (h) рабочая точка симулятора под блюром: detected-demod vs genie-demod
    // -----------------------------------------------------------------------

    /// Гауссово ядро (нормированное, радиус ceil(3σ)) — как в тракте sim.
    fn gauss_kernel(sigma: f64) -> Vec<f64> {
        if sigma <= 0.0 {
            return vec![1.0];
        }
        let r = (3.0 * sigma).ceil() as usize;
        let mut k = Vec::with_capacity(2 * r + 1);
        let mut sum = 0.0;
        for i in 0..=(2 * r) {
            let x = i as f64 - r as f64;
            let wv = (-(x * x) / (2.0 * sigma * sigma)).exp();
            k.push(wv);
            sum += wv;
        }
        for wv in &mut k {
            *wv /= sum;
        }
        k
    }

    /// Сепарабельный гауссов блюр линейного RGB, clamp-to-edge (как в sim).
    fn blur_linear(buf: &[[f32; 3]], w: usize, h: usize, sigma: f64) -> Vec<[f32; 3]> {
        if sigma <= 0.0 {
            return buf.to_vec();
        }
        let k = gauss_kernel(sigma);
        let r = (k.len() / 2) as isize;
        let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
        let mut tmp = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f64; 3];
                for (ki, &wv) in k.iter().enumerate() {
                    let sx = clamp(x as isize + ki as isize - r, w);
                    let p = buf[y * w + sx];
                    for c in 0..3 {
                        acc[c] += wv * p[c] as f64;
                    }
                }
                tmp[y * w + x] = [acc[0] as f32, acc[1] as f32, acc[2] as f32];
            }
        }
        let mut out = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f64; 3];
                for (ki, &wv) in k.iter().enumerate() {
                    let sy = clamp(y as isize + ki as isize - r, h);
                    let p = tmp[sy * w + x];
                    for c in 0..3 {
                        acc[c] += wv * p[c] as f64;
                    }
                }
                out[y * w + x] = [acc[0] as f32, acc[1] as f32, acc[2] as f32];
            }
        }
        out
    }

    /// Билинейная выборка линейного RGB (центры пикселей i+0.5), зажим к краю.
    fn bilin_lin(buf: &[[f32; 3]], w: usize, h: usize, x: f64, y: f64) -> [f32; 3] {
        let fx = (x - 0.5).clamp(0.0, (w - 1) as f64);
        let fy = (y - 0.5).clamp(0.0, (h - 1) as f64);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let a = buf[y0 * w + x0][c] as f64;
            let b = buf[y0 * w + x1][c] as f64;
            let cc = buf[y1 * w + x0][c] as f64;
            let d = buf[y1 * w + x1][c] as f64;
            let top = a + (b - a) * tx;
            let bot = cc + (d - cc) * tx;
            out[c] = (top + (bot - top) * ty) as f32;
        }
        out
    }

    /// Строит снимок камеры по рабочей точке sim: §7.4-профиль (cell 16) ->
    /// display-linear (гамма на канал) -> downscale ×0.5 (→ px/cell 8) ->
    /// гауссов блюр σ (px камеры) -> аддитивный сенсорный шум -> clamp[0,1].
    /// Возвращает (линейный RGB-снимок, w, h). Прямая (genie) геометрия проста:
    /// display-координата (u,v) -> (u/2, v/2) снимка.
    fn build_camera(
        p: &CalibProfile,
        cells: &[u8],
        sigma_blur: f64,
        noise: f64,
        seed: u64,
    ) -> (Vec<[f32; 3]>, usize, usize) {
        let frame = render_symbol(p, cells);
        let sz = frame.size_px;
        let g = gammas(&p);
        // display в линейном свете (как emit_linear в sim).
        let disp: Vec<[f32; 3]> = frame
            .rgb
            .iter()
            .map(|d| {
                [
                    (d[0] as f64 / 255.0).powf(g[0]) as f32,
                    (d[1] as f64 / 255.0).powf(g[1]) as f32,
                    (d[2] as f64 / 255.0).powf(g[2]) as f32,
                ]
            })
            .collect();
        // downscale ×0.5: центр камерного пикселя (i+0.5) ← display 2·(i+0.5).
        let w2 = sz / 2;
        let h2 = sz / 2;
        let mut cam = vec![[0.0f32; 3]; w2 * h2];
        for j in 0..h2 {
            for i in 0..w2 {
                cam[j * w2 + i] =
                    bilin_lin(&disp, sz, sz, 2.0 * (i as f64 + 0.5), 2.0 * (j as f64 + 0.5));
            }
        }
        cam = blur_linear(&cam, w2, h2, sigma_blur);
        let mut rng = XorShift64(seed);
        for px in &mut cam {
            for c in 0..3 {
                let v = px[c] as f64 + rng.gaussian() * noise;
                px[c] = v.clamp(0.0, 1.0) as f32;
            }
        }
        (cam, w2, h2)
    }

    /// Средние (genie SER, detected SER) по нескольким сидам на заданной σ блюра.
    fn blur_point_sers(sigma: f64, seeds: &[u64]) -> (f64, f64) {
        let p = prof(3, ChromaMode::Chroma2, 16, 1); // §7.4-подобный, cell 16
        let noise = 2.0 / 255.0; // сенсорный шум ≈ σ=2 градации серого
        let mut g_acc = 0.0;
        let mut d_acc = 0.0;
        for (si, &seed) in seeds.iter().enumerate() {
            let cells = rand_cells(&p, 0xB1_0000 ^ seed ^ (si as u64));
            let (cam, w, h) = build_camera(&p, &cells, sigma, noise, seed);
            let sample = |x: f64, y: f64| bilin_lin(&cam, w, h, x, y);

            // genie: точная геометрия тракта (display -> снимок ×0.5).
            let genie_map = |u: f64, v: f64| (u * 0.5, v * 0.5);
            let genie = symbol::demod_symbol(&p, &genie_map, &sample);
            g_acc += ser(&genie, &cells);

            // detected: детекция ЗЧ-рамки по G-плоскости снимка + frame_map.
            let g_plane: Vec<f32> = cam.iter().map(|px| px[1]).collect();
            let d = detect_symbol(w, h, &g_plane).expect("детекция под блюром");
            assert_eq!(d.rotation_quadrants, 0);
            let map = frame_map(&p, &d);
            let det = symbol::demod_symbol(&p, &map, &sample);
            d_acc += ser(&det, &cells);
        }
        let n = seeds.len() as f64;
        (g_acc / n, d_acc / n)
    }

    /// Рабочая точка sim σ=1: детектированная геометрия не должна отставать от
    /// genie больше чем в 1.5× по SER (раньше — ~4× из-за краевого биения
    /// пер-сторонних 1-D лагов под сглаживанием).
    #[test]
    fn blur_operating_point_sigma1() {
        let seeds = [0x1111_2222u64, 0x3333_4444, 0x5555_6666, 0x7777_8888, 0x9999_AAAA];
        let (genie, detected) = blur_point_sers(1.0, &seeds);
        eprintln!(
            "[h] σ=1  genie SER = {:.4}%  detected SER = {:.4}%  ratio = {:.2}×",
            genie * 100.0,
            detected * 100.0,
            detected / genie.max(1e-9)
        );
        // ПРИКОЛОЧЕНО измерением (см. отчёт): без тонкого выравнивания было ~2.7×
        // (краевое биение пер-сторонних 1-D лагов под блюром), с ним ≈ 1.1×.
        // Порог 1.4× — цель ≤1.5× с запасом; регресс выравнивания сразу его
        // превысит. σ=1 — самая жёсткая точка (при σ=2 payload и так деградирует).
        assert!(
            detected <= 1.4 * genie,
            "detected SER {:.4}% > 1.4× genie {:.4}% (отношение {:.2}×)",
            detected * 100.0,
            genie * 100.0,
            detected / genie.max(1e-9)
        );
    }

    /// σ=2: та же рабочая точка при более сильном блюре — только репорт, без
    /// жёсткого порога (не должно регрессировать против текущего).
    #[test]
    fn blur_operating_point_sigma2() {
        let seeds = [0x1111_2222u64, 0x3333_4444, 0x5555_6666, 0x7777_8888, 0x9999_AAAA];
        let (genie, detected) = blur_point_sers(2.0, &seeds);
        eprintln!(
            "[h] σ=2  genie SER = {:.4}%  detected SER = {:.4}%  ratio = {:.2}×",
            genie * 100.0,
            detected * 100.0,
            detected / genie.max(1e-9)
        );
    }
}
