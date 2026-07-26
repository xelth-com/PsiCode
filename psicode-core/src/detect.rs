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
use alloc::vec;
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

/// Отладочный вывод стадий детекции (переменная окружения PSICODE_DETECT_DEBUG).
fn dbg_on() -> bool {
    std::env::var_os("PSICODE_DETECT_DEBUG").is_some()
}

/// Максимум кандидатов-квадов, прогоняемых через гейт ориентации (в порядке
/// [`coarse_candidates`]: сначала полная маска яркости, затем карта активности,
/// затем связные компоненты). Ограничивает стоимость: 16 корреляций на кандидат.
const CAND_LIMIT: usize = 8;

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

    // 1. грубая детекция: НАБОР кандидатов-четырёхугольников (§ алгоритм v0.1).
    //    Кандидат 0 — экстремумы полной маски «яркость − фон» (символ во весь
    //    кадр: сим, плотный кроп). Далее — крупные плотно-квадратные компоненты
    //    карты активности (символ в загромождённой сцене: живой снимок, где
    //    титул/редактор по краям утягивают глобальные экстремумы) и связные
    //    компоненты маски яркости. Ложных кандидатов отсеет гейт ориентации.
    let candidates = coarse_candidates(w, h, luma);
    if candidates.is_empty() {
        return Err(DetectError::NotFound);
    }

    // 2. ПРОХОД 1 (дешёвый): гейт ориентации по СЫРОЙ пер-сторонней корреляции
    //    на каждый кандидат (§3.2) -> лучшая ориентация r, score и отрыв. Для
    //    сим-кадров и плотных кропов грубая рамка уже на кольце -> score высок,
    //    ориентация верна. Берём глобально лучший (кандидат, r) по score.
    let mut cheap: Option<([(f64, f64); 4], usize, f64, f64)> = None;
    for corners in candidates.iter().take(CAND_LIMIT) {
        let hc = match build_h(corners) {
            Some(x) => x,
            None => continue,
        };
        let (r, s, margin) = orient_gate(luma, w, h, &hc);
        if dbg_on() {
            std::eprintln!("[detect] cand {corners:?}: r{r} score {s:.4} margin {margin:.4}");
        }
        // строгое сравнение: при равенстве score побеждает ПЕРВЫЙ кандидат
        // (полная маска яркости) — так поведение на сим-кадрах не меняется.
        if cheap.as_ref().map_or(true, |b| s > b.2) {
            cheap = Some((*corners, r, s, margin));
        }
    }
    // если дешёвый победитель уверенно прошёл гейт — выравниваем и возвращаем.
    // Сим-кадры и tight1 идут этим путём (фолбэк ниже НЕ запускается -> пины и
    // их время не трогаются).
    if let Some((cc, cr, cs, cm)) = cheap {
        if cs >= SCORE_MIN && cm >= AMBIG_MARGIN {
            let (ac, _corr) = align_ring(luma, w, h, &cc, cr);
            if let Some(d) = finalize_detection(luma, w, h, &ac, cr) {
                return Ok(d);
            }
        }
    }

    // 3. ПРОХОД 2 (фолбэк): ВЫРАВНИВАНИЕ-КАК-ГЕЙТ (§3.2). Живые снимки (tight2/3,
    //    кроп) стартуют с кривой рамкой -> сырой гейт их топит И путает
    //    ориентацию. Выравниваем КАЖДЫЙ кандидат во ВСЕХ 4 ориентациях и судим по
    //    ВЫРОВНЕННОЙ корреляции кольца: верная (кандидат, r) локается к ~1.0,
    //    ложные ЗЧ-корни другой стороны — не выше ~0.36 (граница кросс-корр).
    //    Дороже, но запускается лишь когда дешёвый проход ничего не принял.
    let mut fb: Option<([(f64, f64); 4], usize, f64, f64)> = None; // углы, r, corr, 2-я corr
    for corners in candidates.iter().take(CAND_LIMIT) {
        if build_h(corners).is_none() {
            continue;
        }
        let mut oc = [f64::MIN; 4];
        let mut oa: [[(f64, f64); 4]; 4] = [*corners; 4];
        for r in 0..4 {
            let (ac, corr) = align_ring(luma, w, h, corners, r);
            oa[r] = ac;
            oc[r] = corr;
        }
        let (br, bc, sc) = argmax2(&oc);
        if dbg_on() {
            std::eprintln!("[detect] fb cand: aligned {oc:?} -> r{br} {bc:.4} (2nd {sc:.4})");
        }
        if fb.as_ref().map_or(true, |b| bc > b.2) {
            fb = Some((oa[br], br, bc, sc));
        }
    }
    let (ac, best_r, corr, second) = match fb {
        Some(t) => t,
        None => return Err(DetectError::NotFound),
    };
    if corr < SCORE_MIN {
        return Err(DetectError::NotFound);
    }
    if corr - second < AMBIG_MARGIN {
        return Err(DetectError::Ambiguous);
    }
    finalize_detection(luma, w, h, &ac, best_r).ok_or(DetectError::NotFound)
}

/// Финализация детекции по (выровненные углы, ориентация): собирает гомографию,
/// считает финальный score суммой пер-сторонних корреляций (§3.2) и гейтит его.
/// None — вырожденная геометрия или score ниже [`SCORE_MIN`].
fn finalize_detection(
    luma: &[f32],
    w: usize,
    h: usize,
    corners: &[(f64, f64); 4],
    best_r: usize,
) -> Option<Detection> {
    let h0 = build_h(corners)?;
    let mut fs = 0.0;
    for side in 0..4 {
        let ri = (side + 4 - best_r) % 4;
        let (cc, _l) = side_corr(luma, w, h, &h0, side, ZC_ROOTS[ri]);
        fs += cc;
    }
    let score = (fs / 4.0).clamp(0.0, 1.0);
    if score < SCORE_MIN {
        return None;
    }
    Some(Detection {
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

/// Гейт ориентации при гомографии `hom`: пер-сторонняя корреляция со всеми
/// четырьмя корнями ЗЧ (§3.2). Снимок = канон, повёрнутый на r четвертей CW,
/// тогда сторона i (по часовой: 0=верх,1=право,2=низ,3=лево) несёт корень с
/// индексом (i−r) mod 4. Возвращает (лучшая ориентация r, её score∈[0,1],
/// отрыв от второй ориентации в тех же единицах). Служит и отбором кандидата,
/// и проверкой неоднозначности ориентации.
fn orient_gate(luma: &[f32], w: usize, h: usize, hom: &[[f64; 3]; 3]) -> (usize, f64, f64) {
    let mut cm = [[0.0f64; 4]; 4];
    for side in 0..4 {
        for ri in 0..4 {
            let (c, _l) = side_corr(luma, w, h, hom, side, ZC_ROOTS[ri]);
            cm[side][ri] = c;
        }
    }
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
    (
        best_r,
        (best_total / 4.0).clamp(0.0, 1.0),
        (best_total - second_total) / 4.0,
    )
}

/// Кандидаты грубой детекции: четвёрки внешних углов [tl, tr, br, bl] в
/// непрерывных координатах снимка. Порядок (важен для tie-break в гейте):
///   0) экстремумы ПОЛНОЙ маски «яркость − фон» — символ во весь кадр (сим,
///      плотный кроп); при равенстве score побеждает он -> поведение сим не
///      меняется относительно baseline;
///   1..) крупные плотно-квадратные компоненты КАРТЫ АКТИВНОСТИ — символ в
///      загромождённой сцене (живой снимок: титул/редактор по краям утягивают
///      глобальные экстремумы, но ров тихой зоны изолирует символ);
///   ..) связные компоненты маски яркости (запас). Ложных кандидатов отсеет
///      гейт ориентации в [`detect_symbol`].
fn coarse_candidates(w: usize, h: usize, luma: &[f32]) -> Vec<[(f64, f64); 4]> {
    let mut out: Vec<[(f64, f64); 4]> = Vec::new();

    // робастные перцентили 1%/99% для оценки контрастного размаха (устойчиво
    // к выбросам). Работаем в исходном (линейном) домене яркости.
    let (p1, p99) = percentiles(luma, 0.01, 0.99);
    let span = p99 - p1;
    let have_span = span >= 1e-6;
    // Фон — тихая зона (§3.1): равномерное средне-серое, но в ЛИНЕЙНОМ домене
    // оно не 0.5, поэтому опору берём как среднюю яркость по рамке снимка.
    // Активны пиксели, далёкие от фона: клетки кольца почти чёрные/белые.
    let bg = border_mean(luma, w, h);
    let thr = 0.15 * span;
    if dbg_on() {
        std::eprintln!("[detect] p1 {p1:.4} p99 {p99:.4} span {span:.4} bg {bg:.4} thr {thr:.4}");
    }

    // маска яркости и её глобальные экстремумы (baseline-путь) — кандидат 0.
    let mask: Vec<bool> = if have_span {
        luma.iter().map(|&v| (v as f64 - bg).abs() > thr).collect()
    } else {
        Vec::new()
    };
    if have_span {
        let mut whole = Extremes::new();
        for j in 0..h {
            let row = j * w;
            for i in 0..w {
                if mask[row + i] {
                    whole.add(i, j);
                }
            }
        }
        if whole.count >= 4 * GRID {
            if let Some(q) = whole.quad() {
                out.push(q);
            }
        }
    }

    // кандидаты карты активности (основной путь для живых снимков).
    out.extend(activity_candidates(w, h, luma));

    // связные компоненты маски яркости (запас: если карта активности промахнулась).
    if have_span && out.len() < CAND_LIMIT {
        let mut visited = vec![false; w * h];
        let mut comps: Vec<Extremes> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        for start in 0..w * h {
            if !mask[start] || visited[start] {
                continue;
            }
            let mut ext = Extremes::new();
            visited[start] = true;
            stack.push(start);
            while let Some(px) = stack.pop() {
                let i = px % w;
                let j = px / w;
                ext.add(i, j);
                if i > 0 && mask[px - 1] && !visited[px - 1] {
                    visited[px - 1] = true;
                    stack.push(px - 1);
                }
                if i + 1 < w && mask[px + 1] && !visited[px + 1] {
                    visited[px + 1] = true;
                    stack.push(px + 1);
                }
                if j > 0 && mask[px - w] && !visited[px - w] {
                    visited[px - w] = true;
                    stack.push(px - w);
                }
                if j + 1 < h && mask[px + w] && !visited[px + w] {
                    visited[px + w] = true;
                    stack.push(px + w);
                }
            }
            if ext.count >= 4 * GRID {
                comps.push(ext);
            }
        }
        comps.sort_by(|a, b| b.count.cmp(&a.count));
        for c in comps.iter().take(MAX_CANDIDATES) {
            if let Some(q) = c.quad() {
                out.push(q);
            }
        }
    }
    out
}

/// Максимум связных компонент маски яркости, добавляемых как кандидаты.
const MAX_CANDIDATES: usize = 4;

/// Сторона блока при агрегации карты активности (px). Активность — градиентная
/// энергия яркости, усреднённая по блокам: символ (плотная сетка чёрно-белых/
/// цветных клеток) даёт высокую активность в КАЖДОМ блоке, а тихая зона / серое
/// окно / небо — почти нулевую. Так тихая зона работает «рвом», отделяющим
/// символ от загромождения (редактор, титул), чего маска «яркость − фон» не
/// даёт (титул тоже не серый и слипается с символом).
const ACT_BLOCK: usize = 8;
/// Радиус бокс-размытия карты активности в БЛОКАХ. Заполняет «дыры» в центрах
/// крупных клеток (внутри одной клетки градиента нет), делая символ сплошным
/// пятном при размере клетки ~8..28 px. Меньше — символ рассыпается на крупной
/// клетке; больше — ров тихой зоны затягивается и символ слипается с мусором.
const ACT_BLUR_R: usize = 4;
/// Порог активной клетки карты = доля от 99.5-го перцентиля активности.
const ACT_THR_FRAC: f64 = 0.28;
/// Радиус эрозии активной маски в БЛОКАХ (морфологическое «открытие»). Рвёт
/// ТОНКИЕ мостики активности от загромождения (титул окна, кнопки, строки
/// редактора) к телу символа. Заодно компенсирует систематический вынос края
/// наружу от бокс-размытия (эрозия ≈ на тот же радиус возвращает границу к
/// внешнему краю кольца).
const ACT_ERODE_R: usize = 5;
/// Максимум квадов-кандидатов от активной карты (крупнейшие компоненты).
const ACT_MAX_CANDIDATES: usize = 4;

/// Кандидаты по КАРТЕ АКТИВНОСТИ: градиентная энергия -> агрегация по блокам ->
/// бокс-размытие -> порог -> эрозия -> связные компоненты -> ранжирование по
/// площади × плотности × квадратности. Возвращает квады внешних углов
/// крупнейших плотно-квадратных компонент (обычно первая — сам символ).
fn activity_candidates(w: usize, h: usize, luma: &[f32]) -> Vec<[(f64, f64); 4]> {
    if w < 2 * ACT_BLOCK || h < 2 * ACT_BLOCK {
        return Vec::new();
    }
    // 1. градиентная энергия на пиксель (|Δx| + |Δy|), сразу в блоки.
    let bw = w / ACT_BLOCK;
    let bh = h / ACT_BLOCK;
    let mut act = vec![0.0f64; bw * bh];
    for j in 0..bh * ACT_BLOCK {
        let row = j * w;
        let rown = if j + 1 < h { (j + 1) * w } else { row };
        let bj = j / ACT_BLOCK;
        for i in 0..bw * ACT_BLOCK {
            let c = luma[row + i] as f64;
            let gx = if i + 1 < w {
                (luma[row + i + 1] as f64 - c).abs()
            } else {
                0.0
            };
            let gy = (luma[rown + i] as f64 - c).abs();
            act[bj * bw + i / ACT_BLOCK] += gx + gy;
        }
    }
    let inv = 1.0 / (ACT_BLOCK * ACT_BLOCK) as f64;
    for a in &mut act {
        *a *= inv;
    }
    // 2. бокс-размытие через интегральное изображение.
    let blurred = box_blur(&act, bw, bh, ACT_BLUR_R);
    // 3. порог по 99.5-му перцентилю (устойчив к одиночным горячим блокам).
    let p995 = percentile_f64(&blurred, 0.995);
    let thr = ACT_THR_FRAC * p995;
    if thr < 1e-9 {
        return Vec::new();
    }
    let mask0: Vec<bool> = blurred.iter().map(|&v| v > thr).collect();
    let mask = erode(&mask0, bw, bh, ACT_ERODE_R);

    // 4. связные компоненты (4-связность) на сетке блоков.
    struct Comp {
        ext: Extremes,
        imin: usize,
        imax: usize,
        jmin: usize,
        jmax: usize,
    }
    let mut visited = vec![false; bw * bh];
    let mut stack: Vec<usize> = Vec::new();
    let mut comps: Vec<Comp> = Vec::new();
    let min_blocks = (bw * bh) / 100;
    for start in 0..bw * bh {
        if !mask[start] || visited[start] {
            continue;
        }
        let mut ext = Extremes::new();
        let (mut imin, mut imax, mut jmin, mut jmax) = (bw, 0usize, bh, 0usize);
        visited[start] = true;
        stack.push(start);
        while let Some(px) = stack.pop() {
            let i = px % bw;
            let j = px / bw;
            ext.add(i, j);
            imin = imin.min(i);
            imax = imax.max(i);
            jmin = jmin.min(j);
            jmax = jmax.max(j);
            if i > 0 && mask[px - 1] && !visited[px - 1] {
                visited[px - 1] = true;
                stack.push(px - 1);
            }
            if i + 1 < bw && mask[px + 1] && !visited[px + 1] {
                visited[px + 1] = true;
                stack.push(px + 1);
            }
            if j > 0 && mask[px - bw] && !visited[px - bw] {
                visited[px - bw] = true;
                stack.push(px - bw);
            }
            if j + 1 < bh && mask[px + bw] && !visited[px + bw] {
                visited[px + bw] = true;
                stack.push(px + bw);
            }
        }
        if ext.count >= min_blocks {
            comps.push(Comp {
                ext,
                imin,
                imax,
                jmin,
                jmax,
            });
        }
    }
    // 5. ранжирование: площадь × плотность bbox × квадратность.
    let mut ranked: Vec<(f64, usize)> = comps
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let bw_c = (c.imax - c.imin + 1) as f64;
            let bh_c = (c.jmax - c.jmin + 1) as f64;
            let solidity = c.ext.count as f64 / (bw_c * bh_c);
            let square = bw_c.min(bh_c) / bw_c.max(bh_c);
            (c.ext.count as f64 * solidity * square, k)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut out = Vec::new();
    let s = ACT_BLOCK as f64;
    for &(_, k) in ranked.iter().take(ACT_MAX_CANDIDATES) {
        let e = &comps[k].ext;
        // внешние углы блоков-экстремумов -> px (внешний край блока).
        let q = [
            (e.tl.1 as f64 * s, e.tl.2 as f64 * s),
            ((e.tr.1 as f64 + 1.0) * s, e.tr.2 as f64 * s),
            ((e.br.1 as f64 + 1.0) * s, (e.br.2 as f64 + 1.0) * s),
            (e.bl.1 as f64 * s, (e.bl.2 as f64 + 1.0) * s),
        ];
        let mut ok = true;
        for m in 0..4 {
            if norm(sub(q[(m + 1) % 4], q[m])) < 4.0 {
                ok = false;
            }
        }
        if ok {
            out.push(q);
        }
    }
    if dbg_on() {
        std::eprintln!(
            "[detect] активность: блоков {bw}x{bh}, thr {thr:.5}, компонент {}, квадов {}",
            comps.len(),
            out.len()
        );
        for q in &out {
            std::eprintln!("[detect]   act-quad {q:?}");
        }
    }
    out
}

/// Бокс-размытие 2-D через интегральное изображение (окно (2r+1)²), край — зажим.
fn box_blur(a: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut ii = vec![0.0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut rowsum = 0.0;
        for x in 0..w {
            rowsum += a[y * w + x];
            ii[(y + 1) * (w + 1) + (x + 1)] = ii[y * (w + 1) + (x + 1)] + rowsum;
        }
    }
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            let sm = ii[y1 * (w + 1) + x1] - ii[y0 * (w + 1) + x1] - ii[y1 * (w + 1) + x0]
                + ii[y0 * (w + 1) + x0];
            out[y * w + x] = sm / ((y1 - y0) * (x1 - x0)) as f64;
        }
    }
    out
}

/// Эрозия булевой маски квадратным элементом радиуса `r` (блок выживает, если
/// ВСЕ блоки окна (2r+1)² активны). Область за пределами карты — неактивна,
/// поэтому краевые блоки эродируются.
fn erode(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut ii = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut rowsum = 0u32;
        for x in 0..w {
            rowsum += mask[y * w + x] as u32;
            ii[(y + 1) * (w + 1) + (x + 1)] = ii[y * (w + 1) + (x + 1)] + rowsum;
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if x < r || y < r || x + r >= w || y + r >= h {
                continue;
            }
            let (x0, y0, x1, y1) = (x - r, y - r, x + r + 1, y + r + 1);
            // группируем как (a+d)−(b+c): каждая скобка неотрицательна и сумма
            // окна >= 0, поэтому нет промежуточного underflow (u32, debug-режим).
            let add = ii[y1 * (w + 1) + x1] + ii[y0 * (w + 1) + x0];
            let sub = ii[y0 * (w + 1) + x1] + ii[y1 * (w + 1) + x0];
            out[y * w + x] = add - sub == ((x1 - x0) * (y1 - y0)) as u32;
        }
    }
    out
}

/// Перцентиль по копии буфера f64 (доля 0..1), частичная сортировка.
fn percentile_f64(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let idx = ((q * (s.len() - 1) as f64).round() as usize).min(s.len() - 1);
    s[idx]
}

/// Трекер диагональных экстремумов множества пикселей/блоков + счётчик.
struct Extremes {
    count: usize,
    tl: (isize, usize, usize), // минимум i+j
    tr: (isize, usize, usize), // максимум i−j
    br: (isize, usize, usize), // максимум i+j
    bl: (isize, usize, usize), // максимум j−i
}

impl Extremes {
    fn new() -> Self {
        Extremes {
            count: 0,
            tl: (isize::MAX, 0, 0),
            tr: (isize::MIN, 0, 0),
            br: (isize::MIN, 0, 0),
            bl: (isize::MIN, 0, 0),
        }
    }

    fn add(&mut self, i: usize, j: usize) {
        self.count += 1;
        let ii = i as isize;
        let jj = j as isize;
        if ii + jj < self.tl.0 {
            self.tl = (ii + jj, i, j);
        }
        if ii - jj > self.tr.0 {
            self.tr = (ii - jj, i, j);
        }
        if ii + jj > self.br.0 {
            self.br = (ii + jj, i, j);
        }
        if jj - ii > self.bl.0 {
            self.bl = (jj - ii, i, j);
        }
    }

    /// Квад [tl, tr, br, bl] с полупиксельной привязкой к внешней границе кольца
    /// (угол = внешний край экстремального пикселя); None — вырожден.
    fn quad(&self) -> Option<[(f64, f64); 4]> {
        let q = [
            (self.tl.1 as f64, self.tl.2 as f64),
            (self.tr.1 as f64 + 1.0, self.tr.2 as f64),
            (self.br.1 as f64 + 1.0, self.br.2 as f64 + 1.0),
            (self.bl.1 as f64, self.bl.2 as f64 + 1.0),
        ];
        for k in 0..4 {
            if norm(sub(q[(k + 1) % 4], q[k])) < 4.0 {
                return None;
            }
        }
        Some(q)
    }
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

/// Полудиапазон пер-углового 2-D поиска анти-алиас выравнивания, в клетках (±).
const ALIGN_RANGE_CELLS: i32 = 4;
/// Порог активации грубой фазы выравнивания по стартовой корреляции кольца.
/// Живой снимок стартует с corr ~0.2 (грубая рамка криво), сим-кадр — с ~0.9+
/// (экстремумы уже на кольце). Ниже порога -> запускаем агрессивный пер-угловой
/// поиск; выше -> сразу субклеточный спуск (тождественно старому refine_ring),
/// поэтому сим-кадры и пин блюра НЕ трогаются грубой фазой.
const ALIGN_ACTIVATE: f64 = 0.6;
/// Вес штрафа тихой зоны в композитной цели трансляции (анти-алиас якорь).
/// При 0 остаётся чистая корреляция кольца (у неё есть ложные максимумы на
/// сдвиге в толщину кольца); >0 — штраф гасит их. Приколочено развёрткой на
/// живых снимках (tight1): см. WORKER_NOTES.
const QUIET_LAMBDA: f64 = 0.7;
/// Смещения вдоль стороны для отсчётов кольца (клетки): центр держит знак под
/// блюром, при-краевые точки чувствительны к субклеточному сдвигу.
const TANG: [f64; 3] = [-0.35, 0.0, 0.35];

/// Строит ПЕР-СТОРОННИЕ канонические отсчёты двойного кольца (u, v, эталон ±1) и
/// щупы тихой зоны (u, v) для ориентации `best_r`. Внутреннее кольцо = инверсия
/// внешнего, кроме угловых n (там глубина 1 уходит на сторону соседа — иной
/// корень). Щуп тихой зоны — центр первой клетки НАРУЖУ от внешнего края
/// (depth=−1). Возврат: [pts_side; 4], [quiet_side; 4].
#[allow(clippy::type_complexity)]
fn ring_samples_side(best_r: usize) -> ([Vec<(f64, f64, f64)>; 4], [Vec<(f64, f64)>; 4]) {
    let mut pts: [Vec<(f64, f64, f64)>; 4] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut quiet: [Vec<(f64, f64)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for side in 0..4 {
        let ri = (side + 4 - best_r) % 4;
        let root = ZC_ROOTS[ri];
        for n in 0..GRID {
            let t = if zc_binary(root, n) { 1.0 } else { -1.0 };
            for &off in &TANG {
                let (ou, ov) = ring_point(side, 0.0, n as f64 + off);
                pts[side].push((ou, ov, t));
                if n > 0 && n < GRID - 1 {
                    let (iu, iv) = ring_point(side, 1.0, n as f64 + off);
                    pts[side].push((iu, iv, -t));
                }
            }
            let (qu, qv) = ring_point(side, -1.0, n as f64);
            quiet[side].push((qu, qv));
        }
    }
    (pts, quiet)
}

/// Нормированная корреляция сырых отсчётов `pts` (u, v, ±1) при гомографии `hom`
/// с ±1-эталоном + амплитуда кольца (СКО). None — константная выборка.
fn corr_of(luma: &[f32], w: usize, h: usize, hom: &[[f64; 3]; 3], pts: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    let n = pts.len() as f64;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(u, v, t) in pts {
        let (x, y) = apply_h(hom, u, v);
        let s = sample_luma(luma, w, h, x, y);
        sa += s;
        sb += t;
        saa += s * s;
        sbb += t * t;
        sab += s * t;
    }
    let nvar_a = saa - sa * sa / n;
    let nvar_b = sbb - sb * sb / n;
    if nvar_a < 1e-12 || nvar_b < 1e-12 {
        return None;
    }
    Some((
        (sab - sa * sb / n) / (nvar_a.sqrt() * nvar_b.sqrt()),
        (nvar_a / n).sqrt(),
    ))
}

/// Штраф тихой зоны: СКО щупов `quiet` (попавших в кадр), нормированное на
/// амплитуду кольца `ring_amp`. Отключён (0), если щупов < GRID (кроп без тихой
/// зоны / край) — как для сим-профилей quiet=0.
fn quiet_penalty(luma: &[f32], w: usize, h: usize, hom: &[[f64; 3]; 3], quiet: &[(f64, f64)], ring_amp: f64) -> f64 {
    let (mut qs, mut qss, mut qn) = (0.0f64, 0.0f64, 0usize);
    for &(u, v) in quiet {
        let (x, y) = apply_h(hom, u, v);
        if x < 0.0 || x > (w - 1) as f64 || y < 0.0 || y > (h - 1) as f64 {
            continue;
        }
        let s = sample_luma(luma, w, h, x, y);
        qs += s;
        qss += s * s;
        qn += 1;
    }
    if qn >= GRID && ring_amp > 1e-6 {
        let qvar = (qss - qs * qs / qn as f64) / qn as f64;
        qvar.max(0.0).sqrt() / ring_amp
    } else {
        0.0
    }
}

/// АНТИ-АЛИАС 2-D выравнивание по ПОЛНОМУ двойному ЗЧ-кольцу (§3.2). Заменяет и
/// пер-сторонний 1-D лаг, и старый refine_ring. Целевая функция:
///   J = corr_ring − λ·penalty_quiet,
/// где corr_ring — нормированная корреляция полного двойного кольца (внешнее +
/// инвертированное внутреннее, 3 точки вдоль стороны) с ±1-эталоном, а
/// penalty_quiet — контраст (СКО) полосы на 1 клетку НАРУЖУ от кольца,
/// нормированный на амплитуду кольца. В верном положении полоса-щуп лежит в
/// тихой зоне (серо -> контраст ~0); при сдвиге «на толщину кольца» (где чистая
/// corr тоже высока из-за инверсии внутреннего кольца) щуп попадает на настоящее
/// кольцо -> штраф гасит J. Так истинное положение — единственный максимум, и
/// безопасны и грубый поиск трансляции, и крупные шаги спуска.
///
/// Две фазы: (1) ГРУБАЯ пер-угловая — каждый угол по 2-D сетке ±[`ALIGN_RANGE_CELLS`]
/// клетки (шаг 0.5), максимизируя сумму композитных J двух примыкающих сторон;
/// охватывает трансляцию + поворот + одиночную сторону в клаттере, но запускается
/// лишь при низкой стартовой corr (< [`ALIGN_ACTIVATE`]) — живой снимок; (2)
/// СУБКЛЕТОЧНЫЙ спуск паттерн-поиском по чистой corr (шаги 0.25 → 0.05 клетки).
/// Для выровненных кадров (сим) фаза 1 пропускается, а фаза 2 тождественна
/// прежнему refine_ring — поэтому пины (в т.ч. блюр σ=1 ≤1.4×) не трогаются.
/// Ориентацию не трогает. Возвращает (выровненные углы, полную corr кольца в
/// найденной точке) — corr служит гейтом выравнивания в фолбэке детекции.
fn align_ring(
    luma: &[f32],
    w: usize,
    h: usize,
    corners: &[(f64, f64); 4],
    best_r: usize,
) -> ([(f64, f64); 4], f64) {
    let (pts_s, quiet_s) = ring_samples_side(best_r);
    let pts_all: Vec<(f64, f64, f64)> = pts_s.iter().flatten().copied().collect();

    // Пер-сторонняя композитная J: corr стороны − λ·штраф её тихой зоны. Именно
    // ПЕР-СТОРОННЯЯ (не полная) метрика развязывает стороны — иначе промах одной
    // стороны в клаттер тонет в сумме по кольцу и жадный поиск его не видит.
    let side_j = |c: &[(f64, f64); 4], side: usize| -> f64 {
        let hom = match build_h(c) {
            Some(x) => x,
            None => return f64::MIN,
        };
        match corr_of(luma, w, h, &hom, &pts_s[side]) {
            Some((corr, amp)) => corr - QUIET_LAMBDA * quiet_penalty(luma, w, h, &hom, &quiet_s[side], amp),
            None => f64::MIN,
        }
    };
    // Полная корреляция кольца (для субклеточного спуска и отчёта).
    let full_corr = |c: &[(f64, f64); 4]| -> f64 {
        match build_h(c) {
            Some(hom) => corr_of(luma, w, h, &hom, &pts_all).map(|t| t.0).unwrap_or(f64::MIN),
            None => f64::MIN,
        }
    };
    // Пер-сторонняя чистая corr (для отладки).
    let side_corr_dbg = |c: &[(f64, f64); 4], side: usize| -> f64 {
        match build_h(c) {
            Some(hom) => corr_of(luma, w, h, &hom, &pts_s[side]).map(|t| t.0).unwrap_or(f64::MIN),
            None => f64::MIN,
        }
    };

    let mut c = *corners;
    let corr0 = full_corr(&c);
    if corr0 <= f64::MIN {
        return (c, 0.0); // вырожденная геометрия — не трогаем.
    }
    let cell_px = (norm(sub(c[1], c[0]))
        + norm(sub(c[2], c[1]))
        + norm(sub(c[3], c[2]))
        + norm(sub(c[0], c[3])))
        / (4.0 * G);
    if dbg_on() {
        std::eprintln!(
            "[detect] align in: corr {corr0:.3} sides [{:.2} {:.2} {:.2} {:.2}]",
            side_corr_dbg(&c, 0),
            side_corr_dbg(&c, 1),
            side_corr_dbg(&c, 2),
            side_corr_dbg(&c, 3)
        );
    }

    // --- фаза 1: ПЕР-УГЛОВОЙ 2-D поиск (§3.2 анти-алиас), только для живых ---
    // Грубая рамка карты активности криво стоит по НЕСКОЛЬКИМ степеням свободы
    // сразу (трансляция + поворот + одиночная сторона в клаттере), а
    // целочисленный промах даёт визуально чистый, но полностью неверный demod.
    // Двигаем КАЖДЫЙ угол по 2-D сетке ±ALIGN_RANGE_CELLS клетки (шаг 0.5),
    // максимизируя сумму композитных J двух ПРИМЫКАЮЩИХ сторон (их corr кольца −
    // λ·штраф их тихой зоны). 2-D охватывает и тангенциальный, и перпендикулярный
    // сдвиг угла; якорь тихой зоны ломает алиас, ограниченный диапазон — без
    // разбегания. Итерируем (углы делят стороны). Гейт по стартовой corr: на
    // сим-кадрах (corr высок) фаза пропускается — субклеточный спуск ниже
    // тождествен старому refine_ring, поэтому пины не трогаются.
    // угол k примыкает к сторонам (k+3)%4 и k.
    const CORNER_SIDES: [(usize, usize); 4] = [(3, 0), (0, 1), (1, 2), (2, 3)];
    if corr0 < ALIGN_ACTIVATE {
        let r = 2 * ALIGN_RANGE_CELLS; // шаг 0.5 клетки
        for _iter in 0..3 {
            let mut moved = false;
            for (k, &(s1, s2)) in CORNER_SIDES.iter().enumerate() {
                let mut best_j = side_j(&c, s1) + side_j(&c, s2);
                let mut best_off = (0.0f64, 0.0f64);
                for dj in -r..=r {
                    for di in -r..=r {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let (dx, dy) = (di as f64 * 0.5 * cell_px, dj as f64 * 0.5 * cell_px);
                        let mut trial = c;
                        trial[k].0 += dx;
                        trial[k].1 += dy;
                        let j = side_j(&trial, s1) + side_j(&trial, s2);
                        if j > best_j {
                            best_j = j;
                            best_off = (dx, dy);
                        }
                    }
                }
                if best_off != (0.0, 0.0) {
                    c[k].0 += best_off.0;
                    c[k].1 += best_off.1;
                    moved = true;
                }
            }
            if dbg_on() {
                std::eprintln!(
                    "[detect] align it: corr {:.3} sides [{:.2} {:.2} {:.2} {:.2}]",
                    full_corr(&c),
                    side_corr_dbg(&c, 0),
                    side_corr_dbg(&c, 1),
                    side_corr_dbg(&c, 2),
                    side_corr_dbg(&c, 3)
                );
            }
            if !moved {
                break;
            }
        }
    }

    // --- фаза 2: субклеточный спуск паттерн-поиском по ЧИСТОЙ корреляции ---
    // МЕЛКИЕ шаги (0.25 → 0.05 клетки), по одному углу за раз. После фазы 1 мы в
    // пределах ~0.5 клетки от истины, а ближайший алиас — на толщине кольца
    // (~2 клетки), поэтому мелкий спуск в него не свалится и якорь не нужен.
    // Тот же приём давал субклеточную точность под блюром (пин σ=1 ≤1.4×).
    const ACCEPT_MARGIN: f64 = 5e-4;
    let mut cur = full_corr(&c);
    for &step_cells in &[0.25f64, 0.1, 0.05] {
        let st = step_cells * cell_px;
        for _ in 0..3 {
            let mut improved = false;
            for k in 0..4 {
                for axis in 0..2 {
                    for &dir in &[1.0f64, -1.0] {
                        let mut trial = c;
                        if axis == 0 {
                            trial[k].0 += dir * st;
                        } else {
                            trial[k].1 += dir * st;
                        }
                        let s = full_corr(&trial);
                        if s > cur + ACCEPT_MARGIN {
                            cur = s;
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
    if dbg_on() {
        std::eprintln!(
            "[detect] align out: corr {cur:.3} sides [{:.2} {:.2} {:.2} {:.2}]",
            side_corr_dbg(&c, 0),
            side_corr_dbg(&c, 1),
            side_corr_dbg(&c, 2),
            side_corr_dbg(&c, 3)
        );
    }
    (c, cur)
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
#[cfg(test)]
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
    // (i) детекция в ЗАГРОМОЖДЁННОЙ сцене: символ на сером поле с полосами-
    //     клаттером по краям («титул окна» + «слайвер редактора»). Baseline с
    //     глобальными экстремумами тянул углы в клаттер (кадр не находился);
    //     карта активности + связные компоненты изолируют символ рвом серого
    //     поля. Регресс детектора активности сразу проваливает этот пин.
    // -----------------------------------------------------------------------

    /// Кладёт drive-RGB `frame` (sz×sz) в серый холст с полем `margin` и рисует
    /// клаттер: яркий чёрно-белый «титул» вдоль верха и тёмный «редактор» слева.
    /// Возвращает (холст, W, H, off_x, off_y).
    fn embed_with_clutter(
        frame: &[[u8; 3]],
        sz: usize,
        margin: usize,
    ) -> (Vec<[u8; 3]>, usize, usize, usize, usize) {
        let w = sz + 2 * margin;
        let h = sz + 2 * margin;
        let mut buf = vec![[128u8; 3]; w * h]; // серое поле (drive 128)
        let (ox, oy) = (margin, margin);
        for y in 0..sz {
            for x in 0..sz {
                buf[(oy + y) * w + ox + x] = frame[y * sz + x];
            }
        }
        draw_clutter(&mut buf, w, h);
        (buf, w, h, ox, oy)
    }

    /// Рисует по краям холста высококонтрастный клаттер (титул + редактор).
    fn draw_clutter(buf: &mut [[u8; 3]], w: usize, h: usize) {
        for y in 0..24 {
            for x in 0..w {
                buf[y * w + x] = if (x / 8) % 2 == 0 {
                    [10, 10, 10]
                } else {
                    [245, 245, 245]
                };
            }
        }
        for y in 0..h {
            for x in 0..18 {
                let v = if (y / 6) % 5 == 0 { 200 } else { 25 };
                buf[y * w + x] = [v, v, v];
            }
        }
    }

    #[test]
    fn cluttered_scene_detection() {
        let p = prof(3, ChromaMode::Chroma2, 8, 1);
        let cells = rand_cells(&p, 0xC1AC_7E11);
        let frame = render_symbol(&p, &cells);
        let sz = frame.size_px;
        let g = gammas(&p);
        let margin = 96usize;
        let (buf, w, h, ox, oy) = embed_with_clutter(&frame.rgb, sz, margin);
        let luma = luma_g(&buf, g[1]);

        let d = detect_symbol(w, h, &luma).expect("детекция символа в клаттере");
        assert_eq!(d.rotation_quadrants, 0, "поворот в клаттере");
        // углы должны лечь на символ (внутри поля), не в клаттер по краям.
        let q = p.quiet_zone_cells() as f64;
        let cell = p.cell_size_px as f64;
        let (lo, hi) = (q * cell, (q + G) * cell);
        let gt = [
            (ox as f64 + lo, oy as f64 + lo),
            (ox as f64 + hi, oy as f64 + lo),
            (ox as f64 + hi, oy as f64 + hi),
            (ox as f64 + lo, oy as f64 + hi),
        ];
        let ce = max_corner_err_cells(&detected_corners(&d), &gt, cell);
        eprintln!("[i] clutter corner err = {ce:.4} cells, score = {:.4}", d.score);
        assert!(ce < 0.5, "углы утянуло в клаттер: err {ce:.4} клетки");

        let map = frame_map(&p, &d);
        let sample = make_sampler(&buf, w, h, g);
        let got = symbol::demod_symbol(&p, &map, &sample);
        let s = ser(&got, &cells);
        eprintln!("[i] clutter SER = {:.4}%", s * 100.0);
        assert!(s < 0.05, "SER {:.4}% в клаттере велик", s * 100.0);

        // негатив: тот же холст БЕЗ символа (только клаттер + серое поле).
        let mut empty = vec![[128u8; 3]; w * h];
        draw_clutter(&mut empty, w, h);
        let luma_e = luma_g(&empty, g[1]);
        assert!(
            detect_symbol(w, h, &luma_e).is_err(),
            "клаттер без символа принят за кадр"
        );
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
