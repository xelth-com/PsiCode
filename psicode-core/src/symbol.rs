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

/// √3 — отношение коэффициентов осей отображения постоянной яркости (§5.1-CL).
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Масштаб прямоугольного созвездия PAM×PAM внутри гамута (§5.1-CL).
///
/// [`ConstLumaMap`] построен на ДИСКЕ |z| ≤ 1: там ограничение гамута сводится
/// к одному радиусу (см. док структуры). Созвездие же Mode A — прямоугольная
/// решётка на КВАДРАТЕ [−1, 1]², и в угловой точке (∓1, ±1) отклонение каналов
/// R/B равно `u·b·(1+√3)` вместо `u·2b` на границе диска. Множитель `2/(1+√3)`
/// возвращает угол РОВНО на границу гамута: область ограничений — правильный
/// шестиугольник, вписанная окружность которого и есть единичный диск, поэтому
/// в диагональном направлении запас есть, и это точный максимум для
/// прямоугольной решётки, а не «с потолка».
pub const CL_LATTICE_SCALE: f64 = 2.0 / (1.0 + SQRT3);

/// Битов на клетку payload: luma_bits + биты хромы режима (§5.2).
///
/// В семействе постоянной яркости (`ChromaMode::ConstLuma*`) сумма та же, но
/// `luma_bits` — биты ДЕЙСТВИТЕЛЬНОЙ оси `2G−R−B`, а не абсолютной яркости.
pub fn bits_per_cell(p: &CalibProfile) -> u32 {
    p.luma_bits as u32 + p.chroma_bits() as u32
}

/// Отображение ПОСТОЯННОЙ ЯРКОСТИ (§5.1-CL, режимы `ChromaMode::ConstLuma*`):
/// комплексный символ z = (x, y) живёт целиком в ЦВЕТНОСТИ при неизменной сумме
/// драйвов.
///
/// # Прямое
/// ```text
/// R = u·(1 − b·x + c·y)
/// G = u·(1 + 2·b·x)
/// B = u·(1 − b·x − c·y)
/// ```
/// откуда `R + G + B ≡ 3u = S` тождественно, при любых (x, y).
///
/// # Обратное (из ЛИНЕАРИЗОВАННЫХ драйвов клетки)
/// ```text
/// S_meas = R + G + B          // ИЗМЕРЕННАЯ поклеточно — это нормативно
/// x = (2G − R − B) / (2·S_meas·b)
/// y = 3·(R − B)   / (2·S_meas·c)
/// ```
/// Проверка алгебры: `2G − R − B = 6u·b·x = 2S·b·x`, `R − B = 2u·c·y =
/// (2S/3)·c·y`. Общий множитель λ на всех трёх драйвах (чем и становится поле
/// освещённости/ошибка экспозиции после обращения гаммы дисплея) сокращается в
/// этих отношениях ТОЧНО. С НОМИНАЛЬНОЙ суммой S инвариантность исчезает
/// полностью — делить нужно именно на измеренную.
///
/// # Почему c = √3·b
/// Векторы отклонений каналов в плоскости (x, y): `G: (2b, 0)`, `R: (−b, c)`,
/// `B: (−b, −c)`. Их нормы равны ⇔ `c = √3·b`; тогда векторы образуют
/// равносторонний треугольник (120°), ограничение гамута на диске сводится к
/// ОДНОМУ радиусу, и одновременно выравнивается шум по осям (из обращения
/// σ_x ∝ √6/(2Sb), σ_y ∝ 3√2/(2Sc); равенство ⇔ c = √3·b).
///
/// # Оптимум гамута
/// Обозначив A = u·2b (свинг драйва на канал на границе диска), ограничения
/// `u + A ≤ white`, `u − A ≥ black` дают максимум A при `u = (white+black)/2`,
/// где `A = (white−black)/2`. Для эталонного профиля §7.4 (black 5, white 255):
/// u = 130, S = 390, A = 125, b = 0.4808, c = 0.8327 — драйв ровно 5..255.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstLumaMap {
    /// Нейтраль на канал, u = S/3.
    pub u: f64,
    /// Коэффициент действительной оси `2G−R−B`.
    pub b: f64,
    /// Коэффициент мнимой оси `R−B`, c = √3·b.
    pub c: f64,
    /// Постоянная сумма драйвов S = 3u.
    pub s: f64,
    /// Свинг драйва канала на границе диска, A = u·2b.
    pub amp: f64,
}

impl ConstLumaMap {
    /// Прямое отображение z -> тройка drive (без квантования и клампа).
    pub fn drive(&self, x: f64, y: f64) -> [f64; 3] {
        [
            self.u * (1.0 - self.b * x + self.c * y),
            self.u * (1.0 + 2.0 * self.b * x),
            self.u * (1.0 - self.b * x - self.c * y),
        ]
    }

    /// Обратное отображение: тройка ОЦЕНЁННЫХ драйвов -> ẑ. Делит на ИЗМЕРЕННУЮ
    /// сумму каналов — в этом весь смысл отображения (см. док структуры).
    pub fn z_from_drive(&self, d: [f64; 3]) -> (f64, f64) {
        let s = (d[0] + d[1] + d[2]).max(1e-9);
        (
            (2.0 * d[1] - d[0] - d[2]) / (2.0 * s * self.b),
            3.0 * (d[0] - d[2]) / (2.0 * s * self.c),
        )
    }
}

/// Параметры отображения постоянной яркости для профиля (§5.1-CL): выводятся из
/// уровней чёрного/белого профиля, а не из констант эталона.
pub fn const_luma_map(p: &CalibProfile) -> ConstLumaMap {
    let (black_255, white_255) = levels(p);
    const_luma_from_levels(black_255 as f64, white_255 as f64)
}

/// Вывод [`ConstLumaMap`] из drive-уровней чёрного/белого.
fn const_luma_from_levels(black: f64, white: f64) -> ConstLumaMap {
    let u = 0.5 * (white + black);
    let amp = (white - u).min(u - black).max(0.0);
    let b = if u > 0.0 { amp / (2.0 * u) } else { 0.0 };
    ConstLumaMap {
        u,
        b,
        c: SQRT3 * b,
        s: 3.0 * u,
        amp,
    }
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
/// Цветокоррекция §3.4: поканальные gain/offset для ЯРКОСТНОГО семейства §5.1
/// (v0, побитово как было) и полная матрица 3×3 для семейства ПОСТОЯННОЙ
/// ЯРКОСТИ §5.1-CL. Хроматическому коду смешивание каналов фатально (замер на
/// живом телефоне: до 26% синего протекает в зелёный), см. [`ChannelSolve`].
pub fn demod_symbol(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> Vec<u8> {
    // Матрица 3×3 — только семейству постоянной яркости; яркостное семейство
    // §5.1 идёт прежним поканальным путём БАЙТ-В-БАЙТ (живая передача и тест
    // замороженного формата опираются на него).
    demod_symbol_inner(p, map, sample, p.chroma_mode.is_const_luma())
}

/// Тело [`demod_symbol`] с явным выбором ветки цветокоррекции §3.4 —
/// `want_matrix = false` принудительно оставляет поканальную нормировку
/// (используется тестом развязки как контроль).
fn demod_symbol_inner(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
    want_matrix: bool,
) -> Vec<u8> {
    let (black_255, white_255) = levels(p);
    let gammas = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    let solve = ChannelSolve::from_reference_row(
        p,
        black_255,
        white_255,
        &gammas,
        map,
        sample,
        want_matrix,
    );
    demod_with_solve(p, map, sample, &solve, &gammas)
}

/// Демодуляция payload-сетки при УЖЕ решённой цветокоррекции §3.4. Вынесена из
/// [`demod_symbol_inner`], чтобы источником матрицы могла быть не только
/// референсная строка, но и внутриполосный калибровочный кадр
/// ([`crate::calframe`]) — путь v0 при этом остаётся байт-в-байт прежним.
fn demod_with_solve(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
    solve: &ChannelSolve,
    gammas: &[f64; 3],
) -> Vec<u8> {
    let (black_255, white_255) = levels(p);
    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    // GreenOnly и Mono несут chroma_bits=0, поэтому Im-ветка отключается сама
    // собой; выбор семейства §5.1 / §5.1-CL инкапсулирован в CellCodec.
    let codec = CellCodec::new(p, black_255, white_255);

    // --- демодуляция payload-клеток ---
    let mut out = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
    for pr in 0..PAYLOAD_ROWS {
        let cy = RING + 1 + pr; // под референсной строкой
        for pc in 0..PAYLOAD_COLS {
            let cx = RING + pc;
            let s = sample_cell(quiet, cell, cx, cy, map, sample);
            out[pr * PAYLOAD_COLS + pc] = codec.decode(solve.drive(s, gammas));
        }
    }
    out
}

/// Полная развязка каналов §3.4 как ОТДЕЛЬНАЯ величина: `t̂ = N·s + q`, где
/// `s` — измеренный линеаризованный RGB клетки, `t_c = (d_c/255)^γ_c`.
///
/// Существует, чтобы матрицу можно было ОЦЕНИТЬ ОДИН РАЗ (по внутриполосному
/// калибровочному кадру, [`crate::calframe`]) и переиспользовать на десятках
/// последующих payload-кадров, вместо переоценки по односкеточной референсной
/// строке §3.4 на каждом кадре.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelMatrix {
    /// Матрица развязки 3×3 (строка = выходной канал).
    pub n: [[f64; 3]; 3],
    /// Смещение (чёрный уровень / вуаль), на выходной канал.
    pub q: [f64; 3],
}

impl ChannelMatrix {
    /// Единичная матрица без смещения (нейтральный элемент).
    pub fn identity() -> Self {
        ChannelMatrix {
            n: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            q: [0.0; 3],
        }
    }

    /// Измеренный линеаризованный RGB клетки -> drive 0..255 на канал.
    pub fn drive(&self, s: [f64; 3], gammas: &[f64; 3]) -> [f64; 3] {
        ChannelSolve::Matrix {
            n: self.n,
            q: self.q,
        }
        .drive(s, gammas)
    }

    /// Линеаризованный drive `t̂ = N·s + q` (без обращения гаммы).
    pub fn apply(&self, s: [f64; 3]) -> [f64; 3] {
        let mut t = [0.0f64; 3];
        for c in 0..3 {
            t[c] = self.n[c][0] * s[0] + self.n[c][1] * s[1] + self.n[c][2] * s[2] + self.q[c];
        }
        t
    }

    /// Относительная невязка Фробениуса к эталонной матрице: `‖N̂−N‖_F/‖N‖_F`.
    /// Метрика A/B-сравнения источников калибровки (смещение `q` не входит: оно
    /// поглощает чёрный уровень и меряется отдельно).
    pub fn frobenius_rel_error(&self, truth: &ChannelMatrix) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for c in 0..3 {
            for j in 0..3 {
                let d = self.n[c][j] - truth.n[c][j];
                num += d * d;
                den += truth.n[c][j] * truth.n[c][j];
            }
        }
        if den <= 0.0 {
            0.0
        } else {
            (num / den).sqrt()
        }
    }

    /// Ошибка ФОРМЫ смешивания: `min_α ‖α·N̂ − N‖_F / ‖N‖_F`.
    ///
    /// Почему именно она — основная метрика A/B: ОБЩИЙ скалярный множитель у
    /// матрицы физически не наблюдаем и операционно безвреден. Поле
    /// освещённости входит в снимок мультипликативно и ОДИНАКОВО во все три
    /// канала, поэтому любая оценка `N` по патчам несёт множитель `1/f̄` со
    /// средним полем `f̄` в местах патчей. Демодулятор §5.1-CL делит на
    /// ИЗМЕРЕННУЮ поклеточную сумму каналов, и этот множитель сокращается
    /// ТОЧНО; в яркостном семействе §5.1 масштаб задают якоря K/W, а не
    /// матрица. Вредна только СМЕСЬ каналов — её и меряем.
    pub fn shape_rel_error(&self, truth: &ChannelMatrix) -> f64 {
        let mut dot = 0.0;
        let mut self_sq = 0.0;
        let mut den = 0.0;
        for c in 0..3 {
            for j in 0..3 {
                dot += self.n[c][j] * truth.n[c][j];
                self_sq += self.n[c][j] * self.n[c][j];
                den += truth.n[c][j] * truth.n[c][j];
            }
        }
        if den <= 0.0 || self_sq <= 0.0 {
            return f64::INFINITY;
        }
        let alpha = dot / self_sq;
        let mut num = 0.0;
        for c in 0..3 {
            for j in 0..3 {
                let d = alpha * self.n[c][j] - truth.n[c][j];
                num += d * d;
            }
        }
        (num / den).sqrt()
    }
}

/// Демодуляция payload-сетки с ВНЕШНЕЙ матрицей развязки §3.4 (из
/// калибровочного кадра). Референсная строка снимка при этом НЕ читается —
/// именно ради этого калибровочные кадры и существуют.
pub fn demod_symbol_matrix(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
    m: &ChannelMatrix,
) -> Vec<u8> {
    let gammas = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    let solve = ChannelSolve::Matrix { n: m.n, q: m.q };
    demod_with_solve(p, map, sample, &solve, &gammas)
}

/// Оценка матрицы развязки §3.4 ПО РЕФЕРЕНСНОЙ СТРОКЕ — тот же МНК, что внутри
/// [`demod_symbol`], но как самостоятельная величина. Нужна для head-to-head
/// сравнения «референсная строка против калибровочного кадра» (см.
/// `psicode-core/examples/calib_ab.rs`). `None` — система вырождена.
pub fn estimate_matrix_reference_row(
    p: &CalibProfile,
    gammas: &[f64; 3],
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> Option<ChannelMatrix> {
    let (black_255, white_255) = levels(p);
    match ChannelSolve::from_reference_row(p, black_255, white_255, gammas, map, sample, true) {
        ChannelSolve::Matrix { n, q } => Some(ChannelMatrix { n, q }),
        ChannelSolve::PerChannel { .. } => None,
    }
}

/// Обращение сенсорной модели §3.4: измеренный линеаризованный RGB клетки ->
/// drive 0..255 на канал.
///
/// # Поканальная ветка (§5.1, режимы 0..=4 — побитово путь v0)
/// Модель на канал: `s = a·(d/255)^γ + b`. Референсная строка даёт два якоря
/// (все K-клетки и все W-клетки), из них решаются a и b, затем инвертируем
/// `d = 255·((s − b)/a)^(1/γ)`. Это снимает произвольные per-channel
/// gain/offset датчика (белый баланс, чёрный уровень), но НЕ смешивание.
///
/// # Матричная ветка (§5.1-CL, `ChromaMode::ConstLuma*`)
/// Тракт «дисплей -> оптика -> сенсор -> CCM ISP» смешивает каналы линейно, и
/// поканальная модель это смешивание не снимает. Для 1-битного ЯРКОСТНОГО кода
/// это безобидно (порог монотонен), для кода ЦВЕТНОСТИ — фатально: замер на
/// живом телефоне (Galaxy A22, 6 кадров) даёт off-diagonal до −0.26 (синий в
/// зелёный) и −0.20 (зелёный в красный), из-за чего ось Re = 2G−R−B схлопывается
/// до ~35% размаха и уезжает с центра — одно из четырёх созвездных состояний
/// оказывается по НЕВЕРНУЮ сторону порога (BER оси Re ≈ 0.24, при том что ось
/// Im = R−B работает: BER ≈ 0.03).
///
/// Поэтому здесь по ВСЕЙ референсной строке (§3.4 даёт K W R G B C M Y и серую
/// лесенку именно для этого) решается МНК-задача
/// `t̂ = N·s + q`, где `t_c = (d_c/255)^γ_c` — линеаризованный drive канала.
/// 16 различных цветов патчей на 4 параметра выходного канала; примари R/G/B
/// линейно независимы, так что система заведомо переопределена. При вырождении
/// (плохая геометрия, битый снимок) — откат на поканальную ветку.
enum ChannelSolve {
    /// Поканальные усиление и смещение: `s = a·t + b`.
    PerChannel { a: [f64; 3], b: [f64; 3] },
    /// Полная развязка каналов: `t̂ = N·s + q`.
    Matrix { n: [[f64; 3]; 3], q: [f64; 3] },
}

impl ChannelSolve {
    fn from_reference_row(
        p: &CalibProfile,
        black_255: u8,
        white_255: u8,
        gammas: &[f64; 3],
        map: &dyn Fn(f64, f64) -> (f64, f64),
        sample: &dyn Fn(f64, f64) -> [f32; 3],
        want_matrix: bool,
    ) -> Self {
        let quiet = p.quiet_zone_cells() as usize;
        let cell = p.cell_size_px as usize;
        let black = [black_255; 3];
        let white = [white_255; 3];

        let mut s_k = [0.0f64; 3];
        let mut s_w = [0.0f64; 3];
        let mut nk = 0usize;
        let mut nw = 0usize;
        // нормальные уравнения МНК: базис [s0, s1, s2, 1] на выходной канал.
        let mut ata = [[0.0f64; 4]; 4];
        let mut atb = [[0.0f64; 3]; 4];
        for ic in 0..INTERIOR {
            let pat = ref_pattern(ic % REF_PERIOD, black_255, white_255);
            let cx = RING + ic;
            let cy = RING; // референсная строка — первая строка внутренней области
            let is_k = pat == black;
            let is_w = pat == white;
            if !is_k && !is_w && !want_matrix {
                continue; // яркостное семейство читает только якоря K/W
            }
            let s = sample_cell(quiet, cell, cx, cy, map, sample);
            if is_k {
                for c in 0..3 {
                    s_k[c] += s[c];
                }
                nk += 1;
            } else if is_w {
                for c in 0..3 {
                    s_w[c] += s[c];
                }
                nw += 1;
            }
            if want_matrix {
                let basis = [s[0], s[1], s[2], 1.0];
                for c in 0..3 {
                    let t = (pat[c] as f64 / 255.0).powf(gammas[c]);
                    for i in 0..4 {
                        atb[i][c] += basis[i] * t;
                    }
                }
                for i in 0..4 {
                    for j in 0..4 {
                        ata[i][j] += basis[i] * basis[j];
                    }
                }
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
        let fallback = ChannelSolve::PerChannel {
            a: a_gain,
            b: b_off,
        };
        if !want_matrix {
            return fallback;
        }
        match solve4x3(&ata, &atb) {
            Some((n, q)) => ChannelSolve::Matrix { n, q },
            None => fallback,
        }
    }

    /// Измеренный линеаризованный RGB клетки -> drive 0..255 на канал.
    fn drive(&self, s: [f64; 3], gammas: &[f64; 3]) -> [f64; 3] {
        let mut d = [0.0f64; 3];
        for c in 0..3 {
            let t = match self {
                ChannelSolve::PerChannel { a, b } => (s[c] - b[c]) / a[c],
                ChannelSolve::Matrix { n, q } => {
                    n[c][0] * s[0] + n[c][1] * s[1] + n[c][2] * s[2] + q[c]
                }
            };
            d[c] = (255.0 * t.max(0.0).powf(1.0 / gammas[c])).clamp(0.0, 255.0);
        }
        d
    }
}

/// Решает нормальные уравнения `AᵀA · x_c = AᵀB_c` для трёх выходных каналов
/// сразу (гаусс с выбором ведущего элемента) и раскладывает решение в матрицу
/// 3×3 плюс смещение. `None` — система вырождена (не хватило разных цветов).
pub(crate) fn solve4x3(ata: &[[f64; 4]; 4], atb: &[[f64; 3]; 4]) -> Option<([[f64; 3]; 3], [f64; 3])> {
    // расширенная матрица [AᵀA | AᵀB] размера 4×7
    let mut m = [[0.0f64; 7]; 4];
    for i in 0..4 {
        m[i][..4].copy_from_slice(&ata[i]);
        for c in 0..3 {
            m[i][4 + c] = atb[i][c];
        }
    }
    // масштаб для порога вырожденности: наибольший диагональный элемент AᵀA.
    let scale = (0..4).fold(0.0f64, |acc, i| acc.max(ata[i][i].abs()));
    if scale <= 0.0 {
        return None;
    }
    for col in 0..4 {
        let mut piv = col;
        for r in col + 1..4 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-9 * scale {
            return None; // вырождено: недостаточно линейно независимых цветов
        }
        m.swap(col, piv);
        let d = m[col][col];
        for j in col..7 {
            m[col][j] /= d;
        }
        for r in 0..4 {
            if r != col {
                let f = m[r][col];
                for j in col..7 {
                    m[r][j] -= f * m[col][j];
                }
            }
        }
    }
    let mut n = [[0.0f64; 3]; 3];
    let mut q = [0.0f64; 3];
    for c in 0..3 {
        for j in 0..3 {
            n[c][j] = m[j][4 + c];
        }
        q[c] = m[3][4 + c];
        if !n[c].iter().all(|v| v.is_finite()) || !q[c].is_finite() {
            return None;
        }
    }
    Some((n, q))
}

/// [LIVE] Радиус тонкого окна локального порога (клетки): 1 = 3×3, масштаб
/// эффективного блюра камеры/ISI при ~13 px/клетку.
const LOCAL_FINE_RADIUS: i32 = 1;
/// [LIVE] Радиус грубого окна локального порога (клетки): плавный фон освещения
/// (виньетка/блик). 12 = окно 25×25 клеток.
const LOCAL_COARSE_RADIUS: i32 = 12;
/// [LIVE] Вес грубого члена двухмасштабного порога: порог эквивалентен
/// (mean_fine + A·mean_coarse)/(1+A). Подобран на живых дампах телефона (широкий
/// оптимум A∈[0.4,0.5], K∈[10,14]).
const LOCAL_COARSE_WEIGHT: f64 = 0.5;

/// [LIVE] Адаптивная демодуляция 1-битной яркостной сетки (Mono/GreenOnly с
/// luma_bits=1, chroma_bits=0) для РЕАЛЬНОЙ камеры. Заменяет ГЛОБАЛЬНЫЙ порог
/// (MID/якоря референсной строки §3.4) на ЛОКАЛЬНЫЙ ДВУХМАСШТАБНЫЙ порог по
/// яркости клеток: тонкое окно 3×3 снимает блюр и локальный наклон, грубое окно
/// радиуса `LOCAL_COARSE_RADIUS` оценивает плавный фон (виньетку/блик) и снимает
/// неоднозначность в локально-однородных блоках. Решение: белая, если
/// `(g − mean_fine) + A·(g − mean_coarse) > 0`.
///
/// Мотивация: на живых снимках телефона фон яркости гуляет 0.62..0.86 по кадру
/// (центральный блик + угловая виньетка), и глобальный порог демодулятора даёт
/// SER ≈ 0.13; локальный двухмасштабный порог — SER ≈ 0.0004 на тех же дампах.
///
/// Для не-1-битных или хромо-режимов делегирует в [`demod_symbol`] (двухуровневый
/// порог неприменим к многоуровневой яркости/хроме). Соглашения `map`/`sample` —
/// как у [`demod_symbol`]; порог монотонен, поэтому годится и сырой, и
/// линеаризованный зелёный канал `sample`.
pub fn demod_symbol_local(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> Vec<u8> {
    // Локальный порог определён только для двоичной яркости без хромы.
    if p.luma_bits != 1 || p.chroma_bits() != 0 {
        return demod_symbol(p, map, sample);
    }
    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    let ii = INTERIOR;
    // сетка зелёного по всей внутренней области (реф-строка ir=0, payload
    // ir=1..=PAYLOAD_ROWS, строка счётчика ir=INTERIOR−1) — краевым payload-
    // клеткам нужен контекст соседей за пределами payload.
    let mut g = vec![0.0f64; ii * ii];
    for ir in 0..ii {
        for ic in 0..ii {
            let s = sample_cell(quiet, cell, RING + ic, RING + ir, map, sample);
            g[ir * ii + ic] = s[1]; // Mono: R=G=B, зелёный ~ яркость
        }
    }
    // предвычисляем среднее по двум окнам (интегральное изображение, O(1)/клетку).
    let mfine = box_mean(&g, ii, LOCAL_FINE_RADIUS);
    let mcoarse = box_mean(&g, ii, LOCAL_COARSE_RADIUS);
    let aw = LOCAL_COARSE_WEIGHT;
    let mut out = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
    for pr in 0..PAYLOAD_ROWS {
        let ir = pr + 1; // payload под референсной строкой
        for pc in 0..PAYLOAD_COLS {
            let idx = ir * ii + pc;
            let gc = g[idx];
            let score = (gc - mfine[idx]) + aw * (gc - mcoarse[idx]);
            out[pr * PAYLOAD_COLS + pc] = (score > 0.0) as u8;
        }
    }
    out
}

/// Среднее по квадратному окну радиуса `rad` (клампованные края) через
/// интегральное изображение: O(n) вместо O(n·окно). `n` — сторона сетки.
fn box_mean(src: &[f64], n: usize, rad: i32) -> Vec<f64> {
    // sat[(r)(n+1)+(c)] = сумма src[0..r][0..c] (с нулевой окантовкой сверху/слева).
    let mut sat = vec![0.0f64; (n + 1) * (n + 1)];
    for r in 0..n {
        let mut row = 0.0;
        for c in 0..n {
            row += src[r * n + c];
            sat[(r + 1) * (n + 1) + (c + 1)] = sat[r * (n + 1) + (c + 1)] + row;
        }
    }
    let area = |r0: usize, c0: usize, r1: usize, c1: usize| -> f64 {
        // сумма по [r0,r1)×[c0,c1)
        sat[r1 * (n + 1) + c1] - sat[r0 * (n + 1) + c1] - sat[r1 * (n + 1) + c0]
            + sat[r0 * (n + 1) + c0]
    };
    let mut out = vec![0.0f64; n * n];
    for r in 0..n {
        let r0 = (r as i32 - rad).max(0) as usize;
        let r1 = (r as i32 + rad + 1).min(n as i32) as usize;
        for c in 0..n {
            let c0 = (c as i32 - rad).max(0) as usize;
            let c1 = (c as i32 + rad + 1).min(n as i32) as usize;
            let cnt = ((r1 - r0) * (c1 - c0)) as f64;
            out[r * n + c] = area(r0, c0, r1, c1) / cnt;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Внутренние помощники
// ---------------------------------------------------------------------------

/// Drive-значения чёрного и белого уровней (§5.1): (black_255, white_255).
pub(crate) fn levels(p: &CalibProfile) -> (u8, u8) {
    let white_255 = (255.0 * p.white_level_pct() as f64 / 100.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    let black_255 = (255.0 * p.black_level_pct() as f64 / 100.0)
        .round()
        .clamp(0.0, 255.0) as u8;
    (black_255, white_255)
}

/// Амплитуды (A_L, A_C) для §5.1 (v0, экспериментальные доли usable).
///
/// Вызывается только для «яркостного» семейства (режимы 0..=4). У `ConstLuma*`
/// собственное отображение ([`ConstLumaMap`]), и эти амплитуды к нему
/// неприменимы — арм оставлен лишь ради исчерпывающего match.
fn amplitudes(p: &CalibProfile, black_255: u8, white_255: u8) -> (f64, f64) {
    let usable = ((white_255 as f64 - MID).min(MID - black_255 as f64)).max(0.0);
    match p.chroma_mode {
        ChromaMode::Mono | ChromaMode::GreenOnly => (usable, 0.0),
        ChromaMode::Chroma1
        | ChromaMode::Chroma2
        | ChromaMode::Chroma3
        | ChromaMode::ConstLuma1
        | ChromaMode::ConstLuma2
        | ChromaMode::ConstLuma3 => {
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

/// Отображение пары осей созвездия в цвет: какое из двух семейств §5.1 активно.
enum CodecKind {
    /// §5.1 v0 (режимы 0..=4): Re на абсолютной яркости G, Im на оси R−B.
    Legacy { a_l: f64, a_c: f64, greenonly: bool },
    /// §5.1-CL (`ConstLuma*`): постоянная сумма драйвов, z целиком в цветности.
    ConstLuma(ConstLumaMap),
}

/// Кодек одной payload-клетки Mode A: символ <-> drive-RGB.
///
/// Общее у обоих семейств — раскладка символа на ДВЕ оси и Грей-код (§5.2):
/// `symbol = (re_gray << chroma_bits) | im_gray`, каждая ось независимо. Ветка
/// [`CodecKind::Legacy`] побитово повторяет путь v0 (режимы 0..=4) — ради
/// байт-совместимости живой передачи и замороженного формата; ветка
/// [`CodecKind::ConstLuma`] реализует §5.1-CL.
struct CellCodec {
    /// Биты МНИМОЙ оси (`chroma_bits` профиля).
    chroma_bits: u32,
    /// Уровней на ДЕЙСТВИТЕЛЬНОЙ оси, 2^luma_bits.
    l_levels: u32,
    /// Уровней на МНИМОЙ оси, 2^chroma_bits.
    c_levels: u32,
    kind: CodecKind,
}

impl CellCodec {
    fn new(p: &CalibProfile, black_255: u8, white_255: u8) -> Self {
        let chroma_bits = p.chroma_bits() as u32;
        let kind = if p.chroma_mode.is_const_luma() {
            CodecKind::ConstLuma(const_luma_from_levels(black_255 as f64, white_255 as f64))
        } else {
            let (a_l, a_c) = amplitudes(p, black_255, white_255);
            CodecKind::Legacy {
                a_l,
                a_c,
                greenonly: matches!(p.chroma_mode, ChromaMode::GreenOnly),
            }
        };
        CellCodec {
            chroma_bits,
            l_levels: 1u32 << (p.luma_bits as u32),
            c_levels: 1u32 << chroma_bits,
            kind,
        }
    }

    /// Символ клетки Mode A -> drive-RGB. Оси декодируются из Грей-кода
    /// раздельно (§5.2): symbol = (luma_gray << chroma_bits) | chroma_gray.
    fn encode(&self, s: u32) -> [u8; 3] {
        let luma_gray = s >> self.chroma_bits;
        let chroma_mask = (1u32 << self.chroma_bits) - 1; // при chroma_bits=0 даёт 0
        let chroma_gray = s & chroma_mask;
        let b_l = gray_to_binary(luma_gray);
        let re = -1.0 + 2.0 * b_l as f64 / (self.l_levels - 1) as f64;
        let im = if self.chroma_bits > 0 {
            let b_c = gray_to_binary(chroma_gray);
            -1.0 + 2.0 * b_c as f64 / (self.c_levels - 1) as f64
        } else {
            0.0
        };
        match &self.kind {
            CodecKind::Legacy {
                a_l,
                a_c,
                greenonly,
            } => encode_field(re, im, *a_l, *a_c, *greenonly),
            CodecKind::ConstLuma(m) => {
                // созвездие сжато в гамут (см. CL_LATTICE_SCALE)
                let d = m.drive(CL_LATTICE_SCALE * re, CL_LATTICE_SCALE * im);
                [quant(d[0]), quant(d[1]), quant(d[2])]
            }
        }
    }

    /// Линеаризованные драйвы клетки -> символ Mode A.
    fn decode(&self, d: [f64; 3]) -> u8 {
        let (re_hat, im_hat) = match &self.kind {
            // §5.1 обратное отображение: Re из G, Im из (R − B)/2.
            CodecKind::Legacy { a_l, a_c, .. } => (
                (d[1] - MID) / a_l,
                if self.chroma_bits > 0 {
                    (d[0] - d[2]) / (2.0 * a_c)
                } else {
                    0.0
                },
            ),
            // §5.1-CL: обе оси — отношения к ИЗМЕРЕННОЙ сумме каналов.
            CodecKind::ConstLuma(m) => {
                let (x, y) = m.z_from_drive(d);
                (x / CL_LATTICE_SCALE, y / CL_LATTICE_SCALE)
            }
        };
        let b_l = nearest_level(re_hat, self.l_levels);
        let luma_gray = binary_to_gray(b_l);
        let sym = if self.chroma_bits > 0 {
            let b_c = nearest_level(im_hat, self.c_levels);
            let chroma_gray = binary_to_gray(b_c);
            (luma_gray << self.chroma_bits) | chroma_gray
        } else {
            luma_gray
        };
        sym as u8
    }
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
    let codec = CellCodec::new(p, black_255, white_255);

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
            sym[y * GRID + x] = codec.encode(s);
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
pub(crate) fn sample_cell(
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
            border: crate::profile::BorderMode::LegacyInverted,
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

    /// [LIVE] demod_symbol_local на ЧИСТОМ идеальном канале (1 бит, Mono) обязан
    /// давать SER 0: локальный порог самокалибруется, а грубое окно 25×25 на
    /// случайной сетке не бывает одноцветным -> нет неоднозначных клеток. Это
    /// условие безопасно для e2e-теста приёмника (точный демод чистого кадра).
    #[test]
    fn local_demod_exact_on_clean_1bpc() {
        let p = prof(1, ChromaMode::Mono, 16, 0);
        let mut rng = XorShift64(0xA5A5_1234_DEAD_BEEF);
        let cells: Vec<u8> = (0..PAYLOAD_COLS * PAYLOAD_ROWS)
            .map(|_| (rng.next() & 1) as u8)
            .collect();
        let frame = render_symbol(&p, &cells);
        let size = frame.size_px;
        let rgb = frame.rgb.clone();
        let gg = p.gamma_g() as f64;
        let sample = move |x: f64, y: f64| -> [f32; 3] {
            let xi = (x.max(0.0) as usize).min(size - 1);
            let yi = (y.max(0.0) as usize).min(size - 1);
            let d = rgb[yi * size + xi];
            let f = |c: u8| (c as f64 / 255.0).powf(gg) as f32;
            [f(d[0]), f(d[1]), f(d[2])]
        };
        let map = |u: f64, v: f64| (u, v);
        let got = demod_symbol_local(&p, &map, &sample);
        let errors = got.iter().zip(&cells).filter(|(a, b)| a != b).count();
        assert_eq!(errors, 0, "локальный демод не точен на чистом канале: {errors}");
    }

    // -----------------------------------------------------------------------
    // §5.1-CL: отображение постоянной яркости (ChromaMode::ConstLuma*)
    // -----------------------------------------------------------------------

    /// Три режима постоянной яркости и разумные глубины действительной оси.
    const CL_CASES: [(u8, ChromaMode); 4] = [
        (1, ChromaMode::ConstLuma1),
        (2, ChromaMode::ConstLuma1),
        (3, ChromaMode::ConstLuma1),
        (2, ChromaMode::ConstLuma2),
    ];

    /// Псевдослучайная нагрузка клеток под маску bits_per_cell.
    fn rand_cells(p: &CalibProfile, seed: u64) -> Vec<u8> {
        let bits = bits_per_cell(p);
        let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        let mut rng = XorShift64(seed);
        (0..PAYLOAD_COLS * PAYLOAD_ROWS)
            .map(|_| (rng.next() as u32 & mask) as u8)
            .collect()
    }

    /// Идеальный сэмплер поверх кадра: дисплей `drive^γ` -> МУЛЬТИПЛИКАТИВНОЕ
    /// поле освещённости `field(y)` (одинаковое по трём каналам, как реальная
    /// виньетка/наклон подсветки) -> линейный сенсор. `map` — тождество.
    fn field_sampler(
        p: &CalibProfile,
        frame: &Frame,
        field: impl Fn(f64) -> f64 + 'static,
    ) -> impl Fn(f64, f64) -> [f32; 3] {
        let size = frame.size_px;
        let rgb = frame.rgb.clone();
        let g = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
        move |x: f64, y: f64| -> [f32; 3] {
            let xi = (x.max(0.0) as usize).min(size - 1);
            let yi = (y.max(0.0) as usize).min(size - 1);
            let f = field(yi as f64 / (size - 1) as f64);
            let d = rgb[yi * size + xi];
            [
                (f * (d[0] as f64 / 255.0).powf(g[0])) as f32,
                (f * (d[1] as f64 / 255.0).powf(g[1])) as f32,
                (f * (d[2] as f64 / 255.0).powf(g[2])) as f32,
            ]
        }
    }

    /// Доля ошибочных клеток после render -> sample -> demod.
    fn ser_through(p: &CalibProfile, cells: &[u8], sample: &dyn Fn(f64, f64) -> [f32; 3]) -> f64 {
        let map = |u: f64, v: f64| (u, v);
        let got = demod_symbol(p, &map, sample);
        got.iter().zip(cells).filter(|(a, b)| a != b).count() as f64 / cells.len() as f64
    }

    /// Константы гамута §5.1-CL для эталонных уровней §7.4 (black 5, white 255):
    /// u = 130, S = 390, A = 125 (драйв ровно 5..255), b = 0.4808, c = √3·b.
    #[test]
    fn const_luma_gamut_constants_match_reference_profile() {
        let p = prof(1, ChromaMode::ConstLuma1, 16, 0);
        let (black, white) = levels(&p);
        assert_eq!((black, white), (5, 255), "эталонные уровни §7.4");
        let m = const_luma_map(&p);
        assert!((m.u - 130.0).abs() < 1e-9, "u = {}", m.u);
        assert!((m.s - 390.0).abs() < 1e-9, "S = {}", m.s);
        assert!((m.amp - 125.0).abs() < 1e-9, "A = {}", m.amp);
        assert!((m.b - 0.480_769_230_769).abs() < 1e-9, "b = {}", m.b);
        assert!((m.c - SQRT3 * m.b).abs() < 1e-15, "c = {}", m.c);
        assert!((m.c - 0.832_716_734_408).abs() < 1e-9, "c = {}", m.c);
        // u ± A попадает ровно в чёрный/белый
        assert!((m.u + m.amp - white as f64).abs() < 1e-9);
        assert!((m.u - m.amp - black as f64).abs() < 1e-9);
    }

    /// Гамут: для ЛЮБОГО z единичного диска все три драйва лежат в [black, white]
    /// (и граница достигается) — при уровнях эталона и при других уровнях.
    #[test]
    fn const_luma_drive_stays_in_gamut_on_unit_disk() {
        for (wq, bq) in [(15u8, 2u8), (0, 0), (7, 15), (15, 15)] {
            let mut p = prof(1, ChromaMode::ConstLuma1, 16, 0);
            p.white_level_q = wq;
            p.black_level_q = bq;
            let (black, white) = levels(&p);
            let m = const_luma_map(&p);
            let mut max_dev = 0.0f64;
            for i in 0..720 {
                let th = core::f64::consts::PI * i as f64 / 360.0;
                for r in [0.0, 0.5, 1.0] {
                    let d = m.drive(r * th.cos(), r * th.sin());
                    for c in 0..3 {
                        assert!(
                            d[c] >= black as f64 - 1e-9 && d[c] <= white as f64 + 1e-9,
                            "w{wq} b{bq}: канал {c} драйв {} вне [{black}, {white}]",
                            d[c]
                        );
                        max_dev = max_dev.max((d[c] - m.u).abs());
                    }
                    // сумма постоянна тождественно
                    assert!((d[0] + d[1] + d[2] - m.s).abs() < 1e-9);
                }
            }
            // созвездие не вырождено и упирается в границу гамута
            assert!((max_dev - m.amp).abs() < 1e-6, "w{wq} b{bq}: свинг {max_dev}");
        }
    }

    /// Постоянство суммы драйвов R+G+B по ВСЕМ payload-клеткам отрендеренного
    /// символа (после 8-битного квантования — с точностью до ±1 градации).
    #[test]
    fn const_luma_render_keeps_channel_sum_constant() {
        for (luma, mode) in CL_CASES {
            let p = prof(luma, mode, 8, 1);
            let cells = rand_cells(&p, 0xBEEF_0000_1234_5678 ^ luma as u64);
            let sym = build_symbol_cells(&p, &cells, 0);
            let m = const_luma_map(&p);
            let mut seen = std::collections::HashSet::new();
            for pr in 0..PAYLOAD_ROWS {
                let y = RING + 1 + pr;
                for pc in 0..PAYLOAD_COLS {
                    let d = sym[y * GRID + (RING + pc)];
                    let s = d[0] as i32 + d[1] as i32 + d[2] as i32;
                    seen.insert(s);
                    assert!(
                        (s as f64 - m.s).abs() <= 1.5,
                        "{mode:?} luma{luma}: сумма {s} != S = {}",
                        m.s
                    );
                }
            }
            // разброс — только шум округления, не сигнал
            assert!(seen.len() <= 3, "{mode:?} luma{luma}: сумм {}", seen.len());
        }
    }

    /// Round-trip §5.1-CL через идеальный канал: точное восстановление клеток.
    #[test]
    fn const_luma_clean_roundtrip_ideal_channel() {
        for (luma, mode) in CL_CASES {
            for cell in [8u8, 16u8] {
                let p = prof(luma, mode, cell, 0);
                assert_roundtrip(&p, 1.0, 0.0, 0x0FED_CBA9_8765_4321 ^ cell as u64);
                // и с произвольным поканальным усилением/подъёмом чёрного
                assert_roundtrip(&p, 0.8, 0.05, 0xDEAD_BEEF_0BAD_F00D ^ cell as u64);
            }
        }
    }

    /// ГЛАВНЫЙ тест смены отображения: под МУЛЬТИПЛИКАТИВНЫМ полем освещённости
    /// (линейный наклон 0.62 -> 0.86 по кадру, замер на живом телефоне) режимы
    /// постоянной яркости восстанавливают клетки ТОЧНО, а legacy-хрома §5.1 —
    /// нет. Механизм: обращение §5.1-CL делит на ИЗМЕРЕННУЮ сумму каналов
    /// клетки, и общий множитель поля сокращается; §5.1 же кладёт Re на
    /// абсолютную яркость, а её нормировка (§3.4) снимается по референсной
    /// СТРОКЕ сверху кадра — то есть по ОДНОЙ точке поля.
    #[test]
    fn const_luma_is_immune_to_illumination_field_legacy_is_not() {
        // поле: 0.62 сверху -> 0.86 снизу, одинаково по трём каналам
        let field = |t: f64| 0.62 + (0.86 - 0.62) * t;

        // 1. новое отображение: SER = 0 на всех конфигурациях
        for (luma, mode) in CL_CASES {
            let p = prof(luma, mode, 16, 0);
            let cells = rand_cells(&p, 0x5EED_1111_2222_3333 ^ luma as u64);
            let frame = render_symbol(&p, &cells);
            let sample = field_sampler(&p, &frame, field);
            let ser = ser_through(&p, &cells, &sample);
            assert_eq!(ser, 0.0, "{mode:?} luma{luma}: SER {ser:.4} под полем");
        }

        // 2. Лобовое сравнение при РАВНОЙ ёмкости: luma_bits + Chroma1 против
        //    luma_bits + ConstLuma1 под ТЕМ ЖЕ полем.
        let mut worst_legacy = 0.0f64;
        for luma in [2u8, 3] {
            let legacy = prof(luma, ChromaMode::Chroma1, 16, 0);
            let lcells = rand_cells(&legacy, 0x5EED_1111_2222_3333);
            let lframe = render_symbol(&legacy, &lcells);
            let lsample = field_sampler(&legacy, &lframe, field);
            let lser = ser_through(&legacy, &lcells, &lsample);

            let cl = prof(luma, ChromaMode::ConstLuma1, 16, 0);
            let ccells = rand_cells(&cl, 0x5EED_1111_2222_3333);
            let cframe = render_symbol(&cl, &ccells);
            let csample = field_sampler(&cl, &cframe, field);
            let cser = ser_through(&cl, &ccells, &csample);

            assert_eq!(cser, 0.0, "ConstLuma1+luma{luma}: SER {cser:.4}");
            worst_legacy = worst_legacy.max(lser);
            println!(
                "поле 0.62..0.86, {} бит/клетку: legacy Chroma1 SER {lser:.4} vs ConstLuma1 SER {cser:.4}",
                luma as u32 + 1
            );
        }
        // При 3 уровнях наклона поля 2-битная яркостная ось ещё вытягивает
        // (ошибка чуть меньше половины шага), 3-битная — уже нет. Именно поэтому
        // §5.1 упирается в 1–2 бита на живом канале, а §5.1-CL — нет.
        assert!(
            worst_legacy > 0.10,
            "тест некорректен: legacy §5.1 обязан рассыпаться под полем (худший SER {worst_legacy:.4})"
        );
    }

    /// [LIVE] §3.4 матрица 3×3: под ЛИНЕЙНЫМ СМЕШИВАНИЕМ каналов (замер на
    /// Galaxy A22 даёт off-diagonal до −0.26: синий течёт в зелёный) семейство
    /// постоянной яркости обязано восстанавливать клетки ТОЧНО — развязка
    /// решается по референсной строке §3.4 (K W R G B C M Y + лесенка).
    /// Проверяем и то, что БЕЗ развязки код действительно рассыпается: иначе
    /// тест ничего не доказывает.
    #[test]
    fn const_luma_survives_channel_crosstalk_via_reference_row() {
        // матрица смешивания сенсора/ISP: строки суммируются к 1 (нейтраль
        // остаётся нейтралью, поэтому якоря K/W §3.4 смешивания НЕ видят —
        // ровно та ситуация, в которой поканальная нормировка бессильна).
        let mix = [
            [0.78, 0.14, 0.08],
            [0.12, 0.62, 0.26],
            [0.07, 0.28, 0.65],
        ];
        let mut worst_without = 0.0f64;
        for (luma, mode) in CL_CASES {
            let p = prof(luma, mode, 16, 0);
            let cells = rand_cells(&p, 0xC0DE_1234_5678_9ABC ^ luma as u64);
            let frame = render_symbol(&p, &cells);
            let size = frame.size_px;
            let rgb = frame.rgb.clone();
            let g = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
            let sample = move |x: f64, y: f64| -> [f32; 3] {
                let xi = (x.max(0.0) as usize).min(size - 1);
                let yi = (y.max(0.0) as usize).min(size - 1);
                let d = rgb[yi * size + xi];
                let lin = [
                    (d[0] as f64 / 255.0).powf(g[0]),
                    (d[1] as f64 / 255.0).powf(g[1]),
                    (d[2] as f64 / 255.0).powf(g[2]),
                ];
                let mut o = [0.0f32; 3];
                for c in 0..3 {
                    // 0.8 усиления и +0.03 подъёма чёрного поверх смешивания
                    let v = mix[c][0] * lin[0] + mix[c][1] * lin[1] + mix[c][2] * lin[2];
                    o[c] = (0.8 * v + 0.03) as f32;
                }
                o
            };
            let ser = ser_through(&p, &cells, &sample);
            assert_eq!(ser, 0.0, "{mode:?} luma{luma}: SER {ser:.4} под смешиванием");
            // контроль: ТОТ ЖЕ снимок через поканальную ветку разваливается —
            // развязка каналов не косметика.
            let map = |u: f64, v: f64| (u, v);
            let got = demod_symbol_inner(&p, &map, &sample, false);
            let bad = got.iter().zip(&cells).filter(|(a, b)| a != b).count();
            worst_without = worst_without.max(bad as f64 / cells.len() as f64);
        }
        assert!(
            worst_without > 0.05,
            "тест некорректен: без матрицы §5.1-CL обязан страдать (худший SER {worst_without:.4})"
        );
    }

    /// Поле, СИЛЬНО обрезающее динамику (0.35..1.0), не ломает §5.1-CL: важна
    /// только пропорция каналов, а не абсолютный уровень.
    #[test]
    fn const_luma_survives_deep_field() {
        let p = prof(2, ChromaMode::ConstLuma1, 16, 0);
        let cells = rand_cells(&p, 0x9999_8888_7777_6666);
        let frame = render_symbol(&p, &cells);
        let sample = field_sampler(&p, &frame, |t| 0.35 + 0.65 * t);
        assert_eq!(ser_through(&p, &cells, &sample), 0.0);
    }

    /// [LIVE] Под сильным ПРОСТРАНСТВЕННЫМ фоном (аддитивный градиент «блика»,
    /// растущий сверху вниз) глобальный порог демодулятора, калиброванный по
    /// верхней референсной строке, разваливается на нижних рядах, а локальный
    /// двухмасштабный порог его отслеживает. Проверяем сам МЕХАНИЗМ выигрыша.
    #[test]
    fn local_demod_survives_illumination_gradient() {
        let p = prof(1, ChromaMode::Mono, 16, 0);
        let mut rng = XorShift64(0x1357_9BDF_2468_ACE0);
        let cells: Vec<u8> = (0..PAYLOAD_COLS * PAYLOAD_ROWS)
            .map(|_| (rng.next() & 1) as u8)
            .collect();
        let frame = render_symbol(&p, &cells);
        let size = frame.size_px;
        let rgb = frame.rgb.clone();
        let gg = p.gamma_g() as f64;
        // сенсор: идеальная гамма + аддитивный «блик», линейно растущий вниз
        // (0 сверху -> +0.5 снизу). Смещение поднимает пол яркости так, что
        // нижний ЧЁРНЫЙ по сырому сигналу перекрывает верхний уровень порога.
        let sample = move |x: f64, y: f64| -> [f32; 3] {
            let xi = (x.max(0.0) as usize).min(size - 1);
            let yi = (y.max(0.0) as usize).min(size - 1);
            let glare = 0.5 * (yi as f64 / size as f64);
            let d = rgb[yi * size + xi];
            let f = |c: u8| ((c as f64 / 255.0).powf(gg) + glare).clamp(0.0, 1.5) as f32;
            [f(d[0]), f(d[1]), f(d[2])]
        };
        let map = |u: f64, v: f64| (u, v);
        let n = (PAYLOAD_COLS * PAYLOAD_ROWS) as f64;
        let g_glob = demod_symbol(&p, &map, &sample);
        let g_loc = demod_symbol_local(&p, &map, &sample);
        let e_glob = g_glob.iter().zip(&cells).filter(|(a, b)| a != b).count() as f64 / n;
        let e_loc = g_loc.iter().zip(&cells).filter(|(a, b)| a != b).count() as f64 / n;
        assert!(
            e_glob > 0.08,
            "тест некорректен: глобальный порог должен разваливаться (SER {e_glob:.4})"
        );
        assert!(
            e_loc < 0.02,
            "локальный порог не удержал градиент: SER {e_loc:.4} (глоб {e_glob:.4})"
        );
    }
}
