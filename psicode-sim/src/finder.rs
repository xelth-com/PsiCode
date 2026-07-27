//! `finder` — исследование ЦЕНТРАЛЬНОГО опознавательного знака против ЗЧ-рамки.
//!
//! # Что проверяется
//!
//! Сегодня символ ищется корреляцией ЗЧ-рамки, и искать приходится ВСЕ ЧЕТЫРЕ
//! стороны СОВМЕСТНО: одна сторона не задаёт ни положения, ни масштаба, а ЗЧ
//! палиндромна, поэтому одна сторона не различает даже направления обхода.
//! Предложение владельца: положить в ЦЕНТР знак (как «бычий глаз» Aztec), но
//! выбранный так, чтобы он хорошо брался ДВУМЕРНОЙ фурье-корреляцией. Тогда
//! положение даёт один глобальный поиск, а рамка перестаёт быть искателем и
//! становится только уточнителем масштаба, поворота и перспективы.
//!
//! Этот модуль НИЧЕГО не меняет в формате. Он только МЕРЯЕТ, во что обойдётся
//! такая замена, и даёт числа для решения.
//!
//! # Кандидаты
//!
//! | | что это | площадь |
//! |---|---|---|
//! | `marray` | M-массив: m-последовательность, свёрнутая по КТО в двумерный псевдослучайный массив с идеальной ПЕРИОДИЧЕСКОЙ автокорреляцией | n² клеток |
//! | `chirp` | бинаризованный двумерный чирп (зонная пластинка Френеля), `sign cos(π r²/n)` | n² клеток |
//! | `bullseye` | концентрические квадратные кольца шириной 1 клетка — это буквально Aztec | n² клеток |
//! | `v1border` | САМА рамка v1 как двумерный шаблон 61×61 с кольцевой маской | 0 (уже есть) |
//!
//! # Метрики
//!
//! * PSR = пик / максимум боковых лепестков на РЕАЛЬНЫХ загромождённых кадрах;
//! * запас = пик на позитиве / максимум ложной тревоги на негативах — именно он
//!   решает, можно ли доверять глобальному поиску БЕЗ стадии проверки;
//! * устойчивость к блюру (σ камеры), к ПЕРСПЕКТИВЕ (наклон экрана) и к
//!   рассогласованию МАСШТАБА (фурье-корреляция масштабно НЕ инвариантна);
//! * цена в мс на кадр 1920×1080 × число ступеней масштабной лестницы;
//! * цена в площади полезной нагрузки.
//!
//! # Устройство
//!
//! Своё БПФ смешанного основания (множители 2/3/5 — 1920 и 1080 оба 5-гладкие,
//! так что кадр преобразуется БЕЗ дополнения), нормированная взаимная корреляция
//! через интегральные изображения (знаменатель NCC берётся за O(1) на сдвиг) и
//! свой рендерер сцены с гомографией наклона. Внешних зависимостей нет.

use crate::rng::Rng;
use psicode_core::zcborder::{self, BorderSpec, Carrier, RING, V1_ROOTS};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// комплексное число и БПФ смешанного основания
// ---------------------------------------------------------------------------

/// Комплексный отсчёт БПФ.
#[derive(Clone, Copy, Default, Debug)]
pub struct Cpx {
    pub re: f64,
    pub im: f64,
}

impl Cpx {
    #[inline]
    const fn new(re: f64, im: f64) -> Self {
        Cpx { re, im }
    }
    const ZERO: Cpx = Cpx { re: 0.0, im: 0.0 };
    #[inline]
    fn mul(self, o: Cpx) -> Cpx {
        Cpx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    /// `self · conj(o)`.
    #[inline]
    fn mul_conj(self, o: Cpx) -> Cpx {
        Cpx::new(self.re * o.re + self.im * o.im, self.im * o.re - self.re * o.im)
    }
    #[inline]
    fn add(self, o: Cpx) -> Cpx {
        Cpx::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    fn conj(self) -> Cpx {
        Cpx::new(self.re, -self.im)
    }
    #[inline]
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    #[inline]
    fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Максимальное простое основание, которое обрабатывает бабочка.
const MAX_RADIX: usize = 8;

/// План одномерного БПФ длины `n`: таблица поворотных множителей `W_n^k`.
struct Plan {
    n: usize,
    tw: Vec<Cpx>,
}

impl Plan {
    /// Построить план. `n` обязан раскладываться на простые ≤ [`MAX_RADIX`].
    fn new(n: usize) -> Self {
        assert!(n > 0);
        // проверка гладкости
        let mut m = n;
        for p in [2usize, 3, 5, 7] {
            while m % p == 0 {
                m /= p;
            }
        }
        assert_eq!(m, 1, "длина БПФ {n} не 7-гладкая");
        let mut tw = Vec::with_capacity(n);
        for k in 0..n {
            let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            tw.push(Cpx::new(a.cos(), a.sin()));
        }
        Plan { n, tw }
    }

    /// Прямое БПФ: `out` длины `n`, `inp` читается с шагом `stride`.
    /// `inp` и `out` не должны пересекаться.
    fn run(&self, inp: &[Cpx], stride: usize, out: &mut [Cpx]) {
        self.rec(inp, stride, self.n, out);
    }

    fn rec(&self, inp: &[Cpx], stride: usize, n: usize, out: &mut [Cpx]) {
        if n == 1 {
            out[0] = inp[0];
            return;
        }
        let r = smallest_factor(n);
        let m = n / r;
        for q in 0..r {
            self.rec(&inp[q * stride..], stride * r, m, &mut out[q * m..q * m + m]);
        }
        // шаг «бабочки» строго на месте: для фиксированного k множество
        // индексов {q·m+k} совпадает с множеством {j·m+k}
        let step_n = self.n / n; // W_n^t = tw[t · N/n]
        let step_r = self.n / r; // W_r^t = tw[t · N/r]
        let mut a = [Cpx::ZERO; MAX_RADIX];
        for k in 0..m {
            for q in 0..r {
                let t = (q * k) % n;
                a[q] = out[q * m + k].mul(self.tw[t * step_n]);
            }
            for j in 0..r {
                let mut s = Cpx::ZERO;
                for q in 0..r {
                    let t = (j * q) % r;
                    s = s.add(a[q].mul(self.tw[t * step_r]));
                }
                out[j * m + k] = s;
            }
        }
    }
}

fn smallest_factor(n: usize) -> usize {
    for p in [2usize, 3, 5, 7] {
        if n % p == 0 {
            return p;
        }
    }
    n
}

/// Двумерное БПФ `w × h` (row-major), многопоточное.
pub struct Fft2 {
    w: usize,
    h: usize,
    pw: Plan,
    ph: Plan,
    threads: usize,
}

impl Fft2 {
    fn new(w: usize, h: usize, threads: usize) -> Self {
        Fft2 { w, h, pw: Plan::new(w), ph: Plan::new(h), threads: threads.max(1) }
    }

    /// Пакет одномерных БПФ по строкам буфера `w × rows`, на месте.
    fn rows_pass(plan: &Plan, data: &mut [Cpx], w: usize, threads: usize) {
        let chunk = (data.len() / w).div_ceil(threads).max(1);
        std::thread::scope(|s| {
            for part in data.chunks_mut(chunk * w) {
                s.spawn(move || {
                    let mut scratch = vec![Cpx::ZERO; w];
                    for row in part.chunks_mut(w) {
                        scratch.copy_from_slice(row);
                        plan.run(&scratch, 1, row);
                    }
                });
            }
        });
    }

    fn transpose(src: &[Cpx], w: usize, h: usize, dst: &mut [Cpx]) {
        const B: usize = 32;
        let mut y0 = 0;
        while y0 < h {
            let y1 = (y0 + B).min(h);
            let mut x0 = 0;
            while x0 < w {
                let x1 = (x0 + B).min(w);
                for y in y0..y1 {
                    for x in x0..x1 {
                        dst[x * h + y] = src[y * w + x];
                    }
                }
                x0 = x1;
            }
            y0 = y1;
        }
    }

    /// Прямое двумерное БПФ на месте. `scratch` — буфер того же размера.
    fn forward(&self, data: &mut [Cpx], scratch: &mut [Cpx]) {
        Self::rows_pass(&self.pw, data, self.w, self.threads);
        Self::transpose(data, self.w, self.h, scratch);
        Self::rows_pass(&self.ph, scratch, self.h, self.threads);
        Self::transpose(scratch, self.h, self.w, data);
    }

    /// Обратное двумерное БПФ на месте (с нормировкой 1/(w·h)).
    fn inverse(&self, data: &mut [Cpx], scratch: &mut [Cpx]) {
        for v in data.iter_mut() {
            *v = v.conj();
        }
        self.forward(data, scratch);
        let s = 1.0 / (self.w * self.h) as f64;
        for v in data.iter_mut() {
            *v = Cpx::new(v.re * s, -v.im * s);
        }
    }
}

// ---------------------------------------------------------------------------
// плоскости и интегральные изображения
// ---------------------------------------------------------------------------

/// Вещественная плоскость (яркость или одна компонента цветности).
#[derive(Clone)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f64>,
}

impl Plane {
    fn new(w: usize, h: usize) -> Self {
        Plane { w, h, d: vec![0.0; w * h] }
    }
    #[inline]
    fn at(&self, x: usize, y: usize) -> f64 {
        self.d[y * self.w + x]
    }
}

/// Интегральное изображение (суммы и суммы квадратов) — знаменатель NCC за O(1).
struct Integral {
    w: usize,
    h: usize,
    s1: Vec<f64>,
    s2: Vec<f64>,
}

impl Integral {
    fn new(p: &Plane) -> Self {
        let (w, h) = (p.w, p.h);
        let mut s1 = vec![0.0; (w + 1) * (h + 1)];
        let mut s2 = vec![0.0; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut r1 = 0.0;
            let mut r2 = 0.0;
            for x in 0..w {
                let v = p.at(x, y);
                r1 += v;
                r2 += v * v;
                s1[(y + 1) * (w + 1) + x + 1] = s1[y * (w + 1) + x + 1] + r1;
                s2[(y + 1) * (w + 1) + x + 1] = s2[y * (w + 1) + x + 1] + r2;
            }
        }
        Integral { w, h, s1, s2 }
    }
    /// Сумма по прямоугольнику `[x, x+bw) × [y, y+bh)`; выход за край — 0.
    #[inline]
    fn box_sum(&self, x: usize, y: usize, bw: usize, bh: usize) -> (f64, f64) {
        let x1 = (x + bw).min(self.w);
        let y1 = (y + bh).min(self.h);
        let st = self.w + 1;
        let a = self.s1[y1 * st + x1] - self.s1[y * st + x1] - self.s1[y1 * st + x] + self.s1[y * st + x];
        let b = self.s2[y1 * st + x1] - self.s2[y * st + x1] - self.s2[y1 * st + x] + self.s2[y * st + x];
        (a, b)
    }
}

// ---------------------------------------------------------------------------
// шаблоны-кандидаты
// ---------------------------------------------------------------------------

/// Шумовой пол сенсора в долях полной шкалы: σ ≈ 2 градации из 255 по телеметрии
/// эталонного профиля (§7.4). Ниже него локальный контраст — это шум.
const NOISE_FLOOR: f64 = 2.0 / 255.0;

/// Форма маски шаблона: чем ограничена его опора.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mask {
    /// сплошной прямоугольник `tw × th`
    Box,
    /// кольцо: внешний прямоугольник минус внутренний, отступ `off` пикселей
    Ring { off: usize },
    /// уголок «Г»: полосы шириной `off` вдоль ВЕРХНЕГО и ЛЕВОГО краёв
    Corner { off: usize },
}

/// Кандидат: тип знака.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cand {
    MArray,
    Chirp,
    Bullseye,
    /// вся рамка v1 как один жёсткий шаблон 61×61
    V1Border,
    /// УГОЛОК рамки v1: блок `n × n` из левого верхнего угла символа.
    ///
    /// Смысл: площадь нагрузки он не стоит (это уже существующая рамка), а
    /// жёсткость у него та же, что у центрального знака `n × n`, — то есть он
    /// проверяет, обязательно ли платить площадью за малость шаблона.
    V1Corner,
}

impl Cand {
    fn label(self) -> &'static str {
        match self {
            Cand::MArray => "m-array",
            Cand::Chirp => "chirp",
            Cand::Bullseye => "bullseye",
            Cand::V1Border => "v1border",
            Cand::V1Corner => "v1corner",
        }
    }
    /// Занимает ли кандидат клетки НАГРУЗКИ (уголок и полная рамка — нет).
    fn costs_payload(self) -> bool {
        !matches!(self, Cand::V1Border | Cand::V1Corner)
    }
}

/// Двумерный шаблон в ПИКСЕЛЯХ: значения (комплексные — для цветности),
/// нулевые вне маски, со снятым средним по маске.
struct Tmpl {
    tw: usize,
    th: usize,
    /// значения, `th·tw`, вне маски строго 0
    v: Vec<Cpx>,
    mask: Mask,
    /// число пикселей в маске
    m: usize,
    /// ‖v‖ = sqrt(Σ|v|²)
    norm: f64,
    complex: bool,
}

/// Клеточная карта кандидата: значения ±1 (вещественные) или комплексные ЗЧ,
/// `None` — клетка вне знака (для рамки — вся внутренность).
fn cand_cells(cand: Cand, n: usize, complex_border: bool) -> (usize, Vec<Option<Cpx>>) {
    match cand {
        Cand::MArray => (n, marray_cells(n)),
        Cand::Chirp => (n, chirp_cells(n)),
        Cand::Bullseye => (n, bullseye_cells(n)),
        Cand::V1Border => {
            let c = border_map(complex_border);
            (GRID, c)
        }
        Cand::V1Corner => {
            // левый верхний блок n × n рамки: две полосы глубины RING, буквой «Г»
            let c = border_map(complex_border);
            let mut out = Vec::with_capacity(n * n);
            for y in 0..n {
                for x in 0..n {
                    out.push(c[y * GRID + x]);
                }
            }
            (n, out)
        }
    }
}

/// Клеточная карта рамки v1 (61×61), `None` во внутренней области.
fn border_map(complex: bool) -> Vec<Option<Cpx>> {
    let spec = BorderSpec {
        n: GRID,
        roots: V1_ROOTS,
        carrier: if complex { Carrier::ComplexChroma } else { Carrier::BinaryLuma },
    };
    zcborder::render_cells(&spec)
        .into_iter()
        .map(|o| o.map(|(re, im)| Cpx::new(re, im)))
        .collect()
}

/// Период сдвигового регистра Фибоначчи длины `k` с маской отводов `taps`.
/// Возвращает 0, если состояние зациклилось раньше полного периода.
fn lfsr_period(k: usize, taps: u32) -> usize {
    let full = (1usize << k) - 1;
    let mut st: u32 = 1;
    for step in 1..=full {
        let fb = (st & taps).count_ones() & 1;
        st = (st >> 1) | (fb << (k - 1));
        if st == 1 {
            return step;
        }
    }
    0
}

/// m-последовательность длины `2^k − 1`.
///
/// Маска отводов не выписывается таблицей, а ИЩЕТСЯ: берётся наименьшая, дающая
/// ПОЛНЫЙ период `2^k − 1`. Это самопроверяющаяся конструкция — примитивность
/// полинома не постулируется, а подтверждается прогоном регистра.
fn m_sequence(k: usize) -> Vec<i8> {
    let full = (1usize << k) - 1;
    let mut taps = 0u32;
    for cand in (1u32 << (k - 1))..(1u32 << k) {
        if lfsr_period(k, cand) == full {
            taps = cand;
            break;
        }
    }
    assert_ne!(taps, 0, "не нашлось примитивной маски для k={k}");
    let mut st: u32 = 1;
    let mut out = Vec::with_capacity(full);
    for _ in 0..full {
        out.push(if st & 1 == 1 { 1i8 } else { -1 });
        let fb = (st & taps).count_ones() & 1;
        st = (st >> 1) | (fb << (k - 1));
    }
    debug_assert_eq!(st, 1);
    out
}

/// Пара взаимно простых сомножителей `n1·n2 = 2^k−1` с `min(n1,n2) ≥ need`.
fn folding_for(need: usize) -> (usize, usize, usize) {
    for &(k, n1, n2) in &[(6usize, 7usize, 9usize), (8, 15, 17), (10, 31, 33), (12, 63, 65)] {
        if n1.min(n2) >= need {
            return (k, n1, n2);
        }
    }
    (12, 63, 65)
}

/// M-массив: свёртка m-последовательности по Китайской теореме об остатках.
///
/// `a[i][j] = s[t]`, где `t ≡ i (mod n1)`, `t ≡ j (mod n2)`; при `gcd(n1,n2)=1`
/// это биекция, и массив наследует ИДЕАЛЬНУЮ периодическую автокорреляцию
/// последовательности. Окно `n × n` берётся с периодическим заворотом; период
/// выбирается так, чтобы он был НЕ МЕНЬШЕ окна — иначе в окне окажется точная
/// копия куска себя, и корреляция получит боковой лепесток размером с пик.
fn marray_cells(n: usize) -> Vec<Option<Cpx>> {
    let (k, n1, n2) = folding_for(n);
    let s = m_sequence(k);
    let len = s.len();
    let mut arr = vec![0i8; n1 * n2];
    for t in 0..len {
        arr[(t % n2) * n1 + (t % n1)] = s[t];
    }
    let mut out = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let v = arr[(y % n2) * n1 + (x % n1)] as f64;
            out.push(Some(Cpx::new(v, 0.0)));
        }
    }
    out
}

/// Бинаризованный двумерный чирп (зонная пластинка Френеля).
///
/// `sign cos(π·r²/n)`: границы зон при `r² = n(m+½)`, шаг зон у края
/// `dr = n/(2r) = 1` клетка при `r = n/2` — ровно Найквист по краю.
fn chirp_cells(n: usize) -> Vec<Option<Cpx>> {
    let c = (n as f64 - 1.0) * 0.5;
    let mut out = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let dx = x as f64 - c;
            let dy = y as f64 - c;
            let ph = std::f64::consts::PI * (dx * dx + dy * dy) / n as f64;
            out.push(Some(Cpx::new(if ph.cos() >= 0.0 { 1.0 } else { -1.0 }, 0.0)));
        }
    }
    out
}

/// «Бычий глаз» Aztec: концентрические КВАДРАТНЫЕ кольца шириной 1 клетка.
fn bullseye_cells(n: usize) -> Vec<Option<Cpx>> {
    let c = (n as f64 - 1.0) * 0.5;
    let mut out = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let r = (x as f64 - c).abs().max((y as f64 - c).abs());
            let ring = (r + 0.5).floor() as i64;
            out.push(Some(Cpx::new(if ring % 2 == 0 { 1.0 } else { -1.0 }, 0.0)));
        }
    }
    out
}

/// Отрендерить клеточную карту в пиксельный шаблон при `ppc` пикселей на клетку.
///
/// Значение пикселя — среднее по 3×3 подвыборке (то, что видит камера с
/// апертурой пикселя). Среднее по маске снимается: канал добавляет неизвестное
/// смещение, и постоянная составляющая иначе утекала бы в пик.
///
/// Маска задаётся АНАЛИТИЧЕСКИ (сплошной прямоугольник или кольцо глубины
/// `RING·ppc`), а не по факту попадания подвыборок. Это принципиально: знаменатель
/// NCC берётся из интегральных изображений по ТОЙ ЖЕ фигуре, и малейшее
/// расхождение опоры числителя и знаменателя ломает неравенство Коши–Буняковского
/// — тогда на плоских участках кадра знаменатель схлопывается и «NCC» вылетает
/// далеко за единицу. Клеточная выборка поэтому зажимается в сетку, а не
/// отбрасывается на краю.
fn make_tmpl(cand: Cand, n_cells: usize, ppc: f64, complex_border: bool) -> Tmpl {
    let (n, cells) = cand_cells(cand, n_cells, complex_border);
    let side = (n as f64 * ppc).round() as usize;
    let tw = side;
    let th = side;
    let ring_off = ((RING as f64 * ppc).round() as usize).clamp(1, side / 2);
    let mask = match cand {
        Cand::V1Border => Mask::Ring { off: ring_off },
        Cand::V1Corner => Mask::Corner { off: ring_off },
        _ => Mask::Box,
    };
    let mut v = vec![Cpx::ZERO; tw * th];
    let mut inmask = vec![false; tw * th];
    const SS: usize = 3;
    for py in 0..th {
        for px in 0..tw {
            let keep = match mask {
                Mask::Box => true,
                Mask::Ring { off } => px < off || py < off || px + off >= tw || py + off >= th,
                Mask::Corner { off } => px < off || py < off,
            };
            if !keep {
                continue;
            }
            inmask[py * tw + px] = true;
            let mut acc = Cpx::ZERO;
            let mut hits = 0usize;
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = (px as f64 + (sx as f64 + 0.5) / SS as f64) / ppc;
                    let fy = (py as f64 + (sy as f64 + 0.5) / SS as f64) / ppc;
                    let cx = (fx.floor() as isize).clamp(0, n as isize - 1) as usize;
                    let cy = (fy.floor() as isize).clamp(0, n as isize - 1) as usize;
                    if let Some(c) = cells[cy * n + cx] {
                        acc = acc.add(c);
                        hits += 1;
                    }
                }
            }
            if hits > 0 {
                let s = 1.0 / hits as f64;
                v[py * tw + px] = Cpx::new(acc.re * s, acc.im * s);
            }
        }
    }
    let m = inmask.iter().filter(|b| **b).count();
    // снять среднее по маске
    let mut mean = Cpx::ZERO;
    for i in 0..v.len() {
        if inmask[i] {
            mean = mean.add(v[i]);
        }
    }
    let inv = 1.0 / m as f64;
    let mean = Cpx::new(mean.re * inv, mean.im * inv);
    let mut norm2 = 0.0;
    for i in 0..v.len() {
        if inmask[i] {
            v[i] = Cpx::new(v[i].re - mean.re, v[i].im - mean.im);
            norm2 += v[i].norm2();
        } else {
            v[i] = Cpx::ZERO;
        }
    }
    Tmpl {
        tw,
        th,
        v,
        mask,
        m,
        norm: norm2.sqrt(),
        complex: complex_border && cand == Cand::V1Border,
    }
}

// ---------------------------------------------------------------------------
// поверхность нормированной взаимной корреляции
// ---------------------------------------------------------------------------

/// Кэш прямого БПФ изображения (переиспользуется всеми шаблонами и ступенями).
struct ImgSpec {
    f: Vec<Cpx>,
    integ: Integral,
    /// для комплексного случая — интеграл по второй компоненте
    integ_im: Option<Integral>,
    w: usize,
    h: usize,
}

fn img_spec(fft: &Fft2, re: &Plane, im: Option<&Plane>, scratch: &mut Vec<Cpx>) -> ImgSpec {
    let (w, h) = (re.w, re.h);
    let mut f: Vec<Cpx> = Vec::with_capacity(w * h);
    match im {
        Some(p) => {
            for i in 0..w * h {
                f.push(Cpx::new(re.d[i], p.d[i]));
            }
        }
        None => {
            for i in 0..w * h {
                f.push(Cpx::new(re.d[i], 0.0));
            }
        }
    }
    scratch.resize(w * h, Cpx::ZERO);
    fft.forward(&mut f, scratch);
    ImgSpec {
        f,
        integ: Integral::new(re),
        integ_im: im.map(Integral::new),
        w,
        h,
    }
}

/// Суммы `Σ`, `Σ|·|²` по опоре маски шаблона в позиции `(x, y)`.
#[inline]
fn mask_stats(spec: &ImgSpec, t: &Tmpl, x: usize, y: usize) -> (Cpx, f64) {
    let (o1, o2) = spec.integ.box_sum(x, y, t.tw, t.th);
    let (mut s1r, mut s2) = (o1, o2);
    let (mut s1i, _) = (0.0, 0.0);
    if let Some(ii) = &spec.integ_im {
        let (a, b) = ii.box_sum(x, y, t.tw, t.th);
        s1i = a;
        s2 += b;
    }
    // вычитаемая «дырка» внутри опоры: у кольца — внутренний прямоугольник,
    // у уголка — правый нижний прямоугольник
    let hole = match t.mask {
        Mask::Box => None,
        Mask::Ring { off } => {
            Some((off, off, t.tw.saturating_sub(2 * off), t.th.saturating_sub(2 * off)))
        }
        Mask::Corner { off } => {
            Some((off, off, t.tw.saturating_sub(off), t.th.saturating_sub(off)))
        }
    };
    if let Some((dx, dy, iw, ih)) = hole {
        let (a, b) = spec.integ.box_sum(x + dx, y + dy, iw, ih);
        s1r -= a;
        s2 -= b;
        if let Some(ii) = &spec.integ_im {
            let (a2, b2) = ii.box_sum(x + dx, y + dy, iw, ih);
            s1i -= a2;
            s2 -= b2;
        }
    }
    (Cpx::new(s1r, s1i), s2)
}

/// Поверхность NCC (циклическая корреляция; область достоверности — сдвиги,
/// при которых окно шаблона целиком внутри кадра).
struct Surface {
    w: usize,
    /// достоверная область: `0..vw × 0..vh`
    vw: usize,
    vh: usize,
    d: Vec<f32>,
}

fn ncc_surface(fft: &Fft2, spec: &ImgSpec, t: &Tmpl, scratch: &mut Vec<Cpx>) -> Surface {
    let (w, h) = (spec.w, spec.h);
    // шаблон в нулевом дополнении, с сопряжением: corr = IFFT(F · conj(T))
    let mut tf = vec![Cpx::ZERO; w * h];
    for y in 0..t.th.min(h) {
        for x in 0..t.tw.min(w) {
            tf[y * w + x] = t.v[y * t.tw + x];
        }
    }
    scratch.resize(w * h, Cpx::ZERO);
    fft.forward(&mut tf, scratch);
    for i in 0..w * h {
        tf[i] = spec.f[i].mul_conj(tf[i]);
    }
    fft.inverse(&mut tf, scratch);

    let vw = w.saturating_sub(t.tw) + 1;
    let vh = h.saturating_sub(t.th) + 1;
    let mut d = vec![0.0f32; w * h];
    let m = t.m as f64;
    // Порог дисперсии: участок, чьё СКО ниже шума сенсора, информации не несёт.
    // Без порога плоские (пересвеченные/чёрные) области кадра дают нулевой
    // знаменатель и бесконечный «NCC» — классическая болезнь нормированной
    // корреляции на реальных снимках.
    let var_floor = m * NOISE_FLOOR * NOISE_FLOOR;
    for y in 0..vh {
        for x in 0..vw {
            let (s1, s2) = mask_stats(spec, t, x, y);
            let var = (s2 - s1.norm2() / m).max(var_floor);
            let den = var.sqrt() * t.norm;
            let num = tf[y * w + x];
            let val = if den > 1e-12 {
                if t.complex {
                    num.abs() / den
                } else {
                    num.re / den
                }
            } else {
                0.0
            };
            d[y * w + x] = val as f32;
        }
    }
    let _ = h;
    Surface { w, vw, vh, d }
}

/// Итог по поверхности: пик, его позиция и уровень боковых лепестков.
#[derive(Clone, Copy, Debug, Default)]
struct Peak {
    val: f64,
    x: usize,
    y: usize,
    /// максимум ВНЕ исключающего окна вокруг пика
    side_max: f64,
    /// среднее и СКО боковых лепестков
    side_mean: f64,
    side_sd: f64,
}

impl Peak {
    /// PSR по максимуму: пик / сильнейший конкурент. Это то число, которое
    /// решает, можно ли доверять глобальному поиску без стадии проверки.
    fn psr_max(&self) -> f64 {
        if self.side_max > 1e-9 {
            self.val / self.side_max
        } else {
            f64::INFINITY
        }
    }
    /// Классический PSR в сигмах.
    fn psr_sigma(&self) -> f64 {
        if self.side_sd > 1e-12 {
            (self.val - self.side_mean) / self.side_sd
        } else {
            f64::INFINITY
        }
    }
}

/// Полуширина главного лепестка на уровне 0.5 от пика, в пикселях (максимум по
/// горизонтали и вертикали). Это и есть ЦЕНА ЛОКАЛИЗАЦИИ: чем шире лепесток, тем
/// хуже положение определяется даже при формально высокой корреляции.
fn mainlobe_half_width(s: &Surface, pk: &Peak) -> f64 {
    let half = pk.val * 0.5;
    let mut best = 0.0f64;
    for axis in 0..2 {
        for sgn in [-1isize, 1] {
            let mut r = 0usize;
            loop {
                r += 1;
                let (x, y) = if axis == 0 {
                    (pk.x as isize + sgn * r as isize, pk.y as isize)
                } else {
                    (pk.x as isize, pk.y as isize + sgn * r as isize)
                };
                if x < 0 || y < 0 || x as usize >= s.vw || y as usize >= s.vh || r > 400 {
                    break;
                }
                if (s.d[y as usize * s.w + x as usize].abs() as f64) < half {
                    break;
                }
            }
            best = best.max(r as f64);
        }
    }
    best
}

/// Ошибка положения: расстояние от ОЦЕНЁННОГО центра знака до истинного, px.
fn centre_err(pk: &Peak, t: &Tmpl, truth: (f64, f64)) -> f64 {
    let ex = pk.x as f64 + t.tw as f64 * 0.5;
    let ey = pk.y as f64 + t.th as f64 * 0.5;
    ((ex - truth.0).powi(2) + (ey - truth.1).powi(2)).sqrt()
}

/// Найти пик и статистику боковых лепестков; исключающее окно — `excl` пикселей.
fn analyse(s: &Surface, excl: usize) -> Peak {
    let mut best = f64::NEG_INFINITY;
    let (mut bx, mut by) = (0usize, 0usize);
    for y in 0..s.vh {
        for x in 0..s.vw {
            let v = s.d[y * s.w + x].abs() as f64;
            if v > best {
                best = v;
                bx = x;
                by = y;
            }
        }
    }
    let mut side_max = 0.0f64;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut cnt = 0usize;
    for y in 0..s.vh {
        for x in 0..s.vw {
            if x.abs_diff(bx) <= excl && y.abs_diff(by) <= excl {
                continue;
            }
            let v = s.d[y * s.w + x].abs() as f64;
            if v > side_max {
                side_max = v;
            }
            sum += v;
            sum2 += v * v;
            cnt += 1;
        }
    }
    let n = cnt.max(1) as f64;
    let mean = sum / n;
    let var = (sum2 / n - mean * mean).max(0.0);
    Peak { val: best, x: bx, y: by, side_max, side_mean: mean, side_sd: var.sqrt() }
}

/// Худшая ложная тревога по набору кадров и кадр, который её дал.
fn worst_fa<'a, I: Iterator<Item = &'a (String, ImgSpec)>>(
    fft: &Fft2,
    it: I,
    t: &Tmpl,
    scratch: &mut Vec<Cpx>,
) -> (f64, String) {
    let mut best = 0.0f64;
    let mut who = String::new();
    for (tag, sp) in it {
        let s = ncc_surface(fft, sp, t, scratch);
        let v = max_abs(&s);
        if v > best {
            best = v;
            who = tag.clone();
        }
    }
    (best, who)
}

/// Максимум |NCC| по достоверной области — уровень ЛОЖНОЙ ТРЕВОГИ на негативе.
fn max_abs(s: &Surface) -> f64 {
    let mut best = 0.0f64;
    for y in 0..s.vh {
        for x in 0..s.vw {
            let v = s.d[y * s.w + x].abs() as f64;
            if v > best {
                best = v;
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// рендерер сцены: символ 61×61 + гомография наклона + блюр + шум
// ---------------------------------------------------------------------------

/// Сторона символа в клетках (совпадает с `symbol::GRID`).
const GRID: usize = 61;

/// Параметры синтетической сцены.
#[derive(Clone, Copy)]
struct Scene {
    canvas: usize,
    ppc: f64,
    /// наклон плоскости экрана вокруг вертикальной оси, градусы
    tilt_deg: f64,
    /// наклон вокруг оси 45° (одновременно по обеим), градусы
    diag: bool,
    /// σ гауссова блюра в пикселях камеры
    blur: f64,
    /// σ аддитивного шума, доля полной шкалы
    noise: f64,
    seed: u64,
    /// уровни чёрного/белого символа в долях шкалы
    black: f64,
    white: f64,
}

impl Scene {
    fn clean(canvas: usize, ppc: f64) -> Self {
        Scene {
            canvas,
            ppc,
            tilt_deg: 0.0,
            diag: false,
            blur: 0.0,
            noise: 0.0,
            seed: 0xC0FFEE,
            black: 0.06,
            white: 0.94,
        }
    }
}

/// Клеточная карта символа: рамка v1 (в яркости — знак фазы) + случайная
/// нагрузка + центральный знак кандидата.
fn symbol_cells(cand: Cand, n_cells: usize, seed: u64) -> Vec<f64> {
    let mut g = vec![0.0f64; GRID * GRID];
    let mut rng = Rng::new(seed);
    for v in g.iter_mut() {
        *v = if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
    }
    // рамка v1 в яркостном носителе — она физически занимает эти клетки
    let spec = BorderSpec { n: GRID, roots: V1_ROOTS, carrier: Carrier::BinaryLuma };
    for (i, c) in zcborder::render_cells(&spec).into_iter().enumerate() {
        if let Some((re, _)) = c {
            g[i] = if re >= 0.0 { 1.0 } else { -1.0 };
        }
    }
    if cand.costs_payload() {
        let (n, cells) = cand_cells(cand, n_cells, false);
        let off = (GRID - n) / 2;
        for y in 0..n {
            for x in 0..n {
                if let Some(c) = cells[y * n + x] {
                    g[(off + y) * GRID + off + x] = if c.re >= 0.0 { 1.0 } else { -1.0 };
                }
            }
        }
    }
    g
}

/// Гомография «клетки символа → пиксели» для наклонённой плоскости.
///
/// Плоскость `z = 0`, символ центрирован в начале; поворот на `θ` вокруг оси Y
/// (или вокруг диагонали при `diag`), сдвиг на `d`, пинхол с `f = ppc·d`.
fn tilt_homography(s: &Scene) -> [[f64; 3]; 3] {
    let side = GRID as f64;
    let d = 3.0 * side; // типичное кадрирование: расстояние ≈ 3 ширины символа
    let f = s.ppc * d;
    let th = s.tilt_deg.to_radians();
    let (c, sn) = (th.cos(), th.sin());
    let cx = s.canvas as f64 * 0.5;
    let cy = s.canvas as f64 * 0.5;
    if !s.diag {
        // u = cx + f·cosθ·x/(d − sinθ·x), v = cy + f·y/(d − sinθ·x)
        [
            [f * c - cx * sn, 0.0, cx * d],
            [-cy * sn, f, cy * d],
            [-sn, 0.0, d],
        ]
    } else {
        // честный поворот плоскости вокруг оси (1,1,0)/√2 (формула Родрига):
        // X' = A·x + B·y, Y' = B·x + A·y, Z' = (y−x)·sinθ/√2
        let a = (1.0 + c) * 0.5;
        let b = (1.0 - c) * 0.5;
        let k = sn / std::f64::consts::SQRT_2;
        [
            [f * a - cx * k, f * b + cx * k, cx * d],
            [f * b - cy * k, f * a + cy * k, cy * d],
            [-k, k, d],
        ]
    }
}

fn h_apply(h: &[[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let w = h[2][0] * x + h[2][1] * y + h[2][2];
    ((h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w)
}

fn h_invert(h: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = h;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let id = 1.0 / det;
    let mut o = [[0.0; 3]; 3];
    o[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * id;
    o[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id;
    o[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id;
    o[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * id;
    o[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id;
    o[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id;
    o[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * id;
    o[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id;
    o[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id;
    o
}

/// Разделимый гауссов блюр по σ пикселей камеры.
fn blur(p: &mut Plane, sigma: f64) {
    if sigma <= 1e-6 {
        return;
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut k = Vec::with_capacity((2 * r + 1) as usize);
    let mut s = 0.0;
    for i in -r..=r {
        let v = (-(i * i) as f64 / (2.0 * sigma * sigma)).exp();
        k.push(v);
        s += v;
    }
    for v in k.iter_mut() {
        *v /= s;
    }
    let (w, h) = (p.w, p.h);
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let xx = (x as isize + j as isize - r).clamp(0, w as isize - 1) as usize;
                acc += p.d[y * w + xx] * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let yy = (y as isize + j as isize - r).clamp(0, h as isize - 1) as usize;
                acc += tmp[yy * w + x] * kv;
            }
            p.d[y * w + x] = acc;
        }
    }
}

/// Отрисовать сцену: символ в центре холста, вне символа — ровный фон.
/// Возвращает плоскость и ИСТИННЫЙ ЦЕНТР знака в пикселях. Именно центр, а не
/// угол: при рассогласовании масштаба корреляция совмещает знак с шаблоном по
/// ЦЕНТРУ, и «попал/не попал» осмысленно только в этих координатах.
fn render_scene(cand: Cand, n_cells: usize, s: &Scene) -> (Plane, (f64, f64)) {
    let cells = symbol_cells(cand, n_cells, s.seed ^ 0x5EED);
    let h = tilt_homography(s);
    let hi = h_invert(&h);
    let mut p = Plane::new(s.canvas, s.canvas);
    let bg = 0.5 * (s.black + s.white);
    let half = GRID as f64 * 0.5;
    const SS: usize = 2;
    for py in 0..s.canvas {
        for px in 0..s.canvas {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = px as f64 + (sx as f64 + 0.5) / SS as f64 - 0.5;
                    let fy = py as f64 + (sy as f64 + 0.5) / SS as f64 - 0.5;
                    let (mx, my) = h_apply(&hi, fx, fy);
                    let cx = mx + half;
                    let cy = my + half;
                    if cx < 0.0 || cy < 0.0 || cx >= GRID as f64 || cy >= GRID as f64 {
                        acc += bg;
                    } else {
                        let v = cells[cy as usize * GRID + cx as usize];
                        acc += if v >= 0.0 { s.white } else { s.black };
                    }
                }
            }
            p.d[py * s.canvas + px] = acc / (SS * SS) as f64;
        }
    }
    blur(&mut p, s.blur);
    if s.noise > 0.0 {
        let mut rng = Rng::new(s.seed ^ 0xA11CE);
        for v in p.d.iter_mut() {
            *v += rng.gaussian() * s.noise;
        }
    }
    let (cx, cy) = footprint_centre_cells(cand, n_cells);
    let (tx, ty) = h_apply(&h, cx - half, cy - half);
    (p, (tx, ty))
}

/// Центр опоры кандидата в КЛЕТКАХ символа. Центральные знаки и полная рамка
/// центрированы; уголок сидит в левом верхнем углу.
fn footprint_centre_cells(cand: Cand, n_cells: usize) -> (f64, f64) {
    match cand {
        Cand::V1Corner => (n_cells as f64 * 0.5, n_cells as f64 * 0.5),
        _ => (GRID as f64 * 0.5, GRID as f64 * 0.5),
    }
}

// ---------------------------------------------------------------------------
// загрузка реальных кадров
// ---------------------------------------------------------------------------

/// Три плоскости NV21-дампа: Y (полное разрешение) и U, V (половинное).
struct Frame {
    tag: String,
    y: Plane,
    u: Plane,
    v: Plane,
}

fn read_bytes(p: &Path) -> Option<Vec<u8>> {
    std::fs::read(p).ok()
}

/// Прочитать `dump<i>.{y,u,v,meta}`. Формат `.meta`:
/// `w h yRowStride uRowStride uPixStride vRowStride vPixStride ySize uSize vSize`.
fn load_frame(dir: &Path, i: usize) -> Option<Frame> {
    let base = |ext: &str| -> PathBuf { dir.join(format!("dump{i}.{ext}")) };
    let meta = std::fs::read_to_string(base("meta")).ok()?;
    let f: Vec<usize> = meta.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    if f.len() < 7 {
        return None;
    }
    let (w, h, ystr, ustr, upix) = (f[0], f[1], f[2], f[3], f[4]);
    let yb = read_bytes(&base("y"))?;
    let ub = read_bytes(&base("u"))?;
    let vb = read_bytes(&base("v"))?;
    let mut y = Plane::new(w, h);
    for r in 0..h {
        for c in 0..w {
            let idx = r * ystr + c;
            y.d[r * w + c] = *yb.get(idx).unwrap_or(&0) as f64 / 255.0;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let mut u = Plane::new(cw, ch);
    let mut v = Plane::new(cw, ch);
    for r in 0..ch {
        for c in 0..cw {
            let idx = r * ustr + c * upix;
            u.d[r * cw + c] = (*ub.get(idx).unwrap_or(&128) as f64 - 128.0) / 128.0;
            v.d[r * cw + c] = (*vb.get(idx).unwrap_or(&128) as f64 - 128.0) / 128.0;
        }
    }
    let tag = format!("{}/dump{i}", dir.file_name().and_then(|s| s.to_str()).unwrap_or("?"));
    Some(Frame { tag, y, u, v })
}

fn data_root() -> PathBuf {
    if let Ok(p) = std::env::var("PSICODE_FINDER_DATA") {
        return PathBuf::from(p);
    }
    PathBuf::from(
        r"C:\Users\Dmytro\AppData\Local\Temp\claude\C--Users-Dmytro-psicode\05d92c67-463a-48df-929c-278fa3508f02\scratchpad",
    )
}

/// Уровни чёрного и белого по перцентилям 5 % / 95 % в прямоугольнике кадра.
/// Это ФАКТИЧЕСКИЙ контраст, который камера видит на снятом символе.
fn luma_levels(p: &Plane, x0: usize, y0: usize, x1: usize, y1: usize) -> (f64, f64) {
    let mut hist = [0usize; 256];
    let mut n = 0usize;
    for y in y0..y1.min(p.h) {
        for x in x0..x1.min(p.w) {
            let b = (p.at(x, y) * 255.0).clamp(0.0, 255.0) as usize;
            hist[b] += 1;
            n += 1;
        }
    }
    if n == 0 {
        return (0.06, 0.94);
    }
    let pick = |q: f64| -> f64 {
        let target = (q * n as f64) as usize;
        let mut acc = 0usize;
        for (b, c) in hist.iter().enumerate() {
            acc += c;
            if acc >= target {
                return b as f64 / 255.0;
            }
        }
        1.0
    };
    (pick(0.05), pick(0.95))
}

/// Вклеить отрендеренный символ в РЕАЛЬНЫЙ кадр: единственный способ получить
/// позитив для центральных кандидатов, которых в захватах физически нет.
fn splice(
    frame: &Plane,
    cand: Cand,
    n_cells: usize,
    ppc: f64,
    at: (usize, usize),
    blur_sigma: f64,
    seed: u64,
    levels: (f64, f64),
    tilt_deg: f64,
) -> (Plane, (f64, f64)) {
    let side = (GRID as f64 * ppc).round() as usize;
    let mut sc = Scene::clean(side, ppc);
    sc.blur = blur_sigma;
    sc.seed = seed;
    sc.black = levels.0;
    sc.white = levels.1;
    sc.tilt_deg = tilt_deg;
    let (patch, centre) = render_scene(cand, n_cells, &sc);
    let mut out = frame.clone();
    let (ax, ay) = at;
    for py in 0..side.min(frame.h.saturating_sub(ay)) {
        for px in 0..side.min(frame.w.saturating_sub(ax)) {
            out.d[(ay + py) * frame.w + ax + px] = patch.d[py * side + px];
        }
    }
    (out, (ax as f64 + centre.0, ay as f64 + centre.1))
}

// ---------------------------------------------------------------------------
// Фурье–Меллин
// ---------------------------------------------------------------------------

/// Нижняя граница радиуса лог-полярной развёртки, отсчётов спектра.
///
/// Не 1, а 4: при логарифмическом шаге половина строк развёртки приходилась бы на
/// `r ∈ [1, 8]` — восемь отсчётов спектра вокруг постоянной составляющей,
/// размноженные на сотню строк. Этот блок доминирует, и фазовая корреляция
/// намертво садится в нулевой сдвиг независимо от реального масштаба и поворота.
const LP_RMIN: f64 = 4.0;

/// Шаг лог-полярной радиальной оси: `ln(rmax/rmin)/n` на строку.
fn lp_step(n: usize) -> f64 {
    ((n as f64 * 0.5) / LP_RMIN).ln() / n as f64
}

/// Лог-полярная развёртка спектра (fftshift внутри), `n × n` → `n × n`.
///
/// Спектр предварительно проходит высокочастотный акцент Редди–Чаттерджи
/// `H = (1−X)(2−X)`, `X = cos πξ · cos πη`: без него медленный радиальный спад
/// `|F|` — самая заметная структура развёртки, и корреляция цепляется за него,
/// а не за содержимое знака.
fn log_polar(mag: &[f64], n: usize) -> Vec<f64> {
    let c = n as f64 * 0.5;
    let rmax = c;
    let step = lp_step(n);
    let mut out = vec![0.0; n * n];
    for ri in 0..n {
        let r = LP_RMIN * (step * ri as f64).exp();
        if r > rmax {
            continue;
        }
        for ti in 0..n {
            let th = std::f64::consts::PI * ti as f64 / n as f64; // спектр симметричен
            let dx = r * th.cos();
            let dy = r * th.sin();
            let x = c + dx;
            let y = c + dy;
            // fftshift: спектр лежит с нулём в углу
            let sx = ((x as isize + (n / 2) as isize) as usize) % n;
            let sy = ((y as isize + (n / 2) as isize) as usize) % n;
            let xi = dx / n as f64;
            let eta = dy / n as f64;
            let xx = (std::f64::consts::PI * xi).cos() * (std::f64::consts::PI * eta).cos();
            out[ri * n + ti] = mag[sy * n + sx] * (1.0 - xx) * (2.0 - xx);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// команда
// ---------------------------------------------------------------------------

const SIZES: [usize; 4] = [9, 15, 21, 25];
const CENTRAL: [Cand; 3] = [Cand::MArray, Cand::Chirp, Cand::Bullseye];
/// Опорный масштаб реального захвата v1b (px камеры на клетку).
const PPC_REF: f64 = 10.3;
/// Измеренный σ дефокуса живого канала, px камеры.
const BLUR_REF: f64 = 2.0;

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}

fn f2(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.2}")
    } else {
        "inf".into()
    }
}

/// Доля площади нагрузки, съедаемая знаком `n × n`.
/// Полезных клеток в символе v1: `61² − (8·61 − 16)` = 3721 − 472 = 3249
/// (рамка v1 — четыре полосы 61×2, углы посчитаны один раз).
fn payload_cost(n: usize) -> f64 {
    let payload = GRID * GRID - (8 * GRID - 16);
    n as f64 * n as f64 / payload as f64 * 100.0
}

pub fn cmd_finder() {
    let t0 = Instant::now();
    let th = threads();
    println!("# finder — центральный опознавательный знак против ЗЧ-рамки");
    println!(
        "\nПотоков {th}. Все корреляции — нормированные (NCC), знаменатель через интегральные\n\
         изображения; БПФ смешанного основания 2/3/5, кадр 1920×1080 преобразуется БЕЗ дополнения\n\
         (циклическая корреляция; достоверная область — сдвиги, где шаблон целиком в кадре).\n\
         Опорная точка канала: {PPC_REF} px/клетку, σ_блюра {BLUR_REF} px."
    );

    // необязательный аргумент — прогнать ОДИН раздел (развёртки долгие)
    let only = std::env::args().nth(2);
    let want = |k: &str| only.is_none() || only.as_deref() == Some(k);

    if want("0") {
        sanity_gate(th);
    }
    if want("1") {
        blur_sweep(th);
    }
    if want("2") {
        perspective_sweep(th);
    }
    if want("3") {
        scale_sweep(th);
    }
    if want("4") {
        real_frames(th);
    }
    if want("5") {
        cost_and_ladder(th);
    }
    if want("6") {
        fm_verdict(th);
    }

    println!("\nвсего {:.1} c", t0.elapsed().as_secs_f64());
}

// --- 0. sanity gate ---------------------------------------------------------

fn sanity_gate(th: usize) {
    println!("\n## 0. Sanity gate — без блюра, без шума, без наклона, масштаб точный");
    println!("\nПик обязан лечь в известную позицию. Невязка — в пикселях камеры.\n");
    println!(
        "| кандидат | площадь нагрузки | пик NCC | невязка, px | PSR(max) | PSR(σ) | полуширина лепестка, клеток |"
    );
    println!("|---|---|---|---|---|---|---|");
    let canvas = 1024;
    let fft = Fft2::new(canvas, canvas, th);
    let mut scratch = Vec::new();
    for (cand, n) in all_rows() {
        let sc = Scene::clean(canvas, PPC_REF);
        let (p, truth) = render_scene(cand, n, &sc);
        let spec = img_spec(&fft, &p, None, &mut scratch);
        let t = make_tmpl(cand, n, PPC_REF, false);
        let s = ncc_surface(&fft, &spec, &t, &mut scratch);
        let pk = analyse(&s, excl_for(cand, &t));
        let err = centre_err(&pk, &t, truth);
        println!(
            "| {} | {:.1} % | {:.3} | {:.2} | {} | {} | {:.2} |",
            row_label(cand, n),
            if cand.costs_payload() { payload_cost(n) } else { 0.0 },
            pk.val,
            err,
            f2(pk.psr_max()),
            f2(pk.psr_sigma()),
            mainlobe_half_width(&s, &pk) / PPC_REF
        );
    }

    // отдельно: боковые лепестки полой рамки при сдвиге НА СТОРОНУ
    println!("\n### 0.1 Профиль боковых лепестков рамки вдоль строки пика");
    println!(
        "\nПолый квадрат коррелирует с собой, сдвинутым на сторону: ожидались лепестки на\n\
         ±одной стороне (±61 клетка). Холст берётся 1536 px, чтобы такие сдвиги вообще попадали\n\
         в достоверную область. |NCC| на горизонтальном срезе через пик, в единицах пика.\n"
    );
    let big = 1536usize;
    let fftb = Fft2::new(big, big, th);
    let scb = Scene::clean(big, PPC_REF);
    let (pb, _) = render_scene(Cand::V1Border, 0, &scb);
    let specb = img_spec(&fftb, &pb, None, &mut scratch);
    let tb = make_tmpl(Cand::V1Border, 0, PPC_REF, false);
    let sb = ncc_surface(&fftb, &specb, &tb, &mut scratch);
    let pkb = analyse(&sb, (tb.tw / 4).max(8));
    let shifts = [0.0f64, 2.0, 4.0, 8.0, 16.0, 32.0, 57.0, 59.0, 61.0, 63.0];
    print!("| сдвиг, клеток |");
    for d in shifts {
        print!(" {d:.0} |");
    }
    println!();
    print!("|---|");
    for _ in shifts {
        print!("---|");
    }
    println!();
    print!("| \\|NCC\\|/пик |");
    for &dc in shifts.iter() {
        let dx = (dc * PPC_REF).round() as isize;
        let mut m = 0.0f64;
        for sgn in [-1isize, 1] {
            let x = pkb.x as isize + sgn * dx;
            if x >= 0 && (x as usize) < sb.vw {
                m = m.max(sb.d[pkb.y * sb.w + x as usize].abs() as f64);
            }
        }
        print!(" {:.3} |", m / pkb.val);
    }
    println!();
    println!(
        "\n(В достоверную область {} px влезают сдвиги до ±{:.0} клеток.)",
        sb.vw,
        (sb.vw as f64 - 1.0) / PPC_REF
    );
}

// --- 1. блюр ----------------------------------------------------------------

fn blur_sweep(th: usize) {
    println!("\n## 1. Устойчивость к блюру (σ в px камеры, {PPC_REF} px/клетку)");
    println!(
        "\nШаблон — БЕЗ блюра (приёмник не знает σ). В ячейке: пик NCC / PSR(max).\n\
         «×» — пик уехал больше чем на клетку от истины.\n"
    );
    let sigmas = [0.0f64, 1.0, 2.0, 3.0, 4.0, 6.0];
    print!("| кандидат / n |");
    for s in sigmas {
        print!(" σ={s} |");
    }
    println!();
    print!("|---|");
    for _ in sigmas {
        print!("---|");
    }
    println!();
    let canvas = 1024;
    let fft = Fft2::new(canvas, canvas, th);
    let mut scratch = Vec::new();
    for (cand, n) in all_rows() {
        let t = make_tmpl(cand, n, PPC_REF, false);
        let excl = excl_for(cand, &t);
        print!("| {} |", row_label(cand, n));
        for &sg in sigmas.iter() {
            let mut sc = Scene::clean(canvas, PPC_REF);
            sc.blur = sg;
            sc.noise = 0.01;
            let (p, truth) = render_scene(cand, n, &sc);
            let spec = img_spec(&fft, &p, None, &mut scratch);
            let s = ncc_surface(&fft, &spec, &t, &mut scratch);
            let pk = analyse(&s, excl);
            let err = centre_err(&pk, &t, truth);
            let bad = if err > PPC_REF { "×" } else { "" };
            print!(" {:.2}/{}{bad} |", pk.val, f2(pk.psr_max()));
        }
        println!();
    }
}

// --- 2. перспектива ---------------------------------------------------------

/// Кейстон: отношение px/клетку у БЛИЖНЕГО края к ДАЛЬНЕМУ при данном наклоне.
fn keystone(sc: &Scene) -> f64 {
    let h = tilt_homography(sc);
    let half = GRID as f64 * 0.5;
    let (x0, _) = h_apply(&h, -half, 0.0);
    let (x1, _) = h_apply(&h, -half + 1.0, 0.0);
    let (x2, _) = h_apply(&h, half - 1.0, 0.0);
    let (x3, _) = h_apply(&h, half, 0.0);
    let near = (x1 - x0).abs();
    let far = (x3 - x2).abs();
    if far > 1e-9 {
        near.max(far) / near.min(far)
    } else {
        f64::INFINITY
    }
}

/// Строки таблиц: все центральные кандидаты по размерам плюс рамка.
fn all_rows() -> Vec<(Cand, usize)> {
    let mut rows: Vec<(Cand, usize)> = Vec::new();
    for &c in CENTRAL.iter() {
        for &n in SIZES.iter() {
            rows.push((c, n));
        }
    }
    for &n in SIZES.iter() {
        rows.push((Cand::V1Corner, n));
    }
    rows.push((Cand::V1Border, 0));
    rows
}

fn row_label(cand: Cand, n: usize) -> String {
    if cand == Cand::V1Border {
        "v1border 61×61".to_string()
    } else {
        format!("{} {n}×{n}", cand.label())
    }
}

/// Исключающее окно вокруг пика при подсчёте боковых лепестков.
fn excl_for(cand: Cand, t: &Tmpl) -> usize {
    if cand == Cand::V1Border {
        (t.tw / 4).max(8)
    } else {
        t.tw / 2
    }
}

/// Лучшая гипотеза по ЛЕСТНИЦЕ МАСШТАБОВ: приёмник берёт максимум пика по
/// ступеням, поэтому и здесь берётся максимум, а не «лучшая правильная».
fn best_over_ladder(
    fft: &Fft2,
    spec: &ImgSpec,
    cand: Cand,
    n: usize,
    ks: &[f64],
    truth: (f64, f64),
    scratch: &mut Vec<Cpx>,
    limit: usize,
) -> (f64, f64, f64, f64) {
    let mut best = (f64::NEG_INFINITY, 0.0, f64::INFINITY, 1.0);
    for &k in ks {
        let t = make_tmpl(cand, n, PPC_REF * k, false);
        if t.tw >= limit || t.th >= limit {
            continue;
        }
        let s = ncc_surface(fft, spec, &t, scratch);
        let pk = analyse(&s, excl_for(cand, &t));
        if pk.val > best.0 {
            best = (pk.val, pk.psr_max(), centre_err(&pk, &t, truth), k);
        }
    }
    best
}

fn perspective_sweep(th: usize) {
    println!("\n## 2. Устойчивость к ПЕРСПЕКТИВЕ — главное число");
    println!(
        "\nНаклон плоскости экрана вокруг вертикальной оси, расстояние = 3 ширины символа.\n\
         Приёмнику дана ЛЕСТНИЦА МАСШТАБОВ (×0.60..×1.10, шаг 5 %) — она всё равно нужна, и без\n\
         неё сравнение нечестно: наклон укорачивает символ по одной оси, и это ракурсное\n\
         сжатие лестница отрабатывает. Остаётся ЧИСТЫЙ кейстон, который фронтальным шаблоном\n\
         не берётся В ПРИНЦИПЕ. В ячейке: пик NCC / PSR(max), «×» — промах центра > 1 клетки.\n\
         σ_блюра = {BLUR_REF}, шум 1 %.\n"
    );
    let tilts = [0.0f64, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0];
    let canvas = 1024;
    let ks: Vec<f64> = (0..11).map(|i| 0.60 + 0.05 * i as f64).collect();
    print!("| кейстон (ближ/дальн px/клетку) |");
    for t in tilts {
        let mut sc = Scene::clean(canvas, PPC_REF);
        sc.tilt_deg = t;
        print!(" {:.2} |", keystone(&sc));
    }
    println!();
    print!("| кандидат / n |");
    for t in tilts {
        print!(" {t}° |");
    }
    println!();
    print!("|---|");
    for _ in tilts {
        print!("---|");
    }
    println!();
    let fft = Fft2::new(canvas, canvas, th);
    let mut scratch = Vec::new();
    let mut breakdown: Vec<(String, Option<f64>, Option<f64>)> = Vec::new();
    for (cand, n) in all_rows() {
        let lab = row_label(cand, n);
        print!("| {lab} |");
        let mut brk = None;
        let mut brk_ks = None;
        for &tl in tilts.iter() {
            let mut sc = Scene::clean(canvas, PPC_REF);
            sc.tilt_deg = tl;
            sc.blur = BLUR_REF;
            sc.noise = 0.01;
            let (p, truth) = render_scene(cand, n, &sc);
            let spec = img_spec(&fft, &p, None, &mut scratch);
            let (val, psr, err, _k) =
                best_over_ladder(&fft, &spec, cand, n, &ks, truth, &mut scratch, canvas);
            let fail = err > PPC_REF || psr < 1.5;
            if fail && brk.is_none() {
                brk = Some(tl);
                brk_ks = Some(keystone(&sc));
            }
            print!(" {:.2}/{}{} |", val, f2(psr), if fail { "×" } else { "" });
        }
        println!();
        breakdown.push((lab, brk, brk_ks));
    }
    println!("\n### 2.1 Точка слома по перспективе (первый наклон с промахом > 1 клетки или PSR < 1.5)");
    println!("\n| кандидат | слом при | кейстон в точке слома |");
    println!("|---|---|---|");
    for (lab, b, ks) in breakdown {
        match b {
            Some(v) => println!("| {lab} | {v}° | {:.3} |", ks.unwrap_or(f64::NAN)),
            None => println!("| {lab} | > 50° | — |"),
        }
    }

    println!("\n### 2.2 Диагональный наклон (одновременно по обеим осям), та же лестница");
    println!("\n| кандидат | 10° | 20° | 30° | 40° |");
    println!("|---|---|---|---|---|");
    let mut rows2: Vec<(Cand, usize)> = CENTRAL.iter().map(|&c| (c, 15usize)).collect();
    rows2.push((Cand::V1Border, 0));
    for (cand, n) in rows2 {
        print!("| {} |", row_label(cand, n));
        for tl in [10.0f64, 20.0, 30.0, 40.0] {
            let mut sc = Scene::clean(canvas, PPC_REF);
            sc.tilt_deg = tl;
            sc.diag = true;
            sc.blur = BLUR_REF;
            sc.noise = 0.01;
            let (p, truth) = render_scene(cand, n, &sc);
            let spec = img_spec(&fft, &p, None, &mut scratch);
            let (val, psr, err, _k) =
                best_over_ladder(&fft, &spec, cand, n, &ks, truth, &mut scratch, canvas);
            print!(" {:.2}/{}{} |", val, f2(psr), if err > PPC_REF || psr < 1.5 { "×" } else { "" });
        }
        println!();
    }

    println!("\n### 2.3 То же БЕЗ лестницы (один фронтальный шаблон номинального масштаба)");
    println!(
        "\nРазница между 2 и 2.3 — ровно вклад ракурсного СЖАТИЯ, который лестница отрабатывает.\n"
    );
    println!("| кандидат | 5° | 10° | 15° | 20° | 30° |");
    println!("|---|---|---|---|---|---|");
    let mut rows3: Vec<(Cand, usize)> = CENTRAL.iter().map(|&c| (c, 15usize)).collect();
    rows3.push((Cand::V1Border, 0));
    for (cand, n) in rows3 {
        let t = make_tmpl(cand, n, PPC_REF, false);
        print!("| {} |", row_label(cand, n));
        for tl in [5.0f64, 10.0, 15.0, 20.0, 30.0] {
            let mut sc = Scene::clean(canvas, PPC_REF);
            sc.tilt_deg = tl;
            sc.blur = BLUR_REF;
            sc.noise = 0.01;
            let (p, truth) = render_scene(cand, n, &sc);
            let spec = img_spec(&fft, &p, None, &mut scratch);
            let s = ncc_surface(&fft, &spec, &t, &mut scratch);
            let pk = analyse(&s, excl_for(cand, &t));
            let err = centre_err(&pk, &t, truth);
            print!(
                " {:.2}/{}{} |",
                pk.val,
                f2(pk.psr_max()),
                if err > PPC_REF || pk.psr_max() < 1.5 { "×" } else { "" }
            );
        }
        println!();
    }
}

// --- 3. масштаб -------------------------------------------------------------

/// Рабочий диапазон масштабов приёмника: px камеры на клетку от 4 (предел MTF)
/// до 24 (символ во весь кадр). Отношение 6.0 — его и надо покрыть лестницей.
const LADDER_RANGE: f64 = 6.0;

fn scale_sweep(th: usize) {
    println!("\n## 3. Чувствительность к МАСШТАБУ — сколько ступеней нужно лестнице");
    println!(
        "\nСцена в {PPC_REF} px/клетку, шаблон — в `k·{PPC_REF}`; сетка геометрическая, шаг 2.5 %.\n\
         Фурье-корреляция масштабно НЕ инвариантна, поэтому ширина СВЯЗНОЙ зоны удержания вокруг\n\
         k = 1 задаёт шаг лестницы. Критерий удержания: промах центра ≤ 1 клетки и PSR ≥ 1.5.\n\
         σ = {BLUR_REF}, шум 1 %.\n"
    );
    let canvas = 1024;
    let fft = Fft2::new(canvas, canvas, th);
    let mut scratch = Vec::new();
    const J: i32 = 16;
    let kof = |j: i32| 1.025f64.powi(j);
    println!("| кандидат / n | зона удержания | ширина | ступеней на диапазон 6× | пик в центре зоны |");
    println!("|---|---|---|---|---|");
    let mut out: Vec<(String, f64, f64)> = Vec::new();
    for (cand, n) in all_rows() {
        let lab = row_label(cand, n);
        let mut sc = Scene::clean(canvas, PPC_REF);
        sc.blur = BLUR_REF;
        sc.noise = 0.01;
        let (p, truth) = render_scene(cand, n, &sc);
        let spec = img_spec(&fft, &p, None, &mut scratch);
        let probe = |k: f64, scratch: &mut Vec<Cpx>| -> (bool, f64) {
            let t = make_tmpl(cand, n, PPC_REF * k, false);
            if t.tw >= canvas || t.th >= canvas {
                return (false, 0.0);
            }
            let s = ncc_surface(&fft, &spec, &t, scratch);
            let pk = analyse(&s, excl_for(cand, &t));
            let err = centre_err(&pk, &t, truth);
            (err <= PPC_REF && pk.psr_max() >= 1.5, pk.val)
        };
        let (ok0, v0) = probe(1.0, &mut scratch);
        if !ok0 {
            println!("| {lab} | не держит даже точный масштаб | — | — | {v0:.2} |");
            continue;
        }
        // связная зона: расширяемся от k = 1 наружу, пока держит
        let mut jlo = 0;
        while jlo > -J && probe(kof(jlo - 1), &mut scratch).0 {
            jlo -= 1;
        }
        let mut jhi = 0;
        while jhi < J && probe(kof(jhi + 1), &mut scratch).0 {
            jhi += 1;
        }
        let (lo, hi) = (kof(jlo), kof(jhi));
        let width = hi / lo;
        let rungs = (LADDER_RANGE.ln() / width.ln()).ceil().max(1.0);
        println!(
            "| {lab} | ×{lo:.3}..×{hi:.3} | {width:.3} | {rungs:.0} | {v0:.2} |"
        );
        out.push((lab, width, rungs));
    }
    println!(
        "\nШирина зоны удержания падает как ~1/(сторона шаблона в клетках): при рассогласовании\n\
         масштаба на δ края шаблона размером `L` клеток уезжают на `δ·L/2` клетки, и разъезд в\n\
         пол-клетки уже съедает корреляцию. Отсюда и разрыв между знаком 15×15 и рамкой 61×61."
    );
    let _ = out;
}

// --- 4. реальные кадры ------------------------------------------------------

fn real_frames(th: usize) {
    println!("\n## 4. РЕАЛЬНЫЕ кадры 1920×1080 (Galaxy Note 10 Lite, NV21)");
    let root = data_root();
    let neg_dirs = ["v1live"];
    let mut negs: Vec<Frame> = Vec::new();
    for d in neg_dirs {
        for i in 0..4 {
            if let Some(f) = load_frame(&root.join(d), i) {
                negs.push(f);
            }
        }
    }
    let mut clutter: Vec<Frame> = Vec::new();
    for d in ["n1", "n16"] {
        if let Some(f) = load_frame(&root.join(d), 0) {
            clutter.push(f);
        }
    }
    let v1b: Vec<Frame> = (0..2).filter_map(|i| load_frame(&root.join("v1b"), i)).collect();
    if negs.is_empty() {
        println!("\n(кадры не найдены: {}; пропуск)", root.display());
        return;
    }
    println!(
        "\nНегативы: {} кадров v1live (символа НЕТ) + {} кадров с символами v0 (n1, n16) —\n\
         для центральных кандидатов это тоже чистое загромождение. Позитивы для центральных\n\
         кандидатов получены ВКЛЕЙКОЙ отрендеренного символа ({PPC_REF} px/клетку, σ={BLUR_REF})\n\
         в реальный кадр v1live — иначе позитива не существует, знака в захватах нет.\n",
        negs.len(),
        clutter.len()
    );

    let (w, h) = (negs[0].y.w, negs[0].y.h);
    let fft = Fft2::new(w, h, th);
    let mut scratch = Vec::new();
    let neg_specs: Vec<(String, ImgSpec)> =
        negs.iter().map(|f| (f.tag.clone(), img_spec(&fft, &f.y, None, &mut scratch))).collect();
    let clut_specs: Vec<(String, ImgSpec)> =
        clutter.iter().map(|f| (f.tag.clone(), img_spec(&fft, &f.y, None, &mut scratch))).collect();

    // контраст берётся из НАСТОЯЩЕГО снятого символа, а не выдумывается
    let levels = if let Some(f) = v1b.first() {
        luma_levels(&f.y, 390, 50, 1020, 640)
    } else {
        (0.06, 0.94)
    };
    println!(
        "Контраст вклейки взят из снятого символа v1b (перцентили 5/95 в области символа):\n\
         чёрное {:.3}, белое {:.3} полной шкалы. «Идеальная» вклейка — фронтальная, точный\n\
         масштаб; «реалистичная» — тот же контраст плюс наклон 12° и σ = 2.5.\n",
        levels.0, levels.1
    );

    println!(
        "| кандидат / n | пик (идеальная вклейка) | пик (реалистичная) | худшая ложная тревога | ЗАПАС идеал | ЗАПАС реалист |"
    );
    println!("|---|---|---|---|---|---|");
    for (cand, n) in all_rows() {
        let t = make_tmpl(cand, n, PPC_REF, false);
        let excl = excl_for(cand, &t);
        let lab = row_label(cand, n);
        // позитив: вклейка в кадр негатива №0
        let (spliced, truth) =
            splice(&negs[0].y, cand, n, PPC_REF, (900, 300), BLUR_REF, 0xBEEF, levels, 0.0);
        let sp = img_spec(&fft, &spliced, None, &mut scratch);
        let s = ncc_surface(&fft, &sp, &t, &mut scratch);
        let pk = analyse(&s, excl);
        let err = centre_err(&pk, &t, truth);
        // реалистичная вклейка: наклон + более сильный блюр, поиск по лестнице
        let (spl2, truth2) =
            splice(&negs[0].y, cand, n, PPC_REF, (900, 300), 2.5, 0xBEEF, levels, 12.0);
        let sp2 = img_spec(&fft, &spl2, None, &mut scratch);
        let ks: Vec<f64> = (0..7).map(|i| 0.85 + 0.05 * i as f64).collect();
        let (val2, _psr2, err2, _k2) =
            best_over_ladder(&fft, &sp2, cand, n, &ks, truth2, &mut scratch, w.min(h));
        // худшая ложная тревога по всем негативам
        let (fa, who) = worst_fa(&fft, neg_specs.iter().chain(clut_specs.iter()), &t, &mut scratch);
        println!(
            "| {lab} | {:.3}{} | {:.3}{} | {:.3} ({who}) | **{}** | **{}** |",
            pk.val,
            if err > PPC_REF { " (промах!)" } else { "" },
            val2,
            if err2 > PPC_REF { " (промах!)" } else { "" },
            fa,
            f2(pk.val / fa),
            f2(val2 / fa)
        );
    }

    // рамка на НАСТОЯЩЕМ символе v1
    if !v1b.is_empty() {
        println!("\n### 4.1 Рамка v1 на НАСТОЯЩЕМ снятом символе (v1b), по лестнице масштабов");
        println!("\n| носитель | лучший px/клетку | пик NCC | позиция пика | PSR(max) | худшая ЛТ на негативах | запас |");
        println!("|---|---|---|---|---|---|---|");
        for complex in [false, true] {
            let mut best = (0.0f64, 0.0f64, 0usize, 0usize, 0.0f64);
            for step in 0..9 {
                let ppc = 8.5 + step as f64 * 0.5;
                let t = make_tmpl(Cand::V1Border, 0, ppc, complex);
                let excl = (t.tw / 4).max(8);
                let (sp, _lab) = if complex {
                    let f = &v1b[0];
                    let fc = Fft2::new(f.u.w, f.u.h, th);
                    let spec = img_spec(&fc, &f.u, Some(&f.v), &mut scratch);
                    (Some((fc, spec)), ())
                } else {
                    (None, ())
                };
                let (pkv, pk) = match &sp {
                    Some((fc, spec)) => {
                        // цветность в половинном разрешении -> шаблон в половинном масштабе
                        let t2 = make_tmpl(Cand::V1Border, 0, ppc * 0.5, true);
                        if t2.tw >= spec.w || t2.th >= spec.h {
                            continue;
                        }
                        let s = ncc_surface(fc, spec, &t2, &mut scratch);
                        let p = analyse(&s, (t2.tw / 4).max(8));
                        (p.val, p)
                    }
                    None => {
                        let spec = img_spec(&fft, &v1b[0].y, None, &mut scratch);
                        let s = ncc_surface(&fft, &spec, &t, &mut scratch);
                        let p = analyse(&s, excl);
                        (p.val, p)
                    }
                };
                if pkv > best.0 {
                    best = (pkv, ppc, pk.x, pk.y, pk.psr_max());
                }
            }
            // ложная тревога того же шаблона на негативах
            let t = make_tmpl(Cand::V1Border, 0, if best.1 > 0.0 { best.1 } else { PPC_REF }, complex);
            let mut fa = 0.0f64;
            if complex {
                for f in negs.iter().chain(clutter.iter()) {
                    let fc = Fft2::new(f.u.w, f.u.h, th);
                    let spec = img_spec(&fc, &f.u, Some(&f.v), &mut scratch);
                    let t2 = make_tmpl(Cand::V1Border, 0, best.1.max(PPC_REF) * 0.5, true);
                    if t2.tw >= spec.w || t2.th >= spec.h {
                        continue;
                    }
                    let s = ncc_surface(&fc, &spec, &t2, &mut scratch);
                    fa = fa.max(max_abs(&s));
                }
            } else {
                fa = worst_fa(&fft, neg_specs.iter().chain(clut_specs.iter()), &t, &mut scratch).0;
            }
            println!(
                "| {} | {:.1} | {:.3} | ({}, {}) | {} | {:.3} | {} |",
                if complex { "цветность (U,V), 960×540" } else { "яркость Y, 1920×1080" },
                best.1,
                best.0,
                best.2,
                best.3,
                f2(best.4),
                fa,
                f2(best.0 / fa.max(1e-9))
            );
        }
    }

    // центральные кандидаты в ЦВЕТНОСТИ — проверка довода «загромождение живёт в яркости»
    println!("\n### 4.2 Уровень ложной тревоги в ЦВЕТНОСТИ против ЯРКОСТИ (только негативы)");
    println!(
        "\nДовод рамки v1: она невидима в Y, а загромождение экрана живёт в Y. Проверяем, даёт ли\n\
         это преимущество и центральному знаку. Цветность 960×540, шаблон в половинном масштабе.\n"
    );
    println!("| кандидат / n | ЛТ в яркости | ЛТ в цветности |");
    println!("|---|---|---|");
    for &n in [15usize, 25].iter() {
        for &cand in CENTRAL.iter() {
            let t = make_tmpl(cand, n, PPC_REF, false);
            let fy = worst_fa(&fft, neg_specs.iter().chain(clut_specs.iter()), &t, &mut scratch).0;
            let mut fc_max = 0.0f64;
            let tc = make_tmpl(cand, n, PPC_REF * 0.5, false);
            for f in negs.iter().chain(clutter.iter()) {
                let fc = Fft2::new(f.u.w, f.u.h, th);
                let spec = img_spec(&fc, &f.u, None, &mut scratch);
                if tc.tw >= spec.w {
                    continue;
                }
                let s = ncc_surface(&fc, &spec, &tc, &mut scratch);
                fc_max = fc_max.max(max_abs(&s));
            }
            println!("| {} {n}×{n} | {fy:.3} | {fc_max:.3} |", cand.label());
        }
    }

    // 4.3 прореживание — единственный рычаг, который реально сбивает цену БПФ
    println!("\n### 4.3 Прореживание кадра: сохраняется ли запас на 960×540 и 480×270");
    println!(
        "\nЗнак 15 клеток при {PPC_REF} px/клетку — это 155 px; искать его на полном кадре\n\
         расточительно. Прореживание в K раз режет цену БПФ в K² раз. Вопрос, до какого K\n\
         запас держится (2 px/клетку — предел Найквиста, при K=4 остаётся 2.6).\n"
    );
    println!("| кандидат / n | K | размер кадра | px/клетку | пик на вклейке | худшая ЛТ | ЗАПАС |");
    println!("|---|---|---|---|---|---|---|");
    for &(cand, n) in &[(Cand::MArray, 15usize), (Cand::MArray, 25), (Cand::Chirp, 15), (Cand::V1Border, 0)] {
        for k in [1usize, 2, 4] {
            let ppc = PPC_REF / k as f64;
            let t = make_tmpl(cand, n, ppc, false);
            let (spliced, truth) =
                splice(&negs[0].y, cand, n, PPC_REF, (900, 300), BLUR_REF, 0xBEEF, levels, 0.0);
            let sd = decimate(&spliced, k);
            let fk = Fft2::new(sd.w, sd.h, th);
            if t.tw >= sd.w || t.th >= sd.h {
                continue;
            }
            let sp = img_spec(&fk, &sd, None, &mut scratch);
            let s = ncc_surface(&fk, &sp, &t, &mut scratch);
            let pk = analyse(&s, excl_for(cand, &t));
            let err = centre_err(&pk, &t, (truth.0 / k as f64, truth.1 / k as f64)) * k as f64;
            let mut fa = 0.0f64;
            for f in negs.iter().chain(clutter.iter()) {
                let fd = decimate(&f.y, k);
                let spn = img_spec(&fk, &fd, None, &mut scratch);
                let s = ncc_surface(&fk, &spn, &t, &mut scratch);
                fa = fa.max(max_abs(&s));
            }
            println!(
                "| {} | {k} | {}×{} | {ppc:.2} | {:.3}{} | {fa:.3} | **{}** |",
                row_label(cand, n),
                sd.w,
                sd.h,
                pk.val,
                if err > PPC_REF { " (промах!)" } else { "" },
                f2(pk.val / fa.max(1e-9))
            );
        }
    }
}

/// Прореживание плоскости в `k` раз усреднением блока `k × k`.
fn decimate(p: &Plane, k: usize) -> Plane {
    if k <= 1 {
        return p.clone();
    }
    let (w, h) = (p.w / k, p.h / k);
    let mut o = Plane::new(w, h);
    let inv = 1.0 / (k * k) as f64;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for dy in 0..k {
                for dx in 0..k {
                    acc += p.at(x * k + dx, y * k + dy);
                }
            }
            o.d[y * w + x] = acc * inv;
        }
    }
    o
}

// --- 5. цена ----------------------------------------------------------------

fn cost_and_ladder(th: usize) {
    println!("\n## 5. Цена в мс: фурье-корреляция по кадру 1920×1080");
    let mut scratch: Vec<Cpx> = Vec::new();
    let mut rng = Rng::new(7);
    println!(
        "\nБПФ здесь СВОЁ — рекурсивное, смешанного основания, без специализированных бабочек\n\
         и без SIMD, на f64. Настроенная библиотека класса FFTW/pocketfft на тех же размерах\n\
         быстрее примерно на порядок, поэтому измеренные «мс» — ПОТОЛОК цены, а не пол; рядом\n\
         дана оценка по операциям (5·N·log₂N флопов на двумерное преобразование при 4 Гфлоп/с\n\
         на поток). Все выводы строятся на ОТНОШЕНИЯХ, от качества БПФ не зависящих.\n\
         Каждый замер — ЛУЧШИЙ из нескольких прогонов, вход перед каждым восстанавливается."
    );
    println!(
        "\n| размер кадра | прямое БПФ 2D, 1 поток | прямое, {th} потоков | обратное, {th} потоков | умножение спектров | интегральные изобр. | оценка для настроенного БПФ, {th} потоков |"
    );
    println!("|---|---|---|---|---|---|---|");
    let mut cost: Vec<(usize, usize, f64, f64, f64, f64, f64)> = Vec::new();
    for &(w, h) in &[(1920usize, 1080usize), (960, 540), (480, 270)] {
        let fft1 = Fft2::new(w, h, 1);
        let fftn = Fft2::new(w, h, th);
        scratch.resize(w * h, Cpx::ZERO);
        let mut pl = Plane::new(w, h);
        let mut data: Vec<Cpx> = Vec::with_capacity(w * h);
        for i in 0..w * h {
            let v = rng.next_f64();
            pl.d[i] = v;
            data.push(Cpx::new(v, 0.0));
        }
        let reps = if w > 1000 { 3 } else { 6 };
        let mut buf = data.clone();
        let mut bench = |f: &dyn Fn(&mut Vec<Cpx>, &mut Vec<Cpx>)| -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                buf.copy_from_slice(&data);
                let t = Instant::now();
                f(&mut buf, &mut scratch);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(&buf);
            }
            best
        };
        let ms_fwd1 = bench(&|d, s| fft1.forward(d, s));
        let ms_fwdn = bench(&|d, s| fftn.forward(d, s));
        let ms_invn = bench(&|d, s| fftn.inverse(d, s));
        let ms_mul = bench(&|d, _s| {
            for v in d.iter_mut() {
                *v = v.mul_conj(*v);
            }
        });
        let t = Instant::now();
        let integ = Integral::new(&pl);
        let ms_integ = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&integ);
        // 5·N·log2(N) флопов, 4 Гфлоп/с на поток, КПД распараллеливания 0.5
        let npts = (w * h) as f64;
        let ms_ideal = 5.0 * npts * npts.log2() / 4.0e9 * 1e3 / (th as f64 * 0.5);
        println!(
            "| {w}×{h} | {ms_fwd1:.0} мс | {ms_fwdn:.0} мс | {ms_invn:.0} мс | {ms_mul:.1} мс | {ms_integ:.0} мс | ≈{ms_ideal:.0} мс |"
        );
        cost.push((w, h, ms_fwdn, ms_invn, ms_mul, ms_integ, ms_ideal));
    }

    println!(
        "\nПрямое БПФ кадра считается ОДИН раз и переиспользуется ВСЕМИ ступенями лестницы;\n\
         на ступень приходится (умножение спектров + обратное БПФ). Шаблоны преобразуются\n\
         единожды при старте и кэшируются — в бюджет кадра не входят.\n\
         В скобках — та же сумма для настроенного БПФ (столбец «оценка»)."
    );
    println!("\n| ступеней лестницы | 1920×1080 | 960×540 (K=2) | 480×270 (K=4) |");
    println!("|---|---|---|---|");
    for k in [1usize, 2, 4, 8, 12, 20, 40] {
        print!("| {k} |");
        for c in cost.iter() {
            let total = c.2 + c.5 + (c.3 + c.4) * k as f64;
            let ideal = c.6 * (1.0 + k as f64) + c.5;
            print!(" {total:.0} (≈{ideal:.0}) мс |");
        }
        println!();
    }
    println!(
        "\nДля сравнения: нынешняя пространственная акквизиция — 157 мс (8 потоков, десктоп)."
    );
}

// --- 6. Фурье–Меллин --------------------------------------------------------

/// Окно для Фурье–Меллина: знак `n × n` в центре, повёрнутый на `rot_deg`,
/// вокруг — либо ровный фон, либо реальное соседство (случайная нагрузка символа).
fn fm_window(
    cand: Cand,
    n: usize,
    ppc: f64,
    rot_deg: f64,
    clutter: bool,
    canvas: usize,
    seed: u64,
) -> Plane {
    let (nn, cells) = cand_cells(cand, n, false);
    let mut payload = vec![0.0f64; GRID * GRID];
    let mut rng = Rng::new(seed);
    for v in payload.iter_mut() {
        *v = if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
    }
    let c = canvas as f64 * 0.5;
    let r = rot_deg.to_radians();
    let (co, si) = (r.cos(), r.sin());
    let mut p = Plane::new(canvas, canvas);
    for py in 0..canvas {
        for px in 0..canvas {
            let dx = px as f64 - c;
            let dy = py as f64 - c;
            let ux = (co * dx + si * dy) / ppc;
            let uy = (-si * dx + co * dy) / ppc;
            let fx = ux + nn as f64 * 0.5;
            let fy = uy + nn as f64 * 0.5;
            let v = if fx >= 0.0 && fy >= 0.0 && fx < nn as f64 && fy < nn as f64 {
                cells[fy as usize * nn + fx as usize].map(|c| c.re).unwrap_or(0.0)
            } else if clutter {
                let gx = ux + GRID as f64 * 0.5;
                let gy = uy + GRID as f64 * 0.5;
                if gx >= 0.0 && gy >= 0.0 && gx < GRID as f64 && gy < GRID as f64 {
                    payload[gy as usize * GRID + gx as usize]
                } else {
                    0.0
                }
            } else {
                0.0
            };
            p.d[py * canvas + px] = 0.5 + 0.44 * v;
        }
    }
    blur(&mut p, 1.0);
    p
}

/// Лог-полярный спектр окна: `|БПФ|` → лог-полярная развёртка → БПФ.
fn fm_spec(fft: &Fft2, win: &Plane, n: usize, scratch: &mut Vec<Cpx>) -> Vec<Cpx> {
    let mut a: Vec<Cpx> = win.d.iter().map(|&v| Cpx::new(v, 0.0)).collect();
    scratch.resize(n * n, Cpx::ZERO);
    fft.forward(&mut a, scratch);
    let mag: Vec<f64> = a.iter().map(|c| (1.0 + c.abs()).ln()).collect();
    let lp = log_polar(&mag, n);
    let mut lf: Vec<Cpx> = lp.iter().map(|&v| Cpx::new(v, 0.0)).collect();
    fft.forward(&mut lf, scratch);
    lf
}

/// Фазовая корреляция двух лог-полярных спектров: возвращает (сдвиг по радиусу,
/// сдвиг по углу, высота пика, PSR пика).
fn fm_match(fft: &Fft2, a: &[Cpx], b: &[Cpx], n: usize, scratch: &mut Vec<Cpx>) -> (f64, f64, f64, f64) {
    let mut c: Vec<Cpx> = (0..n * n)
        .map(|i| {
            let z = a[i].mul_conj(b[i]);
            let m = z.abs();
            if m > 1e-12 {
                Cpx::new(z.re / m, z.im / m)
            } else {
                Cpx::ZERO
            }
        })
        .collect();
    scratch.resize(n * n, Cpx::ZERO);
    fft.inverse(&mut c, scratch);
    let (mut best, mut bx, mut by) = (f64::NEG_INFINITY, 0usize, 0usize);
    for y in 0..n {
        for x in 0..n {
            let v = c[y * n + x].re;
            if v > best {
                best = v;
                bx = x;
                by = y;
            }
        }
    }
    let mut side = 0.0f64;
    for y in 0..n {
        for x in 0..n {
            if x.abs_diff(bx) <= 3 && y.abs_diff(by) <= 3 {
                continue;
            }
            side = side.max(c[y * n + x].re);
        }
    }
    // by — сдвиг по РАДИУСУ (лог-масштаб), bx — по УГЛУ
    let dr = if by > n / 2 { by as f64 - n as f64 } else { by as f64 };
    let dt = if bx > n / 2 { bx as f64 - n as f64 } else { bx as f64 };
    (dr, dt, best, if side > 1e-12 { best / side } else { f64::INFINITY })
}

fn fm_verdict(th: usize) {
    println!("\n## 6. Фурье–Меллин: приговор");
    println!(
        "\nФМ — не детектор, а РЕГИСТРАТОР пары изображений: `|БПФ|` убирает сдвиг, лог-полярная\n\
         развёртка превращает масштаб и поворот в сдвиг, вторая корреляция их достаёт. Чтобы\n\
         сделать из него детектор, окно надо СНАЧАЛА поставить туда, где знак — то есть решить\n\
         ровно ту задачу, ради которой он и привлекается. Поэтому его гоняют по сетке окон.\n\
         Ниже: окно 256×256, эталон — знак m-array 15×15 при 8.0 px/клетку в центре.\n"
    );
    let n = 256usize;
    let fft1 = Fft2::new(n, n, 1);
    let fftp = Fft2::new(n, n, th);
    let mut scratch = Vec::new();
    let ppc0 = 8.0f64;
    let refw = fm_window(Cand::MArray, 15, ppc0, 0.0, false, n, 1);
    let rspec = fm_spec(&fft1, &refw, n, &mut scratch);
    let lr = lp_step(n);

    println!("| окно | истинный масштаб | восстановлено | истинный поворот | восстановлено | PSR фазовой корр. |");
    println!("|---|---|---|---|---|---|");
    for &(k, rot, clut) in &[
        (1.0f64, 0.0f64, false),
        (1.0, 20.0, false),
        (1.25, 0.0, false),
        (1.25, 20.0, false),
        (1.0, 0.0, true),
        (1.25, 0.0, true),
        (1.25, 20.0, true),
    ] {
        let w = fm_window(Cand::MArray, 15, ppc0 * k, rot, clut, n, 1);
        let wspec = fm_spec(&fft1, &w, n, &mut scratch);
        let (dr, dt, _pk, psr) = fm_match(&fft1, &wspec, &rspec, n, &mut scratch);
        // спектр масштабируется ОБРАТНО пространству: сдвиг −dr по log r
        let k_est = (-dr * lr).exp();
        let rot_est = dt * 180.0 / n as f64;
        println!(
            "| {} | ×{k:.2} | ×{k_est:.2} | {rot:.0}° | {rot_est:.1}° | {} |",
            if clut { "знак + окружающая нагрузка" } else { "знак на ровном фоне" },
            f2(psr)
        );
    }

    // цена
    let mut acc = 0.0;
    let rounds = 30;
    let t = Instant::now();
    for _ in 0..rounds {
        let s = fm_spec(&fft1, &refw, n, &mut scratch);
        let (_, _, p, _) = fm_match(&fft1, &s, &rspec, n, &mut scratch);
        acc += p;
    }
    let ms1 = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
    let t = Instant::now();
    for _ in 0..rounds {
        let s = fm_spec(&fftp, &refw, n, &mut scratch);
        let (_, _, p, _) = fm_match(&fftp, &s, &rspec, n, &mut scratch);
        acc += p;
    }
    let msp = t.elapsed().as_secs_f64() * 1e3 / rounds as f64;
    std::hint::black_box(acc);
    // идеальная оценка: 3 преобразования 256² по 5·N·log2 N флопов при 4 Гфлоп/с
    let npts = (n * n) as f64;
    let ms_ideal = 3.0 * 5.0 * npts * npts.log2() / 4.0e9 * 1e3;

    println!("\n| цена одного раунда ФМ на окне 256×256 | мс |");
    println!("|---|---|");
    println!("| измерено, 1 поток (своё БПФ) | {ms1:.1} |");
    println!("| измерено, {th} потоков (своё БПФ) | {msp:.1} |");
    println!("| оценка для настроенного БПФ, 1 поток | ≈{ms_ideal:.2} |");

    println!("\n| шаг сетки окон | окон на кадр 1920×1080 | цена измеренная | цена при настроенном БПФ |");
    println!("|---|---|---|---|");
    for stride in [128usize, 64, 32] {
        let wins = ((1920 - n) / stride + 1) * ((1080 - n) / stride + 1);
        println!(
            "| {stride} px | {wins} | {:.0} мс (×{:.0} от 157 мс) | {:.0} мс (×{:.0}) |",
            ms1 * wins as f64,
            ms1 * wins as f64 / 157.0,
            ms_ideal * wins as f64,
            ms_ideal * wins as f64 / 157.0
        );
    }
    println!(
        "\nИ это ЕЩЁ НЕ ВСЁ: ФМ по построению не даёт СДВИГА — он даёт только масштаб и поворот\n\
         для окна, которое уже стоит на знаке. После него всё равно нужна обычная корреляция,\n\
         чтобы получить положение. То есть ФМ не заменяет поиск, а добавляется к нему."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[Cpx]) -> Vec<Cpx> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut s = Cpx::ZERO;
                for (j, v) in x.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                    s = s.add(v.mul(Cpx::new(a.cos(), a.sin())));
                }
                s
            })
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft_for_mixed_radix() {
        for &n in &[8usize, 12, 15, 30, 60] {
            let mut rng = Rng::new(n as u64 + 1);
            let x: Vec<Cpx> = (0..n).map(|_| Cpx::new(rng.gaussian(), rng.gaussian())).collect();
            let plan = Plan::new(n);
            let mut out = vec![Cpx::ZERO; n];
            plan.run(&x, 1, &mut out);
            let want = naive_dft(&x);
            for k in 0..n {
                assert!(
                    (out[k].re - want[k].re).abs() < 1e-8 && (out[k].im - want[k].im).abs() < 1e-8,
                    "n={n} k={k}: {:?} vs {:?}",
                    out[k],
                    want[k]
                );
            }
        }
    }

    #[test]
    fn fft2_roundtrip_is_identity() {
        let (w, h) = (30usize, 20usize);
        let fft = Fft2::new(w, h, 2);
        let mut rng = Rng::new(9);
        let orig: Vec<Cpx> = (0..w * h).map(|_| Cpx::new(rng.next_f64(), rng.next_f64())).collect();
        let mut d = orig.clone();
        let mut sc = vec![Cpx::ZERO; w * h];
        fft.forward(&mut d, &mut sc);
        fft.inverse(&mut d, &mut sc);
        for i in 0..w * h {
            assert!((d[i].re - orig[i].re).abs() < 1e-9, "re {i}");
            assert!((d[i].im - orig[i].im).abs() < 1e-9, "im {i}");
        }
    }

    #[test]
    fn m_sequence_has_full_period_and_balance() {
        for k in [6usize, 8, 10] {
            let s = m_sequence(k);
            assert_eq!(s.len(), (1 << k) - 1);
            let sum: i32 = s.iter().map(|&v| v as i32).sum();
            // m-последовательность сбалансирована с точностью до одного элемента
            assert_eq!(sum.abs(), 1, "k={k} sum={sum}");
        }
    }

    #[test]
    fn marray_folding_is_a_bijection() {
        let (k, n1, n2) = folding_for(15);
        assert_eq!(k, 8);
        let s = m_sequence(k);
        let mut seen = vec![false; n1 * n2];
        for t in 0..s.len() {
            let idx = (t % n2) * n1 + (t % n1);
            assert!(!seen[idx], "коллизия свёртки при t={t}");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|b| *b));
    }

    #[test]
    fn integral_box_sum_matches_direct() {
        let (w, h) = (17usize, 13);
        let mut p = Plane::new(w, h);
        let mut rng = Rng::new(3);
        for v in p.d.iter_mut() {
            *v = rng.next_f64();
        }
        let ii = Integral::new(&p);
        for &(x, y, bw, bh) in &[(0usize, 0usize, 5usize, 4usize), (3, 2, 7, 6), (10, 8, 9, 9)] {
            let (mut s1, mut s2) = (0.0, 0.0);
            for yy in y..(y + bh).min(h) {
                for xx in x..(x + bw).min(w) {
                    let v = p.at(xx, yy);
                    s1 += v;
                    s2 += v * v;
                }
            }
            let (a, b) = ii.box_sum(x, y, bw, bh);
            assert!((a - s1).abs() < 1e-9);
            assert!((b - s2).abs() < 1e-9);
        }
    }

    #[test]
    fn homography_inverse_round_trips() {
        let mut sc = Scene::clean(512, 8.0);
        sc.tilt_deg = 25.0;
        let h = tilt_homography(&sc);
        let hi = h_invert(&h);
        for &(x, y) in &[(0.0f64, 0.0f64), (-20.0, 12.0), (30.0, -5.0)] {
            let (u, v) = h_apply(&h, x, y);
            let (bx, by) = h_apply(&hi, u, v);
            assert!((bx - x).abs() < 1e-6 && (by - y).abs() < 1e-6, "{x},{y} -> {bx},{by}");
        }
    }

    #[test]
    fn templates_are_zero_mean_over_mask() {
        for cand in [Cand::MArray, Cand::Chirp, Cand::Bullseye, Cand::V1Border, Cand::V1Corner] {
            let t = make_tmpl(cand, 15, 6.0, false);
            let s: f64 = t.v.iter().map(|c| c.re).sum();
            assert!(s.abs() < 1e-6, "{:?} mean {s}", cand);
            assert!(t.norm > 0.0);
            assert!(t.m > 0);
        }
    }

    #[test]
    fn mask_support_matches_denominator_footprint() {
        for cand in [Cand::MArray, Cand::Chirp, Cand::Bullseye] {
            let t = make_tmpl(cand, 15, 7.3, false);
            assert_eq!(t.mask, Mask::Box);
            assert_eq!(t.m, t.tw * t.th);
        }
        let t = make_tmpl(Cand::V1Border, 0, 4.0, false);
        match t.mask {
            Mask::Ring { off } => {
                assert_eq!(t.m, t.tw * t.th - (t.tw - 2 * off) * (t.th - 2 * off));
            }
            _ => panic!("рамка обязана быть кольцом"),
        }
        let t = make_tmpl(Cand::V1Corner, 15, 6.0, false);
        match t.mask {
            Mask::Corner { off } => {
                assert_eq!(t.m, t.tw * t.th - (t.tw - off) * (t.th - off));
            }
            _ => panic!("уголок обязан быть буквой Г"),
        }
    }

    /// Регрессия: на вырожденных (плоских) участках знаменатель NCC схлопывался
    /// и «корреляция» вылетала за единицу. Порог дисперсии это закрывает.
    #[test]
    fn ncc_never_exceeds_one_on_degenerate_planes() {
        let (w, h) = (128usize, 128usize);
        let mut p = Plane::new(w, h);
        for y in 40..60 {
            for x in 40..60 {
                p.d[y * w + x] = 1.0;
            }
        }
        let fft = Fft2::new(w, h, 2);
        let mut sc = Vec::new();
        let spec = img_spec(&fft, &p, None, &mut sc);
        for (cand, n, ppc) in
            [
                (Cand::MArray, 9usize, 3.0f64),
                (Cand::Chirp, 9, 3.0),
                (Cand::Bullseye, 9, 3.0),
                (Cand::V1Border, 0, 1.5),
                (Cand::V1Corner, 15, 3.0),
            ]
        {
            let t = make_tmpl(cand, n, ppc, false);
            let s = ncc_surface(&fft, &spec, &t, &mut sc);
            for y in 0..s.vh {
                for x in 0..s.vw {
                    let v = s.d[y * s.w + x].abs();
                    assert!(v <= 1.0 + 1e-3, "{:?}: NCC {v} в ({x}, {y})", cand);
                }
            }
        }
    }

    /// Sanity gate в миниатюре: без блюра и шума пик обязан сесть точно.
    #[test]
    fn clean_scene_peak_lands_on_truth() {
        let canvas = 512usize;
        let ppc = 6.0;
        let fft = Fft2::new(canvas, canvas, 2);
        let mut scratch = Vec::new();
        for cand in [Cand::MArray, Cand::Chirp, Cand::Bullseye, Cand::V1Corner] {
            let sc = Scene::clean(canvas, ppc);
            let (p, truth) = render_scene(cand, 15, &sc);
            let spec = img_spec(&fft, &p, None, &mut scratch);
            let t = make_tmpl(cand, 15, ppc, false);
            let s = ncc_surface(&fft, &spec, &t, &mut scratch);
            let pk = analyse(&s, t.tw / 2);
            let err = centre_err(&pk, &t, truth);
            assert!(err <= 1.5, "{:?}: невязка {err} px, пик {}", cand, pk.val);
        }
    }
}
