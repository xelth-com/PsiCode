//! Выравнивание МЕЖКЛЕТОЧНОЙ ИНТЕРФЕРЕНЦИИ (ISI) на сетке клеток.
//!
//! # Что меряет задача
//!
//! Лестница масштабов (FINDINGS §7) показала: остаток после снятия ПОДОГНАННОГО
//! линейного искажения канала сидит на уровне шума при блоках 636 px и 120 px
//! (0.025/0.067 и 0.013/0.027 против шума 0.026/0.041 и 0.005/0.007) и
//! ВЗЛЕТАЕТ ВДЕСЯТЕРО, как только блок становится размером с клетку
//! (0.078/0.080 против шума 0.007/0.011). Цветопередача и матрица 3×3 при этом
//! не деградируют — значит ломается не цвет, а РАЗДЕЛИМОСТЬ СОСЕДНИХ КЛЕТОК.
//!
//! Исследование канального шума (23 снимка статичного символа) добавило второе:
//! фиксированная (не зависящая от кадра) компонента — НЕ PRNU. Линейная модель
//! на битовом узоре окрестности 5×5 объясняет R² = 0.516 её разброса для ЧЁРНЫХ
//! клеток, сильнейший одиночный член — ЛЕВЫЙ сосед, +3.74 кода когда он белый.
//! Для БЕЛЫХ клеток та же модель объясняет лишь R² = 0.069. Субпиксельная фаза
//! не даёт ничего (0.516 -> 0.522), т.е. это не муар и не фаза сэмплирования.
//!
//! Вывод: помеха ДЕТЕРМИНИРОВАНА и является функцией значений соседей, а такая
//! помеха обратима. Асимметрия чёрное/белое сразу говорит, что ОДНОГО ядра на
//! оба уровня может не хватить — отсюда второй, условный по собственному
//! уровню, решатель ниже.
//!
//! # Три кандидата и что реализовано
//!
//! 1. **Линейный КИХ-выравниватель** ([`equalise`]): оценить ядро помехи и
//!    развернуть свёртку. Дёшево, не распространяет ошибки. Реализован через
//!    усечённый ряд Неймана.
//!
//!    Стандартное возражение — «деконволюция усиливает шум там, где ядро
//!    слабо» — здесь ИЗМЕРЕНО и не подтвердилось: у ядер, снятых с живых
//!    кадров, `|H(ω)| ∈ [0.75, 1.25]` по всей частотной плоскости, провалов
//!    нет, и применяемый фильтр усиливает дисперсию шума в **×1.01…1.05**, то
//!    есть на 0.03…0.2 дБ (см. [`conditioning`]). Причина простая: это слабое
//!    низкочастотное размытие, а не канал с нулями в спектре. Ряд Неймана при
//!    этом сходится на второй итерации — 2, 3 и 4 дают идентичный результат,
//!    расходимости нет ни при каком числе итераций.
//! 2. **Решение с обратной связью** ([`equalise_dfe`]): вычитать вклад УЖЕ
//!    ПРИНЯТЫХ решений соседей. Сильнее, но распространяет ошибки, а страйповая
//!    структура CRC означает, что пачка распространённых ошибок убивает целый
//!    страйп. Реализован ИМЕННО чтобы это померить, а не чтобы поверить.
//! 3. **Совместный классификатор** ([`JointRule`]): решать символ клетки прямо
//!    в пространстве (своё значение, значения соседей), с параметрами,
//!    условными по гипотезе о собственном символе. Это форма правила из HiQ
//!    (LSVM-CMI / QDA-CMI, arXiv:1704.06447), где хроматическое искажение и
//!    межклеточная помеха моделируются СОВМЕСТНО, а не последовательно. Петли
//!    обратной связи нет вообще, поэтому распространения ошибок нет
//!    структурно; и модель, условная по гипотезе, естественно вмещает
//!    измеренную асимметрию чёрное/белое, которой одно ядро вместить не может.
//!
//! # Домен, в котором это законно
//!
//! Свёртка (оптика + ISP) линейна ПО СВЕТУ. В приёмнике светолинейная величина
//! — это `t = N·s + q` (§3.4), линеаризованный драйв ДО обращения гаммы.
//! Поэтому выравниватель ставится МЕЖДУ матрицей развязки и обращением гаммы:
//! поле освещённости при этом проходит насквозь неизменным, и его по-прежнему
//! снимает штатная нормировка (§5.1-CL делит на измеренную сумму каналов,
//! яркостный путь — локальным порогом).
//!
//! Ключ к оценке ядра: поле входит МУЛЬТИПЛИКАТИВНО и ОДИНАКОВО на всю сумму
//! `y_n = f_n·Σ_k h_k x_{n−k}`, поэтому ОТНОШЕНИЯ `h_k/h_0` от него не зависят —
//! общий масштаб (единственное, что портит поле) выбрасывается нормировкой
//! центрального отсчёта в единицу.
//!
//! # Граница применимости, которую важно знать заранее
//!
//! Ядро оценивается ПО РЕШЕНИЯМ ПЕРВОГО ПРОХОДА, поэтому у метода есть область
//! сходимости. При SER первого прохода порядка процента оценка практически
//! несмещена (3721 клетка на 15 параметров, ошибки входят как некоррелированный
//! шум отклика). При SER ~0.3 оценка занижается вдвое и повторные проходы
//! улучшают ЯДРО, но не решения: помеха уже съела созвездие целиком. Это не
//! усиление шума деконволюцией — это обычный отказ бутстрапа вне области
//! сходимости, и лечится он не регуляризацией, а известным ядром из калибровки
//! ([`IsiKernel`] кладётся в кэш; на живых кадрах медианное ядро работает ЛУЧШЕ
//! покадровой оценки).

use alloc::vec;
use alloc::vec::Vec;

/// Максимальный радиус ядра в клетках (2 => окно 5×5, как в исследовании шума).
pub const MAX_RADIUS: usize = 2;
/// Максимум отсчётов ядра при [`MAX_RADIUS`].
pub const MAX_TAPS: usize = (2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1);
/// Число членов гладкого фона в регрессии ядра: 1, r, c, r², rc, c².
const TREND_TERMS: usize = 6;

/// Ядро межклеточной помехи: `y_n = Σ_k h_k · x_{n−k}`, центральный отсчёт
/// нормирован в 1. Значение вне носителя — ноль.
///
/// Хранится в растровом порядке окна `(2·radius+1)²`; неиспользуемый хвост при
/// `radius < MAX_RADIUS` — нули (структура `Copy`, чтобы её можно было положить
/// в калибровочный кэш без аллокаций).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsiKernel {
    /// Радиус носителя в клетках.
    pub radius: usize,
    /// Отсчёты в растровом порядке окна, `taps[centre] == 1.0`.
    pub taps: [f64; MAX_TAPS],
}

impl IsiKernel {
    /// Нейтральное ядро (дельта): выравниватель становится тождеством.
    pub fn identity(radius: usize) -> Self {
        let r = radius.min(MAX_RADIUS);
        let mut taps = [0.0f64; MAX_TAPS];
        taps[centre_index(r)] = 1.0;
        IsiKernel { radius: r, taps }
    }

    /// Сторона окна в клетках.
    #[inline]
    pub fn side(&self) -> usize {
        2 * self.radius + 1
    }

    /// Число значащих отсчётов.
    #[inline]
    pub fn n_taps(&self) -> usize {
        self.side() * self.side()
    }

    /// Отсчёт ядра свёртки `h_k`, `k = (dr, dc)`; вне носителя — 0.
    ///
    /// ВНИМАНИЕ: это индекс СВЁРТКИ, а не смещение соседа. `h_{(dr,dc)}`
    /// умножает `x_{(r−dr, c−dc)}`, то есть соседа со смещением
    /// `(−dr, −dc)`. Для «сколько подмешивает сосед вон оттуда» есть
    /// [`IsiKernel::neighbour`] — путать эти две вещи означает зеркалить
    /// асимметрию ядра, а вся суть замера как раз в ней.
    #[inline]
    pub fn tap(&self, dr: i32, dc: i32) -> f64 {
        let r = self.radius as i32;
        if dr < -r || dr > r || dc < -r || dc > r {
            return 0.0;
        }
        self.taps[((dr + r) as usize) * self.side() + (dc + r) as usize]
    }

    /// Вклад СОСЕДА, стоящего на смещении `(dr, dc)` от клетки (dr вниз,
    /// dc вправо). Это `tap(−dr, −dc)`.
    #[inline]
    pub fn neighbour(&self, dr: i32, dc: i32) -> f64 {
        self.tap(-dr, -dc)
    }

    /// Задать вклад соседа со смещением `(dr, dc)`. Центр `(0,0)` нормирован в
    /// 1 и не меняется.
    pub fn set_neighbour(&mut self, dr: i32, dc: i32, v: f64) {
        let r = self.radius as i32;
        let (a, b) = (-dr, -dc);
        if a < -r || a > r || b < -r || b > r || (dr == 0 && dc == 0) {
            return;
        }
        let side = self.side();
        self.taps[((a + r) as usize) * side + (b + r) as usize] = v;
    }

    /// Суммарная АБСОЛЮТНАЯ энергия внецентральных отсчётов — «сила ISI».
    /// Это же и коэффициент петли: при значении ≪ 1 ряд Неймана сходится, а
    /// обратная связь по решениям не может лавинообразно расходиться.
    pub fn strength(&self) -> f64 {
        let cc = centre_index(self.radius);
        self.taps
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != cc)
            .map(|(_, v)| abs(*v))
            .sum()
    }

    /// Сумма всех отсчётов (для дельты и для сохраняющего DC блюра — 1 + Σ_{k≠0}).
    pub fn dc_gain(&self) -> f64 {
        self.taps.iter().sum()
    }

    /// Максимальное расхождение отсчётов с другим ядром — метрика повторяемости
    /// ядра от снимка к снимку (фиксированное оно или пер-кадровое).
    pub fn max_tap_diff(&self, other: &IsiKernel) -> f64 {
        let r = self.radius.max(other.radius) as i32;
        let mut worst = 0.0f64;
        for dr in -r..=r {
            for dc in -r..=r {
                let d = abs(self.tap(dr, dc) - other.tap(dr, dc));
                if d > worst {
                    worst = d;
                }
            }
        }
        worst
    }
}

/// Форма носителя ядра.
///
/// Анизотропия помехи на квадратной решётке измерена как **13×**: вклад
/// рёберного соседа (расстояние 1.0) против диагонального (√2) — 0.0564 против
/// 0.0043 в геометрическом исследовании и 0.06 против 0.006 в поканальных
/// ядрах, снятых с живых цветных кадров. То есть четыре ДИАГОНАЛЬНЫХ
/// коэффициента полного квадрата почти целиком подгоняют шум, а не физику, —
/// и при этом занимают половину степеней свободы регрессии.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelShape {
    /// Полный квадрат `(2r+1)²`.
    Full,
    /// КРЕСТ: центр плюс рёберные соседи (диагонали принудительно нулевые).
    Cross,
}

impl KernelShape {
    /// Входит ли смещение `(dr, dc)` в носитель.
    #[inline]
    fn holds(&self, dr: i32, dc: i32) -> bool {
        match self {
            KernelShape::Full => true,
            KernelShape::Cross => dr == 0 || dc == 0,
        }
    }
}

/// Обусловленность обращения ядра — прямой ответ на вопрос «а не усилит ли
/// деконволюция шум».
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conditioning {
    /// Минимум `|H(ω)|` по частотной плоскости. Ноль означал бы, что обратного
    /// фильтра не существует ни в каком виде.
    pub h_min: f64,
    /// Максимум `|H(ω)|`.
    pub h_max: f64,
    /// УСИЛЕНИЕ ДИСПЕРСИИ ШУМА фактически применяемым фильтром: `Σ_k g_k²`, где
    /// `g` — усечённый ряд Неймана после `iters` итераций. Единица = шум не
    /// тронут.
    pub noise_gain: f64,
    /// Максимум `|G(ω)·H(ω) − 1|` — сколько ISI осталось после усечения.
    pub residual: f64,
}

/// Эффективный обратный фильтр после `iters` итераций ряда Неймана: результат
/// применения выравнивателя к дельте.
fn inverse_fir(k: &IsiKernel, iters: usize) -> (Vec<f64>, usize, usize) {
    // носитель растёт на radius за итерацию; берём с запасом
    let r = k.radius * (iters + 1) + k.radius;
    let n = 2 * r + 1;
    let mut plane = vec![0.0f64; n * n];
    plane[r * n + r] = 1.0;
    equalise(&mut plane, n, n, k, iters);
    (plane, n, r)
}

/// Обусловленность обращения ядра при `iters` итерациях.
///
/// `h_min` считается по сетке частот `steps × steps` в `[0, π]²` (ядро
/// вещественное, поэтому этого хватает).
#[cfg(feature = "std")]
pub fn conditioning(k: &IsiKernel, iters: usize, steps: usize) -> Conditioning {
    let (g, n, gc) = inverse_fir(k, iters);
    let noise_gain = g.iter().map(|v| v * v).sum::<f64>();
    let r = k.radius as i32;
    let mut h_min = f64::INFINITY;
    let mut h_max: f64 = 0.0;
    let mut residual: f64 = 0.0;
    for a in 0..=steps {
        for b in 0..=steps {
            let (wr, wc) = (
                core::f64::consts::PI * a as f64 / steps as f64,
                core::f64::consts::PI * b as f64 / steps as f64,
            );
            let (mut hre, mut him) = (0.0f64, 0.0f64);
            for dr in -r..=r {
                for dc in -r..=r {
                    let t = k.tap(dr, dc);
                    if t == 0.0 {
                        continue;
                    }
                    let ph = -(wr * dr as f64 + wc * dc as f64);
                    hre += t * ph.cos();
                    him += t * ph.sin();
                }
            }
            let m = (hre * hre + him * him).sqrt();
            h_min = h_min.min(m);
            h_max = h_max.max(m);
            let (mut gre, mut gim) = (0.0f64, 0.0f64);
            for rr in 0..n {
                for cc in 0..n {
                    let v = g[rr * n + cc];
                    if v == 0.0 {
                        continue;
                    }
                    let (dr, dc) = (rr as f64 - gc as f64, cc as f64 - gc as f64);
                    let ph = -(wr * dr + wc * dc);
                    gre += v * ph.cos();
                    gim += v * ph.sin();
                }
            }
            // |G·H − 1|
            let (pre, pim) = (gre * hre - gim * him, gre * him + gim * hre);
            residual = residual.max(((pre - 1.0).powi(2) + pim * pim).sqrt());
        }
    }
    Conditioning {
        h_min,
        h_max,
        noise_gain,
        residual,
    }
}

/// Индекс центрального отсчёта окна радиуса `r` в растровом порядке.
#[inline]
fn centre_index(r: usize) -> usize {
    let side = 2 * r + 1;
    r * side + r
}

/// |x| без std (крейт собирается в no_std-конфигурации).
#[inline]
fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Прямоугольная сетка клеточных величин с зажимом координат к краю.
///
/// Зажим (а не нули) выбран намеренно: за краем payload лежит РЕАЛЬНОЕ
/// содержимое (референсная строка, строка счётчика, ЗЧ-кольцо), и нули были бы
/// заведомо ложным соседом. Штатный вызов передаёт сетку ВСЕГО символа 61×61,
/// так что зажим срабатывает только на самом кольце.
#[derive(Debug, Clone, Copy)]
pub struct Grid<'a> {
    /// Значения в растровом порядке, длиной `rows·cols`.
    pub v: &'a [f64],
    pub rows: usize,
    pub cols: usize,
}

impl<'a> Grid<'a> {
    /// Значение с зажимом координат к краю сетки.
    #[inline]
    pub fn at(&self, r: i32, c: i32) -> f64 {
        let rr = r.clamp(0, self.rows as i32 - 1) as usize;
        let cc = c.clamp(0, self.cols as i32 - 1) as usize;
        self.v[rr * self.cols + cc]
    }
}

// ---------------------------------------------------------------------------
// Оценка ядра
// ---------------------------------------------------------------------------

/// Оценка ядра ISI по паре «измеренная сетка / ИДЕАЛЬНАЯ сетка».
///
/// Идеальная сетка — это переизлучённые (re-encoded) значения ПРОБНЫХ решений
/// первого прохода в том же светолинейном домене, что и измерения; на клетках,
/// где содержимое известно точно (референсная строка, кольцо, строка счётчика),
/// решения тривиально верны. МНК-модель:
///
/// ```text
/// y_n = Σ_k h_k · x_{n−k} + (гладкий фон: 1, r, c, r², rc, c²)
/// ```
///
/// Гладкий фон нужен, потому что поле освещённости и уровень чёрного меняются
/// по кадру (0.62 -> 0.86 по яркости); без него они уехали бы в отсчёты ядра.
/// Затем всё делится на центральный отсчёт — общий масштаб (в который поле как
/// раз и входит) физически не наблюдаем и вреда не приносит.
///
/// `margin` — сколько клеток от края сетки не брать в регрессию (окрестность
/// такой клетки частично зажата, и зажатые повторы завышают близкие отсчёты).
///
/// `None`, если система вырождена или центральный отсчёт получился неположительным.
pub fn estimate_kernel(
    meas: &Grid,
    ideal: &Grid,
    radius: usize,
    margin: usize,
    shape: KernelShape,
) -> Option<IsiKernel> {
    assert_eq!(meas.rows, ideal.rows);
    assert_eq!(meas.cols, ideal.cols);
    let r = radius.min(MAX_RADIUS) as i32;
    let (rows, cols) = (meas.rows, meas.cols);
    if rows <= 2 * margin || cols <= 2 * margin {
        return None;
    }
    // активные отсчёты носителя: (индекс в окне, смещение dr, dc)
    let side = 2 * radius.min(MAX_RADIUS) + 1;
    let mut active: Vec<(usize, i32, i32)> = Vec::with_capacity(side * side);
    for dr in -r..=r {
        for dc in -r..=r {
            if shape.holds(dr, dc) {
                active.push((((dr + r) as usize) * side + (dc + r) as usize, dr, dc));
            }
        }
    }
    let na = active.len();
    let nb = na + TREND_TERMS;
    // нормальные уравнения; базис: активные отсчёты ядра, затем гладкий фон
    let mut ata = vec![0.0f64; nb * nb];
    let mut atb = vec![0.0f64; nb];
    let mut basis = vec![0.0f64; nb];
    let (hr, hc) = (0.5 * rows as f64, 0.5 * cols as f64);
    for rr in margin..rows - margin {
        for cc in margin..cols - margin {
            for (i, &(_, dr, dc)) in active.iter().enumerate() {
                basis[i] = ideal.at(rr as i32 - dr, cc as i32 - dc);
            }
            let ur = (rr as f64 - hr) / hr;
            let uc = (cc as f64 - hc) / hc;
            basis[na] = 1.0;
            basis[na + 1] = ur;
            basis[na + 2] = uc;
            basis[na + 3] = ur * ur;
            basis[na + 4] = ur * uc;
            basis[na + 5] = uc * uc;
            let y = meas.v[rr * cols + cc];
            for a in 0..nb {
                let ba = basis[a];
                if ba == 0.0 {
                    continue;
                }
                atb[a] += ba * y;
                for b in a..nb {
                    ata[a * nb + b] += ba * basis[b];
                }
            }
        }
    }
    for a in 0..nb {
        for b in 0..a {
            ata[a * nb + b] = ata[b * nb + a];
        }
    }
    let x = solve_sym(&mut ata, &mut atb, nb)?;
    let ci = centre_index(radius.min(MAX_RADIUS));
    let centre_col = active.iter().position(|&(i, _, _)| i == ci)?;
    let h0 = x[centre_col];
    if !(h0 > 1e-9) {
        return None;
    }
    let mut taps = [0.0f64; MAX_TAPS];
    for (col, &(i, _, _)) in active.iter().enumerate() {
        let v = x[col] / h0;
        if !v.is_finite() {
            return None;
        }
        taps[i] = v;
    }
    taps[ci] = 1.0;
    Some(IsiKernel {
        radius: radius.min(MAX_RADIUS),
        taps,
    })
}

/// Гаусс с частичным выбором ведущего элемента для `A·x = b` (A портится).
fn solve_sym(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    let mut scale = 0.0f64;
    for i in 0..n {
        let d = abs(a[i * n + i]);
        if d > scale {
            scale = d;
        }
    }
    if scale <= 0.0 {
        return None;
    }
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if abs(a[r * n + col]) > abs(a[piv * n + col]) {
                piv = r;
            }
        }
        if abs(a[piv * n + col]) < 1e-12 * scale {
            return None;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for j in col..n {
            a[col * n + j] /= d;
        }
        b[col] /= d;
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                a[r * n + j] -= f * a[col * n + j];
            }
            b[r] -= f * b[col];
        }
    }
    let out: Vec<f64> = b[..n].to_vec();
    if out.iter().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Линейный выравниватель
// ---------------------------------------------------------------------------

/// Свёртка сетки с ядром (зажим к краю).
///
/// Внутренность и кайма разделены намеренно: во внутренности индексы заведомо
/// в пределах сетки, и зажим (две ветки на каждое из девяти умножений) там
/// чистые накладные расходы. На символе 61×61 внутренность — 87 % клеток, а
/// свёртка исполняется по три канала на итерацию, так что это не микрооптимизация,
/// а заметная доля бюджета кадра.
fn convolve(src: &Grid, k: &IsiKernel, out: &mut [f64]) {
    let r = k.radius;
    let ri = r as i32;
    let (rows, cols) = (src.rows, src.cols);
    // предвычисленные (смещение в буфере, вес) для внутренней области
    let side = 2 * r + 1;
    let mut off: Vec<(isize, f64)> = Vec::with_capacity(side * side);
    for dr in -ri..=ri {
        for dc in -ri..=ri {
            let h = k.tap(dr, dc);
            if h != 0.0 {
                off.push(((-dr as isize) * cols as isize + (-dc as isize), h));
            }
        }
    }
    if rows > 2 * r && cols > 2 * r {
        for rr in r..rows - r {
            let base = rr * cols;
            for cc in r..cols - r {
                let i = (base + cc) as isize;
                let mut acc = 0.0;
                for &(d, h) in &off {
                    acc += h * src.v[(i + d) as usize];
                }
                out[base + cc] = acc;
            }
        }
    }
    // кайма — через зажатый доступ
    for rr in 0..rows {
        let interior_row = rr >= r && rr + r < rows;
        for cc in 0..cols {
            if interior_row && cc >= r && cc + r < cols {
                continue;
            }
            let mut acc = 0.0;
            for dr in -ri..=ri {
                for dc in -ri..=ri {
                    let h = k.tap(dr, dc);
                    if h != 0.0 {
                        acc += h * src.at(rr as i32 - dr, cc as i32 - dc);
                    }
                }
            }
            out[rr * cols + cc] = acc;
        }
    }
}

/// ЛИНЕЙНОЕ выравнивание на месте: приблизить `x` в `y = h ⊛ x` усечённым рядом
/// Неймана (итерация Ван Циттерта) `x_{m+1} = x_m + (y − h ⊛ x_m)`.
///
/// Почему именно так, а не точное обращение: ядро слабое (внецентральная
/// энергия ≈ 0.05), поэтому уже ПЕРВАЯ итерация оставляет остаток O(‖h−δ‖²) ≈
/// 0.003, а усиление шума равно `1 + Σ_{k≠0} h_k²` — то есть доли процента.
/// Точное обращение (БПФ по всей сетке) стоило бы дороже и вносило бы краевые
/// артефакты на 61-клеточном символе, где край — это 7 % клеток.
///
/// Итераций `iters`; `0` — тождество. Возвращает СКО поправки (диагностика).
pub fn equalise(plane: &mut [f64], rows: usize, cols: usize, k: &IsiKernel, iters: usize) -> f64 {
    if iters == 0 || k.radius == 0 {
        return 0.0;
    }
    let y: Vec<f64> = plane.to_vec();
    let mut conv = vec![0.0f64; rows * cols];
    for _ in 0..iters {
        {
            let g = Grid {
                v: plane,
                rows,
                cols,
            };
            convolve(&g, k, &mut conv);
        }
        for i in 0..rows * cols {
            plane[i] += y[i] - conv[i];
        }
    }
    let mut s = 0.0;
    for i in 0..rows * cols {
        let d = plane[i] - y[i];
        s += d * d;
    }
    sqrt(s / (rows * cols) as f64)
}

/// √x без std: Ньютон по начальному приближению из экспоненты (нужен только для
/// диагностических СКО, точность здесь некритична).
fn sqrt(x: f64) -> f64 {
    if !(x > 0.0) {
        return 0.0;
    }
    let mut g = x;
    for _ in 0..40 {
        let ng = 0.5 * (g + x / g);
        if abs(ng - g) <= 1e-15 * ng {
            return ng;
        }
        g = ng;
    }
    g
}

// ---------------------------------------------------------------------------
// Выравниватель с обратной связью по решениям (DFE)
// ---------------------------------------------------------------------------

/// Отчёт о работе DFE: сколько клеток изменило решение и какова длина
/// РАСПРОСТРАНЯЮЩИХСЯ пачек (важнее средней SER: страйп CRC-16 гибнет от одной
/// ошибки, поэтому решает не число ошибок, а их РАСПРЕДЕЛЕНИЕ по страйпам).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DfeReport {
    /// Клеток, у которых решение изменилось относительно решения без DFE.
    pub flipped: usize,
    /// Наибольший модуль поправки, применённой к клетке.
    pub max_correction: f64,
    /// Коэффициент петли `Σ_{k≠0}|h_k|` — оценка сверху для усиления пачки.
    pub loop_gain: f64,
}

/// Выравнивание с ОБРАТНОЙ СВЯЗЬЮ ПО РЕШЕНИЯМ: из значения клетки вычитается
/// вклад соседей, ПРИЧЁМ для уже пройденных в растровом порядке соседей
/// (каузальная половина ядра) берётся ЖЁСТКОЕ решение, а для ещё не пройденных
/// — сырое измерение.
///
/// Так выглядит классический DFE, и именно здесь живёт риск распространения
/// ошибок: неверное жёсткое решение вносит в соседа ошибку `2·h_k`. Петля,
/// однако, ограничена: `Σ_{k≠0}|h_k| ≈ 0.05`, поэтому вклад ошибки затухает за
/// один шаг и лавины быть не может — но это надо ИЗМЕРИТЬ, а не постулировать,
/// чем и занят `psicode-rx/examples/isi_eq.rs`.
///
/// `decide` — жёсткое квантование значения к ближайшей точке созвездия оси.
pub fn equalise_dfe(
    plane: &mut [f64],
    rows: usize,
    cols: usize,
    k: &IsiKernel,
    decide: &dyn Fn(f64) -> f64,
) -> DfeReport {
    let mut rep = DfeReport {
        loop_gain: k.strength(),
        ..Default::default()
    };
    if k.radius == 0 {
        return rep;
    }
    let y: Vec<f64> = plane.to_vec();
    // жёсткие решения, заполняемые по ходу растрового прохода
    let mut hard = vec![f64::NAN; rows * cols];
    let r = k.radius as i32;
    for rr in 0..rows as i32 {
        for cc in 0..cols as i32 {
            let mut acc = 0.0;
            for dr in -r..=r {
                for dc in -r..=r {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let h = k.tap(dr, dc);
                    if h == 0.0 {
                        continue;
                    }
                    // сосед, ДАВШИЙ вклад в текущую клетку, стоит на (rr−dr, cc−dc)
                    let (sr, sc) = (
                        (rr - dr).clamp(0, rows as i32 - 1),
                        (cc - dc).clamp(0, cols as i32 - 1),
                    );
                    let idx = sr as usize * cols + sc as usize;
                    let v = if hard[idx].is_nan() { y[idx] } else { hard[idx] };
                    acc += h * v;
                }
            }
            let idx = rr as usize * cols + cc as usize;
            let corrected = y[idx] - acc;
            if abs(acc) > rep.max_correction {
                rep.max_correction = abs(acc);
            }
            let d_before = decide(y[idx]);
            let d_after = decide(corrected);
            if d_before != d_after {
                rep.flipped += 1;
            }
            plane[idx] = corrected;
            hard[idx] = d_after;
        }
    }
    rep
}

// ---------------------------------------------------------------------------
// Совместное решающее правило (HiQ-подобное)
// ---------------------------------------------------------------------------

/// Совместное решающее правило по вектору признаков «своё значение + значения
/// соседей», с параметрами, УСЛОВНЫМИ ПО ГИПОТЕЗЕ о собственном символе.
///
/// # Почему оно, а не одно ядро
///
/// Замер канального шума даёт для линейной модели по окрестности 5×5
/// R² = 0.516 на ЧЁРНЫХ клетках и R² = 0.069 на БЕЛЫХ. Одно ядро — это
/// утверждение, что помеха от соседей одинакова независимо от собственного
/// уровня клетки; измерение говорит обратное. Здесь для КАЖДОГО уровня `l`
/// собственной оси хранится своя аффинная модель ожидаемого измерения
///
/// ```text
/// m̂(l) = μ_l + Σ_{k≠0} a_{l,k} · v_{n−k}
/// ```
///
/// и решение принимается по минимуму `(v_n − m̂(l))²`. Это дискриминант в
/// совместном пространстве (своё, соседи) — та же идея, что LSVM-CMI/QDA-CMI в
/// HiQ, но в замкнутой форме и с оценкой по МНК, без обучения SVM.
///
/// # Чем оно лучше DFE операционно
///
/// В признаках стоят СЫРЫЕ (мягкие) значения соседей, а не их решения, поэтому
/// петли обратной связи НЕТ ВООБЩЕ и распространения ошибок нет структурно.
/// Это существенно: страйп несёт CRC-16 на 399 клеток, и пачка распространённых
/// ошибок убивает страйп целиком.
///
/// # Вырождение в линейный выравниватель
///
/// Если `a_{l,k}` не зависит от `l`, правило тождественно линейному
/// выравнивателю с ядром `−a_k`: обе гипотезы получают одинаковую поправку, и
/// сравнение сводится к сравнению `v_n − Σ a_k v_{n−k}` с серединой уровней.
/// Разница между этой моделью и линейным ядром — РОВНО та асимметрия, которую
/// мы измерили.
#[derive(Debug, Clone)]
pub struct JointRule {
    /// Радиус окрестности признаков.
    pub radius: usize,
    /// Уровни созвездия оси (в тех же единицах, что сетка).
    pub levels: Vec<f64>,
    /// На уровень: свободный член `μ_l`.
    pub mu: Vec<f64>,
    /// На уровень: веса соседей в растровом порядке окна, центр не используется.
    pub a: Vec<[f64; MAX_TAPS]>,
}

impl JointRule {
    /// Обучение по паре «измеренная сетка / ИСТИННЫЕ (или пробные) уровни».
    ///
    /// `label` даёт для каждой клетки индекс уровня (`usize::MAX` — клетку не
    /// использовать). Для каждого уровня решается МНК на признаках
    /// `[1, соседи]`, где отклик — измеренное значение клетки.
    ///
    /// `None`, если хотя бы у одного уровня не набралось обусловленной системы.
    pub fn train(
        meas: &Grid,
        label: &[usize],
        levels: &[f64],
        radius: usize,
        margin: usize,
    ) -> Option<JointRule> {
        let r = radius.min(MAX_RADIUS);
        let side = 2 * r + 1;
        let nt = side * side;
        let nl = levels.len();
        let (rows, cols) = (meas.rows, meas.cols);
        // базис: [1, соседи кроме центра] -> nt элементов (центр заменён единицей)
        let nb = nt;
        let mut ata = vec![0.0f64; nl * nb * nb];
        let mut atb = vec![0.0f64; nl * nb];
        let mut basis = vec![0.0f64; nb];
        let cc0 = centre_index(r);
        let mut seen = vec![0usize; nl];
        for rr in margin..rows.saturating_sub(margin) {
            for cc in margin..cols.saturating_sub(margin) {
                let idx = rr * cols + cc;
                let l = label[idx];
                if l >= nl {
                    continue;
                }
                let mut i = 0usize;
                for dr in -(r as i32)..=(r as i32) {
                    for dc in -(r as i32)..=(r as i32) {
                        basis[i] = if i == cc0 {
                            1.0 // место центра занимает свободный член μ
                        } else {
                            meas.at(rr as i32 - dr, cc as i32 - dc)
                        };
                        i += 1;
                    }
                }
                let y = meas.v[idx];
                let base = l * nb * nb;
                for a in 0..nb {
                    let ba = basis[a];
                    atb[l * nb + a] += ba * y;
                    for b in a..nb {
                        ata[base + a * nb + b] += ba * basis[b];
                    }
                }
                seen[l] += 1;
            }
        }
        let mut mu = vec![0.0f64; nl];
        let mut coef = vec![[0.0f64; MAX_TAPS]; nl];
        for l in 0..nl {
            if seen[l] < nb * 4 {
                return None; // мало данных на уровень — правило было бы шумом
            }
            let base = l * nb * nb;
            let mut a = ata[base..base + nb * nb].to_vec();
            for i in 0..nb {
                for j in 0..i {
                    a[i * nb + j] = a[j * nb + i];
                }
            }
            let mut b = atb[l * nb..(l + 1) * nb].to_vec();
            let x = solve_sym(&mut a, &mut b, nb)?;
            mu[l] = x[cc0];
            for i in 0..nt {
                if i != cc0 {
                    coef[l][i] = x[i];
                }
            }
        }
        Some(JointRule {
            radius: r,
            levels: levels.to_vec(),
            mu,
            a: coef,
        })
    }

    /// Индекс уровня, объясняющего измерение клетки лучше всех.
    pub fn decide(&self, meas: &Grid, rr: usize, cc: usize) -> usize {
        let r = self.radius as i32;
        let side = 2 * self.radius + 1;
        let cc0 = centre_index(self.radius);
        let v = meas.v[rr * meas.cols + cc];
        let mut best = (f64::INFINITY, 0usize);
        for l in 0..self.levels.len() {
            let mut m = self.mu[l];
            let mut i = 0usize;
            for dr in -r..=r {
                for dc in -r..=r {
                    if i != cc0 {
                        let w = self.a[l][i];
                        if w != 0.0 {
                            m += w * meas.at(rr as i32 - dr, cc as i32 - dc);
                        }
                    }
                    i += 1;
                }
            }
            let _ = side;
            let d = v - m;
            let e = d * d;
            if e < best.0 {
                best = (e, l);
            }
        }
        best.1
    }

    /// Решения по всей сетке.
    pub fn decide_all(&self, meas: &Grid) -> Vec<usize> {
        let mut out = vec![0usize; meas.rows * meas.cols];
        for rr in 0..meas.rows {
            for cc in 0..meas.cols {
                out[rr * meas.cols + cc] = self.decide(meas, rr, cc);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Псевдослучайный ±1 узор (детерминированный, без внешних зависимостей).
    fn pattern(n: usize) -> Vec<f64> {
        let mut s = 0x1234_5678_9abc_def0u64;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                if (s >> 60) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }

    fn apply(k: &IsiKernel, x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; rows * cols];
        let g = Grid { v: x, rows, cols };
        convolve(&g, k, &mut out);
        out
    }

    /// Асимметричное ядро в духе замера: горизонтальные соседи неравны (5:2),
    /// вертикальные тоже. Задаётся В ГЕОМЕТРИИ СОСЕДЕЙ, чтобы тест не зависел
    /// от того, помнит ли читатель зеркальность индекса свёртки.
    fn injected() -> IsiKernel {
        let mut k = IsiKernel::identity(1);
        k.set_neighbour(0, 1, 0.060); // правый
        k.set_neighbour(0, -1, 0.030); // левый
        k.set_neighbour(1, 0, 0.025); // нижний
        k.set_neighbour(-1, 0, 0.020); // верхний
        k
    }

    /// Соседский и свёрточный индексы — зеркальны, и это закреплено тестом.
    #[test]
    fn neighbour_index_is_mirror_of_tap_index() {
        let k = injected();
        assert_eq!(k.neighbour(0, 1), 0.060);
        assert_eq!(k.tap(0, -1), 0.060);
        assert_eq!(k.neighbour(1, 0), 0.025);
        assert_eq!(k.tap(-1, 0), 0.025);
        assert_eq!(k.neighbour(0, 0), 1.0);
    }

    #[test]
    fn identity_kernel_is_neutral() {
        let k = IsiKernel::identity(2);
        assert_eq!(k.strength(), 0.0);
        assert_eq!(k.dc_gain(), 1.0);
        let mut p = pattern(61 * 61);
        let before = p.clone();
        equalise(&mut p, 61, 61, &k, 3);
        assert_eq!(p, before);
    }

    /// РЕГРЕССИЯ: оценка возвращает ВПРЫСНУТОЕ ядро, а выравниватель его снимает.
    #[test]
    fn recovers_injected_kernel_and_removes_it() {
        let (rows, cols) = (61usize, 61usize);
        let x = pattern(rows * cols);
        let k = injected();
        let y = apply(&k, &x, rows, cols);
        let est = estimate_kernel(
            &Grid {
                v: &y,
                rows,
                cols,
            },
            &Grid {
                v: &x,
                rows,
                cols,
            },
            1,
            2,
            KernelShape::Full,
        )
        .expect("оценка ядра");
        assert!(
            est.max_tap_diff(&k) < 1e-6,
            "ядро восстановлено неточно: {:?}",
            est.taps
        );
        // выравнивание возвращает исходную сетку
        let mut p = y.clone();
        equalise(&mut p, rows, cols, &est, 4);
        let mut worst = 0.0f64;
        for r in 3..rows - 3 {
            for c in 3..cols - 3 {
                worst = worst.max(abs(p[r * cols + c] - x[r * cols + c]));
            }
        }
        assert!(worst < 1e-3, "остаток после выравнивания {worst}");
        // и решения становятся безошибочными, тогда как ДО выравнивания ошибки есть
        let err_after = (3..rows - 3)
            .flat_map(|r| (3..cols - 3).map(move |c| (r, c)))
            .filter(|&(r, c)| (p[r * cols + c] > 0.0) != (x[r * cols + c] > 0.0))
            .count();
        assert_eq!(err_after, 0);
    }

    /// Оценка устойчива к ГЛАДКОМУ полю: члены тренда его поглощают.
    #[test]
    fn kernel_estimate_survives_smooth_field() {
        let (rows, cols) = (61usize, 61usize);
        let x = pattern(rows * cols);
        let k = injected();
        let y0 = apply(&k, &x, rows, cols);
        let y: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = ((i / cols) as f64, (i % cols) as f64);
                let f = 0.62 + 0.24 * (r + c) / (2.0 * 60.0);
                y0[i] * f + 0.05 * f
            })
            .collect();
        let est = estimate_kernel(
            &Grid {
                v: &y,
                rows,
                cols,
            },
            &Grid {
                v: &x,
                rows,
                cols,
            },
            1,
            2,
            KernelShape::Full,
        )
        .expect("оценка ядра");
        assert!(
            est.max_tap_diff(&k) < 0.01,
            "поле сдвинуло ядро: {:?} против {:?}",
            est.taps,
            k.taps
        );
    }

    /// Совместное правило вырождается в линейный выравниватель, когда помеха
    /// одинакова для обоих уровней: оба решателя дают одни и те же решения.
    #[test]
    fn joint_rule_matches_linear_when_symmetric() {
        let (rows, cols) = (61usize, 61usize);
        let x = pattern(rows * cols);
        let k = injected();
        let y = apply(&k, &x, rows, cols);
        let label: Vec<usize> = x.iter().map(|&v| (v > 0.0) as usize).collect();
        let g = Grid {
            v: &y,
            rows,
            cols,
        };
        let rule = JointRule::train(&g, &label, &[-1.0, 1.0], 1, 2).expect("обучение");
        let dec = rule.decide_all(&g);
        let mut wrong = 0usize;
        for r in 3..rows - 3 {
            for c in 3..cols - 3 {
                if dec[r * cols + c] != label[r * cols + c] {
                    wrong += 1;
                }
            }
        }
        assert_eq!(wrong, 0, "совместное правило ошиблось {wrong} раз");
    }

    /// ОБУСЛОВЛЕННОСТЬ обращения: у измеренного ядра спектр далёк от нуля,
    /// поэтому усечённый ряд Неймана почти не трогает шум.
    ///
    /// Это ответ на естественное возражение «деконволюция усиливает шум».
    /// Возражение верно для ядра с провалами в спектре; наше ядро — слабое
    /// низкочастотное размытие, `|H|` не опускается ниже ~0.5, а усиление
    /// дисперсии шума фильтром остаётся в пределах единиц процентов.
    #[test]
    fn inverse_is_well_conditioned_for_measured_strength_kernels() {
        for strength in [0.05f64, 0.24, 0.46] {
            let mut k = IsiKernel::identity(1);
            k.set_neighbour(0, -1, 0.30 * strength);
            k.set_neighbour(0, 1, 0.30 * strength);
            k.set_neighbour(-1, 0, 0.15 * strength);
            k.set_neighbour(1, 0, 0.25 * strength);
            let c = conditioning(&k, 2, 32);
            assert!(
                c.h_min > 0.4,
                "сила {strength}: |H| проваливается до {:.3}",
                c.h_min
            );
            assert!(
                c.noise_gain < 1.4,
                "сила {strength}: шум усилен в {:.3} раза",
                c.noise_gain
            );
            assert!(
                c.residual < 0.2,
                "сила {strength}: остаток ISI после усечения {:.3}",
                c.residual
            );
        }
    }

    /// КРЕСТ обнуляет диагонали и оставляет рёберные отсчёты нетронутыми.
    #[test]
    fn cross_shape_zeroes_diagonals() {
        let (rows, cols) = (61usize, 61usize);
        let x = pattern(rows * cols);
        let k = injected();
        let y = apply(&k, &x, rows, cols);
        let est = estimate_kernel(
            &Grid { v: &y, rows, cols },
            &Grid { v: &x, rows, cols },
            1,
            2,
            KernelShape::Cross,
        )
        .expect("оценка ядра");
        for &(a, b) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            assert_eq!(est.neighbour(a, b), 0.0, "диагональ ({a},{b}) не обнулена");
        }
        assert!((est.neighbour(0, 1) - 0.060).abs() < 1e-6);
        assert!((est.neighbour(1, 0) - 0.025).abs() < 1e-6);
    }

    /// DFE снимает помеху и НЕ расходится: коэффициент петли мал.
    #[test]
    fn dfe_converges_with_small_loop_gain() {
        let (rows, cols) = (61usize, 61usize);
        let x = pattern(rows * cols);
        let k = injected();
        let y = apply(&k, &x, rows, cols);
        let mut p = y.clone();
        let rep = equalise_dfe(&mut p, rows, cols, &k, &|v| if v > 0.0 { 1.0 } else { -1.0 });
        assert!(rep.loop_gain < 0.2, "петля {}", rep.loop_gain);
        let err = (3..rows - 3)
            .flat_map(|r| (3..cols - 3).map(move |c| (r, c)))
            .filter(|&(r, c)| (p[r * cols + c] > 0.0) != (x[r * cols + c] > 0.0))
            .count();
        assert_eq!(err, 0);
    }
}
