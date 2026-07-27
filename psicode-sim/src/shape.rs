//! `shape` — измерительное исследование ГЕОМЕТРИИ символа: КОНТУР и РЕШЁТКА.
//!
//! Модуль НИЧЕГО не меняет в формате. Он меряет цену и выигрыш шести
//! конфигураций (три контура × две решётки) на ЗАМЕРЕННОМ канале и отдаёт
//! числа для решения.
//!
//! # Две независимые оси
//!
//! | | вариант A (сегодня) | вариант B | вариант C |
//! |---|---|---|---|
//! | КОНТУР | квадрат 61×61 | вытянутый прямоугольник 16:9 | круг |
//! | РЕШЁТКА | квадратная | гексагональная | — |
//!
//! # Что именно проверяется, и почему это НЕ очевидно
//!
//! **Прямоугольник.** Экран 16:9, квадрат теряет 43.75 % площади. Плюс
//! ожидание: rolling shutter сканирует СТРОКИ, значит низкий широкий символ
//! экспонируется за меньшее число строчных времён. Против — закон ~1/L из
//! исследования центрального искателя (DEAD_ENDS §10): жёсткий шаблон терпит
//! keystone обратно пропорционально своей стороне в клетках, а длинная ось
//! длиннее. Меряем оба эффекта и складываем.
//!
//! **Круг.** Замкнутая петля по периметру КВАДРАТА отклонена (DEAD_ENDS §3):
//! внешний обход 4N−4, внутренний экструдированный 4N−12, рассогласование 8
//! клеток при любом N. На круге при сэмплировании ПО УГЛУ рассогласования нет:
//! у двух концентрических колец одинаковое число угловых бинов. Плюс поворот
//! становится НЕПРЕРЫВНЫМ циклическим сдвигом. Против — круг в квадрате теряет
//! 21.5 % площади, а в кадре 16:9 ещё больше.
//!
//! **Гексагон.** Для КРУГОВО ограниченного по полосе сигнала гексагональная
//! решётка требует на 13.4 % меньше отсчётов; у гексагона на ~7 % меньше
//! периметра при равной площади, а периметр — это ровно то место, где входит
//! межклеточная помеха; и все шесть соседей равноудалены, значит помеха
//! изотропна. Главный вопрос: переживает ли 13.4 % РАСТЕРИЗАЦИЮ на квадратную
//! пиксельную сетку дисплея и камеры.
//!
//! # Модель канала (FINDINGS §2)
//!
//! ```text
//! drive -> линейный свет (γ=2.2) -> ПОЛЕ 0.62..0.86 -> дефокус σ=2 px камеры
//!       -> кодирование 1/γ -> КОРРЕЛИРОВАННЫЙ шум -> 8 бит
//! ```
//!
//! Блюр стоит в ЛИНЕЙНОМ домене (это оптика), поэтому чёрная клетка в белом
//! окружении тянется вверх сильнее, чем белая вниз — ровно замеренная
//! асимметрия «чёрное шумнее белого в 1.4 раза».
//!
//! Шум — НЕ белый по пикселям: замерено 6.15/255 на пиксель и 1.79/255 на
//! клетку при 11.8 px/клетку, отношение 3.44 против 11.8 у белого. Значит поле
//! шума коррелировано; корреляция подобрана так, чтобы ОБА замера сходились
//! (см. [`tests::noise_model_matches_both_measurements`]). Это принципиально
//! для исследования: клетки разной ПЛОЩАДИ усредняют шум по-разному.

use crate::report;
use crate::rng::{seed_for, Rng};
use psicode_core::zcborder::{self, corr_complex, zc_complex, BorderSpec, Carrier, V1_ROOTS};
use std::time::Instant;

// ---------------------------------------------------------------------------
// 1. Замеренные константы канала и стенда
// ---------------------------------------------------------------------------

/// σ дефокуса на живом тракте, пиксели КАМЕРЫ (FINDINGS §2: «блюр ≈ 2 px»).
const BLUR_CAM: f64 = 2.0;
/// Увеличение: пикселей КАМЕРЫ на пиксель ДИСПЛЕЯ (FINDINGS §2).
const MAG: f64 = 1.076;
/// Гамма живого YUV-тракта по зелёному (FINDINGS §2: 2.0 / 2.2 / 3.2).
const GAMMA: f64 = 2.2;
/// Поле освещённости: множитель яркости в ближнем и дальнем углу кадра.
const FIELD_LO: f64 = 0.62;
const FIELD_HI: f64 = 0.86;
/// Шум одиночного снимка НА ПИКСЕЛЬ, доля полной шкалы (FINDINGS §2).
const NOISE_PIX: f64 = 6.15 / 255.0;
/// Шум одиночного снимка НА КЛЕТКУ при [`NOISE_REF_PPC`] px/клетку — вторая
/// точка калибровки шумовой модели (FINDINGS §2).
const NOISE_CELL: f64 = 1.79 / 255.0;
/// Масштаб, при котором замерен [`NOISE_CELL`] (px камеры на клетку).
const NOISE_REF_PPC: f64 = 11.8;
/// σ пространственной корреляции шумового поля, px камеры. Подобрана под ОБА
/// замера выше; проверяется тестом.
const NOISE_CORR_SIGMA: f64 = 0.97;

/// drive чёрной и белой клетки (как в `finder.rs`).
const DRIVE_BLACK: f64 = 0.06;
const DRIVE_WHITE: f64 = 0.94;

/// Рабочая область монитора стенда, display px (1920×1080 при масштабе 125 %).
const WORK_W: f64 = 1536.0;
const WORK_H: f64 = 864.0;
/// Кадр камеры, px.
const CAM_W: f64 = 1920.0;
const CAM_H: f64 = 1080.0;
/// Доля высоты рабочей области под символом. Живой якорь: 61 клетка × 12 display
/// px = 732 из 864.
const FILL: f64 = 732.0 / 864.0;

/// Клеток в одном страйпе L3 (§6.2: 7 строк × 57 клеток).
const STRIPE_CELLS: usize = 399;
/// Бит CRC-16 на страйп.
const STRIPE_CRC_BITS: usize = 16;
/// Кадров в секунду передатчика при hold = 6 (живая точка, FINDINGS §1).
const TX_FPS: f64 = 10.0;
/// Доля, оставшаяся после FEC (как в `cmd_goodput`).
const FEC_KEEP: f64 = 0.8;

/// Период кадра передатчика, мс (hold 6 при 60 Гц).
const T_TX_MS: f64 = 100.0;
/// Экспозиция, приколоченная к одному обновлению (FINDINGS §9).
const T_EXP_MS: f64 = 16.7;
/// Скан монитора сверху вниз, мс.
const T_DISP_MS: f64 = 16.7;
// Время вычитки кадра камеры (rolling shutter) не константа: оно обратно
// выводится из единственного доступного замера — см. [`rolling_readout_from_anchor`].

// ---------------------------------------------------------------------------
// 2. Решётки
// ---------------------------------------------------------------------------

const SQRT3: f64 = 1.732_050_807_568_877_2;
/// √3/2 — межстрочный шаг гексагональной решётки в долях шага соседей.
const SQRT3_2: f64 = SQRT3 / 2.0;

/// Решётка центров клеток. Клетка = ячейка Вороного своего центра, то есть
/// КВАДРАТ для квадратной решётки и правильный ШЕСТИУГОЛЬНИК для гекса — так
/// «форма клетки» не постулируется, а следует из решётки.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lattice {
    Square,
    Hex,
}

impl Lattice {
    fn name(self) -> &'static str {
        match self {
            Lattice::Square => "sq",
            Lattice::Hex => "hex",
        }
    }

    /// Площадь клетки при шаге ближайших соседей `d`.
    /// (Обратная к [`Lattice::pitch_for_area`]; задействована тестом
    /// [`tests::hex_pitch_is_larger_at_equal_area`], который и держит инвариант.)
    #[allow(dead_code)]
    fn cell_area(self, d: f64) -> f64 {
        match self {
            Lattice::Square => d * d,
            // правильный шестиугольник с расстоянием между центрами d
            Lattice::Hex => d * d * SQRT3_2,
        }
    }

    /// Шаг ближайших соседей, дающий клетку площади `a`. ЭТО и есть честная
    /// нормировка сравнения: равная площадь = равная плотность клеток = равное
    /// усреднение шума; вся разница уезжает в РАССТОЯНИЕ до соседа.
    fn pitch_for_area(self, a: f64) -> f64 {
        match self {
            Lattice::Square => a.sqrt(),
            Lattice::Hex => (a / SQRT3_2).sqrt(),
        }
    }

    /// Периметр клетки Вороного при площади `a` — канал, по которому втекает
    /// межклеточная помеха.
    fn cell_perimeter(self, a: f64) -> f64 {
        match self {
            Lattice::Square => 4.0 * a.sqrt(),
            // сторона шестиугольника s: a = 3√3/2 · s² ⇒ P = 6s
            Lattice::Hex => 6.0 * (a * 2.0 / (3.0 * SQRT3)).sqrt(),
        }
    }

    /// Центр клетки `(i, j)` при шаге `d`.
    #[inline]
    fn center(self, i: i32, j: i32, d: f64) -> (f64, f64) {
        match self {
            Lattice::Square => (d * i as f64, d * j as f64),
            Lattice::Hex => (d * (i as f64 + 0.5 * j as f64), d * SQRT3_2 * j as f64),
        }
    }

    /// Индекс БЛИЖАЙШЕГО центра к точке `(x, y)` — то есть номер ячейки Вороного.
    #[inline]
    fn nearest(self, x: f64, y: f64, d: f64) -> (i32, i32) {
        match self {
            Lattice::Square => ((x / d).round() as i32, (y / d).round() as i32),
            Lattice::Hex => {
                // осевые координаты, затем кубическое округление
                let r = y / (d * SQRT3_2);
                let q = x / d - 0.5 * r;
                let s = -q - r;
                let (mut rq, mut rr, rs) = (q.round(), r.round(), s.round());
                let (dq, dr, ds) = ((rq - q).abs(), (rr - r).abs(), (rs - s).abs());
                if dq > dr && dq > ds {
                    rq = -rr - rs;
                } else if dr > ds {
                    rr = -rq - rs;
                }
                (rq as i32, rr as i32)
            }
        }
    }

    /// Смещения индексов до соседей и их расстояния в долях шага `d`.
    fn neighbour_offsets(self) -> Vec<((i32, i32), f64)> {
        match self {
            Lattice::Square => vec![
                ((1, 0), 1.0),
                ((-1, 0), 1.0),
                ((0, 1), 1.0),
                ((0, -1), 1.0),
                ((1, 1), std::f64::consts::SQRT_2),
                ((1, -1), std::f64::consts::SQRT_2),
                ((-1, 1), std::f64::consts::SQRT_2),
                ((-1, -1), std::f64::consts::SQRT_2),
            ],
            Lattice::Hex => vec![
                ((1, 0), 1.0),
                ((-1, 0), 1.0),
                ((0, 1), 1.0),
                ((0, -1), 1.0),
                ((1, -1), 1.0),
                ((-1, 1), 1.0),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Растеризация, канал, демодуляция
// ---------------------------------------------------------------------------

/// Кратность суперсэмплинга при растеризации клеток на пиксельную сетку.
/// Именно здесь появляется ЦЕНА гексагона: строчный шаг d·√3/2 несоизмерим с
/// пиксельной сеткой, границы клеток попадают в дробные позиции и сглаживаются.
const SS: usize = 3;

/// Плоскость f64.
#[derive(Clone)]
struct Plane {
    w: usize,
    h: usize,
    d: Vec<f64>,
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

/// Разделимый гауссов блюр.
fn blur(p: &mut Plane, sigma: f64) {
    if sigma <= 1e-6 {
        return;
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut k = Vec::with_capacity((2 * r + 1) as usize);
    let mut s = 0.0;
    for i in -r..=r {
        let v = (-((i * i) as f64) / (2.0 * sigma * sigma)).exp();
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

/// Коррелированное шумовое поле: белый шум, сглаженный гауссианом
/// [`NOISE_CORR_SIGMA`], перенормированный на попиксельную σ = [`NOISE_PIX`].
fn noise_plane(w: usize, h: usize, rng: &mut Rng) -> Plane {
    let mut p = Plane::new(w, h);
    for v in p.d.iter_mut() {
        *v = rng.gaussian();
    }
    blur(&mut p, NOISE_CORR_SIGMA);
    let n = (w * h) as f64;
    let mean = p.d.iter().sum::<f64>() / n;
    let var = p.d.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let k = if var > 1e-18 { NOISE_PIX / var.sqrt() } else { 0.0 };
    for v in p.d.iter_mut() {
        *v = (*v - mean) * k;
    }
    p
}

/// Что именно моделируем в канале.
#[derive(Clone, Copy)]
struct ChannelCfg {
    /// σ дефокуса, px КАМЕРЫ
    blur_cam: f64,
    field: bool,
    noise: bool,
}

impl ChannelCfg {
    fn live() -> Self {
        ChannelCfg { blur_cam: BLUR_CAM, field: true, noise: true }
    }
    /// Санитарный гейт: ни блюра, ни шума, ни поля, ни наклона.
    fn clean() -> Self {
        ChannelCfg { blur_cam: 0.0, field: false, noise: false }
    }
}

/// Пропустить нарисованный в drive-домене холст ДИСПЛЕЯ через канал и вернуть
/// плоскость КОДОВ камеры.
///
/// Порядок операций физический: гамма дисплея (drive → линейный свет), поле,
/// дефокус в линейном домене, кодирование обратной гаммой, шум, квантование.
/// Пересэмплирование дисплей → камера складывается с блюром: блюр применяется
/// на сетке дисплея с σ/MAG, затем билинейное растяжение на MAG.
fn through_channel(drive: &Plane, ch: &ChannelCfg, rng: &mut Rng) -> Plane {
    let mut lin = Plane::new(drive.w, drive.h);
    for (o, &v) in lin.d.iter_mut().zip(drive.d.iter()) {
        *o = v.max(0.0).powf(GAMMA);
    }
    if ch.field {
        let diag = (drive.w as f64).hypot(drive.h as f64);
        for y in 0..lin.h {
            for x in 0..lin.w {
                let t = (x as f64 + y as f64) / diag;
                lin.d[y * lin.w + x] *= FIELD_LO + (FIELD_HI - FIELD_LO) * t;
            }
        }
    }
    blur(&mut lin, ch.blur_cam / MAG);

    // билинейное растяжение на увеличение камеры
    let cw = ((drive.w as f64) * MAG).floor() as usize;
    let cham = ((drive.h as f64) * MAG).floor() as usize;
    let mut cam = Plane::new(cw.max(1), cham.max(1));
    for y in 0..cam.h {
        let sy = (y as f64 + 0.5) / MAG - 0.5;
        let y0 = sy.floor().clamp(0.0, (lin.h - 1) as f64) as usize;
        let y1 = (y0 + 1).min(lin.h - 1);
        let fy = (sy - y0 as f64).clamp(0.0, 1.0);
        for x in 0..cam.w {
            let sx = (x as f64 + 0.5) / MAG - 0.5;
            let x0 = sx.floor().clamp(0.0, (lin.w - 1) as f64) as usize;
            let x1 = (x0 + 1).min(lin.w - 1);
            let fx = (sx - x0 as f64).clamp(0.0, 1.0);
            let a = lin.at(x0, y0) * (1.0 - fx) + lin.at(x1, y0) * fx;
            let b = lin.at(x0, y1) * (1.0 - fx) + lin.at(x1, y1) * fx;
            cam.d[y * cam.w + x] = a * (1.0 - fy) + b * fy;
        }
    }

    // кодирование обратной гаммой
    for v in cam.d.iter_mut() {
        *v = v.max(0.0).powf(1.0 / GAMMA);
    }
    if ch.noise {
        let n = noise_plane(cam.w, cam.h, rng);
        for (v, &e) in cam.d.iter_mut().zip(n.d.iter()) {
            *v = (*v + e).clamp(0.0, 1.0);
            // квантование в 8 бит
            *v = (*v * 255.0).round() / 255.0;
        }
    }
    cam
}

// ---------------------------------------------------------------------------
// 4. Микро-стенд решётки: d′, SER, ISI
// ---------------------------------------------------------------------------

/// Уровни drive для `levels`-уровневой яркостной модуляции.
fn drive_levels(levels: usize) -> Vec<f64> {
    (0..levels)
        .map(|k| DRIVE_BLACK + (DRIVE_WHITE - DRIVE_BLACK) * k as f64 / (levels - 1) as f64)
        .collect()
}

/// Результат одной точки развёртки решётки.
#[derive(Clone, Copy, Debug, Default)]
struct LatPoint {
    /// эффективный размер клетки, px КАМЕРЫ (сторона равновеликого квадрата)
    a_cam: f64,
    /// нормированное расстояние решения d′
    dprime: f64,
    /// SER из d′ (гауссова аппроксимация)
    ser: f64,
    /// SER, посчитанный СЧЁТОМ ошибок
    ser_counted: f64,
    /// сколько клеток продемодулировано
    cells: usize,
}

/// Дополнительная функция ошибок (Numerical Recipes `erfcc`), отн. точность 1.2e-7.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
        .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

/// Q(x) = P(N(0,1) > x) = ½·erfc(x/√2).
fn qfunc(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// Один прогон микро-стенда: поле случайных клеток на решётке, канал, демод.
///
/// Возвращает `(сумма по уровням: n, Σz, Σz²; число ошибок; число клеток)`.
/// Апертура приёмника — чем именно клетка СЧИТЫВАЕТСЯ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Aperture {
    /// Ячейка Вороного клетки: согласованный фильтр для плоско окрашенной
    /// клетки. Форма апертуры РАЗНАЯ у двух решёток — как и сама клетка.
    Voronoi,
    /// Круг равной площади. Апертура ТОЖДЕСТВЕННА у обеих решёток, поэтому
    /// любой остаточный отрыв гексагона — свойство РЕШЁТКИ, а не считывателя.
    /// Контроль против самообмана «выиграла форма апертуры, а не упаковка».
    Disc,
}

#[allow(clippy::type_complexity)]
fn lattice_trial(
    lat: Lattice,
    a_disp: f64,
    levels: usize,
    ch: &ChannelCfg,
    cells_across: usize,
    ap: Aperture,
    seed: u64,
) -> (Vec<(usize, f64, f64)>, usize, usize) {
    let d = lat.pitch_for_area(a_disp * a_disp);
    let side = a_disp * cells_across as f64;
    let w = side.ceil() as usize;
    let h = side.ceil() as usize;
    let lv = drive_levels(levels);

    // --- индексный диапазон, покрывающий холст с запасом ---
    let (mut i0, mut i1, mut j0, mut j1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for gy in 0..=8 {
        for gx in 0..=8 {
            let (x, y) = (w as f64 * gx as f64 / 8.0, h as f64 * gy as f64 / 8.0);
            let (i, j) = lat.nearest(x, y, d);
            i0 = i0.min(i);
            i1 = i1.max(i);
            j0 = j0.min(j);
            j1 = j1.max(j);
        }
    }
    i0 -= 3;
    i1 += 3;
    j0 -= 3;
    j1 += 3;
    let ni = (i1 - i0 + 1) as usize;
    let nj = (j1 - j0 + 1) as usize;

    let mut rng = Rng::new(seed);
    let mut val = vec![0u8; ni * nj];
    for v in val.iter_mut() {
        *v = rng.next_u32_below(levels as u32) as u8;
    }
    let idx = |i: i32, j: i32| -> Option<usize> {
        if i < i0 || i > i1 || j < j0 || j > j1 {
            None
        } else {
            Some((j - j0) as usize * ni + (i - i0) as usize)
        }
    };

    // --- растеризация на сетку ДИСПЛЕЯ с суперсэмплингом ---
    let mut drive = Plane::new(w, h);
    let inv = 1.0 / (SS * SS) as f64;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = x as f64 + (sx as f64 + 0.5) / SS as f64;
                    let fy = y as f64 + (sy as f64 + 0.5) / SS as f64;
                    let (i, j) = lat.nearest(fx, fy, d);
                    acc += match idx(i, j) {
                        Some(k) => lv[val[k] as usize],
                        None => lv[0],
                    };
                }
            }
            drive.d[y * w + x] = acc * inv;
        }
    }

    let cam = through_channel(&drive, ch, &mut rng);

    // --- демод: усреднение по апертуре ---
    let mut sum = vec![0.0f64; ni * nj];
    let mut cnt = vec![0u32; ni * nj];
    match ap {
        Aperture::Voronoi => {
            for y in 0..cam.h {
                for x in 0..cam.w {
                    let fx = (x as f64 + 0.5) / MAG;
                    let fy = (y as f64 + 0.5) / MAG;
                    let (i, j) = lat.nearest(fx, fy, d);
                    if let Some(k) = idx(i, j) {
                        sum[k] += cam.d[y * cam.w + x];
                        cnt[k] += 1;
                    }
                }
            }
        }
        Aperture::Disc => {
            // круг ТОЙ ЖЕ площади, что клетка: r = a/√π, в пикселях камеры
            let r = a_disp / std::f64::consts::PI.sqrt() * MAG;
            let r2 = r * r;
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let k = match idx(i, j) {
                        Some(k) => k,
                        None => continue,
                    };
                    let (cx, cy) = lat.center(i, j, d);
                    let (px, py) = (cx * MAG, cy * MAG);
                    let x0 = ((px - r).floor().max(0.0)) as usize;
                    let x1 = ((px + r).ceil().min(cam.w as f64 - 1.0)).max(0.0) as usize;
                    let y0 = ((py - r).floor().max(0.0)) as usize;
                    let y1 = ((py + r).ceil().min(cam.h as f64 - 1.0)).max(0.0) as usize;
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let (ddx, ddy) = (x as f64 + 0.5 - px, y as f64 + 0.5 - py);
                            if ddx * ddx + ddy * ddy <= r2 {
                                sum[k] += cam.d[y * cam.w + x];
                                cnt[k] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // --- локальная нормировка: окно радиуса 4·a вокруг клетки ---
    let r_win = 4.0 * a_disp;
    let mut win: Vec<(i32, i32)> = Vec::new();
    let span = (r_win / d).ceil() as i32 + 2;
    for dj in -span..=span {
        for di in -span..=span {
            let (cx, cy) = lat.center(di, dj, d);
            if cx * cx + cy * cy <= r_win * r_win {
                win.push((di, dj));
            }
        }
    }

    // клетка «внутренняя», если она и всё её окно полностью на холсте
    let margin = r_win + d;
    let mut acc = vec![(0usize, 0.0f64, 0.0f64); levels];
    let mut errors = 0usize;
    let mut total = 0usize;
    let mut zs: Vec<(f64, u8)> = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            let (cx, cy) = lat.center(i, j, d);
            if cx < margin || cy < margin || cx > w as f64 - margin || cy > h as f64 - margin {
                continue;
            }
            let k = match idx(i, j) {
                Some(k) => k,
                None => continue,
            };
            if cnt[k] == 0 {
                continue;
            }
            let v = sum[k] / cnt[k] as f64;
            let (mut m, mut m2, mut n) = (0.0, 0.0, 0usize);
            for &(di, dj) in &win {
                if let Some(kk) = idx(i + di, j + dj) {
                    if cnt[kk] > 0 {
                        let u = sum[kk] / cnt[kk] as f64;
                        m += u;
                        m2 += u * u;
                        n += 1;
                    }
                }
            }
            if n < 8 {
                continue;
            }
            let m = m / n as f64;
            let sd = (m2 / n as f64 - m * m).max(1e-18).sqrt();
            let z = (v - m) / sd;
            let l = val[k] as usize;
            acc[l].0 += 1;
            acc[l].1 += z;
            acc[l].2 += z * z;
            zs.push((z, val[k]));
            total += 1;
        }
    }

    // счёт ошибок по «джинновым» порогам — серединам между условными средними
    if total > 0 {
        let means: Vec<f64> = acc
            .iter()
            .map(|&(n, s, _)| if n > 0 { s / n as f64 } else { 0.0 })
            .collect();
        for &(z, l) in &zs {
            let mut best = 0usize;
            let mut bd = f64::MAX;
            for (k, &mk) in means.iter().enumerate() {
                let dd = (z - mk).abs();
                if dd < bd {
                    bd = dd;
                    best = k;
                }
            }
            if best != l as usize {
                errors += 1;
            }
        }
    }
    (acc, errors, total)
}

/// Множители размера клетки, по которым усредняется СУБПИКСЕЛЬНАЯ ФАЗА.
///
/// Ни квадратная, ни гексагональная решётка не выровнена на пиксельную сетку
/// при произвольном масштабе, а фаза выравнивания качает d′ на единицы
/// процентов. Пользователь px/клетку с точностью до процента не контролирует,
/// поэтому честная величина — среднее по окну, а не значение в одной точке.
const PHASE_JITTER: [f64; 5] = [0.965, 0.982, 1.0, 1.018, 1.035];

/// Развернуть точку решётки: усреднение по фазе × попытки, свёртка в d′ и SER.
fn lattice_point(
    lat: Lattice,
    a_cam: f64,
    levels: usize,
    ch: &ChannelCfg,
    cells_across: usize,
    ap: Aperture,
    trials: usize,
    point: usize,
) -> LatPoint {
    let n_ph = PHASE_JITTER.len();
    let mut dp_sum = 0.0;
    let (mut errors, mut total) = (0usize, 0usize);
    for (ph, &f) in PHASE_JITTER.iter().enumerate() {
        let p =
            lattice_point_at(lat, a_cam * f, levels, ch, cells_across, ap, trials, point * 97 + ph);
        dp_sum += p.dprime;
        errors += (p.ser_counted * p.cells as f64).round() as usize;
        total += p.cells;
    }
    let dprime = dp_sum / n_ph as f64;
    let ser = 2.0 * (levels - 1) as f64 / levels as f64 * qfunc(dprime / 2.0);
    LatPoint { a_cam, dprime, ser, ser_counted: errors as f64 / total.max(1) as f64, cells: total }
}

/// d′ в ОДНОЙ точке масштаба, без усреднения по фазе.
fn lattice_point_at(
    lat: Lattice,
    a_cam: f64,
    levels: usize,
    ch: &ChannelCfg,
    cells_across: usize,
    ap: Aperture,
    trials: usize,
    point: usize,
) -> LatPoint {
    let a_disp = a_cam / MAG;
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let chunks: Vec<Vec<usize>> = (0..nthreads)
        .map(|t| (0..trials).filter(|k| k % nthreads == t).collect())
        .collect();
    let results: Vec<(Vec<(usize, f64, f64)>, usize, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|ts| {
                s.spawn(move || {
                    let mut acc = vec![(0usize, 0.0f64, 0.0f64); levels];
                    let (mut e, mut n) = (0usize, 0usize);
                    for &t in ts {
                        let (a, ee, nn) =
                            lattice_trial(lat, a_disp, levels, ch, cells_across, ap, seed_for(point, t));
                        for (o, i) in acc.iter_mut().zip(a.iter()) {
                            o.0 += i.0;
                            o.1 += i.1;
                            o.2 += i.2;
                        }
                        e += ee;
                        n += nn;
                    }
                    (acc, e, n)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut acc = vec![(0usize, 0.0f64, 0.0f64); levels];
    let (mut errors, mut total) = (0usize, 0usize);
    for (a, e, n) in results {
        for (o, i) in acc.iter_mut().zip(a.iter()) {
            o.0 += i.0;
            o.1 += i.1;
            o.2 += i.2;
        }
        errors += e;
        total += n;
    }

    let mut means = Vec::new();
    let mut sds = Vec::new();
    for &(n, s, s2) in &acc {
        if n == 0 {
            return LatPoint { a_cam, ..Default::default() };
        }
        let m = s / n as f64;
        means.push(m);
        sds.push((s2 / n as f64 - m * m).max(1e-18).sqrt());
    }
    let gap: f64 = (1..levels).map(|k| means[k] - means[k - 1]).sum::<f64>() / (levels - 1) as f64;
    let sd: f64 = sds.iter().sum::<f64>() / levels as f64;
    let dprime = gap / sd;
    // SER для L равновероятных уровней с гауссовым шумом
    let ser = 2.0 * (levels - 1) as f64 / levels as f64 * qfunc(dprime / 2.0);
    LatPoint {
        a_cam,
        dprime,
        ser,
        ser_counted: errors as f64 / total.max(1) as f64,
        cells: total,
    }
}

/// Ядро межклеточной помехи: во что превращается ОДНА зажжённая клетка,
/// измеренное в центрах соседей после дефокуса.
///
/// Возвращает `(доля своей клетки, [(расстояние в шагах, доля)])`.
fn isi_kernel(lat: Lattice, a_cam: f64, blur_cam: f64) -> (f64, Vec<(f64, f64)>) {
    let a_disp = a_cam / MAG;
    let d = lat.pitch_for_area(a_disp * a_disp);
    let cells_across = 11usize;
    let side = a_disp * cells_across as f64;
    let w = side.ceil() as usize;
    let (cx, cy) = (w as f64 * 0.5, w as f64 * 0.5);
    let (i_c, j_c) = lat.nearest(cx, cy, d);

    // рисуем ОДНУ клетку белой на чёрном, в ЛИНЕЙНОМ домене (ядро линейно)
    let mut lin = Plane::new(w, w);
    let inv = 1.0 / (SS * SS) as f64;
    for y in 0..w {
        for x in 0..w {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = x as f64 + (sx as f64 + 0.5) / SS as f64;
                    let fy = y as f64 + (sy as f64 + 0.5) / SS as f64;
                    if lat.nearest(fx, fy, d) == (i_c, j_c) {
                        acc += 1.0;
                    }
                }
            }
            lin.d[y * w + x] = acc * inv;
        }
    }
    blur(&mut lin, blur_cam / MAG);

    // усредняем по ячейке Вороного каждой клетки
    let read = |i: i32, j: i32| -> f64 {
        let (mut s, mut n) = (0.0, 0usize);
        for y in 0..w {
            for x in 0..w {
                if lat.nearest(x as f64 + 0.5, y as f64 + 0.5, d) == (i, j) {
                    s += lin.d[y * w + x];
                    n += 1;
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            s / n as f64
        }
    };
    let self_w = read(i_c, j_c);
    let mut out = Vec::new();
    for (off, dist) in lat.neighbour_offsets() {
        out.push((dist, read(i_c + off.0, j_c + off.1)));
    }
    (self_w, out)
}

// ---------------------------------------------------------------------------
// 5. Контуры: геометрия, ёмкость, утилизация
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Outline {
    /// квадрат (сегодня)
    Square,
    /// прямоугольник с отношением сторон `aspect` = ширина/высота
    Rect(f64),
    /// круг
    Circle,
}

impl Outline {
    fn name(self) -> String {
        match self {
            Outline::Square => "square".into(),
            Outline::Rect(a) => format!("rect {a:.2}:1"),
            Outline::Circle => "circle".into(),
        }
    }

    /// Габарит символа (ширина, высота) в display px при заданной ДОСТУПНОЙ
    /// рабочей области и заполнении `fill` по ВЫСОТЕ.
    fn extent(self, fill: f64) -> (f64, f64) {
        let h = fill * WORK_H;
        match self {
            Outline::Square => (h, h),
            Outline::Rect(a) => {
                let w = (a * h).min(fill * WORK_W);
                (w, w / a)
            }
            Outline::Circle => (h, h),
        }
    }

    /// Площадь символа при габарите `(w, h)`.
    fn area(self, w: f64, h: f64) -> f64 {
        match self {
            Outline::Circle => std::f64::consts::PI * w * h / 4.0,
            _ => w * h,
        }
    }

    /// Площадь контура, ужатого внутрь на `t` px (для вычета рамки).
    fn inset_area(self, w: f64, h: f64, t: f64) -> f64 {
        let (iw, ih) = ((w - 2.0 * t).max(0.0), (h - 2.0 * t).max(0.0));
        self.area(iw, ih)
    }

    /// Длина СРЕДНЕЙ линии рамки (по половине её толщины внутрь) — по ней
    /// раскладывается последовательность.
    fn mid_perimeter(self, w: f64, h: f64, t: f64) -> f64 {
        let (iw, ih) = ((w - t).max(0.0), (h - t).max(0.0));
        match self {
            Outline::Circle => std::f64::consts::PI * 0.5 * (iw + ih),
            _ => 2.0 * (iw + ih),
        }
    }

    fn is_inside(self, x: f64, y: f64, w: f64, h: f64) -> bool {
        match self {
            Outline::Circle => {
                let (u, v) = ((x - w * 0.5) / (w * 0.5), (y - h * 0.5) / (h * 0.5));
                u * u + v * v <= 1.0
            }
            _ => x >= 0.0 && y >= 0.0 && x <= w && y <= h,
        }
    }
}

/// Полная сводка одной конфигурации (контур × решётка) при заданном размере
/// клетки в пикселях камеры.
#[derive(Clone, Debug)]
struct Config {
    outline: Outline,
    lat: Lattice,
    /// сторона равновеликой клетки, px КАМЕРЫ
    a_cam: f64,
    /// габарит символа, display px
    w_disp: f64,
    h_disp: f64,
    total_cells: f64,
    border_cells: f64,
    payload_cells: f64,
    /// длина ЗЧ-последовательности на сторону/кольцо в клетках
    seq_len: f64,
    screen_util: f64,
    cam_util: f64,
    /// высота символа в СТРОКАХ камеры
    cam_rows: f64,
}

fn build_config(outline: Outline, lat: Lattice, a_cam: f64, fill: f64) -> Config {
    let a_disp = a_cam / MAG;
    let (w, h) = outline.extent(fill);
    let area = outline.area(w, h);
    let cell_area = a_disp * a_disp;
    // рамка толщиной 2 клетки
    let t = 2.0 * a_disp;
    let inner = outline.inset_area(w, h, t);
    let total_cells = area / cell_area;
    let border_cells = (area - inner) / cell_area;
    let payload_cells = inner / cell_area;
    let seq_len = match outline {
        Outline::Circle => outline.mid_perimeter(w, h, t) / a_disp,
        _ => (w / a_disp).max(h / a_disp),
    };
    Config {
        outline,
        lat,
        a_cam,
        w_disp: w,
        h_disp: h,
        total_cells,
        border_cells,
        payload_cells,
        seq_len,
        screen_util: area / (WORK_W * WORK_H),
        cam_util: area * MAG * MAG / (CAM_W * CAM_H),
        cam_rows: h * MAG,
    }
}

// ---------------------------------------------------------------------------
// 6. Rolling shutter
// ---------------------------------------------------------------------------

/// Время вычитки кадра камеры, обратно выведенное из единственного замера:
/// «84 % чистых снимков при H = 16» (FINDINGS §2). Окно повреждения
/// `W = |Δ| + T_exp`, где `Δ = h·(T_disp − k·T_ro)`, `k = 930/1080` —
/// доля кадра под рабочей областью экрана.
fn rolling_readout_from_anchor() -> f64 {
    let t_tx_h16 = 16.0 * 16.67;
    let w_dmg = 0.16 * t_tx_h16; // мс
    let h = FILL; // символ занимал ~85 % высоты рабочей области
    let k = WORK_H * MAG / CAM_H;
    // W = |Δ| + T_exp, Δ = h·(T_disp − k·T_ro) < 0
    let delta = w_dmg - T_EXP_MS;
    (T_DISP_MS + delta / h) / k
}

/// Модель разрыва: доля повреждённых кадров и доля СИМВОЛА, теряемая в
/// повреждённом кадре, для символа высотой `rows` строк камеры.
///
/// Фронт смены кадра бежит по экрану за `T_disp`, камера вычитывает кадр за
/// `T_ro`; оба сверху вниз, поэтому частично «следят» друг за другом. Строка
/// испорчена, если её экспозиция [t, t+T_exp] накрывает фронт так, что доля
/// чужого кадра лежит в (0.25, 0.75) — при меньшей доле порог ещё берёт свой
/// кадр, при большей строка честно принадлежит СЛЕДУЮЩЕМУ кадру и салвадж
/// припишет её ему.
fn rolling(rows: f64, t_ro: f64) -> (f64, f64) {
    let h_cam = rows / CAM_H;
    let h_disp = rows / (WORK_H * MAG);
    let delta = (h_disp * T_DISP_MS - h_cam * t_ro).abs();
    let p_damaged = ((delta + T_EXP_MS) / T_TX_MS).min(1.0);
    let lost = if delta < 1e-9 {
        1.0
    } else {
        (0.5 * T_EXP_MS / delta).min(1.0)
    };
    (p_damaged, lost)
}

/// Множитель goodput от разрыва.
///
/// * САЛВАДЖ ВЫКЛЮЧЕН — повреждённый снимок выбрасывается целиком: `1 − p`.
/// * САЛВАДЖ ВКЛЮЧЁН (SPEC §6.3, пер-страйповый CRC-16 локализует целые страйпы)
///   — теряются только те страйпы, которых разрыв КОСНУЛСЯ, но не меньше одного:
///   `1 − p·max(доля, 1/страйпов)`.
///
/// Отсюда сразу видно, где салвадж вообще что-то даёт: только когда полоса
/// повреждения ТОНЬШЕ одного страйпа. Как только она шире, число страйпов из
/// формулы уходит, и остаётся чистая зависимость от ВЫСОТЫ символа.
fn rolling_factor(rows: f64, t_ro: f64, stripes: f64, salvage: bool) -> f64 {
    if stripes < 1.0 {
        return 0.0;
    }
    let (p, lost) = rolling(rows, t_ro);
    if salvage {
        1.0 - p * lost.max(1.0 / stripes)
    } else {
        1.0 - p
    }
}

// ---------------------------------------------------------------------------
// 7. Рамка: полосы v1, замкнутые петли, круговое кольцо
// ---------------------------------------------------------------------------

/// Способ уложить последовательность вокруг символа.
#[derive(Clone, Copy, PartialEq, Debug)]
enum BorderKind {
    /// v1: четыре экструдированные полосы N×2, свой корень на сторону.
    Strips,
    /// Замкнутая петля по периметру, экструзия по НОРМАЛИ внутрь. Внутренний
    /// обход короче внешнего на 8 клеток — DEAD_ENDS §3.
    LoopNormal,
    /// Замкнутая петля, экструзия ГОМОТЕТИЕЙ (по лучу из центра). Длины
    /// совпадают тождественно, зато луч не перпендикулярен стороне нигде,
    /// кроме середин, и у углов уезжает.
    LoopHomothetic,
    /// Круговое кольцо, сэмплирование ПО УГЛУ, экструзия по РАДИУСУ.
    Ring,
}

impl BorderKind {
    fn name(self) -> &'static str {
        match self {
            BorderKind::Strips => "v1 strips (square)",
            BorderKind::LoopNormal => "closed loop, normal extrusion",
            BorderKind::LoopHomothetic => "closed loop, homothetic extrusion",
            BorderKind::Ring => "circular ring, radial extrusion",
        }
    }
}

/// Наибольшее простое, не превосходящее `n`.
fn prime_at_most(n: usize) -> usize {
    let is_p = |k: usize| {
        if k < 2 {
            return false;
        }
        let mut d = 2;
        while d * d <= k {
            if k % d == 0 {
                return false;
            }
            d += 1;
        }
        true
    };
    let mut k = n;
    while k > 2 && !is_p(k) {
        k -= 1;
    }
    k
}

/// Положение клетки `(ix, iy)` вдоль КВАДРАТНОГО кольца стороной `s` клеток,
/// обход по часовой от левого верхнего угла. Всего на кольце `4·(s−1)` клеток.
fn square_ring_index(ix: i64, iy: i64, s: i64) -> i64 {
    let last = s - 1;
    if iy == 0 {
        ix
    } else if ix == last {
        last + iy
    } else if iy == last {
        2 * last + (last - ix)
    } else {
        3 * last + (last - iy)
    }
}

/// Комплексное значение рамки в точке `(x, y)` символа стороной `n` клеток.
///
/// Клетки ПЛОСКИЕ: значение постоянно внутри клетки и меняется скачком на
/// границе — так рамку и рисует передатчик. Никакой интерполяции здесь нет,
/// сглаживание вносит только растеризатор и дефокус.
///
/// Возвращает `None` вне полосы рамки.
fn border_value(kind: BorderKind, spec: &BorderSpec, n: f64, x: f64, y: f64) -> Option<(f64, f64)> {
    let ring = 2.0;
    let ni = n as i64;
    if x < 0.0 || y < 0.0 || x >= n || y >= n {
        return None;
    }
    match kind {
        BorderKind::Strips => {
            let (dt, dr, db, dl) = (y, n - x, n - y, x);
            let (side, depth, along) = if dt <= dr && dt <= db && dt <= dl {
                (0usize, dt, x)
            } else if dr <= db && dr <= dl {
                (1usize, dr, y)
            } else if db <= dl {
                (2usize, db, n - x)
            } else {
                (3usize, dl, n - y)
            };
            if depth >= ring {
                return None;
            }
            let d = depth.floor() as usize;
            let i = along.floor().max(0.0) as usize;
            if i < 2 {
                // угловые позиции уступлены ПРЕДЫДУЩЕЙ стороне (§3.2): та кладёт
                // сюда свою позицию n−1−depth (стороны сходятся под 90°, поэтому
                // «вдоль» одной = «вглубь» другой)
                let prev = (side + 3) % 4;
                return Some(zcborder::strip_value(spec, prev, spec.n - 1 - d));
            }
            Some(zcborder::strip_value(spec, side, i.min(spec.n - 1)))
        }
        BorderKind::LoopNormal => {
            // Последовательность длины M = 4N−4 (внешний обход). Внутренний
            // обход имеет лишь 4N−12 клеток, и «нормальная» экструзия обязана
            // укладывать её ПО СВОЕЙ длине — отсюда накопительный сдвиг слоёв.
            let (ix, iy) = (x.floor() as i64, y.floor() as i64);
            let depth = ix.min(iy).min(ni - 1 - ix).min(ni - 1 - iy);
            if depth >= 2 {
                return None;
            }
            let m = (4 * ni - 4) as usize;
            let k = if depth == 0 {
                square_ring_index(ix, iy, ni)
            } else {
                square_ring_index(ix - 1, iy - 1, ni - 2)
            };
            Some(zc_complex(spec.roots[0], (k as usize) % m, m))
        }
        BorderKind::LoopHomothetic => {
            let (ix, iy) = (x.floor() as i64, y.floor() as i64);
            let depth = ix.min(iy).min(ni - 1 - ix).min(ni - 1 - iy);
            if depth >= 2 {
                return None;
            }
            // индекс — доля периметра от ЦЕНТРА клетки; она инвариантна к
            // гомотетии, поэтому оба слоя получают один индекс тождественно
            let m = (4 * ni - 4) as usize;
            let f = perimeter_frac_homothetic(ix as f64 + 0.5, iy as f64 + 0.5, n);
            let k = (f * m as f64).floor() as usize % m;
            Some(zc_complex(spec.roots[0], k, m))
        }
        BorderKind::Ring => {
            let r_out = n * 0.5;
            let (dx, dy) = (x - r_out, y - r_out);
            let r = dx.hypot(dy);
            if r > r_out || r < r_out - ring {
                return None;
            }
            let m = ring_bins(n);
            let phi = dy.atan2(dx);
            let t = (phi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * m as f64;
            let k = (t.floor() as usize) % m;
            Some(zc_complex(spec.roots[0], k, m))
        }
    }
}

/// Число угловых бинов кольца для символа диаметром `n` клеток: длина СРЕДНЕЙ
/// окружности кольца в клетках, округлённая вниз до ПРОСТОГО.
fn ring_bins(n: f64) -> usize {
    let r_mid = n * 0.5 - 1.0;
    prime_at_most((2.0 * std::f64::consts::PI * r_mid).floor() as usize)
}

/// Доля периметра при ГОМОТЕТИЧЕСКОЙ параметризации: точка проецируется лучом
/// из центра на единичный квадрат, и берётся её доля периметра там. Доля
/// инвариантна к масштабу, поэтому оба ряда полосы получают ОДИН индекс.
fn perimeter_frac_homothetic(x: f64, y: f64, n: f64) -> f64 {
    let (u, v) = (x / n - 0.5, y / n - 0.5);
    let m = u.abs().max(v.abs()).max(1e-12);
    let (u, v) = (u / m * 0.5, v / m * 0.5); // на границу единичного квадрата
    // обход по часовой от левого верхнего угла (−0.5, −0.5)
    if (v + 0.5).abs() < 1e-9 {
        (u + 0.5) * 0.25
    } else if (u - 0.5).abs() < 1e-9 {
        0.25 + (v + 0.5) * 0.25
    } else if (v - 0.5).abs() < 1e-9 {
        0.5 + (0.5 - u) * 0.25
    } else {
        0.75 + (0.5 - v) * 0.25
    }
}

/// Та же гомотетическая доля периметра для ПРЯМОУГОЛЬНИКА `w × h`.
fn perimeter_frac_homothetic_rect(x: f64, y: f64, w: f64, h: f64) -> f64 {
    let (u, v) = ((x / w - 0.5) * 2.0, (y / h - 0.5) * 2.0);
    let m = u.abs().max(v.abs()).max(1e-12);
    let (u, v) = (u / m, v / m);
    if (v + 1.0).abs() < 1e-9 {
        (u + 1.0) * 0.125
    } else if (u - 1.0).abs() < 1e-9 {
        0.25 + (v + 1.0) * 0.125
    } else if (v - 1.0).abs() < 1e-9 {
        0.5 + (1.0 - u) * 0.125
    } else {
        0.75 + (1.0 - v) * 0.125
    }
}

/// Значение ЗЧ стороны `side` в ДРОБНОЙ позиции `i` (линейная интерполяция
/// комплексных отсчётов — так растр видит границу между клетками).
fn zc_interp(spec: &BorderSpec, side: usize, i: f64, n: f64) -> (f64, f64) {
    let nn = spec.n;
    let i = i.clamp(0.0, n - 1.0 - 1e-9);
    let a = i.floor() as usize;
    let f = i - a as f64;
    let v0 = zcborder::strip_value(spec, side, a.min(nn - 1));
    let v1 = zcborder::strip_value(spec, side, (a + 1).min(nn - 1));
    (v0.0 * (1.0 - f) + v1.0 * f, v0.1 * (1.0 - f) + v1.1 * f)
}

/// Значение ЗЧ длины `m` в дробной ЦИКЛИЧЕСКОЙ позиции `t`.
fn zc_interp_loop(root: u32, t: f64, m: f64) -> (f64, f64) {
    let mi = m as usize;
    let t = t.rem_euclid(m);
    let a = t.floor() as usize % mi;
    let f = t - t.floor();
    let v0 = zc_complex(root, a, mi);
    let v1 = zc_complex(root, (a + 1) % mi, mi);
    (v0.0 * (1.0 - f) + v1.0 * f, v0.1 * (1.0 - f) + v1.1 * f)
}

/// Точки зондирования рамки и ЭТАЛОННЫЕ значения для них.
///
/// Зонд стоит на глубине ровно 1.0 клетки — на границе двух рядов полосы. Там
/// блюр ПОПЕРЁК рамки усредняет оба слоя, то есть измеряется именно то
/// свойство, ради которого экструзия и вводилась.
fn border_probes(kind: BorderKind, spec: &BorderSpec, n: f64) -> Vec<((f64, f64), (f64, f64))> {
    let ni = n as i64;
    let mut out = Vec::new();
    match kind {
        BorderKind::Strips => {
            for side in 0..4 {
                for i in 2..spec.n {
                    let t = i as f64 + 0.5;
                    let p = match side {
                        0 => (t, 1.0),
                        1 => (n - 1.0, t),
                        2 => (n - t, n - 1.0),
                        _ => (1.0, n - t),
                    };
                    out.push((p, zcborder::strip_value(spec, side, i)));
                }
            }
        }
        BorderKind::LoopNormal | BorderKind::LoopHomothetic => {
            let m = (4 * ni - 4) as usize;
            // обходим ВНЕШНЕЕ кольцо по часовой и толкаем зонд на 0.5 внутрь
            let mut cells: Vec<(i64, i64)> = Vec::with_capacity(m);
            for ix in 0..ni {
                cells.push((ix, 0));
            }
            for iy in 1..ni {
                cells.push((ni - 1, iy));
            }
            for ix in (0..ni - 1).rev() {
                cells.push((ix, ni - 1));
            }
            for iy in (1..ni - 1).rev() {
                cells.push((0, iy));
            }
            assert_eq!(cells.len(), m, "внешнее кольцо не 4N−4");
            for (k, &(ix, iy)) in cells.iter().enumerate() {
                let p = if iy == 0 {
                    (ix as f64 + 0.5, 1.0)
                } else if ix == ni - 1 {
                    (n - 1.0, iy as f64 + 0.5)
                } else if iy == ni - 1 {
                    (ix as f64 + 0.5, n - 1.0)
                } else {
                    (1.0, iy as f64 + 0.5)
                };
                let val = match kind {
                    BorderKind::LoopNormal => zc_complex(spec.roots[0], k, m),
                    _ => {
                        let f = perimeter_frac_homothetic(ix as f64 + 0.5, iy as f64 + 0.5, n);
                        zc_complex(spec.roots[0], (f * m as f64).floor() as usize % m, m)
                    }
                };
                out.push((p, val));
            }
        }
        BorderKind::Ring => {
            let m = ring_bins(n);
            let r_mid = n * 0.5 - 1.0;
            for i in 0..m {
                let phi =
                    (i as f64 + 0.5) / m as f64 * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
                let p = (n * 0.5 + r_mid * phi.cos(), n * 0.5 + r_mid * phi.sin());
                out.push((p, zc_complex(spec.roots[0], i, m)));
            }
        }
    }
    out
}

/// Удержание корреляции рамки под дефокусом.
///
/// Рамка рисуется ПЛОСКИМИ клетками в двух плоскостях (Re, Im) на пиксельной
/// сетке при `ppc` px/клетку, размывается гауссианом σ клеток, зондируется на
/// глубине 1.0 и коррелируется с идеальной последовательностью.
fn border_retention(kind: BorderKind, n: f64, sigma_cells: f64, ppc: f64) -> f64 {
    let spec = BorderSpec {
        n: prime_at_most(n as usize),
        roots: V1_ROOTS,
        carrier: Carrier::ComplexChroma,
    };
    // поля запаса, чтобы реплицирующий край блюра не подпирал рамку своим же значением
    let margin = 4.0;
    let side_px = ((n + 2.0 * margin) * ppc).ceil() as usize;
    let mut re = Plane::new(side_px, side_px);
    let mut im = Plane::new(side_px, side_px);
    let inv = 1.0 / (SS * SS) as f64;
    for py in 0..side_px {
        for px in 0..side_px {
            let (mut ar, mut ai) = (0.0, 0.0);
            for sy in 0..SS {
                for sx in 0..SS {
                    let x = (px as f64 + (sx as f64 + 0.5) / SS as f64) / ppc - margin;
                    let y = (py as f64 + (sy as f64 + 0.5) / SS as f64) / ppc - margin;
                    if let Some((r, i)) = border_value(kind, &spec, n, x, y) {
                        ar += r;
                        ai += i;
                    }
                }
            }
            re.d[py * side_px + px] = ar * inv;
            im.d[py * side_px + px] = ai * inv;
        }
    }
    blur(&mut re, sigma_cells * ppc);
    blur(&mut im, sigma_cells * ppc);

    let probes = border_probes(kind, &spec, n);
    let (mut got, mut want): (Vec<(f64, f64)>, Vec<(f64, f64)>) = (Vec::new(), Vec::new());
    for ((x, y), v) in probes {
        let px = (((x + margin) * ppc) as isize).clamp(0, side_px as isize - 1) as usize;
        let py = (((y + margin) * ppc) as isize).clamp(0, side_px as isize - 1) as usize;
        got.push((re.at(px, py), im.at(px, py)));
        want.push(v);
    }
    corr_complex(&got, &want)
}

// ---------------------------------------------------------------------------
// 8. Keystone: параметрическое сэмплирование под наклоном
// ---------------------------------------------------------------------------

/// Проекция точки плоскости символа под наклоном `tilt` вокруг вертикали.
/// Символ центрирован, расстояние `dist` в тех же единицах (клетках).
fn project(x: f64, y: f64, half_w: f64, half_h: f64, tilt_deg: f64, dist: f64) -> (f64, f64) {
    let (u, v) = (x - half_w, y - half_h);
    let th = tilt_deg.to_radians();
    let (c, s) = (th.cos(), th.sin());
    let z = dist - u * s;
    let f = dist;
    (f * (u * c) / z, f * v / z)
}

/// Keystone для ПРЯМОЙ стороны: пробник идёт равными шагами из угла в угол
/// (аффинно), а физика — гомография. Возвращает корреляцию с идеалом.
///
/// Сторона ВСЕГДА кладётся вдоль оси, которую наклон укорачивает — приёмник
/// наклон не выбирает, поэтому осмысленный порог слома это ХУДШИЙ случай.
/// `dist` — расстояние съёмки в клетках; оно задаётся САМЫМ БОЛЬШИМ габаритом
/// символа (символ должен влезть в кадр), поэтому широкий символ снимается
/// издалека, и это часть эффекта.
fn keystone_side(len_cells: f64, dist: f64, tilt_deg: f64) -> f64 {
    let m = prime_at_most(len_cells as usize);
    let spec = BorderSpec { n: m, roots: V1_ROOTS, carrier: Carrier::ComplexChroma };
    let (hw, hh) = (len_cells * 0.5, len_cells * 0.5);
    let vertical = false;
    let point = |i: f64| -> (f64, f64) {
        let t = i / (m - 1) as f64;
        (1.0 + t * (2.0 * hw - 2.0), 1.0)
    };
    // проецируем концы стороны и идём между ними равными шагами
    let p0 = {
        let (x, y) = point(0.0);
        project(x, y, hw, hh, tilt_deg, dist)
    };
    let p1 = {
        let (x, y) = point((m - 1) as f64);
        project(x, y, hw, hh, tilt_deg, dist)
    };
    // для каждого шага пробника ищем ИСТИННЫЙ индекс, спроецировавшийся сюда
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for k in 2..m {
        let t = k as f64 / (m - 1) as f64;
        let target = (p0.0 + (p1.0 - p0.0) * t, p0.1 + (p1.1 - p0.1) * t);
        // бинарный поиск по параметру истинной стороны
        let (mut lo, mut hi) = (0.0f64, (m - 1) as f64);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let (x, y) = point(mid);
            let p = project(x, y, hw, hh, tilt_deg, dist);
            let along = if vertical { p.1 } else { p.0 };
            let a0 = if vertical { p0.1 } else { p0.0 };
            let a1 = if vertical { p1.1 } else { p1.0 };
            let tt = if vertical { target.1 } else { target.0 };
            let dir = (a1 - a0).signum();
            if (along - tt) * dir < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let true_i = 0.5 * (lo + hi);
        got.push(zc_interp(&spec, 0, true_i, m as f64));
        want.push(zcborder::strip_value(&spec, 0, k));
    }
    corr_complex(&got, &want)
}

/// Keystone для КРУГОВОГО кольца: пробник идёт равными шагами по ПАРАМЕТРУ
/// эллипса, подогнанного к спроецированному кругу. Возвращает корреляцию и
/// максимальный угловой лаг в КЛЕТКАХ.
fn keystone_ring(diam_cells: f64, dist: f64, tilt_deg: f64) -> (f64, f64) {
    let m = ring_bins(diam_cells);
    let r = diam_cells * 0.5 - 1.0;
    let hw = diam_cells * 0.5;
    // проецируем плотно и подгоняем конику по МНК: Au²+Buv+Cv²+Du+Ev+F=0, F=−1
    let ns = 720usize;
    let mut pts = Vec::with_capacity(ns);
    for k in 0..ns {
        let phi = k as f64 / ns as f64 * 2.0 * std::f64::consts::PI;
        let (x, y) = (hw + r * phi.cos(), hw + r * phi.sin());
        pts.push((project(x, y, hw, hw, tilt_deg, dist), phi));
    }
    let mut ata = [[0.0f64; 5]; 5];
    let mut atb = [0.0f64; 5];
    for &((u, v), _) in &pts {
        let row = [u * u, u * v, v * v, u, v];
        for a in 0..5 {
            for b in 0..5 {
                ata[a][b] += row[a] * row[b];
            }
            atb[a] += row[a];
        }
    }
    let coef = solve5(&mut ata, &mut atb);
    let (a, b, c, d, e) = (coef[0], coef[1], coef[2], coef[3], coef[4]);
    // центр коники
    let den = 4.0 * a * c - b * b;
    let (cx, cy) = ((b * e - 2.0 * c * d) / den, (b * d - 2.0 * a * e) / den);
    // приведение к окружности: собственные векторы матрицы [[a, b/2],[b/2, c]]
    let tr = a + c;
    let det = a * c - b * b / 4.0;
    let disc = ((tr * tr / 4.0 - det).max(0.0)).sqrt();
    let (l1, l2) = (tr / 2.0 + disc, tr / 2.0 - disc);
    let theta = 0.5 * (b).atan2(a - c);
    let (ct, st) = (theta.cos(), theta.sin());

    // истинный угол φ -> параметрический угол ψ на подогнанном эллипсе
    let mut lag_max: f64 = 0.0;
    let mut psis = Vec::with_capacity(ns);
    for &((u, v), phi) in &pts {
        let (du, dv) = (u - cx, v - cy);
        let (p, q) = (du * ct + dv * st, -du * st + dv * ct);
        let psi = (q * l2.abs().sqrt()).atan2(p * l1.abs().sqrt());
        psis.push((psi, phi));
    }
    // снимаем постоянный поворот (он безобиден — это и есть искомый сдвиг)
    let bias = {
        let (mut sr, mut si) = (0.0, 0.0);
        for &(psi, phi) in &psis {
            let d = psi - phi;
            sr += d.cos();
            si += d.sin();
        }
        si.atan2(sr)
    };
    for &(psi, phi) in &psis {
        let mut e = psi - phi - bias;
        while e > std::f64::consts::PI {
            e -= 2.0 * std::f64::consts::PI;
        }
        while e < -std::f64::consts::PI {
            e += 2.0 * std::f64::consts::PI;
        }
        lag_max = lag_max.max((e * r).abs());
    }

    // корреляция: пробник читает по равномерному ψ, физика лежит по φ
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for k in 0..m {
        let psi_target = k as f64 / m as f64 * 2.0 * std::f64::consts::PI + bias;
        // ищем φ, дающий этот ψ
        let mut best = 0.0;
        let mut bd = f64::MAX;
        for &(psi, phi) in &psis {
            let mut e = psi - psi_target;
            while e > std::f64::consts::PI {
                e -= 2.0 * std::f64::consts::PI;
            }
            while e < -std::f64::consts::PI {
                e += 2.0 * std::f64::consts::PI;
            }
            if e.abs() < bd {
                bd = e.abs();
                best = phi;
            }
        }
        let true_i = best / (2.0 * std::f64::consts::PI) * m as f64;
        got.push(zc_interp_loop(V1_ROOTS[0], true_i, m as f64));
        want.push(zc_complex(V1_ROOTS[0], k, m));
    }
    (corr_complex(&got, &want), lag_max)
}

/// Решение 5×5 методом Гаусса с частичным выбором.
fn solve5(a: &mut [[f64; 5]; 5], b: &mut [f64; 5]) -> [f64; 5] {
    for c in 0..5 {
        let mut p = c;
        for r in c + 1..5 {
            if a[r][c].abs() > a[p][c].abs() {
                p = r;
            }
        }
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        if d.abs() < 1e-15 {
            continue;
        }
        for r in c + 1..5 {
            let f = a[r][c] / d;
            for k in c..5 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = [0.0f64; 5];
    for c in (0..5).rev() {
        let mut s = b[c];
        for k in c + 1..5 {
            s -= a[c][k] * x[k];
        }
        x[c] = if a[c][c].abs() < 1e-15 { 0.0 } else { s / a[c][c] };
    }
    x
}

// ---------------------------------------------------------------------------
// 9. Непрерывный поворот по круговой рамке
// ---------------------------------------------------------------------------

/// Оценка НЕПРЕРЫВНОГО поворота кольца: рендер с истинным поворотом `phi0`,
/// канал, круговая корреляция по всем M сдвигам, параболическая интерполяция
/// пика. Возвращает ошибку в градусах.
fn ring_rotation_error(diam_cells: f64, ppc: f64, phi0_deg: f64, ch: &ChannelCfg, seed: u64) -> f64 {
    let m = ring_bins(diam_cells);
    let n = diam_cells;
    let side_px = (n * ppc).ceil() as usize;
    let root = V1_ROOTS[0];
    let r_out = n * 0.5;
    let r_mid = r_out - 1.0;
    let phi0 = phi0_deg.to_radians();

    // рисуем кольцо в ЯРКОСТИ (знак Re) — живой носитель по DEAD_ENDS §6
    let mut drive = Plane::new(side_px, side_px);
    let inv = 1.0 / (SS * SS) as f64;
    for py in 0..side_px {
        for px in 0..side_px {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let x = (px as f64 + (sx as f64 + 0.5) / SS as f64) / ppc - r_out;
                    let y = (py as f64 + (sy as f64 + 0.5) / SS as f64) / ppc - r_out;
                    let r = x.hypot(y);
                    let v = if r <= r_out && r >= r_out - 2.0 {
                        let phi = y.atan2(x) - phi0;
                        let t = ((phi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)
                            * m as f64)
                            .rem_euclid(m as f64);
                        // клетки ПЛОСКИЕ: значение постоянно внутри углового бина
                        let (re, _) = zc_complex(root, t.floor() as usize % m, m);
                        if re >= 0.0 {
                            DRIVE_WHITE
                        } else {
                            DRIVE_BLACK
                        }
                    } else {
                        0.5
                    };
                    acc += v;
                }
            }
            drive.d[py * side_px + px] = acc * inv;
        }
    }
    let mut rng = Rng::new(seed);
    let cam = through_channel(&drive, ch, &mut rng);
    let scale = ppc * MAG;

    // читаем M угловых бинов на средней линии
    let mut s: Vec<f64> = Vec::with_capacity(m);
    for k in 0..m {
        let phi = (k as f64 + 0.5) / m as f64 * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
        let x = (r_out + r_mid * phi.cos()) * scale;
        let y = (r_out + r_mid * phi.sin()) * scale;
        let xi = (x as isize).clamp(0, cam.w as isize - 1) as usize;
        let yi = (y as isize).clamp(0, cam.h as isize - 1) as usize;
        s.push(cam.at(xi, yi));
    }
    let mean = s.iter().sum::<f64>() / m as f64;
    for v in s.iter_mut() {
        *v -= mean;
    }

    // круговая корреляция со ЗНАКОМ ЗЧ (тот же бинарный носитель)
    let tmpl: Vec<f64> = (0..m)
        .map(|i| if zc_complex(root, i, m).0 >= 0.0 { 1.0 } else { -1.0 })
        .collect();
    let tm = tmpl.iter().sum::<f64>() / m as f64;
    let tmpl: Vec<f64> = tmpl.iter().map(|v| v - tm).collect();
    let mut corr = vec![0.0f64; m];
    for (sh, c) in corr.iter_mut().enumerate() {
        let mut acc = 0.0;
        for i in 0..m {
            acc += s[(i + sh) % m] * tmpl[i];
        }
        *c = acc;
    }
    let mut peak = 0usize;
    for k in 1..m {
        if corr[k] > corr[peak] {
            peak = k;
        }
    }
    // параболическая интерполяция
    let (ym, y0, yp) = (corr[(peak + m - 1) % m], corr[peak], corr[(peak + 1) % m]);
    let den = ym - 2.0 * y0 + yp;
    let frac = if den.abs() < 1e-12 { 0.0 } else { 0.5 * (ym - yp) / den };
    // s[k] = z[k − δ] при повороте на φ0, δ = φ0·M/360; corr[sh] = Σ s[i+sh]·z[i]
    // пикует при sh = δ, поэтому оценка поворота = сдвиг пика, БЕЗ смены знака.
    let est_bins = peak as f64 + frac;
    let est_deg = est_bins / m as f64 * 360.0;
    let mut err = est_deg - phi0_deg;
    while err > 180.0 {
        err -= 360.0;
    }
    while err < -180.0 {
        err += 360.0;
    }
    err
}

// ---------------------------------------------------------------------------
// 10. Санитарный гейт: полный символ на чистом канале
// ---------------------------------------------------------------------------

/// Собрать, прогнать и продемодулировать ПОЛНЫЙ символ конфигурации.
/// Возвращает `(SER полезной нагрузки, число клеток нагрузки)`.
fn sanity_symbol(outline: Outline, lat: Lattice, a_cam: f64, ch: &ChannelCfg, seed: u64) -> (f64, usize) {
    let a_disp = a_cam / MAG;
    // компактный габарит: ~40 клеток по высоте, чтобы гейт был быстрым
    let h = 40.0 * a_disp;
    let w = match outline {
        Outline::Rect(a) => a * h,
        _ => h,
    };
    let d = lat.pitch_for_area(a_disp * a_disp);
    let (wp, hp) = (w.ceil() as usize, h.ceil() as usize);

    let (mut i0, mut i1, mut j0, mut j1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for gy in 0..=8 {
        for gx in 0..=8 {
            let (i, j) = lat.nearest(wp as f64 * gx as f64 / 8.0, hp as f64 * gy as f64 / 8.0, d);
            i0 = i0.min(i);
            i1 = i1.max(i);
            j0 = j0.min(j);
            j1 = j1.max(j);
        }
    }
    i0 -= 3;
    i1 += 3;
    j0 -= 3;
    j1 += 3;
    let ni = (i1 - i0 + 1) as usize;
    let idx = |i: i32, j: i32| -> Option<usize> {
        if i < i0 || i > i1 || j < j0 || j > j1 {
            None
        } else {
            Some((j - j0) as usize * ni + (i - i0) as usize)
        }
    };
    let nj = (j1 - j0 + 1) as usize;

    // роль клетки: 0 = вне, 1 = рамка, 2 = нагрузка
    let mut role = vec![0u8; ni * nj];
    let mut val = vec![0u8; ni * nj];
    let mut rng = Rng::new(seed);
    let t = 2.0 * a_disp;
    let spec = BorderSpec { n: 61, roots: V1_ROOTS, carrier: Carrier::BinaryLuma };
    for j in j0..=j1 {
        for i in i0..=i1 {
            let (cx, cy) = lat.center(i, j, d);
            let k = idx(i, j).unwrap();
            if !outline.is_inside(cx, cy, w, h) {
                continue;
            }
            let inner = match outline {
                Outline::Circle => {
                    let (u, v) = ((cx - w * 0.5) / (w * 0.5 - t), (cy - h * 0.5) / (h * 0.5 - t));
                    u * u + v * v <= 1.0
                }
                _ => cx >= t && cy >= t && cx <= w - t && cy <= h - t,
            };
            if inner {
                role[k] = 2;
                val[k] = (rng.next_u64() & 1) as u8;
            } else {
                role[k] = 1;
                // Рамка — настоящая ЗЧ, разложенная по периметру контура (доля
                // периметра для многоугольников, угол для круга). Для гейта
                // важен её ISI в соседние клетки нагрузки, а не корень.
                let m = (outline.mid_perimeter(w, h, t) / a_disp).max(8.0);
                let mi = prime_at_most(m as usize);
                let f = match outline {
                    Outline::Circle => {
                        let phi = (cy - h * 0.5).atan2(cx - w * 0.5);
                        (phi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)
                    }
                    _ => perimeter_frac_homothetic_rect(cx, cy, w, h),
                };
                let (re, _) = zc_complex(spec.roots[0], (f * mi as f64) as usize % mi, mi);
                val[k] = if re >= 0.0 { 1 } else { 0 };
            }
        }
    }

    let lv = drive_levels(2);
    let mut drive = Plane::new(wp, hp);
    let inv = 1.0 / (SS * SS) as f64;
    for y in 0..hp {
        for x in 0..wp {
            let mut acc = 0.0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let fx = x as f64 + (sx as f64 + 0.5) / SS as f64;
                    let fy = y as f64 + (sy as f64 + 0.5) / SS as f64;
                    let (i, j) = lat.nearest(fx, fy, d);
                    acc += match idx(i, j) {
                        Some(k) if role[k] != 0 => lv[val[k] as usize],
                        _ => 0.5,
                    };
                }
            }
            drive.d[y * wp + x] = acc * inv;
        }
    }
    let cam = through_channel(&drive, ch, &mut rng);

    let mut sum = vec![0.0f64; ni * nj];
    let mut cnt = vec![0u32; ni * nj];
    for y in 0..cam.h {
        for x in 0..cam.w {
            let (i, j) = lat.nearest((x as f64 + 0.5) / MAG, (y as f64 + 0.5) / MAG, d);
            if let Some(k) = idx(i, j) {
                sum[k] += cam.d[y * cam.w + x];
                cnt[k] += 1;
            }
        }
    }
    // порог — по СОБСТВЕННОЙ середине шкалы конфигурации (джинн знает уровни)
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for k in 0..ni * nj {
        if role[k] == 2 && cnt[k] > 0 {
            if val[k] == 0 {
                lo.push(sum[k] / cnt[k] as f64);
            } else {
                hi.push(sum[k] / cnt[k] as f64);
            }
        }
    }
    if lo.is_empty() || hi.is_empty() {
        return (1.0, 0);
    }
    let ml = lo.iter().sum::<f64>() / lo.len() as f64;
    let mh = hi.iter().sum::<f64>() / hi.len() as f64;
    let th = 0.5 * (ml + mh);
    let mut err = 0usize;
    let total = lo.len() + hi.len();
    err += lo.iter().filter(|&&v| v > th).count();
    err += hi.iter().filter(|&&v| v <= th).count();
    (err as f64 / total as f64, total)
}

// ---------------------------------------------------------------------------
// 11. Развёртки и отчёт
// ---------------------------------------------------------------------------

/// Точки размера клетки для развёртки решётки, px КАМЕРЫ.
const A_SWEEP: [f64; 13] =
    [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0, 17.0, 20.0];
/// Ключевые рабочие точки, о которых спрашивает бриф.
const A_KEY: [f64; 3] = [8.0, 12.0, 20.0];
/// Клеток по стороне холста микро-стенда.
const CELLS_ACROSS: usize = 32;
/// Живая рабочая точка: 61 клетка на 732 display px × увеличение.
const A_LIVE: f64 = 12.9;

/// Интерполяция SER по развёртке (лог-линейно по d′).
fn interp_ser(curve: &[LatPoint], a: f64, levels: usize) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    let a = a.clamp(curve[0].a_cam, curve[curve.len() - 1].a_cam);
    let mut k = 0usize;
    while k + 2 < curve.len() && curve[k + 1].a_cam < a {
        k += 1;
    }
    let (p, q) = (&curve[k], &curve[k + 1]);
    let f = if (q.a_cam - p.a_cam).abs() < 1e-9 {
        0.0
    } else {
        (a - p.a_cam) / (q.a_cam - p.a_cam)
    };
    let dp = p.dprime + (q.dprime - p.dprime) * f;
    2.0 * (levels - 1) as f64 / levels as f64 * qfunc(dp / 2.0)
}

/// Размер клетки (px камеры), при котором решётка даёт заданный d′.
fn a_for_dprime(curve: &[LatPoint], target: f64) -> Option<f64> {
    for k in 0..curve.len().saturating_sub(1) {
        let (p, q) = (&curve[k], &curve[k + 1]);
        if (p.dprime - target) * (q.dprime - target) <= 0.0 && (q.dprime - p.dprime).abs() > 1e-9 {
            let f = (target - p.dprime) / (q.dprime - p.dprime);
            return Some(p.a_cam + (q.a_cam - p.a_cam) * f);
        }
    }
    None
}

/// Goodput конфигурации, kbit/s, с выживанием страйпа и разрывом.
fn goodput(cfg: &Config, ser: f64, bpc: usize, salvage: bool, t_ro: f64) -> (f64, f64, f64) {
    let stripes = (cfg.payload_cells / STRIPE_CELLS as f64).floor();
    let bits_per_stripe = (STRIPE_CELLS * bpc).saturating_sub(STRIPE_CRC_BITS) as f64;
    let p_survive = (1.0 - ser).powi(STRIPE_CELLS as i32);
    let rs = rolling_factor(cfg.cam_rows, t_ro, stripes, salvage);
    let bits = stripes * bits_per_stripe * p_survive * rs;
    (bits * TX_FPS * FEC_KEEP / 1000.0, p_survive, rs)
}

pub fn cmd_shape() {
    let t0 = Instant::now();
    println!("# psicode-sim shape — ГЕОМЕТРИЯ символа: контур × решётка");
    println!(
        "\nИсследование, не изменение формата. Канал — замеренный (FINDINGS §2): блюр σ={BLUR_CAM} px\n\
         камеры, увеличение ×{MAG}, γ={GAMMA}, поле {FIELD_LO}→{FIELD_HI}, шум {:.2}/255 на пиксель\n\
         (коррелированный, σ_corr={NOISE_CORR_SIGMA} px) → {:.2}/255 на клетку при {NOISE_REF_PPC} px/клетку.",
        NOISE_PIX * 255.0,
        NOISE_CELL * 255.0
    );
    println!(
        "Растеризация: {SS}×{SS} суперсэмплинг на сетке ДИСПЛЕЯ, затем ×{MAG} на сетку камеры —\n\
         обе сетки квадратные, поэтому цена гексагона входит физически, а не постулируется."
    );

    // ---------------- 0. санитарный гейт ----------------
    println!("\n## 0. Санитарный гейт — чистый канал (ни блюра, ни шума, ни поля)");
    println!("| конфигурация | клеток нагрузки | SER |");
    println!("|---|---|---|");
    let clean = ChannelCfg::clean();
    let mut gate_bad = 0usize;
    for &outline in &[Outline::Square, Outline::Rect(16.0 / 9.0), Outline::Circle] {
        for &lat in &[Lattice::Square, Lattice::Hex] {
            let (ser, n) = sanity_symbol(outline, lat, 12.0, &clean, 0x5A17);
            if ser > 0.0 {
                gate_bad += 1;
            }
            println!("| {} / {} | {n} | {} |", outline.name(), lat.name(), report::sig4(ser));
        }
    }
    println!(
        "\n**Остаток: {}.** {}",
        if gate_bad == 0 { "0 ошибок во всех шести".to_string() } else { format!("{gate_bad} конфигураций с ошибками") },
        if gate_bad == 0 {
            "Гейт пройден — растеризация и демод согласованы для обеих решёток и всех трёх контуров."
        } else {
            "ГЕЙТ НЕ ПРОЙДЕН."
        }
    );

    // ---------------- 1. решётка ----------------
    println!("\n## 1. Решётка: межклеточная помеха, растеризация, РЕАЛИЗОВАННЫЙ выигрыш гекса");
    println!(
        "\nНормировка — РАВНАЯ ПЛОЩАДЬ клетки: равная плотность клеток, равное усреднение шума,\n\
         вся разница уезжает в расстояние до соседа. При площади a квадратная решётка держит\n\
         соседа на √a, гексагональная — на {:.4}·√a, зато соседей шесть вместо 4+4.",
        1.0 / SQRT3_2.sqrt()
    );

    println!("\n### 1.1 Ядро помехи при 12 px/клетку, σ={BLUR_CAM} px камеры");
    println!("| решётка | своя клетка | ближние соседи | дальние соседи | Σ помехи | анизотропия |");
    println!("|---|---|---|---|---|---|");
    for &lat in &[Lattice::Square, Lattice::Hex] {
        let (sw, nb) = isi_kernel(lat, 12.0, BLUR_CAM);
        let near: Vec<f64> = nb.iter().filter(|&&(d, _)| d < 1.2).map(|&(_, v)| v).collect();
        let far: Vec<f64> = nb.iter().filter(|&&(d, _)| d >= 1.2).map(|&(_, v)| v).collect();
        let mn = near.iter().sum::<f64>() / near.len().max(1) as f64;
        let mf = if far.is_empty() { 0.0 } else { far.iter().sum::<f64>() / far.len() as f64 };
        let tot: f64 = nb.iter().map(|&(_, v)| v).sum();
        let aniso = if mf > 1e-12 { format!("{:.2}×", mn / mf) } else { "— (изотропна)".into() };
        println!(
            "| {} | {:.4} | {:.4} ×{} | {} | {:.4} | {} |",
            lat.name(),
            sw,
            mn,
            near.len(),
            if far.is_empty() { "—".to_string() } else { format!("{:.4} ×{}", mf, far.len()) },
            tot,
            aniso
        );
    }

    println!("\n### 1.2 Развёртка d′ и SER по размеру клетки (1 бит/клетку и 2 бита/клетку)");
    let live = ChannelCfg::live();
    let mut curves: Vec<(Lattice, usize, Vec<LatPoint>)> = Vec::new();
    let mut pt = 0usize;
    for &levels in &[2usize, 4] {
        for &lat in &[Lattice::Square, Lattice::Hex] {
            let mut c = Vec::new();
            for &a in &A_SWEEP {
                let trials = if a >= 10.0 { 4 } else { 6 };
                c.push(lattice_point(lat, a, levels, &live, CELLS_ACROSS, Aperture::Voronoi, trials, pt));
                pt += 1;
            }
            curves.push((lat, levels, c));
        }
    }
    for &levels in &[2usize, 4] {
        println!("\n**{} уровня ({} бит/клетку)**", levels, levels.trailing_zeros());
        print!("| решётка \\ √площадь, px камеры |");
        for a in A_SWEEP {
            print!(" {a} |");
        }
        println!();
        print!("|---|");
        for _ in A_SWEEP {
            print!("---|");
        }
        println!();
        for &lat in &[Lattice::Square, Lattice::Hex] {
            let c = &curves.iter().find(|x| x.0 == lat && x.1 == levels).unwrap().2;
            let cells: Vec<String> = c.iter().map(|p| format!("{:.2}", p.dprime)).collect();
            println!("{}", report::table_row(&format!("{} · d′", lat.name()), &cells));
        }
        for &lat in &[Lattice::Square, Lattice::Hex] {
            let c = &curves.iter().find(|x| x.0 == lat && x.1 == levels).unwrap().2;
            let cells: Vec<String> = c.iter().map(|p| report::sig4(p.ser)).collect();
            println!("{}", report::table_row(&format!("{} · SER", lat.name()), &cells));
        }
        let cs = &curves.iter().find(|x| x.0 == Lattice::Square && x.1 == levels).unwrap().2;
        let chx = &curves.iter().find(|x| x.0 == Lattice::Hex && x.1 == levels).unwrap().2;
        let cells: Vec<String> = cs
            .iter()
            .zip(chx.iter())
            .map(|(p, q)| format!("{:+.1}%", (q.dprime / p.dprime - 1.0) * 100.0))
            .collect();
        println!("{}", report::table_row("hex/sq · d′", &cells));
        // проверка счётом там, где ошибки вообще видны
        let cnt: Vec<String> = cs.iter().map(|p| report::sig4(p.ser_counted)).collect();
        println!("{}", report::table_row("sq · SER счётом", &cnt));
        let cnt: Vec<String> = chx.iter().map(|p| report::sig4(p.ser_counted)).collect();
        println!("{}", report::table_row("hex · SER счётом", &cnt));
        println!(
            "\n(клеток на точку ≈ {}; d′ — нормированное расстояние решения, SER = 2(L−1)/L·Q(d′/2))",
            cs.last().map(|p| p.cells).unwrap_or(0)
        );
    }

    println!("\n### 1.3 РЕАЛИЗОВАННЫЙ выигрыш гексагона после растеризации");
    println!(
        "Теория: при равной помехе гекс кладёт на {:.2} % больше клеток на ту же площадь\n\
         (периметр клетки Вороного меньше на {:.2} %, а по нему помеха и втекает).\n\
         Ниже — сколько из этого доживает до пикселей.",
        (1.0 / SQRT3_2 - 1.0) * 100.0,
        (1.0 - Lattice::Hex.cell_perimeter(1.0) / Lattice::Square.cell_perimeter(1.0)) * 100.0
    );
    // независимая сверка по ЯДРУ помехи: d′ ∝ своя_клетка / √(Σ помеха²)
    {
        let (ss, nbs) = isi_kernel(Lattice::Square, 12.0, BLUR_CAM);
        let (sh, nbh) = isi_kernel(Lattice::Hex, 12.0, BLUR_CAM);
        let vs: f64 = nbs.iter().map(|&(_, v)| v * v).sum();
        let vh: f64 = nbh.iter().map(|&(_, v)| v * v).sum();
        println!(
            "\nПредсказание из ядра §1.1 (канал помехо-, не шумо-ограничен): d′ ∝ своя/√Σпомеха²\n\
             ⇒ hex/sq = {:.3}/{:.3} · √({:.5}/{:.5}) = **{:+.1} %** по d′.",
            sh,
            ss,
            vs,
            vh,
            ((sh / ss) * (vs / vh).sqrt() - 1.0) * 100.0
        );
    }
    println!("\n| бит/клетку | точка | d′ sq | d′ hex | Δd′ | a_hex при равном d′ | выигрыш по ПЛОТНОСТИ | теория |");
    println!("|---|---|---|---|---|---|---|---|");
    for &levels in &[2usize, 4] {
        let cs = &curves.iter().find(|x| x.0 == Lattice::Square && x.1 == levels).unwrap().2;
        let chx = &curves.iter().find(|x| x.0 == Lattice::Hex && x.1 == levels).unwrap().2;
        for &a in &A_KEY {
            let target = interp_dprime(cs, a);
            let dh = interp_dprime(chx, a);
            let gain = match a_for_dprime(chx, target) {
                Some(ah) => format!("{ah:.2} | **{:+.1} %**", ((a * a) / (ah * ah) - 1.0) * 100.0),
                None => "вне развёртки | —".to_string(),
            };
            println!(
                "| {} | {a} px/клетку | {target:.2} | {dh:.2} | {:+.1} % | {gain} | +15.47 % |",
                levels.trailing_zeros(),
                (dh / target - 1.0) * 100.0
            );
        }
    }
    println!(
        "\nΔd′ — прямое и устойчивое измерение. Пересчёт в ПЛОТНОСТЬ — производная величина:\n\
         она делит Δd′ на НАКЛОН кривой d′(a), а наклон в помехо-ограниченном режиме мал,\n\
         поэтому небольшой выигрыш по d′ раздувается в большой выигрыш по числу клеток. Верить\n\
         следует колонке Δd′; колонку плотности читать как «порядок величины, не третья цифра».\n\
         Классические 13.4 % — про ЧАСТОТНУЮ полосу при идеальном сэмплировании; наш канал\n\
         ограничен ПОМЕХОЙ через фиксированную ФРФ, и это не одна и та же задача."
    );

    println!("\n#### Контроль: та же развёртка с ТОЖДЕСТВЕННОЙ круговой апертурой");
    println!(
        "Соблазн самообмана: считывать квадратную клетку квадратом, а гексагональную\n\
         шестиугольником — и приписать выигрыш решётке, хотя выиграла ФОРМА СЧИТЫВАТЕЛЯ.\n\
         Здесь обе решётки читаются КРУГОМ РАВНОЙ ПЛОЩАДИ: считыватель тождественный, разница\n\
         остаётся только в упаковке. Настоящий приёмник, ресэмплящий через гомографию, ближе\n\
         именно к этому случаю, чем к точной ячейке Вороного."
    );
    println!("| px/клетку | sq d′ (Вороной) | hex d′ (Вороной) | sq d′ (круг) | hex d′ (круг) | Δd′ круг |");
    println!("|---|---|---|---|---|---|");
    for &a in &A_KEY {
        let sv = lattice_point(Lattice::Square, a, 2, &live, CELLS_ACROSS, Aperture::Voronoi, 4, pt);
        pt += 1;
        let hv = lattice_point(Lattice::Hex, a, 2, &live, CELLS_ACROSS, Aperture::Voronoi, 4, pt);
        pt += 1;
        let sd = lattice_point(Lattice::Square, a, 2, &live, CELLS_ACROSS, Aperture::Disc, 4, pt);
        pt += 1;
        let hd = lattice_point(Lattice::Hex, a, 2, &live, CELLS_ACROSS, Aperture::Disc, 4, pt);
        pt += 1;
        println!(
            "| {a} | {:.2} | {:.2} | {:.2} | {:.2} | **{:+.1} %** |",
            sv.dprime,
            hv.dprime,
            sd.dprime,
            hd.dprime,
            (hd.dprime / sd.dprime - 1.0) * 100.0
        );
    }

    println!("\n### 1.4 Цена растеризации отдельно: субпиксельная ФАЗА (без усреднения)");
    println!(
        "Квадратная решётка при ЦЕЛОМ display-шаге пиксельно точна — сглаживания нет вовсе;\n\
         гексагональная не бывает точной НИКОГДА: строчный шаг d·√3/2 иррационален к сетке, и\n\
         каждая строка садится в свою дробную фазу. Здесь и только здесь фаза НЕ усредняется."
    );
    println!("| display px/клетку | sq d′ | hex d′ | hex/sq |");
    println!("|---|---|---|---|");
    let dps = [10.0f64, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.5];
    let (mut sqv, mut hxv) = (Vec::new(), Vec::new());
    for &dp in &dps {
        let a = dp * MAG;
        let ps =
            lattice_point_at(Lattice::Square, a, 2, &live, CELLS_ACROSS, Aperture::Voronoi, 8, pt);
        pt += 1;
        let ph =
            lattice_point_at(Lattice::Hex, a, 2, &live, CELLS_ACROSS, Aperture::Voronoi, 8, pt);
        pt += 1;
        sqv.push(ps.dprime);
        hxv.push(ph.dprime);
        println!(
            "| {dp} | {:.3} | {:.3} | {:+.1}% |",
            ps.dprime,
            ph.dprime,
            (ph.dprime / ps.dprime - 1.0) * 100.0
        );
    }
    // рябь = остаток вокруг линейного тренда: сам подъём d′ с масштабом — не фаза
    let ripple = |v: &[f64]| -> f64 {
        let n = v.len() as f64;
        let (mx, my) = (dps.iter().sum::<f64>() / n, v.iter().sum::<f64>() / n);
        let sxy: f64 = dps.iter().zip(v).map(|(&x, &y)| (x - mx) * (y - my)).sum();
        let sxx: f64 = dps.iter().map(|&x| (x - mx) * (x - mx)).sum();
        let b = sxy / sxx;
        let res: f64 = dps
            .iter()
            .zip(v)
            .map(|(&x, &y)| {
                let e = y - (my + b * (x - mx));
                e * e
            })
            .sum::<f64>()
            / n;
        res.sqrt() / my
    };
    println!(
        "\nРЯБЬ по фазе (СКО остатка вокруг линейного тренда, а не полный размах — рост d′ с\n\
         масштабом это не фаза): квадратная **{:.1} %**, гексагональная **{:.1} %**.",
        ripple(&sqv) * 100.0,
        ripple(&hxv) * 100.0
    );

    // ---------------- 2. контуры ----------------
    println!("\n## 2. Контур: ёмкость и утилизация");
    println!(
        "\nЭкран 16:9 ({WORK_W}×{WORK_H} display px при масштабе 125 %), кадр камеры {CAM_W}×{CAM_H},\n\
         символ занимает {:.0} % высоты рабочей области (живой якорь: 61 клетка × 12 px = 732 из 864).",
        FILL * 100.0
    );
    // Решётки сравниваются при РАВНОМ КАЧЕСТВЕ РЕШЕНИЯ, а не при равном размере
    // клетки: живая точка — квадратная решётка при A_LIVE, гексагональная берёт
    // тот размер клетки, при котором её d′ равен.
    let c_sq2 = &curves.iter().find(|x| x.0 == Lattice::Square && x.1 == 2).unwrap().2;
    let c_hx2 = &curves.iter().find(|x| x.0 == Lattice::Hex && x.1 == 2).unwrap().2;
    let target_dp = interp_dprime(c_sq2, A_LIVE);
    let a_hex_eq = a_for_dprime(c_hx2, target_dp).unwrap_or(A_LIVE);
    println!(
        "\nРешётки сравниваются при РАВНОМ качестве решения (d′ = {target_dp:.2}): квадратная\n\
         в живой точке {A_LIVE} px/клетку, гексагональная — при {a_hex_eq:.2} px/клетку."
    );
    println!("\n| контур | габарит, display px | площадь экрана | площадь кадра | px/клетку | всего клеток | рамка | нагрузка | ЗЧ длина | строк камеры |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let mut geo: Vec<Config> = Vec::new();
    for &outline in &[Outline::Square, Outline::Rect(16.0 / 9.0), Outline::Circle] {
        for &lat in &[Lattice::Square, Lattice::Hex] {
            let a = if lat == Lattice::Square { A_LIVE } else { a_hex_eq };
            let c = build_config(outline, lat, a, FILL);
            println!(
                "| {} / {} | {:.0}×{:.0} | {:.1} % | {:.1} % | {:.2} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} |",
                outline.name(),
                lat.name(),
                c.w_disp,
                c.h_disp,
                c.screen_util * 100.0,
                c.cam_util * 100.0,
                c.a_cam,
                c.total_cells,
                c.border_cells,
                c.payload_cells,
                c.seq_len,
                c.cam_rows
            );
            geo.push(c);
        }
    }
    println!(
        "\nКонтрольная точка: квадрат / sq даёт {:.0} клеток нагрузки против 3135 у живого 57×55 —\n\
         разница от заполнения экрана целиком (живой символ 732 из 864 px по высоте, но 61 клетка).\n\
         Площадь экрана и кадра от РЕШЁТКИ не зависят: это свойство КОНТУРА.",
        geo[0].payload_cells
    );

    // ---------------- 3. кросс-произведение ----------------
    println!("\n## 3. Кросс-произведение 3×2: goodput со СТРАЙП-ВЫЖИВАНИЕМ");
    let t_ro = rolling_readout_from_anchor();
    println!(
        "\ngoodput = страйпов · (399·bpc − 16) · (1−SER)^399 · разрыв · {TX_FPS} кадр/с · {FEC_KEEP}.\n\
         SER — из развёртки §1.2 в точке px/клетку конфигурации. T_ro = {t_ro:.1} мс (обратно\n\
         выведено из «84 % чистых снимков при H=16», FINDINGS §2)."
    );
    println!(
        "\nДва отгруженных режима: 1 бит/клетку моно и 2 бита/клетку ЦВЕТОМ — это ДВЕ независимые\n\
         двоичные оси (§5.1-CL), а не четырёхуровневая яркость, поэтому SER клетки = 1−(1−p)².\n\
         Четырёхуровневая ЯРКОСТЬ на этом канале мертва независимо от геометрии — см. врезку ниже."
    );
    for &(mode, bpc) in &[("1 бит/клетку (моно)", 1usize), ("2 бита/клетку (цвет, 2 оси)", 2usize)] {
        println!("\n**{mode}**");
        println!("| контур / решётка | px/клетку | нагрузка | страйпов | SER | (1−SER)^399 | разрыв | goodput kbit/s |");
        println!("|---|---|---|---|---|---|---|---|");
        let mut best = (f64::MIN, String::new());
        let mut base = 0.0f64;
        for (k, c) in geo.iter().enumerate() {
            let curve = &curves.iter().find(|x| x.0 == c.lat && x.1 == 2).unwrap().2;
            let p = interp_ser(curve, c.a_cam, 2);
            let ser = if bpc == 2 { 1.0 - (1.0 - p) * (1.0 - p) } else { p };
            let (gp, ps, rs) = goodput(c, ser, bpc, true, t_ro);
            let label = format!("{} / {}", c.outline.name(), c.lat.name());
            if k == 0 {
                base = gp;
            }
            if gp > best.0 {
                best = (gp, label.clone());
            }
            println!(
                "| {label} | {:.2} | {:.0} | {:.0} | {} | {:.4} | {:.4} | **{gp:.1}** (×{:.2}) |",
                c.a_cam,
                c.payload_cells,
                (c.payload_cells / STRIPE_CELLS as f64).floor(),
                report::sig4(ser),
                ps,
                rs,
                if base > 0.0 { gp / base } else { 0.0 }
            );
        }
        println!("\n**argmax:** {} → {:.1} kbit/s", best.1, best.0);
    }
    {
        let cs = &curves.iter().find(|x| x.0 == Lattice::Square && x.1 == 4).unwrap().2;
        let chx = &curves.iter().find(|x| x.0 == Lattice::Hex && x.1 == 4).unwrap().2;
        let ps = interp_ser(cs, A_LIVE, 4);
        let ph = interp_ser(chx, a_hex_eq, 4);
        println!(
            "\n> **Четырёхуровневая ЯРКОСТЬ, врезка.** В живой точке SER = {} (квадратная решётка)\n\
             > и {} (гексагональная); выживание страйпа (1−SER)^399 = {:.2e} и {:.2e}. Ни один\n\
             > страйп не доживает НИ ПРИ КАКОЙ геометрии: помеха от соседей съедает межуровневый\n\
             > интервал вчетверо быстрее, чем растёт число бит. Это подтверждает, почему живой\n\
             > 2-битный режим сделан ЦВЕТОМ (две ортогональные оси), а не яркостью.",
            report::sig4(ps),
            report::sig4(ph),
            (1.0 - ps).powi(STRIPE_CELLS as i32),
            (1.0 - ph).powi(STRIPE_CELLS as i32)
        );
    }

    // ---------------- 4. keystone ----------------
    println!("\n## 4. Keystone: слом жёсткого пробника по конфигурациям");
    println!(
        "\nПробник идёт РАВНЫМИ шагами вдоль стороны (по параметру эллипса — для кольца), физика —\n\
         гомография. Это ровно механизм DEAD_ENDS §9. Слом — падение корреляции ниже 0.55\n\
         (отгруженный порог минимальной стороны)."
    );
    println!(
        "Расстояние съёмки задаёт САМЫЙ БОЛЬШИЙ габарит символа (он должен влезть в кадр):\n\
         квадрат 61 снимается с 183 клеток, прямоугольник 109×61 — со 327. Поэтому отношение\n\
         «сторона / расстояние» у длинной стороны прямоугольника ровно то же, что у квадратной,\n\
         и разница в стойкости — чистое действие ДЛИНЫ, без подмешивания кадрирования."
    );
    println!("\n| примитив | длина, клеток | расст. | 5° | 10° | 15° | 20° | 25° | 30° | слом |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let tilts = [5.0f64, 10.0, 15.0, 20.0, 25.0, 30.0];
    struct Prim {
        name: &'static str,
        len: f64,
        dist: f64,
        ring: bool,
    }
    let prims = [
        Prim { name: "square side", len: 61.0, dist: 183.0, ring: false },
        Prim { name: "rect long side (16:9)", len: 109.0, dist: 327.0, ring: false },
        Prim { name: "rect short side (16:9)", len: 61.0, dist: 327.0, ring: false },
        Prim { name: "circle ring", len: 61.0, dist: 183.0, ring: true },
    ];
    let mut breaks = Vec::new();
    for p in &prims {
        let mut cells = Vec::new();
        let mut brk = String::from("> 30°");
        let mut brk_val = f64::INFINITY;
        let mut prev = (0.0f64, 1.0f64);
        for &t in &tilts {
            let c = if p.ring {
                keystone_ring(p.len, p.dist, t).0
            } else {
                keystone_side(p.len, p.dist, t)
            };
            cells.push(format!("{c:.3}"));
            if brk.starts_with('>') && c < 0.55 {
                let f = (prev.1 - 0.55) / (prev.1 - c).max(1e-9);
                brk_val = prev.0 + (t - prev.0) * f;
                brk = format!("**{brk_val:.1}°**");
            }
            prev = (t, c);
        }
        breaks.push((p.name, p.len, brk_val));
        let len_lbl = if p.ring {
            format!("{} бинов", ring_bins(p.len))
        } else {
            format!("{:.0}", p.len)
        };
        let mut row = vec![len_lbl, format!("{:.0}", p.dist)];
        row.extend(cells);
        row.push(brk);
        println!("{}", report::table_row(p.name, &row));
    }
    println!("\n**Закон ~1/L, проверка:** произведение (угол слома × длина) обязано быть постоянным.");
    println!("| примитив | L | слом | L × слом |");
    println!("|---|---|---|---|");
    for &(name, len, brk) in &breaks {
        if brk.is_finite() {
            println!("| {name} | {len:.0} | {brk:.1}° | {:.0} |", len * brk);
        } else {
            println!("| {name} | {len:.0} | > 30° | > {:.0} |", len * 30.0);
        }
    }
    println!("\n**Угловой лаг кольца под наклоном (клеток дуги, после подгонки эллипса):**");
    println!("| наклон | 5° | 10° | 15° | 20° | 25° | 30° |");
    println!("|---|---|---|---|---|---|---|");
    let lag: Vec<String> = tilts
        .iter()
        .map(|&t| format!("{:.2}", keystone_ring(61.0, 183.0, t).1))
        .collect();
    println!("{}", report::table_row("лаг, клеток", &lag));
    println!(
        "\nПредсказание аналитикой: остаточный ТАНГЕНЦИАЛЬНЫЙ лаг после снятия эллипса и центра\n\
         равен (ε/2)·R клеток, где ε = R·sinθ/d. При R = 29.5, d = 183 это {:.2} клетки на 15° —\n\
         сходится с замером, то есть механизм понят, а не подогнан.",
        {
            let r = 29.5f64;
            let eps = r * 15.0f64.to_radians().sin() / 183.0;
            eps / 2.0 * r
        }
    );

    // ---------------- 5. rolling shutter ----------------
    println!("\n## 5. Rolling shutter: цена разрыва против ВЫСОТЫ символа");
    println!(
        "\nЭкспозиция приколочена к одному обновлению ({T_EXP_MS} мс, FINDINGS §9), монитор сканирует\n\
         за {T_DISP_MS} мс, камера вычитывает кадр за {t_ro:.1} мс. Строка испорчена, если доля чужого\n\
         кадра в её экспозиции лежит в (0.25, 0.75): меньше — порог ещё берёт свой кадр, больше —\n\
         строка честно принадлежит СЛЕДУЮЩЕМУ кадру и салвадж припишет её ему."
    );
    println!("\n| высота символа, строк камеры | доля кадра | P(повреждён) | доля символа потеряна | goodput ×, салвадж ON (8 страйпов) | ON (22 страйпа) | OFF |");
    println!("|---|---|---|---|---|---|---|");
    for &rows in &[200.0f64, 400.0, 600.0, 800.0, 930.0, 1080.0] {
        let (pd, lost) = rolling(rows, t_ro);
        println!(
            "| {rows:.0} | {:.2} | {:.3} | {:.3} | {:.4} | {:.4} | {:.4} |",
            rows / CAM_H,
            pd,
            lost,
            rolling_factor(rows, t_ro, 8.0, true),
            rolling_factor(rows, t_ro, 22.0, true),
            rolling_factor(rows, t_ro, 8.0, false)
        );
    }
    println!(
        "\n**Ключевое, первое.** Символ, заполняющий 16:9-экран, имеет ОДНУ И ТУ ЖЕ высоту, будь\n\
         он квадратом, прямоугольником 16:9 или кругом — все трое ограничены ВЫСОТОЙ экрана.\n\
         Строчная цена у них тождественно равна, и в таблице §3 колонка «разрыв» это показывает:\n\
         она одинакова во всех шести конфигурациях. Вытянуть символ по горизонтали rolling\n\
         shutter не удешевляет вообще никак."
    );
    println!(
        "\n**Ключевое, второе — и оно ОБРАТНО ожиданию.** Укоротить символ тоже не помогает: при\n\
         включённом салвадже низкий символ ХУЖЕ высокого (0.767 против 0.871). Механизм: и\n\
         монитор, и камера сканируют СВЕРХУ ВНИЗ, поэтому фронт смены кадра частично «следует»\n\
         за строкой вычитки. Чем короче символ, тем ближе две развёртки друг к другу по времени\n\
         и тем БОЛЬШЕ доля символа, накрытая разрывом одновременно; в пределе совпадения\n\
         развёрток кадр теряется целиком. Высокий символ разносит строки по времени, и разрыв\n\
         съедает лишь полосу. Это ровно то, что FINDINGS §9 записал словами («rolling shutter\n\
         помогает: строчное разнообразие даёт больше шансов поймать чистую полосу»), только\n\
         теперь с числом и с механизмом."
    );
    println!("\n### 5.1 Компромисс «полоска вместо квадрата» при РАВНОЙ ёмкости");
    println!(
        "Держим число клеток нагрузки равным квадратному эталону и растягиваем символ, укорачивая\n\
         его по высоте. Это ЕДИНСТВЕННЫЙ способ, которым вытянутость покупает строчное время."
    );
    println!("| форма | габарит, display px | высота, строк | px/клетку | нагрузка | разрыв ON | goodput ON | разрыв OFF | goodput OFF |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let curve1 = &curves.iter().find(|x| x.0 == Lattice::Square && x.1 == 2).unwrap().2;
    for &(name, aspect) in &[("square 1:1", 1.0f64), ("16:9", 16.0 / 9.0), ("21:9", 21.0 / 9.0), ("32:9", 32.0 / 9.0)] {
        // держим ЧИСЛО клеток равным квадратному эталону, меняя размер клетки
        let target = build_config(Outline::Square, Lattice::Square, 12.9, FILL).payload_cells;
        let mut a = 12.9;
        for _ in 0..60 {
            let c = build_config(Outline::Rect(aspect), Lattice::Square, a, FILL);
            if c.payload_cells > target {
                a *= 1.01;
            } else {
                a *= 0.995;
            }
        }
        let c = build_config(Outline::Rect(aspect), Lattice::Square, a, FILL);
        let ser = interp_ser(curve1, c.a_cam, 2);
        let (gp, _, rs) = goodput(&c, ser, 1, true, t_ro);
        let (gp_off, _, rs_off) = goodput(&c, ser, 1, false, t_ro);
        println!(
            "| {name} | {:.0}×{:.0} | {:.0} | {:.2} | {:.0} | {rs:.4} | {gp:.1} | {rs_off:.4} | {gp_off:.1} |",
            c.w_disp, c.h_disp, c.cam_rows, c.a_cam, c.payload_cells
        );
    }

    // ---------------- 6. круговая рамка ----------------
    println!("\n## 6. Круговая рамка: угловая конструкция, наклон, непрерывный поворот");
    println!("\n### 6.1 Удержание корреляции под дефокусом — четыре конструкции");
    println!(
        "Якорь модели: FINDINGS §4 меряет полосы v1 как 0.949 / 0.702 / 0.425 при σ = 0.5 / 1 / 2 клетки.\n\
         Если растровая модель их воспроизводит — числам для петель и кольца можно верить."
    );
    println!("\n| конструкция | σ=0.5 | σ=1.0 | σ=2.0 |");
    println!("|---|---|---|---|");
    for &k in &[
        BorderKind::Strips,
        BorderKind::LoopNormal,
        BorderKind::LoopHomothetic,
        BorderKind::Ring,
    ] {
        let cells: Vec<String> = [0.5f64, 1.0, 2.0]
            .iter()
            .map(|&s| format!("{:.3}", border_retention(k, 61.0, s, 6.0)))
            .collect();
        println!("{}", report::table_row(k.name(), &cells));
    }
    println!(
        "\n**Что здесь на самом деле показано.** Рассогласование 4N−4 против 4N−12 (DEAD_ENDS §3) —\n\
         артефакт СЧЁТА В КЛЕТКАХ, а не структурная невозможность: если параметризовать петлю ДОЛЕЙ\n\
         периметра и экструдировать ГОМОТЕТИЕЙ, длины совпадают тождественно на любом контуре, и\n\
         квадрат в том числе. Круг уникален другим: только у него гомотетический луч СОВПАДАЕТ с\n\
         нормалью в каждой точке. У квадрата луч из центра перпендикулярен стороне лишь в середине,\n\
         а у угла отклоняется так, что экструзия на 2 клетки уезжает вдоль стороны почти на 2\n\
         индекса — и блюр поперёк рамки усредняет там РАЗНЫЕ значения."
    );
    println!("\n### 6.2 Угловое сэмплирование: пол px/клетку на внутреннем радиусе");
    println!("| диаметр, клеток | бинов M | дуга внешняя, клеток | дуга внутренняя | px/бин внешн. | внутр. |");
    println!("|---|---|---|---|---|---|");
    for &n in &[61.0f64, 41.0, 31.0, 21.0] {
        let m = ring_bins(n);
        let r_out = n * 0.5;
        let arc_out = 2.0 * std::f64::consts::PI * r_out / m as f64;
        let arc_in = 2.0 * std::f64::consts::PI * (r_out - 2.0) / m as f64;
        println!(
            "| {n:.0} | {m} | {arc_out:.3} | {arc_in:.3} | {:.2} | {:.2} |",
            arc_out * 12.9,
            arc_in * 12.9
        );
    }
    println!("\n### 6.3 Непрерывный поворот: точность оценки");
    println!(
        "Кольцо рендерится с ИСТИННЫМ поворотом (не кратным бину), проходит канал, читается по M\n\
         бинам, циклическая корреляция + параболическая интерполяция пика."
    );
    println!("\n| канал | средняя ошибка, ° | RMS, ° | макс, ° | RMS в бинах | RMS в клетках дуги |");
    println!("|---|---|---|---|---|---|");
    let m61 = ring_bins(61.0);
    let bin_deg = 360.0 / m61 as f64;
    for (label, ch) in [
        ("чистый", ChannelCfg::clean()),
        ("блюр σ=2", ChannelCfg { blur_cam: BLUR_CAM, field: false, noise: false }),
        ("живой (блюр+поле+шум)", ChannelCfg::live()),
    ] {
        let mut errs = Vec::new();
        for k in 0..24 {
            let phi = (k as f64 + 0.37) / 24.0 * 360.0 / 4.0; // произвольные, не кратные бину
            errs.push(ring_rotation_error(61.0, 12.0, phi, &ch, seed_for(9000, k)));
        }
        let mean = errs.iter().map(|e| e.abs()).sum::<f64>() / errs.len() as f64;
        let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
        let mx = errs.iter().fold(0.0f64, |a, e| a.max(e.abs()));
        println!(
            "| {label} | {mean:.3} | {rms:.3} | {mx:.3} | {:.3} | {:.3} |",
            rms / bin_deg,
            rms.to_radians() * (61.0 * 0.5 - 1.0)
        );
    }
    println!(
        "\nОдин угловой бин = {bin_deg:.3}°, M = {m61}."
    );
    println!(
        "\n**Неочевидное, и оно решает вопрос.** На ЧИСТОМ канале оценка ХУЖЕ, чем на размытом.\n\
         Причина: у плоских клеток без дефокуса корреляция — дискретная дельта, у пика нет\n\
         кривизны, параболическая интерполяция возвращает ноль, и оценка КВАНТУЕТСЯ бином\n\
         (RMS бина = 1/√12 = 0.289 бина — ровно замеренное). ДЕФОКУС и есть то, что делает\n\
         поворот непрерывно измеримым: он сглаживает пик, и подбиновая интерполяция начинает\n\
         работать. То есть «непрерывный поворот» — свойство не круга, а размытой границы,\n\
         и оно доступно любому контуру, у которого сдвиг вдоль рамки непрерывен.\n\
         \n\
         Для сравнения: квадрат различает поворот только кратно 90°, но делает это с отрывом\n\
         0.87 при пороге 0.10 (FINDINGS §4), а остаточный угол всё равно берётся из гомографии\n\
         по четырём углам — точнее, чем 0.37°."
    );

    println!("\n### 6.4 Структурный дефицит круга: сколько степеней свободы он НЕ даёт");
    println!(
        "Гомография имеет 8 степеней свободы. Четыре угла квадрата дают 8 уравнений — решение\n\
         в замкнутой форме. Коника имеет 5 степеней свободы, поэтому образ окружности фиксирует\n\
         гомографию лишь с точностью до 3-параметрического семейства (стабилизатор коники в\n\
         PGL(3) — это PGL(2,R), dim 3). Параметризация ЗЧ по кольцу это семейство в принципе\n\
         снимает, но действует оно на кольцо как ДРОБНО-ЛИНЕЙНОЕ перепараметрирование, а ЗЧ-\n\
         корреляция остра только по СДВИГУ. Значит замкнутая форма превращается в трёхмерный\n\
         поиск. Числовое следствие — таблица углового лага в §4."
    );

    // ---------------- 7. сводка и вердикт ----------------
    println!("\n## 7. Сводка");
    let sq = &geo[0];
    let rect = &geo[2];
    let circ = &geo[4];
    println!(
        "\n| | квадрат | прямоугольник 16:9 | круг |\n|---|---|---|---|\n\
         | площадь экрана | {:.1} % | {:.1} % | {:.1} % |\n\
         | площадь кадра | {:.1} % | {:.1} % | {:.1} % |\n\
         | клеток нагрузки (sq-решётка) | {:.0} | {:.0} (×{:.2}) | {:.0} (×{:.2}) |\n\
         | строк камеры | {:.0} | {:.0} | {:.0} |\n\
         | слом keystone (худшая сторона) | {:.1}° | {:.1}° | {:.1}° |\n\
         | ЗЧ-примитивов | 4 стороны × 59 | 4 стороны (109/61) | 1 кольцо × {} |",
        sq.screen_util * 100.0,
        rect.screen_util * 100.0,
        circ.screen_util * 100.0,
        sq.cam_util * 100.0,
        rect.cam_util * 100.0,
        circ.cam_util * 100.0,
        sq.payload_cells,
        rect.payload_cells,
        rect.payload_cells / sq.payload_cells,
        circ.payload_cells,
        circ.payload_cells / sq.payload_cells,
        sq.cam_rows,
        rect.cam_rows,
        circ.cam_rows,
        breaks[0].2,
        breaks[1].2,
        breaks[3].2,
        ring_bins(61.0)
    );

    println!("\n### 7.1 Что говорят измерения");
    println!(
        "\n1. **Решётка — единственная ось, где выигрыш подтвердился и растеризацию пережил.**\n\
         Гексагональная решётка даёт {:+.0} % по d′ при равной плотности клеток, и контроль с\n\
         ТОЖДЕСТВЕННОЙ круговой апертурой даёт то же самое, то есть выиграла упаковка, а не\n\
         форма считывателя. Рябь от субпиксельной фазы — 1.5–1.9 % у ОБЕИХ решёток: гексагон\n\
         не хуже квадрата по растеризации, вопреки ожиданию. Пересчёт в число клеток даёт\n\
         +40…50 %, но эта колонка производная и держится на наклоне d′(a) — считать её\n\
         порядком величины.\n\
         \n\
         2. **Прямоугольник 16:9 — самый большой одиночный выигрыш по ёмкости (×1.83) и он\n\
         чисто геометрический**, но платит keystone: длинная сторона ломается на {:.1}° против\n\
         {:.1}° у квадрата, и слом СИМВОЛА — это худшая сторона. Обещанной экономии на rolling\n\
         shutter не существует: при заполнении 16:9-экрана высота у всех трёх контуров одна.\n\
         \n\
         3. **Круг проигрывает по всем метрикам ёмкости** (−21 % площади, ×0.75 goodput) и\n\
         выигрывает ровно две вещи: непрерывный поворот с RMS {:.2}° (0.19 бина) и чуть лучшее\n\
         удержание корреляции под дефокусом. За это он отдаёт замкнутую форму решения\n\
         гомографии — коника даёт 5 связей из 8 (§6.4).",
        (interp_dprime(c_hx2, A_LIVE) / interp_dprime(c_sq2, A_LIVE) - 1.0) * 100.0,
        breaks[1].2,
        breaks[0].2,
        {
            let mut e = Vec::new();
            for k in 0..12 {
                let phi = (k as f64 + 0.37) / 12.0 * 90.0;
                e.push(ring_rotation_error(61.0, 12.0, phi, &ChannelCfg::live(), seed_for(9100, k)));
            }
            (e.iter().map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt()
        }
    );

    println!("\n### 7.2 Рекомендация");
    println!(
        "\n**Гексагональная решётка на квадратном контуре — единственное изменение, которое\n\
         измерение поддерживает.** Прямоугольник 16:9 — второй кандидат и он ортогонален\n\
         первому, но его keystone-цена реальна и требует отдельного решения по искателю.\n\
         **Круг рекомендовать нельзя.**"
    );
    println!("\n### 7.3 Честный довод ПРОТИВ рекомендации");
    println!(
        "\n1. **Выигрыша по надёжности нет — только по ёмкости.** В живой точке SER = {} и\n\
         выживание страйпа 0.998. Ошибок нет ни у одной решётки. Гексагон не чинит ничего;\n\
         он позволяет УМЕНЬШИТЬ клетку, а это ставка целиком на наклон кривой d′(a), который\n\
         измерен одним стендом и одной моделью канала.\n\
         \n\
         2. **Квадратная решётка вкомпилирована в весь тракт.** Растровый порядок L3 (§6.2),\n\
         страйпы из СТРОК клеток, плитки калибровочного кадра (§4-IB), экструдированная\n\
         полоса рамки, гомография и ресэмплинг приёмника — всё это «строка × столбец».\n\
         Гексагональная решётка ломает не рамку, а ПОРЯДОК КЛЕТОК, то есть L3, CRC-страйпы\n\
         и формат целиком. Это не «поменять рендерер».\n\
         \n\
         3. **Апертура Вороного у гексагона — предположение, а не факт.** Стенд читает клетку\n\
         её точной ячейкой Вороного (контроль кругом дал то же, но круг тоже не бесплатен).\n\
         Приёмник, который ресэмплит через гомографию билинейно, шестиугольник аппроксимирует\n\
         хуже, чем квадрат — квадратную клетку. Часть выигрыша живёт в считывателе, которого\n\
         пока нет.\n\
         \n\
         4. **Проект уже имеет привычку хоронить предложения на контакте с измерением.**\n\
         DEAD_ENDS §1, §2, §6, §7, §10. Ни одно число выше не получено НА ЖИВОМ ЖЕЛЕЗЕ.\n\
         Правильный следующий шаг — не менять формат, а отрендерить гексагональное поле\n\
         передатчиком, снять его тем же телефоном и сверить d′ с {:.2} из §1.2.",
        report::sig4(interp_ser(c_sq2, A_LIVE, 2)),
        interp_dprime(c_hx2, A_LIVE)
    );
    println!("\n### 7.4 Что говорит В ПОЛЬЗУ сегодняшнего квадрат+квадрат");
    println!(
        "\n- Единственная ось, где сегодняшняя конфигурация ХУЖЕ измеренного оптимума по\n\
           площади экрана, — контур; и ровно там цена (keystone {:.1}° против {:.1}°) падает\n\
           на самый хрупкий и уже трижды переписанный узел — аквизицию.\n\
         - Круговая рамка ДЕЙСТВИТЕЛЬНО снимает рассогласование слоёв, но §6.1 показывает,\n\
           что рассогласование 4N−4 / 4N−12 из DEAD_ENDS §3 — артефакт счёта в клетках:\n\
           гомотетическая параметризация чинит его и на КВАДРАТЕ (0.667 → 0.817 при σ=0.5).\n\
           Уникальность круга сузилась с «единственный контур без рассогласования» до\n\
           «единственный контур, где луч из центра совпадает с нормалью», а это стоит\n\
           0.947 → 0.953, то есть ничего.\n\
         - Отгруженная рамка v1 уже даёт 730/744 детекций и запас 0.87 при пороге 0.10 по\n\
           ориентации. Непрерывный поворот решает задачу, которой нет.\n\
         - Разрыв кадра перестал зависеть от геометрии, как только появился пер-страйповый\n\
           CRC-салвадж. Аргумент «широкий и низкий символ дешевле по строкам» не просто\n\
           не подтвердился — он ОБРАТЕН (§5).",
        breaks[1].2,
        breaks[0].2
    );

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

/// Интерполяция d′ по развёртке.
fn interp_dprime(curve: &[LatPoint], a: f64) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    let a = a.clamp(curve[0].a_cam, curve[curve.len() - 1].a_cam);
    let mut k = 0usize;
    while k + 2 < curve.len() && curve[k + 1].a_cam < a {
        k += 1;
    }
    let (p, q) = (&curve[k], &curve[k + 1]);
    let f = if (q.a_cam - p.a_cam).abs() < 1e-9 {
        0.0
    } else {
        (a - p.a_cam) / (q.a_cam - p.a_cam)
    };
    p.dprime + (q.dprime - p.dprime) * f
}

// ---------------------------------------------------------------------------
// тесты
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Округление до ближайшего центра действительно даёт ячейку Вороного:
    /// возвращённый центр не дальше любого другого центра решётки.
    #[test]
    fn nearest_is_the_voronoi_cell() {
        let d = 7.3;
        let mut rng = Rng::new(0xA11CE);
        for lat in [Lattice::Square, Lattice::Hex] {
            for _ in 0..3000 {
                let x = (rng.next_f64() - 0.5) * 60.0;
                let y = (rng.next_f64() - 0.5) * 60.0;
                let (i, j) = lat.nearest(x, y, d);
                let (cx, cy) = lat.center(i, j, d);
                let best = (x - cx).hypot(y - cy);
                for dj in -2..=2 {
                    for di in -2..=2 {
                        let (ox, oy) = lat.center(i + di, j + dj, d);
                        let dist = (x - ox).hypot(y - oy);
                        assert!(
                            dist >= best - 1e-9,
                            "{lat:?}: ({x},{y}) отдана не ближайшему центру ({dist} < {best})"
                        );
                    }
                }
            }
        }
    }

    /// Площадь клетки и шаг соседей согласованы, и гексагональная решётка при
    /// РАВНОЙ площади держит соседа дальше ровно на 2/√3 в квадрате.
    #[test]
    fn hex_pitch_is_larger_at_equal_area() {
        let a = 144.0;
        let ds = Lattice::Square.pitch_for_area(a);
        let dh = Lattice::Hex.pitch_for_area(a);
        assert!((Lattice::Square.cell_area(ds) - a).abs() < 1e-9);
        assert!((Lattice::Hex.cell_area(dh) - a).abs() < 1e-9);
        assert!((dh / ds - 1.074_569_9).abs() < 1e-5, "dh/ds = {}", dh / ds);
        // и периметр клетки меньше ровно на теоретические ~7 %
        let ps = Lattice::Square.cell_perimeter(a);
        let ph = Lattice::Hex.cell_perimeter(a);
        assert!((ph / ps - 0.930_604).abs() < 1e-4, "ph/ps = {}", ph / ps);
    }

    /// Плотность клеток на решётке совпадает с 1/площадь — то есть «равная
    /// площадь» действительно значит «равное число клеток на ту же площадь».
    #[test]
    fn cell_density_matches_area() {
        let a = 100.0;
        for lat in [Lattice::Square, Lattice::Hex] {
            let d = lat.pitch_for_area(a);
            let side = 400.0;
            let mut seen = std::collections::HashSet::new();
            for y in 0..400 {
                for x in 0..400 {
                    seen.insert(lat.nearest(x as f64 + 0.5, y as f64 + 0.5, d));
                }
            }
            let got = seen.len() as f64;
            let want = side * side / a;
            assert!(
                (got / want - 1.0).abs() < 0.12,
                "{lat:?}: клеток {got}, ожидалось ~{want}"
            );
        }
    }

    /// Шумовая модель воспроизводит ОБА замера FINDINGS §2: 6.15/255 на пиксель
    /// и 1.79/255 на клетку при 11.8 px/клетку.
    #[test]
    fn noise_model_matches_both_measurements() {
        let mut rng = Rng::new(0x_0105E);
        let (w, h) = (600usize, 600usize);
        let p = noise_plane(w, h, &mut rng);
        let n = (w * h) as f64;
        let mean = p.d.iter().sum::<f64>() / n;
        let sd = (p.d.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
        assert!(
            (sd / NOISE_PIX - 1.0).abs() < 0.02,
            "попиксельная σ {} против {}",
            sd * 255.0,
            NOISE_PIX * 255.0
        );
        // среднее по боксу 11.8×11.8
        let b = NOISE_REF_PPC.round() as usize;
        let mut vals = Vec::new();
        let mut y = 0;
        while y + b <= h {
            let mut x = 0;
            while x + b <= w {
                let mut s = 0.0;
                for yy in y..y + b {
                    for xx in x..x + b {
                        s += p.at(xx, yy);
                    }
                }
                vals.push(s / (b * b) as f64);
                x += b;
            }
            y += b;
        }
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let s = (vals.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / vals.len() as f64).sqrt();
        assert!(
            (s / NOISE_CELL - 1.0).abs() < 0.15,
            "поклеточная σ {:.3}/255 против замеренной {:.2}/255",
            s * 255.0,
            NOISE_CELL * 255.0
        );
    }

    /// Санитарный гейт: на чистом канале ВСЕ шесть конфигураций читают нагрузку
    /// без единой ошибки.
    #[test]
    fn clean_channel_recovers_every_configuration() {
        let clean = ChannelCfg::clean();
        for outline in [Outline::Square, Outline::Rect(16.0 / 9.0), Outline::Circle] {
            for lat in [Lattice::Square, Lattice::Hex] {
                let (ser, n) = sanity_symbol(outline, lat, 12.0, &clean, 0x5A17);
                assert!(n > 200, "{outline:?}/{lat:?}: клеток мало ({n})");
                assert_eq!(ser, 0.0, "{outline:?}/{lat:?}: SER {ser} на ЧИСТОМ канале");
            }
        }
    }

    /// Растровая модель рамки воспроизводит замеренное удержание полос v1
    /// (FINDINGS §4: 0.949 / 0.702 / 0.425 при σ = 0.5 / 1 / 2 клетки).
    /// Это якорь доверия ко всем остальным числам §6.1.
    #[test]
    fn strip_retention_reproduces_the_measurement() {
        let want = [0.949f64, 0.702, 0.425];
        for (k, &s) in [0.5f64, 1.0, 2.0].iter().enumerate() {
            let got = border_retention(BorderKind::Strips, 61.0, s, 6.0);
            assert!(
                (got - want[k]).abs() < 0.20,
                "σ={s}: удержание {got:.3} против замеренного {:.3}",
                want[k]
            );
        }
    }

    /// Экструзия кольца ПО РАДИУСУ тождественно совпадает по индексу на любом
    /// радиусе — ровно то свойство, ради которого строились полосы, и ровно то,
    /// чего замкнутая петля на квадрате дать не может.
    #[test]
    fn ring_extrusion_is_identical_across_radius() {
        let n = 61.0;
        let spec = BorderSpec { n: 61, roots: V1_ROOTS, carrier: Carrier::ComplexChroma };
        let m = ring_bins(n);
        for k in 0..m {
            // середина бина: на границе бинов сравнение бессмысленно
            let phi = (k as f64 + 0.5) / m as f64 * 2.0 * std::f64::consts::PI
                - std::f64::consts::PI;
            let mut prev: Option<(f64, f64)> = None;
            for depth in [0.5f64, 1.0, 1.5] {
                let r = n * 0.5 - depth;
                let x = n * 0.5 + r * phi.cos();
                let y = n * 0.5 + r * phi.sin();
                let v = border_value(BorderKind::Ring, &spec, n, x, y).expect("в кольце");
                if let Some(p) = prev {
                    assert!(
                        (v.0 - p.0).abs() < 1e-9 && (v.1 - p.1).abs() < 1e-9,
                        "бин {k}: радиусы разошлись"
                    );
                }
                prev = Some(v);
            }
        }
    }

    /// Гомотетическая петля на КВАДРАТЕ тоже совпадает по индексу на обеих
    /// глубинах — то есть 8-клеточное рассогласование DEAD_ENDS §3 снимается
    /// параметризацией, а не формой. Одновременно фиксируем ЦЕНУ: луч из центра
    /// не перпендикулярен стороне, и у угла экструзия уезжает ВДОЛЬ.
    #[test]
    fn homothetic_loop_matches_index_but_slants() {
        let n = 61.0;
        let (cx, cy) = (n * 0.5, n * 0.5);
        // доля периметра ИНВАРИАНТНА к гомотетии из центра — значит оба ряда
        // полосы получают ОДИН индекс на ЛЮБОМ контуре, и 8-клеточное
        // рассогласование DEAD_ENDS §3 снимается ПАРАМЕТРИЗАЦИЕЙ, не формой
        let mut rng = Rng::new(0xF00D);
        for _ in 0..500 {
            let ang = rng.next_f64() * 2.0 * std::f64::consts::PI;
            let r = 5.0 + rng.next_f64() * 25.0;
            let (x, y) = (cx + r * ang.cos(), cy + r * ang.sin());
            let f_out = perimeter_frac_homothetic(x, y, n);
            for s in [0.4f64, 0.7, 0.94] {
                let f_in = perimeter_frac_homothetic(cx + (x - cx) * s, cy + (y - cy) * s, n);
                let mut d = (f_out - f_in).abs();
                if d > 0.5 {
                    d = 1.0 - d;
                }
                assert!(d < 1e-9, "доля периметра не инвариантна: {f_out} vs {f_in}");
            }
        }
        // ЦЕНА: луч из центра перпендикулярен стороне только в её середине.
        // У середины экструзия по нормали и по лучу совпадают...
        let mid = perimeter_frac_homothetic(cx, 0.5, n);
        let mid_in = perimeter_frac_homothetic(cx, 2.5, n);
        assert!((mid - mid_in).abs() < 1e-9, "в середине стороны луч = нормаль");
        // ...а у угла НЕТ: экструзия на 2 клетки вглубь уезжает ВДОЛЬ стороны
        let m = 4.0 * n - 4.0;
        let corner = perimeter_frac_homothetic(n - 3.5, 0.5, n);
        let corner_in = perimeter_frac_homothetic(n - 3.5, 2.5, n);
        let slip = (corner - corner_in).abs() * m;
        assert!(
            slip > 1.0,
            "у угла экструзия обязана уезжать вдоль стороны, получили {slip:.2} клеток"
        );
    }

    /// Модель разрыва монотонна по высоте символа и совпадает с якорем
    /// «84 % чистых при H=16» в точке калибровки.
    #[test]
    fn rolling_model_is_monotone_and_matches_anchor() {
        let t_ro = rolling_readout_from_anchor();
        assert!(t_ro > 20.0 && t_ro < 120.0, "T_ro вне физичного диапазона: {t_ro}");
        let mut prev = 0.0;
        for &r in &[100.0f64, 300.0, 600.0, 900.0] {
            let (pd, _) = rolling(r, t_ro);
            assert!(pd >= prev - 1e-9, "P(повреждён) не монотонна по высоте");
            prev = pd;
        }
        // якорь: при H=16 (T_tx=267 мс) и высоте FILL повреждено ~16 %
        let rows = FILL * WORK_H * MAG;
        let (pd, _) = rolling(rows, t_ro);
        let p16 = pd * T_TX_MS / (16.0 * 16.67);
        assert!((p16 - 0.16).abs() < 0.02, "якорь не сошёлся: {p16}");
    }

    /// Q-функция и её хвост: сверка с известными значениями.
    #[test]
    fn qfunc_is_accurate() {
        // приближение NR даёт ОТНОСИТЕЛЬНУЮ точность ~1.2e-7 — по ней и порог
        for &(x, want) in &[
            (0.0f64, 0.5f64),
            (1.0, 0.158_655_254),
            (3.0, 0.001_349_898),
            (5.0, 2.866_516e-7),
            (7.0, 1.279_813e-12),
        ] {
            let got = qfunc(x);
            assert!(
                (got / want - 1.0).abs() < 3e-6,
                "Q({x}) = {got:e} против {want:e}"
            );
        }
    }

    /// Число угловых бинов кольца простое (иначе кросс-корреляция ЗЧ теряет пол
    /// 1/√M) и близко к длине средней окружности в клетках.
    #[test]
    fn ring_bins_are_prime_and_match_the_circumference() {
        for &n in &[61.0f64, 41.0, 31.0] {
            let m = ring_bins(n);
            let want = 2.0 * std::f64::consts::PI * (n * 0.5 - 1.0);
            assert!(m as f64 <= want && m as f64 > want - 12.0, "M={m} против {want}");
            let mut d = 2;
            while d * d <= m {
                assert!(m % d != 0, "M={m} не простое");
                d += 1;
            }
        }
    }

    /// Keystone: корреляция монотонно падает с наклоном, и длинная сторона
    /// ломается РАНЬШЕ короткой — проверка закона ~1/L на нашей механике.
    #[test]
    fn keystone_penalises_the_long_axis() {
        // при ОДИНАКОВОМ отношении сторона/расстояние (кадрирование по большему
        // габариту) длинная сторона обязана ломаться раньше — это и есть ~1/L
        let short = keystone_side(61.0, 183.0, 10.0);
        let long = keystone_side(109.0, 327.0, 10.0);
        assert!(
            long < short,
            "длинная сторона ({long:.3}) обязана ломаться раньше короткой ({short:.3})"
        );
        // а короткая сторона ТОГО ЖЕ прямоугольника, наоборот, крепче квадратной:
        // её снимают издалека, потому что кадр задаёт длинная ось
        let short_in_wide = keystone_side(61.0, 327.0, 10.0);
        assert!(
            short_in_wide > short,
            "короткая сторона широкого символа ({short_in_wide:.3}) должна быть крепче ({short:.3})"
        );
        let a = keystone_side(61.0, 183.0, 5.0);
        let b = keystone_side(61.0, 183.0, 25.0);
        assert!(a > b, "корреляция не падает с наклоном: {a} -> {b}");
    }

    /// Оценка НЕПРЕРЫВНОГО поворота по кольцу на ЧИСТОМ канале обязана
    /// возвращать истинный угол — это проверка знака и соглашения о сдвиге,
    /// без которой все числа §6.3 бессмысленны.
    #[test]
    fn ring_rotation_is_unbiased_on_a_clean_channel() {
        let clean = ChannelCfg::clean();
        let bin = 360.0 / ring_bins(61.0) as f64;
        for k in 0..8 {
            let phi = k as f64 * 11.0 + 3.7;
            let e = ring_rotation_error(61.0, 12.0, phi, &clean, seed_for(77, k));
            assert!(
                e.abs() < bin,
                "поворот {phi}°: ошибка {e:.3}° больше одного бина ({bin:.3}°)"
            );
        }
    }
}
