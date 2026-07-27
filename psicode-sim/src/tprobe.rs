//! `tprobe` — Монте-Карло ВРЕМЕННОГО канала дисплей→камера (задача 1-D во времени).
//!
//! Гипотеза владельца: временное смешивание ISP — это ФНЧ во времени, то есть та
//! же задача, что пространственный блюр. Жёсткое переключение кадров (кусочно-
//! постоянный сигнал) — временной аналог резкого края: энергия на высоких
//! временных частотах и МСИ ∝ (выдержка + постоянная смешивания) / период кадра.
//! Кандидат-лекарство тот же, что в пространстве: гладкий временной базис, чтобы
//! ФНЧ канала стал ИЗВЕСТНЫМ скалярным ослаблением, которое обращается, а не
//! неразделимой помехой.
//!
//! # Цепочка канала (что моделируется)
//!
//! 1. Дисплей: сигнал s(t) удерживается H периодов обновления 60 Гц,
//!    T_f = H/60 с. Значение патча — СКАЛЯР (патч большой ⇒ пространственный
//!    блюр по построению нерелевантен).
//! 2. Выдержка: боксkar длиной `t_exp` (эталон 1/60 с ≈ 16.7 мс).
//! 3. Rolling shutter: строка r экспонируется со сдвигом (r+½)/R·`t_read`.
//!    Это ЕДИНСТВЕННЫЙ пространственный эффект: разные строки одного снимка
//!    видят разное время.
//! 4. Камера 30 fps, фаза относительно дисплея неизвестна (стратифицированная
//!    развёртка по фазе).
//! 5. ISP temporal mixing: y_k(r) = (1−α)·x_k(r) + α·y_{k−1}(r) — однополюсный
//!    БИХ по НОМЕРУ КАДРА КАМЕРЫ при фиксированной строке. Шум вносится ДО БИХ
//!    (сенсорный), квантование 8 бит — ПОСЛЕ.
//!
//! # Схемы
//!
//! * **A** — жёсткое переключение (то, что мы возим сегодня): один M-PAM символ
//!   на кадр дисплея, кусочно-постоянный патч.
//! * **B1** — временное FDM на гармониках периода кадра: DC + K пар (cos, sin),
//!   1+2K вещественных измерений на кадр. Разрывен на границах кадров.
//! * **B2** — гладкий базис: полупериодные синусы sin(πnu/T_f), n = 1..K. Все
//!   обнуляются на границах ⇒ s(t) НЕПРЕРЫВЕН по кадрам (ровно та «гладкость»,
//!   ради которой всё затевалось). K измерений на кадр.
//! * **C** — A плюс отдельная СИГНАТУРНАЯ область (1/8 площади кадра), где
//!   циклически крутятся L = 3 известных уровня. Приёмник оценивает сигнатуру
//!   ПОБАНДОВО; если оценка не села на легальный уровень своего кадра — банда
//!   ОТБРАКОВЫВАЕТСЯ до демодуляции. Побандовость даёт локализацию разрыва.
//!
//! # Приёмник (что даётся схемам «бесплатно» — одинаково для всех)
//!
//! Сетка кадров дисплея (период и фаза) известна: дисплей — известный клок, телефон
//! захватывает его за много кадров. Это ЕДИНСТВЕННЫЙ «genie» и он симметричен.
//! Ни одна схема НЕ получает деконволюцию МСИ: B корректирует только известное
//! КОМПЛЕКСНОЕ усиление канала на каждой гармонике (боксkar-sinc × БИХ), то есть
//! стационарную АЧХ/ФЧХ, а остаточная утечка соседних кадров остаётся ошибкой.
//! Неравномерность выборки (пачки строк с зазорами) обрабатывается МНК-проекцией
//! на базис с уже применённым усилением канала — это корректное обобщение
//! «скоррелировать и поделить на затухание», а не деконволюция.

use crate::report;
use crate::rng::{seed_for, Rng};
use std::f64::consts::PI;
use std::time::Instant;

// ---------------------------------------------------------------- физика ----

/// Период обновления дисплея (60 Гц).
const T_REF: f64 = 1.0 / 60.0;
/// Период кадра камеры (30 fps).
const T_CAM: f64 = 1.0 / 30.0;
/// Эталонная выдержка (§ живой замер: пинуем к одному периоду обновления).
const T_EXP_REF: f64 = 1.0 / 60.0;
/// Эталонное время развёртки rolling shutter.
const T_READ_REF: f64 = 0.016;
/// Эталонный α ISP для развёрток 1/3/4 (развёртка 2 меряет зависимость от α).
const ALPHA_REF: f64 = 0.15;

// ------------------------------------------------------------- геометрия ----

/// Строк-отсчётов на кадр камеры внутри патча.
const ROWS: usize = 48;
/// Банд (аналог 8 страйпов L3) — единица приёма/отбраковки/локализации разрыва.
const BANDS: usize = 8;

// -------------------------------------------------------------- алфавит -----

/// Размер PAM-алфавита на одно измерение (3 бита/символ).
const M_PAM: usize = 8;
/// Середина размаха драйва.
const MID: f64 = 0.5;
/// Половина размаха драйва: сигнал живёт в [0.05, 0.95].
const SWING: f64 = 0.45;
/// Половина расстояния между PAM-уровнями в нормированных координатах d ∈ [−1, 1].
const PAM_HALF: f64 = 1.0 / (M_PAM as f64 - 1.0);
/// Максимальная доля примеси соседнего кадра, гарантированно не ломающая символ:
/// β·(полный размах) < (половина шага) ⇒ β < 1/(2(M−1)).
const BETA_TOL: f64 = 0.5 / (M_PAM as f64 - 1.0);

// ----------------------------------------------------------------- шум ------

/// СКО наблюдения на ОДИН строчный отсчёт данных (доля полной шкалы).
/// Консервативно: для большого патча реально ~0.002, здесь 0.01.
const SIGMA_OBS: f64 = 0.01;
/// Сигнатура занимает 1/8 площади ⇒ её шум в √8 раз больше.
const SIG_AREA_DIV: f64 = 8.0;

// -------------------------------------------------------------- прогон ------

const N_FRAMES: usize = 96;
const N_TRIALS: usize = 12;
/// Сдвиг стратифицированной фазы, несоизмеримый с сеткой камеры/дисплея.
const PHASE_NUDGE: f64 = 0.013;
/// Прогрев БИХ / край развёртки — кадры дисплея, исключённые из метрик.
const BURN: usize = 3;
const TAIL: usize = 2;

// ------------------------------------------------------------------ B -------

/// Потолок числа гармоник (дальше делёж амплитуды съедает выигрыш).
const K_CAP: usize = 4;
/// Нижняя граница |H(f)| канала на рабочей гармонике (усиление коррекции ≤ 2.5×).
const H_GAIN_FLOOR: f64 = 0.4;
/// Потолок числа обусловленности нормальной матрицы (столбцы нормированы).
const COND_MAX: f64 = 50.0;

// ------------------------------------------------------------------ C -------

/// Уровни сигнатуры, циклически по номеру кадра дисплея (L = 3).
const SIG_LEVELS: [f64; 3] = [0.05, 0.5, 0.95];
/// Минимальная разность соседних уровней сигнатуры.
const SIG_DELTA_MIN: f64 = 0.45;
/// Порог отбраковки: примесь β детектируется, если β·Δg_min > τ. Ставим τ так,
/// чтобы отбраковывалось ровно то, что способно испортить символ (β > BETA_TOL).
const TAU: f64 = BETA_TOL * SIG_DELTA_MIN;
/// Доля площади кадра, отнятая сигнатурой ⇒ множитель скорости для C.
const C_RATE_SCALE: f64 = 7.0 / 8.0;

/// Строгий порог «чистого» захвата (никакой примеси вообще).
const CLEAN_EPS: f64 = 1e-3;
/// Рабочий бюджет FEC по НЕОБНАРУЖЕННЫМ ошибкам (RS(16,8) чинит до 25%, берём запас).
const FEC_ERR_BUDGET: f64 = 0.15;
/// Порог стирания по невязке МНК, в единицах известного приёмнику σ. При n − dims = 1
/// степени свободы оценка невязки очень шумная, поэтому берём широкий порог: примесь
/// соседнего кадра даёт невязку порядка размаха данных, то есть на порядки выше σ.
const RES_K: f64 = 3.0;

// ------------------------------------------------------------ комплексные ---

#[derive(Clone, Copy)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    const ONE: C64 = C64 { re: 1.0, im: 0.0 };

    fn mul(self, o: C64) -> C64 {
        C64 {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    fn div(self, o: C64) -> C64 {
        let d = o.re * o.re + o.im * o.im;
        C64 {
            re: (self.re * o.re + self.im * o.im) / d,
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }

    fn abs(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// АЧХ/ФЧХ боксkar-выдержки: (1/T)∫₀^T e^{jωτ}dτ = e^{jωT/2}·sinc(ωT/2).
/// Опорная точка времени — НАЧАЛО окна экспозиции.
fn h_box(omega: f64, t_exp: f64) -> C64 {
    let a = omega * t_exp * 0.5;
    if a.abs() < 1e-12 {
        return C64::ONE;
    }
    let m = a.sin() / a;
    C64 {
        re: m * a.cos(),
        im: m * a.sin(),
    }
}

/// АЧХ/ФЧХ однополюсного БИХ ISP на частоте дискретизации 1/T_CAM:
/// H(ω) = (1−α)/(1 − α·e^{−jωT_cam}).
fn h_iir(omega: f64, alpha: f64) -> C64 {
    if alpha <= 0.0 {
        return C64::ONE;
    }
    let w = omega * T_CAM;
    let den = C64 {
        re: 1.0 - alpha * w.cos(),
        im: alpha * w.sin(),
    };
    C64 {
        re: 1.0 - alpha,
        im: 0.0,
    }
    .div(den)
}

// ----------------------------------------------------------------- PAM ------

fn pam_level(i: usize) -> f64 {
    -1.0 + 2.0 * i as f64 / (M_PAM as f64 - 1.0)
}

fn pam_slice(d: f64) -> usize {
    let x = (d + 1.0) * 0.5 * (M_PAM as f64 - 1.0);
    x.round().clamp(0.0, M_PAM as f64 - 1.0) as usize
}

// --------------------------------------------------------------- схемы ------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scheme {
    /// Жёсткое переключение (сегодняшний baseline).
    A,
    /// Гармоники периода кадра: DC + K пар (cos, sin).
    B1,
    /// Гладкий базис: полупериодные синусы, непрерывные на границах.
    B2,
    /// A + побандовая сигнатура-детектор смеси.
    C,
    /// Рекомендованная связка: гладкий базис B2 + сигнатурная полоса C, которая
    /// СТИРАЕТ (erasure) захваты со смесью ДО МНК вместо того, чтобы кормить ими
    /// оценщик. Данные и площадь — как у B2 и C соответственно.
    B2C,
}

impl Scheme {
    fn label(self) -> &'static str {
        match self {
            Scheme::A => "A hard",
            Scheme::B1 => "B1 fourier",
            Scheme::B2 => "B2 smooth",
            Scheme::C => "C hard+sig",
            Scheme::B2C => "B2+sig",
        }
    }

    fn rate_scale(self) -> f64 {
        if self == Scheme::C || self == Scheme::B2C {
            C_RATE_SCALE
        } else {
            1.0
        }
    }

    /// Схемы с побандовым оценщиком значения и голосованием (жёсткое переключение).
    fn is_hard(self) -> bool {
        self == Scheme::A || self == Scheme::C
    }

    /// Схемы с сигнатурной полосой.
    fn has_sig(self) -> bool {
        self == Scheme::C || self == Scheme::B2C
    }

    /// Базис, по которому строится план МНК (B2C = B2).
    fn basis(self) -> Scheme {
        if self == Scheme::B2C {
            Scheme::B2
        } else {
            self
        }
    }
}

const ALL: [Scheme; 4] = [Scheme::A, Scheme::B1, Scheme::B2, Scheme::C];
/// Расширенный набор для разделов 6–8 (добавлена рекомендованная связка B2+sig).
const ALL5: [Scheme; 5] = [
    Scheme::A,
    Scheme::B1,
    Scheme::B2,
    Scheme::C,
    Scheme::B2C,
];

// -------------------------------------------------------------- конфиг ------

#[derive(Clone, Copy)]
struct Cfg {
    h: u32,
    t_exp: f64,
    t_read: f64,
    alpha: f64,
    sigma: f64,
    quant: bool,
    rows: usize,
    n_frames: usize,
    trials: usize,
    /// Фиксированная фаза камеры (развёртка 4); иначе стратифицируем по попыткам.
    fixed_phase: Option<f64>,
    /// ШИМ подсветки: (частота, Гц; относительная амплитуда пульсации).
    pwm: Option<(f64, f64)>,
    /// PRNU — фиксированный по времени разброс усиления «клетки» (СКО, доля).
    prnu: f64,
    /// БИХ ISP работает ПОСЛЕ тоновой кривой (drive-домен), а не по линейному свету.
    drive_domain: bool,
}

impl Cfg {
    fn reference(h: u32) -> Cfg {
        Cfg {
            h,
            t_exp: T_EXP_REF,
            t_read: T_READ_REF,
            alpha: ALPHA_REF,
            sigma: SIGMA_OBS,
            quant: true,
            rows: ROWS,
            n_frames: N_FRAMES,
            trials: N_TRIALS,
            fixed_phase: None,
            pwm: None,
            prnu: 0.0,
            drive_domain: false,
        }
    }

    /// Эталон в «честной» per-cell геометрии: одна клетка получает ОДИН отсчёт
    /// на кадр камеры (позиция клетки покрыта развёрткой по фазе камеры).
    fn per_cell(h: u32) -> Cfg {
        let mut c = Cfg::reference(h);
        c.rows = 1;
        c
    }

    fn t_f(&self) -> f64 {
        self.h as f64 * T_REF
    }

    fn bands(&self) -> usize {
        BANDS.min(self.rows)
    }
}

// ----------------------------------------------------------- дизайн B -------

/// Параметры линка, вычисляемые из ИЗВЕСТНЫХ параметров устройства (H, t_exp,
/// t_read, α) и потому одинаково известные передатчику и приёмнику.
#[derive(Clone, Copy)]
struct Design {
    k: usize,
    dims: usize,
    amp: f64,
}

/// Частота n-й базисной функции схемы.
fn basis_freq(scheme: Scheme, n: usize, t_f: f64) -> f64 {
    match scheme.basis() {
        Scheme::B1 => n as f64 / t_f,
        Scheme::B2 => n as f64 / (2.0 * t_f),
        _ => 0.0,
    }
}

/// Комплексное усиление канала на n-й базисной функции.
fn basis_gain(scheme: Scheme, n: usize, t_f: f64, t_exp: f64, alpha: f64) -> C64 {
    let omega = 2.0 * PI * basis_freq(scheme, n, t_f);
    h_box(omega, t_exp).mul(h_iir(omega, alpha))
}

/// Номинальные времена НАЧАЛА экспозиции строчных отсчётов, попадающих в один
/// период кадра дисплея при фазе 0 (для проверки обусловленности).
fn nominal_samples(cfg: &Cfg, t_f: f64) -> Vec<f64> {
    let n_cam = (t_f / T_CAM).ceil() as usize + 2;
    let mut out = Vec::new();
    for k in 0..n_cam {
        for r in 0..cfg.rows {
            let ts = k as f64 * T_CAM + (r as f64 + 0.5) / cfg.rows as f64 * cfg.t_read;
            let ctr = ts + cfg.t_exp * 0.5;
            if ctr >= 0.0 && ctr < t_f {
                out.push(ts);
            }
        }
    }
    out
}

/// Столбцы плана для сэмпла с локальным временем `u` (от начала кадра дисплея),
/// с уже применённым комплексным усилением канала. Амплитуда НЕ включена.
fn design_row(scheme: Scheme, k: usize, u: f64, t_f: f64, t_exp: f64, alpha: f64, out: &mut [f64]) {
    match scheme.basis() {
        Scheme::B1 => {
            out[0] = 1.0;
            for n in 1..=k {
                let hg = basis_gain(scheme, n, t_f, t_exp, alpha);
                let w = 2.0 * PI * n as f64 / t_f;
                let (s, c) = (w * u).sin_cos();
                out[2 * n - 1] = hg.re * c - hg.im * s;
                out[2 * n] = hg.im * c + hg.re * s;
            }
        }
        Scheme::B2 => {
            for n in 1..=k {
                let hg = basis_gain(scheme, n, t_f, t_exp, alpha);
                let w = PI * n as f64 / t_f;
                let (s, c) = (w * u).sin_cos();
                out[n - 1] = hg.im * c + hg.re * s;
            }
        }
        _ => out[0] = 1.0,
    }
}

/// Отношение max/min ведущего элемента гауссова исключения на нормированной
/// нормальной матрице — прокси числа обусловленности. `+inf` при вырождении.
fn pivot_ratio(g: &mut [f64], d: usize) -> f64 {
    let (mut pmin, mut pmax) = (f64::INFINITY, 0.0f64);
    for i in 0..d {
        let p = g[i * d + i];
        if !(p > 1e-12) {
            return f64::INFINITY;
        }
        pmin = pmin.min(p);
        pmax = pmax.max(p);
        for r in i + 1..d {
            let f = g[r * d + i] / p;
            if f != 0.0 {
                for c in i..d {
                    g[r * d + c] -= f * g[i * d + c];
                }
            }
        }
    }
    pmax / pmin
}

/// Обусловленность геометрии выборки для K базисных функций.
fn cond_for(scheme: Scheme, k: usize, cfg: &Cfg, t_f: f64) -> f64 {
    let d = match scheme.basis() {
        Scheme::B1 => 1 + 2 * k,
        Scheme::B2 => k,
        _ => 1,
    };
    if d == 0 {
        return f64::INFINITY;
    }
    let ts = nominal_samples(cfg, t_f);
    if ts.len() < d + 1 {
        return f64::INFINITY;
    }
    let mut g = vec![0.0; d * d];
    let mut row = vec![0.0; d];
    for &t in &ts {
        design_row(scheme, k, t, t_f, cfg.t_exp, cfg.alpha, &mut row);
        for i in 0..d {
            for j in 0..d {
                g[i * d + j] += row[i] * row[j];
            }
        }
    }
    // нормируем столбцы к единичной норме — тогда диагональ = 1
    let diag: Vec<f64> = (0..d).map(|i| g[i * d + i].max(1e-30).sqrt()).collect();
    for i in 0..d {
        for j in 0..d {
            g[i * d + j] /= diag[i] * diag[j];
        }
    }
    pivot_ratio(&mut g, d)
}

/// Выбор числа гармоник: самое большое K, при котором (а) канал не давит
/// гармонику сильнее H_GAIN_FLOOR и (б) геометрия выборки её различает.
fn design(scheme: Scheme, cfg: &Cfg) -> Design {
    let t_f = cfg.t_f();
    match scheme.basis() {
        Scheme::A | Scheme::C => Design {
            k: 0,
            dims: 1,
            amp: SWING,
        },
        Scheme::B1 => {
            let mut k = K_CAP;
            loop {
                let gains_ok = (1..=k)
                    .all(|n| basis_gain(scheme, n, t_f, cfg.t_exp, cfg.alpha).abs() >= H_GAIN_FLOOR);
                if k == 0 || (gains_ok && cond_for(scheme, k, cfg, t_f) <= COND_MAX) {
                    break;
                }
                k -= 1;
            }
            Design {
                k,
                dims: 1 + 2 * k,
                amp: SWING / (1.0 + k as f64 * std::f64::consts::SQRT_2),
            }
        }
        Scheme::B2 | Scheme::B2C => {
            let mut k = K_CAP;
            loop {
                let gains_ok = (1..=k)
                    .all(|n| basis_gain(scheme, n, t_f, cfg.t_exp, cfg.alpha).abs() >= H_GAIN_FLOOR);
                if k == 1 || (gains_ok && cond_for(scheme, k, cfg, t_f) <= COND_MAX) {
                    break;
                }
                k -= 1;
            }
            Design {
                k,
                dims: k,
                amp: SWING / k as f64,
            }
        }
    }
}

// -------------------------------------------------------------- сигнал ------

struct Signal {
    scheme: Scheme,
    t_f: f64,
    n_frames: usize,
    des: Design,
    /// n_frames × dims координат PAM в [−1, 1].
    data: Vec<f64>,
}

impl Signal {
    #[inline]
    fn d(&self, m: i64, j: usize) -> f64 {
        if m < 0 || m >= self.n_frames as i64 {
            0.0
        } else {
            self.data[m as usize * self.des.dims + j]
        }
    }

    /// ∫ s(u)du по локальному времени внутри кадра `m`.
    fn frame_integral(&self, m: i64, u0: f64, u1: f64) -> f64 {
        let du = u1 - u0;
        if m < 0 || m >= self.n_frames as i64 {
            return MID * du;
        }
        match self.scheme.basis() {
            Scheme::A | Scheme::C => du * (MID + SWING * self.d(m, 0)),
            Scheme::B1 => {
                let a = self.des.amp;
                let mut s = du * (MID + a * self.d(m, 0));
                for n in 1..=self.des.k {
                    let w = 2.0 * PI * n as f64 / self.t_f;
                    let di = self.d(m, 2 * n - 1);
                    let dq = self.d(m, 2 * n);
                    s += a * (di * ((w * u1).sin() - (w * u0).sin()) / w
                        + dq * ((w * u0).cos() - (w * u1).cos()) / w);
                }
                s
            }
            Scheme::B2 | Scheme::B2C => {
                let a = self.des.amp;
                let mut s = du * MID;
                for n in 1..=self.des.k {
                    let w = PI * n as f64 / self.t_f;
                    s += a * self.d(m, n - 1) * ((w * u0).cos() - (w * u1).cos()) / w;
                }
                s
            }
        }
    }

    /// Мгновенное значение (для вырожденной выдержки).
    fn value(&self, t: f64) -> f64 {
        let m = (t / self.t_f).floor();
        let u = t - m * self.t_f;
        let e = 1e-9;
        self.frame_integral(m as i64, u, u + e) / e
    }

    /// Средний уровень за окно экспозиции [t0, t1] (боксkar).
    fn mean(&self, t0: f64, t1: f64) -> f64 {
        let dt = t1 - t0;
        if dt < 1e-12 {
            return self.value(t0);
        }
        let mut acc = 0.0;
        let mut t = t0;
        while t < t1 - 1e-15 {
            let m = (t / self.t_f).floor();
            // защита от вырождения при округлении: граница обязана быть строго правее t
            let bound = {
                let b = (m + 1.0) * self.t_f;
                if b > t {
                    b
                } else {
                    t + self.t_f
                }
            };
            let te = bound.min(t1);
            acc += self.frame_integral(m as i64, t - m * self.t_f, te - m * self.t_f);
            t = te;
        }
        acc / dt
    }
}

/// Тоновая кривая ISP (кодирование линейного света в drive-домен).
const GAMMA: f64 = 2.2;

/// Средний по окну экспозиции коэффициент ШИМ-пульсации подсветки.
///
/// Модель: свет умножается на (1 + m·sin(ω_p t + φ)). Здесь берётся СРЕДНЕЕ
/// пульсации по окну и применяется как скалярное усиление ко всему интегралу.
/// Это точно для кусочно-постоянного сигнала (схемы A/C) и приближение порядка
/// самого удерживаемого члена для B — оправданное тем, что f_ШИМ (сотни Гц —
/// килогерцы) на два порядка выше полосы данных (единицы — десятки Гц).
/// Ключевое здесь то, что боксkar выдержки давит пульсацию как sinc(f·t_exp),
/// а фаза пульсации РАЗНАЯ у разных строк (и разных кадров камеры) — то есть
/// остаток выглядит как структурный мультипликативный шум усиления.
fn pwm_gain(t0: f64, t1: f64, pwm: Option<(f64, f64)>, phi: f64) -> f64 {
    let Some((f, m)) = pwm else { return 1.0 };
    if m <= 0.0 || f <= 0.0 {
        return 1.0;
    }
    let w = 2.0 * PI * f;
    let dt = t1 - t0;
    if w * dt < 1e-9 {
        return 1.0 + m * (w * t0 + phi).sin();
    }
    1.0 + m * ((w * t0 + phi).cos() - (w * t1 + phi).cos()) / (w * dt)
}

/// Средний уровень СИГНАТУРНОГО канала за окно экспозиции (кусочно-постоянный).
fn sig_mean(t0: f64, t1: f64, t_f: f64, n_frames: usize) -> f64 {
    let lvl = |m: i64| -> f64 {
        if m < 0 || m >= n_frames as i64 {
            MID
        } else {
            SIG_LEVELS[(m as usize) % SIG_LEVELS.len()]
        }
    };
    let dt = t1 - t0;
    if dt < 1e-12 {
        return lvl((t0 / t_f).floor() as i64);
    }
    let mut acc = 0.0;
    let mut t = t0;
    while t < t1 - 1e-15 {
        let m = (t / t_f).floor();
        let bound = {
            let b = (m + 1.0) * t_f;
            if b > t {
                b
            } else {
                t + t_f
            }
        };
        let te = bound.min(t1);
        acc += lvl(m as i64) * (te - t);
        t = te;
    }
    acc / dt
}

// -------------------------------------------------- истина по «чистоте» -----

/// Сколько членов хвоста БИХ учитывать, чтобы отброшенный вес был < 1e-4.
fn jmax_for(alpha: f64) -> usize {
    if alpha <= 0.0 {
        0
    } else {
        ((1e-4f64).ln() / alpha.ln()).ceil().max(0.0) as usize
    }
}

const PUR_SPAN: usize = 96;

/// Эффективное ядро наблюдения (боксkar × хвост БИХ) разложено по кадрам
/// дисплея: возвращает (доминирующий кадр, его доля веса).
fn purity(ts: f64, t_exp: f64, alpha: f64, t_f: f64, jmax: usize) -> (i64, f64) {
    let base = (ts / t_f).floor() as i64;
    let mut w = [0.0f64; PUR_SPAN];
    let mut tot = 0.0;
    let te = t_exp.max(1e-12);
    for j in 0..=jmax {
        let wj = if alpha <= 0.0 {
            1.0
        } else {
            (1.0 - alpha) * alpha.powi(j as i32)
        };
        tot += wj;
        let t0 = ts - j as f64 * T_CAM;
        let t1 = t0 + te;
        let mut t = t0;
        while t < t1 - 1e-15 {
            let m = (t / t_f).floor();
            let bound = {
                let b = (m + 1.0) * t_f;
                if b > t {
                    b
                } else {
                    t + t_f
                }
            };
            let tend = bound.min(t1);
            let idx = base - m as i64 + 1;
            if idx >= 0 && (idx as usize) < PUR_SPAN {
                w[idx as usize] += wj * (tend - t) / te;
            }
            t = tend;
        }
    }
    let mut best = 0usize;
    for i in 1..PUR_SPAN {
        if w[i] > w[best] {
            best = i;
        }
    }
    (base + 1 - best as i64, w[best] / tot.max(1e-30))
}

// ------------------------------------------------------------- метрики ------

#[derive(Clone, Copy, Default)]
struct Acc {
    /// банда-захваты в измеряемом окне
    cap: u64,
    cap_clean_strict: u64,
    cap_clean_dec: u64,
    /// принятые приёмником банда-захваты (A/B1/B2 принимают всё)
    acc_n: u64,
    acc_clean_dec: u64,
    acc_ok: u64,
    /// RMS восстановления в PAM-координатах
    err2: f64,
    err_n: u64,
    /// доставка на уровне измерений (dim) кадра дисплея
    deliv: u64,
    correct: u64,
    /// ПОСТРАЙПОВАЯ доставка (A/C): голосование только ПО КАДРАМ КАМЕРЫ внутри
    /// своей полосы, без объединения полос. Моделирует реальность, где каждый
    /// страйп несёт СВОИ данные, а не повтор соседнего. В единицах «эквивалентов
    /// кадра дисплея» = (верных пар (кадр, полоса)) / число полос.
    stripe_ok: f64,
    stripe_deliv: f64,
    /// длительность измеряемого окна, с
    dur: f64,
    /// кадры камеры целиком
    cam: u64,
    cam_torn: u64,
    cam_clean_strict: u64,
    cam_clean_dec: u64,
    /// банды внутри РВАНЫХ кадров камеры
    tb: u64,
    tb_clean: u64,
    tb_acc: u64,
    tb_acc_ok: u64,
    /// кадры дисплея в измеряемом окне / из них выданные приёмником (остальное —
    /// СТИРАНИЯ: приёмник сам знает, что не смог, и не кормит FEC мусором)
    fr_tot: u64,
    fr_deliv: u64,
    /// То же, но со СТИРАНИЕМ ПО ОСТАТКУ МНК (только схемы B): приёмник знает свой
    /// шумовой пол из профиля калибровки и объявляет кадр стёртым, если невязка
    /// подгонки его превышает. Не стоит ни площади, ни отсчётов — чистая приёмная
    /// сторона, в отличие от сигнатурной полосы.
    res_deliv: u64,
    res_correct: u64,
    res_fr_deliv: u64,
}

impl Acc {
    fn add(&mut self, o: &Acc) {
        self.cap += o.cap;
        self.cap_clean_strict += o.cap_clean_strict;
        self.cap_clean_dec += o.cap_clean_dec;
        self.acc_n += o.acc_n;
        self.acc_clean_dec += o.acc_clean_dec;
        self.acc_ok += o.acc_ok;
        self.err2 += o.err2;
        self.err_n += o.err_n;
        self.deliv += o.deliv;
        self.correct += o.correct;
        self.stripe_ok += o.stripe_ok;
        self.stripe_deliv += o.stripe_deliv;
        self.dur += o.dur;
        self.cam += o.cam;
        self.cam_torn += o.cam_torn;
        self.cam_clean_strict += o.cam_clean_strict;
        self.cam_clean_dec += o.cam_clean_dec;
        self.tb += o.tb;
        self.tb_clean += o.tb_clean;
        self.tb_acc += o.tb_acc;
        self.tb_acc_ok += o.tb_acc_ok;
        self.fr_tot += o.fr_tot;
        self.fr_deliv += o.fr_deliv;
        self.res_deliv += o.res_deliv;
        self.res_correct += o.res_correct;
        self.res_fr_deliv += o.res_fr_deliv;
    }

    fn clean_dec(&self) -> f64 {
        ratio(self.cap_clean_dec, self.cap)
    }
    fn accept(&self) -> f64 {
        ratio(self.acc_n, self.cap)
    }
    fn acc_prec(&self) -> f64 {
        ratio(self.acc_ok, self.acc_n)
    }
    fn rms(&self) -> f64 {
        if self.err_n == 0 {
            f64::NAN
        } else {
            (self.err2 / self.err_n as f64).sqrt()
        }
    }
    fn err(&self) -> f64 {
        if self.deliv == 0 {
            f64::NAN
        } else {
            1.0 - self.correct as f64 / self.deliv as f64
        }
    }
    fn rate(&self, scheme: Scheme) -> f64 {
        if self.dur <= 0.0 {
            0.0
        } else {
            self.correct as f64 * scheme.rate_scale() / self.dur
        }
    }
    fn stripe_rate(&self, scheme: Scheme) -> f64 {
        if self.dur <= 0.0 || self.stripe_deliv <= 0.0 {
            f64::NAN
        } else {
            self.stripe_ok * scheme.rate_scale() / self.dur
        }
    }
    fn stripe_err(&self) -> f64 {
        if self.stripe_deliv <= 0.0 {
            f64::NAN
        } else {
            1.0 - self.stripe_ok / self.stripe_deliv
        }
    }
    /// Доля СТЁРТЫХ кадров дисплея (приёмник отказался выдавать).
    fn erasure(&self) -> f64 {
        if self.fr_tot == 0 {
            f64::NAN
        } else {
            1.0 - self.fr_deliv as f64 / self.fr_tot as f64
        }
    }
    /// Метрики с включённым стиранием по остатку МНК.
    fn res_rate(&self, scheme: Scheme) -> f64 {
        if self.dur <= 0.0 {
            f64::NAN
        } else {
            self.res_correct as f64 * scheme.rate_scale() / self.dur
        }
    }
    fn res_err(&self) -> f64 {
        if self.res_deliv == 0 {
            f64::NAN
        } else {
            1.0 - self.res_correct as f64 / self.res_deliv as f64
        }
    }
    fn res_erasure(&self) -> f64 {
        if self.fr_tot == 0 {
            f64::NAN
        } else {
            1.0 - self.res_fr_deliv as f64 / self.fr_tot as f64
        }
    }
    /// Скорость, обнулённая там, где FEC не вытянет долю ошибок.
    fn usable(&self, scheme: Scheme) -> f64 {
        let e = self.err();
        if e.is_nan() || e > FEC_ERR_BUDGET {
            0.0
        } else {
            self.rate(scheme)
        }
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

// ----------------------------------------------------------- решатель -------

/// МНК через нормальные уравнения, гауссово исключение (матрица SPD).
fn solve(g: &mut [f64], b: &mut [f64], d: usize) -> Option<Vec<f64>> {
    let mut mx = 0.0f64;
    for i in 0..d {
        mx = mx.max(g[i * d + i]);
    }
    for i in 0..d {
        g[i * d + i] += 1e-12 * mx;
    }
    for i in 0..d {
        let p = g[i * d + i];
        if !(p.abs() > 1e-14 * mx.max(1e-30)) {
            return None;
        }
        for r in i + 1..d {
            let f = g[r * d + i] / p;
            if f != 0.0 {
                for c in i..d {
                    g[r * d + c] -= f * g[i * d + c];
                }
                b[r] -= f * b[i];
            }
        }
    }
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut s = b[i];
        for c in i + 1..d {
            s -= g[i * d + c] * x[c];
        }
        x[i] = s / g[i * d + i];
    }
    Some(x)
}

// -------------------------------------------------------------- прогон ------

fn quantize(v: f64) -> f64 {
    (v * 255.0).round().clamp(0.0, 255.0) / 255.0
}

fn simulate(scheme: Scheme, cfg: &Cfg, des: Design, point: usize, trial: usize) -> Acc {
    let mut a = Acc::default();
    let t_f = cfg.t_f();
    let rows = cfg.rows;
    let bands = cfg.bands();
    let bh = rows / bands;
    let dims = des.dims;
    let mut rng = Rng::new(seed_for(point, trial));

    // ---- полезная нагрузка кадров дисплея
    let mut truth = vec![0usize; cfg.n_frames * dims];
    let mut data = vec![0.0f64; cfg.n_frames * dims];
    for i in 0..cfg.n_frames * dims {
        let s = rng.next_u32_below(M_PAM as u32) as usize;
        truth[i] = s;
        data[i] = pam_level(s);
    }
    let sig = Signal {
        scheme,
        t_f,
        n_frames: cfg.n_frames,
        des,
        data,
    };

    // Взаимная фаза камера/дисплей периодична с периодом gcd(T_f, T_CAM): сдвиг на
    // T_CAM переименовывает кадр камеры, сдвиг на T_f — кадр дисплея. Для всех наших
    // H одно делит другое, поэтому период = min(T_f, T_CAM). Стратифицировать НАДО
    // именно по нему: развёртка по [0, T_f) при чётном H вырождается (при H = 6 шесть
    // «равномерных» фаз дают всего две различные взаимные фазы).
    // Иррациональный сдвиг убирает точное попадание отсчёта на границу кадра
    // (событие нулевой меры в жизни, но не в арифметике).
    let phase_period = t_f.min(T_CAM);
    let phase = cfg
        .fixed_phase
        .unwrap_or_else(|| ((trial as f64 + 0.5) / cfg.trials as f64 + PHASE_NUDGE) * phase_period);

    // ---- симуляция камеры
    let dur = cfg.n_frames as f64 * t_f;
    let n_cam = ((dur / T_CAM).ceil() as usize).max(1);
    let want_sig = scheme.has_sig();
    let mut yd = vec![0.0f64; n_cam * rows];
    let mut ys = if want_sig {
        vec![0.0f64; n_cam * rows]
    } else {
        Vec::new()
    };
    let mut prev_d = vec![0.0f64; rows];
    let mut prev_s = vec![0.0f64; rows];
    let sig_sigma = cfg.sigma * SIG_AREA_DIV.sqrt();

    // PRNU — ФИКСИРОВАННЫЙ по времени разброс усиления «клетки»: не усредняется
    // ни по кадрам камеры, ни БИХ-ом, поэтому это чистая систематика, а не шум.
    // Сигнатурная полоса — другие клетки, свой разброс.
    // (розыгрыш ТОЛЬКО когда помеха включена — иначе поток ГПСЧ сдвинулся бы и
    // числа разделов 1–5 поехали бы в третьей значащей цифре без причины)
    let draw_prnu = |rng: &mut Rng| -> Vec<f64> {
        if cfg.prnu > 0.0 {
            (0..rows).map(|_| 1.0 + cfg.prnu * rng.gaussian()).collect()
        } else {
            vec![1.0; rows]
        }
    };
    let prnu_d = draw_prnu(&mut rng);
    let prnu_s = draw_prnu(&mut rng);
    // ШИМ подсветки свободно бежит относительно дисплея: своя фаза на попытку.
    let pwm_phi = if cfg.pwm.is_some() {
        2.0 * PI * rng.next_f64()
    } else {
        0.0
    };

    // Одно наблюдение: свет → ШИМ-пульсация → PRNU → сенсорный шум → БИХ ISP
    // (в линейном свете ИЛИ в drive-домене после тоновой кривой) → 8 бит.
    let apply = |x: f64, prev: &mut f64, first: bool, quant: bool| -> f64 {
        if cfg.drive_domain {
            let e = x.clamp(0.0, 1.0).powf(1.0 / GAMMA);
            let mut ef = if first {
                e
            } else {
                (1.0 - cfg.alpha) * e + cfg.alpha * *prev
            };
            if quant {
                ef = quantize(ef);
            }
            *prev = ef;
            ef.powf(GAMMA)
        } else {
            let mut v = if first {
                x
            } else {
                (1.0 - cfg.alpha) * x + cfg.alpha * *prev
            };
            if quant {
                v = quantize(v);
            }
            *prev = v;
            v
        }
    };

    for k in 0..n_cam {
        for r in 0..rows {
            let ts = phase + k as f64 * T_CAM + (r as f64 + 0.5) / rows as f64 * cfg.t_read;
            let g_pwm = pwm_gain(ts, ts + cfg.t_exp, cfg.pwm, pwm_phi);
            let xd = prnu_d[r] * g_pwm * sig.mean(ts, ts + cfg.t_exp) + cfg.sigma * rng.gaussian();
            yd[k * rows + r] = apply(xd, &mut prev_d[r], k == 0, cfg.quant);
            if want_sig {
                let xs = prnu_s[r] * g_pwm * sig_mean(ts, ts + cfg.t_exp, t_f, cfg.n_frames)
                    + sig_sigma * rng.gaussian();
                ys[k * rows + r] = apply(xs, &mut prev_s[r], k == 0, cfg.quant);
            }
        }
    }

    // ---- приём A/C побандово + истина «чистоты» (одинакова для всех схем)
    let jmax = jmax_for(cfg.alpha);
    let lo_m = BURN as i64;
    let hi_m = (cfg.n_frames - TAIL) as i64;
    let mut votes = vec![[0u32; M_PAM]; cfg.n_frames];
    let mut votes_b = vec![[0u32; M_PAM]; cfg.n_frames * bands];
    // Прошла ли (кадр камеры, полоса) проверку сигнатуры — вход МНК для B2C.
    let mut band_ok = vec![false; n_cam * bands];
    let mut row_dom = vec![0i64; rows];
    let mut row_pur = vec![0.0f64; rows];

    for k in 0..n_cam {
        for r in 0..rows {
            let ts = phase + k as f64 * T_CAM + (r as f64 + 0.5) / rows as f64 * cfg.t_read;
            let (dom, pur) = purity(ts, cfg.t_exp, cfg.alpha, t_f, jmax);
            row_dom[r] = dom;
            row_pur[r] = pur;
        }
        // рваность и чистота кадра камеры целиком
        let torn = row_dom.iter().any(|&d| d != row_dom[0]);
        let f_strict = row_pur.iter().all(|&p| p >= 1.0 - CLEAN_EPS) && !torn;
        let f_dec = row_pur.iter().all(|&p| p >= 1.0 - BETA_TOL) && !torn;
        let ctr_ts = phase + k as f64 * T_CAM + 0.5 * cfg.t_read;
        let frame_m = ((ctr_ts + cfg.t_exp * 0.5) / t_f).floor() as i64;
        let frame_in = frame_m >= lo_m && frame_m < hi_m;
        if frame_in {
            a.cam += 1;
            if torn {
                a.cam_torn += 1;
            }
            if f_strict {
                a.cam_clean_strict += 1;
            }
            if f_dec {
                a.cam_clean_dec += 1;
            }
        }

        for b in 0..bands {
            let (r0, r1) = (b * bh, (b + 1) * bh);
            let ctr = phase
                + k as f64 * T_CAM
                + ((r0 + r1) as f64 * 0.5) / rows as f64 * cfg.t_read;
            let m_hat = ((ctr + cfg.t_exp * 0.5) / t_f).floor() as i64;
            if m_hat < lo_m || m_hat >= hi_m {
                continue;
            }
            let dom0 = row_dom[r0];
            let same = (r0..r1).all(|r| row_dom[r] == dom0);
            let pmin = (r0..r1).fold(1.0f64, |acc, r| acc.min(row_pur[r]));
            let clean_strict = same && pmin >= 1.0 - CLEAN_EPS;
            let clean_dec = same && pmin >= 1.0 - BETA_TOL;

            a.cap += 1;
            if clean_strict {
                a.cap_clean_strict += 1;
            }
            if clean_dec {
                a.cap_clean_dec += 1;
            }
            let torn_band = torn && frame_in;
            if torn_band {
                a.tb += 1;
                if clean_dec {
                    a.tb_clean += 1;
                }
            }

            // Проверка сигнатуры (C и B2C): побандово, поэтому разрыв ЛОКАЛИЗУЕТСЯ.
            let accept = if scheme.has_sig() {
                let gsum: f64 = (r0..r1).map(|r| ys[k * rows + r]).sum::<f64>() / bh as f64;
                let exp_g = SIG_LEVELS[(m_hat as usize) % SIG_LEVELS.len()];
                (gsum - exp_g).abs() <= TAU
            } else {
                true
            };
            if scheme == Scheme::B2C {
                // B2C: сигнатура работает СТИРАНИЕМ входных отсчётов МНК.
                if accept {
                    band_ok[k * bands + b] = true;
                    a.acc_n += 1;
                    if clean_dec {
                        a.acc_clean_dec += 1;
                    }
                    if torn_band {
                        a.tb_acc += 1;
                    }
                }
                continue;
            }
            // приём A/C: побандовый оценщик значения + голосование
            if !scheme.is_hard() {
                continue;
            }
            let vsum: f64 = (r0..r1).map(|r| yd[k * rows + r]).sum::<f64>() / bh as f64;
            let d_hat = (vsum - MID) / SWING;
            let sym = pam_slice(d_hat);
            if !accept {
                continue;
            }
            let ok = sym == truth[m_hat as usize * dims];
            a.acc_n += 1;
            if clean_dec {
                a.acc_clean_dec += 1;
            }
            if ok {
                a.acc_ok += 1;
            }
            let e = d_hat - pam_level(truth[m_hat as usize * dims]);
            a.err2 += e * e;
            a.err_n += 1;
            votes[m_hat as usize][sym] += 1;
            votes_b[m_hat as usize * bands + b][sym] += 1;
            if torn_band {
                a.tb_acc += 1;
                if ok {
                    a.tb_acc_ok += 1;
                }
            }
        }
    }

    match scheme {
        Scheme::A | Scheme::C => {
            for m in lo_m..hi_m {
                let v = &votes[m as usize];
                let tot: u32 = v.iter().sum();
                if tot == 0 {
                    continue;
                }
                let mut best = 0usize;
                for s in 1..M_PAM {
                    if v[s] > v[best] {
                        best = s;
                    }
                }
                a.deliv += 1;
                if best == truth[m as usize * dims] {
                    a.correct += 1;
                }
            }
            // построчная (по-страйповая) доставка: голосуем ТОЛЬКО по кадрам камеры
            let (mut bd, mut bo) = (0u64, 0u64);
            for m in lo_m..hi_m {
                for b in 0..bands {
                    let v = &votes_b[m as usize * bands + b];
                    let tot: u32 = v.iter().sum();
                    if tot == 0 {
                        continue;
                    }
                    let mut best = 0usize;
                    for s in 1..M_PAM {
                        if v[s] > v[best] {
                            best = s;
                        }
                    }
                    bd += 1;
                    if best == truth[m as usize * dims] {
                        bo += 1;
                    }
                }
            }
            a.stripe_deliv = bd as f64 / bands as f64;
            a.stripe_ok = bo as f64 / bands as f64;
            a.fr_tot = (hi_m - lo_m) as u64;
            a.fr_deliv = a.deliv;
        }
        Scheme::B1 | Scheme::B2 | Scheme::B2C => {
            // раскладываем строчные отсчёты по кадрам дисплея (центр экспозиции).
            // B2C: отсчёты из полос, провалившихся на сигнатуре, СТИРАЮТСЯ здесь —
            // до МНК, а не после демодуляции.
            // (u, y, прошла ли сигнатуру)
            let mut bucket: Vec<Vec<(f64, f64, bool)>> = vec![Vec::new(); cfg.n_frames];
            for k in 0..n_cam {
                for r in 0..rows {
                    let ts = phase + k as f64 * T_CAM + (r as f64 + 0.5) / rows as f64 * cfg.t_read;
                    let m = ((ts + cfg.t_exp * 0.5) / t_f).floor() as i64;
                    if m < lo_m || m >= hi_m {
                        continue;
                    }
                    let ok = scheme != Scheme::B2C || band_ok[k * bands + r / bh];
                    bucket[m as usize].push((ts - m as f64 * t_f, yd[k * rows + r], ok));
                }
            }
            a.fr_tot = (hi_m - lo_m) as u64;
            let mut row = vec![0.0f64; dims];
            let mut keep: Vec<(f64, f64)> = Vec::new();
            for m in lo_m..hi_m {
                let all = &bucket[m as usize];
                // Правило приёмника B2C. Стирать отсчёты можно ТОЛЬКО пока остаётся
                // чем решать: при per-cell и малом H бюджет отсчётов на кадр дисплея
                // всего 3, и жёсткое стирание съедает саму возможность решить МНК.
                //   1) чистых хватает на МНК — решаем по чистым;
                //   2) не хватает, но большинство чисто — решаем по ВСЕМ (одиночный
                //      загрязнённый отсчёт МНК переживает как выброс);
                //   3) иначе — СТИРАНИЕ кадра (для FEC это erasure, а не ошибка).
                keep.clear();
                let n_clean = all.iter().filter(|s| s.2).count();
                if n_clean >= dims + 1 {
                    keep.extend(all.iter().filter(|s| s.2).map(|s| (s.0, s.1)));
                } else if all.len() >= dims + 1 && 2 * n_clean >= all.len() {
                    keep.extend(all.iter().map(|s| (s.0, s.1)));
                }
                let samples = &keep;
                if samples.len() < dims + 1 {
                    continue;
                }
                let mut g = vec![0.0f64; dims * dims];
                let mut rhs = vec![0.0f64; dims];
                for &(u, y) in samples {
                    design_row(scheme, des.k, u, t_f, cfg.t_exp, cfg.alpha, &mut row);
                    for c in row.iter_mut() {
                        *c *= des.amp;
                    }
                    let yy = y - MID;
                    for i in 0..dims {
                        rhs[i] += row[i] * yy;
                        for j in 0..dims {
                            g[i * dims + j] += row[i] * row[j];
                        }
                    }
                }
                let Some(x) = solve(&mut g, &mut rhs, dims) else {
                    continue;
                };
                a.fr_deliv += 1;
                // невязка подгонки: дешёвый признак «этот кадр модель не описывает»
                let mut r2 = 0.0f64;
                for &(u, y) in samples {
                    design_row(scheme, des.k, u, t_f, cfg.t_exp, cfg.alpha, &mut row);
                    let mut pred = MID;
                    for j in 0..dims {
                        pred += row[j] * des.amp * x[j];
                    }
                    r2 += (y - pred) * (y - pred);
                }
                let resid = (r2 / samples.len() as f64).sqrt();
                let res_ok = resid <= RES_K * cfg.sigma.max(1e-6);
                if res_ok {
                    a.res_fr_deliv += 1;
                }
                for j in 0..dims {
                    let t = truth[m as usize * dims + j];
                    let e = x[j] - pam_level(t);
                    a.err2 += e * e;
                    a.err_n += 1;
                    a.deliv += 1;
                    let ok = pam_slice(x[j]) == t;
                    if ok {
                        a.correct += 1;
                    }
                    if res_ok {
                        a.res_deliv += 1;
                        if ok {
                            a.res_correct += 1;
                        }
                    }
                }
            }
            if scheme != Scheme::B2C {
                // B1/B2 ничего не отбраковывают: accept-статистика = все захваты
                a.acc_n = a.cap;
                a.acc_clean_dec = a.cap_clean_dec;
                a.acc_ok = a.cap_clean_dec;
            }
        }
    }

    a.dur = (hi_m - lo_m) as f64 * t_f;
    a
}

fn run_point(scheme: Scheme, cfg: &Cfg, point: usize) -> (Acc, Design) {
    let des = design(scheme, cfg);
    let mut acc = Acc::default();
    for t in 0..cfg.trials {
        let r = simulate(scheme, cfg, des, point, t);
        acc.add(&r);
    }
    (acc, des)
}

// -------------------------------------------------------------- вывод -------

fn pct(x: f64) -> String {
    if x.is_nan() {
        "—".into()
    } else {
        format!("{:.1}%", 100.0 * x)
    }
}

fn num(x: f64, d: usize) -> String {
    if x.is_nan() {
        "—".into()
    } else {
        format!("{x:.d$}")
    }
}

pub fn cmd_tprobe() {
    let t0 = Instant::now();
    println!("# tprobe — временной канал дисплей→камера (Монте-Карло, 1-D во времени)");
    println!(
        "\nОбщие параметры: дисплей 60 Гц, камера 30 fps, выдержка t_exp = {:.1} мс, \
         развёртка t_read = {:.0} мс (кроме развёртки 3), α_ISP = {ALPHA_REF} (кроме развёртки 2), \
         PAM-{M_PAM} (3 бита/измерение), размах драйва [0.05, 0.95], \
         σ на строчный отсчёт = {SIGMA_OBS}, 8-битное квантование после ISP, \
         {ROWS} строк × {BANDS} банд, {N_FRAMES} кадров дисплея × {N_TRIALS} попыток (фаза стратифицирована).",
        T_EXP_REF * 1000.0,
        T_READ_REF * 1000.0
    );
    println!(
        "«Чистый» захват = вся полоса видит ОДИН кадр дисплея с примесью ≤ β_tol = {:.4} \
         (примесь, гарантированно не ломающая символ). Полуинтервал решения в PAM-координатах = {PAM_HALF:.4}.",
        BETA_TOL
    );

    sanity_gate();
    sweep_h();
    sweep_alpha();
    sweep_read();
    sweep_phase();
    reality_checks();
    sweep_noise();
    sweep_drive();
    design_point();

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

// ------------------------------------------------------------ 0. gate -------

fn sanity_gate() {
    println!("\n## 0. Sanity gate — идеальная мгновенная выборка (t_exp→0, α=0, t_read=0, H=16)");
    println!(
        "\nПри идеальном канале обе ветви обязаны восстановить значения точно и \
         никогда не перепутать кадр дисплея. Второй блок — тот же гейт с включённым \
         шумом и 8-битным квантованием (шумовой пол приёмника)."
    );
    println!("\n| режим | схема | dims/кадр | RMS (PAM-коорд) | ошибок среди доставленного | доставлено симв/с |");
    println!("|---|---|---|---|---|---|");
    for (tag, sigma, quant) in [("идеал σ=0, без квант.", 0.0, false), ("σ=0.01 + 8 бит", SIGMA_OBS, true)] {
        for (i, &s) in ALL.iter().enumerate() {
            let mut cfg = Cfg::reference(16);
            cfg.t_exp = 1e-9;
            cfg.t_read = 0.0;
            cfg.alpha = 0.0;
            cfg.sigma = sigma;
            cfg.quant = quant;
            let (a, d) = run_point(s, &cfg, 100 + i + if quant { 10 } else { 0 });
            println!(
                "| {tag} | {} | {} | {} | {} | {} |",
                s.label(),
                d.dims,
                report::sig4(a.rms()),
                pct(a.err()),
                num(a.rate(s), 2)
            );
        }
    }
}

// --------------------------------------------------------------- 1. H -------

const H_LIST: [u32; 8] = [1, 2, 4, 6, 8, 12, 16, 24];

fn sweep_h() {
    println!("\n## 1. Развёртка по H (α = {ALPHA_REF}, t_read = 16 мс)");

    let mut acc = [[Acc::default(); H_LIST.len()]; 4];
    let mut des = [[Design { k: 0, dims: 1, amp: SWING }; H_LIST.len()]; 4];
    for (si, &s) in ALL.iter().enumerate() {
        for (hi, &h) in H_LIST.iter().enumerate() {
            let cfg = Cfg::reference(h);
            let (a, d) = run_point(s, &cfg, 1000 + si * 100 + hi);
            acc[si][hi] = a;
            des[si][hi] = d;
        }
    }

    let head = |t: &str| {
        let mut s = format!("| {t} \\ H → |");
        for h in H_LIST {
            s.push_str(&format!(" {h} |"));
        }
        s
    };
    let sep = {
        let mut s = String::from("|---|");
        for _ in H_LIST {
            s.push_str("---|");
        }
        s
    };

    println!("\n### 1.1 Доля ЧИСТЫХ (несмешанных) захватов — геометрия канала, одна для всех схем");
    println!("{}", head("критерий"));
    println!("{sep}");
    for (lab, f) in [
        ("банда, строгий (примесь < 1e-3)", 0usize),
        ("банда, decode-релевантный (≤ β_tol)", 1),
        ("кадр камеры целиком, decode-рел.", 2),
    ] {
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| {
                let a = &acc[0][hi];
                pct(match f {
                    0 => ratio(a.cap_clean_strict, a.cap),
                    1 => ratio(a.cap_clean_dec, a.cap),
                    _ => ratio(a.cam_clean_dec, a.cam),
                })
            })
            .collect();
        println!("{}", report::table_row(lab, &cells));
    }

    println!("\n### 1.2 Измерений данных на кадр дисплея (выбранный дизайн B)");
    println!("{}", head("схема"));
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| {
                if s == Scheme::C {
                    format!("1×{:.3}", C_RATE_SCALE)
                } else {
                    format!("{}", des[si][hi].dims)
                }
            })
            .collect();
        println!("{}", report::table_row(s.label(), &cells));
    }

    println!("\n### 1.3 RMS восстановления значения (PAM-координаты; полуинтервал решения {PAM_HALF:.4})");
    println!("{}", head("схема"));
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| report::sig4(acc[si][hi].rms()))
            .collect();
        println!("{}", report::table_row(s.label(), &cells));
    }

    println!("\n### 1.4 ЭФФЕКТИВНАЯ СКОРОСТЬ — доставлено И верно, символов/с (3 бита/символ)");
    println!("{}", head("схема"));
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| num(acc[si][hi].rate(s), 2))
            .collect();
        println!("{}", report::table_row(s.label(), &cells));
    }

    println!(
        "\n### 1.4b ПО-СТРАЙПОВАЯ скорость A/C — голосование только по кадрам камеры\n\
         В таблице 1.4 один кадр дисплея восстанавливается голосованием по всем {BANDS} полосам, \
         то есть полосы считаются повтором одного символа. В реальном коде каждый страйп несёт \
         СВОИ данные, и объединять их нельзя: восстановить кадр можно только собрав все страйпы \
         по разным кадрам камеры. Здесь — та же метрика без межполосного голосования (для B \
         аналог — строка «per-stripe» в разделе 5)."
    );
    println!("{}", head("схема"));
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        if s != Scheme::A && s != Scheme::C {
            continue;
        }
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| num(acc[si][hi].stripe_rate(s), 2))
            .collect();
        println!("{}", report::table_row(s.label(), &cells));
    }
    for (si, &s) in ALL.iter().enumerate() {
        if s != Scheme::A && s != Scheme::C {
            continue;
        }
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| pct(acc[si][hi].stripe_err()))
            .collect();
        println!(
            "{}",
            report::table_row(&format!("{} — ошибок", s.label()), &cells)
        );
    }

    println!("\n### 1.5 Доля ОШИБОК среди доставленного (бюджет FEC RS(16,8) ≈ 25%, рабочий порог 15%)");
    println!("{}", head("схема"));
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| pct(acc[si][hi].err()))
            .collect();
        println!("{}", report::table_row(s.label(), &cells));
    }

    println!("\n### 1.6 ГОДНАЯ скорость (та же, но обнулена там, где ошибок > 15% — FEC не вытянет)");
    println!("{}", head("схема"));
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..H_LIST.len())
            .map(|hi| {
                let a = &acc[si][hi];
                let e = a.err();
                if e.is_nan() || e > 0.15 {
                    "0".into()
                } else {
                    num(a.rate(s), 2)
                }
            })
            .collect();
        println!("{}", report::table_row(s.label(), &cells));
    }

    println!("\n### 1.7 Схема C: отбраковка");
    println!("| H | принято захватов | из принятых реально чистых | из принятых символ верен | принято × верно |");
    println!("|---|---|---|---|---|");
    for (hi, &h) in H_LIST.iter().enumerate() {
        let a = &acc[3][hi];
        println!(
            "| {h} | {} | {} | {} | {} |",
            pct(a.accept()),
            pct(ratio(a.acc_clean_dec, a.acc_n)),
            pct(a.acc_prec()),
            pct(a.accept() * a.acc_prec())
        );
    }
}

// ----------------------------------------------------------- 2. alpha -------

const ALPHAS: [f64; 5] = [0.0, 0.15, 0.3, 0.5, 0.7];

fn sweep_alpha() {
    println!("\n## 2. Развёртка по коэффициенту смешивания ISP α (t_read = 16 мс)");

    for (hii, &h) in [6u32, 16].iter().enumerate() {
        println!("\n### 2.{} H = {h}", hii + 1);
        let mut acc = [[Acc::default(); ALPHAS.len()]; 4];
        let mut des = [[Design { k: 0, dims: 1, amp: SWING }; ALPHAS.len()]; 4];
        for (si, &s) in ALL.iter().enumerate() {
            for (ai, &al) in ALPHAS.iter().enumerate() {
                let mut cfg = Cfg::reference(h);
                cfg.alpha = al;
                let (a, d) = run_point(s, &cfg, 2000 + hii * 500 + si * 50 + ai);
                acc[si][ai] = a;
                des[si][ai] = d;
            }
        }
        let head = {
            let mut s = String::from("| схема \\ α → |");
            for a in ALPHAS {
                s.push_str(&format!(" {a} |"));
            }
            s
        };
        let sep = {
            let mut s = String::from("|---|");
            for _ in ALPHAS {
                s.push_str("---|");
            }
            s
        };

        println!("\n**чистых захватов (decode-рел.), одинаково для всех схем**");
        println!("{head}");
        println!("{sep}");
        let cells: Vec<String> = (0..ALPHAS.len()).map(|ai| pct(acc[0][ai].clean_dec())).collect();
        println!("{}", report::table_row("банда", &cells));
        let cells: Vec<String> = (0..ALPHAS.len())
            .map(|ai| pct(ratio(acc[0][ai].cam_clean_dec, acc[0][ai].cam)))
            .collect();
        println!("{}", report::table_row("кадр камеры", &cells));

        println!("\n**эффективная скорость, симв/с**");
        println!("{head}");
        println!("{sep}");
        for (si, &s) in ALL.iter().enumerate() {
            let cells: Vec<String> = (0..ALPHAS.len()).map(|ai| num(acc[si][ai].rate(s), 2)).collect();
            println!("{}", report::table_row(s.label(), &cells));
        }

        println!("\n**ошибок среди доставленного**");
        println!("{head}");
        println!("{sep}");
        for (si, &s) in ALL.iter().enumerate() {
            let cells: Vec<String> = (0..ALPHAS.len()).map(|ai| pct(acc[si][ai].err())).collect();
            println!("{}", report::table_row(s.label(), &cells));
        }

        println!("\n**RMS восстановления (PAM-коорд.)**");
        println!("{head}");
        println!("{sep}");
        for (si, &s) in ALL.iter().enumerate() {
            let cells: Vec<String> = (0..ALPHAS.len()).map(|ai| report::sig4(acc[si][ai].rms())).collect();
            println!("{}", report::table_row(s.label(), &cells));
        }

        println!("\n**измерений на кадр (дизайн B)**");
        println!("{head}");
        println!("{sep}");
        for si in [1usize, 2] {
            let cells: Vec<String> = (0..ALPHAS.len()).map(|ai| format!("{}", des[si][ai].dims)).collect();
            println!("{}", report::table_row(ALL[si].label(), &cells));
        }

        if h == 16 {
            println!("\n**C: принято / из принятых верно**");
            println!("{head}");
            println!("{sep}");
            let cells: Vec<String> = (0..ALPHAS.len()).map(|ai| pct(acc[3][ai].accept())).collect();
            println!("{}", report::table_row("принято", &cells));
            let cells: Vec<String> = (0..ALPHAS.len()).map(|ai| pct(acc[3][ai].acc_prec())).collect();
            println!("{}", report::table_row("верно из принятых", &cells));
        }
    }

    backout_alpha();
}

/// Обратная оценка α из единственного полевого факта: ~84% чистых захватов при H = 16.
fn backout_alpha() {
    println!("\n### 2.3 Обратная оценка α по полевому факту «~84% чистых захватов при H = 16»");
    println!(
        "\nДопущения: (1) «чистый захват» в поле = кадр камеры, декодированный без порчи, \
         поэтому сопоставляем с долей КАДРОВ КАМЕРЫ, у которых примесь ≤ β_tol = {:.4}; \
         (2) t_exp = 16.7 мс; (3) t_read неизвестно — даём три варианта; \
         (4) интегрирование и БИХ линейны по свету. \
         Геометрия (выдержка + развёртка) уже сама по себе даёт грязные захваты ДАЖЕ при α = 0 — \
         это верхняя граница «чистоты», и она задаёт, сколько места остаётся под α.",
        BETA_TOL
    );
    let grid: [f64; 8] = [0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3];
    let reads: [(f64, &str); 3] = [(0.008, "8 мс"), (0.016, "16 мс"), (0.033, "33 мс")];
    let mut head = String::from("| t_read \\ α → |");
    for a in grid {
        head.push_str(&format!(" {a} |"));
    }
    let mut sep = String::from("|---|");
    for _ in grid {
        sep.push_str("---|");
    }
    println!("\n**доля чистых КАДРОВ КАМЕРЫ при H = 16 (decode-релевантный критерий)**");
    println!("{head}");
    println!("{sep}");
    let mut curves: Vec<(&str, Vec<f64>)> = Vec::new();
    for (ri, &(tr, lab)) in reads.iter().enumerate() {
        let mut vals = Vec::new();
        for (ai, &al) in grid.iter().enumerate() {
            let mut cfg = Cfg::reference(16);
            cfg.t_read = tr;
            cfg.alpha = al;
            let (a, _) = run_point(Scheme::A, &cfg, 3000 + ri * 50 + ai);
            vals.push(ratio(a.cam_clean_dec, a.cam));
        }
        let cells: Vec<String> = vals.iter().map(|&v| pct(v)).collect();
        println!("{}", report::table_row(lab, &cells));
        curves.push((lab, vals));
    }
    println!("\n**считывание оценки (линейная интерполяция до уровня 84%)**");
    for (lab, vals) in &curves {
        let msg;
        if vals[0] < 0.84 {
            msg = format!(
                "уже при α = 0 чистых лишь {} < 84% — одна геометрия объясняет наблюдение целиком, α неотличима от 0",
                pct(vals[0])
            );
        } else {
            let mut found = None;
            for i in 1..grid.len() {
                if vals[i] < 0.84 && vals[i - 1] >= 0.84 {
                    let t = (vals[i - 1] - 0.84) / (vals[i - 1] - vals[i]).max(1e-12);
                    found = Some(grid[i - 1] + t * (grid[i] - grid[i - 1]));
                    break;
                }
            }
            msg = match found {
                Some(a) => format!("α ≈ {a:.3}"),
                None => format!(
                    "не пересекает 84% на сетке (при α = {} ещё {})",
                    grid[grid.len() - 1],
                    pct(vals[grid.len() - 1])
                ),
            };
        }
        println!("- t_read = {lab}: {msg}");
    }
}

// ------------------------------------------------------------ 3. read -------

fn sweep_read() {
    println!("\n## 3. Rolling shutter (H = 6, α = {ALPHA_REF})");
    let reads: [(f64, &str); 4] = [
        (0.0, "0 (global)"),
        (0.008, "8 мс"),
        (0.016, "16 мс"),
        (0.033, "33 мс"),
    ];

    let mut acc = [[Acc::default(); 4]; 4];
    for (si, &s) in ALL.iter().enumerate() {
        for (ri, &(tr, _)) in reads.iter().enumerate() {
            let mut cfg = Cfg::reference(6);
            cfg.t_read = tr;
            let (a, _) = run_point(s, &cfg, 4000 + si * 50 + ri);
            acc[si][ri] = a;
        }
    }

    println!("\n### 3.1 Рваность и спасение рваных кадров");
    println!(
        "| t_read | рваных кадров камеры | банд в рваных кадрах чистых | A: верных банд в рваных | \
         C: принято банд в рваных | C: принято И верно | C: точность принятых |"
    );
    println!("|---|---|---|---|---|---|---|");
    for (ri, &(_, lab)) in reads.iter().enumerate() {
        let a = &acc[0][ri];
        let c = &acc[3][ri];
        println!(
            "| {lab} | {} | {} | {} | {} | {} | {} |",
            pct(ratio(a.cam_torn, a.cam)),
            pct(ratio(a.tb_clean, a.tb)),
            pct(ratio(a.tb_acc_ok, a.tb)),
            pct(ratio(c.tb_acc, c.tb)),
            pct(ratio(c.tb_acc_ok, c.tb)),
            pct(ratio(c.tb_acc_ok, c.tb_acc))
        );
    }

    println!("\n### 3.2 Эффективная скорость и ошибки против t_read");
    let mut head = String::from("| схема \\ t_read → |");
    for &(_, l) in &reads {
        head.push_str(&format!(" {l} |"));
    }
    let sep = "|---|---|---|---|---|";
    println!("\n**симв/с (доставлено и верно)**");
    println!("{head}");
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..reads.len()).map(|ri| num(acc[si][ri].rate(s), 2)).collect();
        println!("{}", report::table_row(s.label(), &cells));
    }
    println!("\n**ошибок среди доставленного**");
    println!("{head}");
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..reads.len()).map(|ri| pct(acc[si][ri].err())).collect();
        println!("{}", report::table_row(s.label(), &cells));
    }
    println!("\n**A/C по-страйпово (без межполосного голосования): симв/с и ошибки**");
    println!("{head}");
    println!("{sep}");
    for (si, &s) in ALL.iter().enumerate() {
        if s != Scheme::A && s != Scheme::C {
            continue;
        }
        let cells: Vec<String> = (0..reads.len())
            .map(|ri| num(acc[si][ri].stripe_rate(s), 2))
            .collect();
        println!("{}", report::table_row(s.label(), &cells));
        let cells: Vec<String> = (0..reads.len())
            .map(|ri| pct(acc[si][ri].stripe_err()))
            .collect();
        println!(
            "{}",
            report::table_row(&format!("{} — ошибок", s.label()), &cells)
        );
    }
}

// ----------------------------------------------------------- 4. phase -------

const N_PHASE: usize = 16;

fn sweep_phase() {
    println!("\n## 4. Дрейф фазы камера/дисплей (H = 6, α = {ALPHA_REF}, t_read = 16 мс)");
    println!(
        "\nПри H = 6 период кадра дисплея = ровно 3 кадра камеры, то есть фаза НЕ уплывает сама — \
         развёртка по фазе на полный период кадра ищет слепую фазу. Показываем среднее и ХУДШЕЕ."
    );
    let t_f = 6.0 * T_REF;
    let mut rates = [[0.0f64; N_PHASE]; 4];
    let mut errs = [[0.0f64; N_PHASE]; 4];
    let mut cleans = [0.0f64; N_PHASE];
    for (si, &s) in ALL.iter().enumerate() {
        for p in 0..N_PHASE {
            let mut cfg = Cfg::reference(6);
            cfg.fixed_phase = Some((p as f64 + 0.5) / N_PHASE as f64 * t_f);
            cfg.trials = 3;
            let (a, _) = run_point(s, &cfg, 5000 + si * 50 + p);
            rates[si][p] = a.rate(s);
            errs[si][p] = a.err();
            if si == 0 {
                cleans[p] = a.clean_dec();
            }
        }
    }
    println!("\n### 4.1 Сводка по фазе");
    println!("| схема | скорость: среднее | скорость: ХУДШАЯ фаза | ошибки: среднее | ошибки: ХУДШАЯ фаза | худшая фаза, доля T_f |");
    println!("|---|---|---|---|---|---|");
    for (si, &s) in ALL.iter().enumerate() {
        let mean_r = rates[si].iter().sum::<f64>() / N_PHASE as f64;
        let mut wi = 0usize;
        for p in 1..N_PHASE {
            if rates[si][p] < rates[si][wi] {
                wi = p;
            }
        }
        let mean_e = errs[si].iter().filter(|e| !e.is_nan()).sum::<f64>() / N_PHASE as f64;
        let max_e = errs[si].iter().cloned().fold(0.0f64, |a, b| if b.is_nan() { a } else { a.max(b) });
        println!(
            "| {} | {} | {} | {} | {} | {:.3} |",
            s.label(),
            num(mean_r, 2),
            num(rates[si][wi], 2),
            pct(mean_e),
            pct(max_e),
            (wi as f64 + 0.5) / N_PHASE as f64
        );
    }
    println!("\n### 4.2 Профиль по фазе (скорость, симв/с)");
    let mut head = String::from("| схема \\ фаза (доля T_f) → |");
    for p in 0..N_PHASE {
        head.push_str(&format!(" {:.2} |", (p as f64 + 0.5) / N_PHASE as f64));
    }
    let mut sep = String::from("|---|");
    for _ in 0..N_PHASE {
        sep.push_str("---|");
    }
    println!("{head}");
    println!("{sep}");
    let cells: Vec<String> = (0..N_PHASE).map(|p| pct(cleans[p])).collect();
    println!("{}", report::table_row("чистых захватов", &cells));
    for (si, &s) in ALL.iter().enumerate() {
        let cells: Vec<String> = (0..N_PHASE).map(|p| num(rates[si][p], 1)).collect();
        println!("{}", report::table_row(s.label(), &cells));
    }
}

// -------------------------------------------------------- 5. реализм -------

fn reality_checks() {
    println!("\n## 5. Проверки реализма — где модель льстит схемам");
    println!(
        "\n(а) «Большой патч» даёт B {ROWS} строчных отсчётов времени на кадр камеры. \
         В настоящем коде каждая КЛЕТКА живёт на своей строке и получает ОДИН отсчёт на кадр камеры, \
         а каждый СТРАЙП (≈6 строк здесь) несёт свои данные. Строки «per-stripe» (rows = 6) и \
         «per-cell» (rows = 1) — тот же прогон с урезанным числом строчных отсчётов на единицу данных \
         (позиция строки покрыта развёрткой по фазе камеры). \
         (б) «σ×5» — тот же прогон при σ = 0.05 на отсчёт."
    );
    println!("\n| режим | H | схема | dims | RMS | ошибок | симв/с |");
    println!("|---|---|---|---|---|---|---|");
    let modes: [(&str, usize, f64); 4] = [
        ("большой патч (база)", ROWS, SIGMA_OBS),
        ("per-stripe (rows = 6)", 6, SIGMA_OBS),
        ("per-cell (rows = 1)", 1, SIGMA_OBS),
        ("большой патч, σ×5", ROWS, 0.05),
    ];
    for (mi, &(lab, rows, sg)) in modes.iter().enumerate() {
        for (hi, &h) in [6u32, 16].iter().enumerate() {
            for (si, &s) in ALL.iter().enumerate() {
                let mut cfg = Cfg::reference(h);
                cfg.rows = rows;
                cfg.sigma = sg;
                let (a, d) = run_point(s, &cfg, 6000 + mi * 200 + hi * 50 + si);
                println!(
                    "| {lab} | {h} | {} | {} | {} | {} | {} |",
                    s.label(),
                    d.dims,
                    report::sig4(a.rms()),
                    pct(a.err()),
                    num(a.rate(s), 2)
                );
            }
        }
    }
}

// ------------------------------------------------------ 6. шум/помехи ------

/// Множители σ относительно базового 0.01 на строчный отсчёт.
const SIG_MULT: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0];
const NOISE_H: [u32; 2] = [6, 16];

fn head_row(label: &str, cols: &[String]) -> (String, String) {
    let mut h = format!("| {label} |");
    let mut s = String::from("|---|");
    for c in cols {
        h.push_str(&format!(" {c} |"));
        s.push_str("---|");
    }
    (h, s)
}

fn sweep_noise() {
    println!("\n## 6. Шум и помехи — per-cell геометрия (rows = 1), честные плановые числа");
    println!(
        "\nВесь раздел считается при rows = 1: одна клетка получает ОДИН временной отсчёт \
         на кадр камеры (позиция клетки по строке эквивалентна сдвигу фазы и покрыта \
         развёрткой по фазе камеры). Именно здесь «большой патч» переставал льстить схеме B."
    );

    // ---- 6.1 σ
    println!("\n### 6.1 Развёртка по σ (множитель к базовому σ = {SIGMA_OBS} на отсчёт)");
    let cols: Vec<String> = SIG_MULT.iter().map(|m| format!("×{m:.0}")).collect();
    let mut data = [[[Acc::default(); SIG_MULT.len()]; ALL5.len()]; NOISE_H.len()];
    for (hi, &h) in NOISE_H.iter().enumerate() {
        for (si, &s) in ALL5.iter().enumerate() {
            for (mi, &mult) in SIG_MULT.iter().enumerate() {
                let mut cfg = Cfg::per_cell(h);
                cfg.sigma = SIGMA_OBS * mult;
                data[hi][si][mi] = run_point(s, &cfg, 7000 + hi * 200 + si * 20 + mi).0;
            }
        }
    }
    for (hi, &h) in NOISE_H.iter().enumerate() {
        let (head, sep) = head_row(&format!("H = {h}, симв/с \\ σ →"), &cols);
        println!("\n{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            let cells: Vec<String> = (0..SIG_MULT.len())
                .map(|mi| num(data[hi][si][mi].rate(s), 2))
                .collect();
            println!("{}", report::table_row(s.label(), &cells));
        }
        let (head, sep) = head_row(&format!("H = {h}, ошибок \\ σ →"), &cols);
        println!("\n{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            let cells: Vec<String> = (0..SIG_MULT.len())
                .map(|mi| pct(data[hi][si][mi].err()))
                .collect();
            println!("{}", report::table_row(s.label(), &cells));
        }
        let (head, sep) = head_row(&format!("H = {h}, ГОДНАЯ скорость \\ σ →"), &cols);
        println!("\n{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            let cells: Vec<String> = (0..SIG_MULT.len())
                .map(|mi| num(data[hi][si][mi].usable(s), 2))
                .collect();
            println!("{}", report::table_row(s.label(), &cells));
        }
    }
    println!("\n**Точка перелома: последний множитель σ, где схема ещё бьёт A по ГОДНОЙ скорости**");
    println!("| H | схема | последний ×σ с выигрышем | там: схема / A | первый ×σ БЕЗ выигрыша | там: схема / A |");
    println!("|---|---|---|---|---|---|");
    for (hi, &h) in NOISE_H.iter().enumerate() {
        for (si, &s) in ALL5.iter().enumerate() {
            if si == 0 {
                continue;
            }
            let win = |mi: usize| data[hi][si][mi].usable(s) > data[hi][0][mi].usable(Scheme::A);
            let last_win = (0..SIG_MULT.len()).filter(|&mi| win(mi)).next_back();
            let first_lose = (0..SIG_MULT.len()).find(|&mi| !win(mi));
            let f = |mi: Option<usize>| match mi {
                Some(mi) => (
                    format!("×{:.0}", SIG_MULT[mi]),
                    format!(
                        "{} / {}",
                        num(data[hi][si][mi].usable(s), 2),
                        num(data[hi][0][mi].usable(Scheme::A), 2)
                    ),
                ),
                None => ("—".into(), "—".into()),
            };
            let (a1, a2) = f(last_win);
            let (b1, b2) = f(first_lose);
            println!("| {h} | {} | {a1} | {a2} | {b1} | {b2} |", s.label());
        }
    }
    println!(
        "\nВажно при чтении: при ×5 и выше НИ ОДНА схема не проходит бюджет FEC — там ломается \
         не временная схема, а САМ АЛФАВИТ (PAM-{M_PAM}: полурасстояние решения {:.4} по шкале \
         драйва против σ = {:.3} на отсчёт). Реальная система на таком шуме ушла бы на PAM-4/PAM-2, \
         поэтому колонки ×5…×50 читать как «алфавит не тот», а не как «B проиграла A».",
        SWING * PAM_HALF,
        SIGMA_OBS * 5.0
    );

    // ---- 6.2 ШИМ подсветки
    println!("\n### 6.2 ШИМ подсветки (мультипликативная пульсация света)");
    println!(
        "\nМодель: свет × (1 + m·sin(2πf t + φ)), φ свободно бежит относительно дисплея. \
         Боксkar выдержки давит пульсацию как sinc(f·t_exp) — при t_exp = 16.7 мс частоты, \
         кратные 60 Гц, зануляются ТОЧНО, поэтому 240 Гц (кратная) и 1000/4000 Гц (некратные) \
         разнесены отдельно. Остаток не усредняется между строками/кадрами (разная фаза) и \
         выглядит как структурный мультипликативный шум усиления. σ базовый."
    );
    let pwm_pts: [(f64, f64, &str); 6] = [
        (240.0, 0.3, "240 Гц, m=0.3"),
        (240.0, 1.0, "240 Гц, m=1.0"),
        (1000.0, 0.3, "1 кГц, m=0.3"),
        (1000.0, 1.0, "1 кГц, m=1.0"),
        (4000.0, 0.3, "4 кГц, m=0.3"),
        (4000.0, 1.0, "4 кГц, m=1.0"),
    ];
    println!("\n| H | помеха | схема | dims | RMS | ошибок | симв/с | Δ симв/с к «без ШИМ» |");
    println!("|---|---|---|---|---|---|---|---|");
    for (hi, &h) in NOISE_H.iter().enumerate() {
        for (si, &s) in ALL5.iter().enumerate() {
            let base = data[hi][si][0].rate(s);
            for (pi, &(f, m, lab)) in pwm_pts.iter().enumerate() {
                let mut cfg = Cfg::per_cell(h);
                cfg.pwm = Some((f, m));
                let (acc, des) = run_point(s, &cfg, 7400 + hi * 300 + si * 30 + pi);
                println!(
                    "| {h} | {lab} | {} | {} | {} | {} | {} | {:+.2} |",
                    s.label(),
                    des.dims,
                    report::sig4(acc.rms()),
                    pct(acc.err()),
                    num(acc.rate(s), 2),
                    acc.rate(s) - base
                );
            }
        }
    }

    // ---- 6.3 PRNU
    println!("\n### 6.3 PRNU — фиксированный разброс усиления клетки (НЕ калиброван)");
    println!(
        "\nПостоянный по времени множитель g ~ N(1, prnu) на клетку данных (и отдельный — \
         на клетку сигнатуры). Не усредняется ни по кадрам камеры, ни БИХ-ом. \
         Худший случай: калибровка усиления клетки НЕ выполнена (в реальном профиле есть \
         white/black level, который её как раз и снимает — см. чтение)."
    );
    let prnus: [f64; 3] = [0.01, 0.03, 0.05];
    let pcols: Vec<String> = prnus.iter().map(|p| format!("{:.0}%", p * 100.0)).collect();
    for (hi, &h) in NOISE_H.iter().enumerate() {
        let mut r_rate = vec![Vec::new(); ALL5.len()];
        let mut r_err = vec![Vec::new(); ALL5.len()];
        for (si, &s) in ALL5.iter().enumerate() {
            for (pi, &p) in prnus.iter().enumerate() {
                let mut cfg = Cfg::per_cell(h);
                cfg.prnu = p;
                let (acc, _) = run_point(s, &cfg, 7800 + hi * 200 + si * 20 + pi);
                r_rate[si].push(num(acc.rate(s), 2));
                r_err[si].push(pct(acc.err()));
            }
        }
        let (head, sep) = head_row(&format!("H = {h}, симв/с \\ PRNU →"), &pcols);
        println!("\n{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            println!("{}", report::table_row(s.label(), &r_rate[si]));
        }
        let (head, sep) = head_row(&format!("H = {h}, ошибок \\ PRNU →"), &pcols);
        println!("\n{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            println!("{}", report::table_row(s.label(), &r_err[si]));
        }
    }
}

// -------------------------------------------------- 7. drive-домен БИХ ------

fn sweep_drive() {
    println!("\n## 7. Где стоит БИХ ISP: линейный свет ПРОТИВ drive-домена (после тоновой кривой)");
    println!(
        "\n`exp.rs` в этом же репозитории смешивает кадры в DRIVE-домене (u8), tprobe до сих пор — \
         в линейном свете. Обе версии не могут быть верны одновременно, и это ровно то, от чего \
         зависит, ТОЧНА ли известная поправка усиления у схемы B. Здесь тот же прогон с БИХ, \
         применённым ПОСЛЕ тоновой кривой γ = {GAMMA}: \
         сенсорный шум → x^(1/γ) → БИХ → 8 бит → приёмник линеаризует обратно (^γ). \
         Тогда смешивание — степенное среднее, а не арифметическое, и линейная модель канала \
         в приёмнике становится ПРИБЛИЖЁННОЙ. Физическую истину отсюда не определить — \
         поэтому обе ветви рядом."
    );

    // ---- 7.1 развёртка по H
    for (gi, &(rows, glab)) in [(1usize, "per-cell (rows = 1)"), (ROWS, "большой патч")]
        .iter()
        .enumerate()
    {
        println!("\n### 7.{} Развёртка по H — {glab}", gi + 1);
        let cols: Vec<String> = H_LIST.iter().map(|h| h.to_string()).collect();
        let mut lin = [[Acc::default(); H_LIST.len()]; ALL5.len()];
        let mut drv = [[Acc::default(); H_LIST.len()]; ALL5.len()];
        for (si, &s) in ALL5.iter().enumerate() {
            for (hi, &h) in H_LIST.iter().enumerate() {
                let mut cfg = Cfg::reference(h);
                cfg.rows = rows;
                lin[si][hi] = run_point(s, &cfg, 8000 + gi * 500 + si * 50 + hi).0;
                cfg.drive_domain = true;
                drv[si][hi] = run_point(s, &cfg, 8200 + gi * 500 + si * 50 + hi).0;
            }
        }
        let (head, sep) = head_row("схема (арм) \\ H →", &cols);
        println!("\n**ГОДНАЯ скорость, симв/с (обнулено при ошибках > {:.0}%)**", FEC_ERR_BUDGET * 100.0);
        println!("{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            let c1: Vec<String> = (0..H_LIST.len()).map(|hi| num(lin[si][hi].usable(s), 2)).collect();
            let c2: Vec<String> = (0..H_LIST.len()).map(|hi| num(drv[si][hi].usable(s), 2)).collect();
            println!("{}", report::table_row(&format!("{} — линейный свет", s.label()), &c1));
            println!("{}", report::table_row(&format!("{} — DRIVE", s.label()), &c2));
        }
        println!("\n**ошибок среди доставленного**");
        println!("{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            let c1: Vec<String> = (0..H_LIST.len()).map(|hi| pct(lin[si][hi].err())).collect();
            let c2: Vec<String> = (0..H_LIST.len()).map(|hi| pct(drv[si][hi].err())).collect();
            println!("{}", report::table_row(&format!("{} — линейный свет", s.label()), &c1));
            println!("{}", report::table_row(&format!("{} — DRIVE", s.label()), &c2));
        }
    }

    // ---- 7.3 развёртка по α
    println!("\n### 7.3 Развёртка по α в обеих ветвях — per-cell");
    let cols: Vec<String> = ALPHAS.iter().map(|a| a.to_string()).collect();
    for &h in &NOISE_H {
        let mut lin = [[Acc::default(); ALPHAS.len()]; ALL5.len()];
        let mut drv = [[Acc::default(); ALPHAS.len()]; ALL5.len()];
        for (si, &s) in ALL5.iter().enumerate() {
            for (ai, &al) in ALPHAS.iter().enumerate() {
                let mut cfg = Cfg::per_cell(h);
                cfg.alpha = al;
                lin[si][ai] = run_point(s, &cfg, 8600 + h as usize * 100 + si * 10 + ai).0;
                cfg.drive_domain = true;
                drv[si][ai] = run_point(s, &cfg, 8900 + h as usize * 100 + si * 10 + ai).0;
            }
        }
        let (head, sep) = head_row(&format!("H = {h}: схема (арм) \\ α →"), &cols);
        println!("\n**ГОДНАЯ скорость, симв/с**");
        println!("{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            let c1: Vec<String> = (0..ALPHAS.len()).map(|ai| num(lin[si][ai].usable(s), 2)).collect();
            let c2: Vec<String> = (0..ALPHAS.len()).map(|ai| num(drv[si][ai].usable(s), 2)).collect();
            println!("{}", report::table_row(&format!("{} — лин.", s.label()), &c1));
            println!("{}", report::table_row(&format!("{} — DRIVE", s.label()), &c2));
        }
        let (head, sep) = head_row(&format!("H = {h}: ошибок \\ α →"), &cols);
        println!("\n{head}");
        println!("{sep}");
        for (si, &s) in ALL5.iter().enumerate() {
            let c1: Vec<String> = (0..ALPHAS.len()).map(|ai| pct(lin[si][ai].err())).collect();
            let c2: Vec<String> = (0..ALPHAS.len()).map(|ai| pct(drv[si][ai].err())).collect();
            println!("{}", report::table_row(&format!("{} — лин.", s.label()), &c1));
            println!("{}", report::table_row(&format!("{} — DRIVE", s.label()), &c2));
        }
    }
}

// ---------------------------------------------- 8. рабочая точка B2+sig ----

fn design_point() {
    println!("\n## 8. Рабочая точка: B2 + сигнатурная полоса, H = 6, per-cell");
    println!(
        "\nЦена сигнатуры: {:.1}% площади кадра (множитель скорости {C_RATE_SCALE}), \
         L = {} уровня в цикле по кадрам дисплея, порог отбраковки τ = {TAU:.4} \
         (= β_tol·Δg_min, то есть режем ровно ту примесь, что способна испортить символ). \
         Сигнатура проверяется ПОБАНДОВО и стирает отсчёты ДО МНК. \
         «Стирания» — доля кадров дисплея, которые приёмник СОЗНАТЕЛЬНО не выдал \
         (для FEC это erasure, вдвое дешевле ошибки).",
        100.0 / SIG_AREA_DIV,
        SIG_LEVELS.len()
    );
    println!(
        "\nТретий кандидат на стирание, не стоящий НИ площади, НИ отсчётов: **невязка МНК**. \
         Приёмник знает свой шумовой пол σ из профиля калибровки; если остаток подгонки \
         кадра > {RES_K}σ, кадр объявляется стёртым. Столбцы «+невязка» — та же схема B \
         с этим признаком."
    );
    println!(
        "\n| арм | схема | dims | симв/с | ошибок | стираний | +невязка: симв/с | +невязка: ошибок | +невязка: стираний |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for (di, &(drive, dlab)) in [(false, "линейный свет"), (true, "DRIVE-домен")]
        .iter()
        .enumerate()
    {
        for (si, &s) in ALL5.iter().enumerate() {
            let mut cfg = Cfg::per_cell(6);
            cfg.drive_domain = drive;
            let (acc, des) = run_point(s, &cfg, 9000 + di * 100 + si * 10);
            // у жёстких схем нет подгонки базиса, поэтому и невязки нет
            let (r1, r2, r3) = if s.is_hard() {
                ("н/п".to_string(), "н/п".to_string(), "н/п".to_string())
            } else {
                (
                    num(acc.res_rate(s), 2),
                    pct(acc.res_err()),
                    pct(acc.res_erasure()),
                )
            };
            println!(
                "| {dlab} | {} | {} | {} | {} | {} | {r1} | {r2} | {r3} |",
                s.label(),
                des.dims,
                num(acc.rate(s), 2),
                pct(acc.err()),
                pct(acc.erasure())
            );
        }
    }
    println!("\n### 8.0 То же для B2 по H (per-cell, линейный свет) — где оптимум");
    println!("| H | dims | симв/с | ошибок | +невязка: симв/с | +невязка: ошибок | +невязка: стираний | A для сравнения |");
    println!("|---|---|---|---|---|---|---|---|");
    for &h in &[4u32, 6, 8, 12, 16] {
        let cfg = Cfg::per_cell(h);
        let (b2, des) = run_point(Scheme::B2, &cfg, 9500 + h as usize);
        let (aa, _) = run_point(Scheme::A, &cfg, 9600 + h as usize);
        println!(
            "| {h} | {} | {} | {} | {} | {} | {} | {} @ {} |",
            des.dims,
            num(b2.rate(Scheme::B2), 2),
            pct(b2.err()),
            num(b2.res_rate(Scheme::B2), 2),
            pct(b2.res_err()),
            pct(b2.res_erasure()),
            num(aa.rate(Scheme::A), 2),
            pct(aa.err())
        );
    }
    println!("\n### 8.1 Та же точка под нагрузкой (линейный свет, per-cell, H = 6)");
    println!("| помеха | схема | симв/с | ошибок | стираний |");
    println!("|---|---|---|---|---|");
    let stress: [(&str, f64, f64, Option<(f64, f64)>); 5] = [
        ("база", SIGMA_OBS, 0.0, None),
        ("σ×10", SIGMA_OBS * 10.0, 0.0, None),
        ("σ×20", SIGMA_OBS * 20.0, 0.0, None),
        ("PRNU 3%", SIGMA_OBS, 0.03, None),
        ("ШИМ 1 кГц m=1", SIGMA_OBS, 0.0, Some((1000.0, 1.0))),
    ];
    for (ti, &(lab, sg, pr, pw)) in stress.iter().enumerate() {
        for (si, &s) in [Scheme::A, Scheme::C, Scheme::B2, Scheme::B2C].iter().enumerate() {
            let mut cfg = Cfg::per_cell(6);
            cfg.sigma = sg;
            cfg.prnu = pr;
            cfg.pwm = pw;
            let (acc, _) = run_point(s, &cfg, 9300 + ti * 40 + si * 5);
            println!(
                "| {lab} | {} | {} | {} | {} |",
                s.label(),
                num(acc.rate(s), 2),
                pct(acc.err()),
                pct(acc.erasure())
            );
        }
    }
}

// --------------------------------------------------------------- тесты ------

#[cfg(test)]
mod tests {
    use super::*;

    /// Идеальная мгновенная выборка: все схемы восстанавливают точно.
    #[test]
    fn ideal_sampling_is_exact() {
        for &s in &ALL {
            let mut cfg = Cfg::reference(16);
            cfg.t_exp = 1e-9;
            cfg.t_read = 0.0;
            cfg.alpha = 0.0;
            cfg.sigma = 0.0;
            cfg.quant = false;
            cfg.trials = 2;
            let (a, _) = run_point(s, &cfg, 900);
            assert!(a.deliv > 0, "{s:?}: ничего не доставлено");
            assert_eq!(a.correct, a.deliv, "{s:?}: ошибки при идеальной выборке");
            assert!(a.rms() < 1e-6, "{s:?}: RMS {} слишком велик", a.rms());
        }
    }

    /// Боксkar-среднее постоянного сигнала = сам сигнал (DC-усиление 1).
    #[test]
    fn boxcar_dc_gain_is_unity() {
        let des = Design { k: 0, dims: 1, amp: SWING };
        let sig = Signal {
            scheme: Scheme::A,
            t_f: 0.1,
            n_frames: 4,
            des,
            data: vec![1.0, 1.0, 1.0, 1.0],
        };
        let v = sig.mean(0.01, 0.01 + 1.0 / 60.0);
        assert!((v - (MID + SWING)).abs() < 1e-12, "v = {v}");
    }

    /// Гладкий базис B2 непрерывен на границах кадров (значение = MID).
    #[test]
    fn b2_basis_is_continuous_across_frames() {
        let des = Design { k: 3, dims: 3, amp: SWING / 3.0 };
        let sig = Signal {
            scheme: Scheme::B2,
            t_f: 0.1,
            n_frames: 4,
            des,
            data: vec![1.0, -1.0, 0.5, -0.5, 1.0, 1.0, 0.0, -1.0, 0.25, 1.0, -1.0, 0.75],
        };
        for m in 1..4 {
            let t = m as f64 * 0.1;
            let before = sig.value(t - 1e-7);
            let after = sig.value(t + 1e-7);
            assert!((before - MID).abs() < 1e-4, "слева {before}");
            assert!((after - MID).abs() < 1e-4, "справа {after}");
        }
    }

    /// АЧХ БИХ: |H(0)| = 1, |H(π)| = (1−α)/(1+α).
    #[test]
    fn iir_response_matches_closed_form() {
        let a = 0.5;
        assert!((h_iir(0.0, a).abs() - 1.0).abs() < 1e-12);
        let w_pi = PI / T_CAM;
        assert!((h_iir(w_pi, a).abs() - (1.0 - a) / (1.0 + a)).abs() < 1e-9);
    }

    /// Ноль sinc выдержки: гармоника ровно на 1/t_exp давится в ноль.
    #[test]
    fn boxcar_null_at_inverse_exposure() {
        let t_exp = 1.0 / 60.0;
        let w = 2.0 * PI * 60.0;
        assert!(h_box(w, t_exp).abs() < 1e-12);
        assert!((h_box(0.0, t_exp).abs() - 1.0).abs() < 1e-12);
    }

    /// Чистота: окно целиком внутри кадра ⇒ 1.0; ровно на переходе ⇒ ~0.5.
    #[test]
    fn purity_detects_straddle() {
        let t_f = 0.1;
        let t_exp = 1.0 / 60.0;
        let (_, p_in) = purity(0.05, t_exp, 0.0, t_f, 0);
        assert!((p_in - 1.0).abs() < 1e-12, "внутри кадра: {p_in}");
        let (_, p_edge) = purity(0.1 - t_exp * 0.5, t_exp, 0.0, t_f, 0);
        assert!((p_edge - 0.5).abs() < 1e-9, "на переходе: {p_edge}");
    }

    /// Порог отбраковки C лежит между шумом сигнатуры и вредной примесью.
    #[test]
    fn c_threshold_is_between_noise_and_damage() {
        let sigma_band = SIGMA_OBS * SIG_AREA_DIV.sqrt() / ((ROWS / BANDS) as f64).sqrt();
        assert!(TAU > 2.5 * sigma_band, "порог {TAU} тонет в шуме {sigma_band}");
        assert!(TAU < 0.5 * SIG_DELTA_MIN, "порог {TAU} шире полушага уровней");
    }

    /// МНК решает точно определённую систему.
    #[test]
    fn solver_recovers_known_vector() {
        let d = 3;
        let phi = [1.0, 0.5, 0.2, 0.0, 1.0, -0.3, 0.4, 0.1, 1.0, 1.0, 1.0, 1.0];
        let x_true = [0.7, -0.4, 0.25];
        let n = 4;
        let mut g = vec![0.0; d * d];
        let mut rhs = vec![0.0; d];
        for s in 0..n {
            let row = &phi[s * d..s * d + d];
            let y: f64 = (0..d).map(|j| row[j] * x_true[j]).sum();
            for i in 0..d {
                rhs[i] += row[i] * y;
                for j in 0..d {
                    g[i * d + j] += row[i] * row[j];
                }
            }
        }
        let x = solve(&mut g, &mut rhs, d).expect("система решаема");
        for j in 0..d {
            assert!((x[j] - x_true[j]).abs() < 1e-9, "x[{j}] = {}", x[j]);
        }
    }
}
