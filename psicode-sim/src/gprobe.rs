//! GLOBAL-HG (`gprobe`): весь символ как ОДНО непрерывное комплексное поле,
//! разложенное по ГЛОБАЛЬНОМУ 2-D базису Эрмита–Гаусса над всей апертурой.
//! Низкие моды (порядки 0–2: ψ00, ψ10, ψ01, ψ11, ψ20, ψ02) зарезервированы под
//! КАЛИБРОВОЧНЫЙ КАНАЛ с априори известными переданными значениями; данные едут
//! на старших модах.
//!
//! Проверяемая идея: низкие моды бесплатно (без площади) отдают
//!   * ψ00 — общий gain (экспозиция + поканальный баланс белого),
//!   * ψ10/ψ01 — ЛИНЕЙНЫЙ НАКЛОН поля освещённости (то, чего patch-эталон дать
//!     не может: точка не определяет градиент),
//!   * ψ11/ψ20/ψ02 — кривизну (виньетирование ≈ квадратично),
//!   * профиль аттенюации по порядку m+n — оценку блюра σ (ψ_mn — собственные
//!     функции преобразования Фурье, дефокус давит моду монотонно по m+n).
//!
//! Отличие от [`crate::modeb`]: там базис живёт на ОДНОМ изолированном блоке
//! 64×64 px, где поле освещённости почти константа и потому не проверяется.
//! Здесь апертура — весь символ, и поле реально меняется поперёк неё.
//!
//! Три арма на ОДНОЙ площади, ОДНОМ диапазоне драйва и ОДНОМ канале:
//!   1. PER-CELL  — независимое значение на клетку (текущая схема), в вариантах
//!      наивной нормировки, локального box-mean, глобального био-квадратичного
//!      фита с решающей обратной связью, и отображения ПОСТОЯННОЙ ЯРКОСТИ
//!      (`ChromaMode::ConstLuma1`, §5.1-CL) — текущий лучший per-cell ответ.
//!   2. PILOTS    — per-cell + разрежённая решётка ЭТАЛОННЫХ клеток, приёмник
//!      снимает affine/био-квадратичное поле. Площадь пилотов честно вычитается
//!      из полезной (так делают HCCB и JAB Code / ISO 23634).
//!   3. GLOBAL-HG — идея под тестом.
//!
//! Канал — ИЗМЕРЕННЫЙ на реальной связке (Samsung Galaxy A22 + 1080p дисплей):
//! шум 6.15 кодов на ПИКСЕЛЬ (усреднение по клетке само даёт замеренные
//! 1.79 кода на клетку), мультипликативное поле 0.62→0.86 по кадру на линейной
//! радиометрии, поканальная гамма ISP 3.8/4.7/5.7, кросстолк 6%/8%.

use crate::image::Image;
use crate::rng::{seed_for, Rng};
use crate::{pipeline, report};
use psicode_core::symbol::{const_luma_map, ConstLumaMap, A_C_FRACTION_CHROMA, A_L_FRACTION_CHROMA};
use psicode_core::{CalibProfile, ChromaMode};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Геометрия символа и базиса
// ---------------------------------------------------------------------------

/// Клеток на сторону символа.
const CELLS: usize = 24;
/// Пикселей КАМЕРЫ на клетку (замер: ~13; берём 12 — делится на бины приёмника).
const PX_CELL: usize = 12;
/// Сторона апертуры в camera-px.
const APER: usize = CELLS * PX_CELL; // 288
/// Поле средне-серого вокруг символа: ≥ ceil(3·σ_max) при σ_max = 8.
const PAD: usize = 24;
/// Старшая 1-D мода: MMAX+1 = CELLS мод на ось ⇒ ровно CELLS² = 576 мод,
/// т.е. РОВНО столько же степеней свободы, сколько клеток у per-cell.
const MMAX: usize = CELLS - 1; // 23
/// Мод в базисе (тензорное произведение {0..MMAX}²).
const NMODES: usize = (MMAX + 1) * (MMAX + 1); // 576
/// Сторона бина приёмника при проекции HG (px). Приёмник усредняет код по бину,
/// затем линеаризует — та же предобработка, что у per-cell (честный паритет).
const RX_BIN: usize = 4;
/// Отсчётов приёмника на ось.
const RXN: usize = APER / RX_BIN; // 72

/// Калибровочные моды (порядки 0..2) — данных НЕ несут.
const CAL_MODES: [(usize, usize); 6] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)];
/// Во сколько раз амплитуда калибровочной моды выше data-моды.
const CAL_BOOST: f64 = 3.0;

// ---------------------------------------------------------------------------
// Измеренная физика канала
// ---------------------------------------------------------------------------

/// Шум одиночного снимка на ПИКСЕЛЬ, в кодах из 255 (замер).
const PIX_NOISE_CODES: f64 = 6.15;
/// Поле освещённости: множитель на линейную радиометрию, от .. до .. по кадру.
const FIELD_LO: f64 = 0.62;
const FIELD_HI: f64 = 0.86;
/// Поканальный (хроматический) дифференциал поля, размах R-к-B по кадру.
const CHROMA_DIFF: f64 = 0.06;
/// Поканальная гамма ISP (замер; далеко от sRGB): code = 255·(радиометрия)^(1/γ).
const GAMMA_ISP: [f64; 3] = [3.8, 4.7, 5.7];
/// Кадров в секунду (60 Гц / hold 6 периодов, §6.3) — для бит/с.
const FPS: f64 = 10.0;

/// Развёртка блюра σ (camera-px).
const SIGMAS: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 8.0];
/// Попыток Monte Carlo на точку.
const TRIALS: usize = 16;
/// Целевая доля клиппинга драйва для HG (policy = clip).
const CLIP_TARGET: f64 = 0.003;
/// Запас проектирования у политики «масштаб». Усиление — ПРОТОКОЛЬНАЯ константа,
/// снятая на конечном ансамбле розыгрышей, поэтому без запаса редкий «неудачный»
/// символ всё же выходит за диапазон. 5% запаса стоят 0.45 дБ и делают клип
/// действительно нулевым — иначе «политика без клипа» была бы обещанием, а не фактом.
const SCALE_MARGIN: f64 = 0.95;
/// Порог «бит доставлен»: предсказанный BER < 1e-2 ⇒ SNR > Q⁻¹(1e-2)² .
const SNR_MIN: f64 = 5.411_9; // (2.326348)²
/// Итераций контура «линеаризация → проекция → фит поля → перепроекция».
const MAXIT: usize = 5;

// ---------------------------------------------------------------------------
// Комплексное число (без внешних зависимостей)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default, PartialEq)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    #[inline]
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

// ---------------------------------------------------------------------------
// Малая линейная алгебра: Холецкий для SPD-систем
// ---------------------------------------------------------------------------

/// Разложение Холецкого A = L·Lᵀ на месте (нижний треугольник). `false` — не SPD.
fn chol(a: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if s <= 1e-300 {
                    return false;
                }
                a[i * n + i] = s.sqrt();
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
        for j in (i + 1)..n {
            a[i * n + j] = 0.0;
        }
    }
    true
}

/// Решение L·Lᵀ·x = b по готовому Холецкому (на месте в `b`).
fn chol_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Обращение SPD-матрицы n×n через Холецкого. `None` — вырождена.
fn spd_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = a.to_vec();
    if !chol(&mut l, n) {
        return None;
    }
    let mut inv = vec![0.0f64; n * n];
    let mut e = vec![0.0f64; n];
    for c in 0..n {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[c] = 1.0;
        chol_solve(&l, n, &mut e);
        for r in 0..n {
            inv[r * n + c] = e[r];
        }
    }
    Some(inv)
}

/// Решение общей SPD-системы A·x = b (A портится). `None` — вырождена.
fn spd_solve(a: &mut [f64], n: usize, b: &mut [f64]) -> bool {
    if !chol(a, n) {
        return false;
    }
    chol_solve(a, n, b);
    true
}

/// ∞-норма квадратной матрицы (макс сумма модулей по строке).
fn norm_inf(a: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Q-функция (хвост нормали) — предсказанный BER по SNR
// ---------------------------------------------------------------------------

/// erfc(x) по A&S 7.1.26 (абс. точность ~1.5e-7 — достаточно для порога 1e-2).
fn erfc_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-ax * ax).exp();
    1.0 - sign * erf
}

/// Q(x) = P(N(0,1) > x).
fn q_func(x: f64) -> f64 {
    0.5 * erfc_approx(x / std::f64::consts::SQRT_2)
}

/// Средний квадрат нормированного уровня L-PAM {(2i−(L−1))/(L−1)}.
fn pam_m2(lv: usize) -> f64 {
    let d = (lv - 1) as f64;
    (0..lv)
        .map(|i| {
            let x = (2 * i) as f64 - d;
            (x / d) * (x / d)
        })
        .sum::<f64>()
        / lv as f64
}

/// Предсказанный BER одной оси L-PAM при отношении сигнал/остаток `snr`
/// (Грей-код: одна ошибка уровня ≈ один ошибочный бит).
///
/// Вывод: snr = α²a²·m2/σ², шаг созвездия d = 2αa/(L−1) ⇒ d/2σ =
/// √(snr/m2)/(L−1); SER = 2(1−1/L)·Q(d/2σ); BER = SER/log2 L.
/// При L = 2 сводится к Q(√snr) — обычный QPSK по оси.
fn ber_from_snr_lv(snr: f64, lv: usize) -> f64 {
    if !snr.is_finite() {
        return 0.0;
    }
    if snr <= 0.0 {
        return 0.5;
    }
    let arg = (snr / pam_m2(lv)).sqrt() / (lv - 1) as f64;
    let ser = 2.0 * (1.0 - 1.0 / lv as f64) * q_func(arg);
    (ser / (lv as f64).log2()).clamp(1e-12, 0.5)
}

/// Уровень L-PAM по индексу, нормированный на [−1, 1].
#[inline]
fn pam_level(i: usize, lv: usize) -> f64 {
    ((2 * i) as f64 - (lv - 1) as f64) / (lv - 1) as f64
}

/// Ближайший индекс уровня L-PAM к нормированному значению.
#[inline]
fn pam_slice(x: f64, lv: usize) -> usize {
    let d = (lv - 1) as f64;
    (((x * d + d) * 0.5).round()).clamp(0.0, d) as usize
}

/// Ошибочных бит между двумя индексами уровней при Грей-кодировании.
#[inline]
fn pam_bit_errors(a: usize, b: usize) -> usize {
    ((a ^ (a >> 1)) ^ (b ^ (b >> 1))).count_ones() as usize
}

/// dB с защитой от нуля/бесконечности.
fn to_db(x: f64) -> f64 {
    if x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * x.log10()
    }
}

// ---------------------------------------------------------------------------
// 1-D базис Эрмита–Гаусса на дискретной сетке
// ---------------------------------------------------------------------------

/// Устойчивая нормированная рекуррентность ψ̃_k(u) = H_k(u)e^{-u²/2}/√(2^k k!√π):
/// ψ̃_0 = π^{-1/4}e^{-u²/2}, ψ̃_1 = √2·u·ψ̃_0, ψ̃_{k+1} = √(2/(k+1))·u·ψ̃_k −
/// √(k/(k+1))·ψ̃_{k-1}. Значения O(1) при любом порядке — в отличие от прямого
/// H_k, который при m=23 доходит до ~1e26 и теряет точность.
fn hg_column(u: f64, mmax: usize, out: &mut [f64]) {
    let g = std::f64::consts::PI.powf(-0.25) * (-0.5 * u * u).exp();
    out[0] = g;
    if mmax == 0 {
        return;
    }
    out[1] = std::f64::consts::SQRT_2 * u * g;
    for k in 1..mmax {
        let kf = k as f64;
        out[k + 1] = (2.0 / (kf + 1.0)).sqrt() * u * out[k] - (kf / (kf + 1.0)).sqrt() * out[k - 1];
    }
}

/// 1-D базис на сетке отсчётов: колонки нормированы к единичной ДИСКРЕТНОЙ
/// L2-норме, поэтому 2-D базис ψ_mn = h_m ⊗ h_n тоже единичной нормы, а шум с
/// дисперсией σ² проецируется на любую моду с той же дисперсией σ² — это делает
/// «амплитуду коэффициента» прямо сопоставимой между армами.
struct Hg1 {
    /// Число мод (0..m-1).
    m: usize,
    /// Число отсчётов.
    n: usize,
    /// n×m, колонка k — мода k.
    h: Vec<f64>,
    /// m×m, (HᵀH)⁻¹ — множитель МНК-проекции (не корреляции!).
    ginv: Vec<f64>,
    /// Максимум |Грам вне-диагонали| (диагональ ровно 1).
    gram_max_off: f64,
    /// Оценка числа обусловленности Грама, κ_∞ = ‖G‖_∞·‖G⁻¹‖_∞.
    cond: f64,
    /// Только для бинированного базиса: L2-норма box-усреднённой моды k ДО
    /// перенормировки. Коэффициенты связаны как c_rx[k][l] = c_tx[k][l]·N_k·N_l,
    /// то есть N_k — прямая мера того, насколько бин-фильтр приёмника давит
    /// моду k. Для небинированного базиса — единицы.
    bin_norm: Vec<f64>,
}

impl Hg1 {
    /// Базис из `m` мод шириной `w` на `n` отсчётах с шагом `step` px,
    /// центр апертуры — `center` px (координаты отсчёта: (i+0.5)·step).
    fn new(m: usize, n: usize, step: f64, center: f64, w: f64) -> Self {
        let mut h = vec![0.0f64; n * m];
        let mut col = vec![0.0f64; m];
        for i in 0..n {
            let x = (i as f64 + 0.5) * step;
            hg_column((x - center) / w, m - 1, &mut col);
            for k in 0..m {
                h[i * m + k] = col[k];
            }
        }
        // нормировка колонок к единичной L2
        for k in 0..m {
            let mut ss = 0.0;
            for i in 0..n {
                ss += h[i * m + k] * h[i * m + k];
            }
            let inv = 1.0 / ss.sqrt();
            for i in 0..n {
                h[i * m + k] *= inv;
            }
        }
        // Грам G = HᵀH
        let mut g = vec![0.0f64; m * m];
        for a in 0..m {
            for b in a..m {
                let mut s = 0.0;
                for i in 0..n {
                    s += h[i * m + a] * h[i * m + b];
                }
                g[a * m + b] = s;
                g[b * m + a] = s;
            }
        }
        let mut gram_max_off = 0.0f64;
        for a in 0..m {
            for b in 0..m {
                if a != b {
                    gram_max_off = gram_max_off.max(g[a * m + b].abs());
                }
            }
        }
        let ginv = spd_inverse(&g, m).expect("Грам 1-D базиса вырожден");
        let cond = norm_inf(&g, m) * norm_inf(&ginv, m);
        Hg1 {
            m,
            n,
            h,
            ginv,
            gram_max_off,
            cond,
            bin_norm: vec![1.0; m],
        }
    }

    /// Базис приёмника, СОГЛАСОВАННЫЙ с бинированием: колонка k — box-среднее
    /// колонки передатчика по бину. Приёмник знает, какое усреднение он сам
    /// применил, поэтому проецировать обязан на УСРЕДНЁННЫЕ функции: иначе
    /// бин-фильтр вносит систематическую немодельную аттенюацию старших мод и
    /// чистый канал перестаёт восстанавливаться точно.
    fn from_binned(tx: &Hg1, bin: usize) -> Self {
        let m = tx.m;
        let n = tx.n / bin;
        let inv = 1.0 / bin as f64;
        let mut h = vec![0.0f64; n * m];
        for i in 0..n {
            for d in 0..bin {
                let src = &tx.h[(i * bin + d) * m..(i * bin + d) * m + m];
                for k in 0..m {
                    h[i * m + k] += src[k] * inv;
                }
            }
        }
        let mut bin_norm = vec![0.0f64; m];
        for k in 0..m {
            let mut ss = 0.0;
            for i in 0..n {
                ss += h[i * m + k] * h[i * m + k];
            }
            bin_norm[k] = ss.sqrt();
            let s = 1.0 / bin_norm[k];
            for i in 0..n {
                h[i * m + k] *= s;
            }
        }
        let mut g = vec![0.0f64; m * m];
        for a in 0..m {
            for b in a..m {
                let mut s = 0.0;
                for i in 0..n {
                    s += h[i * m + a] * h[i * m + b];
                }
                g[a * m + b] = s;
                g[b * m + a] = s;
            }
        }
        let mut gram_max_off = 0.0f64;
        for a in 0..m {
            for b in 0..m {
                if a != b {
                    gram_max_off = gram_max_off.max(g[a * m + b].abs());
                }
            }
        }
        let ginv = spd_inverse(&g, m).expect("Грам бинированного базиса вырожден");
        let cond = norm_inf(&g, m) * norm_inf(&ginv, m);
        Hg1 { m, n, h, ginv, gram_max_off, cond, bin_norm }
    }

    /// Индекс моды (mx, my) в линейном списке.
    #[inline]
    fn idx(&self, mx: usize, my: usize) -> usize {
        mx * self.m + my
    }

    /// МНК-проекция поля f (n×n, row-major по y) на 2-D базис:
    /// C = G⁻¹·(Hᵀ F H)·G⁻¹ (точная МНК по РЕАЛЬНОЙ сетке отсчётов, не
    /// корреляция — усечённый гауссиан под наивной корреляцией не ортогонален).
    /// Разделимость даёт Kronecker-структуру Грама, поэтому 2-D задача решается
    /// двумя 1-D умножениями и никаких матриц 576×576 не строится.
    fn project(&self, f: &[f64]) -> Vec<f64> {
        let (m, n) = (self.m, self.n);
        // T[k][j] = Σ_i H[i][k]·F[i][j]   (по x)
        let mut t = vec![0.0f64; m * n];
        for i in 0..n {
            let hrow = &self.h[i * m..i * m + m];
            let frow = &f[i * n..i * n + n];
            for k in 0..m {
                let hk = hrow[k];
                if hk == 0.0 {
                    continue;
                }
                let tk = &mut t[k * n..k * n + n];
                for j in 0..n {
                    tk[j] += hk * frow[j];
                }
            }
        }
        // B[k][l] = Σ_j T[k][j]·H[j][l]   (по y)
        let mut b = vec![0.0f64; m * m];
        for k in 0..m {
            for j in 0..n {
                let tv = t[k * n + j];
                if tv == 0.0 {
                    continue;
                }
                let hrow = &self.h[j * m..j * m + m];
                for l in 0..m {
                    b[k * m + l] += tv * hrow[l];
                }
            }
        }
        // C = G⁻¹ B G⁻¹
        let mut tmp = vec![0.0f64; m * m];
        for a in 0..m {
            for l in 0..m {
                let mut s = 0.0;
                for k in 0..m {
                    s += self.ginv[a * m + k] * b[k * m + l];
                }
                tmp[a * m + l] = s;
            }
        }
        let mut c = vec![0.0f64; m * m];
        for a in 0..m {
            for bq in 0..m {
                let mut s = 0.0;
                for l in 0..m {
                    s += tmp[a * m + l] * self.ginv[l * m + bq];
                }
                c[a * m + bq] = s;
            }
        }
        c
    }

    /// Синтез поля F = H·C·Hᵀ на сетке базиса.
    fn synth(&self, c: &[f64]) -> Vec<f64> {
        let (m, n) = (self.m, self.n);
        // U[i][l] = Σ_k H[i][k]·C[k][l]
        let mut u = vec![0.0f64; n * m];
        for i in 0..n {
            let hrow = &self.h[i * m..i * m + m];
            let urow = &mut u[i * m..i * m + m];
            for k in 0..m {
                let hk = hrow[k];
                if hk == 0.0 {
                    continue;
                }
                let crow = &c[k * m..k * m + m];
                for l in 0..m {
                    urow[l] += hk * crow[l];
                }
            }
        }
        // F[i][j] = Σ_l U[i][l]·H[j][l]
        let mut f = vec![0.0f64; n * n];
        for i in 0..n {
            let urow = &u[i * m..i * m + m];
            let frow = &mut f[i * n..i * n + n];
            for j in 0..n {
                let hrow = &self.h[j * m..j * m + m];
                let mut s = 0.0;
                for l in 0..m {
                    s += urow[l] * hrow[l];
                }
                frow[j] = s;
            }
        }
        f
    }
}

// ---------------------------------------------------------------------------
// Физика: отображение символа в драйв, поле освещённости, канал
// ---------------------------------------------------------------------------

/// Отображение комплексного z = (Re, Im) в тройку драйвов.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mapping {
    /// §5.1: R = M + A_L·Re + A_C·Im, G = M + A_L·Re, B = M + A_L·Re − A_C·Im.
    /// Re сидит на АБСОЛЮТНОЙ яркости ⇒ поле освещённости бьёт напрямую.
    Luma,
    /// §5.1-CL (`ChromaMode::ConstLuma1`): сумма драйвов постоянна, z целиком в
    /// цветности; ОБЩИЙ множитель поля сокращается в обращении.
    ConstLuma,
}

/// Эталонный профиль §7.4 SPEC (та же конфигурация, что psicode-sim::main).
fn reference_profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        luma_bits: 3,
        chroma_mode: ChromaMode::ConstLuma1,
        gamma_g_q: 28, // γ_G = 2.200
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15, // 100%
        black_level_q: 2,
        noise_sigma_q: 12,
        mtf_limit_px: 6,
        torn_frames_q: 5,
        crosstalk_rg_q: 3, // 6%
        crosstalk_gb_q: 4, // 8%
        quiet_zone: 1,
        fec_overhead: 2,
        border: psicode_core::profile::BorderMode::LegacyInverted,
    }
}

/// Вся физика тракта: уровни драйва, амплитуды двух отображений, гаммы, помехи.
#[derive(Clone, Copy)]
struct Phys {
    black: f64,
    white: f64,
    /// Середина §5.1.
    mid: f64,
    a_l: f64,
    a_c: f64,
    cl: ConstLumaMap,
    /// Гамма ДИСПЛЕЯ по каналам: радиометрия = (drive/255)^γ.
    g_disp: [f64; 3],
    /// Гамма ISP по каналам: code = 255·радиометрия^(1/γ).
    g_isp: [f64; 3],
    crosstalk: (f64, f64),
    /// σ шума на пиксель в кодах (0..255).
    noise_codes: f64,
    /// Множитель показателя ЛИНЕАРИЗАЦИИ У ПРИЁМНИКА (1.0 — согласованные гаммы).
    /// Отличие от 1 — рассогласование тон-кривой: тогда поле перестаёт быть
    /// мультипликативным в драйв-домене, и «курица-яйцо» с гаммой становится
    /// настоящей.
    rx_expo_scale: f64,
}

impl Phys {
    fn new(p: &CalibProfile) -> Self {
        let white = (255.0 * p.white_level_pct() as f64 / 100.0).round();
        let black = (255.0 * p.black_level_pct() as f64 / 100.0).round();
        let mid = 128.0;
        let usable = ((white - mid).min(mid - black)).max(0.0);
        Phys {
            black,
            white,
            mid,
            a_l: A_L_FRACTION_CHROMA * usable,
            a_c: A_C_FRACTION_CHROMA * usable,
            cl: const_luma_map(p),
            g_disp: [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64],
            g_isp: GAMMA_ISP,
            crosstalk: (
                p.crosstalk_rg_pct() as f64 / 100.0,
                p.crosstalk_gb_pct() as f64 / 100.0,
            ),
            noise_codes: PIX_NOISE_CODES,
            rx_expo_scale: 1.0,
        }
    }

    /// Нейтраль (пьедестал) отображения: драйв при z = 0.
    #[inline]
    fn ped(&self, map: Mapping) -> f64 {
        match map {
            Mapping::Luma => self.mid,
            Mapping::ConstLuma => self.cl.u,
        }
    }

    /// Отклонения драйва от пьедестала по каналам при символе z.
    #[inline]
    fn dev(&self, map: Mapping, re: f64, im: f64) -> [f64; 3] {
        match map {
            Mapping::Luma => [
                self.a_l * re + self.a_c * im,
                self.a_l * re,
                self.a_l * re - self.a_c * im,
            ],
            Mapping::ConstLuma => {
                let (u, b, c) = (self.cl.u, self.cl.b, self.cl.c);
                [u * (-b * re + c * im), u * (2.0 * b * re), u * (-b * re - c * im)]
            }
        }
    }

    /// Полный драйв (без клампа).
    #[inline]
    fn drive(&self, map: Mapping, re: f64, im: f64) -> [f64; 3] {
        let ped = self.ped(map);
        let d = self.dev(map, re, im);
        [ped + d[0], ped + d[1], ped + d[2]]
    }

    /// Максимальный масштаб g, при котором drive(g·z) ещё внутри [black, white]
    /// ПО ВСЕМ ТРЁМ каналам. Жёсткое ограничение диапазона драйва.
    #[inline]
    fn max_scale(&self, map: Mapping, re: f64, im: f64) -> f64 {
        let ped = self.ped(map);
        let up = self.white - ped;
        let down = ped - self.black;
        let d = self.dev(map, re, im);
        let mut g = f64::INFINITY;
        for &v in &d {
            if v > 1e-12 {
                g = g.min(up / v);
            } else if v < -1e-12 {
                g = g.min(down / (-v));
            }
        }
        g
    }

    /// Показатель сквозного преобразования code → drive: drive = 255·(code/255)^e.
    #[inline]
    fn expo(&self, c: usize) -> f64 {
        self.g_isp[c] / self.g_disp[c]
    }

    /// Драйв, восстановленный из НОРМИРОВАННОГО кода [0,1] (наивно, без поля).
    #[inline]
    fn drive_from_code01(&self, c: usize, v: f64) -> f64 {
        255.0 * v.max(0.0).powf(self.expo(c) * self.rx_expo_scale)
    }

    /// ИСТИННОЕ поле в ДРАЙВ-домене: drive_наив = drive·λ^(1/γ_disp).
    /// (Вывод: code = 255·(λ·(d/255)^γ)^(1/γ_isp) ⇒ 255·(code/255)^(γ_isp/γ) =
    /// d·λ^(1/γ) — гамма ISP сокращается ТОЧНО, остаётся гамма дисплея.)
    #[inline]
    fn drive_field(&self, c: usize, lambda: f64) -> f64 {
        lambda.powf(1.0 / self.g_disp[c])
    }
}

/// Режим поля освещённости.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FieldMode {
    /// Поля нет (λ ≡ 1).
    None,
    /// Ахроматический пандус 0.62 → 0.86 по диагонали кадра (замер).
    Achromatic,
    /// Тот же пандус + поканальный дифференциал CHROMA_DIFF размахом по кадру.
    Chromatic,
}

impl FieldMode {
    fn label(self) -> &'static str {
        match self {
            FieldMode::None => "нет",
            FieldMode::Achromatic => "ахром 0.62→0.86",
            FieldMode::Chromatic => "ахром + хром 6%",
        }
    }
}

/// Нормированная координата пандуса t ∈ [0,1] по апертуре (диагональ).
#[inline]
fn field_t(x_aper: f64, y_aper: f64) -> f64 {
    0.5 * (x_aper + y_aper) / (APER as f64 - 1.0)
}

/// λ по каналам в точке апертуры (координаты в px относительно её левого верха).
#[inline]
fn field_lambda(mode: FieldMode, x_aper: f64, y_aper: f64) -> [f64; 3] {
    match mode {
        FieldMode::None => [1.0; 3],
        FieldMode::Achromatic => {
            let l = FIELD_LO + (FIELD_HI - FIELD_LO) * field_t(x_aper, y_aper);
            [l; 3]
        }
        FieldMode::Chromatic => {
            let t = field_t(x_aper, y_aper);
            let l = FIELD_LO + (FIELD_HI - FIELD_LO) * t;
            // чистый ГРАДИЕНТ дифференциала (без сдвига среднего): R/B расходятся
            // на CHROMA_DIFF от края к краю. Постоянный перекос тривиален —
            // интересен именно градиентный, его отображение §5.1-CL не снимает.
            let d = 0.5 * CHROMA_DIFF * (2.0 * t - 1.0);
            [l * (1.0 + d), l, l * (1.0 - d)]
        }
    }
}

/// Полный канал: драйв апертуры (APER², 3, в кодах 0..255) → снимок камеры,
/// нормированные КОДЫ [0,1] размера APER².
///
/// Порядок стадий: клип драйва в [black,white] и квантование 8 бит → радиометрия
/// (d/255)^γ_disp на padded-холст нейтрали → поле освещённости (мультипликативно
/// на линейную радиометрию) → блюр σ (оптика) → кроп → кросстолк (сенсор) →
/// ISP-гамма в коды → шум 6.15 кода на пиксель → клип [0,255].
fn channel(
    drive: &[[f64; 3]],
    ph: &Phys,
    map: Mapping,
    fld: FieldMode,
    sigma: f64,
    rng: &mut Rng,
) -> (Image, usize) {
    let ps = APER + 2 * PAD;
    let mut img = Image::new(ps, ps);
    let ped = ph.ped(map);
    let mut clips = 0usize;

    // padded-холст: нейтраль отображения, под тем же полем
    for py in 0..ps {
        for px in 0..ps {
            let xa = px as f64 - PAD as f64;
            let ya = py as f64 - PAD as f64;
            let lam = field_lambda(fld, xa, ya);
            let mut v = [0.0f32; 3];
            for c in 0..3 {
                v[c] = ((ped / 255.0).powf(ph.g_disp[c]) * lam[c]) as f32;
            }
            img.set(px, py, v);
        }
    }

    // апертура
    for y in 0..APER {
        for x in 0..APER {
            let d = drive[y * APER + x];
            let lam = field_lambda(fld, x as f64, y as f64);
            let mut v = [0.0f32; 3];
            for c in 0..3 {
                if d[c] < ph.black - 1e-9 || d[c] > ph.white + 1e-9 {
                    clips += 1;
                }
                // жёсткое ограничение диапазона драйва + 8-битное квантование
                let q = d[c].clamp(ph.black, ph.white).round();
                v[c] = ((q / 255.0).powf(ph.g_disp[c]) * lam[c]) as f32;
            }
            img.set(PAD + x, PAD + y, v);
        }
    }

    let blurred = pipeline::blur(&img, sigma);
    let mut crop = Image::new(APER, APER);
    for y in 0..APER {
        for x in 0..APER {
            crop.set(x, y, blurred.at(PAD + x, PAD + y));
        }
    }
    pipeline::crosstalk(&mut crop, ph.crosstalk.0, ph.crosstalk.1);

    // ISP: радиометрия → код [0,1]
    for c in 0..3 {
        let inv = 1.0 / ph.g_isp[c];
        for p in crop.data.iter_mut() {
            p[c] = (p[c].max(0.0) as f64).powf(inv) as f32;
        }
    }
    pipeline::add_noise(&mut crop, ph.noise_codes / 255.0, rng);
    pipeline::clamp01(&mut crop);
    (crop, clips)
}

/// Средний КОД по окну `win` px в центре бина `bin` px, затем линеаризация в
/// драйв. Усреднение в КОД-домене (как делает реальный демодулятор), поэтому
/// σ бина = 6.15/win — ровно замеренная физика.
/// Возвращает (APER/bin)² троек драйва, row-major.
fn binned_drives(code: &Image, ph: &Phys, bin: usize, win: usize) -> Vec<[f64; 3]> {
    let n = APER / bin;
    let off = (bin - win) / 2;
    let inv = 1.0 / (win * win) as f64;
    let mut out = vec![[0.0f64; 3]; n * n];
    for by in 0..n {
        for bx in 0..n {
            let mut acc = [0.0f64; 3];
            for dy in 0..win {
                let y = by * bin + off + dy;
                for dx in 0..win {
                    let x = bx * bin + off + dx;
                    let p = code.at(x, y);
                    for c in 0..3 {
                        acc[c] += p[c] as f64;
                    }
                }
            }
            let mut d = [0.0f64; 3];
            for c in 0..3 {
                d[c] = ph.drive_from_code01(c, acc[c] * inv);
            }
            out[by * n + bx] = d;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Учёт степеней свободы: SNR по осям, предсказанный BER, доставленные биты
// ---------------------------------------------------------------------------

/// Накопитель одной ВЕЩЕСТВЕННОЙ оси одной степени свободы.
#[derive(Clone, Copy, Default)]
struct Axis {
    s_rc: f64,
    s_cc: f64,
    s_rr: f64,
}
impl Axis {
    #[inline]
    fn add(&mut self, c: f64, r: f64) {
        self.s_rc += r * c;
        self.s_cc += c * c;
        self.s_rr += r * r;
    }
    /// SNR = мощность сигнала / остаточная дисперсия (EVM-стиль): усиление α
    /// снимается МНК, остаток — всё, что не объясняется линейным α·c.
    fn snr(&self) -> f64 {
        if self.s_cc <= 0.0 {
            return 0.0;
        }
        let sig = self.s_rc * self.s_rc / self.s_cc;
        let noise = self.s_rr - sig;
        if noise <= 1e-300 {
            f64::INFINITY
        } else {
            sig / noise
        }
    }
}

/// Степень свободы (клетка или мода): две оси QPSK + эмпирические ошибки.
#[derive(Clone, Copy, Default)]
struct Dof {
    re: Axis,
    im: Axis,
    errs: usize,
    bits: usize,
}

/// Итог арма на точке развёртки.
#[derive(Clone, Default)]
struct ArmResult {
    /// Доставленных бит на символ: число ОСЕЙ с предсказанным BER < 1e-2.
    delivered: usize,
    /// Сырых бит (2 на степень свободы полезной нагрузки).
    raw: usize,
    /// Эмпирический BER по всем осям.
    ber: f64,
    /// Средний SNR по осям, дБ.
    snr_db: f64,
    /// Остаточная ошибка поля после собственной коррекции арма, RMS в %.
    field_res: f64,
}

/// Свести накопленные степени свободы в итог арма. `lv` — уровней на ось
/// (2 = QPSK, 2 бита на степень свободы; 4 = 16-QAM, 4 бита).
fn finish(dofs: &[Dof], field_res: f64, lv: usize) -> ArmResult {
    let bpa = (lv as f64).log2() as usize; // бит на ось
    let mut delivered = 0usize;
    let mut errs = 0usize;
    let mut bits = 0usize;
    let mut snr_sum = 0.0;
    let mut snr_n = 0usize;
    for d in dofs {
        for ax in [d.re, d.im] {
            let s = ax.snr();
            if ber_from_snr_lv(s, lv) < 1e-2 {
                delivered += bpa;
            }
            if s.is_finite() {
                snr_sum += s;
                snr_n += 1;
            }
        }
        errs += d.errs;
        bits += d.bits;
    }
    ArmResult {
        delivered,
        raw: dofs.len() * 2 * bpa,
        ber: if bits == 0 {
            0.0
        } else {
            errs as f64 / bits as f64
        },
        snr_db: if snr_n == 0 {
            f64::INFINITY
        } else {
            to_db(snr_sum / snr_n as f64)
        },
        field_res,
    }
}

// ---------------------------------------------------------------------------
// Полиномиальная модель поля (порядки 0–2: та же ёмкость, что ψ00..ψ02)
// ---------------------------------------------------------------------------

/// Членов био-квадратичной модели: {1, X, Y, XY, X², Y²}.
const POLY_N: usize = 6;

/// Термы полинома в нормированных координатах X, Y ∈ [−1, 1].
#[inline]
fn poly_terms(xn: f64, yn: f64) -> [f64; POLY_N] {
    [1.0, xn, yn, xn * yn, xn * xn, yn * yn]
}

/// Нормированная координата пикселя апертуры.
#[inline]
fn norm_coord(x_px: f64) -> f64 {
    (x_px - APER as f64 / 2.0) / (APER as f64 / 2.0)
}

/// Порядок модели поля, который снимает арм.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FieldFit {
    /// Ничего не снимаем (f̂ ≡ 1).
    Flat,
    /// Локальное box-среднее по ±k клеткам (текущий двухмасштабный порог).
    Box(usize),
    /// Глобальный аффинный фит (3 члена).
    Affine,
    /// Глобальный био-квадратичный фит (6 членов).
    Quad,
}

impl FieldFit {
    fn terms(self) -> usize {
        match self {
            FieldFit::Flat | FieldFit::Box(_) => 0,
            FieldFit::Affine => 3,
            FieldFit::Quad => POLY_N,
        }
    }
}

/// Взвешенный МНК-фит β поля: min Σ_s (d_s − (Σ_j β_j P_j(s))·q_s)², где q_s —
/// ОЖИДАЕМЫЙ переданный драйв в точке s. Возвращает β длины `nt`.
/// Это общая машина: пилоты дают q_s = пьедестал (эталонные клетки), решающая
/// обратная связь — q_s из принятых решений, HG — q_s из низких мод.
fn fit_field(samples: &[(f64, f64, f64, f64)], nt: usize) -> Option<Vec<f64>> {
    // samples: (xn, yn, d, q)
    let mut a = vec![0.0f64; nt * nt];
    let mut b = vec![0.0f64; nt];
    for &(xn, yn, d, q) in samples {
        let p = poly_terms(xn, yn);
        for i in 0..nt {
            let pi = p[i] * q;
            for j in 0..nt {
                a[i * nt + j] += pi * p[j] * q;
            }
            b[i] += pi * d;
        }
    }
    if spd_solve(&mut a, nt, &mut b) {
        Some(b)
    } else {
        None
    }
}

/// Значение полиномиального поля в точке.
#[inline]
fn poly_eval(beta: &[f64], xn: f64, yn: f64) -> f64 {
    let p = poly_terms(xn, yn);
    let mut s = 0.0;
    for (j, &b) in beta.iter().enumerate() {
        s += b * p[j];
    }
    s
}

/// Истинное поле в ДРАЙВ-домене в центрах клеток: `[cell][channel]`.
fn true_cell_field(ph: &Phys, fld: FieldMode) -> Vec<[f64; 3]> {
    let mut out = vec![[1.0f64; 3]; CELLS * CELLS];
    for cy in 0..CELLS {
        for cx in 0..CELLS {
            let x = (cx * PX_CELL) as f64 + PX_CELL as f64 / 2.0;
            let y = (cy * PX_CELL) as f64 + PX_CELL as f64 / 2.0;
            let lam = field_lambda(fld, x, y);
            let mut f = [0.0f64; 3];
            for c in 0..3 {
                f[c] = ph.drive_field(c, lam[c]);
            }
            out[cy * CELLS + cx] = f;
        }
    }
    out
}

/// RMS относительной остаточной ошибки поля по клеткам и каналам, в %.
fn field_residual(est: &[[f64; 3]], truth: &[[f64; 3]]) -> f64 {
    let mut ss = 0.0;
    let mut n = 0usize;
    for (e, t) in est.iter().zip(truth) {
        for c in 0..3 {
            let r = e[c] / t[c] - 1.0;
            ss += r * r;
            n += 1;
        }
    }
    100.0 * (ss / n as f64).sqrt()
}

/// Остаточная ошибка, которую отображение ПОСТОЯННОЙ ЯРКОСТИ НЕ снимает: общий
/// множитель сокращается точно, а поканальный дифференциал — нет. Меряем разброс
/// f_c вокруг среднего геометрического по каналам.
fn field_residual_common_removed(truth: &[[f64; 3]]) -> f64 {
    let mut ss = 0.0;
    let mut n = 0usize;
    for t in truth {
        let g = (t[0] * t[1] * t[2]).powf(1.0 / 3.0);
        for c in 0..3 {
            let r = t[c] / g - 1.0;
            ss += r * r;
            n += 1;
        }
    }
    100.0 * (ss / n as f64).sqrt()
}

// ---------------------------------------------------------------------------
// АРМ 1/2: PER-CELL и PILOTS
// ---------------------------------------------------------------------------

/// Амплитуда крайнего уровня по оси. Luma: углы квадрата ложатся ровно в
/// [black, white]. ConstLuma: масштаб ПРЯМОУГОЛЬНОЙ решётки внутри гамута,
/// 2/(1+√3) — тот же множитель, что `CL_LATTICE_SCALE` в psicode-core §5.1-CL
/// (угол созвездия попадает ровно на границу гамута).
#[inline]
fn axis_amp(map: Mapping) -> f64 {
    match map {
        Mapping::Luma => 1.0,
        Mapping::ConstLuma => 2.0 / (1.0 + 3.0f64.sqrt()),
    }
}

/// Клетка — пилот (эталон)? Решётка с шагом `period`, если он задан.
#[inline]
fn is_pilot(cx: usize, cy: usize, period: Option<usize>) -> bool {
    match period {
        Some(p) => cx % p == 0 && cy % p == 0,
        None => false,
    }
}

/// Розыгрыш символов клеток: L-PAM по каждой оси на полезных, z = 0 на пилотных.
/// Возвращает (символы, индексы уровней (re, im)).
fn draw_cells(
    map: Mapping,
    pilots: Option<usize>,
    lv: usize,
    rng: &mut Rng,
) -> (Vec<C>, Vec<(usize, usize)>) {
    let a = axis_amp(map);
    let mut v = vec![C::default(); CELLS * CELLS];
    let mut ix = vec![(0usize, 0usize); CELLS * CELLS];
    for cy in 0..CELLS {
        for cx in 0..CELLS {
            let i = cy * CELLS + cx;
            if is_pilot(cx, cy, pilots) {
                v[i] = C { re: 0.0, im: 0.0 };
                ix[i] = (usize::MAX, usize::MAX);
            } else {
                let ir = rng.next_u32_below(lv as u32) as usize;
                let ii = rng.next_u32_below(lv as u32) as usize;
                v[i] = C {
                    re: a * pam_level(ir, lv),
                    im: a * pam_level(ii, lv),
                };
                ix[i] = (ir, ii);
            }
        }
    }
    (v, ix)
}

/// Рендер per-cell: кусочно-постоянный драйв по клеткам.
fn render_percell(cells: &[C], ph: &Phys, map: Mapping) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0f64; 3]; APER * APER];
    for cy in 0..CELLS {
        for cx in 0..CELLS {
            let z = cells[cy * CELLS + cx];
            let d = ph.drive(map, z.re, z.im);
            for dy in 0..PX_CELL {
                let y = cy * PX_CELL + dy;
                for dx in 0..PX_CELL {
                    out[y * APER + cx * PX_CELL + dx] = d;
                }
            }
        }
    }
    out
}

/// Обращение отображения при известной поправке поля f̂.
#[inline]
fn unmap(ph: &Phys, map: Mapping, d: [f64; 3], f: [f64; 3]) -> C {
    let dc = [d[0] / f[0], d[1] / f[1], d[2] / f[2]];
    match map {
        Mapping::Luma => C {
            re: (dc[1] - ph.mid) / ph.a_l,
            im: (dc[0] - dc[2]) / (2.0 * ph.a_c),
        },
        Mapping::ConstLuma => {
            let (x, y) = ph.cl.z_from_drive(dc);
            C { re: x, im: y }
        }
    }
}

/// Жёсткое решение L-PAM по обеим осям на известную амплитуду.
#[inline]
fn slice(z: C, a: f64, lv: usize) -> C {
    C {
        re: a * pam_level(pam_slice(z.re / a, lv), lv),
        im: a * pam_level(pam_slice(z.im / a, lv), lv),
    }
}

/// Локальное box-среднее драйва по ±k клеткам (clamp-to-edge) / пьедестал.
fn box_field(dr: &[[f64; 3]], ped: f64, k: usize) -> Vec<[f64; 3]> {
    let mut out = vec![[1.0f64; 3]; CELLS * CELLS];
    for cy in 0..CELLS {
        let y0 = cy.saturating_sub(k);
        let y1 = (cy + k).min(CELLS - 1);
        for cx in 0..CELLS {
            let x0 = cx.saturating_sub(k);
            let x1 = (cx + k).min(CELLS - 1);
            let mut acc = [0.0f64; 3];
            let mut n = 0usize;
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let d = dr[y * CELLS + x];
                    for c in 0..3 {
                        acc[c] += d[c];
                    }
                    n += 1;
                }
            }
            let mut f = [1.0f64; 3];
            for c in 0..3 {
                f[c] = acc[c] / (n as f64 * ped);
            }
            out[cy * CELLS + cx] = f;
        }
    }
    out
}

/// Нормированные координаты центра клетки.
#[inline]
fn cell_norm(cx: usize, cy: usize) -> (f64, f64) {
    (
        norm_coord((cx * PX_CELL) as f64 + PX_CELL as f64 / 2.0),
        norm_coord((cy * PX_CELL) as f64 + PX_CELL as f64 / 2.0),
    )
}

/// Декод per-cell с собственной оценкой поля.
///
/// * `Flat`     — f̂ ≡ 1 (наивная глобальная нормировка).
/// * `Box(k)`   — локальное box-среднее (данные нуль-средние ⇒ box-среднее
///                драйва = f·пьедестал). Один проход, как у shipped-демодулятора.
/// * `Affine`/`Quad` при `pilots = None` — ГЛОБАЛЬНЫЙ полиномиальный фит с
///   решающей обратной связью: f̂ → решения → q̂ → перефит (`iters` итераций).
///   Это сильная БЕСПЛАТНАЯ база: площади не стоит ничего.
/// * `Affine`/`Quad` при `pilots = Some(p)` — фит ТОЛЬКО по эталонным клеткам
///   (q ≡ пьедестал), как HCCB/JAB. Итерации не нужны — пилот точен.
fn percell_decode(
    dr: &[[f64; 3]],
    ph: &Phys,
    map: Mapping,
    fit: FieldFit,
    pilots: Option<usize>,
    lv: usize,
    iters: usize,
) -> (Vec<C>, Vec<[f64; 3]>) {
    let ped = ph.ped(map);
    let a = axis_amp(map);
    let n = CELLS * CELLS;

    let mut f = vec![[1.0f64; 3]; n];
    match fit {
        FieldFit::Flat => {}
        FieldFit::Box(k) => f = box_field(dr, ped, k),
        FieldFit::Affine | FieldFit::Quad => {
            let nt = fit.terms();
            if let Some(p) = pilots {
                // фит по пилотам: q ≡ пьедестал
                for c in 0..3 {
                    let mut s = Vec::new();
                    for cy in (0..CELLS).step_by(p) {
                        for cx in (0..CELLS).step_by(p) {
                            let (xn, yn) = cell_norm(cx, cy);
                            s.push((xn, yn, dr[cy * CELLS + cx][c], ped));
                        }
                    }
                    if let Some(b) = fit_field(&s, nt) {
                        for cy in 0..CELLS {
                            for cx in 0..CELLS {
                                let (xn, yn) = cell_norm(cx, cy);
                                f[cy * CELLS + cx][c] = poly_eval(&b, xn, yn);
                            }
                        }
                    }
                }
            } else {
                // решающая обратная связь: q̂ из жёстких решений предыдущего круга
                let mut q = vec![[ped; 3]; n];
                for _ in 0..iters.max(1) {
                    for c in 0..3 {
                        let mut s = Vec::with_capacity(n);
                        for cy in 0..CELLS {
                            for cx in 0..CELLS {
                                let i = cy * CELLS + cx;
                                let (xn, yn) = cell_norm(cx, cy);
                                s.push((xn, yn, dr[i][c], q[i][c]));
                            }
                        }
                        if let Some(b) = fit_field(&s, nt) {
                            for cy in 0..CELLS {
                                for cx in 0..CELLS {
                                    let (xn, yn) = cell_norm(cx, cy);
                                    f[cy * CELLS + cx][c] = poly_eval(&b, xn, yn);
                                }
                            }
                        }
                    }
                    for i in 0..n {
                        let z = slice(unmap(ph, map, dr[i], f[i]), a, lv);
                        let d = ph.dev(map, z.re, z.im);
                        for c in 0..3 {
                            q[i][c] = ped + d[c];
                        }
                    }
                }
            }
        }
    }

    let z: Vec<C> = (0..n).map(|i| unmap(ph, map, dr[i], f[i])).collect();
    (z, f)
}

// ---------------------------------------------------------------------------
// АРМ 3: GLOBAL-HG
// ---------------------------------------------------------------------------

/// Решение общей системы n×n методом Гаусса с частичным выбором (T НЕ
/// симметрична: строки — моды, столбцы — члены полинома).
fn lu_solve(a: &mut [f64], n: usize, b: &mut [f64]) -> bool {
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return false;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for r in (col + 1)..n {
            let f = a[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                a[r * n + j] -= f * a[col * n + j];
            }
            b[r] -= f * b[col];
        }
    }
    for r in (0..n).rev() {
        let mut s = b[r];
        for j in (r + 1)..n {
            s -= a[r * n + j] * b[j];
        }
        b[r] = s / a[r * n + r];
    }
    true
}

/// Линейный индекс моды (mx, my).
#[inline]
fn midx(mx: usize, my: usize) -> usize {
    mx * (MMAX + 1) + my
}
/// Порядок моды m+n по линейному индексу.
#[inline]
fn morder(k: usize) -> usize {
    k / (MMAX + 1) + k % (MMAX + 1)
}

/// Ширина огибающей, при которой поворотная точка СТАРШЕЙ моды приходится ровно
/// на край апертуры: w = (L/2)/√(2·MMAX+1). Больше — низкие моды «глобальнее»,
/// но старшие обрезаются краем (усечение ⇒ неортогональность); меньше — наоборот.
fn w_fit() -> f64 {
    (APER as f64 / 2.0) / ((2.0 * MMAX as f64 + 1.0).sqrt())
}

/// Конфигурация HG-арма.
struct HgSetup {
    /// Базис на сетке ПЕРЕДАТЧИКА (пиксели дисплея).
    tx: Hg1,
    /// Базис на сетке ПРИЁМНИКА (бины RX_BIN px).
    rx: Hg1,
    w: f64,
    /// true — калибровочная мода (данных не несёт).
    is_cal: Vec<bool>,
    /// Известные переданные коэффициенты калибровочных мод (TX-домен).
    cal_re: Vec<f64>,
    cal_im: Vec<f64>,
    /// Усиление поля при политике «клип» и при политике «масштаб» (без клипа).
    g_clip: f64,
    g_scale: f64,
    /// Матрица T[c] (6×nt): моды-строки ψ_LOW, столбцы — члены полинома поля.
    t: Vec<Vec<f64>>,
    /// Число членов модели поля.
    nt: usize,
    /// Уровней на ось (2 = QPSK, 4 = 16-QAM).
    lv: usize,
    /// q0 на сетке приёмника: ожидаемый драйв от пьедестала + известных низких мод.
    q0: Vec<[f64; 3]>,
}

/// Розыгрыш коэффициентов: калибровочные — известные и усиленные, остальные
/// L-PAM по каждой оси. Возвращает (Re, Im, индексы уровней).
fn draw_modes(setup: &HgSetup, lv: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>, Vec<(usize, usize)>) {
    let a = std::f64::consts::FRAC_1_SQRT_2; // масштаб произволен: g его снимает
    let mut cre = setup.cal_re.clone();
    let mut cim = setup.cal_im.clone();
    let mut ix = vec![(usize::MAX, usize::MAX); NMODES];
    for k in 0..NMODES {
        if setup.is_cal[k] {
            continue;
        }
        let ir = rng.next_u32_below(lv as u32) as usize;
        let ii = rng.next_u32_below(lv as u32) as usize;
        cre[k] = a * pam_level(ir, lv);
        cim[k] = a * pam_level(ii, lv);
        ix[k] = (ir, ii);
    }
    (cre, cim, ix)
}

/// Усреднение поля с сетки TX (APER²) на сетку RX (RXN²) box-фильтром бина.
fn bin_field(f: &[f64]) -> Vec<f64> {
    let inv = 1.0 / (RX_BIN * RX_BIN) as f64;
    let mut out = vec![0.0f64; RXN * RXN];
    for by in 0..RXN {
        for bx in 0..RXN {
            let mut s = 0.0;
            for dy in 0..RX_BIN {
                let y = by * RX_BIN + dy;
                for dx in 0..RX_BIN {
                    s += f[y * APER + bx * RX_BIN + dx];
                }
            }
            out[by * RXN + bx] = s * inv;
        }
    }
    out
}

impl HgSetup {
    /// Построить арм: базисы, роли мод, калибровочные значения, усиление под
    /// жёсткое ограничение диапазона драйва, матрицу T оценки поля.
    fn new(ph: &Phys, map: Mapping, w: f64, nt: usize, lv: usize, clip_policy: bool) -> Self {
        let center = APER as f64 / 2.0;
        let tx = Hg1::new(MMAX + 1, APER, 1.0, center, w);
        let rx = Hg1::from_binned(&tx, RX_BIN);

        let mut is_cal = vec![false; NMODES];
        let mut cal_re = vec![0.0f64; NMODES];
        let mut cal_im = vec![0.0f64; NMODES];
        let a = CAL_BOOST * std::f64::consts::FRAC_1_SQRT_2;
        for &(mx, my) in &CAL_MODES {
            let k = midx(mx, my);
            is_cal[k] = true;
            cal_re[k] = a;
            cal_im[k] = a;
        }

        let mut s = HgSetup {
            tx,
            rx,
            w,
            is_cal,
            cal_re,
            cal_im,
            g_clip: 1.0,
            g_scale: 1.0,
            t: Vec::new(),
            nt,
            lv,
            q0: Vec::new(),
        };
        let (gc, gs) = s.calibrate_gain(ph, map);
        // `g_clip` — ДЕЙСТВУЮЩЕЕ усиление арма: политика «клип» (квантиль
        // CLIP_TARGET) либо политика «масштаб» (ни одного клипа).
        s.g_clip = if clip_policy { gc } else { gs };
        s.g_scale = gs;
        s.build_t(ph, map);
        s
    }

    /// Усиление поля под ЖЁСТКОЕ ограничение драйва [black, white]:
    /// * `g_scale` — максимум без единого клипа (политика «масштаб»);
    /// * `g_clip`  — квантиль CLIP_TARGET (политика «клип», как в OFDM).
    /// Детерминированные сиды.
    fn calibrate_gain(&self, ph: &Phys, map: Mapping) -> (f64, f64) {
        let mut all: Vec<f64> = Vec::with_capacity(8 * APER * APER);
        for seed in 0..8usize {
            let mut rng = Rng::new(seed_for(90_100, seed));
            let (cre, cim, _) = draw_modes(self, self.lv, &mut rng);
            let fre = self.tx.synth(&cre);
            let fim = self.tx.synth(&cim);
            for i in 0..APER * APER {
                all.push(ph.max_scale(map, fre[i], fim[i]));
            }
        }
        let n = all.len();
        let ki = ((CLIP_TARGET * n as f64) as usize).min(n - 1);
        all.select_nth_unstable_by(ki, |a, b| a.partial_cmp(b).unwrap());
        let g_clip = all[ki];
        let g_scale = SCALE_MARGIN * all[..=ki].iter().cloned().fold(f64::INFINITY, f64::min);
        (g_clip, g_scale)
    }

    /// Матрица оценки поля: T[c][k][j] = proj_МНК(P_j·q0_c)_k по низким модам.
    /// Столбец j = 0 (единица) описывает ЧИСТЫЙ ПЬЕДЕСТАЛ — то есть общее
    /// усиление; столбцы X и Y живут ТОЛЬКО в нечётных модах ψ10/ψ01 (чётность!),
    /// то есть наклон поля читается ровно оттуда, откуда обещано.
    fn build_t(&mut self, ph: &Phys, map: Mapping) {
        // q0 на сетке RX: пьедестал + известные низкие моды (переданные значения)
        let fre = bin_field(&self.tx.synth(&self.cal_re));
        let fim = bin_field(&self.tx.synth(&self.cal_im));
        let ped = ph.ped(map);
        let g = self.g_clip;
        let mut q0 = vec![[ped; 3]; RXN * RXN];
        for i in 0..RXN * RXN {
            let d = ph.dev(map, g * fre[i], g * fim[i]);
            for c in 0..3 {
                q0[i][c] = ped + d[c];
            }
        }
        let nt = self.nt;
        let mut t = Vec::with_capacity(3);
        for c in 0..3 {
            let mut mat = vec![0.0f64; CAL_MODES.len() * nt];
            for j in 0..nt {
                let mut fld = vec![0.0f64; RXN * RXN];
                for by in 0..RXN {
                    let yn = norm_coord((by as f64 + 0.5) * RX_BIN as f64);
                    for bx in 0..RXN {
                        let xn = norm_coord((bx as f64 + 0.5) * RX_BIN as f64);
                        let p = poly_terms(xn, yn);
                        fld[by * RXN + bx] = p[j] * q0[by * RXN + bx][c];
                    }
                }
                let pr = self.rx.project(&fld);
                for (row, &(mx, my)) in CAL_MODES.iter().enumerate() {
                    mat[row * nt + j] = pr[self.rx.idx(mx, my)];
                }
            }
            t.push(mat);
        }
        self.t = t;
        self.q0 = q0;
    }

    /// Рендер HG: синтез поля по всей апертуре и отображение в драйв.
    fn render(&self, ph: &Phys, map: Mapping, cre: &[f64], cim: &[f64], g: f64) -> Vec<[f64; 3]> {
        let fre = self.tx.synth(cre);
        let fim = self.tx.synth(cim);
        let mut out = vec![[0.0f64; 3]; APER * APER];
        for i in 0..APER * APER {
            out[i] = ph.drive(map, g * fre[i], g * fim[i]);
        }
        out
    }

    /// Комплексное поле символа из троек драйва приёмника (обращение отображения
    /// ПОТОЧЕЧНО, после деления на оценку поля).
    fn demap_grid(&self, ph: &Phys, map: Mapping, dr: &[[f64; 3]], f: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>) {
        let n = RXN * RXN;
        let mut sre = vec![0.0f64; n];
        let mut sim = vec![0.0f64; n];
        for i in 0..n {
            let z = unmap(ph, map, dr[i], f[i]);
            sre[i] = z.re;
            sim[i] = z.im;
        }
        (sre, sim)
    }
}

/// Слепая оценка модального усиления порядка `o` после аттенюации блюром:
/// старт по СКЗ, затем уточнение ПО РЕШЕНИЯМ (g = Σr·d / Σd², d — текущие
/// жёсткие решения) — стандартная схема захвата усиления. Чистая СКЗ смещена
/// шумом и разбросом усиления внутри порядка, чего при L > 2 хватает, чтобы
/// сдвинуть уровень через границу решения. Возвращает множитель к номинальной
/// амплитуде оси; нужна и контуру оценки поля, и подсчёту бит-ошибок.
fn order_gain(setup: &HgSetup, cre: &[f64], cim: &[f64], o: usize, a_nom: f64, lv: usize) -> f64 {
    // Порядков в тензорном наборе 2·MMAX+1, но мод в них НЕРАВНОМЕРНО: в
    // порядке 46 она одна, в порядке 23 — двадцать четыре. По одной-двум модам
    // слепое усиление не оценить, поэтому пулим соседние порядки, пока не
    // наберётся хотя бы MIN_POOL мод (аттенюация по порядку меняется медленно,
    // так что смещение от пулинга мало).
    const MIN_POOL: usize = 8;
    let mut idx: Vec<usize> = Vec::new();
    for r in 0..=3usize {
        idx = (0..NMODES)
            .filter(|&k| !setup.is_cal[k] && morder(k).abs_diff(o) <= r)
            .collect();
        if idx.len() >= MIN_POOL {
            break;
        }
    }
    if idx.is_empty() {
        return 0.0;
    }
    let mut ss = 0.0;
    for &k in &idx {
        ss += cre[k] * cre[k] + cim[k] * cim[k];
    }
    let rms = (ss / (2 * idx.len()) as f64).sqrt();
    let mut g = rms / (a_nom * pam_m2(lv).sqrt()).max(1e-12);
    for _ in 0..3 {
        if g <= 1e-12 {
            break;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for &k in &idx {
            for &r in &[cre[k], cim[k]] {
                let d = a_nom * pam_level(pam_slice(r / (g * a_nom), lv), lv);
                num += r * d;
                den += d * d;
            }
        }
        if den > 1e-12 {
            g = num / den;
        } else {
            break;
        }
    }
    g
}

/// Итог декода HG на одной попытке.
struct HgRx {
    /// Оценённые коэффициенты (RX-домен).
    cre: Vec<f64>,
    cim: Vec<f64>,
    /// Остаточная ошибка поля по итерациям (%), если задан эталон.
    trace: Vec<f64>,
    /// ρ = (|ĉ20|+|ĉ02|)/(2|ĉ00|) ДО коррекции поля (итерация 0) и ПОСЛЕ.
    rho_raw: f64,
    rho_fin: f64,
}

/// Декод GLOBAL-HG.
///
/// Контур: линеаризация (γ известна) → МНК-проекция → фит поля ПО НИЗКИМ МОДАМ
/// → деление → перепроекция. Вычитание утечки данных в низкие моды делается по
/// ЖЁСТКИМ решениям — иначе контур вырождается в тождество (soft-оценка низких
/// мод содержит ровно то, что мы пытаемся объяснить полем).
fn hg_decode(
    setup: &HgSetup,
    ph: &Phys,
    map: Mapping,
    dr: &[[f64; 3]],
    iters: usize,
    truth: Option<&[[f64; 3]]>,
) -> HgRx {
    let n = RXN * RXN;
    let ped = ph.ped(map);
    let g = setup.g_clip;
    let ncal = CAL_MODES.len();
    let nt = setup.nt;

    // «сырые» проекции измеренного драйва на низкие моды — считаются один раз
    let mut r_raw = vec![0.0f64; 3 * ncal];
    for c in 0..3 {
        let fld: Vec<f64> = (0..n).map(|i| dr[i][c]).collect();
        let pr = setup.rx.project(&fld);
        for (row, &(mx, my)) in CAL_MODES.iter().enumerate() {
            r_raw[c * ncal + row] = pr[setup.rx.idx(mx, my)];
        }
    }

    let mut beta = vec![vec![0.0f64; nt]; 3];
    for b in beta.iter_mut() {
        b[0] = 1.0;
    }
    let mut fgrid = vec![[1.0f64; 3]; n];
    let mut cre = vec![0.0f64; NMODES];
    let mut cim = vec![0.0f64; NMODES];
    let mut trace = Vec::new();
    let mut f_cell = vec![[1.0f64; 3]; CELLS * CELLS];
    let (i00, i20, i02) = (
        setup.rx.idx(0, 0),
        setup.rx.idx(2, 0),
        setup.rx.idx(0, 2),
    );
    let mut rho_raw = 0.0;
    let a_nom = std::f64::consts::FRAC_1_SQRT_2;
    let lv = setup.lv;

    for it in 0..iters.max(1) {
        // (1) деление на текущую оценку поля + обращение отображения + проекция.
        // §5.1-CL инвариантно к ОБЩЕМУ множителю (делит на измеренную сумму
        // каналов), поэтому делить его на полную оценку поля вредно — шум оценки
        // попадёт в цветность. Оставляем только ПОКАНАЛЬНЫЙ дифференциал.
        let fuse: Vec<[f64; 3]> = match map {
            Mapping::Luma => fgrid.clone(),
            Mapping::ConstLuma => fgrid
                .iter()
                .map(|f| {
                    let g = (f[0] * f[1] * f[2]).abs().powf(1.0 / 3.0).max(1e-9);
                    [f[0] / g, f[1] / g, f[2] / g]
                })
                .collect(),
        };
        let (sre, sim) = setup.demap_grid(ph, map, dr, &fuse);
        cre = setup.rx.project(&sre);
        cim = setup.rx.project(&sim);
        if it == 0 {
            let a00 = C { re: cre[i00], im: cim[i00] }.abs();
            let a20 = C { re: cre[i20], im: cim[i20] }.abs();
            let a02 = C { re: cre[i02], im: cim[i02] }.abs();
            rho_raw = if a00 > 0.0 { 0.5 * (a20 + a02) / a00 } else { 0.0 };
        }

        // (2) поле в центрах клеток -> метрика остатка
        for cy in 0..CELLS {
            for cx in 0..CELLS {
                let (xn, yn) = cell_norm(cx, cy);
                for c in 0..3 {
                    f_cell[cy * CELLS + cx][c] = poly_eval(&beta[c], xn, yn);
                }
            }
        }
        if let Some(t) = truth {
            trace.push(field_residual(&f_cell, t));
        }
        if it + 1 == iters.max(1) {
            break;
        }

        // (3) ЖЁСТКИЕ решения на старших модах -> реконструкция утечки данных.
        // Мягкая оценка здесь НЕПРИГОДНА: контур выродился бы в тождество
        // (soft-восстановление воспроизводит ровно то, что объясняем полем).
        let mut gain = vec![0.0f64; 2 * (MMAX + 1)];
        for (o, gp) in gain.iter_mut().enumerate() {
            *gp = order_gain(setup, &cre, &cim, o, a_nom, lv);
        }
        let mut hre = vec![0.0f64; NMODES];
        let mut him = vec![0.0f64; NMODES];
        for k in 0..NMODES {
            if setup.is_cal[k] {
                continue;
            }
            let ga = gain[morder(k)] * a_nom;
            if ga <= 1e-12 {
                continue;
            }
            hre[k] = ga * pam_level(pam_slice(cre[k] / ga, lv), lv);
            him[k] = ga * pam_level(pam_slice(cim[k] / ga, lv), lv);
        }
        let sh_re = setup.rx.synth(&hre);
        let sh_im = setup.rx.synth(&him);

        // (4) вычитаем проекцию f̂·dev(ŝ_HIGH) из измерений низких мод
        let mut rhs = vec![0.0f64; 3 * ncal];
        for c in 0..3 {
            let fld: Vec<f64> = (0..n)
                .map(|i| fgrid[i][c] * ph.dev(map, sh_re[i], sh_im[i])[c])
                .collect();
            let pr = setup.rx.project(&fld);
            for (row, &(mx, my)) in CAL_MODES.iter().enumerate() {
                rhs[c * ncal + row] = r_raw[c * ncal + row] - pr[setup.rx.idx(mx, my)];
            }
        }

        // (5) решаем T·β = rhs по низким модам (6 уравнений, nt неизвестных)
        for c in 0..3 {
            let mut a = vec![0.0f64; nt * nt];
            let mut b = vec![0.0f64; nt];
            // нормальные уравнения на случай nt < 6 (аффинная модель)
            for row in 0..ncal {
                for i in 0..nt {
                    let ti = setup.t[c][row * nt + i];
                    for j in 0..nt {
                        a[i * nt + j] += ti * setup.t[c][row * nt + j];
                    }
                    b[i] += ti * rhs[c * ncal + row];
                }
            }
            if lu_solve(&mut a, nt, &mut b) {
                // защита от расходимости: усиление поля физично в (0.2, 3)
                if b[0].is_finite() && b[0] > 0.2 && b[0] < 3.0 {
                    beta[c] = b;
                }
            }
        }
        for by in 0..RXN {
            let yn = norm_coord((by as f64 + 0.5) * RX_BIN as f64);
            for bx in 0..RXN {
                let xn = norm_coord((bx as f64 + 0.5) * RX_BIN as f64);
                for c in 0..3 {
                    fgrid[by * RXN + bx][c] = poly_eval(&beta[c], xn, yn);
                }
            }
        }
    }

    let _ = (ped, g, f_cell);
    let a00 = C { re: cre[i00], im: cim[i00] }.abs();
    let a20 = C { re: cre[i20], im: cim[i20] }.abs();
    let a02 = C { re: cre[i02], im: cim[i02] }.abs();
    HgRx {
        rho_fin: if a00 > 0.0 { 0.5 * (a20 + a02) / a00 } else { 0.0 },
        rho_raw,
        cre,
        cim,
        trace,
    }
}

// ---------------------------------------------------------------------------
// Прогон армов на точке развёртки
// ---------------------------------------------------------------------------

/// Окна усреднения клетки (px в стороне). Полное окно 12 — согласованный фильтр
/// без блюра; узкие окна режут меж-клеточный ISI ценой шума. Арм PER-CELL
/// получает ЛУЧШЕЕ из них на каждой точке — не соломенное чучело.
const WINDOWS: [usize; 5] = [12, 8, 6, 4, 2];
/// Итераций решающей обратной связи у полиномиального фита per-cell.
const ITER_PC: usize = 3;

/// Результат per-cell арма вместе с выбранным окном.
struct PcOut {
    win: usize,
    res: ArmResult,
}

/// Прогон per-cell/pilots: один канал — несколько моделей поля и окон.
#[allow(clippy::too_many_arguments)]
fn eval_percell(
    ph: &Phys,
    map: Mapping,
    pilots: Option<usize>,
    fits: &[FieldFit],
    fld: FieldMode,
    sigma: f64,
    lv: usize,
    point: usize,
) -> Vec<PcOut> {
    let truth = true_cell_field(ph, fld);
    let n = CELLS * CELLS;
    let payload: Vec<usize> = (0..n)
        .filter(|&i| !is_pilot(i % CELLS, i / CELLS, pilots))
        .collect();
    let nf = fits.len();
    let nw = WINDOWS.len();
    let mut acc: Vec<Vec<Dof>> = vec![vec![Dof::default(); payload.len()]; nf * nw];
    let mut fres = vec![0.0f64; nf * nw];

    let a = axis_amp(map);
    let bpa = (lv as f64).log2() as usize;
    for t in 0..TRIALS {
        let mut rng = Rng::new(seed_for(point, t));
        let (cells, ix) = draw_cells(map, pilots, lv, &mut rng);
        let drive = render_percell(&cells, ph, map);
        let (code, _) = channel(&drive, ph, map, fld, sigma, &mut rng);
        for (wi, &win) in WINDOWS.iter().enumerate() {
            let dr = binned_drives(&code, ph, PX_CELL, win);
            for (fi, &fit) in fits.iter().enumerate() {
                let (z, f) = percell_decode(&dr, ph, map, fit, pilots, lv, ITER_PC);
                let slot = fi * nw + wi;
                fres[slot] += field_residual(&f, &truth);
                for (di, &i) in payload.iter().enumerate() {
                    let c = cells[i];
                    let r = z[i];
                    let d = &mut acc[slot][di];
                    d.re.add(c.re, r.re);
                    d.im.add(c.im, r.im);
                    d.errs += pam_bit_errors(ix[i].0, pam_slice(r.re / a, lv))
                        + pam_bit_errors(ix[i].1, pam_slice(r.im / a, lv));
                    d.bits += 2 * bpa;
                }
            }
        }
    }

    let mut out = Vec::with_capacity(nf);
    for fi in 0..nf {
        let mut best: Option<PcOut> = None;
        for (wi, &win) in WINDOWS.iter().enumerate() {
            let slot = fi * nw + wi;
            let r = finish(&acc[slot], fres[slot] / TRIALS as f64, lv);
            if best.as_ref().map_or(true, |b| r.delivered > b.res.delivered) {
                best = Some(PcOut { win, res: r });
            }
        }
        out.push(best.unwrap());
    }
    out
}

/// Итог HG-арма: агрегат + разрез по порядку моды + след сходимости.
struct HgOut {
    res: ArmResult,
    /// SNR (дБ) по порядку m+n.
    order_snr: Vec<f64>,
    /// Доставленных бит по порядку m+n.
    order_bits: Vec<usize>,
    /// Остаточная ошибка поля по итерациям контура, %.
    trace: Vec<f64>,
    /// Доля клиппинга драйва.
    clip_frac: f64,
    /// ρ = (|ĉ20|+|ĉ02|)/(2|ĉ00|) ДО коррекции поля и ПОСЛЕ.
    rho_raw: f64,
    rho_fin: f64,
    /// Число data-осей по порядку m+n (0 — порядок целиком калибровочный).
    order_n: Vec<usize>,
}

/// Прогон GLOBAL-HG.
fn eval_hg(
    setup: &HgSetup,
    ph: &Phys,
    map: Mapping,
    fld: FieldMode,
    sigma: f64,
    point: usize,
    trials: usize,
) -> HgOut {
    let truth = true_cell_field(ph, fld);
    let data: Vec<usize> = (0..NMODES).filter(|&k| !setup.is_cal[k]).collect();
    let lv = setup.lv;
    let bpa = (lv as f64).log2() as usize;
    let mut acc = vec![Dof::default(); data.len()];
    let mut fres = 0.0;
    let mut trace_sum: Vec<f64> = Vec::new();
    let mut clip = 0usize;
    let mut clip_tot = 0usize;
    let (mut rho_raw, mut rho_fin) = (0.0f64, 0.0f64);

    for t in 0..trials {
        let mut rng = Rng::new(seed_for(point, t));
        let (cre, cim, ix) = draw_modes(setup, lv, &mut rng);
        let drive = setup.render(ph, map, &cre, &cim, setup.g_clip);
        let (code, cl) = channel(&drive, ph, map, fld, sigma, &mut rng);
        clip += cl;
        clip_tot += 3 * APER * APER;
        let dr = binned_drives(&code, ph, RX_BIN, RX_BIN);
        let rx = hg_decode(setup, ph, map, &dr, MAXIT, Some(&truth));
        fres += rx.trace.last().copied().unwrap_or(0.0);
        if trace_sum.is_empty() {
            trace_sum = vec![0.0; rx.trace.len()];
        }
        for (a, b) in trace_sum.iter_mut().zip(&rx.trace) {
            *a += *b;
        }
        rho_raw += rx.rho_raw;
        rho_fin += rx.rho_fin;

        // слепое модальное усиление по порядку — только для СЧЁТА бит-ошибок
        // (решения по SNR/EVM его не требуют: PAM-порог масштабно-зависим).
        let mut gain = vec![0.0f64; 2 * (MMAX + 1)];
        for (o, gp) in gain.iter_mut().enumerate() {
            *gp = order_gain(setup, &rx.cre, &rx.cim, o, std::f64::consts::FRAC_1_SQRT_2, lv);
        }
        for (di, &k) in data.iter().enumerate() {
            let d = &mut acc[di];
            d.re.add(cre[k], rx.cre[k]);
            d.im.add(cim[k], rx.cim[k]);
            let ga = (gain[morder(k)] * std::f64::consts::FRAC_1_SQRT_2).max(1e-12);
            d.errs += pam_bit_errors(ix[k].0, pam_slice(rx.cre[k] / ga, lv))
                + pam_bit_errors(ix[k].1, pam_slice(rx.cim[k] / ga, lv));
            d.bits += 2 * bpa;
        }
    }

    let res = finish(&acc, fres / trials as f64, lv);
    let maxord = 2 * MMAX + 1;
    let mut order_snr = vec![0.0f64; maxord];
    let mut order_bits = vec![0usize; maxord];
    let mut order_n = vec![0usize; maxord];
    for (di, &k) in data.iter().enumerate() {
        let o = morder(k);
        for ax in [acc[di].re, acc[di].im] {
            let s = ax.snr();
            if s.is_finite() {
                order_snr[o] += s;
                order_n[o] += 1;
            } else {
                order_n[o] += 1;
                order_snr[o] += 1e12;
            }
            if ber_from_snr_lv(s, lv) < 1e-2 {
                order_bits[o] += bpa;
            }
        }
    }
    for o in 0..maxord {
        order_snr[o] = if order_n[o] == 0 {
            f64::NEG_INFINITY
        } else {
            to_db(order_snr[o] / order_n[o] as f64)
        };
    }

    HgOut {
        res,
        order_snr,
        order_bits,
        order_n,
        trace: trace_sum.iter().map(|v| v / trials as f64).collect(),
        clip_frac: clip as f64 / clip_tot as f64,
        rho_raw: rho_raw / trials as f64,
        rho_fin: rho_fin / trials as f64,
    }
}

/// Пара сетапов HG под ОБЕИМИ политиками жёсткого диапазона драйва: «клип»
/// (амплитуда выше, но CLIP_TARGET пикселей срезано) и «масштаб» (ни одного
/// клипа, амплитуда ниже). Арм получает ЛУЧШУЮ из двух на каждой точке —
/// иначе выбор политики был бы скрытым гандикапом.
struct HgPair {
    clip: HgSetup,
    scale: HgSetup,
}

impl HgPair {
    fn new(ph: &Phys, map: Mapping, w: f64, nt: usize, lv: usize) -> Self {
        HgPair {
            clip: HgSetup::new(ph, map, w, nt, lv, true),
            scale: HgSetup::new(ph, map, w, nt, lv, false),
        }
    }
    /// Прогон обеих политик; возвращает лучшую по доставленным битам и её метку.
    fn eval(
        &self,
        ph: &Phys,
        map: Mapping,
        fld: FieldMode,
        sigma: f64,
        point: usize,
        trials: usize,
    ) -> (HgOut, &'static str) {
        let a = eval_hg(&self.clip, ph, map, fld, sigma, point, trials);
        let b = eval_hg(&self.scale, ph, map, fld, sigma, point + 1, trials);
        if a.res.delivered >= b.res.delivered {
            (a, "клип")
        } else {
            (b, "масштаб")
        }
    }
    fn by_label(&self, label: &str) -> &HgSetup {
        if label == "клип" {
            &self.clip
        } else {
            &self.scale
        }
    }
}

/// Линейная инверсия монотонной калибровки ρ_rel(σ).
fn invert_cal(table: &[(f64, f64)], rho: f64) -> f64 {
    if rho >= table[0].1 {
        return table[0].0;
    }
    let last = table.len() - 1;
    if rho <= table[last].1 {
        return table[last].0;
    }
    for wnd in table.windows(2) {
        let (s0, r0) = wnd[0];
        let (s1, r1) = wnd[1];
        if rho <= r0 && rho >= r1 {
            let f = (r0 - rho) / (r0 - r1);
            return s0 + f * (s1 - s0);
        }
    }
    table[last].0
}

// ---------------------------------------------------------------------------
// Развёртка и отчёт
// ---------------------------------------------------------------------------

/// Физика без шума, кросстолка и рассогласований — для sanity-ворот.
fn clean_phys(ph: &Phys) -> Phys {
    let mut q = *ph;
    q.noise_codes = 0.0;
    q.crosstalk = (0.0, 0.0);
    q
}

/// Пиковый и среднеквадратичный размах драйва HG-поля при g = 1 (crest-фактор —
/// та самая PAPR-проблема OFDM: у per-cell QPSK crest ровно 1).
fn hg_crest(setup: &HgSetup, ph: &Phys, map: Mapping) -> f64 {
    let mut rng = Rng::new(seed_for(90_200, 0));
    let (cre, cim, _) = draw_modes(setup, setup.lv, &mut rng);
    let fre = setup.tx.synth(&cre);
    let fim = setup.tx.synth(&cim);
    let (mut peak, mut ss, mut n) = (0.0f64, 0.0f64, 0usize);
    for i in 0..APER * APER {
        let d = ph.dev(map, fre[i], fim[i]);
        for &v in &d {
            peak = peak.max(v.abs());
            ss += v * v;
            n += 1;
        }
    }
    peak / (ss / n as f64).sqrt()
}

/// Строка таблицы «арм × что-то».
fn row(label: &str, cells: &[String]) -> String {
    report::table_row(label, cells)
}

/// Заголовок markdown-таблицы по подписям столбцов.
fn head(first: &str, cols: &[String]) -> String {
    let mut s = format!("| {first}");
    for c in cols {
        s.push_str(&format!(" | {c}"));
    }
    s.push_str(" |\n|---");
    for _ in cols {
        s.push_str("|---");
    }
    s.push('|');
    s
}

/// Спецификация опорного (не-HG) арма.
struct RowSpec {
    name: &'static str,
    map: Mapping,
    pilots: Option<usize>,
    fit: FieldFit,
}

/// Пять опорных армов: три per-cell (наивный / shipped-локальный / глобальный
/// полиномиальный с решающей обратной связью), пилотный (HCCB/JAB) и
/// отображение постоянной яркости §5.1-CL.
const SPECS: [RowSpec; 5] = [
    RowSpec { name: "PER-CELL naive", map: Mapping::Luma, pilots: None, fit: FieldFit::Flat },
    RowSpec { name: "PER-CELL box±2 (shipped)", map: Mapping::Luma, pilots: None, fit: FieldFit::Box(2) },
    RowSpec { name: "PER-CELL biquad+DFB", map: Mapping::Luma, pilots: None, fit: FieldFit::Quad },
    RowSpec { name: "PILOTS 6×6 biquad", map: Mapping::Luma, pilots: Some(4), fit: FieldFit::Quad },
    RowSpec { name: "PER-CELL §5.1-CL", map: Mapping::ConstLuma, pilots: None, fit: FieldFit::Flat },
];

/// Уровней на ось в двух прогонах: 2 (QPSK, 2 бита/степень свободы) и
/// 4 (16-QAM, 4 бита). При двух уровнях порог решения стоит в НУЛЕ, и поле
/// освещённости почти безвредно — дискриминирующим опыт становится на четырёх.
const LEVELS: [usize; 2] = [2, 4];

pub fn cmd_gprobe() {
    let t0 = Instant::now();
    let p = reference_profile();
    let ph = Phys::new(&p);
    let w = w_fit();

    println!("# psicode-sim gprobe — GLOBAL-HG против PER-CELL и PILOTS");
    println!();
    println!(
        "Апертура {APER}×{APER} camera-px = {CELLS}×{CELLS} клеток по {PX_CELL} px; \
         {NMODES} мод HG ({}×{} тензорно) — РОВНО столько же степеней свободы, \
         сколько клеток. Одна и та же модуляция во всех армах.",
        MMAX + 1,
        MMAX + 1
    );
    println!(
        "Канал (замеры на Galaxy A22 + 1080p): шум {PIX_NOISE_CODES} кода/ПИКСЕЛЬ, поле \
         {FIELD_LO}→{FIELD_HI} мультипликативно на линейную радиометрию, γ ISP \
         [{:.1},{:.1},{:.1}], γ дисплея [{:.3},{:.3},{:.3}], кросстолк {:.0}/{:.0}%.",
        GAMMA_ISP[0], GAMMA_ISP[1], GAMMA_ISP[2],
        ph.g_disp[0], ph.g_disp[1], ph.g_disp[2],
        ph.crosstalk.0 * 100.0, ph.crosstalk.1 * 100.0
    );
    println!(
        "Драйв: [{}, {}] жёстко; §5.1 A_L={:.1}, A_C={:.1}; §5.1-CL u={:.0}, amp={:.0}. \
         {TRIALS} попыток/точку, детерминированные сиды (psicode-sim::rng).",
        ph.black, ph.white, ph.a_l, ph.a_c, ph.cl.u, ph.cl.amp
    );
    println!();

    let hg2 = HgSetup::new(&ph, Mapping::Luma, w, POLY_N, 2, true);

    // ---------------- 0. диагностика базиса ----------------
    println!("## 0. Базис: усечение, ортогональность, охват калибровочных мод");
    println!(
        "Ширина огибающей w = {:.2} px выбрана так, чтобы поворотная точка СТАРШЕЙ моды \
         (m={MMAX}) легла ровно на край апертуры: w = (L/2)/√(2m+1).",
        w
    );
    println!("| сетка | отсчётов | max\\|Грам вне-диаг\\| | κ_∞(Грам) |");
    println!("|---|---|---|---|");
    println!(
        "| передатчик (1 px) | {} | {:.3e} | {:.2} |",
        hg2.tx.n, hg2.tx.gram_max_off, hg2.tx.cond
    );
    println!(
        "| приёмник ({RX_BIN} px бин) | {} | {:.3e} | {:.2} |",
        hg2.rx.n, hg2.rx.gram_max_off, hg2.rx.cond
    );
    println!();
    println!(
        "Приёмник усредняет код по бину {RX_BIN}×{RX_BIN} px, поэтому его базис построен как \
         box-среднее передающего (`Hg1::from_binned`) — иначе бин-фильтр давил бы старшие моды \
         систематически и ВНЕ модели. Сам фильтр мягок: старшая мода теряет относительно \
         нулевой лишь {:.1}% амплитуды по двум осям (её локальная длина волны ≈ клетка, а бин \
         вчетверо мельче).",
        100.0 * (1.0 - (hg2.rx.bin_norm[MMAX] / hg2.rx.bin_norm[0]).powi(2))
    );
    println!(
        "Проекция — МНК по РЕАЛЬНОЙ сетке отсчётов (C = G⁻¹·HᵀFH·G⁻¹), НЕ корреляция: \
         усечённый краем гауссиан под наивной корреляцией не ортогонален. 2-D Грам = G⊗G \
         (разделимость), поэтому max вне-диагонали 2-D равен 1-D, а κ(2-D) = κ(1-D)²."
    );
    let cal_r = w * 5.0f64.sqrt() / (APER as f64 / 2.0);
    println!(
        "**Охват калибровочного канала:** поворотная точка моды порядка 2 лежит на радиусе \
         w·√5 = {:.0} px = {:.0}% полуширины апертуры, т.е. ψ00..ψ02 физически щупают лишь \
         ~{:.0}% ПЛОЩАДИ символа. Низкие HG-моды сконцентрированы в ЦЕНТРЕ: их \
         «глобальность» модельная, а не пространственная — плечо для оценки наклона короткое.",
        w * 5.0f64.sqrt(), 100.0 * cal_r, 100.0 * cal_r * cal_r
    );
    println!();

    // ---------------- 0б. бюджет амплитуды / PAPR ----------------
    let crest = hg_crest(&hg2, &ph, Mapping::Luma);
    let pc_amp = ph.a_l * PX_CELL as f64;
    let hg_amp = hg2.g_clip * ph.a_l * std::f64::consts::FRAC_1_SQRT_2;
    let hg_amp_ns = hg2.g_scale * ph.a_l * std::f64::consts::FRAC_1_SQRT_2;
    println!("## 0б. Бюджет амплитуды под ЖЁСТКИМ диапазоном драйва (PAPR)");
    println!(
        "Сравнение честное: коэффициент в базисе ЕДИНИЧНОЙ L2-нормы. Шум проецируется на \
         любую такую функцию с ОДНОЙ И ТОЙ ЖЕ дисперсией, поэтому отношение амплитуд есть \
         в точности отношение SNR."
    );
    println!("| схема | амплитуда/степень свободы (коды драйва) | к per-cell |");
    println!("|---|---|---|");
    println!("| PER-CELL, клетка {PX_CELL}×{PX_CELL} | {pc_amp:.0} | 0.0 дБ |");
    println!(
        "| GLOBAL-HG, политика «клип» ({:.1}%) | {hg_amp:.0} | {:+.1} дБ |",
        100.0 * CLIP_TARGET, 20.0 * (hg_amp / pc_amp).log10()
    );
    println!(
        "| GLOBAL-HG, политика «масштаб» (0 клипа, запас {:.0}%) | {hg_amp_ns:.0} | {:+.1} дБ |",
        100.0 * (1.0 - SCALE_MARGIN),
        20.0 * (hg_amp_ns / pc_amp).log10()
    );
    println!(
        "| GLOBAL-HG калибровочная мода (×{CAL_BOOST}) | {:.0} | {:+.1} дБ |",
        hg_amp * CAL_BOOST, 20.0 * (hg_amp * CAL_BOOST / pc_amp).log10()
    );
    println!();
    println!(
        "Crest-фактор синтезированного HG-поля (пик/СКЗ размаха драйва) = **{crest:.2}**; у \
         per-cell он ровно **1.00** — созвездие ПОСТОЯННОГО МОДУЛЯ упирается в [{}, {}] в \
         КАЖДОЙ клетке. Это структурный, а не реализационный проигрыш глобального \
         разложения: сумма многих мод со случайными знаками почти гауссова, а гауссово поле, \
         втиснутое в конечный диапазон драйва, теряет ~20·lg(crest) дБ мощности. Политика \
         «клип» возвращает часть — ценой искажения (доля клипа контролируется).",
        ph.black, ph.white
    );
    println!(
        "Стоимость калибровки HG: **площадь 0%** (моды, а не клетки), **полоса {:.2}%** \
         ({}/{NMODES} мод), **мощность {:.1}%**. Для сравнения: пилотная решётка 6×6 стоит \
         **{:.1}% ПЛОЩАДИ** ({} из {} клеток).",
        100.0 * CAL_MODES.len() as f64 / NMODES as f64,
        CAL_MODES.len(),
        100.0 * 6.0 * CAL_BOOST * CAL_BOOST / (570.0 + 6.0 * CAL_BOOST * CAL_BOOST),
        100.0 * 36.0 / (CELLS * CELLS) as f64,
        36,
        CELLS * CELLS
    );
    println!();

    // ---------------- 1. sanity-ворота ----------------
    println!("## 1. Sanity-ворота: σ=0, без шума, без поля, без кросстолка, согласованные γ");
    let cph = clean_phys(&ph);
    println!("| арм | уровней/ось | ошибочных бит | остаточная амплитуда ошибки |");
    println!("|---|---|---|---|");
    for &lv in &LEVELS {
        for sp in &SPECS {
            let o = eval_percell(&cph, sp.map, sp.pilots, &[sp.fit], FieldMode::None, 0.0, lv, 11_000);
            let r = &o[0].res;
            println!(
                "| {} | {lv} | {} | {} |",
                sp.name,
                (r.ber * (r.raw * TRIALS) as f64).round() as usize,
                if r.snr_db.is_finite() { format!("{:.1e}", 10f64.powf(-r.snr_db / 20.0)) } else { "0".into() }
            );
        }
        for (nm, map) in [("GLOBAL-HG §5.1", Mapping::Luma), ("GLOBAL-HG §5.1-CL", Mapping::ConstLuma)] {
            let s = HgSetup::new(&cph, map, w, POLY_N, lv, false); // политика «масштаб»
            let o = eval_hg(&s, &cph, map, FieldMode::None, 0.0, 11_100, 4);
            println!(
                "| {nm} | {lv} | {} | {} |",
                (o.res.ber * (o.res.raw * 4) as f64).round() as usize,
                if o.res.snr_db.is_finite() { format!("{:.1e}", 10f64.powf(-o.res.snr_db / 20.0)) } else { "0".into() }
            );
        }
    }
    println!(
        "\nВ воротах политика драйва у HG — «масштаб» (ни одного клипа), чтобы проверялся \
         тракт, а не искажение клиппера. Квантование драйва 8 бит включено во всех армах."
    );
    println!(
        "\n**Единственный арм, НЕ проходящий ворота, — box±2**, и это не баг, а его свойство: \
         локальное box-среднее по 25 клеткам оценивает поле по САМИМ ДАННЫМ, а среднее 25 \
         нуль-средних отсчётов имеет СКО ≈ 1/5 амплитуды данных. Даже при полном отсутствии \
         поля это даёт ~20% ложной «поправки». При 1–3 битах яркости (там, где схема и \
         применяется) это терпимо, при 4-PAM — уже нет. Остальные армы восстанавливают \
         нагрузку ТОЧНО при обоих созвездиях."
    );
    println!();

    // ---------------- 2. валидация модели шума ----------------
    println!("## 2. Валидация модели шума: замеренные 1.79 кода/клетку получаются усреднением");
    let (nrow, trow) = measure_cell_noise(&ph);
    let wcols: Vec<String> = WINDOWS.iter().map(|w| format!("{w}×{w} = {} px", w * w)).collect();
    println!("{}", head("окно усреднения клетки", &wcols));
    println!("{}", row("σ кода на клетку (измерено в модели)", &nrow));
    println!("{}", row("6.15/окно (теория)", &trow));
    println!(
        "\nЗамеренная на телефоне медиана **1.79 кода/клетку** ложится ровно на 6.15/√12 = 1.78 \
         — те самые 7–16 px, что усредняет shipped-демодулятор. Модель шума задана на \
         ПИКСЕЛЬНОМ уровне и воспроизводит замер; полное окно {PX_CELL}×{PX_CELL} даёт 0.51 \
         кода, и этот бесплатный выигрыш в сравнении ОТДАН per-cell (арм получает лучшее окно)."
    );
    println!();

    // ---------------- 3. остаточная ошибка поля ----------------
    println!("## 3. МЕХАНИЗМ: остаточная ошибка поля после СОБСТВЕННОЙ коррекции арма (σ=1)");
    println!(
        "RMS(f̂/f − 1) по клеткам и каналам, %. f — ИСТИННОЕ поле в ДРАЙВ-домене, \
         f = λ^(1/γ_дисплея): гамма ISP сокращается точно (см. `Phys::drive_field`)."
    );
    let fmodes = [FieldMode::Achromatic, FieldMode::Chromatic];
    let fcols: Vec<String> = fmodes.iter().map(|f| f.label().to_string()).collect();
    println!("{}", head("арм \\ поле", &fcols));
    for (name, map, pilots, fit) in [
        ("PER-CELL naive (f̂≡1)", Mapping::Luma, None, FieldFit::Flat),
        ("PER-CELL box±2 (shipped)", Mapping::Luma, None, FieldFit::Box(2)),
        ("PER-CELL affine+DFB", Mapping::Luma, None, FieldFit::Affine),
        ("PER-CELL biquad+DFB", Mapping::Luma, None, FieldFit::Quad),
        ("PILOTS 6×6 affine", Mapping::Luma, Some(4), FieldFit::Affine),
        ("PILOTS 6×6 biquad", Mapping::Luma, Some(4), FieldFit::Quad),
        ("PILOTS 4×4 biquad", Mapping::Luma, Some(6), FieldFit::Quad),
    ] {
        let cells: Vec<String> = fmodes
            .iter()
            .map(|&fm| {
                let o = eval_percell(&ph, map, pilots, &[fit], fm, 1.0, 4, 13_000);
                format!("{:.2}", o[0].res.field_res)
            })
            .collect();
        println!("{}", row(name, &cells));
    }
    {
        let s = HgPair::new(&ph, Mapping::Luma, w, POLY_N, 4);
        let cells: Vec<String> = fmodes
            .iter()
            .map(|&fm| format!("{:.2}", s.eval(&ph, Mapping::Luma, fm, 1.0, 13_100, TRIALS).0.res.field_res))
            .collect();
        println!("{}", row("**GLOBAL-HG (низкие моды, biquad)**", &cells));
        let sa = HgPair::new(&ph, Mapping::Luma, w, 3, 4);
        let cells: Vec<String> = fmodes
            .iter()
            .map(|&fm| format!("{:.2}", sa.eval(&ph, Mapping::Luma, fm, 1.0, 13_200, TRIALS).0.res.field_res))
            .collect();
        println!("{}", row("GLOBAL-HG (низкие моды, только affine)", &cells));
    }
    {
        let cells: Vec<String> = fmodes
            .iter()
            .map(|&fm| format!("{:.2}", field_residual_common_removed(&true_cell_field(&ph, fm))))
            .collect();
        println!("{}", row("PER-CELL §5.1-CL (инвариант)", &cells));
    }
    println!(
        "\nСтрока §5.1-CL — не оценка, а то, что отображению ПОСТОЯННОЙ ЯРКОСТИ принципиально \
         НЕ мешает: общий множитель сокращается в обращении точно, остаётся только \
         поканальный дифференциал. Оценивать там нечего — и это её главное свойство."
    );
    println!();

    // ---------------- 4/5. головные развёртки по уровням созвездия ----------------
    let scols: Vec<String> = SIGMAS.iter().map(|s| format!("σ={s}")).collect();
    let acols: Vec<String> = [FieldMode::None, FieldMode::Achromatic, FieldMode::Chromatic]
        .iter().map(|f| f.label().to_string()).collect();
    let mut verdict: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new();

    for &lv in &LEVELS {
        let bpa = (lv as f64).log2() as usize;
        let hgl = HgPair::new(&ph, Mapping::Luma, w, POLY_N, lv);
        let hgc = HgPair::new(&ph, Mapping::ConstLuma, w, POLY_N, lv);
        println!(
            "## 4.{bpa} ГОЛОВНАЯ: доставленные биты/символ vs блюр σ — {lv} уровня/ось \
             ({} бита/степень свободы), поле ахроматическое",
            2 * bpa
        );
        println!(
            "«Доставлено» = биты осей с ПРЕДСКАЗАННЫМ BER < 1e-2 (SNR снимается EVM-оценкой по \
             {TRIALS} попыткам, порог по L-PAM; при двух уровнях это SNR > {:.1} дБ). Сырьё: \
             per-cell {} бит, pilots {} бит (площадь пилотов вычтена), HG {} бит (6 мод отданы \
             под калибровку).",
            to_db(SNR_MIN),
            CELLS * CELLS * 2 * bpa,
            (CELLS * CELLS - 36) * 2 * bpa,
            (NMODES - CAL_MODES.len()) * 2 * bpa
        );
        println!("{}", head("арм \\ σ (camera-px)", &scols));
        let mut base = vec![0usize; SIGMAS.len()];
        let mut wins: Vec<(String, Vec<usize>)> = Vec::new();
        for sp in &SPECS {
            let mut cells = Vec::new();
            let mut ws = Vec::new();
            for (i, &s) in SIGMAS.iter().enumerate() {
                let o = eval_percell(&ph, sp.map, sp.pilots, &[sp.fit], FieldMode::Achromatic, s, lv, 14_000 + i);
                cells.push(format!("{}", o[0].res.delivered));
                ws.push(o[0].win);
                base[i] = base[i].max(o[0].res.delivered);
            }
            println!("{}", row(sp.name, &cells));
            wins.push((sp.name.to_string(), ws));
        }
        let hgr: Vec<(HgOut, &str)> = SIGMAS
            .iter()
            .enumerate()
            .map(|(i, &s)| hgl.eval(&ph, Mapping::Luma, FieldMode::Achromatic, s, 14_500 + 2 * i, TRIALS))
            .collect();
        let hgs: Vec<&HgOut> = hgr.iter().map(|(o, _)| o).collect();
        println!("{}", row("**GLOBAL-HG §5.1**", &hgs.iter().map(|o| format!("{}", o.res.delivered)).collect::<Vec<_>>()));
        let hgcr: Vec<(HgOut, &str)> = SIGMAS
            .iter()
            .enumerate()
            .map(|(i, &s)| hgc.eval(&ph, Mapping::ConstLuma, FieldMode::Achromatic, s, 14_700 + 2 * i, TRIALS))
            .collect();
        println!("{}", row("GLOBAL-HG §5.1-CL", &hgcr.iter().map(|(o, _)| format!("{}", o.res.delivered)).collect::<Vec<_>>()));
        println!("{}", row("средний SNR HG §5.1, дБ", &hgs.iter().map(|o| format!("{:.1}", o.res.snr_db)).collect::<Vec<_>>()));
        println!("{}", row("политика драйва HG §5.1", &hgr.iter().map(|(_, l)| l.to_string()).collect::<Vec<_>>()));
        println!("{}", row("клип драйва HG §5.1, %", &hgs.iter().map(|o| format!("{:.3}", 100.0 * o.clip_frac)).collect::<Vec<_>>()));
        println!();
        println!("Выбранное окно усреднения клетки (арм получает ЛУЧШЕЕ на каждой точке):");
        for (n, ws) in &wins {
            println!(
                "- {n}: {}",
                ws.iter().zip(SIGMAS.iter()).map(|(w, s)| format!("σ{s}→{w}px")).collect::<Vec<_>>().join(", ")
            );
        }
        println!();

        println!("## 5.{bpa} Абляция поля при σ=1 — {lv} уровня/ось");
        println!("{}", head("арм \\ поле", &acols));
        for sp in &SPECS {
            let cells: Vec<String> = [FieldMode::None, FieldMode::Achromatic, FieldMode::Chromatic]
                .iter()
                .enumerate()
                .map(|(i, &fm)| {
                    format!("{}", eval_percell(&ph, sp.map, sp.pilots, &[sp.fit], fm, 1.0, lv, 15_000 + i)[0].res.delivered)
                })
                .collect();
            println!("{}", row(sp.name, &cells));
        }
        let cells: Vec<String> = [FieldMode::None, FieldMode::Achromatic, FieldMode::Chromatic]
            .iter()
            .enumerate()
            .map(|(i, &fm)| format!("{}", hgl.eval(&ph, Mapping::Luma, fm, 1.0, 15_500 + 2 * i, TRIALS).0.res.delivered))
            .collect();
        println!("{}", row("**GLOBAL-HG §5.1**", &cells));
        println!();

        verdict.push((lv, base, hgs.iter().map(|o| o.res.delivered).collect()));
    }

    // ---------------- 6. контур и гамма ----------------
    println!("## 6. Контур «линеаризация → проекция → фит поля → перепроекция»");
    println!(
        "**Главное, что должен знать читатель:** при СОГЛАСОВАННЫХ гаммах курицы-яйца НЕТ. \
         Гамма ISP сокращается аналитически: 255·(code/255)^(γ_isp/γ_disp) = d·λ^(1/γ_disp). \
         Поле остаётся ЧИСТО мультипликативным в драйв-домене, и линеаризация не требует \
         знания усиления. Итерация нужна лишь для вычитания утечки ДАННЫХ в низкие моды — \
         и делать её приходится по ЖЁСТКИМ решениям: с мягкими контур вырождается в \
         тождество (soft-восстановление воспроизводит ровно то, что мы объясняем полем)."
    );
    let hgl4 = HgPair::new(&ph, Mapping::Luma, w, POLY_N, 4);
    let (hgt, pol6) = hgl4.eval(&ph, Mapping::Luma, FieldMode::Chromatic, 1.0, 16_000, TRIALS);
    let icols: Vec<String> = (0..hgt.trace.len()).map(|i| format!("итер {i}")).collect();
    println!("{}", head("остаточная ошибка поля, %", &icols));
    println!("{}", row("GLOBAL-HG (хром. поле, σ=1)", &hgt.trace.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>()));
    let conv = hgt.trace.windows(2).position(|w| (w[1] - w[0]).abs() < 0.02).map(|i| i + 1);
    println!(
        "\nСходимость: {}",
        match conv {
            Some(i) => format!("**сходится**, стабилизируется на итерации {i} (шаг < 0.02 п.п.)"),
            None => "**НЕ стабилизировалась** за отведённые итерации".into(),
        }
    );
    println!("\n**Рассогласованная тон-кривая** — вот где курица-яйцо настоящая:");
    println!("| ошибка показателя линеаризации | остат. поле HG, % | доставлено HG | доставлено PER-CELL biquad |");
    println!("|---|---|---|---|");
    for (i, &k) in [0.9f64, 0.95, 1.0, 1.05, 1.1].iter().enumerate() {
        let mut q = ph;
        q.rx_expo_scale = k;
        let hgm = eval_hg(hgl4.by_label(pol6), &q, Mapping::Luma, FieldMode::Achromatic, 1.0, 17_000 + i, 8);
        let pc = eval_percell(&q, Mapping::Luma, None, &[FieldFit::Quad], FieldMode::Achromatic, 1.0, 4, 17_500 + i);
        println!(
            "| {:+.0}% | {:.2} | {} | {} |",
            (k - 1.0) * 100.0, hgm.res.field_res, hgm.res.delivered, pc[0].res.delivered
        );
    }
    println!();

    // ---------------- 7. полезный порядок мод и оценка блюра ----------------
    println!("## 7. Полезный порядок мод vs σ и оценка блюра по аттенюации ψ");
    println!(
        "Политика драйва здесь ФИКСИРОВАНА на «масштаб» (ни одного клипа): измеряется \
         чистая физика аттенюации мод, а не искажение клиппера."
    );
    let hg7 = HgPair::new(&ph, Mapping::Luma, w, POLY_N, 4);
    let hgs4: Vec<HgOut> = SIGMAS
        .iter()
        .enumerate()
        .map(|(i, &s)| eval_hg(&hg7.scale, &ph, Mapping::Luma, FieldMode::Achromatic, s, 20_000 + i, TRIALS))
        .collect();
    let ocols: Vec<String> = SIGMAS.iter().map(|s| format!("σ={s}")).collect();
    println!("{}", head("порядок m+n \\ σ", &ocols));
    for o in 0..14usize {
        let cells: Vec<String> = hgs4
            .iter()
            .map(|h| {
                if h.order_n[o] == 0 {
                    "калибр.".into()
                } else if h.order_snr[o].is_finite() {
                    format!("{:.0}", h.order_snr[o])
                } else {
                    "∞".into()
                }
            })
            .collect();
        println!("{}", row(&format!("SNR дБ, порядок {o}"), &cells));
    }
    println!("\n(Порядки 0–2 целиком отданы под калибровку — данных там нет по построению.)");
    println!();
    let top: Vec<String> = hgs4
        .iter()
        .map(|h| match (0..2 * MMAX + 1).rev().find(|&o| h.order_bits[o] > 0) {
            Some(o) => format!("{o}"),
            None => "—".into(),
        })
        .collect();
    println!("{}", head("метрика \\ σ", &ocols));
    println!("{}", row("старший порядок с доставленными битами", &top));

    // Проектная калибровочная кривая снимается на ЭТАЛОННОМ канале (полный шум и
    // кросстолк) БЕЗ поля — так её и снимали бы в лаборатории. Измерения ниже
    // берутся на том же канале с полем и без, чтобы конфаунд был виден отдельно.
    let cal_sigmas = [0.0f64, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
    let cal: Vec<HgOut> = cal_sigmas
        .iter()
        .enumerate()
        .map(|(i, &s)| eval_hg(&hg7.scale, &ph, Mapping::Luma, FieldMode::None, s, 18_000 + i, 6))
        .collect();
    let nofld: Vec<HgOut> = SIGMAS
        .iter()
        .enumerate()
        .map(|(i, &s)| eval_hg(&hg7.scale, &ph, Mapping::Luma, FieldMode::None, s, 21_000 + i, TRIALS))
        .collect();
    let t_raw: Vec<(f64, f64)> = cal_sigmas.iter().zip(&cal).map(|(&s, c)| (s, c.rho_raw / cal[0].rho_raw)).collect();
    let t_fin: Vec<(f64, f64)> = cal_sigmas.iter().zip(&cal).map(|(&s, c)| (s, c.rho_fin / cal[0].rho_fin)).collect();
    println!("{}", row("истинная σ", &SIGMAS.iter().map(|s| format!("{s}")).collect::<Vec<_>>()));
    println!("{}", row("**σ̂, ПОЛЯ НЕТ, до коррекции**", &nofld.iter().map(|h| format!("{:.2}", invert_cal(&t_raw, h.rho_raw / cal[0].rho_raw))).collect::<Vec<_>>()));
    println!("{}", row("ρ_отн, поля нет", &nofld.iter().map(|h| format!("{:.3}", h.rho_raw / cal[0].rho_raw)).collect::<Vec<_>>()));
    println!("{}", row("σ̂, поле ЕСТЬ, до коррекции", &hgs4.iter().map(|h| format!("{:.2}", invert_cal(&t_raw, h.rho_raw / cal[0].rho_raw))).collect::<Vec<_>>()));
    println!("{}", row("ρ_отн, поле есть, до коррекции", &hgs4.iter().map(|h| format!("{:.3}", h.rho_raw / cal[0].rho_raw)).collect::<Vec<_>>()));
    println!("{}", row("σ̂, поле есть, ПОСЛЕ коррекции", &hgs4.iter().map(|h| format!("{:.2}", invert_cal(&t_fin, h.rho_fin / cal[0].rho_fin))).collect::<Vec<_>>()));
    println!("{}", row("ρ_отн, поле есть, после коррекции", &hgs4.iter().map(|h| format!("{:.3}", h.rho_fin / cal[0].rho_fin)).collect::<Vec<_>>()));
    println!(
        "\n**Три строки — три разных вывода.** (1) БЕЗ поля механизм РАБОТАЕТ: ρ монотонно \
         падает с σ, и σ̂ идёт за истинной σ — гауссова аттенюация мод по m+n реальна и \
         читаема. (2) С полем ДО коррекции ρ уезжает: наивно линеаризованный драйв несёт \
         член (f−1)·M/A_L, а он ЧЁТНЫЙ по обеим осям и садится ровно на ψ00/ψ20/ψ02 — то есть \
         поле подделывает ту же наблюдаемую, по которой мы хотели читать блюр. (3) ПОСЛЕ \
         коррекции ρ залипает на 1.000 при ЛЮБОЙ σ: фит поля с квадратичными членами X², Y² \
         СЪЕДАЕТ гауссову аттенюацию низких мод целиком — параметры АЛИАСИРОВАНЫ. Шести \
         низких мод хватает либо на поле, либо на блюр, но не на оба сразу."
    );
    println!("\nПроектная калибровочная кривая ρ_отн(σ), эталонный канал БЕЗ поля:");
    println!("{}", head("σ", &cal_sigmas.iter().map(|s| format!("{s}")).collect::<Vec<_>>()));
    println!("{}", row("ρ_отн ДО коррекции", &t_raw.iter().map(|(_, r)| format!("{r:.3}")).collect::<Vec<_>>()));
    println!("{}", row("ρ_отн ПОСЛЕ коррекции", &t_fin.iter().map(|(_, r)| format!("{r:.3}")).collect::<Vec<_>>()));
    println!();

    // ---------------- 8. усечение ----------------
    println!("## 8. Усечение: чем оплачена «глобальность» низких мод (развёртка по w)");
    println!(
        "| w/w_fit | w, px | max\\|Грам вне-диаг\\| (RX) | κ_∞ (1-D) | охват ψ≤2, % полуширины | \
         остат. поле, % | доставлено (σ=1, 4 уровня) |"
    );
    println!("|---|---|---|---|---|---|---|");
    for &k in &[0.7f64, 1.0, 1.3, 1.6] {
        let s = HgPair::new(&ph, Mapping::Luma, w * k, POLY_N, 4);
        let o = s.eval(&ph, Mapping::Luma, FieldMode::Achromatic, 1.0, 19_000 + 2 * (k * 10.0) as usize, 8).0;
        println!(
            "| {k:.1} | {:.1} | {:.2e} | {:.1} | {:.0} | {:.2} | {} |",
            s.clip.w, s.clip.rx.gram_max_off, s.clip.rx.cond,
            100.0 * s.clip.w * 5.0f64.sqrt() / (APER as f64 / 2.0),
            o.res.field_res, o.res.delivered
        );
    }
    println!();

    // ---------------- 9. вердикт ----------------
    println!("## 9. Вердикт");
    for (lv, base, hgv) in &verdict {
        println!("\n**{lv} уровня/ось ({} бита/степень свободы):**\n", 2 * (*lv as f64).log2() as usize);
        println!("| σ | лучший per-cell/pilots | GLOBAL-HG | Δ бит | Δ бит/с |");
        println!("|---|---|---|---|---|");
        for (i, &s) in SIGMAS.iter().enumerate() {
            let d = hgv[i] as i64 - base[i] as i64;
            println!("| {s} | {} | {} | {d:+} | {:+.0} |", base[i], hgv[i], d as f64 * FPS);
        }
    }
    println!("\nбит/с = доставленные биты × {FPS} кадр/с (60 Гц / hold 6 периодов, §6.3).");
    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

/// Измеренная в модели σ кода на клетку по окнам усреднения против теории
/// 6.15/окно — прямая проверка того, что шум задан на ПИКСЕЛЬНОМ уровне.
fn measure_cell_noise(ph: &Phys) -> (Vec<String>, Vec<String>) {
    let cells = vec![C { re: 0.0, im: 0.0 }; CELLS * CELLS];
    let drive = render_percell(&cells, ph, Mapping::Luma);
    let mut codes: Vec<Image> = Vec::new();
    for t in 0..8 {
        let mut rng = Rng::new(seed_for(12_000, t));
        codes.push(channel(&drive, ph, Mapping::Luma, FieldMode::None, 0.0, &mut rng).0);
    }
    let mut meas = Vec::new();
    let mut theo = Vec::new();
    for &win in &WINDOWS {
        let off = (PX_CELL - win) / 2;
        let inv = 1.0 / (win * win) as f64;
        let mut vals = Vec::new();
        for code in &codes {
            for cy in 0..CELLS {
                for cx in 0..CELLS {
                    let mut acc = 0.0;
                    for dy in 0..win {
                        for dx in 0..win {
                            acc += code.at(cx * PX_CELL + off + dx, cy * PX_CELL + off + dy)[1] as f64;
                        }
                    }
                    vals.push(acc * inv * 255.0);
                }
            }
        }
        let mu = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / vals.len() as f64;
        meas.push(format!("{:.2}", var.sqrt()));
        theo.push(format!("{:.2}", PIX_NOISE_CODES / win as f64));
    }
    (meas, theo)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Все тесты держатся на дешёвых точках (σ = 0 ⇒ блюр вырожден в копию),
    /// чтобы `cargo test` оставался быстрым в debug-профиле.
    fn setup() -> (CalibProfile, Phys, f64) {
        let p = reference_profile();
        let ph = Phys::new(&p);
        (p, ph, w_fit())
    }

    /// Дискретный базис нормирован (диагональ Грама ровно 1), а вне-диагональ —
    /// честная мера НЕортогональности усечённого гауссиана. Пин верхней границы
    /// и обусловленности: при w = w_fit усечение остаётся управляемым.
    #[test]
    fn hg_basis_is_normalized_and_conditioned() {
        let (_p, _ph, w) = setup();
        let tx = Hg1::new(MMAX + 1, APER, 1.0, APER as f64 / 2.0, w);
        for k in 0..tx.m {
            let n: f64 = (0..tx.n).map(|i| tx.h[i * tx.m + k].powi(2)).sum();
            assert!((n - 1.0).abs() < 1e-12, "норма моды {k} = {n}");
        }
        assert!(tx.gram_max_off < 0.05, "вне-диагональ Грама {}", tx.gram_max_off);
        assert!(tx.cond < 2.0, "κ_∞ Грама {}", tx.cond);
        let rx = Hg1::from_binned(&tx, RX_BIN);
        assert_eq!(rx.n, RXN);
        assert!(rx.gram_max_off < 0.05, "вне-диагональ RX-Грама {}", rx.gram_max_off);
    }

    /// МНК-проекция обращает синтез на своей же сетке: project(synth(c)) = c.
    /// Это и есть смысл G⁻¹-множителя — наивная корреляция такого не даёт.
    #[test]
    fn hg_projection_inverts_synthesis() {
        let (_p, _ph, w) = setup();
        let b = Hg1::new(MMAX + 1, APER, 1.0, APER as f64 / 2.0, w);
        let mut rng = Rng::new(seed_for(4242, 1));
        let c: Vec<f64> = (0..NMODES).map(|_| if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 }).collect();
        let f = b.synth(&c);
        let r = b.project(&f);
        let worst = c.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-6, "невязка проекции {worst}");
    }

    /// Приёмный базис СОГЛАСОВАН с бинированием: спроецировав box-усреднённое
    /// поле, получаем те же коэффициенты. Без этого бин-фильтр давил бы старшие
    /// моды систематически и чистый канал не восстанавливался бы точно.
    #[test]
    fn binned_basis_matches_box_averaged_field() {
        let (_p, _ph, w) = setup();
        let tx = Hg1::new(MMAX + 1, APER, 1.0, APER as f64 / 2.0, w);
        let rx = Hg1::from_binned(&tx, RX_BIN);
        let mut rng = Rng::new(seed_for(4243, 2));
        let c: Vec<f64> = (0..NMODES).map(|_| if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 }).collect();
        let binned = bin_field(&tx.synth(&c));
        let r = rx.project(&binned);
        // box-усреднение линейно и РАЗДЕЛИМО, поэтому связь коэффициентов точная и
        // ПОМОДОВАЯ: c_rx[k][l] = c_tx[k][l]·N_k·N_l, где N — норма усреднённой моды.
        let n = &rx.bin_norm;
        let mut worst = 0.0f64;
        for mx in 0..=MMAX {
            for my in 0..=MMAX {
                let k = midx(mx, my);
                let want = c[k] * n[mx] * n[my];
                worst = worst.max((r[k] - want).abs());
            }
        }
        assert!(worst < 1e-9, "невязка бинированной проекции {worst}");
        // норма моды 0 падает ровно как 1/√бин (box-среднее гладкой функции)
        assert!(
            (n[0] - 1.0 / (RX_BIN as f64).sqrt()).abs() < 1e-3,
            "норма низшей моды {} != 1/√{RX_BIN}",
            n[0]
        );
        // старшие моды давятся сильнее нулевой, но при w = w_fit — слабо (их
        // локальная длина волны ~клетка, а бин вчетверо мельче)
        assert!(n[MMAX] < n[0], "бин-фильтр обязан давить старшую моду сильнее нулевой");
        assert!(n[MMAX] > 0.9 * n[0], "бин-фильтр давит старшую моду слишком сильно");
    }

    /// SANITY-ВОРОТА: σ=0, без шума, без поля, без кросстолка, согласованные γ —
    /// КАЖДЫЙ арм обязан восстановить нагрузку ТОЧНО, при обоих созвездиях.
    #[test]
    fn sanity_gate_every_arm_recovers_exactly() {
        let (_p, ph, w) = setup();
        let cph = clean_phys(&ph);
        for &lv in &LEVELS {
            let a = |m| axis_amp(m);
            for (name, map, pilots, fit) in [
                ("per-cell naive", Mapping::Luma, None, FieldFit::Flat),
                ("per-cell biquad", Mapping::Luma, None, FieldFit::Quad),
                ("pilots biquad", Mapping::Luma, Some(4), FieldFit::Quad),
                ("per-cell §5.1-CL", Mapping::ConstLuma, None, FieldFit::Flat),
            ] {
                let mut rng = Rng::new(seed_for(5150, lv));
                let (cells, ix) = draw_cells(map, pilots, lv, &mut rng);
                let drive = render_percell(&cells, &cph, map);
                let (code, clips) = channel(&drive, &cph, map, FieldMode::None, 0.0, &mut rng);
                assert_eq!(clips, 0, "{name}: созвездие вышло за диапазон драйва");
                let dr = binned_drives(&code, &cph, PX_CELL, PX_CELL);
                let (z, _f) = percell_decode(&dr, &cph, map, fit, pilots, lv, ITER_PC);
                let mut errs = 0usize;
                for (i, &(ir, ii)) in ix.iter().enumerate() {
                    if ir == usize::MAX {
                        continue; // пилот
                    }
                    errs += pam_bit_errors(ir, pam_slice(z[i].re / a(map), lv))
                        + pam_bit_errors(ii, pam_slice(z[i].im / a(map), lv));
                }
                assert_eq!(errs, 0, "{name} (lv{lv}): {errs} ошибочных бит на чистом канале");
            }
            // GLOBAL-HG: политика «масштаб» — жёсткий диапазон соблюдён без клипа
            for (name, map) in [("HG §5.1", Mapping::Luma), ("HG §5.1-CL", Mapping::ConstLuma)] {
                let s = HgSetup::new(&cph, map, w, POLY_N, lv, false);
                let mut rng = Rng::new(seed_for(5160, lv));
                let (cre, cim, ix) = draw_modes(&s, lv, &mut rng);
                let drive = s.render(&cph, map, &cre, &cim, s.g_clip);
                let (code, clips) = channel(&drive, &cph, map, FieldMode::None, 0.0, &mut rng);
                assert_eq!(clips, 0, "{name}: политика «масштаб» всё же клипнула драйв");
                let dr = binned_drives(&code, &cph, RX_BIN, RX_BIN);
                let rx = hg_decode(&s, &cph, map, &dr, MAXIT, None);
                let mut gain = vec![0.0f64; 2 * (MMAX + 1)];
                for (o, gp) in gain.iter_mut().enumerate() {
                    *gp = order_gain(&s, &rx.cre, &rx.cim, o, std::f64::consts::FRAC_1_SQRT_2, lv);
                }
                let mut errs = 0usize;
                for k in 0..NMODES {
                    if s.is_cal[k] {
                        continue;
                    }
                    let ga = (gain[morder(k)] * std::f64::consts::FRAC_1_SQRT_2).max(1e-12);
                    errs += pam_bit_errors(ix[k].0, pam_slice(rx.cre[k] / ga, lv))
                        + pam_bit_errors(ix[k].1, pam_slice(rx.cim[k] / ga, lv));
                }
                assert_eq!(errs, 0, "{name} (lv{lv}): {errs} ошибочных бит на чистом канале");
            }
        }
    }

    /// Шум задан на ПИКСЕЛЬНОМ уровне: σ клетки = 6.15/окно. Именно так замер
    /// 1.79 кода/клетку получается из 6.15 кода/пиксель усреднением ~12 px.
    #[test]
    fn pixel_noise_averages_to_measured_cell_sigma() {
        let (_p, ph, _w) = setup();
        let (meas, theo) = measure_cell_noise(&ph);
        for (m, t) in meas.iter().zip(&theo) {
            let (mv, tv) = (m.parse::<f64>().unwrap(), t.parse::<f64>().unwrap());
            assert!(
                (mv - tv).abs() / tv < 0.10,
                "σ клетки {mv} против теории {tv} — модель шума не пиксельная"
            );
        }
    }

    /// PAPR: глобальное разложение структурно теряет амплитуду под ЖЁСТКИМ
    /// диапазоном драйва, у per-cell crest ровно 1. Пин механизма — если
    /// однажды окажется иначе, значит арм тайком вышел за диапазон.
    #[test]
    fn global_expansion_pays_papr() {
        let (_p, ph, w) = setup();
        let s = HgSetup::new(&ph, Mapping::Luma, w, POLY_N, 2, true);
        let crest = hg_crest(&s, &ph, Mapping::Luma);
        assert!(crest > 3.0, "crest-фактор HG {crest} подозрительно мал");
        let pc = ph.a_l * PX_CELL as f64;
        let hg = s.g_clip * ph.a_l * std::f64::consts::FRAC_1_SQRT_2;
        assert!(hg < pc, "амплитуда HG {hg} не ниже per-cell {pc} — проверь диапазон драйва");
        assert!(s.g_scale <= s.g_clip, "политика «масштаб» должна быть не выше «клипа»");
    }

    /// Оценка поля: глобальный полиномиальный фит с решающей обратной связью
    /// снимает пандус 0.62→0.86 на порядок лучше наивной нормировки И лучше
    /// пилотной решётки той же модели — при НУЛЕВОЙ плате площадью.
    #[test]
    fn field_fit_beats_naive_and_is_not_worse_than_pilots() {
        let (_p, ph, _w) = setup();
        let naive = eval_percell(&ph, Mapping::Luma, None, &[FieldFit::Flat], FieldMode::Achromatic, 0.0, 2, 6100);
        let quad = eval_percell(&ph, Mapping::Luma, None, &[FieldFit::Quad], FieldMode::Achromatic, 0.0, 2, 6100);
        let pilot = eval_percell(&ph, Mapping::Luma, Some(4), &[FieldFit::Quad], FieldMode::Achromatic, 0.0, 2, 6101);
        assert!(naive[0].res.field_res > 10.0, "наивный остаток поля {} — поле не приложено?", naive[0].res.field_res);
        assert!(
            quad[0].res.field_res < 0.2 * naive[0].res.field_res,
            "полиномиальный фит не снял поле: {} против наивных {}",
            quad[0].res.field_res, naive[0].res.field_res
        );
        assert!(
            quad[0].res.delivered >= pilot[0].res.delivered,
            "бесплатный фит {} проиграл платящим площадью пилотам {}",
            quad[0].res.delivered, pilot[0].res.delivered
        );
    }

    /// Отображение ПОСТОЯННОЙ ЯРКОСТИ инвариантно к ОБЩЕМУ множителю поля и
    /// НЕ инвариантно к поканальному дифференциалу — ровно то, что утверждает
    /// §5.1-CL и что ограничивает область его применимости.
    #[test]
    fn const_luma_kills_common_field_but_not_chromatic() {
        let (_p, ph, _w) = setup();
        let a = field_residual_common_removed(&true_cell_field(&ph, FieldMode::Achromatic));
        let c = field_residual_common_removed(&true_cell_field(&ph, FieldMode::Chromatic));
        assert!(a < 1.0, "общий множитель не сократился: {a}%");
        assert!(c > 2.0 * a, "хроматический дифференциал обязан оставаться: {c}% против {a}%");
    }

    /// Оценка блюра по аттенюации ψ: ДО фита поля ρ монотонно падает с σ
    /// (механизм работает), ПОСЛЕ фита — залипает на единице, потому что
    /// квадратичные члены поля и гауссова аттенюация мод АЛИАСИРУЮТ друг друга.
    /// Это ключевой отрицательный результат, поэтому он запинен.
    #[test]
    fn blur_probe_is_aliased_by_the_field_fit() {
        let (_p, ph, w) = setup();
        let cph = clean_phys(&ph);
        let s = HgSetup::new(&cph, Mapping::Luma, w, POLY_N, 2, false);
        let a = eval_hg(&s, &cph, Mapping::Luma, FieldMode::None, 0.0, 6200, 2);
        let b = eval_hg(&s, &cph, Mapping::Luma, FieldMode::None, 4.0, 6201, 2);
        assert!(
            b.rho_raw < 0.98 * a.rho_raw,
            "ρ ДО коррекции не падает с блюром: {} против {}",
            b.rho_raw, a.rho_raw
        );
        assert!(
            (b.rho_fin / a.rho_fin - 1.0).abs() < 0.02,
            "ρ ПОСЛЕ коррекции неожиданно чувствителен к σ: {} против {}",
            b.rho_fin, a.rho_fin
        );
    }

    /// L-PAM сводится к обычной QPSK-оси при двух уровнях.
    #[test]
    fn pam_ber_reduces_to_qpsk_at_two_levels() {
        for &snr in &[0.5f64, 2.0, 5.4119, 20.0] {
            let a = ber_from_snr_lv(snr, 2);
            let b = q_func(snr.sqrt());
            assert!((a - b).abs() < 1e-12, "L-PAM при L=2 != Q(√snr): {a} != {b}");
        }
        // порог «доставлено»: SNR = Q⁻¹(1e-2)² даёт ровно BER 1e-2
        assert!((ber_from_snr_lv(SNR_MIN, 2) - 1e-2).abs() < 2e-4);
        // четыре уровня требуют заметно большего SNR при том же BER
        assert!(ber_from_snr_lv(SNR_MIN, 4) > 5e-2);
    }

    /// Линейная алгебра: Холецкий и Гаусс решают то, что должны.
    #[test]
    fn linear_solvers_are_correct() {
        // SPD 3×3
        let a = [4.0, 1.0, 1.0, 1.0, 3.0, 0.5, 1.0, 0.5, 2.0];
        let inv = spd_inverse(&a, 3).expect("SPD");
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[i * 3 + k] * inv[k * 3 + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-12, "A·A⁻¹[{i}][{j}] = {s}");
            }
        }
        // общая система через Гаусса с выбором
        // строки (0,2) и (1,3), правая часть (2,5): 2y=2 -> y=1; x+3y=5 -> x=2.
        // Нулевой ведущий элемент проверяет частичный выбор.
        let mut m = [0.0, 2.0, 1.0, 3.0];
        let mut b = [2.0, 5.0];
        assert!(lu_solve(&mut m, 2, &mut b));
        assert!((b[0] - 2.0).abs() < 1e-12 && (b[1] - 1.0).abs() < 1e-12, "{b:?}");
    }

    /// Оба созвездия обоих отображений укладываются в [black, white] ТОЧНО:
    /// диапазон драйва — жёсткое ограничение, а не пожелание.
    #[test]
    fn constellations_fit_the_drive_range() {
        let (_p, ph, _w) = setup();
        for map in [Mapping::Luma, Mapping::ConstLuma] {
            let a = axis_amp(map);
            for &lv in &LEVELS {
                for ir in 0..lv {
                    for ii in 0..lv {
                        let d = ph.drive(map, a * pam_level(ir, lv), a * pam_level(ii, lv));
                        for (c, &v) in d.iter().enumerate() {
                            assert!(
                                v >= ph.black - 1e-6 && v <= ph.white + 1e-6,
                                "{map:?} lv{lv} ({ir},{ii}) канал {c}: драйв {v} вне [{}, {}]",
                                ph.black, ph.white
                            );
                        }
                    }
                }
                // и хотя бы один УГОЛ упирается в границу — диапазон использован весь.
                // (§5.1 симметрична относительно M, поэтому её потолок — M+usable = 251
                // при white = 255: связывает более узкая, ЧЁРНАЯ сторона.)
                let touch = [(a, a), (a, -a), (-a, a), (-a, -a)].iter().any(|&(x, y)| {
                    ph.drive(map, x, y)
                        .iter()
                        .any(|&v| (v - ph.black).abs() < 1.0 || (v - ph.white).abs() < 1.0)
                });
                assert!(touch, "{map:?}: созвездие не дотягивается до границы диапазона");
            }
        }
    }
}


