//! Рид-Соломон над GF(32): два перемежённых кодовых слова RS(16, 8).
//!
//! Код профиля — 32 символа (160 бит). Одно RS-слово над GF(32) длиннее 31
//! символа невозможно, поэтому код собирается из двух: чётные позиции —
//! слово A, нечётные — слово B. Каждое слово: 8 информационных + 8
//! проверочных символов (скорость ровно 1/2), исправляет до 4 ошибок в
//! позициях, неизвестных декодеру (ровно то, что нужно для опечаток при
//! ручном вводе).
//!
//! Гарантии всего 32-символьного кода: любые <=4 опечатки; до 8 при
//! равномерном распределении; сплошная порча до 8 подряд идущих символов
//! (две соседние группы по 4 целиком) — перемежение делит её между A и B.
//!
//! Систематический код: первые 8 символов слова — сообщение как есть.
//! Корни порождающего полинома: alpha^0 .. alpha^7 (fcr = 0).

use crate::gf32 as gf;
use alloc::{vec, vec::Vec};

pub const N: usize = 16; // длина одного кодового слова (максимум для GF(32): 31)
pub const K: usize = 8; // информационные символы одного слова
pub const NSYM: usize = N - K; // 8 проверочных
const T: usize = NSYM / 2; // исправляем до 4 ошибок на слово

pub const MSG_LEN: usize = 2 * K; // 16 информационных символов во всём коде
pub const CODE_LEN: usize = 2 * N; // 32 символа во всём коде

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsError {
    /// Ошибок больше, чем код может исправить.
    TooManyErrors,
}

impl core::fmt::Display for RsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RsError::TooManyErrors => f.write_str("too many errors to correct"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RsError {}

/// Порождающий полином g(x) = prod_{i=0}^{NSYM-1} (x + alpha^i), старшим
/// коэффициентом вперёд (NSYM+1 коэффициентов). Считается в compile-time,
/// в рантайме кодер только читает эту константу.
const GEN: [u8; NSYM + 1] = build_generator();

const fn build_generator() -> [u8; NSYM + 1] {
    // g стартует как полином «1» (один коэффициент), домножаем на (x + alpha^i).
    let mut g = [0u8; NSYM + 1];
    g[0] = 1;
    let mut len = 1usize; // текущее число коэффициентов
    let mut i = 0;
    while i < NSYM {
        let root = gf::exp(i);
        // out = g * (x + root); старшим коэффициентом вперёд, длина len+1
        let mut out = [0u8; NSYM + 1];
        let mut a = 0;
        while a < len {
            out[a] = gf::add(out[a], g[a]); // g[a] * 1 (член x)
            out[a + 1] = gf::add(out[a + 1], gf::mul(g[a], root)); // g[a] * root
            a += 1;
        }
        g = out;
        len += 1;
        i += 1;
    }
    g
}

fn poly_eval(p: &[u8], x: u8) -> u8 {
    let mut y = 0u8;
    for &c in p {
        y = gf::add(gf::mul(y, x), c);
    }
    y
}

/// Кодирует 8 символов (значения 0..31) в одно слово из 16 символов.
/// Без аллокаций: работает на массивах фиксированного размера.
pub fn encode(msg: &[u8; K]) -> [u8; N] {
    debug_assert!(msg.iter().all(|&s| s < 32));
    // деление msg * x^NSYM на g(x), остаток -> проверочные символы
    let mut rem = [0u8; N];
    rem[..K].copy_from_slice(msg);
    for i in 0..K {
        let coef = rem[i];
        if coef != 0 {
            for j in 1..GEN.len() {
                rem[i + j] = gf::add(rem[i + j], gf::mul(GEN[j], coef));
            }
        }
    }
    let mut out = [0u8; N];
    out[..K].copy_from_slice(msg);
    out[K..].copy_from_slice(&rem[K..]);
    out
}

fn syndromes(code: &[u8; N]) -> ([u8; NSYM], bool) {
    let mut s = [0u8; NSYM];
    let mut any = false;
    for i in 0..NSYM {
        s[i] = poly_eval(code, gf::exp(i));
        any |= s[i] != 0;
    }
    (s, any)
}

/// Берлекэмп-Мэсси: локатор ошибок sigma(x), старшим коэффициентом вперёд.
fn berlekamp_massey(synd: &[u8; NSYM]) -> Vec<u8> {
    // здесь работаем младшим коэффициентом вперёд, в конце развернём
    let mut sigma = vec![1u8]; // текущий локатор
    let mut prev = vec![1u8]; // локатор на момент последнего обновления
    let mut l = 0usize; // число ошибок, покрытых sigma
    let mut m = 1usize; // шагов с последнего обновления
    let mut b = 1u8; // расхождение на момент последнего обновления

    for n in 0..NSYM {
        // расхождение d = S_n + sum_{i=1..l} sigma_i * S_{n-i}
        let mut d = synd[n];
        for i in 1..=l.min(sigma.len() - 1) {
            d = gf::add(d, gf::mul(sigma[i], synd[n - i]));
        }
        if d == 0 {
            m += 1;
        } else if 2 * l <= n {
            let old = sigma.clone();
            let coef = gf::div(d, b);
            // sigma -= coef * x^m * prev
            let mut shifted = vec![0u8; m];
            shifted.extend_from_slice(&prev);
            for (i, &c) in shifted.iter().enumerate() {
                if i < sigma.len() {
                    sigma[i] = gf::add(sigma[i], gf::mul(coef, c));
                } else {
                    sigma.push(gf::mul(coef, c));
                }
            }
            l = n + 1 - l;
            prev = old;
            b = d;
            m = 1;
        } else {
            let coef = gf::div(d, b);
            let mut shifted = vec![0u8; m];
            shifted.extend_from_slice(&prev);
            for (i, &c) in shifted.iter().enumerate() {
                if i < sigma.len() {
                    sigma[i] = gf::add(sigma[i], gf::mul(coef, c));
                } else {
                    sigma.push(gf::mul(coef, c));
                }
            }
            m += 1;
        }
    }
    sigma.reverse(); // -> старшим коэффициентом вперёд
    sigma
}

/// Ищет позиции ошибок перебором Ченя. Возвращает индексы в кодовом слове.
fn find_error_positions(sigma_hi_first: &[u8]) -> Option<Vec<usize>> {
    let errs = sigma_hi_first.len() - 1;
    if errs == 0 {
        return Some(vec![]);
    }
    if errs > T {
        return None;
    }
    let mut pos = Vec::with_capacity(errs);
    // корень X^-1 = alpha^i соответствует позиции ошибки j = N-1-i
    for i in 0..31usize {
        if poly_eval(sigma_hi_first, gf::exp(i)) == 0 {
            let j = (31 - i) % 31; // X = alpha^j — локатор
            if j >= N {
                return None; // корень указывает за пределы кодового слова
            }
            pos.push(N - 1 - j);
        }
    }
    if pos.len() != errs {
        return None; // локатор не разложился на различимые корни
    }
    Some(pos)
}

/// Форни: величины ошибок для найденных позиций.
fn correct(code: &mut [u8; N], synd: &[u8; NSYM], positions: &[usize]) {
    // полином синдромов младшим коэффициентом вперёд
    let s_lo: Vec<u8> = synd.to_vec();
    // локаторы X_k = alpha^(N-1-pos)
    let xs: Vec<u8> = positions.iter().map(|&p| gf::exp(N - 1 - p)).collect();

    // Omega(x) = S(x)*Lambda(x) mod x^NSYM, всё младшим вперёд
    let mut lambda_lo = vec![1u8];
    for &x in &xs {
        // (1 - X*x) => младшим вперёд: [1, X]
        lambda_lo = {
            let a = &lambda_lo;
            let b = [1u8, x];
            let mut out = vec![0u8; a.len() + 1];
            for (i, &ai) in a.iter().enumerate() {
                for (j, &bj) in b.iter().enumerate() {
                    out[i + j] = gf::add(out[i + j], gf::mul(ai, bj));
                }
            }
            out
        };
    }
    let mut omega = vec![0u8; NSYM];
    for i in 0..NSYM {
        for j in 0..=i.min(lambda_lo.len() - 1) {
            omega[i] = gf::add(omega[i], gf::mul(s_lo[i - j], lambda_lo[j]));
        }
    }

    // Lambda'(x): формальная производная (в GF(2^m) выживают нечётные степени)
    let mut lambda_deriv = vec![0u8; lambda_lo.len().saturating_sub(1)];
    for i in (1..lambda_lo.len()).step_by(2) {
        lambda_deriv[i - 1] = lambda_lo[i];
    }

    let eval_lo = |p: &[u8], x: u8| -> u8 {
        let mut y = 0u8;
        for &c in p.iter().rev() {
            y = gf::add(gf::mul(y, x), c);
        }
        y
    };

    for (k, &p) in positions.iter().enumerate() {
        let x_inv = gf::inv(xs[k]);
        let num = eval_lo(&omega, x_inv);
        let den = eval_lo(&lambda_deriv, x_inv);
        // fcr = 0: величина = X_k * Omega(X_k^-1) / Lambda'(X_k^-1)
        let magnitude = gf::mul(xs[k], gf::div(num, den));
        code[p] = gf::add(code[p], magnitude);
    }
}

/// Декодирует одно слово на месте. Возвращает число исправленных символов.
pub fn decode(code: &mut [u8; N]) -> Result<usize, RsError> {
    debug_assert!(code.iter().all(|&s| s < 32));
    let (synd, has_errors) = syndromes(code);
    if !has_errors {
        return Ok(0);
    }
    let sigma = berlekamp_massey(&synd);
    let positions = find_error_positions(&sigma).ok_or(RsError::TooManyErrors)?;
    if positions.is_empty() {
        return Err(RsError::TooManyErrors);
    }
    correct(code, &synd, &positions);
    // контрольная проверка: после исправления синдромы обязаны обнулиться
    let (_, still) = syndromes(code);
    if still {
        return Err(RsError::TooManyErrors);
    }
    Ok(positions.len())
}

/// Кодирует 16 информационных символов в 32: слово A из первых 8, слово B из
/// последних 8, перемежение A0 B0 A1 B1 ...
pub fn encode_pair(msg: &[u8; MSG_LEN]) -> [u8; CODE_LEN] {
    let mut a = [0u8; K];
    let mut b = [0u8; K];
    a.copy_from_slice(&msg[..K]);
    b.copy_from_slice(&msg[K..]);
    let ca = encode(&a);
    let cb = encode(&b);
    let mut out = [0u8; CODE_LEN];
    for i in 0..N {
        out[2 * i] = ca[i];
        out[2 * i + 1] = cb[i];
    }
    out
}

/// Декодирует пару перемежённых слов на месте.
/// Возвращает суммарное число исправленных символов.
pub fn decode_pair(code: &mut [u8; CODE_LEN]) -> Result<usize, RsError> {
    let mut a = [0u8; N];
    let mut b = [0u8; N];
    for i in 0..N {
        a[i] = code[2 * i];
        b[i] = code[2 * i + 1];
    }
    let fixed = decode(&mut a)? + decode(&mut b)?;
    for i in 0..N {
        code[2 * i] = a[i];
        code[2 * i + 1] = b[i];
    }
    Ok(fixed)
}

/// Информационные символы из (исправленного) 32-символьного кода.
pub fn extract_message(code: &[u8; CODE_LEN]) -> [u8; MSG_LEN] {
    let mut msg = [0u8; MSG_LEN];
    for i in 0..K {
        msg[i] = code[2 * i];
        msg[K + i] = code[2 * i + 1];
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    /// маленький детерминированный ГПСЧ для тестов
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_msg(rng: &mut Lcg) -> [u8; K] {
        let mut m = [0u8; K];
        for s in m.iter_mut() {
            *s = rng.below(32) as u8;
        }
        m
    }

    #[test]
    fn clean_roundtrip() {
        let mut rng = Lcg(1);
        for _ in 0..200 {
            let msg = random_msg(&mut rng);
            let mut code = encode(&msg);
            assert_eq!(decode(&mut code), Ok(0));
            assert_eq!(&code[..K], &msg[..]);
        }
    }

    #[test]
    fn corrects_up_to_four_errors() {
        let mut rng = Lcg(2);
        for round in 0..500 {
            let msg = random_msg(&mut rng);
            let clean = encode(&msg);
            let nerr = 1 + (round % T); // 1..=4
            let mut code = clean;
            let mut touched = std::collections::HashSet::new();
            for _ in 0..nerr {
                loop {
                    let p = rng.below(N as u64) as usize;
                    if touched.insert(p) {
                        let delta = 1 + rng.below(31) as u8; // гарантированно != 0
                        code[p] = gf::add(code[p], delta);
                        break;
                    }
                }
            }
            let fixed = decode(&mut code).expect("must correct <=4 errors");
            assert_eq!(fixed, nerr, "round {round}");
            assert_eq!(code, clean, "round {round}");
        }
    }

    #[test]
    fn five_errors_mostly_detected() {
        // 5 ошибок за пределами гарантии одного слова: код обязан либо честно
        // отказаться, либо мискорректировать (это неизбежно математически).
        // Проверяем, что молчаливая "успешная" мискоррекция редка — её
        // страхует CRC-8 уровнем выше.
        let mut rng = Lcg(3);
        let mut silent_bad = 0;
        let total = 400;
        for _ in 0..total {
            let msg = random_msg(&mut rng);
            let clean = encode(&msg);
            let mut code = clean;
            let mut touched = std::collections::HashSet::new();
            while touched.len() < T + 1 {
                let p = rng.below(N as u64) as usize;
                if touched.insert(p) {
                    let delta = 1 + rng.below(31) as u8;
                    code[p] = gf::add(code[p], delta);
                }
            }
            if let Ok(_) = decode(&mut code) {
                if code != clean {
                    silent_bad += 1;
                }
            }
        }
        // эмпирически заметно меньше половины; главное — что это не 100% и что
        // CRC добьёт остаток
        assert!(silent_bad < total / 2, "miscorrection rate suspiciously high: {silent_bad}/{total}");
    }

    fn random_msg_pair(rng: &mut Lcg) -> [u8; MSG_LEN] {
        let mut m = [0u8; MSG_LEN];
        for s in m.iter_mut() {
            *s = rng.below(32) as u8;
        }
        m
    }

    #[test]
    fn pair_roundtrip_and_extract() {
        let mut rng = Lcg(4);
        for _ in 0..200 {
            let msg = random_msg_pair(&mut rng);
            let mut code = encode_pair(&msg);
            assert_eq!(decode_pair(&mut code), Ok(0));
            assert_eq!(extract_message(&code), msg);
        }
    }

    #[test]
    fn pair_corrects_any_four_errors() {
        // <=4 ошибки в произвольных позициях: каждое слово видит <=4 — гарантия
        let mut rng = Lcg(5);
        for round in 0..500 {
            let msg = random_msg_pair(&mut rng);
            let clean = encode_pair(&msg);
            let nerr = 1 + (round % 4);
            let mut code = clean;
            let mut touched = std::collections::HashSet::new();
            for _ in 0..nerr {
                loop {
                    let p = rng.below(CODE_LEN as u64) as usize;
                    if touched.insert(p) {
                        let delta = 1 + rng.below(31) as u8;
                        code[p] = gf::add(code[p], delta);
                        break;
                    }
                }
            }
            let fixed = decode_pair(&mut code).expect("must correct <=4 errors");
            assert_eq!(fixed, nerr, "round {round}");
            assert_eq!(code, clean, "round {round}");
        }
    }

    #[test]
    fn pair_corrects_contiguous_burst_of_eight() {
        // 8 подряд испорченных символов делятся перемежением 4/4 между словами
        let mut rng = Lcg(6);
        for start in 0..=(CODE_LEN - 8) {
            let msg = random_msg_pair(&mut rng);
            let clean = encode_pair(&msg);
            let mut code = clean;
            for p in start..start + 8 {
                let delta = 1 + rng.below(31) as u8;
                code[p] = gf::add(code[p], delta);
            }
            let fixed = decode_pair(&mut code).expect("burst of 8 must be corrected");
            assert_eq!(fixed, 8, "start {start}");
            assert_eq!(code, clean, "start {start}");
        }
    }
}
