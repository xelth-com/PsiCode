//! Арифметика GF(2^5) = GF(32).
//!
//! Примитивный полином: x^5 + x^2 + 1 (0b100101).
//! Таблицы exp/log строятся в compile-time (const fn), в рантайме — только
//! индексация массивов.

/// x^5 + x^2 + 1
const PRIM_POLY: u16 = 0b10_0101;

const fn build_tables() -> ([u8; 62], [u8; 32]) {
    let mut exp = [0u8; 62]; // удвоенная длина, чтобы mul обходился без mod 31
    let mut log = [0u8; 32];
    let mut x: u16 = 1;
    let mut i = 0;
    while i < 31 {
        exp[i] = x as u8;
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0b10_0000 != 0 {
            x ^= PRIM_POLY;
        }
        i += 1;
    }
    // дублируем, чтобы exp[log a + log b] работал без взятия остатка
    let mut j = 31;
    while j < 62 {
        exp[j] = exp[j - 31];
        j += 1;
    }
    (exp, log)
}

const TABLES: ([u8; 62], [u8; 32]) = build_tables();

#[inline]
pub const fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

#[inline]
pub const fn mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        0
    } else {
        TABLES.0[(TABLES.1[a as usize] as usize) + (TABLES.1[b as usize] as usize)]
    }
}

#[inline]
pub fn div(a: u8, b: u8) -> u8 {
    debug_assert!(b != 0, "division by zero in GF(32)");
    if a == 0 {
        0
    } else {
        let la = TABLES.1[a as usize] as isize;
        let lb = TABLES.1[b as usize] as isize;
        TABLES.0[((la - lb).rem_euclid(31)) as usize]
    }
}

#[inline]
pub fn inv(a: u8) -> u8 {
    div(1, a)
}

/// alpha^p (p может быть любым неотрицательным)
#[inline]
pub const fn exp(p: usize) -> u8 {
    TABLES.0[p % 31]
}

/// log_alpha(a); паникует на a == 0
#[inline]
pub fn log(a: u8) -> u8 {
    debug_assert!(a != 0, "log of zero in GF(32)");
    TABLES.1[a as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_axioms_exhaustive() {
        // поле маленькое — проверяем всё
        for a in 1u8..32 {
            assert_eq!(mul(a, inv(a)), 1, "a * a^-1 != 1 for a={a}");
            for b in 1u8..32 {
                let p = mul(a, b);
                assert_ne!(p, 0, "zero divisor: {a}*{b}");
                assert_eq!(div(p, b), a);
                assert_eq!(mul(a, b), mul(b, a));
                for c in 0u8..32 {
                    // дистрибутивность
                    assert_eq!(mul(a, add(b, c)), add(mul(a, b), mul(a, c)));
                }
            }
        }
    }

    #[test]
    fn exp_log_roundtrip() {
        for p in 0..31usize {
            assert_eq!(log(exp(p)) as usize, p);
        }
    }
}
