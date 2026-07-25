//! Детерминированный ГПСЧ: xorshift64* + Гаусс через Box–Muller.
//!
//! Никакого системного времени и внешней энтропии: сид каждой попытки
//! выводится из (индекс точки развёртки, индекс попытки) через [`seed_for`],
//! поэтому весь Monte Carlo воспроизводим побитно.

/// splitmix64 — финализатор для перемешивания сида в полноценное состояние.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Сид попытки из индексов точки развёртки и попытки. Разные (point, trial)
/// дают заведомо разные, хорошо перемешанные состояния; одинаковые — одинаковые.
pub fn seed_for(point_idx: usize, trial_idx: usize) -> u64 {
    let mixed = (point_idx as u64)
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add((trial_idx as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    splitmix64(mixed)
}

/// xorshift64* — быстрый воспроизводимый ГПСЧ. Состояние ненулевое по построению.
pub struct Rng {
    state: u64,
    /// закэшированный второй результат Box–Muller
    spare: Option<f64>,
}

impl Rng {
    /// Создать ГПСЧ с явным сидом (нулевой сид подменяется на константу —
    /// у xorshift нулевое состояние вырождено).
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Rng { state, spare: None }
    }

    /// Следующее 64-битное псевдослучайное число.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Равномерное [0, 1) с 53-битной мантиссой.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // старшие 53 бита -> [0,1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Равномерное целое из [0, n). При n = 0 возвращает 0.
    /// (Смещение по модулю для наших n ≤ 256 пренебрежимо мало.)
    #[inline]
    pub fn next_u32_below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as u32
    }

    /// Стандартный нормальный отсчёт N(0, 1), Box–Muller (пара кэшируется).
    pub fn gaussian(&mut self) -> f64 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        // u1 > 0, чтобы ln был определён
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::new(0xDEAD_BEEF);
        let mut b = Rng::new(0xDEAD_BEEF);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let mut diff = 0;
        for _ in 0..1000 {
            if a.next_u64() != b.next_u64() {
                diff += 1;
            }
        }
        // почти все отсчёты обязаны различаться
        assert!(diff > 990, "sequences too similar: {diff}/1000 differ");
    }

    #[test]
    fn seed_for_is_deterministic_and_distinct() {
        assert_eq!(seed_for(3, 7), seed_for(3, 7));
        assert_ne!(seed_for(3, 7), seed_for(7, 3));
        assert_ne!(seed_for(0, 0), seed_for(0, 1));
    }

    #[test]
    fn gaussian_stats_are_sane() {
        let mut r = Rng::new(12345);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for _ in 0..n {
            let g = r.gaussian();
            sum += g;
            sum2 += g * g;
        }
        let mean = sum / n as f64;
        let var = sum2 / n as f64 - mean * mean;
        // среднее ≈ 0, дисперсия ≈ 1
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }

    #[test]
    fn uniform_below_stays_in_range() {
        let mut r = Rng::new(99);
        for _ in 0..10_000 {
            let v = r.next_u32_below(32);
            assert!(v < 32);
        }
    }
}
