//! [ПРОВЕРКА] FNV-1a хэши отрендеренных символов v0 — сторож байт-в-байт пути.
//!
//! Печатает хэш RGB-байт `render_symbol_counter` для набора профилей. Запускается
//! на ДВУХ ревизиях (до и после правки рендерера) и сравнивается: путь v0 несёт
//! замороженный формат и живые передачи, он обязан совпасть побайтово.
//!
//! запуск: cargo run --release -p psicode-core --example render_hash

use psicode_core::profile::BorderMode;
use psicode_core::symbol::{self, PAYLOAD_COLS, PAYLOAD_ROWS};
use psicode_core::{CalibProfile, ChromaMode};

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

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

fn main() {
    let modes = [
        ("Mono", ChromaMode::Mono, 1u8),
        ("GreenOnly", ChromaMode::GreenOnly, 2),
        ("Chroma2", ChromaMode::Chroma2, 3),
        ("ConstLuma1", ChromaMode::ConstLuma1, 1),
        ("ConstLuma3", ChromaMode::ConstLuma3, 2),
    ];
    for (name, mode, luma_bits) in modes {
        for &quiet in &[0u8, 1, 3] {
            for &cell in &[8u8, 12, 16] {
                for &counter in &[0u8, 7] {
                    let p = CalibProfile {
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
                        noise_sigma_q: 12,
                        mtf_limit_px: 6,
                        torn_frames_q: 5,
                        crosstalk_rg_q: 3,
                        crosstalk_gb_q: 4,
                        quiet_zone: quiet,
                        fec_overhead: 2,
                        border: BorderMode::LegacyInverted,
                    };
                    let bits = symbol::bits_per_cell(&p);
                    let mask = if bits >= 8 { 0xFFu32 } else { (1u32 << bits) - 1 };
                    let mut rng = XorShift64(0xDEAD_BEEF ^ (cell as u64) ^ ((quiet as u64) << 8));
                    let cells: Vec<u8> = (0..PAYLOAD_COLS * PAYLOAD_ROWS)
                        .map(|_| (rng.next() as u32 & mask) as u8)
                        .collect();
                    let f = symbol::render_symbol_counter(&p, &cells, counter);
                    let flat: Vec<u8> = f.rgb.iter().flat_map(|c| c.iter().copied()).collect();
                    println!(
                        "{name:<11} quiet{quiet} cell{cell:<3} ctr{counter} size{:<5} {:016x}",
                        f.size_px,
                        fnv1a(&flat)
                    );
                }
            }
        }
    }
}
