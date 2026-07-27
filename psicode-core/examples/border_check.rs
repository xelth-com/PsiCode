//! Проверка: инвертировано ли внутреннее кольцо рамки, и совпадают ли строки
//! полосы. Печатает первые клетки верхней стороны для v0 и v1.
//!
//! v0 (§3.2 LegacyInverted): внутреннее кольцо = инверсия внешнего.
//! v1 (ExtrudedStrips): обе строки полосы ОДИНАКОВЫ.

use psicode_core::profile::BorderMode;
use psicode_core::symbol::{render_symbol, GRID};
use psicode_core::CalibProfile;

fn profile(border: BorderMode) -> CalibProfile {
    use psicode_core::ChromaMode;
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 12,
        frame_hold_periods: 6,
        luma_bits: 1,
        chroma_mode: ChromaMode::Mono,
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
        quiet_zone: 1,
        fec_overhead: 2,
        border,
    }
}

/// Яркость клетки (cx, cy) отрендеренного кадра: берём центр клетки.
fn cell_luma(f: &psicode_core::symbol::Frame, cell: usize, cx: usize, cy: usize) -> u8 {
    let quiet = f.quiet_cells;
    let px = (quiet + cx) * cell + cell / 2;
    let py = (quiet + cy) * cell + cell / 2;
    f.rgb[py * f.size_px + px][1]
}

fn dump(name: &str, border: BorderMode) {
    let p = profile(border);
    let cell = p.cell_size_px as usize;
    let cells = vec![0u8; psicode_core::symbol::PAYLOAD_COLS * psicode_core::symbol::PAYLOAD_ROWS];
    let f = render_symbol(&p, &cells);

    println!("\n=== {name} ===");
    println!("кадр {}x{} px, тихая зона {} клеток", f.size_px, f.size_px, f.quiet_cells);

    let n = 24.min(GRID);
    let row0: Vec<char> = (0..n).map(|x| if cell_luma(&f, cell, x, 0) > 128 { '#' } else { '.' }).collect();
    let row1: Vec<char> = (0..n).map(|x| if cell_luma(&f, cell, x, 1) > 128 { '#' } else { '.' }).collect();
    println!("верх, внешняя строка: {}", row0.iter().collect::<String>());
    println!("верх, внутренняя    : {}", row1.iter().collect::<String>());

    // Углы затираются порядком покраски, поэтому считаем и по всей стороне,
    // и без двух клеток с каждого края — иначе пара угловых клеток маскирует
    // то, что вся остальная сторона строго инверсна.
    let count = |lo: usize, hi: usize| {
        let mut same = 0usize;
        let mut inv = 0usize;
        for x in lo..hi {
            let a = cell_luma(&f, cell, x, 0) > 128;
            let b = cell_luma(&f, cell, x, 1) > 128;
            if a == b {
                same += 1;
            } else {
                inv += 1;
            }
        }
        (same, inv, hi - lo)
    };
    let (s_all, i_all, n_all) = count(0, GRID);
    let (s_in, i_in, n_in) = count(2, GRID - 2);
    println!("вся сторона ({n_all}):   совпадают {s_all}, инверсны {i_all}");
    println!("без углов   ({n_in}):   совпадают {s_in}, инверсны {i_in}");
    println!(
        "вердикт: {}",
        if i_in == n_in {
            "ОДНА НОРМАЛЬНАЯ + ОДНА ИНВЕРСНАЯ (кроме затёртых углов)"
        } else if s_in == n_in {
            "ДВЕ ОДИНАКОВЫЕ (выдавливание)"
        } else {
            "ни то ни другое — смешанное"
        }
    );
}

fn main() {
    dump("v0 LegacyInverted", BorderMode::LegacyInverted);
    dump("v1 ExtrudedStrips", BorderMode::ExtrudedStrips);
}
