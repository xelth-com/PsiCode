//! [ДИАГНОСТИКА] Лестница масштабов для отображения постоянной яркости §5.1-CL.
//!
//! Полный символ Mode A (57×55 клеток) смешивает В ОДНОМ измерении все причины
//! отказа цвета сразу. Swatch-паттерн их разделяет: геометрия символа остаётся
//! ШТАТНОЙ (тихая зона, ЗЧ-кольцо §3.2, референсная строка §3.4, строка счётчика
//! §3.3 — детекция и калибровка приёмника работают без единой правки), а вместо
//! payload-сетки рисуется `n × n` крупных ОДНОРОДНЫХ блоков известной цветности,
//! разделённых средне-серым «рвом».
//!
//! # Зачем лестница
//! Механизмы отказа делятся на два семейства, и `n` их разводит:
//! - **не зависящие от масштаба** — белый баланс, гамма, СМЕШИВАНИЕ КАНАЛОВ
//!   (§3.4), гамут, диапазон драйва. Ломают уже при `n = 1`, где блок ~600 px
//!   поперёк: субдискретизация хромы 4:2:0 на таком блоке не видна, и
//!   пространственный шумодав ISP до середины блока не достаёт;
//! - **зависящие от масштаба** — 4:2:0, контентно-зависимое подавление хромы,
//!   ISI/блюр. При `n = 1` их нет, и они нарастают с `n`.
//!
//! # Истина
//! Цветность блока — детерминированная функция от `(n, номер кадра mod 256,
//! индекс блока)`. Младшие 8 бит номера кадра приёмник читает из строки счётчика
//! §3.3 ([`crate::symbol::read_counters`]), поэтому анализатору достаточно
//! снимка: истина восстанавливается точно, без обратного канала.
//!
//! # Два режима
//! - **углы** (`sweep = false`) — ровно четыре точки алфавита `ConstLuma1`
//!   (решётка, сжатая на [`CL_LATTICE_SCALE`]): числа напрямую сравнимы с
//!   живым payload;
//! - **развёртка** (`sweep = true`) — плотная решётка «подсолнуха» по ВСЕЙ
//!   плоскости цветности `|z| ≤ 1`, покрываемая за 256 кадров. Это превращает
//!   «сломано» в «вот такое искажение»: по паре (истина, измерение) считается
//!   смещение (белый баланс/гамма), линейная часть 2×2 (поворот и сдвиг = НЕ
//!   снятое смешивание каналов) и остаток (шум).
//!
//! Модуль под фичей `std` (как [`crate::symbol`]: нужна вещественная математика).

use crate::profile::CalibProfile;
use crate::symbol::{
    const_luma_map, render_symbol_counter, Frame, CL_LATTICE_SCALE, PAYLOAD_COLS, PAYLOAD_ROWS,
    RING,
};
use alloc::vec;
use alloc::vec::Vec;

/// Ширина серого рва вокруг блока, в клетках payload (по каждой стороне плитки).
/// Между двумя соседними блоками получается вдвое больше — 4 клетки (~48 px при
/// cell 12), чего хватает, чтобы блюр/ISI соседних блоков не доставал до тела
/// блока, но паттерн оставался плотным.
pub const SWATCH_GUARD: usize = 2;

/// Самая мелкая ступень лестницы. При `n = 16` плитка — 3×3 клетки, блок — 1×1
/// (12 px при cell 12), то есть РОВНО масштаб живой payload-клетки: именно там
/// 4:2:0 и шумодав хромы ISP обязаны себя показать. Дальше плитка перестаёт
/// вмещать ров, и блоки начали бы соприкасаться.
pub const SWATCH_MAX_N: usize = 16;

/// Период развёртки по номеру кадра: строка счётчика §3.3 несёт 8 бит, поэтому
/// истина обязана зависеть ТОЛЬКО от `frame mod 256`.
pub const SWATCH_PERIOD: u64 = 256;

/// Зерно детерминированного назначения углов созвездия.
const SWATCH_SEED: u64 = 0x5741_5443_4830_0001;
/// Золотой угол (радианы) — шаг решётки «подсолнуха».
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653_3;
/// Средне-серая нейтраль (§5.1): ров между блоками, z = 0.
const GRAY: [u8; 3] = [128; 3];

/// splitmix64 — детерминированный смеситель (тот же, что у tx `single`).
fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Число блоков паттерна.
pub fn swatch_blocks(n: usize) -> usize {
    n.max(1) * n.max(1)
}

/// Прямоугольник блока `b` в КЛЕТКАХ payload-сетки: `(pc0, pr0, cols, rows)`,
/// уже без рва. Блоки нумеруются построчно (b = row·n + col).
///
/// Плитки делят 57×55 максимально ровно; ров `guard` снимается с каждой стороны
/// плитки. На МЕЛКИХ ступенях лестницы (большие `n`) плитка перестаёт вмещать
/// полный ров, и он сжимается до `(плитка − 1) / 2`, но не исчезает: блоки
/// НИКОГДА не соприкасаются, пока плитка ≥ 3 клеток (то есть до `n = 18`).
/// Это принципиально: именно мелкие ступени вскрывают зависящие от масштаба
/// механизмы (4:2:0, шумодав хромы, ISI), и без рва их нельзя было бы отличить
/// от межблочного перетекания.
pub fn swatch_rect(n: usize, b: usize, guard: usize) -> (usize, usize, usize, usize) {
    let n = n.max(1);
    let (br, bc) = (b / n, b % n);
    let span = |i: usize, total: usize| -> (usize, usize) {
        let a = i * total / n;
        let z = (i + 1) * total / n;
        (a, z)
    };
    let (c0, c1) = span(bc, PAYLOAD_COLS);
    let (r0, r1) = span(br, PAYLOAD_ROWS);
    let shrink = |a: usize, z: usize| -> (usize, usize) {
        let w = z.saturating_sub(a);
        if w < 3 {
            // плитка меньше «блок + ров с двух сторон» — вырождено
            return (a, 0);
        }
        let g = guard.min((w - 1) / 2);
        (a + g, w - 2 * g)
    };
    let (pc0, cols) = shrink(c0, c1);
    let (pr0, rows) = shrink(r0, r1);
    (pc0, pr0, cols, rows)
}

/// Цветность `(x, y)` блока `b` кадра `frame` — ЕДИНСТВЕННЫЙ источник истины,
/// общий для передатчика и анализатора. Зависит только от `frame mod 256`.
pub fn swatch_point(n: usize, frame: u64, b: usize, sweep: bool) -> (f64, f64) {
    let n = n.max(1);
    let nb = n * n;
    let f = frame % SWATCH_PERIOD;
    if sweep {
        // решётка «подсолнуха» по диску: равномерная плотность по площади,
        // соседние индексы разнесены на золотой угол -> внутри одного кадра
        // блоки НЕ похожи друг на друга (иначе кадровый дрейф был бы неотличим
        // от смещения созвездия).
        let total = (SWATCH_PERIOD as usize) * nb;
        let g = f as usize * nb + b;
        let r = ((g as f64 + 0.5) / total as f64).sqrt();
        let th = g as f64 * GOLDEN_ANGLE;
        (r * th.cos(), r * th.sin())
    } else {
        // ровно алфавит payload ConstLuma1: четыре угла сжатой решётки
        let s = mix64(
            SWATCH_SEED
                ^ f.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (b as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
        ) & 3;
        let x = if s & 2 != 0 { 1.0 } else { -1.0 };
        let y = if s & 1 != 0 { 1.0 } else { -1.0 };
        (CL_LATTICE_SCALE * x, CL_LATTICE_SCALE * y)
    }
}

/// Все точки кадра в порядке блоков.
pub fn swatch_points(n: usize, frame: u64, sweep: bool) -> Vec<(f64, f64)> {
    (0..swatch_blocks(n))
        .map(|b| swatch_point(n, frame, b, sweep))
        .collect()
}

/// Рендер swatch-кадра. Геометрия символа — штатная (`p.cell_size_px` задаёт
/// размер клетки в display-px), строка счётчика несёт `frame & 0xFF`.
pub fn render_swatch(p: &CalibProfile, n: usize, frame: u64, sweep: bool, guard: usize) -> Frame {
    // База: обычный символ v0 с нулевым payload — берём из него ЗЧ-кольцо,
    // референсную строку §3.4 и строку счётчика §3.3 без изменений.
    let cells = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
    let mut fr = render_symbol_counter(p, &cells, (frame & 0xFF) as u8);

    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    let size = fr.size_px;
    let paint = |pc: usize, pr: usize, color: [u8; 3], rgb: &mut Vec<[u8; 3]>| {
        let px0 = (quiet + RING + pc) * cell;
        let py0 = (quiet + RING + 1 + pr) * cell;
        for dy in 0..cell {
            let row = (py0 + dy) * size + px0;
            for dx in 0..cell {
                rgb[row + dx] = color;
            }
        }
    };
    // 1. весь payload -> нейтральный ров
    for pr in 0..PAYLOAD_ROWS {
        for pc in 0..PAYLOAD_COLS {
            paint(pc, pr, GRAY, &mut fr.rgb);
        }
    }
    // 2. блоки
    let m = const_luma_map(p);
    for b in 0..swatch_blocks(n) {
        let (pc0, pr0, cols, rows) = swatch_rect(n, b, guard);
        if cols == 0 || rows == 0 {
            continue;
        }
        let (x, y) = swatch_point(n, frame, b, sweep);
        let d = m.drive(x, y);
        let color = [
            d[0].round().clamp(0.0, 255.0) as u8,
            d[1].round().clamp(0.0, 255.0) as u8,
            d[2].round().clamp(0.0, 255.0) as u8,
        ];
        for pr in pr0..pr0 + rows {
            for pc in pc0..pc0 + cols {
                paint(pc, pr, color, &mut fr.rgb);
            }
        }
    }
    fr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ChromaMode;
    use crate::symbol::{read_counters, GRID};

    fn prof(cell: u8) -> CalibProfile {
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: cell,
            frame_hold_periods: 6,
            luma_bits: 1,
            chroma_mode: ChromaMode::ConstLuma1,
            gamma_g_q: 28,
            gamma_r_delta_q: 8,
            gamma_b_delta_q: 10,
            white_level_q: 15,
            black_level_q: 2,
            noise_sigma_q: 0,
            mtf_limit_px: 6,
            torn_frames_q: 0,
            crosstalk_rg_q: 0,
            crosstalk_gb_q: 0,
            quiet_zone: 1,
            fec_overhead: 2,
        }
    }

    /// Блоки не пересекаются, лежат внутри payload и НИКОГДА не соприкасаются —
    /// на всех ступенях лестницы, включая мелкие (блок ~1 клетка).
    #[test]
    fn blocks_are_disjoint_inside_payload_and_separated() {
        for n in 1..=SWATCH_MAX_N {
            let mut used = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
            for b in 0..swatch_blocks(n) {
                let (pc0, pr0, cols, rows) = swatch_rect(n, b, SWATCH_GUARD);
                assert!(cols > 0 && rows > 0, "n {n}, блок {b} вырожден");
                assert!(pc0 + cols <= PAYLOAD_COLS, "n {n}, блок {b} за краем по X");
                assert!(pr0 + rows <= PAYLOAD_ROWS, "n {n}, блок {b} за краем по Y");
                for pr in pr0..pr0 + rows {
                    for pc in pc0..pc0 + cols {
                        let i = pr * PAYLOAD_COLS + pc;
                        assert_eq!(used[i], 0, "n {n}: блоки пересекаются в ({pc}, {pr})");
                        used[i] = 1;
                    }
                }
            }
            // ров: у любой занятой клетки соседи по диагонали из ДРУГОГО блока
            // быть не могут — проверяем расстояние между блоками по осям.
            for b in 0..swatch_blocks(n) {
                for b2 in b + 1..swatch_blocks(n) {
                    let (c0, r0, cw, rh) = swatch_rect(n, b, SWATCH_GUARD);
                    let (c2, r2, cw2, rh2) = swatch_rect(n, b2, SWATCH_GUARD);
                    let gap_x = if c0 + cw <= c2 {
                        c2 - (c0 + cw)
                    } else if c2 + cw2 <= c0 {
                        c0 - (c2 + cw2)
                    } else {
                        0
                    };
                    let gap_y = if r0 + rh <= r2 {
                        r2 - (r0 + rh)
                    } else if r2 + rh2 <= r0 {
                        r0 - (r2 + rh2)
                    } else {
                        0
                    };
                    assert!(
                        gap_x.max(gap_y) >= 2,
                        "n {n}: блоки {b} и {b2} разделены лишь {}/{} клетками",
                        gap_x,
                        gap_y
                    );
                }
            }
        }
    }

    /// Углы созвездия — ровно алфавит ConstLuma1, и все четыре встречаются.
    #[test]
    fn corner_mode_covers_the_four_constellation_points() {
        let want = CL_LATTICE_SCALE;
        let mut seen = [false; 4];
        for n in 1..=SWATCH_MAX_N {
            for frame in 0..64u64 {
                for b in 0..swatch_blocks(n) {
                    let (x, y) = swatch_point(n, frame, b, false);
                    assert!((x.abs() - want).abs() < 1e-12, "x {x}");
                    assert!((y.abs() - want).abs() < 1e-12, "y {y}");
                    seen[((x > 0.0) as usize) * 2 + (y > 0.0) as usize] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "не все углы встретились: {seen:?}");
    }

    /// Развёртка не выходит за единичный диск (гамут §5.1-CL) и покрывает его
    /// достаточно равномерно: в каждом из 4 квадрантов есть точки, и радиусы
    /// растут до ~1.
    #[test]
    fn sweep_stays_in_unit_disk_and_covers_it() {
        let n = 2usize;
        let mut quad = [0usize; 4];
        let mut rmax: f64 = 0.0;
        for frame in 0..SWATCH_PERIOD {
            for b in 0..swatch_blocks(n) {
                let (x, y) = swatch_point(n, frame, b, true);
                let r = (x * x + y * y).sqrt();
                assert!(r <= 1.0 + 1e-12, "кадр {frame} блок {b}: |z| = {r}");
                rmax = rmax.max(r);
                quad[((x > 0.0) as usize) * 2 + (y > 0.0) as usize] += 1;
            }
        }
        assert!(rmax > 0.99, "развёртка не доходит до края: rmax {rmax}");
        assert!(quad.iter().all(|&c| c > 50), "перекос по квадрантам: {quad:?}");
    }

    /// Рендер: размер как у штатного символа, счётчик кадров читается обратно,
    /// а тело блока однородно и равно ожидаемому драйву.
    #[test]
    fn render_matches_symbol_geometry_and_block_colours() {
        let p = prof(8);
        let quiet = p.quiet_zone_cells() as usize;
        let cell = p.cell_size_px as usize;
        for &(n, sweep) in &[(1usize, false), (2, true), (4, false), (8, true), (16, false)] {
            for frame in [0u64, 7, 129] {
                let fr = render_swatch(&p, n, frame, sweep, SWATCH_GUARD);
                assert_eq!(fr.size_px, (GRID + 2 * quiet) * cell);
                let m = const_luma_map(&p);
                for b in 0..swatch_blocks(n) {
                    let (pc0, pr0, cols, rows) = swatch_rect(n, b, SWATCH_GUARD);
                    let (x, y) = swatch_point(n, frame, b, sweep);
                    let d = m.drive(x, y);
                    let want = [
                        d[0].round().clamp(0.0, 255.0) as u8,
                        d[1].round().clamp(0.0, 255.0) as u8,
                        d[2].round().clamp(0.0, 255.0) as u8,
                    ];
                    for pr in [pr0, pr0 + rows - 1] {
                        for pc in [pc0, pc0 + cols - 1] {
                            let px = (quiet + RING + pc) * cell + cell / 2;
                            let py = (quiet + RING + 1 + pr) * cell + cell / 2;
                            assert_eq!(
                                fr.rgb[py * fr.size_px + px],
                                want,
                                "n {n} кадр {frame} блок {b} клетка ({pc}, {pr})"
                            );
                        }
                    }
                }
                // строка счётчика читается обратно через идеальный сэмплер
                let size = fr.size_px;
                let rgb = fr.rgb.clone();
                let g = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
                let sample = move |x: f64, y: f64| -> [f32; 3] {
                    let xi = (x.max(0.0) as usize).min(size - 1);
                    let yi = (y.max(0.0) as usize).min(size - 1);
                    let d = rgb[yi * size + xi];
                    [
                        (d[0] as f64 / 255.0).powf(g[0]) as f32,
                        (d[1] as f64 / 255.0).powf(g[1]) as f32,
                        (d[2] as f64 / 255.0).powf(g[2]) as f32,
                    ]
                };
                let map = |u: f64, v: f64| (u, v);
                let (a, z) = read_counters(&p, &map, &sample);
                assert_eq!((a as u64, z as u64), (frame & 0xFF, frame & 0xFF));
            }
        }
    }
}
