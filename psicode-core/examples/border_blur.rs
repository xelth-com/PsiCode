//! [ИЗМЕРЕНИЕ] Удержание корреляции ЗЧ-рамки под расфокусом: v0 (инвертированное
//! двойное кольцо) против v1 (экструдированные полосы), бинарный носитель против
//! комплексного.
//!
//! Меряется ровно то, что видит детектор: клеточная сетка рендерится в пиксели,
//! размывается гауссианом с σ в КЛЕТКАХ, сэмплируется по щупам коррелятора и
//! нормированно коррелируется с эталоном. Плюс «удержание энергии» — дисперсия
//! (AC-энергия) сэмплированной рамки после блюра к дисперсии до него: это та
//! величина, по которой раньше получили 5 % у инвертированного кольца и 23 % у
//! неинвертированного.
//!
//! Окружение реалистичное: внутри символа случайный payload, снаружи —
//! случайное загромождение (тихой зоны НЕТ), иначе блюр на краю мерялся бы
//! против несуществующего серого поля.
//!
//! запуск: cargo run --release -p psicode-core --example border_blur

use psicode_core::zcborder::{
    corr_complex, render_cells, side_reference, strip_cell, BorderSpec, Carrier, RING,
};

/// Пикселей на клетку при рендере (достаточно для σ до ~2 клеток).
const CELL: usize = 16;
/// Поле загромождения вокруг символа в клетках (тихой зоны нет — это МУСОР).
const MARGIN_CELLS: usize = 6;

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
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Какую конструкцию рамки рендерим.
#[derive(Clone, Copy, PartialEq)]
enum Build {
    /// v0: внешнее кольцо — ЗЧ, внутреннее — его инверсия.
    LegacyInverted,
    /// v1: четыре полосы N×RING, оба ряда одинаковы.
    Strips,
}

/// Клеточная карта символа: комплексное значение на клетку (для бинарного
/// носителя мнимая часть нулевая). `None` — клетка не принадлежит рамке.
fn build_grid(spec: &BorderSpec, build: Build, rng: &mut XorShift64) -> Vec<(f64, f64)> {
    let n = spec.n;
    let mut g = vec![(0.0f64, 0.0f64); n * n];
    // payload внутри: случайный, как в реальном кадре
    for v in g.iter_mut() {
        *v = random_cell(spec.carrier, rng);
    }
    match build {
        Build::Strips => {
            for (idx, c) in render_cells(spec).iter().enumerate() {
                if let Some(v) = c {
                    g[idx] = *v;
                }
            }
        }
        Build::LegacyInverted => {
            // Раскладка v0: корень на сторону, внешний контур + ИНВЕРСИЯ внутрь.
            // Наивный приоритет углов «верх последним» — как в symbol.rs.
            let last = n - 1;
            let mut outer = vec![(0.0f64, 0.0f64); n * n];
            let mut is_ring = vec![false; n * n];
            let put = |o: &mut Vec<(f64, f64)>, r: &mut Vec<bool>, x: usize, y: usize, v: (f64, f64)| {
                o[y * n + x] = v;
                r[y * n + x] = true;
            };
            for side in [3usize, 2, 1, 0] {
                for i in 0..n {
                    let v = psicode_core::zcborder::strip_value(spec, side, i);
                    let (x, y) = strip_cell(side, i, 0, n);
                    put(&mut outer, &mut is_ring, x, y, v);
                }
            }
            for y in 0..n {
                for x in 0..n {
                    if is_ring[y * n + x] {
                        g[y * n + x] = outer[y * n + x];
                    }
                }
            }
            // внутреннее кольцо = инверсия примыкающей внешней клетки
            for i in 1..last {
                let inv = |v: (f64, f64)| (-v.0, -v.1);
                g[i * n + 1] = inv(outer[i * n]); // лево
                g[(last - 1) * n + i] = inv(outer[last * n + i]); // низ
                g[i * n + (last - 1)] = inv(outer[i * n + last]); // право
                g[n + i] = inv(outer[i]); // верх
            }
        }
    }
    g
}

fn random_cell(carrier: Carrier, rng: &mut XorShift64) -> (f64, f64) {
    match carrier {
        Carrier::BinaryLuma => {
            if rng.next() & 1 == 0 {
                (1.0, 0.0)
            } else {
                (-1.0, 0.0)
            }
        }
        Carrier::ComplexChroma => {
            let a = rng.unit() * core::f64::consts::TAU;
            (a.cos(), a.sin())
        }
    }
}

/// Растеризует клеточную сетку в поле (Re, Im) с полем загромождения вокруг.
fn rasterize(g: &[(f64, f64)], n: usize, carrier: Carrier, rng: &mut XorShift64) -> (Vec<(f64, f64)>, usize) {
    let side_cells = n + 2 * MARGIN_CELLS;
    let w = side_cells * CELL;
    let mut img = vec![(0.0f64, 0.0f64); w * w];
    // загромождение снаружи: случайные КРУПНЫЕ пятна (окна/текст), а не белый шум
    let mut clutter = vec![(0.0f64, 0.0f64); side_cells * side_cells];
    for c in clutter.iter_mut() {
        *c = random_cell(carrier, rng);
    }
    for cy in 0..side_cells {
        for cx in 0..side_cells {
            let inside = cx >= MARGIN_CELLS
                && cy >= MARGIN_CELLS
                && cx < MARGIN_CELLS + n
                && cy < MARGIN_CELLS + n;
            let v = if inside {
                g[(cy - MARGIN_CELLS) * n + (cx - MARGIN_CELLS)]
            } else {
                clutter[cy * side_cells + cx]
            };
            for py in 0..CELL {
                for px in 0..CELL {
                    img[(cy * CELL + py) * w + cx * CELL + px] = v;
                }
            }
        }
    }
    (img, w)
}

/// Сепарабельный гауссов блюр поля (Re, Im), clamp-to-edge. σ в ПИКСЕЛЯХ.
fn blur(img: &[(f64, f64)], w: usize, sigma_px: f64) -> Vec<(f64, f64)> {
    if sigma_px <= 0.0 {
        return img.to_vec();
    }
    let r = (3.0 * sigma_px).ceil() as isize;
    let k: Vec<f64> = (-r..=r)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma_px * sigma_px)).exp())
        .collect();
    let ks: f64 = k.iter().sum();
    let k: Vec<f64> = k.iter().map(|v| v / ks).collect();
    let cl = |i: isize| i.clamp(0, w as isize - 1) as usize;
    let mut tmp = vec![(0.0f64, 0.0f64); w * w];
    for y in 0..w {
        for x in 0..w {
            let (mut a, mut b) = (0.0, 0.0);
            for (t, &kv) in k.iter().enumerate() {
                let sx = cl(x as isize + t as isize - r);
                let p = img[y * w + sx];
                a += kv * p.0;
                b += kv * p.1;
            }
            tmp[y * w + x] = (a, b);
        }
    }
    let mut out = vec![(0.0f64, 0.0f64); w * w];
    for y in 0..w {
        for x in 0..w {
            let (mut a, mut b) = (0.0, 0.0);
            for (t, &kv) in k.iter().enumerate() {
                let sy = cl(y as isize + t as isize - r);
                let p = tmp[sy * w + x];
                a += kv * p.0;
                b += kv * p.1;
            }
            out[y * w + x] = (a, b);
        }
    }
    out
}

/// Сэмплирует щупы коррелятора и возвращает (корреляция с эталоном, AC-энергия).
///
/// `depth_probe` — на какой ГЛУБИНЕ полосы читать: v1 читает центр полосы
/// (1.0 — граница двух одинаковых рядов, максимум удержания под блюром), v0
/// вынужден читать центр ВНЕШНЕГО кольца (0.5), потому что на 1.0 у него
/// встречаются кольцо и его инверсия и остаётся ровно серое.
fn probe(
    img: &[(f64, f64)],
    w: usize,
    spec: &BorderSpec,
    depth_probe: f64,
) -> (f64, f64) {
    let mut got: Vec<(f64, f64)> = Vec::new();
    let mut want: Vec<(f64, f64)> = Vec::new();
    for side in 0..4 {
        for (i, re, im) in side_reference(spec, side) {
            // центр клетки (i, depth_probe) стороны в клеточных координатах символа
            let (cx, cy) = cell_center(side, i as f64 + 0.5, depth_probe, spec.n);
            let px = (MARGIN_CELLS as f64 + cx) * CELL as f64;
            let py = (MARGIN_CELLS as f64 + cy) * CELL as f64;
            got.push(bilin(img, w, px, py));
            want.push((re, im));
        }
    }
    let c = corr_complex(&got, &want);
    let n = got.len() as f64;
    let (mr, mi) = got.iter().fold((0.0, 0.0), |a, v| (a.0 + v.0 / n, a.1 + v.1 / n));
    let energy = got
        .iter()
        .map(|v| (v.0 - mr).powi(2) + (v.1 - mi).powi(2))
        .sum::<f64>()
        / n;
    (c, energy)
}

/// Центр точки (along, depth) стороны в клеточных координатах символа.
fn cell_center(side: usize, along: f64, depth: f64, n: usize) -> (f64, f64) {
    let nn = n as f64;
    match side & 3 {
        0 => (along, depth),
        1 => (nn - depth, along),
        2 => (nn - along, nn - depth),
        _ => (depth, nn - along),
    }
}

fn bilin(img: &[(f64, f64)], w: usize, x: f64, y: f64) -> (f64, f64) {
    let fx = (x - 0.5).clamp(0.0, (w - 1) as f64);
    let fy = (y - 0.5).clamp(0.0, (w - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(w - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let g = |ix: usize, iy: usize| img[iy * w + ix];
    let mix = |a: (f64, f64), b: (f64, f64), t: f64| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
    let top = mix(g(x0, y0), g(x1, y0), tx);
    let bot = mix(g(x0, y1), g(x1, y1), tx);
    mix(top, bot, ty)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(61);
    let sigmas = [0.0f64, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
    let arms: [(&str, Build, Carrier, f64); 3] = [
        ("v0 инверт. кольцо, бинарный  ", Build::LegacyInverted, Carrier::BinaryLuma, 0.5),
        ("v1 полосы,        бинарный  ", Build::Strips, Carrier::BinaryLuma, 1.0),
        ("v1 полосы,        комплексный", Build::Strips, Carrier::ComplexChroma, 1.0),
    ];

    println!("ЗЧ-рамка под расфокусом, N = {n}, {CELL} px/клетку, RING = {RING}.");
    println!("σ — в КЛЕТКАХ. Окружение: случайный payload внутри, загромождение снаружи, тихой зоны нет.");
    println!();
    print!("{:30}", "конструкция");
    for s in sigmas {
        print!(" σ={s:<5}");
    }
    println!();
    println!("{}", "-".repeat(30 + 7 * sigmas.len()));

    let mut retention: Vec<(String, Vec<f64>)> = Vec::new();
    for (name, build, carrier, depth_probe) in arms {
        let spec = BorderSpec {
            n,
            roots: [3, 1, 4, 2],
            carrier,
        };
        // усредняем по нескольким реализациям payload/загромождения
        let mut corr_acc = vec![0.0f64; sigmas.len()];
        let mut en_acc = vec![0.0f64; sigmas.len()];
        const TRIALS: usize = 8;
        for t in 0..TRIALS {
            let mut rng = XorShift64(0xB0_4DE7_u64.wrapping_mul(t as u64 + 1) | 1);
            let g = build_grid(&spec, build, &mut rng);
            let (img, w) = rasterize(&g, n, carrier, &mut rng);
            for (si, &s) in sigmas.iter().enumerate() {
                let b = blur(&img, w, s * CELL as f64);
                let (c, e) = probe(&b, w, &spec, depth_probe);
                corr_acc[si] += c / TRIALS as f64;
                en_acc[si] += e / TRIALS as f64;
            }
        }
        print!("{name:30}");
        for c in &corr_acc {
            print!(" {c:<6.3}");
        }
        println!();
        let e0 = en_acc[0].max(1e-12);
        retention.push((name.to_string(), en_acc.iter().map(|e| e / e0).collect()));
    }

    println!();
    println!("Удержание ЭНЕРГИИ (дисперсия сэмплов рамки / она же при σ=0), %:");
    print!("{:30}", "конструкция");
    for s in sigmas {
        print!(" σ={s:<5}");
    }
    println!();
    println!("{}", "-".repeat(30 + 7 * sigmas.len()));
    for (name, r) in &retention {
        print!("{name:30}");
        for v in r {
            print!(" {:<6.1}", v * 100.0);
        }
        println!();
    }
}
