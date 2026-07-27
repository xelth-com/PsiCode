//! [ДИАГНОСТИКА] Рабочий диапазон px/клетку у захвата рамки v1.
//!
//! Размер клетки — СВОБОДНЫЙ параметр передатчика, а расфокус камеры задан
//! оптикой и в камерных пикселях от размера клетки НЕ зависит. Поэтому диапазон
//! меряется так: канал фиксирован (перспектива живого кадра, гауссов расфокус
//! σ камерных px, поле освещённости и шум сенсора с замеренного телефона), а
//! клетка гоняется по лестнице. Печатает четыре гейтовые метрики и промах углов.
//!
//! запуск: cargo run --release -p psicode-core --example pxsweep -- [сигма_px]

use psicode_core::acquire::{
    accepted, acquire_best_unfiltered, AcquireOpts, Field, Quad, PROBE_DEPTH_STRIP,
};
use psicode_core::zcborder::{render_cells, BorderSpec, Carrier};

fn v1_spec() -> BorderSpec {
    BorderSpec { n: 61, roots: [3, 1, 4, 2], carrier: Carrier::BinaryLuma }
}

struct Rng(u64);
impl Rng {
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
    fn gauss(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

/// Четырёхугольник с трапецией живого кадра (низ шире верха на 8.3 %),
/// вписанный так, что средняя сторона = `cell` × 61.
fn keystone_quad(cell: f64, at: (f64, f64)) -> Quad {
    let n = 61.0;
    let side = cell * n;
    // отношение низ/верх с замеренного кадра: 656/606
    let k = 656.0 / 606.0;
    let top = side * 2.0 / (1.0 + k);
    let bot = top * k;
    let cx = at.0 + side * 0.5;
    Quad {
        corners: [
            (cx - top * 0.5, at.1),
            (cx + top * 0.5, at.1),
            (cx + bot * 0.5, at.1 + side),
            (cx - bot * 0.5, at.1 + side),
        ],
    }
}

/// Обращение 3×3 (клетки -> однородные пиксели).
fn inv3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let co = |r: usize, c: usize| {
        let (r0, r1) = ((r + 1) % 3, (r + 2) % 3);
        let (c0, c1) = ((c + 1) % 3, (c + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let d = m[0][0] * co(0, 0) + m[0][1] * co(0, 1) + m[0][2] * co(0, 2);
    let mut out = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[c][r] = co(r, c) / d;
        }
    }
    out
}

/// Матрица «клетки (0..n) -> однородные пиксели» по четырём углам.
fn cell_matrix(q: &Quad, n: f64) -> [[f64; 3]; 3] {
    let [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = q.corners;
    let (dx1, dx2, sx) = (x1 - x2, x3 - x2, x0 - x1 + x2 - x3);
    let (dy1, dy2, sy) = (y1 - y2, y3 - y2, y0 - y1 + y2 - y3);
    let den = dx1 * dy2 - dx2 * dy1;
    let (g, h) = if den.abs() < 1e-9 {
        (0.0, 0.0)
    } else {
        ((sx * dy2 - dx2 * sy) / den, (dx1 * sy - sx * dy1) / den)
    };
    let iv = 1.0 / n;
    [
        [(x1 - x0 + g * x1) * iv, (x3 - x0 + h * x3) * iv, x0],
        [(y1 - y0 + g * y1) * iv, (y3 - y0 + h * y3) * iv, y0],
        [g * iv, h * iv, 1.0],
    ]
}

/// Кадр «как его видит камера»: символ по гомографии, загромождение вокруг,
/// поле освещённости, расфокус σ камерных px, шум сенсора 1.79 кода из 255.
fn render(spec: &BorderSpec, w: usize, h: usize, q: &Quad, sigma: f64, seed: u64) -> Vec<f32> {
    let n = spec.n;
    let mut rng = Rng(seed | 1);
    let border = render_cells(spec);
    let grid: Vec<f64> = (0..n * n)
        .map(|k| match border[k] {
            Some(v) => v.0,
            None => {
                if rng.next() & 1 == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
        })
        .collect();
    let inv = inv3(cell_matrix(q, n as f64));
    let mut img = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let ww = inv[2][0] * px + inv[2][1] * py + inv[2][2];
            let cu = (inv[0][0] * px + inv[0][1] * py + inv[0][2]) / ww;
            let cv = (inv[1][0] * px + inv[1][1] * py + inv[1][2]) / ww;
            let v = if ww > 0.0 && cu >= 0.0 && cv >= 0.0 && cu < n as f64 && cv < n as f64 {
                // гамма дисплея уже «снята»: белая клетка 0.88, чёрная 0.10
                if grid[cv as usize * n + cu as usize] > 0.0 { 0.88 } else { 0.10 }
            } else if (y / 11) % 3 == 0 && (x / 8) % 2 == 0 {
                0.80
            } else {
                0.14
            };
            img[y * w + x] = v as f32;
        }
    }
    // поле освещённости 0.62..0.86
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 / w as f64 - 0.45;
            let dy = y as f64 / h as f64 - 0.40;
            img[y * w + x] *= (0.62 + 0.24 * (-(dx * dx + dy * dy) * 4.0).exp()) as f32;
        }
    }
    // сепарабельный гауссиан
    if sigma > 0.01 {
        let r = (3.0 * sigma).ceil() as isize;
        let kk: Vec<f64> = (-r..=r)
            .map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp())
            .collect();
        let ks: f64 = kk.iter().sum();
        let kk: Vec<f64> = kk.iter().map(|v| v / ks).collect();
        let cl = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
        let mut tmp = img.clone();
        for y in 0..h {
            for x in 0..w {
                let mut a = 0.0;
                for (t, &kv) in kk.iter().enumerate() {
                    a += kv * img[y * w + cl(x as isize + t as isize - r, w)] as f64;
                }
                tmp[y * w + x] = a as f32;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let mut a = 0.0;
                for (t, &kv) in kk.iter().enumerate() {
                    a += kv * tmp[cl(y as isize + t as isize - r, h) * w + x] as f64;
                }
                img[y * w + x] = a as f32;
            }
        }
    }
    for v in img.iter_mut() {
        *v = (*v as f64 + rng.gauss() * (1.79 / 255.0)).clamp(0.0, 1.0) as f32;
    }
    img
}

fn main() {
    let sigma: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);
    let spec = v1_spec();
    println!("расфокус σ = {sigma:.1} камерных px, трапеция живого кадра (низ/верх 1.083)");
    println!(
        "{:>7} {:>7} {:>7} {:>7} {:>29} {:>9} {:>10}",
        "px/кл", "score", "отрыв", "полоса", "стороны", "угол,кл", "итог"
    );
    println!("{}", "-".repeat(92));
    for &cell in &[
        3.0f64, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0,
    ] {
        let side = (cell * 61.0).ceil() as usize;
        let (w, h) = (side + 200, side + 160);
        let q = keystone_quad(cell, (100.0, 80.0));
        let img = render(&spec, w, h, &q, sigma, 0xC0FF_EE00 + cell as u64);
        let f = Field { w, h, re: &img, im: None };
        // диапазон поиска — ±25 % вокруг истины: меряем ЗАХВАТ, а не стоимость
        // лестницы масштабов.
        let opts = AcquireOpts {
            px_per_cell: (cell * 0.8, cell * 1.25),
            probe_depth: PROBE_DEPTH_STRIP,
            ..Default::default()
        };
        match acquire_best_unfiltered(&spec, &f, &opts) {
            None => println!("{cell:>7.1} {:>7}", "—"),
            Some(a) => {
                let worst = (0..4)
                    .map(|k| {
                        (a.quad.corners[k].0 - q.corners[k].0)
                            .hypot(a.quad.corners[k].1 - q.corners[k].1)
                    })
                    .fold(0.0f64, f64::max);
                println!(
                    "{cell:>7.1} {:>7.4} {:>7.3} {:>7.2} [{:.3} {:.3} {:.3} {:.3}] {:>9.2} {:>10}",
                    a.score,
                    a.margin,
                    a.strip_ratio,
                    a.sides[0],
                    a.sides[1],
                    a.sides[2],
                    a.sides[3],
                    worst / a.px_per_cell,
                    if accepted(&a, opts.gate) { "ПРИНЯТ" } else { "отвергнут" }
                );
            }
        }
    }
}
