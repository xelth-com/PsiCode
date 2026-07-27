//! [ДИАГНОСТИКА, ВРЕМЕННАЯ] Подобие против гомографии на РЕАЛЬНЫХ кадрах v1.
//!
//! Захват моделирует раскладку ПОДОБИЕМ (origin, один масштаб, угол). Снимок с
//! руки — проективный. Этот прогон берёт результат захвата, отпускает четыре
//! угла на свободу (8 степеней вместо 4) и смотрит, поднимаются ли пер-сторонние
//! корреляции. Если да — виновата модель геометрии, а не рамка.
//!
//! запуск: cargo run --release -p psicode-core --example v1fit -- <dump-dir> [кадр]

use psicode_core::acquire::{acquire_best_unfiltered, AcquireOpts, Field, PROBE_DEPTH_STRIP};
use psicode_core::zcborder::{corr_complex, side_reference, BorderSpec, Carrier};

fn v1_spec() -> BorderSpec {
    BorderSpec { n: 61, roots: [3, 1, 4, 2], carrier: Carrier::BinaryLuma }
}

fn load(dir: &str, idx: usize) -> Option<(Vec<f32>, usize, usize)> {
    let meta = std::fs::read_to_string(format!("{dir}/dump{idx}.meta")).ok()?;
    let f: Vec<usize> = meta.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    if f.len() < 3 {
        return None;
    }
    let y = std::fs::read(format!("{dir}/dump{idx}.y")).ok()?;
    let (w, h, st) = (f[0], f[1], f[2]);
    let mut v = vec![0.0f32; w * h];
    for j in 0..h {
        for i in 0..w {
            v[j * w + i] = *y.get(j * st + i).unwrap_or(&0) as f32 / 255.0;
        }
    }
    Some((v, w, h))
}

fn bilin(p: &[f32], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let fx = (x - 0.5).clamp(0.0, (w - 1) as f64);
    let fy = (y - 0.5).clamp(0.0, (h - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let g = |ix: usize, iy: usize| p[iy * w + ix] as f64;
    let top = g(x0, y0) + (g(x1, y0) - g(x0, y0)) * tx;
    let bot = g(x0, y1) + (g(x1, y1) - g(x0, y1)) * tx;
    top + (bot - top) * ty
}

/// Проективное отображение единичного квадрата в четырёхугольник [tl,tr,br,bl].
#[derive(Clone, Copy)]
struct Homo {
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64,
}

fn homo_from_quad(q: &[(f64, f64); 4]) -> Homo {
    let (x0, y0) = q[0];
    let (x1, y1) = q[1];
    let (x2, y2) = q[2];
    let (x3, y3) = q[3];
    let dx1 = x1 - x2;
    let dx2 = x3 - x2;
    let sx = x0 - x1 + x2 - x3;
    let dy1 = y1 - y2;
    let dy2 = y3 - y2;
    let sy = y0 - y1 + y2 - y3;
    let den = dx1 * dy2 - dx2 * dy1;
    let (g, h) = if den.abs() < 1e-12 {
        (0.0, 0.0)
    } else {
        ((sx * dy2 - dx2 * sy) / den, (dx1 * sy - sx * dy1) / den)
    };
    Homo {
        a: x1 - x0 + g * x1,
        b: x3 - x0 + h * x3,
        c: x0,
        d: y1 - y0 + g * y1,
        e: y3 - y0 + h * y3,
        f: y0,
        g,
        h,
    }
}

impl Homo {
    /// Клеточные координаты (0..n) -> пиксели.
    #[inline]
    fn map(&self, cu: f64, cv: f64, n: f64) -> (f64, f64) {
        let (u, v) = (cu / n, cv / n);
        let w = self.g * u + self.h * v + 1.0;
        (
            (self.a * u + self.b * v + self.c) / w,
            (self.d * u + self.e * v + self.f) / w,
        )
    }
}

/// Щупы стороны в КЛЕТОЧНЫХ координатах (повтор `acquire::side_probes`, он приватен).
fn side_probes(spec: &BorderSpec, side: usize, depth0: f64) -> Vec<(f64, f64, (f64, f64))> {
    let n = spec.n as f64;
    let mut out = Vec::new();
    for (i, re, im) in side_reference(spec, side) {
        for d in [-0.5f64, 0.5] {
            for t in [-0.3f64, 0.0, 0.3] {
                let a = i as f64 + t + 0.5;
                let depth = depth0 + d;
                let (cu, cv) = match side & 3 {
                    0 => (a, depth),
                    1 => (n - depth, a),
                    2 => (n - a, n - depth),
                    _ => (depth, n - a),
                };
                out.push((cu, cv, (re, im)));
            }
        }
    }
    out
}

fn side_corr(
    spec: &BorderSpec,
    pl: &[f32],
    w: usize,
    h: usize,
    hm: &Homo,
    side: usize,
    depth0: f64,
) -> f64 {
    let n = spec.n as f64;
    let pr = side_probes(spec, side, depth0);
    let got: Vec<(f64, f64)> = pr
        .iter()
        .map(|&(cu, cv, _)| {
            let (x, y) = hm.map(cu, cv, n);
            (bilin(pl, w, h, x, y), 0.0)
        })
        .collect();
    let want: Vec<(f64, f64)> = pr.iter().map(|&(_, _, v)| v).collect();
    corr_complex(&got, &want)
}

fn mean_sides(spec: &BorderSpec, pl: &[f32], w: usize, h: usize, q: &[(f64, f64); 4], d: f64) -> f64 {
    let hm = homo_from_quad(q);
    (0..4).map(|s| side_corr(spec, pl, w, h, &hm, s, d)).sum::<f64>() / 4.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: v1fit <dump-dir> [кадр]");
        std::process::exit(2);
    });
    let only: Option<usize> = args.get(2).and_then(|s| s.parse().ok());
    let spec = v1_spec();
    let depth = PROBE_DEPTH_STRIP;
    let opts = AcquireOpts {
        px_per_cell: (7.0, 20.0),
        probe_depth: depth,
        ..Default::default()
    };

    for idx in 0..64 {
        if let Some(k) = only {
            if idx != k {
                continue;
            }
        }
        let (pl, w, h) = match load(&dir, idx) {
            Some(t) => t,
            None => break,
        };
        let f = Field { w, h, re: &pl, im: None };
        let a = match acquire_best_unfiltered(&spec, &f, &opts) {
            Some(a) => a,
            None => {
                println!("{idx:>3}  кандидатов нет");
                continue;
            }
        };
        // канонические углы захвата
        let n = spec.n as f64;
        let mut q: [(f64, f64); 4] = a.quad.corners;
        let s0: Vec<f64> = (0..4)
            .map(|s| side_corr(&spec, &pl, w, h, &homo_from_quad(&q), s, depth))
            .collect();
        let q_sim = q;
        println!(
            "{idx:>3} ПОДОБИЕ    ср {:.3} стороны [{:.3} {:.3} {:.3} {:.3}] px/кл {:.2} углы {:?}",
            s0.iter().sum::<f64>() / 4.0,
            s0[0], s0[1], s0[2], s0[3],
            a.px_per_cell,
            q_sim.map(|p| (p.0.round() as i32, p.1.round() as i32))
        );

        // --- ПОСТОРОННЕ: каждая сторона подгоняется НЕЗАВИСИМО, 4 степени
        // (два своих конца). Проверяем, сходится ли это от подобия.
        let mut per: [[(f64, f64); 2]; 4] = [[(0.0, 0.0); 2]; 4];
        let mut per_c = [0.0f64; 4];
        for s in 0..4 {
            // концы стороны s в порядке обхода: q[s] -> q[(s+1)%4]
            let mut ends = [q_sim[s], q_sim[(s + 1) % 4]];
            let ev = |e: &[(f64, f64); 2]| -> f64 {
                // строим вырожденный квад, у которого сторона s лежит на e:
                // корреляция стороны s зависит только от её концов и нормали
                let mut c = q_sim;
                c[s] = e[0];
                c[(s + 1) % 4] = e[1];
                side_corr(&spec, &pl, w, h, &homo_from_quad(&c), s, depth)
            };
            let mut bs = ev(&ends);
            for &step in &[6.0f64, 3.0, 1.5, 0.75, 0.3, 0.1] {
                let mut guard = 0;
                loop {
                    guard += 1;
                    let mut improved = false;
                    for k in 0..2 {
                        for axis in 0..2 {
                            for sgn in [1.0f64, -1.0] {
                                let mut c = ends;
                                if axis == 0 { c[k].0 += sgn * step } else { c[k].1 += sgn * step }
                                let v = ev(&c);
                                if v > bs + 1e-7 {
                                    bs = v;
                                    ends = c;
                                    improved = true;
                                }
                            }
                        }
                    }
                    if !improved || guard >= 60 { break }
                }
            }
            per[s] = ends;
            per_c[s] = bs;
        }
        println!(
            "{idx:>3} ПО-СТОРОНАМ ср {:.3} стороны [{:.3} {:.3} {:.3} {:.3}]",
            per_c.iter().sum::<f64>() / 4.0,
            per_c[0], per_c[1], per_c[2], per_c[3]
        );
        let _ = per;

        // паттерн-поиск по восьми координатам углов
        let mut bs = mean_sides(&spec, &pl, w, h, &q, depth);
        for &step in &[4.0f64, 2.0, 1.0, 0.5, 0.25, 0.1] {
            let mut guard = 0;
            loop {
                guard += 1;
                let mut improved = false;
                for k in 0..4 {
                    for axis in 0..2 {
                        for sgn in [1.0f64, -1.0] {
                            let mut c = q;
                            if axis == 0 {
                                c[k].0 += sgn * step;
                            } else {
                                c[k].1 += sgn * step;
                            }
                            let v = mean_sides(&spec, &pl, w, h, &c, depth);
                            if v > bs + 1e-7 {
                                bs = v;
                                q = c;
                                improved = true;
                            }
                        }
                    }
                }
                if !improved || guard >= 60 {
                    break;
                }
            }
        }
        let hm = homo_from_quad(&q);
        let s1: Vec<f64> = (0..4).map(|s| side_corr(&spec, &pl, w, h, &hm, s, depth)).collect();
        // структурное отношение на подогнанном четырёхугольнике
        let mut off = 0.0;
        for s in 0..4 {
            for d in [-2.0f64, 2.0] {
                off += side_corr(&spec, &pl, w, h, &hm, s, depth + d);
            }
        }
        off /= 8.0;
        let ratio = if off > 1e-6 { bs / off } else { f64::INFINITY };
        let len = |i: usize, j: usize| {
            ((q[i].0 - q[j].0).powi(2) + (q[i].1 - q[j].1).powi(2)).sqrt() / n
        };
        println!(
            "{idx:>3} ГОМОГРАФИЯ ср {:.3} стороны [{:.3} {:.3} {:.3} {:.3}] полоса {:.2}",
            bs, s1[0], s1[1], s1[2], s1[3], ratio
        );
        println!(
            "    px/кл по сторонам верх {:.2} право {:.2} низ {:.2} лево {:.2}   углы {:?}",
            len(0, 1), len(1, 2), len(2, 3), len(3, 0),
            q.map(|p| (p.0.round() as i32, p.1.round() as i32))
        );
        // СВЕРКА: те же углы, но щупы штатного захвата (пер-сторонняя аффинная
        // полоса). Расхождение здесь означает, что читают в разных местах.
        let chk = psicode_core::acquire::score_sides(
            &spec,
            &f,
            depth,
            &psicode_core::acquire::Quad { corners: q },
        );
        println!(
            "    ШТАТНЫЙ сэмплер на ТЕХ ЖЕ углах: [{:.3} {:.3} {:.3} {:.3}]",
            chk[0], chk[1], chk[2], chk[3]
        );
    }
}
