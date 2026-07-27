//! [ДИАГНОСТИКА] Распределение score захвата на РЕАЛЬНЫХ кадрах: отдельно там,
//! где символ ЕСТЬ, и там, где его НЕТ.
//!
//! Порог принятия имеет смысл только как ЗАЗОР между двумя этими распределениями,
//! а не как круглое число. Живой отказ (score 0.4359 при пороге 0.40 и заведомо
//! отсутствующем на экране символе) — ровно про то, что зазор не был измерен.
//!
//! Печатает по кадру: лучший score, отрыв ориентации, пер-сторонние корреляции,
//! структурное отношение полосы, масштаб и положение.
//!
//! запуск: cargo run --release -p psicode-core --example acq_diag -- <dump-dir> [v1|v0]

use psicode_core::acquire::{
    accepted, acquire_best_unfiltered, AcquireOpts, Field, PROBE_DEPTH_LEGACY,
    PROBE_DEPTH_STRIP,
};
use psicode_core::zcborder::{BorderSpec, Carrier};
use std::time::Instant;

fn spec_of(kind: &str) -> (BorderSpec, f64) {
    if kind == "v0" {
        (
            BorderSpec { n: 61, roots: [1, 2, 3, 4], carrier: Carrier::BinaryLuma },
            PROBE_DEPTH_LEGACY,
        )
    } else {
        (
            BorderSpec { n: 61, roots: [3, 1, 4, 2], carrier: Carrier::BinaryLuma },
            PROBE_DEPTH_STRIP,
        )
    }
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: acq_diag <dump-dir> [v1|v0]");
        std::process::exit(2);
    });
    let kind = args.get(2).cloned().unwrap_or_else(|| "v1".into());
    let (spec, depth) = spec_of(&kind);
    // У кольца v0 стороны под расфокусом заведомо неровные, поэтому его гейт
    // ослаблен — иначе диагностика отвергала бы настоящие символы v0.
    let gate = if kind == "v0" { (0.40, 0.05, 1.0) } else { AcquireOpts::default().gate };
    let opts = AcquireOpts {
        px_per_cell: (7.0, 20.0),
        probe_depth: depth,
        gate,
        ..Default::default()
    };

    println!("{dir}  рамка {kind}");
    println!(
        "{:>4} {:>7} {:>7} {:>7} {:>29} {:>7} {:>17}",
        "кадр", "score", "отрыв", "полоса", "стороны", "px/кл", "положение"
    );
    println!("{}", "-".repeat(92));
    let mut n = 0usize;
    let mut acc = Vec::new();
    let mut best: Vec<f64> = Vec::new();
    for idx in 0..64 {
        let (luma, w, h) = match load(&dir, idx) {
            Some(t) => t,
            None => break,
        };
        n += 1;
        let f = Field { w, h, re: &luma, im: None };
        let t0 = Instant::now();
        let got = acquire_best_unfiltered(&spec, &f, &opts);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        match got {
            Some(a) => {
                if accepted(&a, opts.gate) { acc.push(a.score); }
                best.push(a.score);
                let o = a.place.origin;
                println!(
                    "{idx:>4} {:>7.4} {:>7.3} {:>7.2} [{:.3} {:.3} {:.3} {:.3}] {:>7.2} ({:>6.0},{:>6.0}) {ms:.0}мс {}",
                    a.score, a.margin, a.strip_ratio,
                    a.sides[0], a.sides[1], a.sides[2], a.sides[3],
                    a.px_per_cell, o.0, o.1,
                    if accepted(&a, opts.gate) { "ПРИНЯТ" } else { "отвергнут" }
                );
            }
            None => println!("{idx:>4} {:>7} {ms:.0}мс", "—"),
        }
    }
    println!("{}", "-".repeat(92));
    best.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if best.is_empty() {
        println!("кандидатов нет ни на одном из {n} кадров");
    } else {
        println!(
            "ПРИНЯТО {} из {n}.  Лучший кандидат (гейт обойдён): min {:.4} медиана {:.4} max {:.4}",
            acc.len(),
            best[0],
            best[best.len() / 2],
            best[best.len() - 1]
        );
    }
}
