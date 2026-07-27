//! [ИЗМЕРЕНИЕ] Захват на РЕАЛЬНЫХ снимках телефона, ПОЛНЫЙ кадр 1920×1080,
//! БЕЗ кропа: прежний блоб-конвейер (`detect::detect_symbol_acquire`) против
//! корреляционного захвата (`acquire::acquire`).
//!
//! Это и есть заявленный головной замер. Прежний путь требует ручного кропа
//! вокруг символа (см. аргумент кропа в `psicode-rx/examples/cell_noise.rs` и
//! `swatch_diag.rs`) — именно потому, что на загромождённом экране компонента
//! символа слипается с окном терминала. Здесь обоим дают ОДИН И ТОТ ЖЕ полный
//! кадр, и отдельно показывается, что прежний путь на кропе действительно
//! работает — то есть провал на полном кадре не «сломанный детектор», а его
//! зависимость от рва тихой зоны.
//!
//! ВАЖНО о том, что именно лежит в этих снимках: это символы СТАРОГО формата
//! (бинарная яркостная рамка, внутреннее кольцо — инверсия внешнего, тихая зона
//! есть). Новый ЗЧ-рендер в них снять неоткуда. Поэтому корреляционный захват
//! здесь настроен на LEGACY-рамку (`Carrier::BinaryLuma`, корни по сторонам
//! [1,2,3,4], щуп на глубине 0.5 — центр ВНЕШНЕГО кольца) и тихой зоной НЕ
//! пользуется. Тем самым проверяется ровно алгоритм захвата на реальных
//! данных; выигрыш от новой рамки (полосы + комплексный носитель) меряется
//! отдельно в `border_blur.rs` и в тестах `zcborder`.
//!
//! запуск: cargo run --release -p psicode-core --example acq_real -- <dump-dir> [кроп x,y,w,h]

use psicode_core::acquire::{acquire, AcquireOpts, Field, PROBE_DEPTH_LEGACY, PROBE_DEPTH_STRIP};
use psicode_core::detect;
use psicode_core::zcborder::{render_cells, BorderSpec, Carrier};
use std::time::Instant;

/// Рамка v1: экструдированные полосы, корни по сторонам [3,1,4,2].
fn v1_spec() -> BorderSpec {
    BorderSpec { n: 61, roots: [3, 1, 4, 2], carrier: Carrier::BinaryLuma }
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

/// Среднее по квадратному окну радиуса `r` (интегральное изображение).
fn box_mean(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut ii = vec![0.0f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0.0;
        for x in 0..w {
            row += v[y * w + x] as f64;
            ii[(y + 1) * (w + 1) + (x + 1)] = ii[y * (w + 1) + (x + 1)] + row;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            let sm = ii[y1 * (w + 1) + x1] - ii[y0 * (w + 1) + x1] - ii[y1 * (w + 1) + x0]
                + ii[y0 * (w + 1) + x0];
            out[y * w + x] = (sm / ((y1 - y0) * (x1 - x0)) as f64) as f32;
        }
    }
    out
}

/// Вкомпоновывает СИНТЕТИЧЕСКИЙ символ рамки v1 (полосы, тихой зоны нет) в
/// РЕАЛЬНЫЙ кадр на место прежнего символа. Возвращает px на клетку.
///
/// Честность построения: загромождение, поле освещённости и амплитуда контраста
/// берутся ИЗ СНИМКА (локальное среднее радиуса 30 px сохраняет блик и
/// виньетку), блюр ставится ИЗМЕРЕННЫЙ (σ = 2 камерных px). Синтетичен только
/// сам символ: снять v1 с этого телефона неоткуда — снимки старого формата.
fn implant_v1(
    luma: &mut [f32],
    w: usize,
    h: usize,
    crop: (usize, usize, usize, usize),
    seed: u64,
) -> f64 {
    let (x0, y0, cw, ch) = crop;
    let n = 61usize;
    // тот же ЭКРАННЫЙ габарит, что занимал старый символ ВМЕСТЕ с тихой зоной
    let side_px = cw.min(ch) as f64;
    let cell = side_px / n as f64;

    let mean = box_mean(luma, w, h, 30);
    let mut vals: Vec<f32> = Vec::new();
    for j in (0..ch).step_by(4) {
        for i in (0..cw).step_by(4) {
            let (xx, yy) = (x0 + i, y0 + j);
            if xx < w && yy < h {
                vals.push(luma[yy * w + xx]);
            }
        }
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let amp = ((vals[vals.len() * 95 / 100] - vals[vals.len() * 5 / 100]) / 2.0) as f64;

    let spec = v1_spec();
    let border = render_cells(&spec);
    let mut rng = XorShift64(seed | 1);
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

    let ox = x0 as f64 + (cw as f64 - side_px) / 2.0;
    let oy = y0 as f64 + (ch as f64 - side_px) / 2.0;
    let mut patch: Vec<f32> = luma.to_vec();
    for j in 0..h {
        for i in 0..w {
            let cu = (i as f64 - ox) / cell;
            let cv = (j as f64 - oy) / cell;
            if cu < 0.0 || cv < 0.0 {
                continue;
            }
            let (iu, iv) = (cu as usize, cv as usize);
            if iu >= n || iv >= n {
                continue;
            }
            let m = mean[j * w + i] as f64;
            patch[j * w + i] = (m + amp * grid[iv * n + iu]).clamp(0.0, 1.0) as f32;
        }
    }
    let sigma = 2.0f64;
    let r = (3.0 * sigma).ceil() as isize;
    let kk: Vec<f64> = (-r..=r)
        .map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let ks: f64 = kk.iter().sum();
    let kk: Vec<f64> = kk.iter().map(|v| v / ks).collect();
    let cl = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
    let mut tmp = patch.clone();
    for j in 0..h {
        for i in 0..w {
            let mut a = 0.0;
            for (t, &kv) in kk.iter().enumerate() {
                a += kv * patch[j * w + cl(i as isize + t as isize - r, w)] as f64;
            }
            tmp[j * w + i] = a as f32;
        }
    }
    for j in 0..h {
        for i in 0..w {
            let cu = (i as f64 - ox) / cell;
            let cv = (j as f64 - oy) / cell;
            if !(cu >= -1.0 && cv >= -1.0 && cu < n as f64 + 1.0 && cv < n as f64 + 1.0) {
                continue;
            }
            let mut a = 0.0;
            for (t, &kv) in kk.iter().enumerate() {
                a += kv * tmp[cl(j as isize + t as isize - r, h) * w + i] as f64;
            }
            luma[j * w + i] = a as f32;
        }
    }
    cell
}

/// Рамка старого формата так, как её видит коррелятор: бинарная яркость, корни
/// по сторонам [верх, право, низ, лево] = [1, 2, 3, 4] (`symbol::ZC_ROOTS`).
fn legacy_spec() -> BorderSpec {
    BorderSpec {
        n: 61,
        roots: [1, 2, 3, 4],
        carrier: Carrier::BinaryLuma,
    }
}

struct Dump {
    y: Vec<u8>,
    w: usize,
    h: usize,
    stride: usize,
}

fn load(dir: &str, idx: usize) -> Option<Dump> {
    let meta = std::fs::read_to_string(format!("{dir}/dump{idx}.meta")).ok()?;
    let f: Vec<usize> = meta.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    if f.len() < 3 {
        return None;
    }
    let y = std::fs::read(format!("{dir}/dump{idx}.y")).ok()?;
    Some(Dump {
        y,
        w: f[0],
        h: f[1],
        stride: f[2],
    })
}

/// Плоскость яркости [0,1] с необязательным кропом.
fn plane(d: &Dump, crop: Option<(usize, usize, usize, usize)>) -> (Vec<f32>, usize, usize) {
    let (x0, y0, w, h) = crop.unwrap_or((0, 0, d.w, d.h));
    let w = w.min(d.w.saturating_sub(x0));
    let h = h.min(d.h.saturating_sub(y0));
    let mut v = vec![0.0f32; w * h];
    for j in 0..h {
        for i in 0..w {
            v[j * w + i] = d.y[(y0 + j) * d.stride + x0 + i] as f32 / 255.0;
        }
    }
    (v, w, h)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: acq_real <dump-dir> [x,y,w,h]");
        std::process::exit(2);
    });
    let crop = args.get(2).and_then(|s| {
        let p: Vec<usize> = s.split(',').filter_map(|t| t.parse().ok()).collect();
        (p.len() == 4).then(|| (p[0], p[1], p[2], p[3]))
    });

    let spec = legacy_spec();
    // PSICODE_ACQ_THREADS=1 принудительно возвращает однопоточный путь (замеры).
    let threads = std::env::var("PSICODE_ACQ_THREADS").ok().and_then(|v| v.parse().ok());
    let opts = AcquireOpts {
        // «Ничего не знаем о дистанции»: полный рабочий диапазон приёмника.
        px_per_cell: (8.0, 17.0),
        probe_depth: PROBE_DEPTH_LEGACY,
        threads,
        // кольцо v0: стороны под расфокусом неровные, гейт ослаблен (см. acquire)
        gate: (0.40, 0.05, 1.0),
        ..Default::default()
    };

    println!("каталог {dir}");
    println!(
        "{:>4} | {:^34} | {:^34}",
        "кадр", "СТАРЫЙ блоб-конвейер (полный кадр)", "НОВЫЙ ЗЧ-захват (полный кадр)"
    );
    println!(
        "{:>4} | {:>8} {:>8} {:>7} {:>7} | {:>8} {:>8} {:>7} {:>7}",
        "", "итог", "score", "px/кл", "мс", "итог", "score", "px/кл", "мс"
    );
    println!("{}", "-".repeat(82));

    let synth = std::env::var_os("PSICODE_SYNTH_V1").is_some();
    let mut old_ok = 0usize;
    let mut new_ok = 0usize;
    let mut v1_ok = 0usize;
    let mut v1_ms = 0.0f64;
    let mut v1_score = 0.0f64;
    let mut old_ms = 0.0f64;
    let mut new_ms = 0.0f64;
    let mut frames = 0usize;

    for idx in 0..64 {
        let d = match load(&dir, idx) {
            Some(d) => d,
            None => break,
        };
        frames += 1;
        let (luma, w, h) = plane(&d, None);

        // --- старый путь: блоб-кандидаты + align-as-gate ---
        let t0 = Instant::now();
        let old = detect::detect_symbol_acquire(w, h, &luma);
        let dt_old = t0.elapsed().as_secs_f64() * 1000.0;
        old_ms += dt_old;

        // --- новый путь: корреляция ЗЧ ---
        let f = Field {
            w,
            h,
            re: &luma,
            im: None,
        };
        let t1 = Instant::now();
        let new = acquire(&spec, &f, &opts);
        let dt_new = t1.elapsed().as_secs_f64() * 1000.0;
        new_ms += dt_new;

        let (o_res, o_score, o_ppc) = match &old {
            Ok(det) => {
                old_ok += 1;
                // px/клетку из гомографии: длина верхней стороны / 61
                let c = |u: f64, v: f64| {
                    let hm = det.homography;
                    let dd = hm[2][0] * u + hm[2][1] * v + hm[2][2];
                    (
                        (hm[0][0] * u + hm[0][1] * v + hm[0][2]) / dd,
                        (hm[1][0] * u + hm[1][1] * v + hm[1][2]) / dd,
                    )
                };
                let a = c(0.0, 0.0);
                let b = c(61.0, 0.0);
                let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt() / 61.0;
                ("НАЙДЕН", det.score, len)
            }
            Err(e) => (
                match e {
                    detect::DetectError::NotFound => "нет",
                    detect::DetectError::Ambiguous => "неодн.",
                    detect::DetectError::BadInput => "вход",
                },
                0.0,
                0.0,
            ),
        };
        let (n_res, n_score, n_ppc) = match &new {
            Some(a) => {
                new_ok += 1;
                ("НАЙДЕН", a.score, a.px_per_cell)
            }
            None => ("нет", 0.0, 0.0),
        };
        let mut extra = String::new();
        if synth {
            if let Some(cr) = crop {
                let (mut l2, w2, h2) = plane(&d, None);
                let cell = implant_v1(&mut l2, w2, h2, cr, 0x5171_0000 + idx as u64);
                let f2 = Field { w: w2, h: h2, re: &l2, im: None };
                let o2 = AcquireOpts {
                    px_per_cell: (8.0, 17.0),
                    probe_depth: PROBE_DEPTH_STRIP,
                    threads,
                    seed_min: std::env::var("PSICODE_SEED_MIN").ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(AcquireOpts::default().seed_min),
                    ..Default::default()
                };
                let t2 = Instant::now();
                let r2 = psicode_core::acquire::acquire_best_unfiltered(&v1_spec(), &f2, &o2);
                let dt2 = t2.elapsed().as_secs_f64() * 1000.0;
                v1_ms += dt2;
                extra = match r2 {
                    Some(a) => {
                        v1_ok += 1;
                        v1_score += a.score;
                        format!(
                            " | v1 {:.3} стороны[{:.2} {:.2} {:.2} {:.2}] полоса {:.1} отрыв {:.2} {:.2}px (истина {:.2})",
                            a.score, a.sides[0], a.sides[1], a.sides[2], a.sides[3],
                            a.strip_ratio, a.margin, a.px_per_cell, cell
                        )
                    }
                    None => format!(" | v1 нет ({dt2:.0}мс)"),
                };
            }
        }
        println!(
            "{idx:>4} | {o_res:>8} {o_score:>8.3} {o_ppc:>7.2} {dt_old:>7.0} | \
             {n_res:>8} {n_score:>8.3} {n_ppc:>7.2} {dt_new:>7.0}{extra}"
        );
    }

    if frames == 0 {
        eprintln!("нет дампов в {dir}");
        std::process::exit(1);
    }
    println!("{}", "-".repeat(82));
    println!(
        "ИТОГО полный кадр 1920×1080 без кропа: старый {}/{} ({:.0} мс/кадр), новый {}/{} ({:.0} мс/кадр)",
        old_ok, frames, old_ms / frames as f64, new_ok, frames, new_ms / frames as f64
    );

    if synth {
        let avg = if v1_ok > 0 { v1_score / v1_ok as f64 } else { 0.0 };
        println!(
            "РАМКА v1 (полосы, БЕЗ тихой зоны), синтезирована в РЕАЛЬНОЙ сцене: {}/{} \
             (score {:.3} ср., {:.0} мс/кадр)",
            v1_ok, frames, avg, v1_ms / frames as f64
        );
    }

    // --- контроль: тот же старый путь, но на КРОПЕ вокруг символа ---
    if let Some(cr) = crop {
        let mut ok = 0usize;
        let mut ms = 0.0f64;
        let mut n = 0usize;
        for idx in 0..64 {
            let d = match load(&dir, idx) {
                Some(d) => d,
                None => break,
            };
            n += 1;
            let (luma, w, h) = plane(&d, Some(cr));
            let t = Instant::now();
            let r = detect::detect_symbol_acquire(w, h, &luma);
            ms += t.elapsed().as_secs_f64() * 1000.0;
            if r.is_ok() {
                ok += 1;
            }
        }
        println!(
            "КОНТРОЛЬ старый путь на РУЧНОМ КРОПЕ {:?}: {}/{} ({:.0} мс/кадр) — \
             провал выше не «сломанный детектор», а зависимость от рва тихой зоны",
            cr, ok, n, ms / n as f64
        );
    }
}
