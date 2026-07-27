//! [ДИАГНОСТИКА] Анализатор ЛЕСТНИЦЫ МАСШТАБОВ §5.1-CL по живым YUV-дампам.
//!
//! Пара к `psicode-tx swatch` (см. `psicode_core::swatch`): передатчик рисует
//! `n×n` крупных однородных блоков известной цветности вместо payload, а этот
//! анализатор восстанавливает истину ИЗ САМОГО СНИМКА (строка счётчика §3.3
//! несёт номер кадра) и меряет, во что превратилась плоскость цветности.
//!
//! ```text
//! usage: swatch_diag <dir|dump-prefix> <crop x0,y0,w,h> <cell> <n> [corners|sweep]
//!   пример: swatch_diag C:/dumps/a22_n1 750,0,900,900 12 1 corners
//! ```
//! `<dir>` — каталог, в котором лежат `*.y` / `*.u` / `*.v` / `*.meta`; берутся
//! все найденные дампы, точки со всех кадров складываются в одну выборку.
//! Кроп обязателен: на полном кадре 1920×1080 символ делит экран с другими
//! окнами и грубая стадия детекции промахивается.
//!
//! # Что печатается и как читать
//! * **референсная строка §3.4** — известные K W R G B C M Y: прямой замер
//!   белого баланса и смешивания каналов, до всякого payload;
//! * **по блокам** — истина `(x, y)` -> измерение, разброс ВНУТРИ блока
//!   (шум на масштабе клетки) и смещение среднего блока (систематика);
//! * **аффинная подгонка** `ẑ = A·z + t` по всем блокам всех кадров:
//!   - `t` -> белый баланс / гамма / чёрный уровень,
//!   - поворот и сдвиг (антисимметричная и симметричная части `A`) ->
//!     НЕ снятое смешивание каналов (§3.4, матрица 3×3),
//!   - сингулярные числа `A` -> схлопывание оси (одна ось умерла),
//!   - остаток после снятия `A` и `t` -> шум.
//!
//! ТРИ ветки цветокоррекции считаются РЯДОМ:
//! * `СЫРОЙ` — §3.4 не применяется вовсе, только снятие тон-кривой. Обращение
//!   §5.1-CL делит на измеренную сумму каналов, поэтому цветность осмысленна.
//!   Ветка НЕ зависит от референсной строки — а строка эта ОДНА клетка высотой,
//!   и НЧ-фильтр хромы ISP разрушает её раньше, чем крупные блоки. Без сырой
//!   ветки поломку калибровки не отличить от поломки данных;
//! * `v0` — поканальные gain/offset по якорям K/W (путь v0);
//! * `3×3` — полная развязка каналов по всей референсной строке.
//!
//! # Как читать лестницу
//! Смотреть на «усиление осей» СЫРОЙ ветки как функцию `n`:
//! * усиление ПОСТОЯННО по `n` -> механизм НЕ зависит от масштаба (белый
//!   баланс, гамма, смешивание каналов) -> лечится матрицей §3.4;
//! * усиление ПАДАЕТ с ростом `n` -> механизм зависит от масштаба (4:2:0,
//!   шумодав хромы ISP, ISI) -> матрица не поможет, нужна клетка крупнее либо
//!   демодуляция, знающая про субдискретизацию хромы.

use psicode_core::swatch::{swatch_blocks, swatch_point, swatch_rect, SWATCH_GUARD};
use psicode_core::symbol::{self, ConstLumaMap, CL_LATTICE_SCALE};
use psicode_core::{detect, tone};
use psicode_rx::tx_chromatic_profile;
use psicode_rx::yuv::YuvFrame;
use std::fs;

const SQRT3: f64 = 1.732_050_807_568_877_2;
/// Отступ ВНУТРЬ блока, в клетках: тело блока читаем без краёв, чтобы блюр от
/// серого рва не попадал в измерение.
const BLOCK_MARGIN: usize = 1;

/// Клетка референсного паттерна §3.4 (копия приватной symbol::ref_pattern).
fn ref_pattern(idx: usize, k: u8, w: u8) -> [u8; 3] {
    match idx {
        0 | 8 => [k, k, k],
        1 | 9 => [w, w, w],
        2 => [w, k, k],
        3 => [k, w, k],
        4 => [k, k, w],
        5 => [k, w, w],
        6 => [w, k, w],
        7 => [w, w, k],
        _ => {
            let step = (idx - 10) as f64;
            let d = (k as f64 + (w as f64 - k as f64) * step / 5.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            [d, d, d]
        }
    }
}

const REF_NAMES: [&str; 16] = [
    "K", "W", "R", "G", "B", "C", "M", "Y", "K2", "W2", "g0", "g1", "g2", "g3", "g4", "g5",
];

/// Одно измерение блока: истина, среднее по телу блока и разброс внутри него.
struct BlockMeas {
    frame: u64,
    block: usize,
    truth: (f64, f64),
    meas: (f64, f64),
    sd: (f64, f64),
    cells: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: swatch_diag <dir|dump-prefix> <crop x0,y0,w,h> <cell> <n> [corners|sweep]"
        );
        std::process::exit(2);
    }
    let target = &args[1];
    let crop = &args[2];
    let cell_px: u8 = args[3].parse().expect("cell");
    let n: usize = args[4].parse().expect("n");
    let sweep = args.get(5).map(|s| s.as_str()) == Some("sweep");

    // --- список дампов ---
    let mut prefixes: Vec<String> = Vec::new();
    let path = std::path::Path::new(target);
    if path.is_dir() {
        let mut v: Vec<String> = fs::read_dir(path)
            .expect("dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("y"))
            .map(|e| e.path().with_extension("").to_string_lossy().into_owned())
            .collect();
        v.sort();
        prefixes = v;
    } else {
        prefixes.push(target.clone());
    }
    if prefixes.is_empty() {
        eprintln!("не найдено ни одного *.y в {target}");
        std::process::exit(1);
    }
    println!(
        "лестница масштабов: n = {n} ({} блок(ов)), режим {}, cell {cell_px} px, дампов {}",
        swatch_blocks(n),
        if sweep { "РАЗВЁРТКА" } else { "углы" },
        prefixes.len()
    );
    let (_, _, bc, br) = swatch_rect(n, 0, SWATCH_GUARD);
    println!(
        "блок {bc}×{br} клеток ≈ {}×{} px снимка (тело без {BLOCK_MARGIN}-клеточного края)",
        bc * cell_px as usize,
        br * cell_px as usize
    );

    let mut all_raw: Vec<BlockMeas> = Vec::new();
    let mut all_v0: Vec<BlockMeas> = Vec::new();
    let mut all_m3: Vec<BlockMeas> = Vec::new();
    let mut gammas_seen: Vec<[f64; 3]> = Vec::new();
    let mut ref_printed = false;

    for prefix in &prefixes {
        match analyse_one(
            prefix,
            crop,
            cell_px,
            n,
            sweep,
            !ref_printed,
            &mut [&mut all_raw, &mut all_v0, &mut all_m3],
        ) {
            Ok(g) => {
                gammas_seen.push(g);
                ref_printed = true;
            }
            Err(e) => println!("  {prefix}: ПРОПУСК ({e})"),
        }
    }

    if all_m3.is_empty() {
        eprintln!("ни одного пригодного кадра");
        std::process::exit(1);
    }
    // сводка по гаммам
    let ng = gammas_seen.len() as f64;
    let mut gm = [0.0f64; 3];
    for g in &gammas_seen {
        for c in 0..3 {
            gm[c] += g[c] / ng;
        }
    }
    println!(
        "\nтон (самокалибровка §3.4) по {} кадрам: γ = {:.2}/{:.2}/{:.2}",
        gammas_seen.len(),
        gm[0],
        gm[1],
        gm[2]
    );

    for (name, set) in [
        ("СЫРОЙ (без §3.4 вообще)", &all_raw),
        ("v0 (поканальный по K/W)", &all_v0),
        ("3×3 (§3.4, вся реф-строка)", &all_m3),
    ] {
        println!("\n================ ЦВЕТОКОРРЕКЦИЯ: {name} ================");
        report(set, n, sweep);
    }

    if std::env::var("PSI_DUMP").is_ok() {
        use std::io::Write;
        let out = format!("swatch_pairs_n{n}.tsv");
        let mut f = fs::File::create(&out).unwrap();
        writeln!(f, "branch\tframe\tblock\ttx\tty\tmx\tmy\tsdx\tsdy").unwrap();
        for (br_name, set) in [("raw", &all_raw), ("v0", &all_v0), ("m3", &all_m3)] {
            for m in set.iter() {
                writeln!(
                    f,
                    "{br_name}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                    m.frame, m.block, m.truth.0, m.truth.1, m.meas.0, m.meas.1, m.sd.0, m.sd.1
                )
                .unwrap();
            }
        }
        println!("\nпары (истина, измерение) -> {out}");
    }
}

/// Разбор одного дампа: детекция, тон, референсная строка, все блоки.
/// Возвращает оценённые гаммы.
#[allow(clippy::too_many_arguments)]
fn analyse_one(
    prefix: &str,
    crop: &str,
    cell_px: u8,
    n: usize,
    sweep: bool,
    print_ref: bool,
    out: &mut [&mut Vec<BlockMeas>; 3],
) -> Result<[f64; 3], String> {
    let meta = fs::read_to_string(format!("{prefix}.meta")).map_err(|e| e.to_string())?;
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|t| t.parse().unwrap_or(0))
        .collect();
    if m.len() < 5 {
        return Err("битый .meta".into());
    }
    let (mut w, mut h, mut y_stride, mut uv_stride, uv_px) = (m[0], m[1], m[2], m[3], m[4]);
    let mut yb = fs::read(format!("{prefix}.y")).map_err(|e| e.to_string())?;
    let mut ub = fs::read(format!("{prefix}.u")).map_err(|e| e.to_string())?;
    let mut vb = fs::read(format!("{prefix}.v")).map_err(|e| e.to_string())?;

    // кроп "x0,y0,w,h" (чётные)
    let c: Vec<usize> = crop
        .split(',')
        .map(|t| t.parse().unwrap_or(0))
        .collect();
    if c.len() == 4 {
        let (x0, y0, cw, ch) = (c[0] & !1, c[1] & !1, c[2] & !1, c[3] & !1);
        if x0 + cw > w || y0 + ch > h {
            return Err("кроп за границей кадра".into());
        }
        let mut ny = vec![0u8; cw * ch];
        for j in 0..ch {
            let src = (y0 + j) * y_stride + x0;
            ny[j * cw..(j + 1) * cw].copy_from_slice(&yb[src..src + cw]);
        }
        let cs = if uv_px == 2 { cw } else { cw / 2 };
        let mut nu = vec![0u8; cs * (ch / 2)];
        let mut nv = vec![0u8; cs * (ch / 2)];
        for j in 0..ch / 2 {
            for i in 0..cw / 2 {
                let s = (y0 / 2 + j) * uv_stride + (x0 / 2 + i) * uv_px;
                if s < ub.len() {
                    nu[j * cs + i * uv_px] = ub[s];
                }
                if s < vb.len() {
                    nv[j * cs + i * uv_px] = vb[s];
                }
            }
        }
        yb = ny;
        ub = nu;
        vb = nv;
        w = cw;
        h = ch;
        y_stride = cw;
        uv_stride = cs;
    }
    let fr = YuvFrame {
        y: &yb,
        u: &ub,
        v: &vb,
        w,
        h,
        y_stride,
        uv_stride,
        uv_pixel_stride: uv_px,
    };

    // --- детекция ---
    let luma: Vec<f32> = (0..w * h).map(|i| fr.y_norm(i % w, i / w)).collect();
    let mut det = detect::detect_symbol(w, h, &luma)
        .or_else(|_| detect::detect_symbol_acquire(w, h, &luma))
        .map_err(|_| "детекция не нашла символ".to_string())?;
    for _ in 0..10 {
        match detect::track_symbol(w, h, &luma, &det) {
            Ok(d2) if d2.score > det.score + 1e-4 => det = d2,
            _ => break,
        }
    }

    let mut p = tx_chromatic_profile();
    p.cell_size_px = cell_px;
    let map = detect::frame_map(&p, &det);
    let quiet = p.quiet_zone_cells() as usize;
    let cellpx = p.cell_size_px as usize;
    let black_255 = (255.0 * p.black_level_pct() as f64 / 100.0).round() as u8;
    let white_255 = (255.0 * p.white_level_pct() as f64 / 100.0).round() as u8;

    // --- сэмплеры ---
    let raw = |x: f64, y: f64| -> [f32; 3] {
        let xc = x.clamp(0.0, (w - 1) as f64);
        let yc = y.clamp(0.0, (h - 1) as f64);
        let (x0, y0) = (xc.floor() as usize, yc.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = ((xc - x0 as f64) as f32, (yc - y0 as f64) as f32);
        let mut o = [0.0f32; 3];
        let s00 = fr.rgb_at(x0, y0);
        let s10 = fr.rgb_at(x1, y0);
        let s01 = fr.rgb_at(x0, y1);
        let s11 = fr.rgb_at(x1, y1);
        for ch in 0..3 {
            let a = s00[ch] * (1.0 - fx) + s10[ch] * fx;
            let b = s01[ch] * (1.0 - fx) + s11[ch] * fx;
            o[ch] = a * (1.0 - fy) + b * fy;
        }
        o
    };
    let g = tone::estimate_channel_gammas(&p, &map, &raw);
    let lin = |x: f64, y: f64| -> [f32; 3] {
        let s = raw(x, y);
        [
            (s[0] as f64).max(0.0).powf(g[0]) as f32,
            (s[1] as f64).max(0.0).powf(g[1]) as f32,
            (s[2] as f64).max(0.0).powf(g[2]) as f32,
        ]
    };
    let sample_cell = |cx: usize, cy: usize| -> [f64; 3] {
        let u = ((quiet + cx) * cellpx) as f64 + cellpx as f64 / 2.0;
        let v = ((quiet + cy) * cellpx) as f64 + cellpx as f64 / 2.0;
        let d = cellpx as f64 / 4.0;
        let mut acc = [0.0f64; 3];
        for &(sx, sy) in &[(-d, -d), (-d, d), (d, -d), (d, d)] {
            let (x, y) = map(u + sx, v + sy);
            let s = lin(x, y);
            for ch in 0..3 {
                acc[ch] += s[ch] as f64;
            }
        }
        for ch in 0..3 {
            acc[ch] /= 4.0;
        }
        acc
    };

    // --- номер кадра из строки счётчика §3.3 ---
    let (c_start, c_end) = symbol::read_counters(&p, &map, &lin);
    if c_start != c_end {
        return Err(format!("рваный снимок: счётчик {c_start} != {c_end}"));
    }
    let frame = c_start as u64;

    // --- цветокоррекция §3.4: две ветки рядом ---
    let gp = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    let target = |d: [f64; 3]| -> [f64; 3] {
        [
            (d[0] / 255.0).powf(gp[0]),
            (d[1] / 255.0).powf(gp[1]),
            (d[2] / 255.0).powf(gp[2]),
        ]
    };
    let mut ref_s: Vec<([f64; 3], [f64; 3], [u8; 3])> = Vec::new();
    for ic in 0..symbol::INTERIOR {
        let pat = ref_pattern(ic % 16, black_255, white_255);
        let s = sample_cell(symbol::RING + ic, symbol::RING);
        ref_s.push((
            s,
            target([pat[0] as f64, pat[1] as f64, pat[2] as f64]),
            pat,
        ));
    }
    // v0: поканальные a/b по якорям K/W
    let (mut sk, mut sw, mut nk, mut nw) = ([0.0f64; 3], [0.0f64; 3], 0usize, 0usize);
    for (s, _, pat) in &ref_s {
        if *pat == [black_255; 3] {
            for ch in 0..3 {
                sk[ch] += s[ch];
            }
            nk += 1;
        } else if *pat == [white_255; 3] {
            for ch in 0..3 {
                sw[ch] += s[ch];
            }
            nw += 1;
        }
    }
    let (mut a_gain, mut b_off) = ([1.0f64; 3], [0.0f64; 3]);
    for ch in 0..3 {
        sk[ch] /= nk.max(1) as f64;
        sw[ch] /= nw.max(1) as f64;
        let dk = (black_255 as f64 / 255.0).powf(gp[ch]);
        let dw = (white_255 as f64 / 255.0).powf(gp[ch]);
        let a = (sw[ch] - sk[ch]) / (dw - dk);
        a_gain[ch] = if a.abs() < 1e-12 { 1e-12 } else { a };
        b_off[ch] = sk[ch] - a_gain[ch] * dk;
    }
    // 3×3: МНК по всей референсной строке
    let pts: Vec<([f64; 3], [f64; 3])> = ref_s.iter().map(|(s, t, _)| (*s, *t)).collect();
    let (nm, q) = fit3x3(&pts);

    let drive_v0 = |s: [f64; 3]| -> [f64; 3] {
        let mut d = [0.0f64; 3];
        for ch in 0..3 {
            let t = ((s[ch] - b_off[ch]) / a_gain[ch]).max(0.0);
            d[ch] = (255.0 * t.powf(1.0 / gp[ch])).clamp(0.0, 255.0);
        }
        d
    };
    // СЫРАЯ ветка: никакой §3.4 — только снятие тон-кривой. Обращение §5.1-CL
    // делит на ИЗМЕРЕННУЮ сумму каналов, поэтому общий множитель сокращается и
    // цветность остаётся осмысленной; НЕ сокращается только поканальное
    // усиление (белый баланс). Ветка нужна, чтобы отличить «сломаны данные» от
    // «сломана калибровка»: референсная строка — ОДНА клетка высотой, и любой
    // НЧ-фильтр хромы ISP разрушает её РАНЬШЕ, чем крупные блоки.
    let drive_raw = |s: [f64; 3]| -> [f64; 3] {
        let mut d = [0.0f64; 3];
        for ch in 0..3 {
            d[ch] = (255.0 * s[ch].max(0.0).powf(1.0 / gp[ch])).clamp(0.0, 255.0);
        }
        d
    };
    let drive_m3 = |s: [f64; 3]| -> [f64; 3] {
        let mut d = [0.0f64; 3];
        for ch in 0..3 {
            let t = (nm[ch][0] * s[0] + nm[ch][1] * s[1] + nm[ch][2] * s[2] + q[ch]).max(0.0);
            d[ch] = (255.0 * t.powf(1.0 / gp[ch])).clamp(0.0, 255.0);
        }
        d
    };

    if print_ref {
        println!("\n=== РЕФЕРЕНСНАЯ СТРОКА §3.4 ({prefix}) ===");
        println!(" патч  ожид.drive      восст. v0          восст. 3×3");
        let mut acc = vec![([0.0f64; 3], [0.0f64; 3], 0usize); 16];
        for (i, (s, _, _)) in ref_s.iter().enumerate() {
            let k = i % 16;
            let d0 = drive_v0(*s);
            let d3 = drive_m3(*s);
            for ch in 0..3 {
                acc[k].0[ch] += d0[ch];
                acc[k].1[ch] += d3[ch];
            }
            acc[k].2 += 1;
        }
        for k in 0..16 {
            let cnt = acc[k].2.max(1) as f64;
            let want = ref_pattern(k, black_255, white_255);
            println!(
                " {:>4} {:>3},{:>3},{:>3}   {:5.0},{:5.0},{:5.0}   {:5.0},{:5.0},{:5.0}",
                REF_NAMES[k],
                want[0],
                want[1],
                want[2],
                acc[k].0[0] / cnt,
                acc[k].0[1] / cnt,
                acc[k].0[2] / cnt,
                acc[k].1[0] / cnt,
                acc[k].1[1] / cnt,
                acc[k].1[2] / cnt
            );
        }
        println!("матрица 3×3 §3.4 (t̂ = N·s + q), строки нормированы на диагональ:");
        for ch in 0..3 {
            let d = if nm[ch][ch].abs() > 1e-12 {
                nm[ch][ch]
            } else {
                1.0
            };
            println!(
                "   {} [{:+7.3} {:+7.3} {:+7.3}]  (диаг {:+.4}, смещ {:+.4})",
                ["R", "G", "B"][ch],
                nm[ch][0] / d,
                nm[ch][1] / d,
                nm[ch][2] / d,
                nm[ch][ch],
                q[ch]
            );
        }
    }

    // --- блоки ---
    let cl = const_luma(black_255 as f64, white_255 as f64);
    let name = std::path::Path::new(prefix)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(prefix);
    println!(
        "\n{name}: score {:.4}, rot {}, кадр (счётчик) {frame}, γ {:.2}/{:.2}/{:.2}",
        det.score, det.rotation_quadrants, g[0], g[1], g[2]
    );
    for b in 0..swatch_blocks(n) {
        let (pc0, pr0, cols, rows) = swatch_rect(n, b, SWATCH_GUARD);
        if cols == 0 || rows == 0 {
            continue;
        }
        // на мелких ступенях блок сам по себе 1–2 клетки: отступ снимаем.
        let margin = if cols.min(rows) >= 3 { BLOCK_MARGIN } else { 0 };
        let truth = swatch_point(n, frame, b, sweep);
        for (which, drive) in [
            (0usize, &drive_raw as &dyn Fn([f64; 3]) -> [f64; 3]),
            (1usize, &drive_v0 as &dyn Fn([f64; 3]) -> [f64; 3]),
            (2usize, &drive_m3 as &dyn Fn([f64; 3]) -> [f64; 3]),
        ] {
            let mut zx: Vec<f64> = Vec::new();
            let mut zy: Vec<f64> = Vec::new();
            for pr in pr0 + margin..pr0 + rows - margin {
                for pc in pc0 + margin..pc0 + cols - margin {
                    let s = sample_cell(symbol::RING + pc, symbol::RING + 1 + pr);
                    let (x, y) = cl.z_from_drive(drive(s));
                    zx.push(x);
                    zy.push(y);
                }
            }
            let (mx, sx) = mean_sd(&zx);
            let (my, sy) = mean_sd(&zy);
            let bm = BlockMeas {
                frame,
                block: b,
                truth,
                meas: (mx, my),
                sd: (sx, sy),
                cells: zx.len(),
            };
            out[which].push(bm);
        }
    }
    Ok(g)
}

/// Отчёт по накопленной выборке блоков.
fn report(set: &[BlockMeas], n: usize, sweep: bool) {
    // 1. таблица блоков (для углов — компактно по 4 состояниям)
    if !sweep {
        println!("по углам созвездия (истина -> измерение, среднее по всем блокам/кадрам):");
        for &(tx, ty) in &[
            (-1.0f64, -1.0f64),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        ] {
            let sel: Vec<&BlockMeas> = set
                .iter()
                .filter(|m| {
                    (m.truth.0 > 0.0) == (tx > 0.0) && (m.truth.1 > 0.0) == (ty > 0.0)
                })
                .collect();
            if sel.is_empty() {
                continue;
            }
            let k = sel.len() as f64;
            let mx = sel.iter().map(|m| m.meas.0).sum::<f64>() / k;
            let my = sel.iter().map(|m| m.meas.1).sum::<f64>() / k;
            let sx = (sel.iter().map(|m| (m.meas.0 - mx).powi(2)).sum::<f64>() / k).sqrt();
            let sy = (sel.iter().map(|m| (m.meas.1 - my).powi(2)).sum::<f64>() / k).sqrt();
            let ix = sel.iter().map(|m| m.sd.0).sum::<f64>() / k;
            let iy = sel.iter().map(|m| m.sd.1).sum::<f64>() / k;
            println!(
                "  ({:+.3}, {:+.3}) -> ({:+.3}, {:+.3})  разброс блоков ({:.3}, {:.3})  \
                 шум ВНУТРИ блока ({:.3}, {:.3})  n {}",
                tx * CL_LATTICE_SCALE,
                ty * CL_LATTICE_SCALE,
                mx,
                my,
                sx,
                sy,
                ix,
                iy,
                sel.len()
            );
        }
    } else {
        // развёртка: сводка по кольцам радиуса
        println!("развёртка по кольцам |z| (истина -> измерение):");
        for k in 0..5usize {
            let (lo, hi) = (k as f64 * 0.2, (k + 1) as f64 * 0.2);
            let sel: Vec<&BlockMeas> = set
                .iter()
                .filter(|m| {
                    let r = (m.truth.0 * m.truth.0 + m.truth.1 * m.truth.1).sqrt();
                    r >= lo && r < hi + 1e-9
                })
                .collect();
            if sel.is_empty() {
                continue;
            }
            let kf = sel.len() as f64;
            let rt = sel
                .iter()
                .map(|m| (m.truth.0 * m.truth.0 + m.truth.1 * m.truth.1).sqrt())
                .sum::<f64>()
                / kf;
            let rm = sel
                .iter()
                .map(|m| (m.meas.0 * m.meas.0 + m.meas.1 * m.meas.1).sqrt())
                .sum::<f64>()
                / kf;
            let ix = sel.iter().map(|m| m.sd.0).sum::<f64>() / kf;
            let iy = sel.iter().map(|m| m.sd.1).sum::<f64>() / kf;
            println!(
                "  |z| {lo:.1}..{hi:.1}: истина {rt:.3} -> измер {rm:.3} (усиление {:.3}), \
                 шум внутри блока ({ix:.3}, {iy:.3}), n {}",
                if rt > 1e-6 { rm / rt } else { 0.0 },
                sel.len()
            );
        }
    }

    // 2. аффинная подгонка ẑ = A·z + t
    let (a, t) = fit_affine2(set);
    println!("аффинная подгонка ẑ = A·z + t по {} блокам:", set.len());
    println!(
        "  A = [{:+.4} {:+.4}]   t = ({:+.4}, {:+.4})",
        a[0][0], a[0][1], t[0], t[1]
    );
    println!("      [{:+.4} {:+.4}]", a[1][0], a[1][1]);
    // разложение
    let rot = (a[1][0] - a[0][1]).atan2(a[0][0] + a[1][1]);
    let shear = 0.5 * (a[0][1] + a[1][0]);
    let (s1, s2) = singular_values(&a);
    println!(
        "  усиление осей: Re {:+.3}, Im {:+.3} | поворот {:+.2}° | симметричный сдвиг {:+.4}",
        a[0][0],
        a[1][1],
        rot.to_degrees(),
        shear
    );
    println!(
        "  сингулярные числа {:.3} / {:.3} (схлопывание оси = отношение {:.2})",
        s1,
        s2,
        if s1 > 1e-9 { s2 / s1 } else { 0.0 }
    );
    // 3. остаток
    let (mut rx, mut ry, mut r0x, mut r0y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for m in set {
        let px = a[0][0] * m.truth.0 + a[0][1] * m.truth.1 + t[0];
        let py = a[1][0] * m.truth.0 + a[1][1] * m.truth.1 + t[1];
        rx += (m.meas.0 - px).powi(2);
        ry += (m.meas.1 - py).powi(2);
        r0x += (m.meas.0 - m.truth.0).powi(2);
        r0y += (m.meas.1 - m.truth.1).powi(2);
    }
    let k = set.len() as f64;
    println!(
        "  СЫРАЯ ошибка rms (x {:.3}, y {:.3}) -> ОСТАТОК после снятия A и t (x {:.3}, y {:.3})",
        (r0x / k).sqrt(),
        (r0y / k).sqrt(),
        (rx / k).sqrt(),
        (ry / k).sqrt()
    );
    // 4. шум внутри блока — ключевое число лестницы
    let ix = set.iter().map(|m| m.sd.0).sum::<f64>() / k;
    let iy = set.iter().map(|m| m.sd.1).sum::<f64>() / k;
    let cells = set.iter().map(|m| m.cells).max().unwrap_or(0);
    println!(
        "  шум ВНУТРИ блока (по клеткам, {cells} клеток/блок): σx {ix:.3}, σy {iy:.3}  [n = {n}]"
    );
}

// ---------------------------------------------------------------------------
// Мелкая линейная алгебра
// ---------------------------------------------------------------------------

fn mean_sd(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
    (m, s)
}

fn const_luma(black: f64, white: f64) -> ConstLumaMap {
    let u = 0.5 * (white + black);
    let amp = (white - u).min(u - black).max(0.0);
    let b = if u > 0.0 { amp / (2.0 * u) } else { 0.0 };
    ConstLumaMap {
        u,
        b,
        c: SQRT3 * b,
        s: 3.0 * u,
        amp,
    }
}

/// МНК `t = N·s + q` по парам (измеренный линеаризованный RGB, целевой драйв).
fn fit3x3(pts: &[([f64; 3], [f64; 3])]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut nmat = [[0.0f64; 3]; 3];
    let mut q = [0.0f64; 3];
    for c in 0..3 {
        let mut m = [[0.0f64; 5]; 4];
        for (s, t) in pts {
            let bas = [s[0], s[1], s[2], 1.0];
            for i in 0..4 {
                for j in 0..4 {
                    m[i][j] += bas[i] * bas[j];
                }
                m[i][4] += bas[i] * t[c];
            }
        }
        solve_inplace(&mut m, 4, 1);
        nmat[c] = [m[0][4], m[1][4], m[2][4]];
        q[c] = m[3][4];
    }
    (nmat, q)
}

/// МНК `ẑ = A·z + t` (2×2 плюс смещение) по блокам.
fn fit_affine2(set: &[BlockMeas]) -> ([[f64; 2]; 2], [f64; 2]) {
    let mut m = [[0.0f64; 5]; 4];
    // базис [zx, zy, 1]; две правые части (mx, my). Используем 3×3 систему.
    let mut a = [[0.0f64; 5]; 4];
    for bm in set {
        let bas = [bm.truth.0, bm.truth.1, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += bas[i] * bas[j];
            }
            a[i][3] += bas[i] * bm.meas.0;
            a[i][4] += bas[i] * bm.meas.1;
        }
    }
    m.copy_from_slice(&a);
    solve_inplace(&mut m, 3, 2);
    (
        [[m[0][3], m[1][3]], [m[0][4], m[1][4]]],
        [m[2][3], m[2][4]],
    )
}

/// Гаусс с выбором ведущего для системы `n×n` с `rhs` правыми частями,
/// расположенными в столбцах `n..n+rhs`. Решение оказывается на их месте.
fn solve_inplace(m: &mut [[f64; 5]; 4], n: usize, rhs: usize) {
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        m.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-15 {
            continue;
        }
        for j in col..n + rhs {
            m[col][j] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[r][col];
                for j in col..n + rhs {
                    m[r][j] -= f * m[col][j];
                }
            }
        }
    }
}

/// Сингулярные числа матрицы 2×2 (по собственным числам AᵀA).
fn singular_values(a: &[[f64; 2]; 2]) -> (f64, f64) {
    let m00 = a[0][0] * a[0][0] + a[1][0] * a[1][0];
    let m01 = a[0][0] * a[0][1] + a[1][0] * a[1][1];
    let m11 = a[0][1] * a[0][1] + a[1][1] * a[1][1];
    let tr = m00 + m11;
    let det = m00 * m11 - m01 * m01;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    (
        (tr / 2.0 + disc).max(0.0).sqrt(),
        (tr / 2.0 - disc).max(0.0).sqrt(),
    )
}
