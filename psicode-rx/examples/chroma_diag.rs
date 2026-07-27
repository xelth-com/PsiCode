//! Диагностика хромо-тракта §5.1-CL на ЖИВЫХ YUV-дампах телефона.
//!
//! usage: chroma_diag <dump-prefix> <streamed-file|--none> <sid-hex> [crop x0,y0,w,h] [cell]
//!
//! Отвечает на три вопроса, ради которых написан:
//!   1. куда РЕАЛЬНО садится созвездие (x, y) — скан по payload-клеткам;
//!   2. что показывает референсная строка §3.4 (K W R G B C M Y + лесенка) —
//!      прямой замер белого баланса и матрицы 3×3;
//!   3. поосевой SER (ось Re = 2G−R−B против оси Im = R−B) против ИСТИНЫ
//!      передатчика, с проверкой качества «лока» на seq.

use psicode_core::fountain::{crc32c, FountainEncoder};
use psicode_core::l3::{self, FrameHeader, TransferInfo, FLAG_TRANSFER_INFO};
use psicode_core::symbol::{self, ConstLumaMap};
use psicode_core::{detect, tone};
use psicode_rx::tx_chromatic_profile;
use psicode_rx::yuv::YuvFrame;
use std::fs;

const SYMBOLS_PER_FRAME: usize = 8;
const REPAIR_EVERY: u32 = 4;
const SQRT3: f64 = 1.732_050_807_568_877_2;
/// зеркало приватной symbol::CL_LATTICE_SCALE.
const CL_SCALE: f64 = 2.0 / (1.0 + SQRT3);

fn symbol_size_for(bpc: u32) -> usize {
    let cap: usize = l3::STRIPE_ROWS
        .iter()
        .map(|&r| (r * l3::PAYLOAD_COLS * bpc as usize - 16) / 8)
        .sum();
    (cap - l3::FRAME_HEADER_LEN - l3::TRANSFER_INFO_LEN) / SYMBOLS_PER_FRAME
}

/// Один систематический проход потока (зеркало `frames.rs::systematic_pass`
/// при `repair_every = 4`): source 0..K−1 с вплетённым repair. `total`
/// игнорируется — проход имеет естественную длину K + K/4.
fn emission_order(k: u32, _total: usize) -> Vec<u32> {
    let mut v = Vec::new();
    let (mut src, mut rep, mut since) = (0u32, k, 0u32);
    while src < k {
        v.push(src);
        src += 1;
        since += 1;
        if since == REPAIR_EVERY {
            v.push(rep);
            rep += 1;
            since = 0;
        }
    }
    v
}

/// Клетка референсного паттерна §3.4 (копия приватной symbol::ref_pattern).
fn ref_pattern(idx: usize, k: u8, w: u8) -> [u8; 3] {
    match idx {
        0 => [k, k, k],
        1 => [w, w, w],
        2 => [w, k, k],
        3 => [k, w, k],
        4 => [k, k, w],
        5 => [k, w, w],
        6 => [w, k, w],
        7 => [w, w, k],
        8 => [k, k, k],
        9 => [w, w, w],
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = args.get(1).expect("usage: chroma_diag <prefix> <file|--none> <sid> [crop] [cell]");
    let file = args.get(2).cloned().unwrap_or_else(|| "--none".into());
    let sid = u32::from_str_radix(args.get(3).map(|s| s.as_str()).unwrap_or("0"), 16).unwrap_or(0);

    // --- дамп ---
    let meta = fs::read_to_string(format!("{prefix}.meta")).expect("meta");
    let m: Vec<usize> = meta.split_whitespace().map(|t| t.parse().unwrap()).collect();
    let (mut w, mut h, mut y_stride, mut uv_stride, uv_px) = (m[0], m[1], m[2], m[3], m[4]);
    let mut yb = fs::read(format!("{prefix}.y")).expect("y");
    let mut ub = fs::read(format!("{prefix}.u")).expect("u");
    let mut vb = fs::read(format!("{prefix}.v")).expect("v");

    if let Some(spec) = args.get(4) {
        if spec.contains(',') {
            let c: Vec<usize> = spec.split(',').map(|t| t.parse().unwrap()).collect();
            let (x0, y0, cw, ch) = (c[0] & !1, c[1] & !1, c[2] & !1, c[3] & !1);
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
            println!("кроп: {x0},{y0} {cw}x{ch}");
        }
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
        .expect("не найдено");
    for _ in 0..10 {
        match detect::track_symbol(w, h, &luma, &det) {
            Ok(d2) if d2.score > det.score + 1e-4 => det = d2,
            _ => break,
        }
    }
    println!("детекция: score {:.4}, rot {}", det.score, det.rotation_quadrants);

    let mut p = tx_chromatic_profile();
    if let Some(cs) = args.get(5).and_then(|s| s.parse::<u8>().ok()) {
        p.cell_size_px = cs;
    }
    let bpc = symbol::bits_per_cell(&p);
    println!(
        "профиль: luma_bits {} + chroma {} = {} бит/клетку, cell {}, quiet {}",
        p.luma_bits,
        p.chroma_bits(),
        bpc,
        p.cell_size_px,
        p.quiet_zone_cells()
    );
    let map = detect::frame_map(&p, &det);
    let (black_255, white_255) = (
        (255.0 * p.black_level_pct() as f64 / 100.0).round() as u8,
        (255.0 * p.white_level_pct() as f64 / 100.0).round() as u8,
    );
    println!("уровни: black {black_255}, white {white_255}");

    // px/клетку в координатах снимка
    let cellf = p.cell_size_px as f64;
    let quiet = p.quiet_zone_cells() as usize;
    let cellpx = p.cell_size_px as usize;
    let cc = |cx: f64, cy: f64| {
        map(
            (quiet as f64 + cx) * cellf + cellf / 2.0,
            (quiet as f64 + cy) * cellf + cellf / 2.0,
        )
    };
    let (ax, ay) = cc(30.0, 30.0);
    let (bx, by) = cc(31.0, 30.0);
    println!(
        "px/клетку {:.2}",
        ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
    );

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
        for c in 0..3 {
            let a = s00[c] * (1.0 - fx) + s10[c] * fx;
            let b = s01[c] * (1.0 - fx) + s11[c] * fx;
            o[c] = a * (1.0 - fy) + b * fy;
        }
        o
    };
    // СЫРЫЕ Y/U/V той же точки (билинейно по Y, ближайший по хроме).
    let rawyuv = |x: f64, y: f64| -> [f64; 3] {
        let xi = (x.clamp(0.0, (w - 1) as f64)) as usize;
        let yi = (y.clamp(0.0, (h - 1) as f64)) as usize;
        let yy = yb.get(yi * y_stride + xi).copied().unwrap_or(0) as f64;
        let ci = (yi >> 1) * uv_stride + (xi >> 1) * uv_px;
        [
            yy,
            ub.get(ci).copied().unwrap_or(128) as f64,
            vb.get(ci).copied().unwrap_or(128) as f64,
        ]
    };

    let g = tone::estimate_channel_gammas(&p, &map, &raw);
    println!("тон: γ = {:.2}/{:.2}/{:.2}", g[0], g[1], g[2]);
    let lin = |x: f64, y: f64| -> [f32; 3] {
        let s = raw(x, y);
        [
            (s[0] as f64).max(0.0).powf(g[0]) as f32,
            (s[1] as f64).max(0.0).powf(g[1]) as f32,
            (s[2] as f64).max(0.0).powf(g[2]) as f32,
        ]
    };

    // сэмпл клетки (cx,cy) как symbol::sample_cell: среднее 2×2 ±cell/4.
    let sample_cell = |cx: usize, cy: usize, src: &dyn Fn(f64, f64) -> [f32; 3]| -> [f64; 3] {
        let u = ((quiet + cx) * cellpx) as f64 + cellpx as f64 / 2.0;
        let v = ((quiet + cy) * cellpx) as f64 + cellpx as f64 / 2.0;
        let d = cellpx as f64 / 4.0;
        let mut acc = [0.0f64; 3];
        for &(sx, sy) in &[(-d, -d), (-d, d), (d, -d), (d, d)] {
            let (x, y) = map(u + sx, v + sy);
            let s = src(x, y);
            for c in 0..3 {
                acc[c] += s[c] as f64;
            }
        }
        for c in 0..3 {
            acc[c] /= 4.0;
        }
        acc
    };
    let sample_cell_yuv = |cx: usize, cy: usize| -> [f64; 3] {
        let u = ((quiet + cx) * cellpx) as f64 + cellpx as f64 / 2.0;
        let v = ((quiet + cy) * cellpx) as f64 + cellpx as f64 / 2.0;
        let d = cellpx as f64 / 4.0;
        let mut acc = [0.0f64; 3];
        for &(sx, sy) in &[(-d, -d), (-d, d), (d, -d), (d, d)] {
            let (x, y) = map(u + sx, v + sy);
            let s = rawyuv(x, y);
            for c in 0..3 {
                acc[c] += s[c];
            }
        }
        for c in 0..3 {
            acc[c] /= 4.0;
        }
        acc
    };

    // --- нормировка §3.4 по K/W референсной строки (как demod_symbol) ---
    let black = [black_255; 3];
    let white = [white_255; 3];
    let (mut s_k, mut s_w) = ([0.0f64; 3], [0.0f64; 3]);
    let (mut nk, mut nw) = (0usize, 0usize);
    for ic in 0..symbol::INTERIOR {
        let pat = ref_pattern(ic % 16, black_255, white_255);
        if pat == black {
            let s = sample_cell(symbol::RING + ic, symbol::RING, &lin);
            for c in 0..3 {
                s_k[c] += s[c];
            }
            nk += 1;
        } else if pat == white {
            let s = sample_cell(symbol::RING + ic, symbol::RING, &lin);
            for c in 0..3 {
                s_w[c] += s[c];
            }
            nw += 1;
        }
    }
    let (mut a_gain, mut b_off) = ([1.0f64; 3], [0.0f64; 3]);
    for c in 0..3 {
        s_k[c] /= nk.max(1) as f64;
        s_w[c] /= nw.max(1) as f64;
        let dkc = (black_255 as f64 / 255.0).powf(g[c]);
        let dwc = (white_255 as f64 / 255.0).powf(g[c]);
        let a = (s_w[c] - s_k[c]) / (dwc - dkc);
        a_gain[c] = if a.abs() < 1e-12 { 1e-12 } else { a };
        b_off[c] = s_k[c] - a_gain[c] * dkc;
    }
    println!(
        "якоря §3.4: s_K = [{:.4} {:.4} {:.4}], s_W = [{:.4} {:.4} {:.4}]",
        s_k[0], s_k[1], s_k[2], s_w[0], s_w[1], s_w[2]
    );
    println!(
        "            a = [{:.4} {:.4} {:.4}], b = [{:.4} {:.4} {:.4}]",
        a_gain[0], a_gain[1], a_gain[2], b_off[0], b_off[1], b_off[2]
    );

    // линеаризованный драйв клетки (в точности как demod_symbol)
    let drive_of = |cx: usize, cy: usize| -> [f64; 3] {
        let s = sample_cell(cx, cy, &lin);
        let mut d = [0.0f64; 3];
        for c in 0..3 {
            let base = ((s[c] - b_off[c]) / a_gain[c]).max(0.0);
            d[c] = (255.0 * base.powf(1.0 / g[c])).clamp(0.0, 255.0);
        }
        d
    };

    // ====================================================================
    // 1. РЕФЕРЕНСНАЯ СТРОКА §3.4: что показывает камера на ИЗВЕСТНЫХ цветах
    // ====================================================================
    println!("\n=== РЕФЕРЕНСНАЯ СТРОКА §3.4 (среднее по повторам периода) ===");
    println!(" idx  ожид.drive        raw RGB(0..1)            Y   Cb   Cr     восст.drive         ошибка");
    let mut acc: Vec<([f64; 3], [f64; 3], [f64; 3], usize)> =
        vec![([0.0; 3], [0.0; 3], [0.0; 3], 0); 16];
    let mut ref_lin = vec![[0.0f64; 3]; 16];
    for ic in 0..symbol::INTERIOR {
        let k = ic % 16;
        let r = sample_cell(symbol::RING + ic, symbol::RING, &raw);
        let d = drive_of(symbol::RING + ic, symbol::RING);
        let yv = sample_cell_yuv(symbol::RING + ic, symbol::RING);
        let sl = sample_cell(symbol::RING + ic, symbol::RING, &lin);
        for c in 0..3 {
            acc[k].0[c] += r[c];
            acc[k].1[c] += d[c];
            acc[k].2[c] += yv[c];
            ref_lin[k][c] += sl[c];
        }
        acc[k].3 += 1;
    }
    for k in 0..16 {
        let n = acc[k].3.max(1) as f64;
        for c in 0..3 {
            ref_lin[k][c] /= n;
        }
    }
    for k in 0..16 {
        let n = acc[k].3.max(1) as f64;
        let r = [acc[k].0[0] / n, acc[k].0[1] / n, acc[k].0[2] / n];
        let d = [acc[k].1[0] / n, acc[k].1[1] / n, acc[k].1[2] / n];
        let yv = [acc[k].2[0] / n, acc[k].2[1] / n, acc[k].2[2] / n];
        let want = ref_pattern(k, black_255, white_255);
        let err = [
            d[0] - want[0] as f64,
            d[1] - want[1] as f64,
            d[2] - want[2] as f64,
        ];
        println!(
            " {:>3} {:>3},{:>3},{:>3}   {:.3},{:.3},{:.3}   {:5.1},{:5.1},{:5.1}   {:5.1},{:5.1},{:5.1}   {:+6.1},{:+6.1},{:+6.1}",
            REF_NAMES[k], want[0], want[1], want[2],
            r[0], r[1], r[2], yv[0], yv[1], yv[2],
            d[0], d[1], d[2], err[0], err[1], err[2]
        );
    }

    // ====================================================================
    // 2. СОЗВЕЗДИЕ: куда садятся payload-клетки
    // ====================================================================
    let mfromlv = |b: f64, wl: f64| -> ConstLumaMap {
        let u = 0.5 * (wl + b);
        let amp = (wl - u).min(u - b).max(0.0);
        let bb = if u > 0.0 { amp / (2.0 * u) } else { 0.0 };
        ConstLumaMap {
            u,
            b: bb,
            c: SQRT3 * bb,
            s: 3.0 * u,
            amp,
        }
    };
    let cl = mfromlv(black_255 as f64, white_255 as f64);
    println!(
        "\nConstLumaMap: u {:.1}, b {:.4}, c {:.4}, S {:.1}, A {:.1}, scale {:.4}",
        cl.u, cl.b, cl.c, cl.s, cl.amp, CL_SCALE
    );

    let pcn = symbol::PAYLOAD_COLS;
    let prn = symbol::PAYLOAD_ROWS;
    let n_cells = pcn * prn;
    let mut xs = vec![0.0f64; n_cells];
    let mut ys = vec![0.0f64; n_cells];
    let mut drives = vec![[0.0f64; 3]; n_cells];
    let mut raws = vec![[0.0f64; 3]; n_cells];
    let mut lins = vec![[0.0f64; 3]; n_cells];
    for pr in 0..prn {
        for pc in 0..pcn {
            let cx = symbol::RING + pc;
            let cy = symbol::RING + 1 + pr;
            let d = drive_of(cx, cy);
            let (x, y) = cl.z_from_drive(d);
            let i = pr * pcn + pc;
            xs[i] = x / CL_SCALE;
            ys[i] = y / CL_SCALE;
            drives[i] = d;
            raws[i] = sample_cell(cx, cy, &raw);
            lins[i] = sample_cell(cx, cy, &lin);
        }
    }
    let stat = |v: &[f64]| -> (f64, f64) {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
        (m, s)
    };
    let (mx, sx) = stat(&xs);
    let (my, sy) = stat(&ys);
    println!(
        "созвездие: x  mean {:+.3} sd {:.3}   y  mean {:+.3} sd {:.3}   (идеал mean 0, sd 1)",
        mx, sx, my, sy
    );
    // корреляция осей — сдвиг/срез
    let cov = xs
        .iter()
        .zip(&ys)
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / n_cells as f64;
    println!("корреляция осей x,y: {:+.3}", cov / (sx * sy).max(1e-12));
    // маргинальные гистограммы
    let hist = |v: &[f64], name: &str| {
        let mut bins = [0usize; 21];
        for &t in v {
            let b = (((t + 2.0) / 4.0) * 20.0).round().clamp(0.0, 20.0) as usize;
            bins[b] += 1;
        }
        print!("гист {name} [-2..2]:");
        for b in bins {
            print!(" {b}");
        }
        println!();
    };
    hist(&xs, "x");
    hist(&ys, "y");
    // 2D-гистограмма 9x9 по [-1.5, 1.5]
    println!("2D-гистограмма (x вправо, y вниз), диапазон ±1.5, 9x9:");
    let mut h2 = [[0usize; 9]; 9];
    for i in 0..n_cells {
        let bx = (((xs[i] + 1.5) / 3.0) * 8.0).round().clamp(0.0, 8.0) as usize;
        let by = (((ys[i] + 1.5) / 3.0) * 8.0).round().clamp(0.0, 8.0) as usize;
        h2[by][bx] += 1;
    }
    for row in h2.iter() {
        let s: Vec<String> = row.iter().map(|c| format!("{c:5}")).collect();
        println!("   {}", s.join(""));
    }
    // средние драйвы по четвертям решения
    let mut q: Vec<([f64; 3], usize)> = vec![([0.0; 3], 0); 4];
    for i in 0..n_cells {
        let k = ((xs[i] > 0.0) as usize) * 2 + (ys[i] > 0.0) as usize;
        for c in 0..3 {
            q[k].0[c] += drives[i][c];
        }
        q[k].1 += 1;
    }
    println!("средний восст. драйв по решённым квадрантам (x<0/x>0, y<0/y>0):");
    for k in 0..4 {
        let n = q[k].1.max(1) as f64;
        println!(
            "   x{} y{}: n {:4}  drive [{:6.1} {:6.1} {:6.1}]  сумма {:6.1}",
            if k >= 2 { '+' } else { '-' },
            if k % 2 == 1 { '+' } else { '-' },
            q[k].1,
            q[k].0[0] / n,
            q[k].0[1] / n,
            q[k].0[2] / n,
            (q[k].0[0] + q[k].0[1] + q[k].0[2]) / n
        );
    }

    if file == "--none" {
        return;
    }

    // ====================================================================
    // 3. ИСТИНА ПЕРЕДАТЧИКА -> ПООСЕВОЙ SER
    // ====================================================================
    let data = fs::read(&file).expect("streamed file");
    let symbol_size = symbol_size_for(bpc);
    let enc = FountainEncoder::new(&data, symbol_size);
    let k = enc.k();
    // Поток передатчика БЕСКОНЕЧЕН (frames.rs::Streamer): систематический проход
    // (source + repair каждые 4), затем сплошной repair с инкрементом ESI.
    let emit = emission_order(k, 0);
    let next_repair = k + k / REPAIR_EVERY;
    let esi_at = |j: usize| -> u32 {
        if j < emit.len() {
            emit[j]
        } else {
            next_repair + (j - emit.len()) as u32
        }
    };
    let ti = TransferInfo {
        transfer_length: data.len() as u64,
        symbol_size: symbol_size as u16,
        k,
        checksum: crc32c(&data),
    };
    // счётчик кадров §3.3 даёт seq & 0xFF -> точный якорь по seq
    let (c_start, c_end) = symbol::read_counters(&p, &map, &lin);
    println!(
        "\nK={k}, проход {} кадров, symbol_size={symbol_size}, bpc={bpc}; счётчик кадра: start {c_start}, end {c_end}",
        emit.len().div_ceil(SYMBOLS_PER_FRAME)
    );
    let seq_max: usize = args
        .get(7)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let n_frames = seq_max;
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(n_frames);
    for seq in 0..n_frames {
        let base = seq * SYMBOLS_PER_FRAME;
        let mut bytes = Vec::with_capacity(SYMBOLS_PER_FRAME * symbol_size);
        for i in 0..SYMBOLS_PER_FRAME {
            bytes.extend_from_slice(&enc.symbol(esi_at(base + i)));
        }
        let mut hd = FrameHeader::new(sid, esi_at(base), SYMBOLS_PER_FRAME as u8);
        let t = if seq % 8 == 0 {
            hd.flags |= FLAG_TRANSFER_INFO;
            Some(&ti)
        } else {
            None
        };
        frames.push(l3::build_frame(&hd, t, &bytes, bpc));
    }

    // жёсткие решения приёмника (symbol = (re_bit << 1) | im_bit при 1+1 бит)
    let got: Vec<u8> = (0..n_cells)
        .map(|i| {
            let re = (xs[i] > 0.0) as u8;
            let im = (ys[i] > 0.0) as u8;
            (re << 1) | im
        })
        .collect();
    // сверка с продакшн-демодом
    let got_core = symbol::demod_symbol(&p, &map, &lin);
    let dis = got.iter().zip(&got_core).filter(|(a, b)| a != b).count();
    println!("расхождение с symbol::demod_symbol: {dis} из {n_cells}");

    // поосевой SER по страйпам, ЛОК по каждой оси отдельно
    let rows = l3::STRIPE_ROWS;
    let mut r0 = 0usize;
    let mut tot_re = (0usize, 0usize);
    let mut tot_im = (0usize, 0usize);
    for (si, &rh) in rows.iter().enumerate() {
        let tot = rh * pcn;
        let score = |cells: &[u8], axis_shift: u32| -> usize {
            let mut wrong = 0;
            for r in r0..r0 + rh {
                for c in 0..pcn {
                    let i = r * pcn + c;
                    if ((got[i] >> axis_shift) & 1) != ((cells[i] >> axis_shift) & 1) {
                        wrong += 1;
                    }
                }
            }
            wrong
        };
        // лок по оси Re
        let mut best_re = (usize::MAX, 0usize);
        let mut second_re = usize::MAX;
        let mut best_im = (usize::MAX, 0usize);
        let mut second_im = usize::MAX;
        let mut best_sym = (usize::MAX, 0usize);
        for (s, cells) in frames.iter().enumerate() {
            let e_re = score(cells, 1);
            let e_im = score(cells, 0);
            if e_re < best_re.0 {
                second_re = best_re.0;
                best_re = (e_re, s);
            } else if e_re < second_re {
                second_re = e_re;
            }
            if e_im < best_im.0 {
                second_im = best_im.0;
                best_im = (e_im, s);
            } else if e_im < second_im {
                second_im = e_im;
            }
            let mut ew = 0;
            for r in r0..r0 + rh {
                for c in 0..pcn {
                    let i = r * pcn + c;
                    if got[i] != cells[i] {
                        ew += 1;
                    }
                }
            }
            if ew < best_sym.0 {
                best_sym = (ew, s);
            }
        }
        // поосевой SER на seq, залоченном по ЛУЧШЕЙ оси Re
        let lock = best_re.1;
        let e_re_at_lock = score(&frames[lock], 1);
        let e_im_at_lock = score(&frames[lock], 0);
        println!(
            "страйп {si}: Re лок seq {:2} {:3}/{tot} ({:.3}), 2-й лучший {:3} | Im на том же seq {:3}/{tot} ({:.3}) | Im собств.лок seq {:2} {:3} 2-й {:3} | символ {:3}/{tot} ({:.3})",
            lock,
            e_re_at_lock,
            e_re_at_lock as f64 / tot as f64,
            second_re,
            e_im_at_lock,
            e_im_at_lock as f64 / tot as f64,
            best_im.1,
            best_im.0,
            second_im,
            best_sym.0,
            best_sym.0 as f64 / tot as f64
        );
        if si > 0 {
            tot_re.0 += e_re_at_lock;
            tot_re.1 += tot;
            tot_im.0 += e_im_at_lock;
            tot_im.1 += tot;
        }
        r0 += rh;
    }
    // топ-3 seq по каждой оси на страйп: виден ли НАСТОЯЩИЙ лок
    println!("\nтоп-3 seq по оси (меньше = лучше):");
    let mut r0 = 0usize;
    for (si, &rh) in rows.iter().enumerate() {
        let tot = rh * pcn;
        let mut re_list: Vec<(usize, usize)> = Vec::new();
        let mut im_list: Vec<(usize, usize)> = Vec::new();
        for (s, cells) in frames.iter().enumerate() {
            let (mut ere, mut eim) = (0usize, 0usize);
            for r in r0..r0 + rh {
                for c in 0..pcn {
                    let i = r * pcn + c;
                    if ((got[i] >> 1) & 1) != ((cells[i] >> 1) & 1) {
                        ere += 1;
                    }
                    if (got[i] & 1) != (cells[i] & 1) {
                        eim += 1;
                    }
                }
            }
            re_list.push((ere, s));
            im_list.push((eim, s));
        }
        re_list.sort();
        im_list.sort();
        println!(
            "  страйп {si} (n {tot}): Re {:?} | Im {:?}",
            &re_list[..3],
            &im_list[..3]
        );
        r0 += rh;
    }
    // принудительный seq (аргумент 6): поосевой BER всех страйпов на ОДНОМ кадре
    if let Some(fs) = args.get(6).and_then(|s| s.parse::<usize>().ok()) {
        println!("\nПРИНУДИТЕЛЬНЫЙ seq {fs}: поосевой BER по страйпам");
        let cells = &frames[fs.min(n_frames - 1)];
        let mut r0 = 0usize;
        for (si, &rh) in rows.iter().enumerate() {
            let tot = rh * pcn;
            let (mut ere, mut eim) = (0usize, 0usize);
            for r in r0..r0 + rh {
                for c in 0..pcn {
                    let i = r * pcn + c;
                    if ((got[i] >> 1) & 1) != ((cells[i] >> 1) & 1) {
                        ere += 1;
                    }
                    if (got[i] & 1) != (cells[i] & 1) {
                        eim += 1;
                    }
                }
            }
            println!(
                "  страйп {si}: Re {:3}/{tot} ({:.3})  Im {:3}/{tot} ({:.3})",
                ere,
                ere as f64 / tot as f64,
                eim,
                eim as f64 / tot as f64
            );
            r0 += rh;
        }
    }
    println!(
        "ИТОГО (страйпы 1..7, без sid-зависимого 0): Re BER {:.4} ({}/{}), Im BER {:.4} ({}/{})",
        tot_re.0 as f64 / tot_re.1 as f64,
        tot_re.0,
        tot_re.1,
        tot_im.0 as f64 / tot_im.1 as f64,
        tot_im.0,
        tot_im.1
    );

    // ====================================================================
    // 4. СОЗВЕЗДИЕ, РАСКРАШЕННОЕ ИСТИНОЙ (ГЛОБАЛЬНЫЙ лок по обеим осям)
    // ====================================================================
    // Кадр на экране один, поэтому лок общий на весь символ: берём seq с
    // минимумом суммарных побитовых ошибок. Лок считаем достоверным, если он
    // отрывается от второго кандидата (случайность дала бы ~0.5·2·n).
    let mut best = (usize::MAX, 0usize);
    let mut second = usize::MAX;
    for (s, cells) in frames.iter().enumerate() {
        let mut wrong = 0;
        for i in 0..n_cells {
            if ((got[i] >> 1) & 1) != ((cells[i] >> 1) & 1) {
                wrong += 1;
            }
            if (got[i] & 1) != (cells[i] & 1) {
                wrong += 1;
            }
        }
        if wrong < best.0 {
            second = best.0;
            best = (wrong, s);
        } else if wrong < second {
            second = wrong;
        }
    }
    let locked = (best.0 as f64) < 0.6 * second as f64;
    println!(
        "\nГЛОБАЛЬНЫЙ лок: seq {} ({} бит.ошибок из {}), 2-й кандидат {} -> {}",
        best.1,
        best.0,
        2 * n_cells,
        second,
        if locked { "ДОСТОВЕРНО" } else { "НЕТ ЛОКА" }
    );
    let truth = frames[best.1].clone();
    let have_truth = vec![locked; n_cells];
    let nt = have_truth.iter().filter(|&&t| t).count();
    println!("\nклеток с достоверной истиной: {nt} из {n_cells}");
    if nt > 0 {
        println!("СРЕДНИЕ ПО ИСТИННЫМ СИМВОЛАМ (x, y) и восст. драйв:");
        for s in 0..4u8 {
            let idx: Vec<usize> = (0..n_cells)
                .filter(|&i| have_truth[i] && truth[i] == s)
                .collect();
            if idx.is_empty() {
                continue;
            }
            let n = idx.len() as f64;
            let mxx = idx.iter().map(|&i| xs[i]).sum::<f64>() / n;
            let myy = idx.iter().map(|&i| ys[i]).sum::<f64>() / n;
            let sxx = (idx.iter().map(|&i| (xs[i] - mxx).powi(2)).sum::<f64>() / n).sqrt();
            let syy = (idx.iter().map(|&i| (ys[i] - myy).powi(2)).sum::<f64>() / n).sqrt();
            let mut dr = [0.0f64; 3];
            let mut rw = [0.0f64; 3];
            for &i in &idx {
                for c in 0..3 {
                    dr[c] += drives[i][c];
                    rw[c] += raws[i][c];
                }
            }
            // ожидание
            let ex = if (s >> 1) & 1 == 1 { 1.0 } else { -1.0 };
            let ey = if s & 1 == 1 { 1.0 } else { -1.0 };
            let want = cl.drive(CL_SCALE * ex, CL_SCALE * ey);
            println!(
                "  симв {s} (Re {:+.0}, Im {:+.0}) n {:4}: изм (x {:+.3}±{:.3}, y {:+.3}±{:.3})  drive [{:6.1} {:6.1} {:6.1}] vs ожид [{:6.1} {:6.1} {:6.1}]  raw [{:.3} {:.3} {:.3}]",
                ex, ey, idx.len(), mxx, sxx, myy, syy,
                dr[0] / n, dr[1] / n, dr[2] / n,
                want[0], want[1], want[2],
                rw[0] / n, rw[1] / n, rw[2] / n
            );
        }
    }

    // ====================================================================
    // 5. МАТРИЦА 3×3 (§3.4): объясняет ли ЛИНЕЙНОЕ смешивание каналов провал?
    // ====================================================================
    let gp = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    // цель канала c для драйва d: t = (d/255)^γ_проф — то же пространство,
    // в котором demod_symbol строит свою поканальную модель s = a·t + b.
    let target = |d: [f64; 3]| -> [f64; 3] {
        [
            (d[0] / 255.0).powf(gp[0]),
            (d[1] / 255.0).powf(gp[1]),
            (d[2] / 255.0).powf(gp[2]),
        ]
    };
    // МНК t = N·s + q по набору (s, t): решаем 4×4 нормальные уравнения на канал.
    let fit = |pts: &[([f64; 3], [f64; 3], f64)]| -> ([[f64; 3]; 3], [f64; 3]) {
        let mut nmat = [[0.0f64; 3]; 3];
        let mut q = [0.0f64; 3];
        for c in 0..3 {
            let mut a = [[0.0f64; 4]; 4];
            let mut rhs = [0.0f64; 4];
            for (s, t, wt) in pts {
                let bas = [s[0], s[1], s[2], 1.0];
                for i in 0..4 {
                    for j in 0..4 {
                        a[i][j] += wt * bas[i] * bas[j];
                    }
                    rhs[i] += wt * bas[i] * t[c];
                }
            }
            // гауссово исключение с выбором ведущего
            let mut m = [[0.0f64; 5]; 4];
            for i in 0..4 {
                m[i][..4].copy_from_slice(&a[i]);
                m[i][4] = rhs[i];
            }
            for col in 0..4 {
                let mut piv = col;
                for r in col + 1..4 {
                    if m[r][col].abs() > m[piv][col].abs() {
                        piv = r;
                    }
                }
                m.swap(col, piv);
                let d = m[col][col];
                if d.abs() < 1e-14 {
                    continue;
                }
                for j in col..5 {
                    m[col][j] /= d;
                }
                for r in 0..4 {
                    if r != col {
                        let f = m[r][col];
                        for j in col..5 {
                            m[r][j] -= f * m[col][j];
                        }
                    }
                }
            }
            nmat[c] = [m[0][4], m[1][4], m[2][4]];
            q[c] = m[3][4];
        }
        (nmat, q)
    };
    let apply = |nm: &[[f64; 3]; 3], q: &[f64; 3], s: [f64; 3]| -> [f64; 3] {
        let mut d = [0.0f64; 3];
        for c in 0..3 {
            let t = (nm[c][0] * s[0] + nm[c][1] * s[1] + nm[c][2] * s[2] + q[c]).max(0.0);
            d[c] = (255.0 * t.powf(1.0 / gp[c])).clamp(0.0, 255.0);
        }
        d
    };
    let ber = |dec: &dyn Fn(usize) -> u8| -> (f64, f64) {
        let (mut ere, mut eim, mut n) = (0usize, 0usize, 0usize);
        for i in 0..n_cells {
            if !have_truth[i] {
                continue;
            }
            let g = dec(i);
            if ((g >> 1) & 1) != ((truth[i] >> 1) & 1) {
                ere += 1;
            }
            if (g & 1) != (truth[i] & 1) {
                eim += 1;
            }
            n += 1;
        }
        (ere as f64 / n as f64, eim as f64 / n as f64)
    };
    let show = |name: &str, nm: &[[f64; 3]; 3], q: &[f64; 3]| {
        println!("{name}:");
        for c in 0..3 {
            println!(
                "   [{:+8.4} {:+8.4} {:+8.4}] + {:+8.4}",
                nm[c][0], nm[c][1], nm[c][2], q[c]
            );
        }
    };

    // РЕАЛИЗУЕМЫЕ (без истины) варианты приёмника
    println!("\n=== РЕАЛИЗУЕМЫЕ варианты (истина только для СЧЁТА ошибок) ===");
    let ber_of = |dec: &[u8]| -> (f64, f64) {
        let (mut ere, mut eim, mut n) = (0usize, 0usize, 0usize);
        for i in 0..n_cells {
            if !have_truth[i] {
                continue;
            }
            if ((dec[i] >> 1) & 1) != ((truth[i] >> 1) & 1) {
                ere += 1;
            }
            if (dec[i] & 1) != (truth[i] & 1) {
                eim += 1;
            }
            n += 1;
        }
        (ere as f64 / n as f64, eim as f64 / n as f64)
    };
    // чистые страйпы (0 ошибок -> CRC-16 страйпа пройдёт) + их номера:
    // страйп 0 несёт FrameHeader/TransferInfo, без него кадр отбрасывается.
    let stripes_clean = |dec: &[u8]| -> String {
        let mut r0 = 0usize;
        let mut ok: Vec<usize> = Vec::new();
        for (si, &rh) in rows.iter().enumerate() {
            let mut bad = 0;
            for r in r0..r0 + rh {
                for c in 0..pcn {
                    let i = r * pcn + c;
                    if dec[i] != truth[i] {
                        bad += 1;
                    }
                }
            }
            if bad == 0 {
                ok.push(si);
            }
            r0 += rh;
        }
        format!("{}/8 {:?}", ok.len(), ok)
    };
    let decide = |zx: &[f64], zy: &[f64]| -> Vec<u8> {
        (0..n_cells)
            .map(|i| (((zx[i] > 0.0) as u8) << 1) | (zy[i] > 0.0) as u8)
            .collect()
    };
    // двухмасштабное локальное вычитание DC по сетке payload (аналог
    // demod_symbol_local для 1-битной яркости, но в осях созвездия).
    let local_dc = |v: &[f64]| -> Vec<f64> {
        let mean_rad = |rad: i32| -> Vec<f64> {
            let mut o = vec![0.0f64; n_cells];
            for r in 0..prn as i32 {
                for c in 0..pcn as i32 {
                    let (mut s, mut n) = (0.0, 0.0);
                    for dr in -rad..=rad {
                        for dc in -rad..=rad {
                            let (rr, cx2) = (r + dr, c + dc);
                            if rr >= 0 && rr < prn as i32 && cx2 >= 0 && cx2 < pcn as i32 {
                                s += v[rr as usize * pcn + cx2 as usize];
                                n += 1.0;
                            }
                        }
                    }
                    o[r as usize * pcn + c as usize] = s / n;
                }
            }
            o
        };
        let f = mean_rad(1);
        let cs = mean_rad(12);
        (0..n_cells)
            .map(|i| (v[i] - f[i]) + 0.5 * (v[i] - cs[i]))
            .collect()
    };
    let report = |name: &str, dec: &[u8]| {
        let (a, b) = ber_of(dec);
        println!(
            "  {name:<40} Re BER {a:.4}  Im BER {b:.4}  чистых страйпов {}",
            stripes_clean(dec)
        );
    };
    report("A. базовый (поканальный, реплика v0)", &decide(&xs, &ys));
    report("A'. ПРОДАКШН symbol::demod_symbol", &got_core);
    let gx = xs.iter().sum::<f64>() / n_cells as f64;
    let gy = ys.iter().sum::<f64>() / n_cells as f64;
    let xdc: Vec<f64> = xs.iter().map(|t| t - gx).collect();
    let ydc: Vec<f64> = ys.iter().map(|t| t - gy).collect();
    report("B. базовый + глобальный DC", &decide(&xdc, &ydc));
    report(
        "C. базовый + локальный 2-масштабный DC",
        &decide(&local_dc(&xs), &local_dc(&ys)),
    );
    // (a) подгонка по РЕФЕРЕНСНОЙ СТРОКЕ (то, что можно сделать в проде)
    let mut ref_pts: Vec<([f64; 3], [f64; 3], f64)> = Vec::new();
    for k in 0..16 {
        let n = acc[k].3.max(1) as f64;
        let s = ref_lin[k];
        let want = ref_pattern(k, black_255, white_255);
        ref_pts.push((
            s,
            target([want[0] as f64, want[1] as f64, want[2] as f64]),
            n,
        ));
    }
    let (nr, qr) = fit(&ref_pts);
    show("N по РЕФЕРЕНСНОЙ СТРОКЕ", &nr, &qr);
    let dr_cells: Vec<[f64; 3]> = (0..n_cells).map(|i| apply(&nr, &qr, lins[i])).collect();
    let dec_ref = |i: usize| -> u8 {
        let (x, y) = cl.z_from_drive(dr_cells[i]);
        (((x > 0.0) as u8) << 1) | (y > 0.0) as u8
    };
    let (bre, bim) = ber(&dec_ref);
    println!("   -> Re BER {bre:.4}, Im BER {bim:.4}");
    axis_stats("   реф-3×3", &dr_cells, &cl, &truth, &have_truth);

    // D/E/F: реф-3×3 в связке с DC-центрированием и итеративной
    // решающе-направленной подгонкой матрицы по САМИМ payload-клеткам.
    {
        let zx: Vec<f64> = (0..n_cells)
            .map(|i| cl.z_from_drive(dr_cells[i]).0 / CL_SCALE)
            .collect();
        let zy: Vec<f64> = (0..n_cells)
            .map(|i| cl.z_from_drive(dr_cells[i]).1 / CL_SCALE)
            .collect();
        report("D. реф-3×3", &decide(&zx, &zy));
        let mx2 = zx.iter().sum::<f64>() / n_cells as f64;
        let my2 = zy.iter().sum::<f64>() / n_cells as f64;
        report(
            "E. реф-3×3 + глобальный DC",
            &decide(
                &zx.iter().map(|t| t - mx2).collect::<Vec<_>>(),
                &zy.iter().map(|t| t - my2).collect::<Vec<_>>(),
            ),
        );
        report(
            "F. реф-3×3 + локальный DC",
            &decide(&local_dc(&zx), &local_dc(&zy)),
        );
        // G: решающе-направленная итерация — стартуем с базового решения и
        // 4 раза подгоняем 3×3 по СОБСТВЕННЫМ решениям (истина не нужна).
        let mut dec = decide(&xs, &ys);
        for it in 1..=5 {
            let pts: Vec<([f64; 3], [f64; 3], f64)> = (0..n_cells)
                .map(|i| {
                    let s = dec[i];
                    let ex = if (s >> 1) & 1 == 1 { 1.0 } else { -1.0 };
                    let ey = if s & 1 == 1 { 1.0 } else { -1.0 };
                    (lins[i], target(cl.drive(CL_SCALE * ex, CL_SCALE * ey)), 1.0)
                })
                .collect();
            let (nd, qd) = fit(&pts);
            let dd: Vec<[f64; 3]> = (0..n_cells).map(|i| apply(&nd, &qd, lins[i])).collect();
            let zx: Vec<f64> = (0..n_cells)
                .map(|i| cl.z_from_drive(dd[i]).0 / CL_SCALE)
                .collect();
            let zy: Vec<f64> = (0..n_cells)
                .map(|i| cl.z_from_drive(dd[i]).1 / CL_SCALE)
                .collect();
            dec = decide(&zx, &zy);
            report(&format!("G. решающе-направленная 3×3, итер {it}"), &dec);
            if it == 5 {
                show("   матрица после 5 итераций", &nd, &qd);
                axis_stats("   ре-3×3", &dd, &cl, &truth, &have_truth);
            }
        }
    }

    // (b) ОРАКУЛ: лучшая линейная развязка, подогнанная по САМИМ payload-клеткам
    if nt > 0 {
        let pts: Vec<([f64; 3], [f64; 3], f64)> = (0..n_cells)
            .filter(|&i| have_truth[i])
            .map(|i| {
                let s = truth[i];
                let ex = if (s >> 1) & 1 == 1 { 1.0 } else { -1.0 };
                let ey = if s & 1 == 1 { 1.0 } else { -1.0 };
                (lins[i], target(cl.drive(CL_SCALE * ex, CL_SCALE * ey)), 1.0)
            })
            .collect();
        let (no, qo) = fit(&pts);
        show("N ОРАКУЛ (по истинным payload-клеткам)", &no, &qo);
        let do_cells: Vec<[f64; 3]> = (0..n_cells).map(|i| apply(&no, &qo, lins[i])).collect();
        let dec_or = |i: usize| -> u8 {
            let (x, y) = cl.z_from_drive(do_cells[i]);
            (((x > 0.0) as u8) << 1) | (y > 0.0) as u8
        };
        let (bre, bim) = ber(&dec_or);
        println!("   -> Re BER {bre:.4}, Im BER {bim:.4}");
        axis_stats("   оракул-3×3", &do_cells, &cl, &truth, &have_truth);

        // (c) ОРАКУЛ БЕЗ ФИЗИКИ: прямая линейная развязка в осях (x, y) —
        //     верхняя граница того, что даёт ЛЮБАЯ линейная коррекция.
        let pts2: Vec<([f64; 3], [f64; 3], f64)> = (0..n_cells)
            .filter(|&i| have_truth[i])
            .map(|i| {
                let s = truth[i];
                let ex = if (s >> 1) & 1 == 1 { 1.0 } else { -1.0 };
                let ey = if s & 1 == 1 { 1.0 } else { -1.0 };
                (lins[i], [ex, ey, 0.0], 1.0)
            })
            .collect();
        let (nx, qx) = fit(&pts2);
        show("N ОРАКУЛ прямо в (x, y)", &nx, &qx);
        let (mut ere, mut eim, mut nn) = (0usize, 0usize, 0usize);
        let (mut sx0, mut sx1, mut sy0, mut sy1) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n_cells {
            if !have_truth[i] {
                continue;
            }
            let s = lins[i];
            let xh = nx[0][0] * s[0] + nx[0][1] * s[1] + nx[0][2] * s[2] + qx[0];
            let yh = nx[1][0] * s[0] + nx[1][1] * s[1] + nx[1][2] * s[2] + qx[1];
            if ((xh > 0.0) as u8) != ((truth[i] >> 1) & 1) {
                ere += 1;
            }
            if ((yh > 0.0) as u8) != (truth[i] & 1) {
                eim += 1;
            }
            if (truth[i] >> 1) & 1 == 1 {
                sx1.push(xh);
            } else {
                sx0.push(xh);
            }
            if truth[i] & 1 == 1 {
                sy1.push(yh);
            } else {
                sy0.push(yh);
            }
            nn += 1;
        }
        let ms = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let s = (v.iter().map(|t| (t - m) * (t - m)).sum::<f64>() / v.len() as f64).sqrt();
            (m, s)
        };
        let (m0, s0) = ms(&sx0);
        let (m1, s1) = ms(&sx1);
        let (n0, t0) = ms(&sy0);
        let (n1, t1) = ms(&sy1);
        println!(
            "   -> Re BER {:.4}, Im BER {:.4}; Re: {:+.3}±{:.3} / {:+.3}±{:.3} (Q {:.2}), Im: {:+.3}±{:.3} / {:+.3}±{:.3} (Q {:.2})",
            ere as f64 / nn as f64,
            eim as f64 / nn as f64,
            m0, s0, m1, s1, (m1 - m0) / (s0 + s1),
            n0, t0, n1, t1, (n1 - n0) / (t0 + t1)
        );

        // ------------------------------------------------------------------
        // (d) ЧТО ОСТАЛОСЬ: шум, ISI соседей или пространственный дрейф?
        //     Оракульная линейная проекция lin -> (x, y) на разных базисах.
        // ------------------------------------------------------------------
        println!("\n=== остаток: разложение по базисам (оракул, прямо в (x, y)) ===");
        let neigh = |i: usize| -> [f64; 3] {
            let (r, c) = (i / pcn, i % pcn);
            let mut a = [0.0f64; 3];
            let mut n = 0.0f64;
            for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let (rr, cc) = (r as i32 + dr, c as i32 + dc);
                if rr >= 0 && rr < prn as i32 && cc >= 0 && cc < pcn as i32 {
                    let j = rr as usize * pcn + cc as usize;
                    for k in 0..3 {
                        a[k] += lins[j][k];
                    }
                    n += 1.0;
                }
            }
            for k in 0..3 {
                a[k] /= n.max(1.0);
            }
            a
        };
        let run = |name: &str, basis: &dyn Fn(usize) -> Vec<f64>, idxs: &[usize]| {
            let nb = basis(idxs[0]).len();
            let mut ata = vec![0.0f64; nb * nb];
            let mut atb = vec![0.0f64; nb * 2];
            for &i in idxs {
                let b = basis(i);
                let s = truth[i];
                let tx = if (s >> 1) & 1 == 1 { 1.0 } else { -1.0 };
                let ty = if s & 1 == 1 { 1.0 } else { -1.0 };
                for r in 0..nb {
                    for c in 0..nb {
                        ata[r * nb + c] += b[r] * b[c];
                    }
                    atb[r * 2] += b[r] * tx;
                    atb[r * 2 + 1] += b[r] * ty;
                }
            }
            // гауссово исключение для 2 правых частей
            let mut m = vec![0.0f64; nb * (nb + 2)];
            for r in 0..nb {
                for c in 0..nb {
                    m[r * (nb + 2) + c] = ata[r * nb + c];
                }
                m[r * (nb + 2) + nb] = atb[r * 2];
                m[r * (nb + 2) + nb + 1] = atb[r * 2 + 1];
            }
            for col in 0..nb {
                let mut piv = col;
                for r in col + 1..nb {
                    if m[r * (nb + 2) + col].abs() > m[piv * (nb + 2) + col].abs() {
                        piv = r;
                    }
                }
                for c in 0..nb + 2 {
                    m.swap(col * (nb + 2) + c, piv * (nb + 2) + c);
                }
                let d = m[col * (nb + 2) + col];
                if d.abs() < 1e-12 {
                    continue;
                }
                for c in col..nb + 2 {
                    m[col * (nb + 2) + c] /= d;
                }
                for r in 0..nb {
                    if r != col {
                        let f = m[r * (nb + 2) + col];
                        for c in col..nb + 2 {
                            m[r * (nb + 2) + c] -= f * m[col * (nb + 2) + c];
                        }
                    }
                }
            }
            let wx: Vec<f64> = (0..nb).map(|r| m[r * (nb + 2) + nb]).collect();
            let wy: Vec<f64> = (0..nb).map(|r| m[r * (nb + 2) + nb + 1]).collect();
            let (mut ere, mut eim) = (0usize, 0usize);
            let (mut vx, mut vy) = (0.0f64, 0.0f64);
            for &i in idxs {
                let b = basis(i);
                let xh: f64 = (0..nb).map(|r| wx[r] * b[r]).sum();
                let yh: f64 = (0..nb).map(|r| wy[r] * b[r]).sum();
                let s = truth[i];
                let tx = if (s >> 1) & 1 == 1 { 1.0 } else { -1.0 };
                let ty = if s & 1 == 1 { 1.0 } else { -1.0 };
                if (xh > 0.0) != (tx > 0.0) {
                    ere += 1;
                }
                if (yh > 0.0) != (ty > 0.0) {
                    eim += 1;
                }
                vx += (xh - tx) * (xh - tx);
                vy += (yh - ty) * (yh - ty);
            }
            let n = idxs.len() as f64;
            println!(
                "  {name:<38} n {:4}  Re BER {:.4}  Im BER {:.4}  rms x {:.3}  rms y {:.3}",
                idxs.len(),
                ere as f64 / n,
                eim as f64 / n,
                (vx / n).sqrt(),
                (vy / n).sqrt()
            );
        };
        let all: Vec<usize> = (0..n_cells).filter(|&i| have_truth[i]).collect();
        let b_lin = |i: usize| vec![lins[i][0], lins[i][1], lins[i][2], 1.0];
        run("3×3 глобально", &b_lin, &all);
        let b_row = |i: usize| {
            let r = (i / pcn) as f64 / prn as f64;
            let c = (i % pcn) as f64 / pcn as f64;
            let s = lins[i];
            vec![s[0], s[1], s[2], 1.0, r, c, s[0] * r, s[1] * r, s[2] * r, s[0] * c, s[1] * c, s[2] * c]
        };
        run("3×3 + линейный дрейф по (row, col)", &b_row, &all);
        let b_isi = |i: usize| {
            let s = lins[i];
            let a = neigh(i);
            vec![s[0], s[1], s[2], a[0], a[1], a[2], 1.0]
        };
        run("3×3 + 4-соседа (эквалайзер ISI)", &b_isi, &all);
        let b_both = |i: usize| {
            let r = (i / pcn) as f64 / prn as f64;
            let c = (i % pcn) as f64 / pcn as f64;
            let s = lins[i];
            let a = neigh(i);
            vec![s[0], s[1], s[2], a[0], a[1], a[2], 1.0, r, c, s[0] * r, s[1] * r, s[2] * r, s[0] * c, s[1] * c, s[2] * c]
        };
        run("3×3 + соседи + дрейф", &b_both, &all);
        // пер-страйповая подгонка: чисто пространственная адаптация
        let mut r0 = 0usize;
        for (si, &rh) in rows.iter().enumerate() {
            let idxs: Vec<usize> = (r0 * pcn..(r0 + rh) * pcn)
                .filter(|&i| have_truth[i])
                .collect();
            if idxs.len() > 20 {
                run(&format!("3×3 в страйпе {si}"), &b_lin, &idxs);
            }
            r0 += rh;
        }
    }
}

/// Средние (x, y) по истинным символам для набора восстановленных драйвов.
fn axis_stats(
    tag: &str,
    dr: &[[f64; 3]],
    cl: &ConstLumaMap,
    truth: &[u8],
    have: &[bool],
) {
    for s in 0..4u8 {
        let idx: Vec<usize> = (0..dr.len())
            .filter(|&i| have[i] && truth[i] == s)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let n = idx.len() as f64;
        let z: Vec<(f64, f64)> = idx
            .iter()
            .map(|&i| {
                let (x, y) = cl.z_from_drive(dr[i]);
                (x / CL_SCALE, y / CL_SCALE)
            })
            .collect();
        let mx = z.iter().map(|t| t.0).sum::<f64>() / n;
        let my = z.iter().map(|t| t.1).sum::<f64>() / n;
        let sx = (z.iter().map(|t| (t.0 - mx).powi(2)).sum::<f64>() / n).sqrt();
        let sy = (z.iter().map(|t| (t.1 - my).powi(2)).sum::<f64>() / n).sqrt();
        println!(
            "{tag} симв {s}: x {:+.3}±{:.3}, y {:+.3}±{:.3}  (n {})",
            mx, sx, my, sy, idx.len()
        );
    }
}
