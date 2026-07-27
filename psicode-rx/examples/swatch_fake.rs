//! [ДИАГНОСТИКА] Синтетические YUV-дампы swatch-паттерна — самопроверка
//! анализатора `swatch_diag` БЕЗ телефона.
//!
//! ```text
//! usage: swatch_fake <outdir> <n> <corners|sweep> <кадров> <канал> [cell] [x0] [y0]
//!   канал: список через запятую — clean | mix | blur=<px> | noise=<σ>
//!   напр.: clean, mix, mix,blur=9, mix,blur=9,noise=2
//! ```
//! Модель тракта (в том же порядке, что живой): дисплей `drive^γ` -> ЛИНЕЙНОЕ
//! СМЕШИВАНИЕ каналов 3×3 (ISP CCM + оптика) -> усиление/подъём чёрного ->
//! тон-кривая камеры `s^(1/γ_кам)` -> BT.601 limited range -> 4:2:0 (box 2×2,
//! опционально ещё и НЧ-фильтр хромы) -> шум.
//!
//! Смысл: прогнать лестницу на канале с ИЗВЕСТНЫМИ параметрами и убедиться, что
//! анализатор возвращает именно их — тогда числам на живых снимках можно верить.

use psicode_core::swatch::render_swatch;
use psicode_rx::tx_chromatic_profile;
use std::fs;

const W: usize = 1920;
const H: usize = 1080;

/// Смешивание каналов «как на живом A22»: строки суммируются к 1, поэтому
/// нейтраль остаётся нейтралью и якоря K/W §3.4 смешивания НЕ видят.
const MIX: [[f64; 3]; 3] = [
    [0.78, 0.14, 0.08],
    [0.12, 0.62, 0.26],
    [0.07, 0.28, 0.65],
];
const MIX_ID: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!(
            "usage: swatch_fake <outdir> <n> <corners|sweep> <кадров> <канал> [cell] [x0] [y0]
             канал: список через запятую, напр. clean | mix | mix,blur=9 | mix,blur=9,noise=2"
        );
        std::process::exit(2);
    }
    let outdir = &a[1];
    let n: usize = a[2].parse().expect("n");
    let sweep = a[3] == "sweep";
    let nframes: u64 = a[4].parse().expect("кадров");
    let chan = a[5].clone();
    let cell: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(12);
    let x0: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(760);
    let y0: usize = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(60);

    // спецификация канала: список через запятую — mix, blur=<px>, noise=<σ>
    let mut mix = MIX_ID;
    let mut chroma_blur = 0usize;
    let mut noise = 0.0f64;
    for tok in chan.split(',') {
        let (k, v) = match tok.split_once('=') {
            Some((k, v)) => (k, v.parse::<f64>().unwrap_or(0.0)),
            None => (tok, 0.0),
        };
        match k {
            "clean" => {}
            "mix" => mix = MIX,
            "blur" => chroma_blur = v as usize,
            "noise" => noise = v,
            other => panic!("неизвестный токен канала: {other}"),
        }
    }
    fs::create_dir_all(outdir).expect("mkdir");

    let mut p = tx_chromatic_profile();
    p.cell_size_px = cell as u8;
    let gd = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    // тон-кривая камеры: близкая к живому замеру (R≈G≈2.1, B≈3.2)
    let gc = [2.1f64, 2.2, 3.2];
    let (gain, black) = (0.8f64, 0.03f64);

    println!(
        "swatch_fake: n={n}, {}, кадров {nframes}, канал {chan}, cell {cell}, символ в ({x0}, {y0})",
        if sweep { "sweep" } else { "corners" }
    );
    println!(
        "  смешивание R[{:.2} {:.2} {:.2}] G[{:.2} {:.2} {:.2}] B[{:.2} {:.2} {:.2}], \
         блюр хромы {chroma_blur} px, шум σ {noise}",
        mix[0][0], mix[0][1], mix[0][2], mix[1][0], mix[1][1], mix[1][2], mix[2][0], mix[2][1],
        mix[2][2]
    );

    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    let nextf = move |r: &mut u64| -> f64 {
        // xorshift64 -> [-1, 1)
        *r ^= *r << 13;
        *r ^= *r >> 7;
        *r ^= *r << 17;
        (*r >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };

    for f in 0..nframes {
        let fr = render_swatch(&p, n, f, sweep, psicode_core::swatch::SWATCH_GUARD);
        let side = fr.size_px;
        assert!(x0 + side <= W && y0 + side <= H, "символ не влезает в кадр");

        // 1. сцена в «сырых» RGB камеры [0,1]. Тракт применяем к КАЖДОМУ
        //    drive-пикселю сцены, включая фон: иначе фон не похож на реальный
        //    снимок экрана и грубая стадия детекции ведёт себя иначе, чем в
        //    поле (символ на живом кадре делит экран с другими окнами).
        let through = |d: [u8; 3]| -> [f64; 3] {
            let lin = [
                (d[0] as f64 / 255.0).powf(gd[0]),
                (d[1] as f64 / 255.0).powf(gd[1]),
                (d[2] as f64 / 255.0).powf(gd[2]),
            ];
            let mut o = [0.0f64; 3];
            for c in 0..3 {
                let v = mix[c][0] * lin[0] + mix[c][1] * lin[1] + mix[c][2] * lin[2];
                o[c] = (gain * v + black).clamp(0.0, 1.0).powf(1.0 / gc[c]);
            }
            o
        };
        // фон: средне-серый рабочий стол плюс «окна» — тёмные и светлые
        // прямоугольники слева/снизу, чтобы карта активности видела сцену,
        // похожую на живой снимок.
        let bg = through([96, 96, 96]);
        let mut rgb = vec![bg; W * H];
        let clutter = through([20, 20, 20]);
        let clutter2 = through([220, 220, 220]);
        for (rx, ry, rw, rh, dark) in [
            (40usize, 40usize, 640usize, 900usize, true),
            (80, 120, 520, 40, false),
            (80, 300, 440, 40, false),
            (60, 960, 1800, 90, true),
            (1660, 100, 220, 700, false),
        ] {
            let c = if dark { clutter } else { clutter2 };
            for j in ry..(ry + rh).min(H) {
                for i in rx..(rx + rw).min(W) {
                    rgb[j * W + i] = c;
                }
            }
        }
        for j in 0..side {
            for i in 0..side {
                rgb[(y0 + j) * W + (x0 + i)] = through(fr.rgb[j * side + i]);
            }
        }

        // 2. RGB -> Y/Cb/Cr (BT.601 limited range), хрома в полном разрешении
        let mut yp = vec![0.0f64; W * H];
        let mut cb = vec![0.0f64; W * H];
        let mut cr = vec![0.0f64; W * H];
        for i in 0..W * H {
            let (r, g, b) = (rgb[i][0] * 255.0, rgb[i][1] * 255.0, rgb[i][2] * 255.0);
            yp[i] = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0;
            cb[i] = 128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0;
            cr[i] = 128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0;
        }
        // 3. НЧ-фильтр хромы ISP (box радиуса chroma_blur) — до субдискретизации
        if chroma_blur > 0 {
            cb = box_blur(&cb, W, H, chroma_blur);
            cr = box_blur(&cr, W, H, chroma_blur);
        }
        // 4. 4:2:0: box 2×2 по хроме
        let (cw, chh) = (W / 2, H / 2);
        let mut u = vec![0u8; cw * chh * 2];
        let mut v = vec![0u8; cw * chh * 2];
        for j in 0..chh {
            for i in 0..cw {
                let mut su = 0.0;
                let mut sv = 0.0;
                for dj in 0..2 {
                    for di in 0..2 {
                        let idx = (2 * j + dj) * W + (2 * i + di);
                        su += cb[idx];
                        sv += cr[idx];
                    }
                }
                let q = |x: f64, r: &mut u64| -> u8 {
                    (x / 4.0 + noise * nextf(r)).round().clamp(0.0, 255.0) as u8
                };
                u[j * (cw * 2) + i * 2] = q(su, &mut rng);
                v[j * (cw * 2) + i * 2] = q(sv, &mut rng);
            }
        }
        let mut yb = vec![0u8; W * H];
        for i in 0..W * H {
            yb[i] = (yp[i] + noise * nextf(&mut rng))
                .round()
                .clamp(0.0, 255.0) as u8;
        }

        let prefix = format!("{outdir}/dump{f}");
        fs::write(format!("{prefix}.y"), &yb).unwrap();
        fs::write(format!("{prefix}.u"), &u).unwrap();
        fs::write(format!("{prefix}.v"), &v).unwrap();
        // формат .meta как у живых дампов телефона
        fs::write(
            format!("{prefix}.meta"),
            format!(
                "{W} {H} {W} {} 2 {} 2 {} {} {}\n",
                cw * 2,
                cw * 2,
                W * H,
                u.len(),
                v.len()
            ),
        )
        .unwrap();
    }
    println!("готово: {nframes} дамп(ов) в {outdir}");
}

/// Box-фильтр радиуса `r` (разделимый), края зажаты.
fn box_blur(src: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut tmp = vec![0.0f64; w * h];
    let mut out = vec![0.0f64; w * h];
    for j in 0..h {
        for i in 0..w {
            let (a, b) = (i.saturating_sub(r), (i + r).min(w - 1));
            let mut s = 0.0;
            for k in a..=b {
                s += src[j * w + k];
            }
            tmp[j * w + i] = s / (b - a + 1) as f64;
        }
    }
    for i in 0..w {
        for j in 0..h {
            let (a, b) = (j.saturating_sub(r), (j + r).min(h - 1));
            let mut s = 0.0;
            for k in a..=b {
                s += tmp[k * w + i];
            }
            out[j * w + i] = s / (b - a + 1) as f64;
        }
    }
    out
}
