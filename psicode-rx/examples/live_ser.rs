//! Живой SER: демодуляция сырого YUV-дампа телефона против ТОЧНЫХ кадров
//! передатчика (файл + session_id известны) — измерение реального SER канала.
//! usage: live_ser <dump-prefix> <streamed-file> <session_id_hex>
//!   пример: live_ser .../dump0 SPEC.md 38AE9566

use psicode_core::fountain::{crc32c, FountainEncoder};
use psicode_core::l3::{self, FrameHeader, TransferInfo, FLAG_TRANSFER_INFO};
use psicode_core::{detect, symbol, tone};
use psicode_rx::tx_default_profile;
use psicode_rx::yuv::YuvFrame;
use std::fs;

const SYMBOLS_PER_FRAME: usize = 8;
const REPAIR_EVERY: u32 = 4;

/// symbol_size как в tx: (ёмкость кадра − заголовок 12 − TransferInfo 14) / 8.
fn symbol_size_for(bpc: u32) -> usize {
    let cap: usize = l3::STRIPE_ROWS
        .iter()
        .map(|&r| (r * l3::PAYLOAD_COLS * bpc as usize - 16) / 8)
        .sum();
    (cap - l3::FRAME_HEADER_LEN - l3::TRANSFER_INFO_LEN) / SYMBOLS_PER_FRAME
}

fn emission_order(k: u32, total: usize) -> Vec<u32> {
    let mut v = Vec::with_capacity(total);
    let (mut src, mut rep, mut since) = (0u32, k, 0u32);
    while v.len() < total {
        if src < k {
            v.push(src);
            src += 1;
            since += 1;
            if since == REPAIR_EVERY {
                if v.len() < total {
                    v.push(rep);
                    rep += 1;
                }
                since = 0;
            }
        } else {
            v.push(rep);
            rep += 1;
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prefix = args.get(1).expect("usage: live_ser <dump-prefix> <file> <sid-hex>");
    let file = args.get(2).expect("file");
    let sid = u32::from_str_radix(args.get(3).expect("sid"), 16).expect("hex sid");

    // --- дамп ---
    let meta = fs::read_to_string(format!("{prefix}.meta")).expect("meta");
    let m: Vec<usize> = meta.split_whitespace().map(|t| t.parse().unwrap()).collect();
    let (mut w, mut h, mut y_stride, mut uv_stride, uv_px) = (m[0], m[1], m[2], m[3], m[4]);
    let mut yb = fs::read(format!("{prefix}.y")).expect("y");
    let mut ub = fs::read(format!("{prefix}.u")).expect("u");
    let mut vb = fs::read(format!("{prefix}.v")).expect("v");

    // необязательный кроп "x0,y0,w,h" (чётные): холодная грубая стадия на
    // загромождённом кадре промахивается — кроп вокруг символа решает.
    if let Some(spec) = args.get(4) {
        let c: Vec<usize> = spec.split(',').map(|t| t.parse().unwrap()).collect();
        let (x0, y0, cw, ch) = (c[0] & !1, c[1] & !1, c[2] & !1, c[3] & !1);
        let mut ny = vec![0u8; cw * ch];
        for j in 0..ch {
            let src = (y0 + j) * y_stride + x0;
            ny[j * cw..(j + 1) * cw].copy_from_slice(&yb[src..src + cw]);
        }
        // хрома: полразрешения; сохраняем pixel stride как в исходнике
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
    println!("dump {w}x{h} yS={y_stride} uvS={uv_stride}/{uv_px}");

    // --- детекция на Y-плоскости ---
    let luma: Vec<f32> = (0..w * h).map(|i| fr.y_norm(i % w, i / w)).collect();
    // полный офлайн-детект (с тяжёлым fallback) — максимум качества геометрии;
    // если он не нашёл, пробуем телефонный acquire.
    let mut det = detect::detect_symbol(w, h, &luma)
        .or_else(|_| detect::detect_symbol_acquire(w, h, &luma))
        .expect("не найдено");
    println!("детекция: score {:.4}, rot {}", det.score, det.rotation_quadrants);
    // самоуточнение: итерации track по тому же кадру (как тёплый трекинг
    // телефона, который сходится за несколько кадров).
    for _ in 0..10 {
        match detect::track_symbol(w, h, &luma, &det) {
            Ok(d2) if d2.score > det.score + 1e-4 => det = d2,
            _ => break,
        }
    }
    println!("после уточнения: score {:.4}", det.score);

    let p = tx_default_profile();
    let map = detect::frame_map(&p, &det);

    // сырой (гамма-кодированный) RGB, билинейно
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
    let got = symbol::demod_symbol(&p, &map, &lin);

    // --- режим статичного кадра tx single: истина = детерминированные клетки ---
    if file == "--single" {
        let bpc = symbol::bits_per_cell(&p);
        let mask: u16 = if bpc >= 16 { u16::MAX } else { (1u16 << bpc) - 1 };
        let n = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
        let mut state = 0x0D15_EA5E_5EED_1234u64;
        let truth: Vec<u8> = (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (((z ^ (z >> 31)) >> 24) as u16 & mask) as u8
            })
            .collect();
        let wrong = got.iter().zip(&truth).filter(|(a, b)| a != b).count();
        println!("SINGLE: SER {:.4} ({} / {})", wrong as f64 / n as f64, wrong, n);
        let rows = [7usize, 7, 7, 7, 7, 7, 7, 6];
        let mut r0 = 0usize;
        for (si, &rh) in rows.iter().enumerate() {
            let mut sw = 0usize;
            for r in r0..r0 + rh {
                for c in 0..57 {
                    if got[r * 57 + c] != truth[r * 57 + c] {
                        sw += 1;
                    }
                }
            }
            println!("  страйп {si}: {sw} из {} (SER {:.3})", rh * 57, sw as f64 / (rh * 57) as f64);
            r0 += rh;
        }
        return;
    }

    // --- реконструкция кадров передачи ---
    let data = fs::read(file).expect("streamed file");
    let symbol_size = symbol_size_for(symbol::bits_per_cell(&p));
    let enc = FountainEncoder::new(&data, symbol_size);
    let k = enc.k();
    let n_frames = ((k as usize + k as usize / 4 + SYMBOLS_PER_FRAME) / SYMBOLS_PER_FRAME) + 1;
    let emit = emission_order(k, n_frames * SYMBOLS_PER_FRAME);
    let ti = TransferInfo {
        transfer_length: data.len() as u64,
        symbol_size: symbol_size as u16,
        k,
        checksum: crc32c(&data),
    };
    let bpc = symbol::bits_per_cell(&p);
    println!("K={k}, кадров/проход={n_frames}, bpc={bpc}");

    // клетки всех кадров прохода
    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(n_frames);
    for seq in 0..n_frames {
        let mut bytes = Vec::with_capacity(SYMBOLS_PER_FRAME * symbol_size);
        for i in 0..SYMBOLS_PER_FRAME {
            bytes.extend_from_slice(&enc.symbol(emit[seq * SYMBOLS_PER_FRAME + i]));
        }
        let mut hd = FrameHeader::new(sid, emit[seq * SYMBOLS_PER_FRAME], SYMBOLS_PER_FRAME as u8);
        let t = if seq % 8 == 0 {
            hd.flags |= FLAG_TRANSFER_INFO;
            Some(&ti)
        } else {
            None
        };
        frames.push(l3::build_frame(&hd, t, &bytes, bpc));
    }
    let n = got.len();
    let best = (0..n_frames)
        .map(|s| {
            (
                got.iter().zip(&frames[s]).filter(|(a, b)| a != b).count(),
                s,
            )
        })
        .min()
        .unwrap();
    println!(
        "лучший ЦЕЛЫЙ кадр: seq {}, SER {:.4} ({} / {})",
        best.1,
        best.0 as f64 / n as f64,
        best.0,
        n
    );
    // ПЕР-СТРАЙПОВО: свой лучший seq у каждого страйпа (рваный снимок = разные
    // seq сверху и снизу; SER страйпа против его собственного seq = чистое
    // качество канала без смешения кадров).
    let rows = [7usize, 7, 7, 7, 7, 7, 7, 6];
    let mut r0 = 0usize;
    for (si, &rh) in rows.iter().enumerate() {
        let mut sbest = (usize::MAX, 0usize);
        for (s, cells) in frames.iter().enumerate() {
            let mut wrong = 0usize;
            for r in r0..r0 + rh {
                for c in 0..57 {
                    let idx = r * 57 + c;
                    if got[idx] != cells[idx] {
                        wrong += 1;
                    }
                }
            }
            if wrong < sbest.0 {
                sbest = (wrong, s);
            }
        }
        let tot = rh * 57;
        println!(
            "  страйп {si}: лучший seq {:2}, {:3} ошибок из {tot} (SER {:.3})",
            sbest.1,
            sbest.0,
            sbest.0 as f64 / tot as f64
        );
        r0 += rh;
    }
}
