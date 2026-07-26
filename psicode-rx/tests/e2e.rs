//! Десктоп-тесты приёмного тракта psicode-rx (rlib-путь, без JNI, §8):
//! (a) один кадр -> RGB->YUV601(оба pixel-stride) -> детекция + точный демод;
//! (b) сквозная передача ~2 КБ -> кадры L3 -> YUV -> RxSession -> фонтан+CRC OK;
//! (c) мусорные кадры не роняют сессию и держат состояние здравым;
//! (d) замер времени на синтетическом кадре 1920×1080.
//!
//! tx-сторона (фрейминг/фонтан) зеркалит psicode-sim/src/transfer.rs; YUV-хелперы
//! используют прямое BT.601 limited-range, обратное которому делает rx.

use std::time::Instant;

use psicode_core::fountain::{crc32c, FountainEncoder};
use psicode_core::l3::{self, FrameHeader, TransferInfo};
use psicode_core::symbol::{self, render_symbol_counter};
use psicode_rx::{tx_default_profile, RxSession, RxState};

// ---------------------------------------------------------------------------
// YUV_420_888 хелперы (прямое BT.601 limited-range; rx делает обратное)
// ---------------------------------------------------------------------------

/// RGB (0..255) -> YCbCr limited-range BT.601 (studio swing). Обратно к
/// коэффициентам rx::yuv (матрица-инверс).
fn rgb_to_yuv601(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (rf, gf, bf) = (r as f32, g as f32, b as f32);
    let y = 16.0 + 0.257 * rf + 0.504 * gf + 0.098 * bf;
    let u = 128.0 - 0.148 * rf - 0.291 * gf + 0.439 * bf;
    let v = 128.0 + 0.439 * rf - 0.368 * gf - 0.071 * bf;
    let q = |x: f32| x.round().clamp(0.0, 255.0) as u8;
    (q(y), q(u), q(v))
}

/// Плоскости кадра YUV_420_888.
struct Planes {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
    uv_pixel_stride: usize,
}

/// RGB-кадр (w·h) -> YUV_420_888. `semi_planar=false` — планарный (I420,
/// pixel_stride 1, отдельные плотные U/V); `true` — полупланарный (NV12-стиль,
/// pixel_stride 2, чередование U/V в одном буфере, u→байт U, v→байт V).
/// w и h предполагаются чётными (кадры символа таковы).
fn to_yuv420(rgb: &[[u8; 3]], w: usize, h: usize, semi_planar: bool) -> Planes {
    let mut y = vec![0u8; w * h];
    let cw = w / 2;
    let ch = h / 2;
    // полная хрома по пикселям (усредним 2×2)
    let mut uf = vec![0u8; cw * ch];
    let mut vf = vec![0u8; cw * ch];
    for j in 0..h {
        for i in 0..w {
            let px = rgb[j * w + i];
            let (yy, _, _) = rgb_to_yuv601(px[0], px[1], px[2]);
            y[j * w + i] = yy;
        }
    }
    for cj in 0..ch {
        for ci in 0..cw {
            let mut su = 0u32;
            let mut sv = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let px = rgb[(cj * 2 + dy) * w + (ci * 2 + dx)];
                    let (_, uu, vv) = rgb_to_yuv601(px[0], px[1], px[2]);
                    su += uu as u32;
                    sv += vv as u32;
                }
            }
            uf[cj * cw + ci] = (su / 4) as u8;
            vf[cj * cw + ci] = (sv / 4) as u8;
        }
    }
    if !semi_planar {
        Planes {
            y,
            u: uf,
            v: vf,
            y_stride: w,
            uv_stride: cw,
            uv_pixel_stride: 1,
        }
    } else {
        // NV12-стиль: [U0 V0 U1 V1 ...], u=&buf[0..], v=&buf[1..]
        let mut buf = vec![0u8; cw * ch * 2];
        for k in 0..cw * ch {
            buf[k * 2] = uf[k];
            buf[k * 2 + 1] = vf[k];
        }
        let u = buf.clone();
        let mut v = buf;
        v.remove(0); // сдвиг на 1 байт -> v указывает на первый байт V
        Planes {
            y,
            u,
            v,
            y_stride: w,
            uv_stride: cw * 2,
            uv_pixel_stride: 2,
        }
    }
}

/// Вставить квадратный символ (sz·sz) по центру серого холста w·h.
fn paste_center(canvas: &mut [[u8; 3]], w: usize, h: usize, sym: &[[u8; 3]], sz: usize) {
    let ox = (w - sz) / 2;
    let oy = (h - sz) / 2;
    for j in 0..sz {
        for i in 0..sz {
            canvas[(oy + j) * w + (ox + i)] = sym[j * sz + i];
        }
    }
}

/// splitmix64-подобная псевдослучайность (без внешних зависимостей).
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// tx-сторона фрейминга (зеркалит transfer.rs) — для сквозного теста
// ---------------------------------------------------------------------------

const SYMBOL_SIZE: usize = 140;
const SYMBOLS_PER_FRAME: usize = 8;
const REPAIR_EVERY: u32 = 4;

fn emission_order(k: u32, n_frames: usize) -> Vec<u32> {
    let total = n_frames * SYMBOLS_PER_FRAME;
    let mut v = Vec::with_capacity(total);
    let mut src = 0u32;
    let mut rep = k;
    let mut since = 0u32;
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
    v.truncate(total);
    v
}

fn build_symbol_bytes(enc: &FountainEncoder, emit: &[u32], base: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(SYMBOLS_PER_FRAME * SYMBOL_SIZE);
    for i in 0..SYMBOLS_PER_FRAME {
        out.extend_from_slice(&enc.symbol(emit[base + i]));
    }
    out
}

// ---------------------------------------------------------------------------
// (a) один кадр: детекция + точный демод, оба pixel-stride
// ---------------------------------------------------------------------------

#[test]
fn single_frame_detects_and_demods_exactly_both_strides() {
    let mut p = tx_default_profile();
    p.cell_size_px = 16;
    let bpc = symbol::bits_per_cell(&p); // 3

    // L3-обрамлённый кадр: точный демод <=> все 8 страйпов проходят CRC-16.
    let hdr = FrameHeader::new(0xA1B2_C3D4, 0, SYMBOLS_PER_FRAME as u8);
    let ti = TransferInfo {
        transfer_length: 1000,
        symbol_size: SYMBOL_SIZE as u16,
        k: SYMBOLS_PER_FRAME as u32,
        checksum: 0,
    };
    let sym_bytes: Vec<u8> = (0..SYMBOLS_PER_FRAME * SYMBOL_SIZE)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
        .collect();
    let cells = l3::build_frame(&hdr, Some(&ti), &sym_bytes, bpc);
    let frame = render_symbol_counter(&p, &cells, 0);
    let sz = frame.size_px;

    for semi in [false, true] {
        let pl = to_yuv420(&frame.rgb, sz, sz, semi);
        let mut rx = RxSession::new(p);
        let st = rx.process_frame_yuv(
            &pl.y, &pl.u, &pl.v, sz, sz, pl.y_stride, pl.uv_stride, pl.uv_pixel_stride,
        );
        assert!(st.detected, "semi={semi}: ЗЧ-рамка не найдена");
        assert_eq!(st.rotation, 0, "semi={semi}: неожиданный поворот");
        assert!(st.score > 0.9, "semi={semi}: слабый score {}", st.score);
        // точный демод всей payload-сетки: все 8 страйпов CRC-валидны.
        assert_eq!(
            st.stripes_ok, 8,
            "semi={semi}: демод не точен (страйпов OK {}/8)",
            st.stripes_ok
        );
    }
}

// ---------------------------------------------------------------------------
// (b) сквозная передача ~2 КБ: фонтан завершается, CRC OK, объект совпал
// ---------------------------------------------------------------------------

#[test]
fn streaming_transfer_completes_crc_ok() {
    let mut p = tx_default_profile();
    p.cell_size_px = 16;
    let bpc = symbol::bits_per_cell(&p);
    let sz_cells = symbol::GRID + 2 * p.quiet_zone_cells() as usize;
    let sz = sz_cells * p.cell_size_px as usize;

    // ~2 КБ псевдослучайной нагрузки
    let mut s = 0x0D15_EA5E_5EED_1234u64;
    let payload: Vec<u8> = (0..2000).map(|_| (splitmix(&mut s) >> 24) as u8).collect();
    let checksum = crc32c(&payload);
    let enc = FountainEncoder::new(&payload, SYMBOL_SIZE);
    let k = enc.k();
    let session = 0x5053_4930u32;

    let frame_cap = (10 * k as usize).div_ceil(SYMBOLS_PER_FRAME) + 16;
    let emit = emission_order(k, frame_cap + 2);

    let mut rx = RxSession::new(p);
    let mut got: Option<Vec<u8>> = None;
    let mut seq = 0u32;
    while (seq as usize) < frame_cap && got.is_none() {
        let base = seq as usize * SYMBOLS_PER_FRAME;
        let sym = build_symbol_bytes(&enc, &emit, base);
        let ti = if seq % 8 == 0 {
            Some(TransferInfo {
                transfer_length: payload.len() as u64,
                symbol_size: SYMBOL_SIZE as u16,
                k,
                checksum,
            })
        } else {
            None
        };
        let hdr = FrameHeader::new(session, emit[base], SYMBOLS_PER_FRAME as u8);
        let cells = l3::build_frame(&hdr, ti.as_ref(), &sym, bpc);
        let frame = render_symbol_counter(&p, &cells, (seq & 0xFF) as u8);

        let pl = to_yuv420(&frame.rgb, sz, sz, false);
        let st = rx.process_frame_yuv(
            &pl.y, &pl.u, &pl.v, sz, sz, pl.y_stride, pl.uv_stride, pl.uv_pixel_stride,
        );
        assert!(st.detected, "кадр {seq}: не найдено");
        if st.done {
            got = rx.take_result();
        }
        seq += 1;
    }

    assert_eq!(rx.state(), RxState::Done, "передача не завершилась за {seq} кадров");
    let obj = got.expect("объект должен быть собран");
    assert_eq!(obj, payload, "собранный объект не совпал с нагрузкой (CRC-32C)");
    // забран ровно один раз
    assert!(rx.take_result().is_none(), "результат должен забираться однократно");
    eprintln!("(b) сквозная передача: K={k}, завершено за {seq} кадров, CRC-32C OK");
}

// ---------------------------------------------------------------------------
// (c) мусорные кадры не роняют сессию и держат состояние здравым
// ---------------------------------------------------------------------------

#[test]
fn garbage_frames_do_not_panic_and_keep_state_sane() {
    let mut rx = RxSession::with_cell(16);
    let mut s = 0xDEAD_BEEF_0000_0001u64;

    // разные размеры/шаги/содержимое, включая вырожденные. Немного итераций:
    // захват на мусоре дорог (align без лока), а контракт тут — НЕ ПАНИКА и НЕ
    // ложное завершение, что доказывается и малым набором разнородных кадров.
    for it in 0..12 {
        let w = 8 + (splitmix(&mut s) % 1000) as usize;
        let h = 8 + (splitmix(&mut s) % 700) as usize;
        let ylen = (splitmix(&mut s) % (w * h + 1) as u64) as usize; // иногда короче w·h
        let yb: Vec<u8> = (0..ylen).map(|_| (splitmix(&mut s) >> 24) as u8).collect();
        let ub: Vec<u8> = (0..(ylen / 4).max(1)).map(|_| (splitmix(&mut s) >> 24) as u8).collect();
        let vb = ub.clone();
        let y_stride = w + (splitmix(&mut s) % 8) as usize;
        let uv_stride = (w / 2).max(1);
        let uv_ps = if splitmix(&mut s) & 1 == 0 { 1 } else { 2 };
        let st = rx.process_frame_yuv(&yb, &ub, &vb, w, h, y_stride, uv_stride, uv_ps);
        // контракт: не паникует; на мусоре не «завершается» ложно
        assert!(!st.done, "итерация {it}: ложное завершение на мусоре");
        assert!(!st.crc_ok, "итерация {it}: ложный CRC на мусоре");
    }
    assert_ne!(rx.state(), RxState::Done, "мусор не должен завершать сессию");
    assert!(rx.take_result().is_none(), "мусор не должен давать результат");

    // вырожденные размеры
    let _ = rx.process_frame_yuv(&[], &[], &[], 0, 0, 0, 0, 0);
    let _ = rx.process_frame_yuv(&[0u8; 4], &[128], &[128], 2, 2, 2, 1, 1);
}

// ---------------------------------------------------------------------------
// (d) замер времени на кадре 1920×1080: ЗАХВАТ (раз на лок) vs СЛЕЖЕНИЕ (кадр)
// ---------------------------------------------------------------------------

#[test]
fn timing_report_1080p() {
    let (w, h) = (1920usize, 1080usize);
    let mut p = tx_default_profile();
    // символ ~1040 px по стороне вписывается в 1080p
    let sz_cells = symbol::GRID + 2 * p.quiet_zone_cells() as usize; // 69
    p.cell_size_px = (1040 / sz_cells) as u8; // ~15
    let sz = sz_cells * p.cell_size_px as usize;
    let bpc = symbol::bits_per_cell(&p);

    // валидный L3-кадр, чтобы измерять полный локающий путь (демод + L3 + фонтан)
    let hdr = FrameHeader::new(0x1234_5678, 0, SYMBOLS_PER_FRAME as u8);
    let ti = TransferInfo {
        transfer_length: 1000,
        symbol_size: SYMBOL_SIZE as u16,
        k: SYMBOLS_PER_FRAME as u32,
        checksum: 0,
    };
    let sym_bytes: Vec<u8> = (0..SYMBOLS_PER_FRAME * SYMBOL_SIZE).map(|i| i as u8).collect();
    let cells = l3::build_frame(&hdr, Some(&ti), &sym_bytes, bpc);
    let frame = render_symbol_counter(&p, &cells, 0);

    let mut canvas = vec![[128u8; 3]; w * h];
    paste_center(&mut canvas, w, h, &frame.rgb, sz);
    let pl = to_yuv420(&canvas, w, h, false);
    let run = |rx: &mut RxSession| {
        rx.process_frame_yuv(&pl.y, &pl.u, &pl.v, w, h, pl.y_stride, pl.uv_stride, pl.uv_pixel_stride)
    };

    // ЗАХВАТ: свежая сессия, кадр 1 (acquire; раз на лок).
    let mut rx = RxSession::new(p);
    let t0 = Instant::now();
    let st = run(&mut rx);
    let acquire_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(st.detected, "1080p: символ по центру должен захватиться (score {})", st.score);

    // СЛЕЖЕНИЕ: та же сессия, кадры 2.. (track; каждый кадр после лока).
    let iters = 8;
    let t1 = Instant::now();
    let mut track_st = st;
    for _ in 0..iters {
        track_st = run(&mut rx);
    }
    let track_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    assert!(track_st.detected, "слежение должно держать лок");

    // ПОИСК/мусор: одна свежая сессия на мусорном кадре (acquire не находит).
    let mut sgen = 0x1111_2222_3333_4444u64;
    let garbage: Vec<u8> = (0..w * h).map(|_| (splitmix(&mut sgen) >> 24) as u8).collect();
    let gu = vec![128u8; (w / 2) * (h / 2)];
    let mut gr = RxSession::new(p);
    let t2 = Instant::now();
    let gst = gr.process_frame_yuv(&garbage, &gu, &gu, w, h, w, w / 2, 1);
    let search_ms = t2.elapsed().as_secs_f64() * 1000.0;
    assert!(!gst.detected, "мусор не должен детектиться");

    eprintln!(
        "(d) 1080p таймингы (десктоп i7; телефон ~3-5×): ЗАХВАТ {acquire_ms:.0} мс (раз на лок), \
         СЛЕЖЕНИЕ {track_ms:.1} мс/кадр, поиск/мусор {search_ms:.0} мс; px/клетку≈{:.1}, score {:.3}",
        st.px_per_cell, st.score
    );
}
