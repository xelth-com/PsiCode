//! [ИЗМЕРЕНИЕ] Выравнивание МЕЖКЛЕТОЧНОЙ ИНТЕРФЕРЕНЦИИ на РЕАЛЬНЫХ снимках.
//!
//! Отвечает ровно на шесть вопросов постановки:
//!   1. поклеточные SER/BER до и после, ПООСЕВО (Re = 2G−R−B, Im = R−B);
//!   2. ВЫЖИВАЕМОСТЬ СТРАЙПОВ до и после — по настоящему CRC-16 страйпа, а не
//!      по модели (399 клеток на страйп, одна ошибка убивает страйп целиком);
//!   3. сколько децибел запаса возвращено (Q = полуразнос уровней / σ остатка);
//!   4. декодируется ли цветной набор A22 (страйп 0 несёт FrameHeader);
//!   5. цена в мс на кадр;
//!   6. распространение ошибок у решателя с обратной связью — РАСПРЕДЕЛЕНИЕ
//!      длин пачек, а не средняя SER.
//!
//! Сравниваются ТРИ решателя (см. `psicode_core::isi`):
//!   L — линейный КИХ (обращение свёртки рядом Неймана);
//!   D — обратная связь по решениям (DFE);
//!   J — совместный классификатор в пространстве (своё, соседи), с параметрами,
//!       условными по гипотезе (форма правила LSVM-CMI/QDA-CMI из HiQ).
//!
//! usage:
//!   isi_eq mono   <dump-dir> [x0,y0,w,h] [max-frames] [cell]
//!   isi_eq chroma <dump-dir> [x0,y0,w,h] [truth-file] [seq-max]
//!   isi_eq synth                       — чистый канал, санитарный шлюз
//!
//! Каталог содержит `dump0.meta` и `dump{N}.y` (+ `.u`/`.v` для цвета).

use psicode_core::detect::{self, Detection};
use psicode_core::fountain::{crc32c, FountainEncoder};
use psicode_core::isi::{self, Grid, IsiKernel, JointRule, KernelShape};
use psicode_core::l3::{self, FrameHeader, TransferInfo, FLAG_TRANSFER_INFO};
use psicode_core::symbol::{
    self, IsiConfig, GRID, PAYLOAD_COLS, PAYLOAD_ROWS, RING,
};
use psicode_core::tone;
use psicode_rx::yuv::YuvFrame;
use psicode_rx::{tx_chromatic_profile, tx_default_profile};
use std::fs;
use std::time::Instant;

/// Клеток payload.
const NCELL: usize = PAYLOAD_COLS * PAYLOAD_ROWS;
/// Клеток сетки всего символа.
const NGRID: usize = GRID * GRID;
/// Символов в кадре транспорта (зеркало session.rs).
const SYMBOLS_PER_FRAME: usize = 8;
/// Repair каждые 4 source (зеркало session.rs).
const REPAIR_EVERY: u32 = 4;
/// Отступ от края сетки, с которого клетка берётся откликом регрессии.
const FIT_MARGIN: usize = 3;
/// Ниже этого разделения оси Re созвездие уже схлопнуто — снимок накрыл границу
/// двух кадров передатчика, и лечить его выравнивателем нечем.
const Q_CLEAN_MIN: f64 = 1.8;

// ---------------------------------------------------------------------------
// Загрузка дампов
// ---------------------------------------------------------------------------

/// Один дамп кадра: плоскости Y (+U/V), уже с применённым кропом.
struct Dump {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    w: usize,
    h: usize,
    y_stride: usize,
    uv_stride: usize,
    uv_px: usize,
    colour: bool,
}

impl Dump {
    fn luma_f32(&self) -> Vec<f32> {
        let mut o = vec![0.0f32; self.w * self.h];
        for j in 0..self.h {
            for i in 0..self.w {
                o[j * self.w + i] = self.y[j * self.y_stride + i] as f32 / 255.0;
            }
        }
        o
    }

    fn frame(&self) -> YuvFrame<'_> {
        YuvFrame {
            y: &self.y,
            u: &self.u,
            v: &self.v,
            w: self.w,
            h: self.h,
            y_stride: self.y_stride,
            uv_stride: self.uv_stride,
            uv_pixel_stride: self.uv_px,
        }
    }

    /// Билинейный сэмпл сырого сигнала: цвет -> RGB, моно -> Y во все три канала.
    fn raw(&self, x: f64, y: f64) -> [f32; 3] {
        let xc = x.clamp(0.0, (self.w - 1) as f64);
        let yc = y.clamp(0.0, (self.h - 1) as f64);
        let (x0, y0) = (xc.floor() as usize, yc.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = ((xc - x0 as f64) as f32, (yc - y0 as f64) as f32);
        if !self.colour {
            let at = |i: usize, j: usize| self.y[j * self.y_stride + i] as f32 / 255.0;
            let a = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
            let b = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
            let g = a * (1.0 - fy) + b * fy;
            return [g, g, g];
        }
        let fr = self.frame();
        let (s00, s10, s01, s11) = (
            fr.rgb_at(x0, y0),
            fr.rgb_at(x1, y0),
            fr.rgb_at(x0, y1),
            fr.rgb_at(x1, y1),
        );
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            let a = s00[c] * (1.0 - fx) + s10[c] * fx;
            let b = s01[c] * (1.0 - fx) + s11[c] * fx;
            o[c] = a * (1.0 - fy) + b * fy;
        }
        o
    }
}

/// Загрузка серии `dump{N}` из каталога с общим кропом.
fn load_dumps(dir: &str, crop: Option<[usize; 4]>, max: usize, colour: bool) -> Vec<Dump> {
    let meta = fs::read_to_string(format!("{dir}/dump0.meta")).expect("dump0.meta");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|t| t.parse().expect("meta: числа"))
        .collect();
    let (fw, fh, y_stride, uv_stride, uv_px) = (m[0], m[1], m[2], m[3], m[4]);
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < max {
        let Ok(yb) = fs::read(format!("{dir}/dump{i}.y")) else {
            break;
        };
        let (ub, vb) = if colour {
            (
                fs::read(format!("{dir}/dump{i}.u")).unwrap_or_default(),
                fs::read(format!("{dir}/dump{i}.v")).unwrap_or_default(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let d = match crop {
            None => Dump {
                y: yb,
                u: ub,
                v: vb,
                w: fw,
                h: fh,
                y_stride,
                uv_stride,
                uv_px,
                colour,
            },
            Some([x0, y0, cw, ch]) => {
                let (x0, y0) = (x0 & !1, y0 & !1);
                let (cw, ch) = ((cw.min(fw - x0)) & !1, (ch.min(fh - y0)) & !1);
                let mut ny = vec![0u8; cw * ch];
                for j in 0..ch {
                    let s = (y0 + j) * y_stride + x0;
                    ny[j * cw..(j + 1) * cw].copy_from_slice(&yb[s..s + cw]);
                }
                let cs = if uv_px == 2 { cw } else { cw / 2 };
                let (mut nu, mut nv) = (vec![128u8; cs * (ch / 2)], vec![128u8; cs * (ch / 2)]);
                if colour {
                    for j in 0..ch / 2 {
                        for i2 in 0..cw / 2 {
                            let s = (y0 / 2 + j) * uv_stride + (x0 / 2 + i2) * uv_px;
                            if s < ub.len() {
                                nu[j * cs + i2 * uv_px] = ub[s];
                            }
                            if s < vb.len() {
                                nv[j * cs + i2 * uv_px] = vb[s];
                            }
                        }
                    }
                }
                Dump {
                    y: ny,
                    u: nu,
                    v: nv,
                    w: cw,
                    h: ch,
                    y_stride: cw,
                    uv_stride: cs,
                    uv_px,
                    colour,
                }
            }
        };
        out.push(d);
        i += 1;
    }
    out
}

/// Захват на первом кадре + доводка, затем НЕЗАВИСИМОЕ выравнивание каждого
/// кадра от общего семени (тот же рецепт, что в `cell_noise.rs`).
fn align(dumps: &[Dump]) -> Vec<Option<Detection>> {
    let l0 = dumps[0].luma_f32();
    let (w, h) = (dumps[0].w, dumps[0].h);
    let mut seed = detect::detect_symbol(w, h, &l0)
        .or_else(|_| detect::detect_symbol_acquire(w, h, &l0))
        .expect("детекция на кадре 0 не удалась");
    for _ in 0..10 {
        match detect::track_symbol(w, h, &l0, &seed) {
            Ok(d) if d.score > seed.score + 1e-4 => seed = d,
            _ => break,
        }
    }
    dumps
        .iter()
        .map(|dp| {
            let l = dp.luma_f32();
            let mut d = detect::track_symbol(dp.w, dp.h, &l, &seed).ok();
            for _ in 0..10 {
                let Some(cur) = d.as_ref() else { break };
                match detect::track_symbol(dp.w, dp.h, &l, cur) {
                    Ok(n) if n.score > cur.score + 1e-4 => d = Some(n),
                    _ => break,
                }
            }
            d
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Статистика
// ---------------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Разделение оси в единицах σ: `Q = (μ₊ − μ₋) / (2·σ_pooled)`.
///
/// Именно эта величина решает: страйп из 399 клеток выживает, когда хвост
/// нормали за порогом достаточно тонок, а порог стоит на полуразносе.
fn q_factor(vals: &[f64], cls: &[u8]) -> (f64, f64, f64, f64) {
    let p: Vec<f64> = vals
        .iter()
        .zip(cls)
        .filter(|(_, &c)| c == 1)
        .map(|(v, _)| *v)
        .collect();
    let n: Vec<f64> = vals
        .iter()
        .zip(cls)
        .filter(|(_, &c)| c == 0)
        .map(|(v, _)| *v)
        .collect();
    if p.len() < 2 || n.len() < 2 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let (mp, mn) = (mean(&p), mean(&n));
    let (sp, sn) = (sd(&p), sd(&n));
    let pooled = (((p.len() - 1) as f64 * sp * sp + (n.len() - 1) as f64 * sn * sn)
        / (p.len() + n.len() - 2) as f64)
        .sqrt();
    let q = if pooled > 0.0 {
        (mp - mn) / (2.0 * pooled)
    } else {
        f64::INFINITY
    };
    (q, mp, mn, pooled)
}

/// Моделируемая выживаемость страйпа: `(1−SER)^399`.
fn stripe_model(ser: f64) -> f64 {
    (1.0 - ser).powi(399)
}

/// Реальные CRC страйпов кадра.
fn stripes_ok(cells: &[u8], bpc: u32) -> [bool; 8] {
    l3::parse_frame(cells, bpc).stripes_ok
}

fn fmt_stripes(s: &[bool; 8]) -> String {
    s.iter().map(|&b| if b { '#' } else { '.' }).collect()
}

/// Печать ядра В ГЕОМЕТРИИ СОСЕДЕЙ: элемент на месте (dr, dc) — вклад соседа,
/// стоящего на смещении (dr, dc) от клетки. Ядро свёртки само по себе зеркально
/// (`h_k` умножает `x_{n−k}`), и печатать его как есть — гарантированно
/// перепутать «левый» с «правым», а вся суть замера как раз в асимметрии.
fn print_kernel(tag: &str, k: &IsiKernel) {
    let r = k.radius as i32;
    let c = isi::conditioning(k, 2, 24);
    // анизотропия: рёберные соседи против диагональных (ожидание из геометрии ~13×)
    let edge = k.neighbour(0, -1).abs()
        + k.neighbour(0, 1).abs()
        + k.neighbour(-1, 0).abs()
        + k.neighbour(1, 0).abs();
    let diag = k.neighbour(-1, -1).abs()
        + k.neighbour(-1, 1).abs()
        + k.neighbour(1, -1).abs()
        + k.neighbour(1, 1).abs();
    println!(
        "  {tag}: радиус {}, сила Σ|h≠0| = {:.4}, DC {:.4} (доля своего света {:.3})",
        k.radius,
        k.strength(),
        k.dc_gain(),
        1.0 / k.dc_gain().max(1e-9)
    );
    println!(
        "      обусловленность: |H| ∈ [{:.3}, {:.3}], усиление шума ×{:.4}, остаток ISI {:.4}\
         {}",
        c.h_min,
        c.h_max,
        c.noise_gain,
        c.residual,
        if diag > 1e-9 {
            format!("; анизотропия рёбра/диагонали {:.1}×", edge / diag)
        } else {
            String::new()
        }
    );
    for dr in -r..=r {
        let row: Vec<String> = (-r..=r)
            .map(|dc| format!("{:+.4}", k.neighbour(dr, dc)))
            .collect();
        println!("      {}", row.join(" "));
    }
}

/// Поотсчётная медиана набора ядер — кандидат на КЭШИРУЕМОЕ ядро калибровки.
fn median_kernel(ks: &[IsiKernel]) -> IsiKernel {
    let mut out = IsiKernel::identity(ks[0].radius);
    let r = out.radius as i32;
    let side = out.side();
    for dr in -r..=r {
        for dc in -r..=r {
            let mut v: Vec<f64> = ks.iter().map(|k| k.tap(dr, dc)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m = v[v.len() / 2];
            out.taps[((dr + r) as usize) * side + (dc + r) as usize] = m;
        }
    }
    out.taps[(r as usize) * side + r as usize] = 1.0;
    out
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("mono") => mono(&args),
        Some("chroma") => chroma(&args),
        Some("synth") => synth(),
        Some("sweep") => sweep(),
        Some("axes") => axes(&args),
        _ => {
            eprintln!(
                "usage: isi_eq mono <dir> [crop] [max] [cell] | \
                 isi_eq chroma <dir> [crop] [truth-file] [seq-max] | isi_eq synth"
            );
            std::process::exit(2);
        }
    }
}

fn parse_crop(s: Option<&String>) -> Option<[usize; 4]> {
    s.filter(|t| t.contains(',')).map(|t| {
        let c: Vec<usize> = t.split(',').map(|x| x.parse().expect("кроп")).collect();
        [c[0], c[1], c[2], c[3]]
    })
}

// ===========================================================================
// САНИТАРНЫЙ ШЛЮЗ: чистый канал
// ===========================================================================

/// Чистый канал (без поля, без шума, без блюра): выравниватель обязан быть
/// БЕЗВРЕДЕН, а известная патология двухмасштабного порога — воспроизвестись.
///
/// Порог `demod_symbol_local` оценивает фон по данным с НУЛЕВЫМ средним и
/// потому изобретает поправку там, где корректировать нечего; на чистом канале
/// это даёт десятки ошибок при полном их отсутствии у глобального порога.
fn synth() {
    println!("=== САНИТАРНЫЙ ШЛЮЗ: идеальный канал ===");
    for &(name, chromatic) in &[("моно 1 бит", false), ("QPSK §5.1-CL", true)] {
        let p = if chromatic {
            tx_chromatic_profile()
        } else {
            tx_default_profile()
        };
        let bpc = symbol::bits_per_cell(&p);
        let mask = ((1u32 << bpc) - 1) as u8;
        let mut st = 0x0D15_EA5E_5EED_1234u64;
        let cells: Vec<u8> = (0..NCELL)
            .map(|_| {
                st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = st;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (((z ^ (z >> 31)) >> 24) as u8) & mask
            })
            .collect();
        let fr = symbol::render_symbol(&p, &cells);
        let side = fr.size_px;
        let gam = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
        // приёмник видит ЛИНЕАРИЗОВАННЫЙ сигнал: (d/255)^γ, ровно модель §3.4
        let samp = |x: f64, y: f64| -> [f32; 3] {
            let xi = (x.round().max(0.0) as usize).min(side - 1);
            let yi = (y.round().max(0.0) as usize).min(side - 1);
            let px = fr.rgb[yi * side + xi];
            [
                (px[0] as f64 / 255.0).powf(gam[0]) as f32,
                (px[1] as f64 / 255.0).powf(gam[1]) as f32,
                (px[2] as f64 / 255.0).powf(gam[2]) as f32,
            ]
        };
        let map = |u: f64, v: f64| (u, v);
        let errs = |got: &[u8]| got.iter().zip(&cells).filter(|(a, b)| a != b).count();
        let base = symbol::demod_symbol(&p, &map, &samp);
        let cfg = IsiConfig::default();
        let eq = symbol::demod_symbol_isi(&p, &map, &samp, None, &cfg);
        println!(
            "{name}: глобальный порог {} ош., + ISI {} ош. (ядро сила {:.5})",
            errs(&base),
            errs(&eq.cells),
            eq.kernels[1].strength()
        );
        if !chromatic {
            let loc = symbol::demod_symbol_local(&p, &map, &samp);
            let l_isi = symbol::demod_symbol_local_isi(&p, &map, &samp, &cfg);
            let mut c2 = cfg;
            c2.fine_threshold = false;
            let l_isi_c = symbol::demod_symbol_local_isi(&p, &map, &samp, &c2);
            println!(
                "  локальный двухмасштабный {} ош.; + ISI (тонкое окно вкл.) {} ош.; \
                 + ISI (только грубое окно) {} ош.",
                errs(&loc),
                errs(&l_isi.cells),
                errs(&l_isi_c.cells)
            );
        }
    }
}

// ===========================================================================
// КОНТРОЛИРУЕМАЯ РАЗВЁРТКА: ISI известной силы + шум известной величины
// ===========================================================================

/// Ядро, ИЗМЕРЕННОЕ на цветном наборе A22 (ось Re, медиана по чистым кадрам),
/// в геометрии соседей. Развёртка гоняет ровно его, чтобы отделить вопрос
/// «работает ли выравниватель» от вопросов геометрии, поля и рваных кадров.
fn measured_axis_kernel() -> IsiKernel {
    let mut k = IsiKernel::identity(1);
    k.set_neighbour(0, -1, 0.104); // левый
    k.set_neighbour(0, 1, 0.104); // правый
    k.set_neighbour(-1, 0, 0.056); // верхний
    k.set_neighbour(1, 0, 0.126); // нижний
    for &(a, b) in &[(-1, -1), (-1, 1), (1, -1), (1, 1)] {
        k.set_neighbour(a, b, 0.018);
    }
    k
}

/// Детерминированный гауссов шум (Бокс–Мюллер поверх LCG) — без внешних крейтов.
fn gauss(state: &mut u64) -> f64 {
    let mut u = || {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64).max(1e-12)
    };
    let (a, b) = (u(), u());
    (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
}

/// Развёртка по шуму: настоящий кадр L3 (значит, страйповые CRC осмысленны),
/// известное ядро, гауссов шум. Считает SER, ВЫЖИВАЕМОСТЬ СТРАЙПОВ и
/// РАСПРЕДЕЛЕНИЕ ДЛИН ПАЧЕК для четырёх решателей.
///
/// Ради чего: на реальных снимках SER после выравнивания падает почти в ноль, и
/// распространение ошибок у DFE просто не на чем измерить. Здесь SER задаётся
/// ручкой, и видно, ГДЕ обратная связь начинает делать хуже.
fn sweep() {
    let p = tx_chromatic_profile();
    let bpc = symbol::bits_per_cell(&p);
    let k = measured_axis_kernel();
    println!("=== РАЗВЁРТКА ПО ШУМУ: ISI силы {:.3}, кадр L3 ===", k.strength());
    print_kernel("ядро (геометрия соседей)", &k);
    // настоящий кадр L3: страйп = 399 клеток + CRC-16
    let mut hd = FrameHeader::new(0x0138_c764, 17, SYMBOLS_PER_FRAME as u8);
    hd.flags = 0;
    let bytes: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    let truth = l3::build_frame(&hd, None, &bytes, bpc);
    let idx_pay: Vec<usize> = (0..NCELL)
        .map(|i| (RING + 1 + i / PAYLOAD_COLS) * GRID + RING + i % PAYLOAD_COLS)
        .collect();
    // истинные оси на всей сетке символа; вне payload — нейтраль (кольцо,
    // референсная строка и строка счётчика на осях созвездия дают 0)
    let (mut tx_, mut ty) = (vec![0.0f64; NGRID], vec![0.0f64; NGRID]);
    for (i, &gi) in idx_pay.iter().enumerate() {
        tx_[gi] = if (truth[i] >> 1) & 1 == 1 { 1.0 } else { -1.0 };
        ty[gi] = if truth[i] & 1 == 1 { 1.0 } else { -1.0 };
    }
    let conv = |x: &[f64]| -> Vec<f64> {
        let mut o = vec![0.0f64; NGRID];
        let g = Grid { v: x, rows: GRID, cols: GRID };
        let r = k.radius as i32;
        for rr in 0..GRID as i32 {
            for cc in 0..GRID as i32 {
                let mut a = 0.0;
                for dr in -r..=r {
                    for dc in -r..=r {
                        a += k.tap(dr, dc) * g.at(rr - dr, cc - dc);
                    }
                }
                o[rr as usize * GRID + cc as usize] = a;
            }
        }
        o
    };
    let (bx, by) = (conv(&tx_), conv(&ty));
    println!(
        "\n{:<8} {:>9} {:>9} {:>9} {:>9}   живых страйпов (из 8)",
        "σ", "база", "L линейн", "D обр.св", "J совм."
    );
    let mut rows: Vec<(f64, [usize; 4], [usize; 4], [Vec<usize>; 4])> = Vec::new();
    for &sigma in &[0.05f64, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] {
        let mut st = 0xC0FF_EE00_1234_5678u64;
        let (mut mx, mut my) = (bx.clone(), by.clone());
        for i in 0..NGRID {
            mx[i] += sigma * gauss(&mut st);
            my[i] += sigma * gauss(&mut st);
        }
        let gx = Grid { v: &mx, rows: GRID, cols: GRID };
        let gy = Grid { v: &my, rows: GRID, cols: GRID };
        // ядро оценивается ПО ПРОБНЫМ РЕШЕНИЯМ (как в бою), не по истине
        let base: Vec<u8> = idx_pay
            .iter()
            .map(|&gi| (((mx[gi] > 0.0) as u8) << 1) | (my[gi] > 0.0) as u8)
            .collect();
        let (mut px, mut py) = (vec![0.0f64; NGRID], vec![0.0f64; NGRID]);
        for (i, &gi) in idx_pay.iter().enumerate() {
            px[gi] = if (base[i] >> 1) & 1 == 1 { 1.0 } else { -1.0 };
            py[gi] = if base[i] & 1 == 1 { 1.0 } else { -1.0 };
        }
        let kx = isi::estimate_kernel(
            &gx,
            &Grid { v: &px, rows: GRID, cols: GRID },
            1,
            FIT_MARGIN,
            KernelShape::Cross,
        )
        .unwrap_or(IsiKernel::identity(1));
        let ky = isi::estimate_kernel(
            &gy,
            &Grid { v: &py, rows: GRID, cols: GRID },
            1,
            FIT_MARGIN,
            KernelShape::Cross,
        )
        .unwrap_or(IsiKernel::identity(1));
        let (mut lx, mut ly) = (mx.clone(), my.clone());
        isi::equalise(&mut lx, GRID, GRID, &kx, 2);
        isi::equalise(&mut ly, GRID, GRID, &ky, 2);
        let hard = |v: f64| if v > 0.0 { 1.0 } else { -1.0 };
        let (mut dx, mut dy) = (mx.clone(), my.clone());
        isi::equalise_dfe(&mut dx, GRID, GRID, &kx, &hard);
        isi::equalise_dfe(&mut dy, GRID, GRID, &ky, &hard);
        let (mut lab_x, mut lab_y) = (vec![usize::MAX; NGRID], vec![usize::MAX; NGRID]);
        for (i, &gi) in idx_pay.iter().enumerate() {
            lab_x[gi] = ((base[i] >> 1) & 1) as usize;
            lab_y[gi] = (base[i] & 1) as usize;
        }
        let jx = JointRule::train(&gx, &lab_x, &[-1.0, 1.0], 1, FIT_MARGIN);
        let jy = JointRule::train(&gy, &lab_y, &[-1.0, 1.0], 1, FIT_MARGIN);
        let asm = |xp: &[f64], yp: &[f64]| -> Vec<u8> {
            idx_pay
                .iter()
                .map(|&gi| (((xp[gi] > 0.0) as u8) << 1) | (yp[gi] > 0.0) as u8)
                .collect()
        };
        let cj: Vec<u8> = match (&jx, &jy) {
            (Some(a), Some(b)) => {
                let (da, db) = (a.decide_all(&gx), b.decide_all(&gy));
                idx_pay
                    .iter()
                    .map(|&gi| ((da[gi] as u8) << 1) | db[gi] as u8)
                    .collect()
            }
            _ => base.clone(),
        };
        let got = [base.clone(), asm(&lx, &ly), asm(&dx, &dy), cj];
        let mut errs = [0usize; 4];
        let mut alive = [0usize; 4];
        let mut bursts: [Vec<usize>; 4] = Default::default();
        for (j, g) in got.iter().enumerate() {
            errs[j] = (0..NCELL).filter(|&i| g[i] != truth[i]).count();
            alive[j] = stripes_ok(g, bpc).iter().filter(|&&b| b).count();
            let mut run = 0usize;
            for i in 0..NCELL {
                if g[i] != truth[i] {
                    run += 1;
                } else if run > 0 {
                    bursts[j].push(run);
                    run = 0;
                }
            }
            if run > 0 {
                bursts[j].push(run);
            }
        }
        println!(
            "{sigma:<8.2} {:>9.5} {:>9.5} {:>9.5} {:>9.5}   {} / {} / {} / {}",
            errs[0] as f64 / NCELL as f64,
            errs[1] as f64 / NCELL as f64,
            errs[2] as f64 / NCELL as f64,
            errs[3] as f64 / NCELL as f64,
            alive[0],
            alive[1],
            alive[2],
            alive[3]
        );
        rows.push((sigma, errs, alive, bursts));
    }
    println!("\n--- длины пачек ошибок (кучность решает судьбу страйпа) ---");
    println!("{:<8} {:<10} {:>7} {:>7} {:>7} {:>7} {:>7}", "σ", "решатель", "ошибок", "пачек", "1", "2", "3+");
    for (sigma, errs, _, bursts) in &rows {
        for (j, name) in ["база", "L линейн", "D обр.св", "J совм."].iter().enumerate() {
            let b = &bursts[j];
            if b.is_empty() {
                continue;
            }
            println!(
                "{sigma:<8.2} {name:<10} {:>7} {:>7} {:>7} {:>7} {:>7}",
                errs[j],
                b.len(),
                b.iter().filter(|&&x| x == 1).count(),
                b.iter().filter(|&&x| x == 2).count(),
                b.iter().filter(|&&x| x >= 3).count()
            );
        }
    }
}

// ===========================================================================
// БАЛАНС ОСЕЙ СОЗВЕЗДИЯ (§5.1-CL): чем на самом деле задан перекос
// ===========================================================================

/// Хвост нормали `P(n > q·σ)` = `½·erfc(q/√2)`.
///
/// `erfc` через рациональное приближение Абрамовица–Стигана 7.1.26
/// (|ε| < 1.5·10⁻⁷) — внешних зависимостей в проекте нет, а нужна одна функция.
fn q_tail(q: f64) -> f64 {
    if q < 0.0 {
        return 1.0 - q_tail(-q);
    }
    let x = q / core::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    0.5 * y * (-x * x).exp()
}

/// Выживаемость страйпа при поклеточной вероятности ошибки `p`.
fn survival(p: f64) -> f64 {
    (1.0 - p).powi(399)
}

/// Симметричная 2×2 ковариация из накопителя.
#[derive(Default, Clone, Copy)]
struct Cov2 {
    n: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Cov2 {
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.sxx += x * x;
        self.syy += y * y;
        self.sxy += x * y;
    }
    fn sx(&self) -> f64 {
        (self.sxx / self.n.max(1.0)).sqrt()
    }
    fn sy(&self) -> f64 {
        (self.syy / self.n.max(1.0)).sqrt()
    }
    fn corr(&self) -> f64 {
        let d = self.sx() * self.sy();
        if d > 0.0 {
            (self.sxy / self.n.max(1.0)) / d
        } else {
            0.0
        }
    }
}

/// Локальное среднее по квадратному окну радиуса `rad` на сетке `n×n`
/// (интегральное изображение). Зеркало `symbol::box_mean`, которая приватна.
fn box_mean_grid(src: &[f64], n: usize, rad: i32) -> Vec<f64> {
    let mut sat = vec![0.0f64; (n + 1) * (n + 1)];
    for r in 0..n {
        let mut row = 0.0;
        for c in 0..n {
            row += src[r * n + c];
            sat[(r + 1) * (n + 1) + (c + 1)] = sat[r * (n + 1) + (c + 1)] + row;
        }
    }
    let area = |r0: usize, c0: usize, r1: usize, c1: usize| -> f64 {
        sat[r1 * (n + 1) + c1] - sat[r0 * (n + 1) + c1] - sat[r1 * (n + 1) + c0]
            + sat[r0 * (n + 1) + c0]
    };
    let mut out = vec![0.0f64; n * n];
    for r in 0..n {
        let r0 = (r as i32 - rad).max(0) as usize;
        let r1 = (r as i32 + rad + 1).min(n as i32) as usize;
        for c in 0..n {
            let c0 = (c as i32 - rad).max(0) as usize;
            let c1 = (c as i32 + rad + 1).min(n as i32) as usize;
            out[r * n + c] = area(r0, c0, r1, c1) / (((r1 - r0) * (c1 - c0)) as f64);
        }
    }
    out
}

/// [ИЗМЕРЕНИЕ] Откуда берётся перекос осей `2G−R−B` и `R−B` и что даст
/// перераспределение амплитуды между ними.
///
/// # Что проверяется
///
/// Отображение §5.1-CL выбирает `c = √3·b` ровно ради РАВНОГО отношения
/// сигнал/шум на осях — при НЕЗАВИСИМОМ и равном шуме в R, G, B:
///
/// ```text
/// Var(2G−R−B) = 4σ² + σ² + σ² = 6σ²      swing 6A
/// Var(R−B)    =       σ² + σ² = 2σ²      swing 2C
/// ```
///
/// и при `C/A = √3` оба Q совпадают. Значит ЛЮБОЙ измеренный перекос — это
/// нарушение предположения о равном независимом шуме. Кандидат из физики:
/// байеровская мозаика даёт зелёному вдвое больше фотосайтов, `σ_G ≈ σ/√2`,
/// откуда оптимум `c/b = 2.12` вместо 1.73 — то есть **сдвиг амплитуды НА ось
/// Im**. Здесь это проверяется замером, а не постулируется.
///
/// # Как считается оптимум
///
/// В драйв-единицах `G = u + 2A·x`, `R = u − A·x + C·y`, `B = u − A·x − C·y`,
/// оценки `x̂ = (2G−R−B)/6A`, `ŷ = (R−B)/2C`, порог на полуразносе 1:
///
/// ```text
/// Q_Re = 6A/σ_Re      Q_Im = 2C/σ_Im
/// ```
///
/// Гамут: `|R−u|,|B−u| ≤ amp` даёт `A + C ≤ amp`, `|G−u| ≤ amp` даёт
/// `2A ≤ amp`. Максимум `min(Q_Re, Q_Im)` при связывающем первом ограничении:
///
/// ```text
/// c/b = C/A = 3·σ_Im/σ_Re        A = amp/(1 + c/b)        s = 2/(1 + c/b)
/// ```
///
/// (при `σ_Im/σ_Re = 1/√3` это ровно нынешние `√3` и `2/(1+√3)`). Ограничение
/// по зелёному связывает при `c/b < 1` — тогда оптимум упирается в него.
///
/// usage: isi_eq axes <dump-dir> [crop] [truth-file] [seq-max]
fn axes(args: &[String]) {
    let dir = args.get(2).expect("каталог дампов");
    let crop = parse_crop(args.get(3));
    let truth_file = args.get(4).filter(|s| !s.contains(',')).cloned();
    let seq_max: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let dumps = load_dumps(dir, crop, usize::MAX, true);
    assert!(!dumps.is_empty(), "нет дампов в {dir}");
    let p = tx_chromatic_profile();
    let bpc = symbol::bits_per_cell(&p);
    let dets = align(&dumps);
    let cl = symbol::const_luma_map(&p);
    let cls = symbol::CL_LATTICE_SCALE;
    let frames: Vec<Vec<u8>> = match &truth_file {
        Some(f) => build_truth_frames(f, bpc, seq_max),
        None => Vec::new(),
    };
    let cfg = IsiConfig::default();
    println!(
        "=== БАЛАНС ОСЕЙ §5.1-CL: {} кадров из {dir} ===",
        dumps.len()
    );
    println!(
        "отображение: u {:.1}, b {:.4}, c {:.4}, c/b {:.4}, S {:.1}, amp {:.1}, масштаб решётки {:.4}",
        cl.u,
        cl.b,
        cl.c,
        cl.c / cl.b,
        cl.s,
        cl.amp,
        cls
    );
    // драйв-свинги нынешнего отображения
    let a_drive = cl.u * cl.b * cls;
    let c_drive = cl.u * cl.c * cls;
    println!(
        "драйв-свинги: A = u·b·s = {a_drive:.2}, C = u·c·s = {c_drive:.2}, A+C = {:.2} (amp {:.2})",
        a_drive + c_drive,
        cl.amp
    );

    let idx_pay: Vec<usize> = (0..NCELL)
        .map(|i| (RING + 1 + i / PAYLOAD_COLS) * GRID + RING + i % PAYLOAD_COLS)
        .collect();

    // накопители, объединённые по чистым кадрам
    let (mut ax_raw, mut ax_eq) = (Cov2::default(), Cov2::default());
    // поканальные остатки драйва (после снятия классового среднего и поля)
    let mut ch_raw = [[0.0f64; 3]; 3];
    let mut ch_eq = [[0.0f64; 3]; 3];
    let (mut n_raw, mut n_eq) = (0.0f64, 0.0f64);
    let mut used = 0usize;

    for (fi, dp) in dumps.iter().enumerate() {
        let Some(d) = dets[fi].as_ref() else { continue };
        let map = detect::frame_map(&p, d);
        let raw = |x: f64, y: f64| dp.raw(x, y);
        let g = tone::estimate_channel_gammas(&p, &map, &raw);
        let lin = |x: f64, y: f64| -> [f32; 3] {
            let s = raw(x, y);
            [
                (s[0] as f64).max(0.0).powf(g[0]) as f32,
                (s[1] as f64).max(0.0).powf(g[1]) as f32,
                (s[2] as f64).max(0.0).powf(g[2]) as f32,
            ]
        };
        let pg = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
        let base = symbol::demod_symbol(&p, &map, &lin);
        let (truth, seq) = lock_truth(&frames, &base, bpc);
        let Some(tr) = truth else { continue };
        let eq = symbol::demod_symbol_isi(&p, &map, &lin, None, &cfg);

        let s_grid = symbol::sample_symbol_grid(&p, &map, &lin);
        let m = symbol::estimate_matrix_reference_row(&p, &pg, &map, &lin)
            .expect("матрица развязки вырождена");
        let mut t: [Vec<f64>; 3] = [vec![0.0; NGRID], vec![0.0; NGRID], vec![0.0; NGRID]];
        for (i, s) in s_grid.iter().enumerate() {
            let l = m.apply(*s);
            for c in 0..3 {
                t[c][i] = l[c];
            }
        }
        let mut teq = t.clone();
        for c in 0..3 {
            isi::equalise(&mut teq[c], GRID, GRID, &eq.kernels[c], cfg.iters);
        }
        let drives = |tp: &[Vec<f64>; 3]| -> Vec<[f64; 3]> {
            (0..NGRID)
                .map(|i| {
                    let mut dr = [0.0f64; 3];
                    for c in 0..3 {
                        dr[c] = (255.0 * tp[c][i].max(0.0).powf(1.0 / pg[c])).clamp(0.0, 255.0);
                    }
                    dr
                })
                .collect()
        };
        let d_raw = drives(&t);
        let d_eq = drives(&teq);

        // Q ДО выравнивания — им же отбираем кадры с несхлопнутым созвездием
        let xs: Vec<f64> = idx_pay
            .iter()
            .map(|&i| cl.z_from_drive(d_raw[i]).0 / cls)
            .collect();
        let ys: Vec<f64> = idx_pay
            .iter()
            .map(|&i| cl.z_from_drive(d_raw[i]).1 / cls)
            .collect();
        let cre: Vec<u8> = (0..NCELL).map(|i| (tr[i] >> 1) & 1).collect();
        let cim: Vec<u8> = (0..NCELL).map(|i| tr[i] & 1).collect();
        let (q0, _, _, _) = q_factor(&xs, &cre);
        if q0 < Q_CLEAN_MIN {
            println!("кадр {fi} (seq {seq}): Q_Re {q0:.2} — созвездие схлопнуто, пропуск");
            continue;
        }
        used += 1;

        // --- ковариация В ПЛОСКОСТИ ОСЕЙ, остаток вокруг КЛАССОВЫХ средних ---
        // Классовые средние вычитаются, потому что усиление осей канала не
        // единичное (лестница масштабов: 0.53 по Re, 0.76 по Im) — без этого
        // «остаток» мерил бы усиление, а не шум.
        let axis_cov = |dd: &Vec<[f64; 3]>| -> Cov2 {
            let mut acc = Cov2::default();
            let xv: Vec<f64> = idx_pay
                .iter()
                .map(|&i| cl.z_from_drive(dd[i]).0 / cls)
                .collect();
            let yv: Vec<f64> = idx_pay
                .iter()
                .map(|&i| cl.z_from_drive(dd[i]).1 / cls)
                .collect();
            let (mut mx, mut my, mut cnt) = ([0.0f64; 4], [0.0f64; 4], [0.0f64; 4]);
            for i in 0..NCELL {
                let k = (cre[i] * 2 + cim[i]) as usize;
                mx[k] += xv[i];
                my[k] += yv[i];
                cnt[k] += 1.0;
            }
            for k in 0..4 {
                mx[k] /= cnt[k].max(1.0);
                my[k] /= cnt[k].max(1.0);
            }
            for i in 0..NCELL {
                let k = (cre[i] * 2 + cim[i]) as usize;
                acc.add(xv[i] - mx[k], yv[i] - my[k]);
            }
            acc
        };
        for (dst, src) in [(&mut ax_raw, axis_cov(&d_raw)), (&mut ax_eq, axis_cov(&d_eq))] {
            dst.n += src.n;
            dst.sxx += src.sxx;
            dst.syy += src.syy;
            dst.sxy += src.sxy;
        }

        // --- поканальный остаток драйва: класс + гладкое поле сняты ---
        for (dd, acc, nn) in [
            (&d_raw, &mut ch_raw, &mut n_raw),
            (&d_eq, &mut ch_eq, &mut n_eq),
        ] {
            // остаток относительно классового среднего, на всей сетке символа
            let mut res: [Vec<f64>; 3] = [
                vec![0.0; NGRID],
                vec![0.0; NGRID],
                vec![0.0; NGRID],
            ];
            let mut mean = [[0.0f64; 3]; 4];
            let mut cnt = [0.0f64; 4];
            for (i, &gi) in idx_pay.iter().enumerate() {
                let k = (cre[i] * 2 + cim[i]) as usize;
                for c in 0..3 {
                    mean[k][c] += dd[gi][c];
                }
                cnt[k] += 1.0;
            }
            for k in 0..4 {
                for c in 0..3 {
                    mean[k][c] /= cnt[k].max(1.0);
                }
            }
            for (i, &gi) in idx_pay.iter().enumerate() {
                let k = (cre[i] * 2 + cim[i]) as usize;
                for c in 0..3 {
                    res[c][gi] = dd[gi][c] - mean[k][c];
                }
            }
            // снятие ГЛАДКОГО поля: локальное среднее остатка радиусом 12 клеток
            for c in 0..3 {
                let lm = box_mean_grid(&res[c], GRID, 12);
                for i in 0..NGRID {
                    res[c][i] -= lm[i];
                }
            }
            for &gi in &idx_pay {
                for a in 0..3 {
                    for b in 0..3 {
                        acc[a][b] += res[a][gi] * res[b][gi];
                    }
                }
                *nn += 1.0;
            }
        }
        let (qre, _, _, _) = q_factor(&xs, &cre);
        let (qim, _, _, _) = q_factor(&ys, &cim);
        println!("кадр {fi} (seq {seq}): Q_Re {qre:.2}σ, Q_Im {qim:.2}σ");
    }
    assert!(used > 0, "не набралось ни одного пригодного кадра");
    println!("\nиспользовано кадров: {used}");

    for (tag, ax, ch, nn) in [
        ("ДО выравнивания", &ax_raw, &ch_raw, n_raw),
        ("ПОСЛЕ выравнивания", &ax_eq, &ch_eq, n_eq),
    ] {
        println!("\n--- {tag} ---");
        println!(
            "ковариация в плоскости осей: σ_x {:.4}, σ_y {:.4}, корреляция {:+.3}",
            ax.sx(),
            ax.sy(),
            ax.corr()
        );
        let cov = |a: usize, b: usize| ch[a][b] / nn.max(1.0);
        let sd_c = [cov(0, 0).sqrt(), cov(1, 1).sqrt(), cov(2, 2).sqrt()];
        println!(
            "поканальный остаток драйва (коды 0..255): σ_R {:.2}, σ_G {:.2}, σ_B {:.2}",
            sd_c[0], sd_c[1], sd_c[2]
        );
        println!(
            "  корреляции: RG {:+.3}, GB {:+.3}, RB {:+.3}",
            cov(0, 1) / (sd_c[0] * sd_c[1]).max(1e-12),
            cov(1, 2) / (sd_c[1] * sd_c[2]).max(1e-12),
            cov(0, 2) / (sd_c[0] * sd_c[2]).max(1e-12)
        );
        // σ осей ИЗ поканальной ковариации: 2G−R−B и R−B
        let w_re = [-1.0f64, 2.0, -1.0];
        let w_im = [1.0f64, 0.0, -1.0];
        let quad = |w: &[f64; 3]| -> f64 {
            let mut s = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    s += w[a] * w[b] * cov(a, b);
                }
            }
            s.max(0.0).sqrt()
        };
        let (s_re, s_im) = (quad(&w_re), quad(&w_im));
        println!(
            "  -> σ(2G−R−B) {s_re:.2}, σ(R−B) {s_im:.2}, отношение {:.3} \
             (iid даёт √3 = 1.732, байер σ_G=σ/√2 даёт 1.414)",
            s_re / s_im.max(1e-12)
        );
        // сверка: Q из поканальной ковариации против прямого замера по осям
        let (q_re_pred, q_im_pred) = (6.0 * a_drive / s_re, 2.0 * c_drive / s_im);
        println!(
            "  предсказанные Q: Re {q_re_pred:.2}σ, Im {q_im_pred:.2}σ; \
             прямой замер по осям: Re {:.2}σ, Im {:.2}σ",
            1.0 / ax.sx(),
            1.0 / ax.sy()
        );

        // --- КОНСТРУКТИВНЫЙ ОПТИМУМ ---
        // прямой замер осей надёжнее (он учитывает всё, что портит решение),
        // поэтому σ_Re и σ_Im берём из него: σ_x = σ_Re/(6A), σ_y = σ_Im/(2C).
        let s_re_d = ax.sx() * 6.0 * a_drive;
        let s_im_d = ax.sy() * 2.0 * c_drive;
        let ratio = s_im_d / s_re_d;
        let rho_opt_raw = 3.0 * ratio;
        let rho_opt = rho_opt_raw.max(1.0); // ниже 1 связывает ограничение по G
        println!(
            "\n  ОПТИМУМ: σ_Im/σ_Re {ratio:.4} -> c/b = 3·σ_Im/σ_Re = {rho_opt_raw:.3}\
             {}",
            if rho_opt_raw < 1.0 {
                " (зажат до 1.000 ограничением по зелёному)"
            } else {
                ""
            }
        );
        println!(
            "  масштаб решётки s = 2/(1+c/b): {:.4} (сейчас {:.4})",
            2.0 / (1.0 + rho_opt),
            cls
        );
        let sqrt3 = 3.0f64.sqrt();
        let q_at = |rho: f64| -> (f64, f64) {
            let k = (1.0 + sqrt3) / (1.0 + rho);
            (1.0 / ax.sx() * k, 1.0 / ax.sy() * k * rho / sqrt3)
        };
        // Минимакс по Q — НЕ тот критерий, который максимизирует выживаемость.
        // Клетка гибнет, если ошиблась ЛЮБАЯ ось, поэтому минимизировать надо
        // СУММУ хвостов, а она выпукла: при заметном перекосе выгоднее оставить
        // лучшей оси часть форы, чем выравнивать Q. Сканируем c/b напрямую.
        let p_cell = |rho: f64| -> (f64, f64, f64) {
            let (qr, qi) = q_at(rho);
            let (pr, pi) = (q_tail(qr), q_tail(qi));
            (qr, qi, pr + pi - pr * pi)
        };
        let mut best = (f64::INFINITY, 1.0f64);
        let mut rho = 0.20f64;
        while rho <= 6.0 {
            if 2.0 / (1.0 + rho) * (1.0 + sqrt3) / 2.0 >= 1.0 || rho >= 1.0 {
                // ограничение по зелёному: 2A ≤ amp ⇔ c/b ≥ 1
                if rho >= 1.0 {
                    let (_, _, pc) = p_cell(rho);
                    if pc < best.0 {
                        best = (pc, rho);
                    }
                }
            }
            rho += 0.002;
        }
        println!(
            "\n  {:<32} {:>8} {:>8} {:>11} {:>13}",
            "конфигурация", "Q_Re", "Q_Im", "p(клетка)", "выживаемость"
        );
        for (name, r) in [
            ("сейчас, c/b = √3 = 1.732", sqrt3),
            ("байеровская гипотеза 2.121", 2.1213),
            ("минимакс Q (равные оси)", rho_opt),
            ("МИНИМУМ p(клетка)", best.1),
        ] {
            let (qr, qi, pc) = p_cell(r);
            println!(
                "  {:<32} {qr:>8.2} {qi:>8.2} {pc:>11.5} {:>12.1}%",
                format!("{name} [c/b {r:.3}]"),
                100.0 * survival(pc)
            );
        }
        let (_, _, pc_now) = p_cell(sqrt3);
        println!(
            "  СВЕРКА С РЕАЛЬНОСТЬЮ: гауссова модель даёт p {pc_now:.5} на клетку; \
             прямой замер CRC на этих же кадрах — см. режим `chroma`."
        );
    }
}

// ===========================================================================
// МОНО
// ===========================================================================

fn mono(args: &[String]) {
    let dir = args.get(2).expect("каталог дампов");
    let crop = parse_crop(args.get(3));
    let max: usize = args
        .iter()
        .skip(3)
        .filter(|s| !s.contains(','))
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let dumps = load_dumps(dir, crop, max, false);
    assert!(!dumps.is_empty(), "нет дампов в {dir}");
    let mut p = tx_default_profile();
    if let Some(c) = args.get(5).and_then(|s| s.parse::<u8>().ok()) {
        p.cell_size_px = c;
    }
    if args.iter().any(|a| a == "--v1") {
        p.border = psicode_core::profile::BorderMode::ExtrudedStrips;
    }
    // Нагрузка `psicode-tx single` — не кадр L3, её страйповые CRC заведомо не
    // сходятся; тогда колонка страйпов бессмысленна и не печатается.
    let l3_payload = args.iter().any(|a| a == "--l3");
    let bpc = symbol::bits_per_cell(&p);
    println!(
        "=== ISI / МОНО: {} кадров {}x{} из {dir}; профиль {} бит/клетку, рамка {:?} ===",
        dumps.len(),
        dumps[0].w,
        dumps[0].h,
        bpc,
        p.border
    );
    let dets = align(&dumps);
    let ok: Vec<usize> = (0..dumps.len()).filter(|&i| dets[i].is_some()).collect();
    println!("детекция удалась на {} из {} кадров", ok.len(), dumps.len());

    // эталон: детерминированная нагрузка `psicode-tx single`
    let truth = splitmix_payload(bpc);

    let cfg = IsiConfig::default();
    let mut cfg_coarse = cfg;
    cfg_coarse.fine_threshold = false;

    let mut kernels: Vec<IsiKernel> = Vec::new();
    let mut rows: Vec<(usize, usize, usize, usize, [bool; 8], [bool; 8], [bool; 8])> = Vec::new();
    let (mut t_base, mut t_eq) = (0.0f64, 0.0f64);
    let mut truth_ok = true;
    let (mut e_glob, mut e_glob_isi): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    for &f in &ok {
        let d = dets[f].as_ref().unwrap();
        let map = detect::frame_map(&p, d);
        let dp = &dumps[f];
        let samp = |x: f64, y: f64| dp.raw(x, y);

        let t0 = Instant::now();
        let base = symbol::demod_symbol_local(&p, &map, &samp);
        t_base += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let eq = symbol::demod_symbol_local_isi(&p, &map, &samp, &cfg);
        t_eq += t1.elapsed().as_secs_f64() * 1e3;
        let eqc = symbol::demod_symbol_local_isi(&p, &map, &samp, &cfg_coarse);

        let e = |g: &[u8]| g.iter().zip(&truth).filter(|(a, b)| a != b).count();
        // ЗАМЕНЯЕТ ЛИ выравниватель локальный порог? Глобальный порог §3.4 плюс
        // выравниватель против локального двухмасштабного — прямая проверка.
        // Поле освещённости и межклеточная помеха — РАЗНЫЕ вещи (первое гладкое,
        // вторая на масштабе клетки), выравниватель поле пропускает насквозь.
        e_glob.push(e(&symbol::demod_symbol(&p, &map, &samp)));
        e_glob_isi.push(e(&symbol::demod_symbol_isi(&p, &map, &samp, None, &cfg).cells));
        let (eb, ee, ec) = (e(&base), e(&eq.cells), e(&eqc.cells));
        if eb > NCELL / 4 {
            truth_ok = false;
        }
        rows.push((
            f,
            eb,
            ee,
            ec,
            stripes_ok(&base, bpc),
            stripes_ok(&eq.cells, bpc),
            stripes_ok(&eqc.cells, bpc),
        ));
        if eq.applied {
            kernels.push(eq.kernels[1]);
        }
    }
    if !truth_ok {
        println!(
            "! эталон splitmix не подходит (это не дамп `tx single`) — SER считается ТОЛЬКО по CRC"
        );
    }

    println!("\n--- ядро ISI (яркость), по кадрам ---");
    if !kernels.is_empty() {
        let k0 = median_kernel(&kernels);
        print_kernel("МЕДИАННОЕ по серии", &k0);
        let mut worst = 0.0f64;
        for k in &kernels {
            worst = worst.max(k0.max_tap_diff(k));
        }
        let stre: Vec<String> = kernels.iter().map(|k| format!("{:.3}", k.strength())).collect();
        println!("  сила ядра по кадрам: {}", stre.join(" "));
        let k0 = &k0;
        // разброс каждого отсчёта по серии
        let r = k0.radius as i32;
        println!("  разброс вкладов соседей по {} кадрам (σ):", kernels.len());
        for dr in -r..=r {
            let row: Vec<String> = (-r..=r)
                .map(|dc| {
                    let v: Vec<f64> = kernels.iter().map(|k| k.neighbour(dr, dc)).collect();
                    format!("{:.4}", sd(&v))
                })
                .collect();
            println!("      {}", row.join(" "));
        }
        let mut devs: Vec<f64> = kernels.iter().map(|k| k0.max_tap_diff(k)).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_dev = devs[devs.len() / 2];
        println!(
            "  расхождение отсчёта с медианой: медиана {med_dev:.4}, максимум {worst:.4} \
             (ядро {})",
            if med_dev < 0.01 {
                "ФИКСИРОВАННОЕ -> кэшируемо в калибровке"
            } else {
                "гуляет от кадра к кадру"
            }
        );
    }

    println!("\n--- SER и выживаемость страйпов, по кадрам ---");
    println!(
        "кадр   баз.ош  +ISI   +ISI(груб)   {}",
        if l3_payload {
            "страйпы: база / +ISI / +ISI(груб)"
        } else {
            "(нагрузка tx single — страйповых CRC нет)"
        }
    );
    for (f, eb, ee, ec, sb, se, sc) in &rows {
        if l3_payload {
            println!(
                "{f:>4}  {eb:>6} {ee:>6} {ec:>10}   {} / {} / {}",
                fmt_stripes(sb),
                fmt_stripes(se),
                fmt_stripes(sc)
            );
        } else {
            println!("{f:>4}  {eb:>6} {ee:>6} {ec:>10}");
        }
    }
    let n = rows.len().max(1) as f64;
    let sum = |sel: &dyn Fn(&(usize, usize, usize, usize, [bool; 8], [bool; 8], [bool; 8])) -> usize| {
        rows.iter().map(sel).sum::<usize>() as f64
    };
    let (sb, se, sc) = (
        sum(&|r| r.1) / n / NCELL as f64,
        sum(&|r| r.2) / n / NCELL as f64,
        sum(&|r| r.3) / n / NCELL as f64,
    );
    let alive = |sel: &dyn Fn(&(usize, usize, usize, usize, [bool; 8], [bool; 8], [bool; 8])) -> [bool; 8]| {
        rows.iter()
            .map(|r| sel(r).iter().filter(|&&b| b).count())
            .sum::<usize>() as f64
            / (n * 8.0)
    };
    println!(
        "\nИТОГО SER: база {sb:.6} -> +ISI {se:.6} -> +ISI(только грубое окно) {sc:.6}"
    );
    println!(
        "заменяет ли выравниватель локальный порог? ГЛОБАЛЬНЫЙ порог SER {:.6}, \
         глобальный + ISI {:.6} — против локального {sb:.6}",
        e_glob.iter().sum::<usize>() as f64 / n / NCELL as f64,
        e_glob_isi.iter().sum::<usize>() as f64 / n / NCELL as f64
    );
    if l3_payload {
        println!(
            "выживаемость страйпов (реальный CRC): база {:.3} -> +ISI {:.3} -> +ISI(груб) {:.3}",
            alive(&|r| r.4),
            alive(&|r| r.5),
            alive(&|r| r.6)
        );
    }
    let _ = &alive;
    println!(
        "модель (1−SER)^399:                   база {:.3} -> +ISI {:.3} -> +ISI(груб) {:.3}",
        stripe_model(sb),
        stripe_model(se),
        stripe_model(sc)
    );
    println!(
        "\nцена: демодуляция {:.2} мс/кадр, с выравнивателем {:.2} мс/кадр (+{:.2} мс)",
        t_base / n,
        t_eq / n,
        (t_eq - t_base) / n
    );
}

/// Детерминированная нагрузка `psicode-tx single` (splitmix64, тот же сид).
fn splitmix_payload(bpc: u32) -> Vec<u8> {
    let mask: u16 = if bpc >= 16 { u16::MAX } else { (1u16 << bpc) - 1 };
    let mut st = 0x0D15_EA5E_5EED_1234u64;
    (0..NCELL)
        .map(|_| {
            st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = st;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (((z ^ (z >> 31)) >> 24) as u16 & mask) as u8
        })
        .collect()
}

// ===========================================================================
// ЦВЕТ
// ===========================================================================

fn chroma(args: &[String]) {
    let dir = args.get(2).expect("каталог дампов");
    let crop = parse_crop(args.get(3));
    let truth_file = args.get(4).filter(|s| !s.contains(',')).cloned();
    let seq_max: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let dumps = load_dumps(dir, crop, usize::MAX, true);
    assert!(!dumps.is_empty(), "нет дампов в {dir}");
    let p = tx_chromatic_profile();
    let bpc = symbol::bits_per_cell(&p);
    println!(
        "=== ISI / ЦВЕТ: {} кадров {}x{} из {dir}; §5.1-CL, {} бит/клетку ===",
        dumps.len(),
        dumps[0].w,
        dumps[0].h,
        bpc
    );
    let dets = align(&dumps);
    let cl = symbol::const_luma_map(&p);
    let cls = symbol::CL_LATTICE_SCALE;
    let frames: Vec<Vec<u8>> = match &truth_file {
        Some(f) => build_truth_frames(f, bpc, seq_max),
        None => Vec::new(),
    };

    let cfg = IsiConfig::default();
    let mut cfg5 = cfg;
    cfg5.radius = 2;
    let mut cfg2p = cfg;
    cfg2p.passes = 2;

    // индексы payload-клеток в сетке символа
    let idx_pay: Vec<usize> = (0..NCELL)
        .map(|i| (RING + 1 + i / PAYLOAD_COLS) * GRID + RING + i % PAYLOAD_COLS)
        .collect();

    /// Всё, что нужно от кадра во ВТОРОЙ фазе (проверка фиксированного ядра).
    struct Ctx {
        #[allow(dead_code)]
        gam: [f64; 3],
        t: [Vec<f64>; 3],
        truth: Option<Vec<u8>>,
        seq: usize,
        base: Vec<u8>,
        torn: bool,
        /// Разделение оси Re ДО выравнивания — им отделяются снимки, накрывшие
        /// границу двух кадров передатчика (там созвездие схлопнуто, и ISI ни
        /// при чём).
        qre: f64,
    }

    let mut ctxs: Vec<Option<Ctx>> = Vec::new();
    let mut kern_ch: Vec<[IsiKernel; 3]> = Vec::new();
    let (mut tb, mut te, mut nt) = (0.0f64, 0.0f64, 0.0f64);

    println!(
        "\n{:<5} {:>6} {:>9} {:>9} {:>7} {:>7} {:>7}  {}",
        "кадр", "seq", "счётчик", "score", "Q Re", "Q Im", "торн", "решатели ниже"
    );
    for (fi, dp) in dumps.iter().enumerate() {
        let Some(d) = dets[fi].as_ref() else {
            println!("{fi:<5} детекция не удалась");
            ctxs.push(None);
            continue;
        };
        let map = detect::frame_map(&p, d);
        let raw = |x: f64, y: f64| dp.raw(x, y);
        let g = tone::estimate_channel_gammas(&p, &map, &raw);
        let gam = [g[0], g[1], g[2]];
        let lin = |x: f64, y: f64| -> [f32; 3] {
            let s = raw(x, y);
            [
                (s[0] as f64).max(0.0).powf(gam[0]) as f32,
                (s[1] as f64).max(0.0).powf(gam[1]) as f32,
                (s[2] as f64).max(0.0).powf(gam[2]) as f32,
            ]
        };

        // --- продакшн-пути, с хронометражем ---
        let t0 = Instant::now();
        let base = symbol::demod_symbol(&p, &map, &lin);
        tb += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let eq = symbol::demod_symbol_isi(&p, &map, &lin, None, &cfg);
        te += t1.elapsed().as_secs_f64() * 1e3;
        nt += 1.0;
        let eq5 = symbol::demod_symbol_isi(&p, &map, &lin, None, &cfg5);
        let eq2 = symbol::demod_symbol_isi(&p, &map, &lin, None, &cfg2p);
        if eq.applied {
            kern_ch.push(eq.kernels);
        }

        // --- светолинейные плоскости канала (та же величина, что внутри демода) ---
        let s_grid = symbol::sample_symbol_grid(&p, &map, &lin);
        // Гаммы ПРОФИЛЯ, а не измеренные: demod_symbol строит матрицу развязки
        // именно по ним, и «база» этой площадки обязана совпадать с продакшн-
        // путём клетка-в-клетку, иначе «до/после» сравнивают разные «до».
        let pg = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
        let m = symbol::estimate_matrix_reference_row(&p, &pg, &map, &lin)
            .expect("матрица развязки вырождена");
        let mut t: [Vec<f64>; 3] = [vec![0.0; NGRID], vec![0.0; NGRID], vec![0.0; NGRID]];
        for (i, s) in s_grid.iter().enumerate() {
            let l = m.apply(*s);
            for c in 0..3 {
                t[c][i] = l[c];
            }
        }
        let axes = |tp: &[Vec<f64>; 3]| -> (Vec<f64>, Vec<f64>) {
            let (mut xs, mut ys) = (vec![0.0; NGRID], vec![0.0; NGRID]);
            for i in 0..NGRID {
                let mut dr = [0.0f64; 3];
                for c in 0..3 {
                    dr[c] = (255.0 * tp[c][i].max(0.0).powf(1.0 / pg[c])).clamp(0.0, 255.0);
                }
                let (x, y) = cl.z_from_drive(dr);
                xs[i] = x / cls;
                ys[i] = y / cls;
            }
            (xs, ys)
        };
        let (xs, ys) = axes(&t);

        // --- истина и признак рваного кадра ---
        let (truth, seq) = lock_truth(&frames, &base, bpc);
        let (c0, c1) = symbol::read_counters(&p, &map, &lin);
        let torn = c0 != c1;
        let have = truth.is_some();
        let tr = truth.clone().unwrap_or_else(|| base.clone());
        let cre: Vec<u8> = (0..NCELL).map(|i| (tr[i] >> 1) & 1).collect();
        let cim: Vec<u8> = (0..NCELL).map(|i| tr[i] & 1).collect();
        let pay = |v: &[f64]| -> Vec<f64> { idx_pay.iter().map(|&i| v[i]).collect() };
        let (qre, _, _, _) = q_factor(&pay(&xs), &cre);
        let (qim, _, _, _) = q_factor(&pay(&ys), &cim);
        println!(
            "{fi:<5} {seq:>6} {:>4}/{:<4} {:>9.4} {qre:>7.2} {qim:>7.2} {:>7}",
            c0,
            c1,
            d.score,
            if torn { "ДА" } else { "нет" }
        );
        if !have {
            println!("      ! истина не залочена");
        }

        // --- Q ПОСЛЕ канального выравнивателя (это и есть продакшн-путь) ---
        let mut teq = t.clone();
        for c in 0..3 {
            isi::equalise(&mut teq[c], GRID, GRID, &eq.kernels[c], cfg.iters);
        }
        let (xe, ye) = axes(&teq);
        let (qre2, _, _, _) = q_factor(&pay(&xe), &cre);
        let (qim2, _, _, _) = q_factor(&pay(&ye), &cim);
        println!(
            "      Q после канального выравнивателя: Re {qre:.2} -> {qre2:.2}σ ({:+.2} дБ) | \
             Im {qim:.2} -> {qim2:.2}σ ({:+.2} дБ)",
            20.0 * (qre2 / qre.max(1e-9)).log10(),
            20.0 * (qim2 / qim.max(1e-9)).log10()
        );
        for c in 0..3 {
            print_kernel(&format!("ядро канала {}", ["R", "G", "B"][c]), &eq.kernels[c]);
        }

        // --- три решателя на ОБЩЕЙ площадке: оси созвездия ---
        let counter = c0;
        let ideal_d = symbol::ideal_symbol_drives(&p, &base, counter);
        let (mut ix, mut iy) = (vec![0.0; NGRID], vec![0.0; NGRID]);
        for i in 0..NGRID {
            let dr = [
                ideal_d[i][0] as f64,
                ideal_d[i][1] as f64,
                ideal_d[i][2] as f64,
            ];
            let (x, y) = cl.z_from_drive(dr);
            ix[i] = x / cls;
            iy[i] = y / cls;
        }
        let gx = Grid { v: &xs, rows: GRID, cols: GRID };
        let gy = Grid { v: &ys, rows: GRID, cols: GRID };
        let gix = Grid { v: &ix, rows: GRID, cols: GRID };
        let giy = Grid { v: &iy, rows: GRID, cols: GRID };
        let kx = isi::estimate_kernel(&gx, &gix, cfg.radius, FIT_MARGIN, cfg.shape);
        let ky = isi::estimate_kernel(&gy, &giy, cfg.radius, FIT_MARGIN, cfg.shape);

        let mut lx = xs.clone();
        let mut ly = ys.clone();
        if let Some(k) = &kx {
            isi::equalise(&mut lx, GRID, GRID, k, cfg.iters);
        }
        if let Some(k) = &ky {
            isi::equalise(&mut ly, GRID, GRID, k, cfg.iters);
        }
        let hard = |v: f64| if v > 0.0 { 1.0 } else { -1.0 };
        let (mut dx, mut dy) = (xs.clone(), ys.clone());
        let (mut rx, mut ry) = (Default::default(), Default::default());
        if let Some(k) = &kx {
            rx = isi::equalise_dfe(&mut dx, GRID, GRID, k, &hard);
        }
        if let Some(k) = &ky {
            ry = isi::equalise_dfe(&mut dy, GRID, GRID, k, &hard);
        }
        let (mut lab_x, mut lab_y) = (vec![usize::MAX; NGRID], vec![usize::MAX; NGRID]);
        for (i, &gi) in idx_pay.iter().enumerate() {
            lab_x[gi] = ((base[i] >> 1) & 1) as usize;
            lab_y[gi] = (base[i] & 1) as usize;
        }
        let jx = JointRule::train(&gx, &lab_x, &[-1.0, 1.0], cfg.radius, FIT_MARGIN);
        let jy = JointRule::train(&gy, &lab_y, &[-1.0, 1.0], cfg.radius, FIT_MARGIN);
        let assemble = |xp: &[f64], yp: &[f64]| -> Vec<u8> {
            idx_pay
                .iter()
                .map(|&gi| (((xp[gi] > 0.0) as u8) << 1) | (yp[gi] > 0.0) as u8)
                .collect()
        };
        let cells_l = assemble(&lx, &ly);
        let cells_d = assemble(&dx, &dy);
        let cells_j: Vec<u8> = match (&jx, &jy) {
            (Some(a), Some(b)) => {
                let (da, db) = (a.decide_all(&gx), b.decide_all(&gy));
                idx_pay
                    .iter()
                    .map(|&gi| ((da[gi] as u8) << 1) | db[gi] as u8)
                    .collect()
            }
            _ => base.clone(),
        };

        println!(
            "\n      {:<30} {:>8} {:>8} {:>8}  страйпы",
            "решатель", "Re SER", "Im SER", "симв."
        );
        let variants: Vec<(&str, &Vec<u8>)> = vec![
            ("база demod_symbol", &base),
            ("L каналы 3×3 (продакшн)", &eq.cells),
            ("L каналы 3×3, 2 прохода", &eq2.cells),
            ("L каналы 5×5", &eq5.cells),
            ("L оси 3×3", &cells_l),
            ("D оси 3×3 (обр. связь)", &cells_d),
            ("J совместное правило", &cells_j),
        ];
        for (name, got) in &variants {
            let (re, im, sy) = axis_ser(got, &tr, have);
            let parsed = l3::parse_frame(got, bpc);
            let so = parsed.stripes_ok;
            let hdr = match &parsed.header {
                Some(h) => format!(" ЗАГОЛОВОК sid={:08x} esi={}", h.session_id, h.esi),
                None => String::new(),
            };
            println!(
                "      {name:<30} {re:>8.5} {im:>8.5} {sy:>8.5}  {} {}/8{hdr}",
                fmt_stripes(&so),
                so.iter().filter(|&&b| b).count()
            );
        }
        println!(
            "      DFE: перевёрнуто Re {} / Im {}, петля {:.3}/{:.3}",
            rx.flipped, ry.flipped, rx.loop_gain, ry.loop_gain
        );
        if have {
            burst_report("      пачки база", &base, &tr);
            burst_report("      пачки L   ", &eq.cells, &tr);
            burst_report("      пачки DFE ", &cells_d, &tr);
            blend_report(&frames, &base, seq);
        }

        ctxs.push(Some(Ctx {
            gam,
            t,
            truth,
            seq,
            base,
            torn,
            qre,
        }));
    }

    println!(
        "\nцена: demod_symbol {:.2} мс/кадр, demod_symbol_isi {:.2} мс/кадр (+{:.2} мс)",
        tb / nt,
        te / nt,
        (te - tb) / nt
    );

    // =====================================================================
    // ФИКСИРОВАННОЕ ЯДРО: свойство связки «дисплей+оптика+ISP» или кадра?
    // =====================================================================
    if kern_ch.len() >= 2 {
        println!("\n=== ФИКСИРОВАННОЕ ЯДРО ПРОТИВ ПОКАДРОВОЙ ОЦЕНКИ ===");
        // рваные кадры исключаются из МЕДИАНЫ: их «идеал» построен из решений,
        // которые относятся к ДВУМ разным кадрам передатчика, и оценка занижена.
        let clean: Vec<usize> = (0..ctxs.len())
            .filter(|&i| {
                ctxs[i]
                    .as_ref()
                    .map_or(false, |c| !c.torn && c.qre >= Q_CLEAN_MIN)
            })
            .collect();
        println!(
            "кадры с НЕсхлопнутым созвездием (Q_Re >= {Q_CLEAN_MIN}): {clean:?} из {}",
            ctxs.len()
        );
        let pick: Vec<[IsiKernel; 3]> = if clean.len() >= 2 {
            clean
                .iter()
                .filter_map(|&i| kern_ch.get(i).copied())
                .collect()
        } else {
            kern_ch.clone()
        };
        let med: [IsiKernel; 3] = [
            median_kernel(&pick.iter().map(|k| k[0]).collect::<Vec<_>>()),
            median_kernel(&pick.iter().map(|k| k[1]).collect::<Vec<_>>()),
            median_kernel(&pick.iter().map(|k| k[2]).collect::<Vec<_>>()),
        ];
        for c in 0..3 {
            print_kernel(&format!("МЕДИАННОЕ ядро {}", ["R", "G", "B"][c]), &med[c]);
            let mut worst = 0.0f64;
            for k in &pick {
                worst = worst.max(med[c].max_tap_diff(&k[c]));
            }
            println!("      макс. расхождение по кадрам: {worst:.4}");
        }
        // ЦЕНА рабочего режима: ядро взято из кэша, оценка не выполняется.
        {
            let mut cached = cfg;
            cached.kernel = Some(med);
            let (mut tc, mut nc2) = (0.0f64, 0.0f64);
            for (fi, dp) in dumps.iter().enumerate() {
                let Some(d) = dets[fi].as_ref() else { continue };
                let map = detect::frame_map(&p, d);
                let raw = |x: f64, y: f64| dp.raw(x, y);
                let g = tone::estimate_channel_gammas(&p, &map, &raw);
                let lin = |x: f64, y: f64| -> [f32; 3] {
                    let s = raw(x, y);
                    [
                        (s[0] as f64).max(0.0).powf(g[0]) as f32,
                        (s[1] as f64).max(0.0).powf(g[1]) as f32,
                        (s[2] as f64).max(0.0).powf(g[2]) as f32,
                    ]
                };
                let t = Instant::now();
                let _ = symbol::demod_symbol_isi(&p, &map, &lin, None, &cached);
                tc += t.elapsed().as_secs_f64() * 1e3;
                nc2 += 1.0;
            }
            println!(
                "цена с КЭШИРОВАННЫМ ядром: {:.2} мс/кадр (без выравнивателя {:.2}, +{:.2} мс)",
                tc / nc2,
                tb / nt,
                tc / nc2 - tb / nt
            );
            // Устойчивый замер: один кадр, много повторов, МЕДИАНА — иначе
            // цифра гуляет от фоновой нагрузки машины больше, чем сама разница.
            if let Some((fi, d)) = dets.iter().enumerate().find_map(|(i, d)| d.as_ref().map(|x| (i, x)))
            {
                let dp = &dumps[fi];
                let map = detect::frame_map(&p, d);
                let raw = |x: f64, y: f64| dp.raw(x, y);
                let g = tone::estimate_channel_gammas(&p, &map, &raw);
                let lin = |x: f64, y: f64| -> [f32; 3] {
                    let s = raw(x, y);
                    [
                        (s[0] as f64).max(0.0).powf(g[0]) as f32,
                        (s[1] as f64).max(0.0).powf(g[1]) as f32,
                        (s[2] as f64).max(0.0).powf(g[2]) as f32,
                    ]
                };
                let reps = 40;
                let mut ta = Vec::with_capacity(reps);
                let mut tcc = Vec::with_capacity(reps);
                let mut tee = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let t = Instant::now();
                    let _ = symbol::demod_symbol(&p, &map, &lin);
                    ta.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    let _ = symbol::demod_symbol_isi(&p, &map, &lin, None, &cached);
                    tcc.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    let _ = symbol::demod_symbol_isi(&p, &map, &lin, None, &cfg);
                    tee.push(t.elapsed().as_secs_f64() * 1e3);
                }
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v[v.len() / 2]
                };
                let (a, b, c) = (med(&mut ta), med(&mut tcc), med(&mut tee));
                println!(
                    "МЕДИАНА по {reps} повторам одного кадра: demod_symbol {a:.2} мс, \
                     +ISI(кэш) {b:.2} мс (+{:.2}), +ISI(оценка каждый кадр) {c:.2} мс (+{:.2})",
                    b - a,
                    c - a
                );
            }
        }
        // ---------------------------------------------------------------
        // АБЛЯЦИЯ: форма носителя × число итераций.
        // ---------------------------------------------------------------
        // Прямая проверка двух возражений: (а) не подгоняют ли диагональные
        // отсчёты шум (анизотропия помехи на квадратной решётке ~13×) и (б) не
        // расходится ли обращение с ростом числа итераций (усиление шума
        // деконволюцией). Оба вопроса решаются числом на РЕАЛЬНЫХ снимках.
        println!("\n=== АБЛЯЦИЯ: форма носителя × итерации (только чистые кадры) ===");
        // ОБЩЕЕ ядро на все три канала: ось Im = R−B чувствительна к РАЗНИЦЕ
        // ошибок оценки ядер R и B (независимые подгонки вносят в разность свою
        // невязку), тогда как ось Re = 2G−R−B к ней почти нечувствительна.
        // Проверяем, не выгоднее ли принудительно связать каналы.
        let shared = median_kernel(
            &pick
                .iter()
                .flat_map(|k| k.iter().copied())
                .collect::<Vec<_>>(),
        );
        print_kernel("ОБЩЕЕ ядро на все каналы", &shared);
        println!("{:<8} {:>6} {:>10} {:>16}", "форма", "итер", "SER", "живых страйпов");
        for (sname, shape) in [("крест", KernelShape::Cross), ("квадрат", KernelShape::Full)] {
            for iters in [1usize, 2, 3, 4] {
                let mut c = cfg;
                c.shape = shape;
                c.iters = iters;
                let (mut e, mut al, mut nn) = (0usize, 0usize, 0usize);
                for (fi, dp) in dumps.iter().enumerate() {
                    if !clean.contains(&fi) {
                        continue;
                    }
                    let Some(d) = dets[fi].as_ref() else { continue };
                    let Some(cx) = ctxs[fi].as_ref() else { continue };
                    let Some(tr) = cx.truth.as_ref() else { continue };
                    let map = detect::frame_map(&p, d);
                    let raw = |x: f64, y: f64| dp.raw(x, y);
                    let g = tone::estimate_channel_gammas(&p, &map, &raw);
                    let lin = |x: f64, y: f64| -> [f32; 3] {
                        let s = raw(x, y);
                        [
                            (s[0] as f64).max(0.0).powf(g[0]) as f32,
                            (s[1] as f64).max(0.0).powf(g[1]) as f32,
                            (s[2] as f64).max(0.0).powf(g[2]) as f32,
                        ]
                    };
                    let got = symbol::demod_symbol_isi(&p, &map, &lin, None, &c).cells;
                    let skip = l3::STRIPE_ROWS[0] * PAYLOAD_COLS;
                    e += (skip..NCELL).filter(|&i| got[i] != tr[i]).count();
                    al += stripes_ok(&got, bpc).iter().filter(|&&b| b).count();
                    nn += 1;
                }
                if nn == 0 {
                    continue;
                }
                let n = nn as f64 * (NCELL - l3::STRIPE_ROWS[0] * PAYLOAD_COLS) as f64;
                println!(
                    "{sname:<8} {iters:>6} {:>10.5} {:>12}/{}",
                    e as f64 / n,
                    al,
                    nn * 8
                );
            }
        }

        println!(
            "\n{:<5} {:>10} {:>10} {:>10} {:>10}",
            "кадр", "база SER", "своё ядро", "медиана", "страйпы б/с/м"
        );
        let (mut ab, mut ao, mut am) = (0usize, 0usize, 0usize);
        let (mut sb, mut so, mut sm) = (0usize, 0usize, 0usize);
        // те же суммы, но ТОЛЬКО по кадрам с несхлопнутым созвездием
        let (mut cb, mut co, mut cm, mut cs) = (0usize, 0usize, 0usize, 0usize);
        let (mut csb, mut cso, mut csm, mut css) = (0usize, 0usize, 0usize, 0usize);
        let (mut asx, mut ssx) = (0usize, 0usize);
        for (fi, cx) in ctxs.iter().enumerate() {
            let Some(cx) = cx else { continue };
            let Some(tr) = cx.truth.as_ref() else { continue };
            let decode = |k: Option<&[IsiKernel; 3]>| -> Vec<u8> {
                let mut tp = cx.t.clone();
                if let Some(k) = k {
                    for c in 0..3 {
                        isi::equalise(&mut tp[c], GRID, GRID, &k[c], cfg.iters);
                    }
                }
                // гаммы ПРОФИЛЯ, а не измеренные: так строка «база» этой
                // таблицы совпадает с продакшн-путём symbol::demod_symbol.
                let pg = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
                let codec_decode = |i: usize| -> u8 {
                    let mut dr = [0.0f64; 3];
                    for c in 0..3 {
                        dr[c] = (255.0 * tp[c][i].max(0.0).powf(1.0 / pg[c])).clamp(0.0, 255.0);
                    }
                    let (x, y) = cl.z_from_drive(dr);
                    (((x > 0.0) as u8) << 1) | (y > 0.0) as u8
                };
                idx_pay.iter().map(|&gi| codec_decode(gi)).collect()
            };
            let own = kern_ch.get(fi).copied();
            let g_base = decode(None);
            let g_own = own.map(|k| decode(Some(&k))).unwrap_or_else(|| g_base.clone());
            let g_med = decode(Some(&med));
            let g_sh = decode(Some(&[shared; 3]));
            let e = |g: &Vec<u8>| -> usize {
                let skip = l3::STRIPE_ROWS[0] * PAYLOAD_COLS;
                (skip..NCELL).filter(|&i| g[i] != tr[i]).count()
            };
            let n = (NCELL - l3::STRIPE_ROWS[0] * PAYLOAD_COLS) as f64;
            let (eb, eo, em, es) = (e(&g_base), e(&g_own), e(&g_med), e(&g_sh));
            ab += eb;
            ao += eo;
            am += em;
            asx += es;
            let alive = |g: &Vec<u8>| stripes_ok(g, bpc).iter().filter(|&&b| b).count();
            sb += alive(&g_base);
            so += alive(&g_own);
            sm += alive(&g_med);
            ssx += alive(&g_sh);
            if clean.contains(&fi) {
                cb += eb;
                co += eo;
                cm += em;
                cs += es;
                csb += alive(&g_base);
                cso += alive(&g_own);
                csm += alive(&g_med);
                css += alive(&g_sh);
            }
            println!(
                "{fi:<5} {:>10.5} {:>10.5} {:>10.5}   {}/{}/{}{}",
                eb as f64 / n,
                eo as f64 / n,
                em as f64 / n,
                alive(&g_base),
                alive(&g_own),
                alive(&g_med),
                if cx.torn { "  (рваный)" } else { "" }
            );
            let _ = cx.seq;
            let _ = &cx.base;
        }
        let nfr = ctxs
            .iter()
            .filter(|c| c.as_ref().map_or(false, |x| x.truth.is_some()))
            .count();
        let n = nfr as f64 * (NCELL - l3::STRIPE_ROWS[0] * PAYLOAD_COLS) as f64;
        println!(
            "ИТОГО (все кадры) SER: база {:.5} -> своё ядро {:.5} -> МЕДИАННОЕ ядро {:.5}",
            ab as f64 / n,
            ao as f64 / n,
            am as f64 / n
        );
        println!(
            "ИТОГО (все кадры) живых страйпов: база {sb}/{n8} -> своё {so}/{n8}              -> медиана {sm}/{n8} -> ОБЩЕЕ {ssx}/{n8} (SER общего {:.5})",
            asx as f64 / n,
            n8 = nfr * 8
        );
        let nc = clean.len();
        if nc > 0 && nc < nfr {
            let nn = nc as f64 * (NCELL - l3::STRIPE_ROWS[0] * PAYLOAD_COLS) as f64;
            println!(
                "ТОЛЬКО чистые кадры SER: база {:.5} -> своё {:.5} -> медиана {:.5}                  -> ОБЩЕЕ на каналы {:.5}",
                cb as f64 / nn,
                co as f64 / nn,
                cm as f64 / nn,
                cs as f64 / nn
            );
            println!(
                "ТОЛЬКО чистые кадры живых страйпов: база {csb}/{n8} -> своё {cso}/{n8}                  -> медиана {csm}/{n8} -> ОБЩЕЕ {css}/{n8}",
                n8 = nc * 8
            );
        }
    }
}

/// Поосевые SER: (Re, Im, символьная). Страйп 0 sid-зависим — исключается.
fn axis_ser(got: &[u8], truth: &[u8], have: bool) -> (f64, f64, f64) {
    if !have {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let skip = l3::STRIPE_ROWS[0] * PAYLOAD_COLS;
    let (mut re, mut im, mut sy) = (0usize, 0usize, 0usize);
    let mut n = 0usize;
    for i in skip..NCELL {
        if ((got[i] >> 1) & 1) != ((truth[i] >> 1) & 1) {
            re += 1;
        }
        if (got[i] & 1) != (truth[i] & 1) {
            im += 1;
        }
        if got[i] != truth[i] {
            sy += 1;
        }
        n += 1;
    }
    let n = n as f64;
    (re as f64 / n, im as f64 / n, sy as f64 / n)
}

/// Распределение длин ПАЧЕК ошибок в растровом порядке. Для страйпового CRC
/// решает не число ошибок, а то, кучно ли они лежат: 10 ошибок в одной пачке
/// убивают один страйп, 10 рассыпанных — до восьми.
fn burst_report(tag: &str, got: &[u8], truth: &[u8]) {
    // страйп 0 несёт sid-зависимый заголовок — эталон там заведомо неверен
    let skip = l3::STRIPE_ROWS[0] * PAYLOAD_COLS;
    let mut bursts: Vec<usize> = Vec::new();
    let mut run = 0usize;
    for i in skip..NCELL {
        if got[i] != truth[i] {
            run += 1;
        } else if run > 0 {
            bursts.push(run);
            run = 0;
        }
    }
    if run > 0 {
        bursts.push(run);
    }
    if bursts.is_empty() {
        println!("{tag}: ошибок нет");
        return;
    }
    let total: usize = bursts.iter().sum();
    let longest = *bursts.iter().max().unwrap();
    let hist = |k: usize| bursts.iter().filter(|&&b| b == k).count();
    let mut dead = 0usize;
    let mut lo = skip;
    for &rh in l3::STRIPE_ROWS.iter().skip(1) {
        let hi = lo + rh * PAYLOAD_COLS;
        if (lo..hi.min(NCELL)).any(|i| got[i] != truth[i]) {
            dead += 1;
        }
        lo = hi;
    }
    println!(
        "{tag}: {total} ош. в {} пачках (1:{} 2:{} 3+:{}), макс {longest}; \
         битых страйпов 1..7: {dead}/7",
        bursts.len(),
        hist(1),
        hist(2),
        bursts.iter().filter(|&&b| b >= 3).count()
    );
}

/// Диагностика СМЕШАННОГО кадра: снимок, чья экспозиция накрыла границу двух
/// кадров передатчика, содержит смесь двух РАЗНЫХ нагрузок. Такой кадр не
/// лечится никаким выравнивателем — помеха там не от соседей, а от другого
/// кадра, — и отличать его обязательно, иначе он утащит вниз общую SER и
/// создаст впечатление, что выравниватель не работает.
///
/// Признак: доля ошибок среди клеток, где соседний по времени кадр ОТЛИЧАЕТСЯ
/// от залоченного. У чистого снимка она равна общей SER; у смешанного — резко
/// выше (в пределе 0.5: смесь двух уровней садится на порог).
fn blend_report(frames: &[Vec<u8>], got: &[u8], seq: usize) {
    let skip = l3::STRIPE_ROWS[0] * PAYLOAD_COLS;
    let cur = &frames[seq];
    let base_err = (skip..NCELL).filter(|&i| got[i] != cur[i]).count() as f64
        / (NCELL - skip) as f64;
    let mut out = format!("      смешение: общая SER {base_err:.4}");
    for (lbl, other) in [("пред", seq.wrapping_sub(1)), ("след", seq + 1)] {
        let Some(o) = frames.get(other) else { continue };
        let diff: Vec<usize> = (skip..NCELL).filter(|&i| o[i] != cur[i]).collect();
        if diff.is_empty() {
            continue;
        }
        let bad = diff.iter().filter(|&&i| got[i] != cur[i]).count();
        out.push_str(&format!(
            "; на клетках, где {lbl}. кадр иной ({}): SER {:.4}",
            diff.len(),
            bad as f64 / diff.len() as f64
        ));
    }
    println!("{out}");
}

/// Восстановление потока передатчика для сверки (зеркало `chroma_diag.rs`).
fn build_truth_frames(file: &str, bpc: u32, seq_max: usize) -> Vec<Vec<u8>> {
    let data = fs::read(file).expect("truth file");
    let cap: usize = l3::STRIPE_ROWS
        .iter()
        .map(|&r| (r * l3::PAYLOAD_COLS * bpc as usize - 16) / 8)
        .sum();
    let symbol_size = (cap - l3::FRAME_HEADER_LEN - l3::TRANSFER_INFO_LEN) / SYMBOLS_PER_FRAME;
    let enc = FountainEncoder::new(&data, symbol_size);
    let k = enc.k();
    let mut emit = Vec::new();
    {
        let (mut src, mut rep, mut since) = (0u32, k, 0u32);
        while src < k {
            emit.push(src);
            src += 1;
            since += 1;
            if since == REPAIR_EVERY {
                emit.push(rep);
                rep += 1;
                since = 0;
            }
        }
    }
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
    (0..seq_max)
        .map(|seq| {
            let base = seq * SYMBOLS_PER_FRAME;
            let mut bytes = Vec::with_capacity(SYMBOLS_PER_FRAME * symbol_size);
            for i in 0..SYMBOLS_PER_FRAME {
                bytes.extend_from_slice(&enc.symbol(esi_at(base + i)));
            }
            let mut hd = FrameHeader::new(0, esi_at(base), SYMBOLS_PER_FRAME as u8);
            let t = if seq % 8 == 0 {
                hd.flags |= FLAG_TRANSFER_INFO;
                Some(&ti)
            } else {
                None
            };
            l3::build_frame(&hd, t, &bytes, bpc)
        })
        .collect()
}

/// Лок на кадр транспорта по МИНИМУМУ битовых ошибок, СТРАЙПЫ 1..7 (страйп 0
/// содержит sid-зависимый заголовок). Лок принимается, только если отрыв от
/// второго кандидата велик.
fn lock_truth(frames: &[Vec<u8>], got: &[u8], _bpc: u32) -> (Option<Vec<u8>>, usize) {
    if frames.is_empty() {
        return (None, 0);
    }
    let skip = l3::STRIPE_ROWS[0] * PAYLOAD_COLS;
    let mut best = (usize::MAX, 0usize);
    let mut second = usize::MAX;
    for (s, cells) in frames.iter().enumerate() {
        let mut wrong = 0usize;
        for i in skip..NCELL {
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
    if (best.0 as f64) < 0.6 * second as f64 {
        (Some(frames[best.1].clone()), best.1)
    } else {
        (None, best.1)
    }
}
