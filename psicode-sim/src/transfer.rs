//! Сквозная передача файла в симе (§6.1/§9.2): XOR-фонтан (EXPERIMENTAL) ->
//! кадры L3 -> канал -> РЕАЛЬНАЯ детекция ЗЧ-рамки (§3.2) -> demod -> salvage
//! страйпов -> фонтан-декод -> CRC-32C. Первый полностековый прогон формата.
//!
//! # Раскладка символов в кадре (проектное решение, под §6.2)
//!
//! Конфиг — σ=1-оптимум из BENCHMARKS §6: luma 2 + Chroma1 = **3 бит/клетку**,
//! cell 16 px, px/клетку 8, канал = телеметрия-правда эталона §7.4, σ_blur —
//! свипается. При 3 бит/клетку ёмкости 8 страйпов (§6.2):
//!
//! ```text
//! caps = [147,147,147,147,147,147,147,126] байт  (= (rows·57·3 − 16)/8)
//! итого frame_byte_capacity(3) = 1155 байт
//! ```
//!
//! Служебная шапка: FrameHeader 12 байт [+ TransferInfo 14 байт в каждом 8-м].
//! Символьная область: не-TI кадр 1143 байт, TI-кадр 1129 байт. Берём
//! **symbol_size = 140**, **8 символов/кадр** → 1120 байт символов, влезает и в
//! TI-кадр (1120 ≤ 1129), остаток добивается нулями. Фиксированное число
//! символов на кадр делает индекс кадра N = emission_index/8 тривиально
//! обратимым из заголовка (нужно для атрибуции рваных снимков §6.3).
//!
//! # Порядок ESI (§6.1, fec_overhead=2)
//!
//! Поток encoding-символов: систематические source (ESI 0..K−1) с ВПЛЕТЕНИЕМ
//! одного repair-символа после каждых 4 source; когда систематический проход
//! исчерпан (все K source отправлены), поток продолжается только repair
//! (ESI ≥ K, неограниченно). Кадр `seq` несёт 8 подряд идущих элементов потока
//! `emit[seq·8 .. seq·8+8]`; заголовок.esi = первый ESI кадра. Приёмник
//! воспроизводит тот же `emit` из K (получен в TransferInfo) и по заголовку.esi
//! находит позицию кадра в потоке, размечая ESI всех его символов.
//!
//! # Атрибуция рваного снимка (§6.3)
//!
//! Разрыв компонуется ДО канала (rolling-shutter): строки [0,t) кадра N, строки
//! [t,H) кадра N+1 (позиция t — внутри случайной payload-строки, рвёт ровно один
//! страйп на шве). Строка счётчика (§3.3) лежит ниже payload → в рваном снимке
//! принадлежит N+1, поэтому обе её копии читают (N+1)&0xFF ≠ N&0xFF — так разрыв
//! и детектируется. Тогда верхний непрерывный прогон CRC-валидных страйпов
//! атрибутируется кадру N (заголовок страйпа 0), нижний — кадру N+1 (его ESI
//! предсказуемы: N+1 = N+1 при фиксированных 8 символах/кадр). CRC-16 гарантирует
//! байты, счётчик гарантирует атрибуцию — фонтан не получает ложных пар (esi,bytes).

use std::collections::HashMap;
use std::time::Instant;

use psicode_core::detect::{detect_symbol, frame_map};
use psicode_core::fountain::{crc32c, FountainDecoder, FountainEncoder};
use psicode_core::l3::{self, FrameHeader, ParsedFrame, TransferInfo};
use psicode_core::symbol::{self, demod_symbol, read_counters, render_symbol_counter};
use psicode_core::{CalibProfile, ChromaMode};

use crate::channel::ChannelParams;
use crate::framed::compose_torn;
use crate::pipeline::apply_channel;
use crate::rng::{seed_for, Rng};

/// Размер encoding-символа (байт) — см. раскладку в модульдоке.
const SYMBOL_SIZE: usize = 140;
/// Число символов в кадре (фиксировано — упрощает разметку ESI и §6.3).
const SYMBOLS_PER_FRAME: usize = 8;
/// Полезная нагрузка по умолчанию (~20 КБ, §9.2). Константа.
const PAYLOAD_BYTES: usize = 20_000;
/// Вплетать repair после каждых 4 source (§6.1, fec_overhead=2).
const REPAIR_EVERY: u32 = 4;

/// σ=1-оптимальный профиль (BENCHMARKS §6: luma 2 + Chroma1 = 3 бит/клетку).
/// Телеметрия — как у эталона §7.4 (тот же «правдивый» канал).
fn transfer_profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        luma_bits: 2,
        chroma_mode: ChromaMode::Chroma1,
        gamma_g_q: 28,
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15,
        black_level_q: 2,
        noise_sigma_q: 12,
        mtf_limit_px: 6,
        torn_frames_q: 5,
        crosstalk_rg_q: 3,
        crosstalk_gb_q: 4,
        quiet_zone: 1,
        fec_overhead: 2,
    }
}

/// Детерминированная псевдослучайная нагрузка (splitmix64) заданной длины.
fn make_payload(len: usize) -> Vec<u8> {
    let mut state = 0x0D15_EA5E_5EED_1234u64;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 24) as u8
        })
        .collect()
}

/// Порядок ESI потока (§6.1): source 0..K−1 с вплетением repair каждые
/// `REPAIR_EVERY` source; после систематического прохода — только repair.
/// Генерирует ровно `n_frames · SYMBOLS_PER_FRAME` элементов.
fn emission_order(k: u32, n_frames: usize) -> Vec<u32> {
    let total = n_frames * SYMBOLS_PER_FRAME;
    let mut v = Vec::with_capacity(total);
    let mut src = 0u32;
    let mut rep = k; // следующий repair-ESI
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

/// Байты 8 символов кадра: конкатенация `enc.symbol(emit[base+i])`.
fn build_symbol_bytes(enc: &FountainEncoder, emit: &[u32], base: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(SYMBOLS_PER_FRAME * SYMBOL_SIZE);
    for i in 0..SYMBOLS_PER_FRAME {
        out.extend_from_slice(&enc.symbol(emit[base + i]));
    }
    out
}

/// TransferInfo кадра `seq` (§6.2, каждый 8-й кадр несёт его).
fn ti_for(seq: u32, payload_len: usize, k: u32, checksum: u32) -> Option<TransferInfo> {
    if seq % 8 == 0 {
        Some(TransferInfo {
            transfer_length: payload_len as u64,
            symbol_size: SYMBOL_SIZE as u16,
            k,
            checksum,
        })
    } else {
        None
    }
}

/// Байт-ёмкости и кумулятивные смещения 8 страйпов при `bpc` (§6.2).
fn stripe_layout(bpc: u32) -> ([usize; 8], [usize; 9]) {
    let mut caps = [0usize; 8];
    let mut cum = [0usize; 9];
    for i in 0..8 {
        caps[i] = (l3::STRIPE_ROWS[i] * l3::PAYLOAD_COLS * bpc as usize - 16) / 8;
        cum[i + 1] = cum[i] + caps[i];
    }
    (caps, cum)
}

/// Восстановить символы кадра из выбранных страйпов и скормить фонтану.
/// `include[i]` — учитывать ли страйп i как принадлежащий этому кадру;
/// `j0` — emission-индекс первого символа кадра, `header_len` — длина шапки.
/// Возвращает (получено символов, из них новых для декодера).
#[allow(clippy::too_many_arguments)]
fn recover_and_feed(
    decoder: &mut FountainDecoder,
    emit: &[u32],
    j0: usize,
    header_len: usize,
    stripe_data: &[Option<Vec<u8>>; 8],
    include: &[bool; 8],
    cum: &[usize; 9],
) -> (usize, usize) {
    let total = cum[8];
    let mut stream = vec![0u8; total];
    let mut known = vec![false; total];
    for i in 0..8 {
        if include[i] {
            if let Some(data) = &stripe_data[i] {
                let off = cum[i];
                for (b, &v) in data.iter().enumerate() {
                    if off + b < total {
                        stream[off + b] = v;
                        known[off + b] = true;
                    }
                }
            }
        }
    }
    let (mut recv, mut useful) = (0usize, 0usize);
    for kk in 0..SYMBOLS_PER_FRAME {
        let s = header_len + kk * SYMBOL_SIZE;
        let e = s + SYMBOL_SIZE;
        if e > total {
            break;
        }
        if known[s..e].iter().all(|&x| x) {
            let ei = j0 + kk;
            if ei < emit.len() {
                recv += 1;
                if decoder.push(emit[ei], &stream[s..e]) {
                    useful += 1;
                }
            }
        }
    }
    (recv, useful)
}

/// Разобрать один принятый кадр и скормить его символы фонтану с корректной
/// разметкой ESI и атрибуцией рваного снимка (§6.3). Возврат (получено, новых).
fn process_frame(
    decoder: &mut FountainDecoder,
    emit: &[u32],
    pos: &HashMap<u32, usize>,
    parsed: &ParsedFrame,
    h: &FrameHeader,
    c_start: u8,
    c_end: u8,
    cum: &[usize; 9],
) -> (usize, usize) {
    // индекс кадра N из заголовка (emit[seq·8] уникален -> позиция = seq·8)
    let j0 = match pos.get(&h.esi) {
        Some(&j) => j,
        None => return (0, 0),
    };
    let n_idx = j0 / SYMBOLS_PER_FRAME;

    // данные по страйпам, разложенные по индексу страйпа
    let mut stripe_data: [Option<Vec<u8>>; 8] =
        [None, None, None, None, None, None, None, None];
    for (off, bytes) in &parsed.salvaged {
        for i in 0..8 {
            if cum[i] == *off {
                stripe_data[i] = Some(bytes.clone());
                break;
            }
        }
    }
    let valid = parsed.stripes_ok;
    let hlen_n = if h.has_transfer_info() {
        l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN
    } else {
        l3::FRAME_HEADER_LEN
    };

    let exp_clean = (n_idx & 0xFF) as u8;
    let exp_torn = ((n_idx + 1) & 0xFF) as u8;

    let (mut recv, mut useful) = (0usize, 0usize);
    if c_start == exp_clean && c_end == exp_clean {
        // не рвано: все CRC-валидные страйпы принадлежат кадру N.
        let (r, u) = recover_and_feed(decoder, emit, j0, hlen_n, &stripe_data, &valid, cum);
        recv += r;
        useful += u;
    } else if c_start == exp_torn && c_end == exp_torn {
        // рвано (§6.3): верхний прогон -> N, нижний -> N+1.
        let mut top_end = 0usize;
        while top_end < 8 && valid[top_end] {
            top_end += 1;
        }
        let mut inc_top = [false; 8];
        for b in inc_top.iter_mut().take(top_end) {
            *b = true;
        }
        let (r, u) = recover_and_feed(decoder, emit, j0, hlen_n, &stripe_data, &inc_top, cum);
        recv += r;
        useful += u;

        let mut bot_start = 8usize;
        while bot_start > 0 && valid[bot_start - 1] {
            bot_start -= 1;
        }
        if bot_start > top_end {
            let j0_np1 = (n_idx + 1) * SYMBOLS_PER_FRAME;
            if j0_np1 + SYMBOLS_PER_FRAME <= emit.len() {
                let hlen_np1 = if (n_idx + 1) % 8 == 0 {
                    l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN
                } else {
                    l3::FRAME_HEADER_LEN
                };
                let mut inc_bot = [false; 8];
                for b in inc_bot.iter_mut().take(8).skip(bot_start) {
                    *b = true;
                }
                let (r, u) =
                    recover_and_feed(decoder, emit, j0_np1, hlen_np1, &stripe_data, &inc_bot, cum);
                recv += r;
                useful += u;
            }
        }
    } else {
        // неоднозначно (шумный счётчик): безопасно берём только непрерывный
        // верхний прогон как кадр N.
        let mut top_end = 0usize;
        while top_end < 8 && valid[top_end] {
            top_end += 1;
        }
        let mut inc_top = [false; 8];
        for b in inc_top.iter_mut().take(top_end) {
            *b = true;
        }
        let (r, u) = recover_and_feed(decoder, emit, j0, hlen_n, &stripe_data, &inc_top, cum);
        recv += r;
        useful += u;
    }
    (recv, useful)
}

/// Итог передачи в одном условии (σ, p_torn).
#[derive(Clone, Debug)]
struct CondReport {
    sigma: f64,
    p_torn: f64,
    k: u32,
    frames: usize,
    symbols_recv: usize,
    symbols_lost: usize,
    useful: usize,
    epsilon: f64,
    goodput_kbps: f64,
    completed: bool,
    crc_ok: bool,
    elapsed_s: f64,
}

/// Полностековая передача одного условия. Детерминирована по (условие).
fn run_transfer(sigma: f64, p_torn: f64, payload_len: usize, cond_idx: usize) -> CondReport {
    let t0 = Instant::now();
    let p = transfer_profile();
    let bpc = symbol::bits_per_cell(&p); // 3

    let payload = make_payload(payload_len);
    let checksum = crc32c(&payload);
    let enc = FountainEncoder::new(&payload, SYMBOL_SIZE);
    let k = enc.k();
    let session = 0x5053_4930u32 ^ (cond_idx as u32).wrapping_mul(0x9E37_79B9);

    let mut ch = ChannelParams::from_profile(&p);
    ch.px_per_cell = 8.0;
    ch.set_blur(sigma);

    // кап по кадрам ~ 10·K символов (§9.2), +запас на N+1-lookahead.
    let frame_cap = (10 * k as usize).div_ceil(SYMBOLS_PER_FRAME) + 16;
    let emit = emission_order(k, frame_cap + 2);
    let mut pos: HashMap<u32, usize> = HashMap::with_capacity(emit.len());
    for (i, &e) in emit.iter().enumerate() {
        pos.entry(e).or_insert(i);
    }

    let (_caps, cum) = stripe_layout(bpc);

    let mut dec: Option<FountainDecoder> = None;
    let mut have_session = false;
    let mut cur_session = 0u32;
    let mut done: Option<Vec<u8>> = None;

    let mut rng = Rng::new(seed_for(cond_idx + 1, 0));
    let mut frames = 0usize;
    let mut symbols_recv = 0usize;
    let mut useful = 0usize;

    let mut seq = 0u32;
    while (seq as usize) < frame_cap && done.is_none() {
        // --- передатчик: кадр N ---
        let base = seq as usize * SYMBOLS_PER_FRAME;
        let sym_n = build_symbol_bytes(&enc, &emit, base);
        let ti_n = ti_for(seq, payload.len(), k, checksum);
        let hdr_n = FrameHeader::new(session, emit[base], SYMBOLS_PER_FRAME as u8);
        let cells_n = l3::build_frame(&hdr_n, ti_n.as_ref(), &sym_n, bpc);
        let frame_n = render_symbol_counter(&p, &cells_n, (seq & 0xFF) as u8);

        // --- возможный rolling-shutter разрыв (§6.3) ---
        let torn = rng.next_f64() < p_torn;
        let composite = if torn {
            let base2 = (seq as usize + 1) * SYMBOLS_PER_FRAME;
            let sym2 = build_symbol_bytes(&enc, &emit, base2);
            let ti2 = ti_for(seq + 1, payload.len(), k, checksum);
            let hdr2 = FrameHeader::new(session, emit[base2], SYMBOLS_PER_FRAME as u8);
            let cells2 = l3::build_frame(&hdr2, ti2.as_ref(), &sym2, bpc);
            let frame2 = render_symbol_counter(&p, &cells2, ((seq + 1) & 0xFF) as u8);

            let quiet = p.quiet_zone_cells() as usize;
            let cell = p.cell_size_px as usize;
            let pr = rng.next_u32_below(symbol::PAYLOAD_ROWS as u32) as usize;
            let ir = pr + 1; // payload-строки идут с interior-строки 1 (после реф-строки)
            let top_px = (quiet + symbol::RING + ir) * cell;
            let off = 1 + rng.next_u32_below((cell - 1).max(1) as u32) as usize;
            compose_torn(&frame_n, &frame2, top_px + off)
        } else {
            frame_n
        };
        frames += 1;

        // --- канал + РЕАЛЬНАЯ детекция (как pipeline::run_trial_detect) ---
        let (img, _geom) = apply_channel(&composite, &ch, &mut rng);
        let g_plane: Vec<f32> = img.data.iter().map(|px| px[1]).collect();
        if let Ok(d) = detect_symbol(img.w, img.h, &g_plane) {
            let map = frame_map(&p, &d);
            let sample = |x: f64, y: f64| img.sample(x - 0.5, y - 0.5);
            let cells_rx = demod_symbol(&p, &map, &sample);
            let parsed = l3::parse_frame(&cells_rx, bpc);
            let (c_start, c_end) = read_counters(&p, &map, &sample);

            if let Some(h) = parsed.header {
                if !have_session {
                    have_session = true;
                    cur_session = h.session_id;
                }
                if h.session_id == cur_session {
                    if dec.is_none() {
                        if let Some(ti) = parsed.transfer_info {
                            dec = Some(FountainDecoder::new(
                                ti.k,
                                ti.symbol_size as usize,
                                ti.transfer_length as usize,
                                ti.checksum,
                            ));
                        }
                    }
                    if let Some(decoder) = dec.as_mut() {
                        let (r, u) =
                            process_frame(decoder, &emit, &pos, &parsed, &h, c_start, c_end, &cum);
                        symbols_recv += r;
                        useful += u;
                        if decoder.n_seen() >= decoder.k() {
                            done = decoder.try_finish();
                        }
                    }
                }
            }
        }
        seq += 1;
    }

    let completed = done.is_some();
    let crc_ok = done.as_deref() == Some(&payload[..]);
    let symbols_tx = frames * SYMBOLS_PER_FRAME;
    let symbols_lost = symbols_tx.saturating_sub(symbols_recv);
    let epsilon = if k > 0 {
        useful as f64 / k as f64 - 1.0
    } else {
        0.0
    };
    let goodput_kbps = (payload.len() as f64 * 8.0) / (frames.max(1) as f64 * 0.1) / 1000.0;

    CondReport {
        sigma,
        p_torn,
        k,
        frames,
        symbols_recv,
        symbols_lost,
        useful,
        epsilon,
        goodput_kbps,
        completed,
        crc_ok,
        elapsed_s: t0.elapsed().as_secs_f64(),
    }
}

/// E2E-передача с измерением затрат кадров; печать таблиц.
pub fn cmd_transfer() {
    let p = transfer_profile();
    let bpc = symbol::bits_per_cell(&p);
    println!("# psicode-sim transfer — полностековая передача (§6.1/§9.2, XOR-фонтан EXPERIMENTAL)");
    println!(
        "конфиг: luma {} + {} = {} бит/клетку (BENCHMARKS §6, σ=1-оптимум), cell {} px, px/клетку 8;\n\
         канал = телеметрия-правда §7.4 (кросстолк 6/8%, шум σ2 град, matched γ), σ_blur свипается.",
        p.luma_bits,
        chroma_name(p.chroma_mode),
        bpc,
        p.cell_size_px,
    );
    println!(
        "нагрузка {} байт; symbol_size {} байт, {} символов/кадр = {} байт; frame hold 0.1 с (10 fps).",
        PAYLOAD_BYTES,
        SYMBOL_SIZE,
        SYMBOLS_PER_FRAME,
        SYMBOLS_PER_FRAME * SYMBOL_SIZE
    );
    println!("тракт: bytes -> фонтан -> L3-кадр -> render -> канал -> ДЕТЕКЦИЯ ЗЧ -> demod -> salvage -> фонтан -> CRC-32C\n");

    let conds = [(0.5f64, 0.0f64), (1.0, 0.0), (0.5, 0.2), (1.0, 0.2)];
    let mut reports = Vec::new();
    for (i, &(sigma, p_torn)) in conds.iter().enumerate() {
        reports.push(run_transfer(sigma, p_torn, PAYLOAD_BYTES, i));
    }

    println!("| σ | p_torn | K | кадров | символов принято/потеряно | ε | goodput | CRC-32C |");
    println!("|---|---|---|---|---|---|---|---|");
    for r in &reports {
        let verdict = if r.crc_ok {
            "OK"
        } else if r.completed {
            "FAIL(другой объект)"
        } else {
            "не завершено"
        };
        println!(
            "| {:.1} | {:.0}% | {} | {} | {}/{} | {:+.3} | {:.1} kbit/s | {} |",
            r.sigma,
            r.p_torn * 100.0,
            r.k,
            r.frames,
            r.symbols_recv,
            r.symbols_lost,
            r.epsilon,
            r.goodput_kbps,
            verdict,
        );
    }

    let total_s: f64 = reports.iter().map(|r| r.elapsed_s).sum();
    println!("\nполезных символов = useful (новая инфо); ε = useful/K − 1.");
    for r in &reports {
        println!(
            "- σ={:.1} p_torn={:.0}%: useful={} (K={}), {} за {:.2} c",
            r.sigma,
            r.p_torn * 100.0,
            r.useful,
            r.k,
            if r.crc_ok { "CRC OK" } else if r.completed { "CRC FAIL" } else { "неполно" },
            r.elapsed_s
        );
    }
    println!("\nвсего {total_s:.2} c");
}

/// Короткое имя хромо-режима для шапки.
fn chroma_name(m: ChromaMode) -> &'static str {
    match m {
        ChromaMode::Mono => "Mono",
        ChromaMode::Chroma1 => "Chroma1",
        ChromaMode::Chroma2 => "Chroma2",
        ChromaMode::Chroma3 => "Chroma3",
        ChromaMode::GreenOnly => "GreenOnly",
        ChromaMode::ConstLuma1 => "ConstLuma1",
        ChromaMode::ConstLuma2 => "ConstLuma2",
        ChromaMode::ConstLuma3 => "ConstLuma3",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Малая передача (~2 КБ, σ=0.5, без разрыва) обязана завершиться с CRC OK —
    /// честный полностековый assert (детекция ЗЧ + demod + salvage + фонтан).
    #[test]
    fn e2e_small_transfer_completes() {
        let r = run_transfer(0.5, 0.0, 2000, 0);
        assert!(r.completed, "малая передача не завершилась (кадров {})", r.frames);
        assert!(r.crc_ok, "CRC-32C не совпал");
        assert!(r.epsilon <= 0.5, "overhead ε={:.3} неожиданно велик", r.epsilon);
    }

    /// σ достаточно велика, чтобы ничего не декодировалось: декодер НЕ завершается
    /// ложно (нет паники, нет ложного CRC-прохода).
    #[test]
    fn high_sigma_does_not_falsely_complete() {
        let r = run_transfer(6.0, 0.0, 2000, 1);
        assert!(!r.completed, "при σ=6 не должно быть завершения");
        assert!(!r.crc_ok);
    }

    /// Детерминизм: одинаковые аргументы -> одинаковое число кадров и вердикт.
    #[test]
    fn deterministic_repeatability() {
        let a = run_transfer(0.5, 0.0, 2000, 3);
        let b = run_transfer(0.5, 0.0, 2000, 3);
        assert_eq!(a.frames, b.frames, "число кадров недетерминировано");
        assert_eq!(a.crc_ok, b.crc_ok);
        assert_eq!(a.useful, b.useful);
        assert_eq!(a.symbols_recv, b.symbols_recv);
    }
}
