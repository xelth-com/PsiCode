//! L3-фрейминг сквозь канал: модель rolling-shutter разрыва кадра (§6.3),
//! сквозная попытка «поток байт -> кадры N/N+1 -> tear -> канал -> parse»,
//! оценка FER (§2/§6.3) и goodput частичного декодирования (§4/§6.3).
//!
//! # Модель разрыва (§6.3, §4 SPEC калибровки)
//!
//! Rolling-shutter снимок на смене кадра = строки [0, t) отрендеренного кадра N
//! и строки [t, H) кадра N+1, склеенные ДО канала (физически дисплей сменил
//! содержимое посреди развёртки, поэтому warp/blur/шум действуют на композит).
//! Оба кадра одной геометрии. `t` разыгрывается так, чтобы всегда попадать
//! ВНУТРЬ клеточной строки (суб-клеточный сдвиг) — тогда клеточная строка на шве
//! гарантированно «смешана», и её страйп проваливает CRC, помечая границу.
//!
//! # Правило атрибуции v0 (документируется — под §6.3)
//!
//! 1. `parse_frame` даёт per-stripe CRC. Верхний непрерывный прогон валидных
//!    страйпов от страйпа 0 несёт кадр N (опознаётся заголовком страйпа 0).
//! 2. Строка счётчика (§3.3) читается приёмником; при разрыве выше неё она
//!    принадлежит кадру N+1, поэтому её значение = counter_N+1. Разрыв
//!    детектируется как `counter_row != (seq_top & 0xFF)`, где `seq_top`
//!    восстанавливается из заголовка (esi/count — детерминированный порядок §6.1).
//! 3. Нижний непрерывный прогон валидных страйпов до страйпа 7 несёт кадр N+1
//!    (его ESI предсказуем из заголовка N). Страйп(ы) на шве, провалившие CRC,
//!    теряются.
//! 4. Partial-decode OFF: рваный снимок отбрасывается целиком (0 полезных байт).
//!    ON: салвадж верхнего (N) и нижнего (N+1) прогонов.
//! 5. Fallback (§6.3): если заголовок недоступен (страйп 0 попал на шов), кадр
//!    неатрибутируем — 0 байт и в OFF, и в ON (нижний прогон без заголовка N
//!    не имеет предсказанного ESI).

use crate::channel::ChannelParams;
use crate::rng::{seed_for, Rng};
use psicode_core::l3;
use psicode_core::symbol::{self, Frame};
use psicode_core::CalibProfile;

/// Склейка рваного снимка (§6.3): строки [0, t) из `a` (кадр N), [t, H) из `b`
/// (кадр N+1). Оба кадра обязаны иметь одинаковую сторону.
pub fn compose_torn(a: &Frame, b: &Frame, t: usize) -> Frame {
    assert_eq!(a.size_px, b.size_px, "разрыв: кадры разной геометрии");
    let n = a.size_px;
    let tt = t.min(n);
    let mut rgb = a.rgb.clone();
    for y in tt..n {
        let row = y * n;
        rgb[row..row + n].copy_from_slice(&b.rgb[row..row + n]);
    }
    Frame {
        size_px: n,
        quiet_cells: a.quiet_cells,
        rgb,
    }
}

/// Детерминированные байты символов кадра с номером `seq` (различны по seq).
fn frame_symbol_bytes(seq: u32, len: usize) -> Vec<u8> {
    let mut s = 0x9E37_79B9u32 ^ seq.wrapping_mul(0x85EB_CA6B).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect()
}

/// Байт-ёмкости 8 страйпов при данном bits_per_cell (§6.2).
fn stripe_caps(bpc: u32) -> [usize; 8] {
    let mut caps = [0usize; 8];
    for (c, &rows) in caps.iter_mut().zip(l3::STRIPE_ROWS.iter()) {
        *c = (rows * l3::PAYLOAD_COLS * bpc as usize - 16) / 8;
    }
    caps
}

/// Итог сквозной кадровой попытки. Поля `torn`/`torn_detected`/`stripes_ok` —
/// диагностика (сверяются тестами; в бинаре развёрток используются off/on/lost).
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct FramedTrial {
    /// Был ли снимок рваным (ground truth).
    pub torn: bool,
    /// Детектировал ли приёмник разрыв (счётчик != заголовок).
    pub torn_detected: bool,
    /// Per-stripe валидность CRC после канала.
    pub stripes_ok: [bool; 8],
    /// Полезные байты символов при partial-decode OFF.
    pub useful_off: usize,
    /// Полезные байты символов при partial-decode ON.
    pub useful_on: usize,
    /// FER (§6.3): кадр потерян — страйп-заголовок мёртв ИЛИ мертвы все страйпы.
    pub frame_lost: bool,
}

/// Одна сквозная кадровая попытка (§6.2–6.3). `p` задаёт геометрию/алфавит
/// (bits_per_cell), `ch` — канал, `torn_prob` — вероятность рваного снимка.
/// demod — GENIE-геометрия (детекция §3.2 улучшается параллельным worker'ом,
/// здесь не вызывается — см. отчёт). Сид из (point, trial) — воспроизводимо.
pub fn run_framed_trial(
    p: &CalibProfile,
    ch: &ChannelParams,
    torn_prob: f64,
    point: usize,
    trial: usize,
) -> FramedTrial {
    let bpc = symbol::bits_per_cell(p);
    let caps = stripe_caps(bpc);
    let total_cap: usize = caps.iter().sum();

    let mut rng = Rng::new(seed_for(point, trial));

    // номер кадра N и его сосед; ESI в детерминированном порядке §6.1.
    let seq_n = trial as u32;
    let session = 0x5153_4930u32; // 'QSI0'
    let count: u8 = 9; // символов на кадр (декоративно для заголовка)
    let sym_size: u16 = 256;
    let esi_n = seq_n.wrapping_mul(count as u32) & 0x00FF_FFFF;
    let esi_np1 = (seq_n + 1).wrapping_mul(count as u32) & 0x00FF_FFFF;

    // TransferInfo — в каждом 8-м кадре (§6.2).
    let ti_n = if seq_n % 8 == 0 {
        Some(l3::TransferInfo {
            transfer_length: 1_000_000,
            symbol_size: sym_size,
            k: 4000,
            checksum: 0xC0FF_EE00,
        })
    } else {
        None
    };
    let header_len_n = l3::FRAME_HEADER_LEN + if ti_n.is_some() { l3::TRANSFER_INFO_LEN } else { 0 };

    // байты символов: заполняем весь остаток после шапки.
    let sym_n = frame_symbol_bytes(seq_n, total_cap.saturating_sub(header_len_n));
    let sym_np1 = frame_symbol_bytes(seq_n + 1, total_cap.saturating_sub(l3::FRAME_HEADER_LEN));

    let hdr_n = l3::FrameHeader::new(session, esi_n, count);
    let hdr_np1 = l3::FrameHeader::new(session, esi_np1, count);
    let cells_n = l3::build_frame(&hdr_n, ti_n.as_ref(), &sym_n, bpc);
    let cells_np1 = l3::build_frame(&hdr_np1, None, &sym_np1, bpc);

    let frame_n = symbol::render_symbol_counter(p, &cells_n, (seq_n & 0xFF) as u8);
    let frame_np1 = symbol::render_symbol_counter(p, &cells_np1, ((seq_n + 1) & 0xFF) as u8);

    // разрыв: t внутри случайной payload-строки (суб-клеточный сдвиг -> шов
    // гарантированно рвёт ровно один страйп).
    let torn = rng.next_f64() < torn_prob;
    let composite = if torn {
        let quiet = p.quiet_zone_cells() as usize;
        let cell = p.cell_size_px as usize;
        let pr = rng.next_u32_below(symbol::PAYLOAD_ROWS as u32) as usize; // 0..54
        let ir = pr + 1; // payload-строки идут с interior-строки 1 (после реф-строки)
        let top_px = (quiet + symbol::RING + ir) * cell;
        let off = 1 + rng.next_u32_below((cell - 1).max(1) as u32) as usize;
        compose_torn(&frame_n, &frame_np1, top_px + off)
    } else {
        frame_n
    };

    let (img, geom) = crate::pipeline::apply_channel(&composite, ch, &mut rng);
    let map = |u: f64, v: f64| geom.forward(u, v);
    let sample = |x: f64, y: f64| img.sample(x, y);

    let cells_rx = symbol::demod_symbol(p, &map, &sample);
    let parsed = l3::parse_frame(&cells_rx, bpc);
    let (c_start, _c_end) = symbol::read_counters(p, &map, &sample);

    let stripes_ok = parsed.stripes_ok;
    let frame_lost = !stripes_ok[0] || stripes_ok.iter().all(|&ok| !ok);

    // атрибуция §6.3.
    let (torn_detected, useful_off, useful_on) = match parsed.header {
        Some(h) => {
            let seq_top = if h.count > 0 { h.esi / h.count as u32 } else { 0 };
            let td = (c_start as u32) != (seq_top & 0xFF);
            if !td {
                // одиночный кадр N: салвадж всех валидных страйпов как N.
                let bytes = clean_useful_bytes(&stripes_ok, &caps, header_len_n);
                (false, bytes, bytes)
            } else {
                // рваный: верхний прогон (N) + нижний прогон (N+1) для ON; 0 для OFF.
                let on = torn_useful_bytes(&stripes_ok, &caps, header_len_n);
                (true, 0usize, on)
            }
        }
        // заголовок недоступен (страйп 0 на шве/мёртв): кадр неатрибутируем.
        None => (true, 0usize, 0usize),
    };

    FramedTrial {
        torn,
        torn_detected,
        stripes_ok,
        useful_off,
        useful_on,
        frame_lost,
    }
}

/// Полезные байты символов одиночного (не рваного) кадра: сумма по валидным
/// страйпам, из страйпа 0 вычтена служебная шапка (заголовок [+ TransferInfo]).
fn clean_useful_bytes(stripes_ok: &[bool; 8], caps: &[usize; 8], header_len: usize) -> usize {
    let mut sum = 0usize;
    for i in 0..8 {
        if stripes_ok[i] {
            sum += if i == 0 {
                caps[0].saturating_sub(header_len)
            } else {
                caps[i]
            };
        }
    }
    sum
}

/// Полезные байты рваного снимка при partial-decode ON: верхний прогон валидных
/// страйпов (кадр N, из страйпа 0 вычтена шапка) + нижний прогон (кадр N+1,
/// чистые символьные байты). Средние страйпы на шве потеряны.
fn torn_useful_bytes(stripes_ok: &[bool; 8], caps: &[usize; 8], header_len: usize) -> usize {
    // верхний прогон от страйпа 0
    let mut top_end = 0usize; // [0, top_end) валидны
    while top_end < 8 && stripes_ok[top_end] {
        top_end += 1;
    }
    // нижний прогон до страйпа 7
    let mut bot_start = 8usize; // [bot_start, 8) валидны
    while bot_start > 0 && stripes_ok[bot_start - 1] {
        bot_start -= 1;
    }
    // если прогоны сомкнулись (нет мёртвого шва) — не двойной учёт: считаем это
    // атрибуцией только кадра N (консервативно; редкий tear ровно по границе).
    if top_end >= bot_start {
        return clean_useful_bytes(stripes_ok, caps, header_len);
    }
    let mut sum = 0usize;
    for i in 0..top_end {
        sum += if i == 0 { caps[0].saturating_sub(header_len) } else { caps[i] };
    }
    for &c in caps.iter().take(8).skip(bot_start) {
        sum += c; // нижний прогон — чистые символьные байты кадра N+1
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use psicode_core::ChromaMode;

    fn profile(luma: u8, chroma: ChromaMode) -> CalibProfile {
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: 16,
            frame_hold_periods: 6,
            luma_bits: luma,
            chroma_mode: chroma,
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

    /// Чистый канал, без разрыва: все 8 страйпов валидны, кадр не потерян,
    /// OFF == ON == полная символьная ёмкость.
    #[test]
    fn clean_no_tear_full_salvage() {
        let p = profile(4, ChromaMode::Chroma2); // 6 бит/клетку
        let ch = ChannelParams::clean(&p);
        let r = run_framed_trial(&p, &ch, 0.0, 0, 1); // seq=1 -> без TI
        assert!(r.stripes_ok.iter().all(|&ok| ok), "все страйпы валидны");
        assert!(!r.frame_lost);
        assert!(!r.torn_detected);
        assert_eq!(r.useful_off, r.useful_on);
        // полная ёмкость минус заголовок 12
        let caps = stripe_caps(6);
        let total: usize = caps.iter().sum();
        assert_eq!(r.useful_off, total - l3::FRAME_HEADER_LEN);
    }

    /// Чистый канал, гарантированный разрыв: детектируется, ровно один страйп на
    /// шве мёртв, OFF = 0, ON > 0 (салвадж обоих прогонов).
    #[test]
    fn clean_tear_salvages_partial() {
        let p = profile(4, ChromaMode::Chroma2);
        let ch = ChannelParams::clean(&p);
        // ищем trial с разрывом НЕ в страйпе 0 (заголовок цел)
        let mut found = false;
        for t in 0..40 {
            let r = run_framed_trial(&p, &ch, 1.0, 7, t);
            assert!(r.torn, "torn_prob=1 -> всегда рвём");
            if r.stripes_ok[0] {
                assert!(r.torn_detected, "разрыв должен детектироваться");
                let dead = r.stripes_ok.iter().filter(|&&ok| !ok).count();
                assert_eq!(dead, 1, "ровно один страйп на шве мёртв");
                assert_eq!(r.useful_off, 0, "OFF отбрасывает рваный снимок");
                assert!(r.useful_on > 0, "ON салважит прогоны");
                assert!(!r.frame_lost, "заголовок цел -> кадр не потерян");
                found = true;
                break;
            }
        }
        assert!(found, "не нашлось разрыва вне страйпа 0");
    }

    /// Композиция разрыва берёт верх из a, низ из b.
    #[test]
    fn compose_torn_splits_rows() {
        let p = profile(2, ChromaMode::Mono);
        let cells = vec![0u8; symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS];
        let a = symbol::render_symbol_counter(&p, &cells, 10);
        let b = symbol::render_symbol_counter(&p, &cells, 20);
        let n = a.size_px;
        let t = n / 2;
        let c = compose_torn(&a, &b, t);
        // строка выше шва = a, ниже = b
        assert_eq!(c.rgb[(t - 1) * n], a.rgb[(t - 1) * n]);
        assert_eq!(c.rgb[t * n], b.rgb[t * n]);
    }
}
