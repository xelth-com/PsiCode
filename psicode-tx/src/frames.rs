//! Контент кадров передатчика (§4/§6): ЧИСТЫЕ функции, возвращающие RGB-буферы.
//! Ни окна, ни таймеров, ни ввода — всё здесь юнит-тестируемо без GUI. Оболочка
//! winit/softbuffer (main.rs) только блитит готовые [`RgbFrame`] в окно.
//!
//! Три источника кадров:
//! * [`calibration_frame`] — статический паттерн §4 (клин/лесенки/белое поле/ЗЧ).
//! * [`calibration_counter_frame`] — фаза 2 §4.5: паттерн + анимированный
//!   16-битный счётчик кадров, продублированный сверху и снизу окна (измерение
//!   рвущихся кадров rolling-shutter).
//! * [`Streamer`] — поток кадров L3 файла (§6.1/§6.2), эталонная раскладка ниже.
//! * [`single_frame`] — один статический кадр с детерминированной псевдослучайной
//!   нагрузкой (навести камеру, проверить детекцию ЗЧ на живом экране).
//!
//! # Обобщение symbol_size на любые bits_per_cell (§6.2)
//!
//! psicode-sim/transfer.rs фиксирует symbol_size=140 при 3 бит/клетку. Здесь то
//! же правило обобщено на любой `bpc`: число символов/кадр фиксировано (= 8, что
//! делает индекс кадра N = emission_index/8 тривиально обратимым из заголовка,
//! §6.3), а размер символа берётся так, чтобы 8 символов гарантированно влезали в
//! САМЫЙ тесный кадр — TransferInfo-кадр (§6.2, каждый 8-й несёт лишние 14 байт):
//!
//! ```text
//! min_symbol_capacity = frame_byte_capacity(bpc) − FRAME_HEADER_LEN − TRANSFER_INFO_LEN
//! symbol_size         = floor(min_symbol_capacity / 8)
//! ```
//!
//! Тогда `FRAME_HEADER_LEN + TRANSFER_INFO_LEN + 8·symbol_size ≤ frame_byte_capacity(bpc)`
//! по построению (floor), а для не-TI кадров тем более. Для bpc=3 это даёт 141
//! (transfer.rs округляет вниз до 140 вручную — оба варианта валидны; здесь берём
//! максимальный floor, чтобы не терять ёмкость).

use psicode_core::calibrate;
use psicode_core::fountain::{crc32c, FountainEncoder};
use psicode_core::l3::{self, FrameHeader, TransferInfo};
use psicode_core::symbol::{self, render_symbol_counter};
use psicode_core::CalibProfile;

/// Средне-серый фон/тихая зона (§3.1).
const GRAY: [u8; 3] = [128, 128, 128];
/// Белая клетка счётчика.
const WHITE: [u8; 3] = [255, 255, 255];
/// Чёрная клетка счётчика.
const BLACK: [u8; 3] = [0, 0, 0];

/// Число encoding-символов в кадре (фиксировано, см. модульдок и §6.3).
pub const SYMBOLS_PER_FRAME: usize = 8;
/// Ширина анимированного счётчика §4.5 в клетках (16 бит номера кадра).
pub const CALIB_COUNTER_BITS: usize = 16;

/// Прямоугольный RGB-буфер построчно (row-major). Универсальный тип возврата всех
/// построителей кадров; оболочка окна блитит его целочисленным зумом по центру.
#[derive(Clone, Debug)]
pub struct RgbFrame {
    /// Ширина в пикселях.
    pub w: usize,
    /// Высота в пикселях.
    pub h: usize,
    /// Пиксели построчно, [R, G, B] drive 0..255, длина = w·h.
    pub px: Vec<[u8; 3]>,
}

impl RgbFrame {
    /// Из [`symbol::Frame`] (квадрат size_px×size_px).
    fn from_symbol(f: symbol::Frame) -> Self {
        RgbFrame {
            w: f.size_px,
            h: f.size_px,
            px: f.rgb,
        }
    }
}

/// Размер encoding-символа (байт) для данного `bits_per_cell` (§6.2). См. модульдок.
pub fn symbol_size_for(bits_per_cell: u32) -> usize {
    let cap = l3::frame_byte_capacity(bits_per_cell);
    let min_sym_cap = cap
        .saturating_sub(l3::FRAME_HEADER_LEN)
        .saturating_sub(l3::TRANSFER_INFO_LEN);
    (min_sym_cap / SYMBOLS_PER_FRAME).max(1)
}

/// σ=1-оптимальный эталонный профиль (§6/§7.4): luma 2 + Chroma1 = 3 бит/клетку,
/// cell 16, hold 6, fec 2. Совпадает с psicode-sim `transfer_profile()` —
/// телеметрия при референсных значениях §7.4.
pub fn reference_profile() -> CalibProfile {
    use psicode_core::ChromaMode;
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

// ---------------------------------------------------------------------------
// Калибровка (§4)
// ---------------------------------------------------------------------------

/// Статический калибровочный паттерн §4, квадрат `61·cell` × `61·cell` px.
/// `cell` — целочисленный размер клетки (номинал 12). Тихой зоны у паттерна нет.
pub fn calibration_frame(cell: usize) -> RgbFrame {
    let cell = cell.max(1);
    let size = calibrate::GRID * cell; // 61·cell -> внутри cell = size/61 ровно
    let px = calibrate::render_calibration_pattern(size);
    RgbFrame { w: size, h: size, px }
}

/// Полоса с 16-битным номером кадра (§4.5): ширина `width`, высота `2·cell`,
/// средне-серая, 16 чёрно-белых клеток (каждая `cell` px шириной) по центру,
/// СТАРШИМ битом влево (MSB-first). white = бит 1. Обе копии (верх/низ) строятся
/// из одного значения, поэтому идентичны.
pub fn counter_band(value: u16, width: usize, cell: usize) -> RgbFrame {
    let cell = cell.max(1);
    let h = 2 * cell;
    let mut px = vec![GRAY; width * h];
    let strip_w = CALIB_COUNTER_BITS * cell;
    let x0 = width.saturating_sub(strip_w) / 2; // центрируем по ширине
    for b in 0..CALIB_COUNTER_BITS {
        let bit = (value >> (CALIB_COUNTER_BITS - 1 - b)) & 1 == 1;
        let color = if bit { WHITE } else { BLACK };
        let cx0 = x0 + b * cell;
        for y in 0..h {
            for x in 0..cell {
                let px_x = cx0 + x;
                if px_x < width {
                    px[y * width + px_x] = color;
                }
            }
        }
    }
    RgbFrame { w: width, h, px }
}

/// Фаза 2 калибровки (§4.5): статический паттерн в центре плюс АНИМИРОВАННЫЙ
/// 16-битный счётчик кадров, продублированный полосой сверху и снизу. `frame_no`
/// растёт с каждым обновлением экрана; приёмник ловит 2–3 с и по доле снимков с
/// двумя разными номерами (rolling-shutter) оценивает `torn_frames_q` (§4.5).
pub fn calibration_counter_frame(cell: usize, frame_no: u64) -> RgbFrame {
    let cell = cell.max(1);
    let pattern = calibration_frame(cell);
    let w = pattern.w;
    let value = (frame_no & 0xFFFF) as u16;
    let band = counter_band(value, w, cell); // одна и та же полоса сверху и снизу
    let bh = band.h;
    let h = pattern.h + 2 * bh;
    let mut px = vec![GRAY; w * h];
    blit_into(&mut px, w, &band, 0); // верхняя копия
    blit_into(&mut px, w, &pattern, bh); // паттерн
    blit_into(&mut px, w, &band, bh + pattern.h); // нижняя копия (равна верхней)
    RgbFrame { w, h, px }
}

/// Копирует `src` (той же ширины `dst_w`) в `dst` начиная со строки `y_off`.
fn blit_into(dst: &mut [[u8; 3]], dst_w: usize, src: &RgbFrame, y_off: usize) {
    debug_assert_eq!(src.w, dst_w, "blit_into: ширины должны совпадать");
    let start = y_off * dst_w;
    let len = src.px.len();
    dst[start..start + len].copy_from_slice(&src.px);
}

// ---------------------------------------------------------------------------
// Одиночный кадр (single): детекция на живом экране
// ---------------------------------------------------------------------------

/// Один кадр с ДЕТЕРМИНИРОВАННОЙ псевдослучайной нагрузкой клеток (сид константа,
/// splitmix64 — тот же, что psicode-sim `make_payload`). `counter` — младшие 8 бит
/// номера кадра в строку счётчика (§3.3). `cell_override` подменяет размер клетки
/// профиля ТОЛЬКО для рендера (см. примечание к `Streamer::render`).
pub fn single_frame(profile: &CalibProfile, counter: u8, cell_override: Option<usize>) -> RgbFrame {
    let bpc = symbol::bits_per_cell(profile);
    let mask: u16 = if bpc >= 16 { u16::MAX } else { (1u16 << bpc) - 1 };
    let n = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
    let mut state = 0x0D15_EA5E_5EED_1234u64;
    let cells: Vec<u8> = (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (((z ^ (z >> 31)) >> 24) as u16 & mask) as u8
        })
        .collect();
    let mut disp = *profile;
    if let Some(c) = cell_override {
        disp.cell_size_px = c as u8;
    }
    RgbFrame::from_symbol(render_symbol_counter(&disp, &cells, counter))
}

// ---------------------------------------------------------------------------
// Стриминг файла (§6.1/§6.2)
// ---------------------------------------------------------------------------

/// Поток кадров L3 для передачи файла (§6.1/§6.2). Тракт как в
/// psicode-sim/transfer.rs, но symbol_size обобщён на любой `bpc` (см. модульдок)
/// и поток ESI бесконечен (§6.1: «цикл символов до остановки»).
pub struct Streamer {
    profile: CalibProfile,
    enc: FountainEncoder,
    session_id: u32,
    bpc: u32,
    symbol_size: usize,
    k: u32,
    checksum: u32,
    transfer_length: usize,
    /// Один систематический проход: source 0..K−1 с вплетённым repair (§6.1).
    pass: Vec<u32>,
    /// Первый repair-ESI ПОСЛЕ систематического прохода.
    next_repair: u32,
    frames_per_pass: usize,
}

impl Streamer {
    /// Готовит поток по байтам файла, профилю и `session_id`.
    pub fn new(file_bytes: &[u8], profile: CalibProfile, session_id: u32) -> Self {
        let bpc = symbol::bits_per_cell(&profile);
        let symbol_size = symbol_size_for(bpc);
        let enc = FountainEncoder::new(file_bytes, symbol_size);
        let k = enc.k();
        let checksum = crc32c(file_bytes);
        let (pass, next_repair) = systematic_pass(k, profile.repair_interval_source_symbols());
        let frames_per_pass = pass.len().div_ceil(SYMBOLS_PER_FRAME).max(1);
        Streamer {
            profile,
            enc,
            session_id,
            bpc,
            symbol_size,
            k,
            checksum,
            transfer_length: file_bytes.len(),
            pass,
            next_repair,
            frames_per_pass,
        }
    }

    /// Число source-символов K.
    pub fn k(&self) -> u32 {
        self.k
    }
    /// Размер encoding-символа, байт.
    pub fn symbol_size(&self) -> usize {
        self.symbol_size
    }
    /// Бит на клетку (§5.2).
    pub fn bpc(&self) -> u32 {
        self.bpc
    }
    /// Кадров в одном систематическом проходе (все K source + вплетённый repair).
    pub fn frames_per_pass(&self) -> usize {
        self.frames_per_pass
    }

    /// ESI `j`-го элемента бесконечного потока (§6.1). Внутри систематического
    /// прохода — из таблицы `pass`; после — сплошной repair с ESI ≥ `next_repair`.
    fn emission_esi(&self, j: usize) -> u32 {
        if j < self.pass.len() {
            self.pass[j]
        } else {
            self.next_repair + (j - self.pass.len()) as u32
        }
    }

    /// Клеточные символы кадра `seq` (§6.2): 8 символов потока + FrameHeader,
    /// TransferInfo в каждом 8-м кадре. Возвращает `PAYLOAD_COLS·PAYLOAD_ROWS`
    /// клеток (значение < 2^bpc).
    pub fn frame_cells(&self, seq: u64) -> Vec<u8> {
        let base = seq as usize * SYMBOLS_PER_FRAME;
        let mut sym_bytes = Vec::with_capacity(SYMBOLS_PER_FRAME * self.symbol_size);
        for i in 0..SYMBOLS_PER_FRAME {
            let esi = self.emission_esi(base + i);
            sym_bytes.extend_from_slice(&self.enc.symbol(esi));
        }
        let ti = if seq % 8 == 0 {
            Some(TransferInfo {
                transfer_length: self.transfer_length as u64,
                symbol_size: self.symbol_size as u16,
                k: self.k,
                checksum: self.checksum,
            })
        } else {
            None
        };
        let hdr = FrameHeader::new(
            self.session_id,
            self.emission_esi(base),
            SYMBOLS_PER_FRAME as u8,
        );
        l3::build_frame(&hdr, ti.as_ref(), &sym_bytes, self.bpc)
    }

    /// Рендер кадра `seq` в RGB. Строка счётчика (§3.3) несёт `(seq & 0xFF)`.
    ///
    /// `cell_override` подменяет `cell_size_px` профиля ТОЛЬКО для рендера в
    /// пиксели: содержимое кадра (биты клеток, bpc) от размера клетки не зависит,
    /// меняются лишь размеры буфера. Семантика: `cell_size_px` из профиля — это то,
    /// что ПРИЁМНИК считает размером клетки; для v0-показа мы рисуем 1 display-px на
    /// Frame-px (умноженный на целочисленный зум окна), поэтому px на экране ≠
    /// прескрипция `cell_size_px`.
    pub fn render(&self, seq: u64, cell_override: Option<usize>) -> RgbFrame {
        let cells = self.frame_cells(seq);
        let mut disp = self.profile;
        if let Some(c) = cell_override {
            disp.cell_size_px = c as u8;
        }
        let counter = (seq & 0xFF) as u8;
        RgbFrame::from_symbol(render_symbol_counter(&disp, &cells, counter))
    }
}

/// Один систематический проход потока (§6.1): вернуть `(pass, next_repair_esi)`.
/// `repair_every = None` (fec_overhead=0) — source ×1 без вплетения; `Some(r)` —
/// вплетать один repair после каждых `r` source. `next_repair` — первый repair-ESI
/// для продолжения потока после прохода.
fn systematic_pass(k: u32, repair_every: Option<u32>) -> (Vec<u32>, u32) {
    let mut pass = Vec::new();
    let mut rep = k; // repair-ESI нумеруются с K
    match repair_every {
        None => {
            for s in 0..k {
                pass.push(s);
            }
        }
        Some(r) => {
            let mut since = 0u32;
            let mut src = 0u32;
            while src < k {
                pass.push(src);
                src += 1;
                since += 1;
                if since == r {
                    pass.push(rep);
                    rep += 1;
                    since = 0;
                }
            }
        }
    }
    (pass, rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Детерминированный псевдослучайный файл (splitmix64) заданной длины.
    fn make_file(len: usize) -> Vec<u8> {
        let mut state = 0xF00D_BABE_1234_5678u64;
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

    /// Сэмплер поверх RgbFrame: map=тождество, sample=drive/255 (монотонно —
    /// достаточно для порогового `read_counters`, которому важен лишь порядок).
    fn read_frame_counters(p: &CalibProfile, f: &RgbFrame) -> (u8, u8) {
        let w = f.w;
        let px = f.px.clone();
        let map = |u: f64, v: f64| (u, v);
        let sample = move |x: f64, y: f64| -> [f32; 3] {
            let xi = (x as isize).clamp(0, w as isize - 1) as usize;
            let yi = (y as isize).clamp(0, (f.h as isize) - 1) as usize;
            let d = px[yi * w + xi];
            [d[0] as f32 / 255.0, d[1] as f32 / 255.0, d[2] as f32 / 255.0]
        };
        symbol::read_counters(p, &map, &sample)
    }

    /// Тракт стриминга детерминирован: одинаковый файл/профиль/сессия -> те же
    /// клетки первых N кадров; кадр 0 парсится обратно (§6.2) со всеми валидными
    /// страйпами, совпадающим заголовком и присутствующим TransferInfo; счётчик
    /// кадра растёт mod 256.
    #[test]
    fn stream_frames_reproducible_and_parseable() {
        let file = make_file(4096);
        let prof = reference_profile();
        let sid = 0xABCD_1234u32;
        let a = Streamer::new(&file, prof, sid);
        let b = Streamer::new(&file, prof, sid);

        // воспроизводимость клеток первых кадров
        for seq in 0..6u64 {
            assert_eq!(a.frame_cells(seq), b.frame_cells(seq), "кадр {seq} недетерминирован");
        }

        // кадр 0 парсится обратно (§6.2)
        let cells0 = a.frame_cells(0);
        let parsed = l3::parse_frame(&cells0, a.bpc());
        assert!(parsed.stripes_ok.iter().all(|&ok| ok), "не все страйпы CRC-валидны");
        let h = parsed.header.expect("заголовок кадра 0");
        assert_eq!(h.session_id, sid, "session_id");
        assert_eq!(h.count, SYMBOLS_PER_FRAME as u8, "count");
        assert_eq!(h.esi, a.emission_esi(0), "esi первого символа");
        let ti = parsed.transfer_info.expect("TransferInfo в кадре 0 (0 % 8 == 0)");
        assert_eq!(ti.k, a.k(), "K");
        assert_eq!(ti.symbol_size as usize, a.symbol_size(), "symbol_size");
        assert_eq!(ti.transfer_length as usize, file.len(), "transfer_length");
        assert_eq!(ti.checksum, crc32c(&file), "checksum");

        // счётчик кадра = seq & 0xFF (обе копии), растёт mod 256
        for seq in [0u64, 1, 7, 8, 255, 256, 257] {
            let f = a.render(seq, None);
            let (c0, c1) = read_frame_counters(&prof, &f);
            let want = (seq & 0xFF) as u8;
            assert_eq!(c0, want, "start-копия счётчика, seq={seq}");
            assert_eq!(c1, want, "end-копия счётчика, seq={seq}");
        }
    }

    /// symbol_size math (§6.2): для bpc ∈ 1..=7 восемь символов + заголовок [+ TI]
    /// влезают в ёмкость КАЖДОГО кадра (и TI-кадра, и обычного).
    #[test]
    fn symbol_size_fits_every_frame() {
        for bpc in 1..=7u32 {
            let ss = symbol_size_for(bpc);
            assert!(ss >= 1, "bpc {bpc}: symbol_size 0");
            let cap = l3::frame_byte_capacity(bpc);
            let ti_frame = l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN + SYMBOLS_PER_FRAME * ss;
            let plain_frame = l3::FRAME_HEADER_LEN + SYMBOLS_PER_FRAME * ss;
            assert!(ti_frame <= cap, "bpc {bpc}: TI-кадр {ti_frame} > ёмкость {cap}");
            assert!(plain_frame <= cap, "bpc {bpc}: обычный кадр {plain_frame} > ёмкость {cap}");
        }
    }

    /// Строит поток и убеждается, что кадры реального bpc профиля тоже парсятся
    /// (bpc=3 у эталона) — покрытие боевого пути build->parse на длинном файле.
    #[test]
    fn many_frames_parse_ok() {
        let file = make_file(20_000);
        let s = Streamer::new(&file, reference_profile(), 7);
        for seq in 0..(s.frames_per_pass() as u64).min(40) {
            let parsed = l3::parse_frame(&s.frame_cells(seq), s.bpc());
            assert!(parsed.stripes_ok.iter().all(|&ok| ok), "кадр {seq}: страйп бит");
            let h = parsed.header.expect("заголовок");
            assert_eq!(h.esi, s.emission_esi(seq as usize * SYMBOLS_PER_FRAME));
        }
    }

    /// Анимированный счётчик §4.5: рендерится, меняется с индексом кадра, обе
    /// копии (верхняя и нижняя полосы) идентичны.
    #[test]
    fn calibration_counter_strip_changes_and_duplicates() {
        let cell = 4usize;
        let a = calibration_counter_frame(cell, 5);
        let b = calibration_counter_frame(cell, 6);
        // меняется с индексом кадра
        assert_ne!(a.px, b.px, "счётчик обязан меняться с индексом кадра");
        // геометрия: паттерн 61·cell + две полосы по 2·cell
        let band_h = 2 * cell;
        assert_eq!(a.w, calibrate::GRID * cell);
        assert_eq!(a.h, calibrate::GRID * cell + 2 * band_h);
        // обе копии равны
        let w = a.w;
        let top = &a.px[0..band_h * w];
        let bot = &a.px[(a.h - band_h) * w..];
        assert_eq!(top, bot, "верхняя и нижняя копии счётчика должны совпадать");
        // полоса не пуста (есть и белые, и чёрные клетки при value=5)
        assert!(top.iter().any(|&c| c == WHITE), "нет белых клеток");
        assert!(top.iter().any(|&c| c == BLACK), "нет чёрных клеток");
    }

    /// Дымовой тест без окна: размеры буфера верны для пары размеров клетки, а
    /// содержимое не вырождено в сплошной серый.
    #[test]
    fn frame_dimensions_and_nonblank() {
        let prof = reference_profile();
        let quiet = prof.quiet_zone_cells() as usize;
        let file = make_file(2048);
        let s = Streamer::new(&file, prof, 1);
        for cell in [8usize, 16] {
            let f = s.render(0, Some(cell));
            let expect = (symbol::GRID + 2 * quiet) * cell;
            assert_eq!(f.w, expect, "ширина при cell={cell}");
            assert_eq!(f.h, expect, "высота при cell={cell}");
            assert_eq!(f.px.len(), expect * expect, "длина буфера при cell={cell}");
            assert!(f.px.iter().any(|&c| c != GRAY), "кадр вырожден в серый при cell={cell}");
        }
        // калибровочный паттерн: квадрат 61·cell
        for cell in [10usize, 12] {
            let f = calibration_frame(cell);
            assert_eq!(f.w, calibrate::GRID * cell);
            assert_eq!(f.h, calibrate::GRID * cell);
            assert!(f.px.iter().any(|&c| c != GRAY), "паттерн вырожден в серый при cell={cell}");
        }
        // single: тоже верного размера и не пуст
        let f = single_frame(&prof, 42, Some(12));
        assert_eq!(f.w, (symbol::GRID + 2 * quiet) * 12);
        assert!(f.px.iter().any(|&c| c != GRAY));
    }

    /// fec_overhead управляет вплетением repair (§6.1): fec=0 -> проход = ровно K
    /// source; fec=2 -> repair каждые 4 source.
    #[test]
    fn emission_interleave_respects_fec() {
        // fec=2 (эталон): repair каждые 4 source -> проход длиннее K
        let file = make_file(8000);
        let s = Streamer::new(&file, reference_profile(), 3);
        let k = s.k() as usize;
        // первые элементы: 0,1,2,3, затем repair(=K), 4,5,6,7, repair(K+1)...
        assert_eq!(s.emission_esi(0), 0);
        assert_eq!(s.emission_esi(4), k as u32, "5-й элемент — первый repair");
        assert_eq!(s.emission_esi(5), 4);

        // fec=0: проход = ровно source 0..K−1, дальше сплошной repair
        let mut p0 = reference_profile();
        p0.fec_overhead = 0;
        let s0 = Streamer::new(&file, p0, 3);
        let k0 = s0.k() as usize;
        for j in 0..k0 {
            assert_eq!(s0.emission_esi(j), j as u32, "fec=0: source подряд");
        }
        assert_eq!(s0.emission_esi(k0), k0 as u32, "fec=0: после прохода — repair");
    }
}
