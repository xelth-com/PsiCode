//! L3-фрейминг (§6.2–6.3): FrameHeader/TransferInfo, CRC-16 по страйпам,
//! упаковка байтового потока в клеточные символы кадра.
//!
//! Модуль `no_std` + `alloc` (только core+alloc, без вещественной математики).
//!
//! # Раскладка кадра (проектное решение v0 — под сворачивание в SPEC §6.2)
//!
//! payload-область — 57×55 клеток (§3.3), в растровом порядке разбита на 8
//! горизонтальных страйпов по строкам: `[7,7,7,7,7,7,7,6]` (та же нарезка, что
//! §6.2 «страйп из H/8 строк»). Внутри страйпа клетки в растровом порядке
//! (строки сверху вниз, столбцы слева направо) образуют битовый поток: каждая
//! клетка отдаёт `bits_per_cell` бит СТАРШИМИ ВПЕРЁД (MSB-first). Пусть
//! `cap = n_cells·bits_per_cell` — полная ёмкость страйпа в битах.
//!
//! * ПОСЛЕДНИЕ 16 бит страйпа — CRC-16/CCITT (poly 0x1021, init 0xFFFF, без
//!   рефлексии, без финального XOR) по ВСЕМ предшествующим `cap−16` битам
//!   страйпа (включая биты-паддинга — см. ниже). Это и есть «per-stripe CRC-16»
//!   из §6.2, локализующая уцелевшие страйпы рваного снимка (§6.3).
//! * Первые `cap−16` бит — область данных. Она несёт `floor((cap−16)/8)` ЦЕЛЫХ
//!   байт (MSB-first), а остаток `(cap−16) mod 8` бит после последнего целого
//!   байта — паддинг-нули (входят в CRC, но не несут данных).
//!
//! Байтовый поток кадра, укладываемый непрерывно по областям данных страйпов
//! 0→7: `FrameHeader` (12 байт) [+ `TransferInfo` (14 байт), если взведён флаг],
//! далее байты encoding-символов транспорта. Байты символов продолжаются из
//! страйпа в страйп, прерываясь только служебными 16 битами CRC каждого страйпа.
//! Границы байт всегда выровнены по границам страйпов (каждый страйп несёт целое
//! число байт), поэтому НИ ОДИН байт не пересекает границу страйпа — салвадж
//! §6.3 забирает байты только из CRC-валидных страйпов без «рваных» байт на швах.
//!
//! Все многобайтовые поля — big-endian. `esi`/`k` — 24-битные на проводе, u32 в
//! памяти; `transfer_length` — 40-битное на проводе, u64 в памяти.

use alloc::vec;
use alloc::vec::Vec;

/// Ширина payload-сетки в клетках (§3.3; зеркалит `symbol::PAYLOAD_COLS`).
pub const PAYLOAD_COLS: usize = 57;
/// Высота payload-сетки в клетках (§3.3; зеркалит `symbol::PAYLOAD_ROWS`).
pub const PAYLOAD_ROWS: usize = 55;
/// Число страйпов кадра (§6.2).
pub const STRIPES: usize = 8;
/// Нарезка 55 payload-строк на 8 страйпов (§6.2): 7·7 + 6.
pub const STRIPE_ROWS: [usize; STRIPES] = [7, 7, 7, 7, 7, 7, 7, 6];

/// Магия заголовка кадра (§6.2): кодпоинт «Ψ» U+03A8.
pub const FRAME_MAGIC: u16 = 0x03A8;
/// Версия формата кадра v0.
pub const FRAME_VERSION: u8 = 1;
/// Бит флага «в кадре присутствует TransferInfo» (§6.2, каждый 8-й кадр).
pub const FLAG_TRANSFER_INFO: u8 = 0x01;
/// Размер FrameHeader на проводе, байт (§6.2).
pub const FRAME_HEADER_LEN: usize = 12;
/// Размер TransferInfo на проводе, байт (§6.2).
pub const TRANSFER_INFO_LEN: usize = 14;

/// Полином CRC-16/CCITT (§6.2).
const CRC16_POLY: u16 = 0x1021;
/// Начальное значение CRC-16/CCITT-FALSE (§6.2).
const CRC16_INIT: u16 = 0xFFFF;

/// Заголовок кадра (§6.2), 12 байт на проводе, big-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Магия формата: должна быть [`FRAME_MAGIC`].
    pub magic: u16,
    /// Версия формата.
    pub version: u8,
    /// Битовые флаги (см. [`FLAG_TRANSFER_INFO`]).
    pub flags: u8,
    /// Идентификатор сессии передачи: случаен на передачу, постоянен внутри неё.
    pub session_id: u32,
    /// ESI первого символа кадра (на проводе 24 бита).
    pub esi: u32,
    /// Число encoding-символов в этом кадре.
    pub count: u8,
}

impl FrameHeader {
    /// Новый заголовок с корректными magic/version и нулевыми флагами.
    pub fn new(session_id: u32, esi: u32, count: u8) -> Self {
        FrameHeader {
            magic: FRAME_MAGIC,
            version: FRAME_VERSION,
            flags: 0,
            session_id,
            esi: esi & 0x00FF_FFFF,
            count,
        }
    }

    /// Сериализация в 12 байт big-endian (§6.2).
    pub fn to_bytes(&self) -> [u8; FRAME_HEADER_LEN] {
        let mut b = [0u8; FRAME_HEADER_LEN];
        b[0..2].copy_from_slice(&self.magic.to_be_bytes());
        b[2] = self.version;
        b[3] = self.flags;
        b[4..8].copy_from_slice(&self.session_id.to_be_bytes());
        // esi — 24 бита big-endian
        b[8] = (self.esi >> 16) as u8;
        b[9] = (self.esi >> 8) as u8;
        b[10] = self.esi as u8;
        b[11] = self.count;
        b
    }

    /// Разбор заголовка из первых 12 байт. `None`, если байт мало или magic
    /// не совпал (никогда не паникует).
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < FRAME_HEADER_LEN {
            return None;
        }
        let magic = u16::from_be_bytes([b[0], b[1]]);
        if magic != FRAME_MAGIC {
            return None;
        }
        let session_id = u32::from_be_bytes([b[4], b[5], b[6], b[7]]);
        let esi = ((b[8] as u32) << 16) | ((b[9] as u32) << 8) | b[10] as u32;
        Some(FrameHeader {
            magic,
            version: b[2],
            flags: b[3],
            session_id,
            esi,
            count: b[11],
        })
    }

    /// Взведён ли флаг присутствия TransferInfo.
    pub fn has_transfer_info(&self) -> bool {
        self.flags & FLAG_TRANSFER_INFO != 0
    }

    /// Фильтр сессии (§6.2): принять кадр только если его `session_id` совпадает
    /// с текущей сессией приёмника. Символы чужой сессии MUST отбрасываться.
    pub fn accepts_session(&self, current_session: u32) -> bool {
        self.session_id == current_session
    }
}

/// Информация о передаче (§6.2), 14 байт, присутствует в каждом 8-м кадре.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferInfo {
    /// Полная длина передаваемого объекта (на проводе 40 бит).
    pub transfer_length: u64,
    /// Размер encoding-символа в байтах.
    pub symbol_size: u16,
    /// Число source-символов K (на проводе 24 бита).
    pub k: u32,
    /// Контрольная сумма объекта (CRC-32C).
    pub checksum: u32,
}

impl TransferInfo {
    /// Сериализация в 14 байт big-endian (§6.2).
    pub fn to_bytes(&self) -> [u8; TRANSFER_INFO_LEN] {
        let mut b = [0u8; TRANSFER_INFO_LEN];
        // transfer_length — 40 бит big-endian
        let tl = self.transfer_length & 0x00FF_FFFF_FFFF;
        b[0] = (tl >> 32) as u8;
        b[1] = (tl >> 24) as u8;
        b[2] = (tl >> 16) as u8;
        b[3] = (tl >> 8) as u8;
        b[4] = tl as u8;
        b[5..7].copy_from_slice(&self.symbol_size.to_be_bytes());
        // k — 24 бита big-endian
        b[7] = (self.k >> 16) as u8;
        b[8] = (self.k >> 8) as u8;
        b[9] = self.k as u8;
        b[10..14].copy_from_slice(&self.checksum.to_be_bytes());
        b
    }

    /// Разбор из первых 14 байт. `None`, если байт мало (не паникует).
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < TRANSFER_INFO_LEN {
            return None;
        }
        let transfer_length = ((b[0] as u64) << 32)
            | ((b[1] as u64) << 24)
            | ((b[2] as u64) << 16)
            | ((b[3] as u64) << 8)
            | b[4] as u64;
        let symbol_size = u16::from_be_bytes([b[5], b[6]]);
        let k = ((b[7] as u32) << 16) | ((b[8] as u32) << 8) | b[9] as u32;
        let checksum = u32::from_be_bytes([b[10], b[11], b[12], b[13]]);
        Some(TransferInfo {
            transfer_length,
            symbol_size,
            k,
            checksum,
        })
    }
}

/// Результат разбора кадра (§6.2–6.3). Разбор терпит любой мусор без паники.
#[derive(Debug, Clone)]
pub struct ParsedFrame {
    /// Прошёл ли CRC каждый из 8 страйпов.
    pub stripes_ok: [bool; STRIPES],
    /// Заголовок из страйпа 0 (только если страйп 0 CRC-валиден и magic верен).
    pub header: Option<FrameHeader>,
    /// TransferInfo из страйпа 0 (если заголовок валиден и флаг взведён).
    pub transfer_info: Option<TransferInfo>,
    /// Салвадж §6.3: для каждого CRC-валидного страйпа — (глобальное смещение
    /// первого байта страйпа в байтовом потоке кадра, байты данных страйпа).
    /// Смещение отсчитывается от начала потока (заголовок = смещение 0).
    pub salvaged: Vec<(usize, Vec<u8>)>,
}

impl ParsedFrame {
    /// Длина служебной «шапки» потока (заголовок [+ TransferInfo]) в байтах.
    /// При неизвестном заголовке предполагает отсутствие TransferInfo.
    pub fn header_len(&self) -> usize {
        match &self.header {
            Some(h) if h.has_transfer_info() => FRAME_HEADER_LEN + TRANSFER_INFO_LEN,
            _ => FRAME_HEADER_LEN,
        }
    }

    /// Салвадж только байтов encoding-символов (за вычетом шапки), с их
    /// смещением в потоке символов транспорта. Байты шапки отбрасываются.
    pub fn salvaged_symbol_bytes(&self) -> Vec<(usize, Vec<u8>)> {
        let hl = self.header_len();
        let mut out = Vec::new();
        for (off, bytes) in &self.salvaged {
            let end = off + bytes.len();
            if end <= hl {
                continue; // весь страйп — служебная шапка
            }
            let (skip, sym_off) = if *off >= hl {
                (0usize, off - hl)
            } else {
                (hl - off, 0usize)
            };
            out.push((sym_off, bytes[skip..].to_vec()));
        }
        out
    }
}

/// Ёмкость страйпа в битах при данном `bits_per_cell`.
#[inline]
fn stripe_cap_bits(rows: usize, bits_per_cell: u32) -> usize {
    rows * PAYLOAD_COLS * bits_per_cell as usize
}

/// Число ЦЕЛЫХ байт данных, помещающихся в страйп (§6.2: `floor((cap−16)/8)`).
#[inline]
fn stripe_data_bytes(rows: usize, bits_per_cell: u32) -> usize {
    (stripe_cap_bits(rows, bits_per_cell) - 16) / 8
}

/// CRC-16/CCITT (poly 0x1021, init 0xFFFF, без рефлексии/финального XOR) по
/// последовательности бит СТАРШИМИ ВПЕРЁД — обрабатывает произвольное число
/// бит (не обязательно кратное 8), как требует раскладка страйпа (§6.2).
fn crc16_bits<I: Iterator<Item = bool>>(bits: I) -> u16 {
    let mut crc = CRC16_INIT;
    for bit in bits {
        let top = (crc >> 15) & 1 == 1;
        crc <<= 1;
        if top ^ bit {
            crc ^= CRC16_POLY;
        }
    }
    crc
}

/// Дописать байт в битовый буфер, СТАРШИМ битом вперёд.
#[inline]
fn push_byte(bits: &mut Vec<bool>, b: u8) {
    for k in (0..8).rev() {
        bits.push((b >> k) & 1 == 1);
    }
}

/// Дописать u16 в битовый буфер, СТАРШИМ битом вперёд.
#[inline]
fn push_u16(bits: &mut Vec<bool>, v: u16) {
    for k in (0..16).rev() {
        bits.push((v >> k) & 1 == 1);
    }
}

/// Собрать клеточные символы кадра (§6.2): заголовок [+ TransferInfo] и байты
/// символов раскладываются по 8 страйпам, каждый страйп замыкается CRC-16.
///
/// * `bits_per_cell` ∈ 1..=7 — сколько бит несёт одна клетка (§5.2).
/// * `symbol_bytes` — байты encoding-символов транспорта; излишек, не влезший в
///   кадр, отбрасывается, недостача добивается нулями.
/// * Флаг [`FLAG_TRANSFER_INFO`] в заголовке синхронизируется с наличием
///   `transfer_info` (взводится/снимается автоматически).
///
/// Возвращает ровно `PAYLOAD_COLS·PAYLOAD_ROWS` клеточных символов в растровом
/// порядке (значение каждой клетки < `2^bits_per_cell`).
pub fn build_frame(
    header: &FrameHeader,
    transfer_info: Option<&TransferInfo>,
    symbol_bytes: &[u8],
    bits_per_cell: u32,
) -> Vec<u8> {
    assert!(
        (1..=7).contains(&bits_per_cell),
        "bits_per_cell должен быть 1..=7"
    );
    let bpc = bits_per_cell as usize;

    // байтовый поток кадра: заголовок [+ TransferInfo] + байты символов
    let mut hdr = *header;
    if transfer_info.is_some() {
        hdr.flags |= FLAG_TRANSFER_INFO;
    } else {
        hdr.flags &= !FLAG_TRANSFER_INFO;
    }
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(&hdr.to_bytes());
    if let Some(ti) = transfer_info {
        stream.extend_from_slice(&ti.to_bytes());
    }
    stream.extend_from_slice(symbol_bytes);

    let mut cells = vec![0u8; PAYLOAD_COLS * PAYLOAD_ROWS];
    let mut byte_pos = 0usize;
    let mut cell_pos = 0usize;
    for &rows in &STRIPE_ROWS {
        let n_cells = rows * PAYLOAD_COLS;
        let cap_bits = n_cells * bpc;
        let data_bits = cap_bits - 16;
        let n_bytes = data_bits / 8;

        let mut bits: Vec<bool> = Vec::with_capacity(cap_bits);
        for _ in 0..n_bytes {
            let b = stream.get(byte_pos).copied().unwrap_or(0);
            byte_pos += 1;
            push_byte(&mut bits, b);
        }
        // паддинг-нули до границы данных (входят в CRC)
        while bits.len() < data_bits {
            bits.push(false);
        }
        let crc = crc16_bits(bits.iter().copied());
        push_u16(&mut bits, crc);
        debug_assert_eq!(bits.len(), cap_bits);

        // упаковка cap_bits бит в n_cells клеток по bpc бит (MSB-first)
        for c in 0..n_cells {
            let mut v = 0u8;
            for k in 0..bpc {
                v = (v << 1) | bits[c * bpc + k] as u8;
            }
            cells[cell_pos + c] = v;
        }
        cell_pos += n_cells;
    }
    cells
}

/// Разобрать клеточные символы кадра обратно (§6.2–6.3). Терпит любой мусор и
/// любой размер входа без паники: недостающие клетки трактуются как нули.
/// Возвращает per-stripe валидность CRC, заголовок (если страйп 0 цел) и салвадж
/// байтов из CRC-валидных страйпов.
pub fn parse_frame(cells: &[u8], bits_per_cell: u32) -> ParsedFrame {
    let bpc = bits_per_cell.clamp(1, 7) as usize;

    let mut stripes_ok = [false; STRIPES];
    let mut salvaged: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut header = None;
    let mut transfer_info = None;

    let mut cell_pos = 0usize;
    let mut byte_off = 0usize;
    for (i, &rows) in STRIPE_ROWS.iter().enumerate() {
        let n_cells = rows * PAYLOAD_COLS;
        let cap_bits = n_cells * bpc;
        let data_bits = cap_bits - 16;
        let n_bytes = data_bits / 8;

        // биты страйпа из клеток (младшие bpc бит клетки, MSB-first)
        let mut bits: Vec<bool> = Vec::with_capacity(cap_bits);
        for c in 0..n_cells {
            let v = cells.get(cell_pos + c).copied().unwrap_or(0);
            for k in (0..bpc).rev() {
                bits.push((v >> k) & 1 == 1);
            }
        }
        cell_pos += n_cells;
        while bits.len() < cap_bits {
            bits.push(false);
        }

        // CRC по data_bits, сверка с хвостовыми 16 битами
        let calc = crc16_bits(bits[..data_bits].iter().copied());
        let mut stored = 0u16;
        for &b in &bits[data_bits..cap_bits] {
            stored = (stored << 1) | b as u16;
        }
        let ok = calc == stored;
        stripes_ok[i] = ok;

        // байты данных страйпа
        let mut dbytes = Vec::with_capacity(n_bytes);
        for bi in 0..n_bytes {
            let mut b = 0u8;
            for k in 0..8 {
                b = (b << 1) | bits[bi * 8 + k] as u8;
            }
            dbytes.push(b);
        }

        if i == 0 && ok {
            if let Some(h) = FrameHeader::from_bytes(&dbytes) {
                if h.has_transfer_info() {
                    transfer_info = TransferInfo::from_bytes(&dbytes[FRAME_HEADER_LEN..]);
                }
                header = Some(h);
            }
        }
        if ok {
            salvaged.push((byte_off, dbytes));
        }
        byte_off += n_bytes;
    }

    ParsedFrame {
        stripes_ok,
        header,
        transfer_info,
        salvaged,
    }
}

/// Суммарная ёмкость кадра под байты (сумма целых байт всех страйпов) при
/// данном `bits_per_cell` — полезно вызывающему для нарезки символов.
pub fn frame_byte_capacity(bits_per_cell: u32) -> usize {
    STRIPE_ROWS
        .iter()
        .map(|&rows| stripe_data_bytes(rows, bits_per_cell))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64: детерминированный ГПСЧ без внешних зависимостей.
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

    fn sample_header() -> FrameHeader {
        let mut h = FrameHeader::new(0xDEAD_BEEF, 0x0012_3456, 7);
        h.version = FRAME_VERSION;
        h
    }

    #[test]
    fn header_bytes_roundtrip() {
        let h = sample_header();
        let b = h.to_bytes();
        assert_eq!(b.len(), FRAME_HEADER_LEN);
        let g = FrameHeader::from_bytes(&b).expect("magic верен");
        assert_eq!(h, g);
        // esi усечён до 24 бит
        assert_eq!(g.esi, 0x0012_3456);
    }

    #[test]
    fn header_rejects_bad_magic_and_short() {
        let mut b = sample_header().to_bytes();
        b[0] ^= 0xFF;
        assert!(FrameHeader::from_bytes(&b).is_none());
        assert!(FrameHeader::from_bytes(&[0u8; 5]).is_none());
    }

    #[test]
    fn transfer_info_bytes_roundtrip() {
        let ti = TransferInfo {
            transfer_length: 0x00AB_CDEF_1234, // 40 бит
            symbol_size: 512,
            k: 0x00AB_CDEF & 0x00FF_FFFF,
            checksum: 0xC0FF_EE00,
        };
        let b = ti.to_bytes();
        assert_eq!(b.len(), TRANSFER_INFO_LEN);
        assert_eq!(TransferInfo::from_bytes(&b), Some(ti));
    }

    /// Roundtrip build→parse для всех bits_per_cell: все страйпы валидны,
    /// заголовок и байты символов восстанавливаются точно.
    #[test]
    fn build_parse_roundtrip_all_bpc() {
        for bpc in 1..=7u32 {
            let cap = frame_byte_capacity(bpc);
            // байты символов: заполняем весь остаток после заголовка+TI
            let sym_len = cap - (FRAME_HEADER_LEN + TRANSFER_INFO_LEN);
            let mut rng = XorShift64(0x1234_5678 ^ bpc as u64);
            let sym: Vec<u8> = (0..sym_len).map(|_| rng.next() as u8).collect();
            let ti = TransferInfo {
                transfer_length: 1_000_000,
                symbol_size: 256,
                k: 4000,
                checksum: 0x1234_5678,
            };
            let h = sample_header();

            let cells = build_frame(&h, Some(&ti), &sym, bpc);
            assert_eq!(cells.len(), PAYLOAD_COLS * PAYLOAD_ROWS);
            for &c in &cells {
                assert!((c as u32) < (1 << bpc), "клетка вне алфавита bpc={bpc}");
            }

            let parsed = parse_frame(&cells, bpc);
            assert!(parsed.stripes_ok.iter().all(|&ok| ok), "bpc={bpc}");
            let ph = parsed.header.expect("заголовок");
            assert_eq!(ph.session_id, h.session_id);
            assert_eq!(ph.esi, h.esi);
            assert_eq!(ph.count, h.count);
            assert!(ph.has_transfer_info());
            assert_eq!(parsed.transfer_info, Some(ti));

            // склеиваем салвадж символов и сверяем с исходными байтами символов
            let mut recon = vec![0u8; sym_len];
            for (off, bytes) in parsed.salvaged_symbol_bytes() {
                for (j, &b) in bytes.iter().enumerate() {
                    if off + j < recon.len() {
                        recon[off + j] = b;
                    }
                }
            }
            assert_eq!(recon, sym, "байты символов bpc={bpc}");
        }
    }

    /// Одна перевёрнутая клетка убивает РОВНО один страйп (тот, куда она попала).
    #[test]
    fn single_flipped_cell_kills_one_stripe() {
        let bpc = 5u32;
        let cap = frame_byte_capacity(bpc);
        let sym: Vec<u8> = (0..cap - FRAME_HEADER_LEN).map(|i| i as u8).collect();
        let h = sample_header();
        let cells = build_frame(&h, None, &sym, bpc);

        // индекс клетки в каждом страйпе -> ожидаемый номер страйпа
        let mut cell_pos = 0usize;
        for (i, &rows) in STRIPE_ROWS.iter().enumerate() {
            let n_cells = rows * PAYLOAD_COLS;
            let victim = cell_pos + n_cells / 2; // клетка внутри страйпа i
            let mut broken = cells.clone();
            broken[victim] ^= 1; // переворот младшего бита -> точно другой символ
            let parsed = parse_frame(&broken, bpc);
            let dead: Vec<usize> = (0..STRIPES).filter(|&s| !parsed.stripes_ok[s]).collect();
            assert_eq!(dead, vec![i], "перевёрнута клетка страйпа {i}");
            cell_pos += n_cells;
        }
    }

    /// Разбор НИКОГДА не паникует на произвольном мусоре и любом размере входа.
    #[test]
    fn parse_never_panics_on_garbage() {
        let mut rng = XorShift64(0xA5A5_5A5A_0F0F_F0F0);
        for _ in 0..2000 {
            let len = (rng.next() % 4000) as usize; // и меньше, и больше 3135
            let cells: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let bpc = 1 + (rng.next() % 7) as u32;
            let parsed = parse_frame(&cells, bpc);
            // контракт: результат определён, салвадж не длиннее числа страйпов
            assert!(parsed.salvaged.len() <= STRIPES);
        }
    }

    #[test]
    fn session_id_filter_helper() {
        let h = FrameHeader::new(0x0102_0304, 0, 1);
        assert!(h.accepts_session(0x0102_0304));
        assert!(!h.accepts_session(0x0102_0305));
    }

    /// Салвадж рваного кадра: бьём CRC средних страйпов — верхний и нижний
    /// прогоны выживают, заголовок из уцелевшего страйпа 0 читается.
    #[test]
    fn salvage_survives_middle_corruption() {
        let bpc = 4u32;
        let cap = frame_byte_capacity(bpc);
        let sym: Vec<u8> = (0..cap - FRAME_HEADER_LEN).map(|i| (i * 7) as u8).collect();
        let h = sample_header();
        let cells = build_frame(&h, None, &sym, bpc);

        // порча страйпов 3 и 4 (середина): по одной клетке в каждом
        let mut cell_pos = 0usize;
        let mut broken = cells.clone();
        for (i, &rows) in STRIPE_ROWS.iter().enumerate() {
            let n_cells = rows * PAYLOAD_COLS;
            if i == 3 || i == 4 {
                broken[cell_pos + 1] ^= 1;
            }
            cell_pos += n_cells;
        }
        let parsed = parse_frame(&broken, bpc);
        assert!(parsed.stripes_ok[0], "страйп 0 должен выжить");
        assert!(!parsed.stripes_ok[3] && !parsed.stripes_ok[4], "3,4 мертвы");
        assert!(parsed.header.is_some(), "заголовок из целого страйпа 0");
        // салвадж — только из валидных страйпов (не из 3 и 4)
        assert_eq!(parsed.salvaged.len(), STRIPES - 2);
    }
}
