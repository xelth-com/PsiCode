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

// ============================================================================
// ЭКСПЕРИМЕНТАЛЬНЫЙ РАЗДЕЛ (§6.2 live-grade SER, кандидаты под сворачивание).
//
// НИЧЕГО выше этой линии не меняется: фрозен-формат §6.2 (build_frame/
// parse_frame, FrameHeader, TransferInfo, CRC-16 по 8 страйпам) остаётся
// байт-в-байт. Здесь — ТОЛЬКО аддитивные альтернативные кодеки-кандидаты для
// живого канала, где измеренный SER клетки 1–13% убивает 399-клеточный страйп
// (survival = (1−p)^399 ≈ 5·10⁻⁶ при p=3%), и поток не стартует (k=0 навсегда).
//
//   V1 — мелкозернистый CRC поблочно (строка = 57 клеток / 2 строки = 114):
//        survival блока = (1−p)^57, локализует уцелевших; заголовок живёт в
//        первых 1–2 блоках вместо всего страйпа 0 → резко падает время до
//        первого читаемого заголовка. См. [`BlockLayout`], [`build_frame_blocks`].
//   V2 — внутренний FEC: перемежённый RS над GF(256) по байтам кадра. Корректит
//        t байтовых ошибок на кодовое слово. См. [`rs256_encode`]/[`rs256_decode`].
//        (Перемежение и раскладку по клеткам делает оценщик psicode-sim.)
//   V3 — голосование повторов на приёмнике: БЕЗ изменения провода. Реализуется
//        целиком на стороне rx (см. psicode-sim::l3live и отчёт); ядру достаточно
//        существующего parse_frame/parse_frame_blocks по проголосованным клеткам.
//
// Всё помечено эксперим.: под сворачивание в SPEC отдельным решением.
// ============================================================================

/// Число клеток payload-сетки (зеркалит `PAYLOAD_COLS·PAYLOAD_ROWS`).
pub const PAYLOAD_CELLS: usize = PAYLOAD_COLS * PAYLOAD_ROWS;

/// Полином CRC-8 (CRC-8/SMBus, poly 0x07, init 0x00, без рефлексии/финального
/// XOR) — эксперим. V1: мелкозернистый CRC. Ложное принятие битой строки ≈ 2⁻⁸
/// (0.39%) — вдвое дешевле по накладным, чем CRC-16, но слабее по детекции;
/// см. [`BlockLayout::PER_2ROW_CRC16`] для CRC-16 при той же ~14% избыточности.
const CRC8_POLY: u8 = 0x07;

/// CRC-8 по последовательности бит СТАРШИМИ ВПЕРЁД (произвольное число бит).
fn crc8_bits<I: Iterator<Item = bool>>(bits: I) -> u8 {
    let mut crc: u8 = 0x00;
    for bit in bits {
        let top = (crc >> 7) & 1 == 1;
        crc <<= 1;
        if top ^ bit {
            crc ^= CRC8_POLY;
        }
    }
    crc
}

/// Раскладка кадра на блоки строк, каждый со своим CRC (эксперим. V1).
///
/// Payload-сетка (55 строк) режется на блоки по `rows_per_block` строк
/// (последний блок берёт остаток). Каждый блок несёт `crc_bits`-битный CRC
/// (8 или 16) в хвосте и целое число байт данных в начале — как страйп §6.2,
/// но мельче. Байт-поток кадра (`FrameHeader`[+`TransferInfo`]+байты символов)
/// укладывается непрерывно блок за блоком; ни один байт не пересекает границу
/// блока (каждый блок несёт целое число байт), поэтому салвадж берёт байты
/// только из CRC-валидных блоков.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    /// Строк в блоке (последний может быть короче).
    pub rows_per_block: usize,
    /// Ширина CRC блока в битах: 8 или 16.
    pub crc_bits: usize,
}

impl BlockLayout {
    /// V1a: по одной строке (57 клеток) + CRC-8. Избыточность ≈ 8/57 = 14%.
    pub const PER_ROW_CRC8: BlockLayout = BlockLayout { rows_per_block: 1, crc_bits: 8 };
    /// V1b: по две строки (114 клеток) + CRC-16. Та же ~14%, но детекция 2⁻¹⁶.
    pub const PER_2ROW_CRC16: BlockLayout = BlockLayout { rows_per_block: 2, crc_bits: 16 };

    /// Список блоков как число строк в каждом (последний — остаток).
    pub fn blocks(&self) -> Vec<usize> {
        let rpb = self.rows_per_block.max(1);
        let mut v = Vec::new();
        let mut r = PAYLOAD_ROWS;
        while r > 0 {
            let b = rpb.min(r);
            v.push(b);
            r -= b;
        }
        v
    }

    /// Байт данных в блоке из `rows` строк при данном `bits_per_cell`.
    pub fn block_data_bytes(&self, rows: usize, bits_per_cell: u32) -> usize {
        let cap = rows * PAYLOAD_COLS * bits_per_cell as usize;
        cap.saturating_sub(self.crc_bits) / 8
    }

    /// Суммарная байт-ёмкость кадра при этой раскладке и `bits_per_cell`.
    pub fn frame_byte_capacity(&self, bits_per_cell: u32) -> usize {
        self.blocks()
            .iter()
            .map(|&r| self.block_data_bytes(r, bits_per_cell))
            .sum()
    }
}

/// Результат поблочного разбора (эксперим. V1). Терпит любой мусор без паники.
#[derive(Debug, Clone)]
pub struct ParsedBlockFrame {
    /// Прошёл ли CRC каждый блок (в порядке раскладки).
    pub blocks_ok: Vec<bool>,
    /// Заголовок, если ведущие блоки, покрывающие его байты, CRC-валидны.
    pub header: Option<FrameHeader>,
    /// TransferInfo, если заголовок валиден, флаг взведён и его байты покрыты.
    pub transfer_info: Option<TransferInfo>,
    /// Для каждого валидного блока: (смещение первого байта блока в потоке,
    /// байты данных блока). Смещение от начала потока (заголовок = 0).
    pub salvaged: Vec<(usize, Vec<u8>)>,
}

/// Собрать клетки кадра в поблочной раскладке `layout` (эксперим. V1).
/// Семантика идентична [`build_frame`], но CRC ставится на каждый мелкий блок.
pub fn build_frame_blocks(
    header: &FrameHeader,
    transfer_info: Option<&TransferInfo>,
    symbol_bytes: &[u8],
    bits_per_cell: u32,
    layout: &BlockLayout,
) -> Vec<u8> {
    assert!(
        (1..=7).contains(&bits_per_cell),
        "bits_per_cell должен быть 1..=7"
    );
    assert!(
        layout.crc_bits == 8 || layout.crc_bits == 16,
        "crc_bits должен быть 8 или 16"
    );
    let bpc = bits_per_cell as usize;

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

    let mut cells = vec![0u8; PAYLOAD_CELLS];
    let mut byte_pos = 0usize;
    let mut cell_pos = 0usize;
    for &rows in &layout.blocks() {
        let n_cells = rows * PAYLOAD_COLS;
        let cap_bits = n_cells * bpc;
        let data_bits = cap_bits - layout.crc_bits;
        let n_bytes = data_bits / 8;

        let mut bits: Vec<bool> = Vec::with_capacity(cap_bits);
        for _ in 0..n_bytes {
            let b = stream.get(byte_pos).copied().unwrap_or(0);
            byte_pos += 1;
            push_byte(&mut bits, b);
        }
        while bits.len() < data_bits {
            bits.push(false);
        }
        if layout.crc_bits == 8 {
            let crc = crc8_bits(bits.iter().copied());
            for k in (0..8).rev() {
                bits.push((crc >> k) & 1 == 1);
            }
        } else {
            let crc = crc16_bits(bits.iter().copied());
            push_u16(&mut bits, crc);
        }
        debug_assert_eq!(bits.len(), cap_bits);

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

/// Разобрать клетки поблочной раскладки обратно (эксперим. V1). Не паникует.
pub fn parse_frame_blocks(
    cells: &[u8],
    bits_per_cell: u32,
    layout: &BlockLayout,
) -> ParsedBlockFrame {
    let bpc = bits_per_cell.clamp(1, 7) as usize;
    let crc_w = if layout.crc_bits == 8 { 8 } else { 16 };

    let mut blocks_ok: Vec<bool> = Vec::new();
    let mut salvaged: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut header = None;
    let mut transfer_info = None;

    // ведущий непрерывный префикс валидных блоков от блока 0 — для заголовка.
    let mut prefix: Vec<u8> = Vec::new();
    let mut prefix_contiguous = true;

    let mut cell_pos = 0usize;
    let mut byte_off = 0usize;
    for &rows in &layout.blocks() {
        let n_cells = rows * PAYLOAD_COLS;
        let cap_bits = n_cells * bpc;
        let data_bits = cap_bits.saturating_sub(crc_w);
        let n_bytes = data_bits / 8;

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

        let ok = if crc_w == 8 {
            let calc = crc8_bits(bits[..data_bits].iter().copied());
            let mut stored = 0u8;
            for &b in &bits[data_bits..cap_bits] {
                stored = (stored << 1) | b as u8;
            }
            calc == stored
        } else {
            let calc = crc16_bits(bits[..data_bits].iter().copied());
            let mut stored = 0u16;
            for &b in &bits[data_bits..cap_bits] {
                stored = (stored << 1) | b as u16;
            }
            calc == stored
        };
        blocks_ok.push(ok);

        let mut dbytes = Vec::with_capacity(n_bytes);
        for bi in 0..n_bytes {
            let mut b = 0u8;
            for k in 0..8 {
                b = (b << 1) | bits[bi * 8 + k] as u8;
            }
            dbytes.push(b);
        }

        if prefix_contiguous {
            if ok {
                prefix.extend_from_slice(&dbytes);
            } else {
                prefix_contiguous = false;
            }
        }
        if ok {
            salvaged.push((byte_off, dbytes));
        }
        byte_off += n_bytes;
    }

    if prefix.len() >= FRAME_HEADER_LEN {
        if let Some(h) = FrameHeader::from_bytes(&prefix) {
            if h.has_transfer_info() && prefix.len() >= FRAME_HEADER_LEN + TRANSFER_INFO_LEN {
                transfer_info = TransferInfo::from_bytes(&prefix[FRAME_HEADER_LEN..]);
            }
            header = Some(h);
        }
    }

    ParsedBlockFrame {
        blocks_ok,
        header,
        transfer_info,
        salvaged,
    }
}

// --- V2: RS над GF(256) (эксперим., перемежение делает оценщик) ---

/// Арифметика GF(2⁸), примитивный полином 0x11D (x⁸+x⁴+x³+x²+1). Таблицы
/// exp/log в compile-time; в рантайме — индексация. Эксперим. под V2.
mod gf256 {
    const PRIM: u16 = 0x11D;

    const fn build() -> ([u8; 512], [u8; 256]) {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u16 = 1;
        let mut i = 0;
        while i < 255 {
            exp[i] = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= PRIM;
            }
            i += 1;
        }
        let mut j = 255;
        while j < 512 {
            exp[j] = exp[j - 255];
            j += 1;
        }
        (exp, log)
    }

    const T: ([u8; 512], [u8; 256]) = build();

    #[inline]
    pub const fn add(a: u8, b: u8) -> u8 {
        a ^ b
    }
    #[inline]
    pub const fn mul(a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            0
        } else {
            T.0[T.1[a as usize] as usize + T.1[b as usize] as usize]
        }
    }
    #[inline]
    pub fn div(a: u8, b: u8) -> u8 {
        debug_assert!(b != 0, "division by zero in GF(256)");
        if a == 0 {
            0
        } else {
            let la = T.1[a as usize] as isize;
            let lb = T.1[b as usize] as isize;
            T.0[(la - lb).rem_euclid(255) as usize]
        }
    }
    #[inline]
    pub fn inv(a: u8) -> u8 {
        div(1, a)
    }
    #[inline]
    pub const fn exp(p: usize) -> u8 {
        T.0[p % 255]
    }
}

/// Ошибка декодера RS(256).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rs256Error {
    /// Ошибок больше, чем `nsym/2`, — исправить нельзя.
    TooManyErrors,
}

/// Порождающий полином степени `nsym`, старшим коэффициентом вперёд.
fn rs256_generator(nsym: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 0..nsym {
        let root = gf256::exp(i);
        let mut out = vec![0u8; g.len() + 1];
        for a in 0..g.len() {
            out[a] = gf256::add(out[a], g[a]);
            out[a + 1] = gf256::add(out[a + 1], gf256::mul(g[a], root));
        }
        g = out;
    }
    g
}

fn rs256_eval(p: &[u8], x: u8) -> u8 {
    let mut y = 0u8;
    for &c in p {
        y = gf256::add(gf256::mul(y, x), c);
    }
    y
}

/// Систематический RS над GF(256): к `msg` дописывает `nsym` проверочных байт.
/// Кодовое слово длиной `msg.len()+nsym` (≤ 255). Корни g: alpha⁰..alpha^{nsym-1}.
pub fn rs256_encode(msg: &[u8], nsym: usize) -> Vec<u8> {
    let n = msg.len() + nsym;
    debug_assert!(n <= 255, "RS(256): длина слова > 255");
    let gen = rs256_generator(nsym);
    let mut rem = vec![0u8; n];
    rem[..msg.len()].copy_from_slice(msg);
    for i in 0..msg.len() {
        let coef = rem[i];
        if coef != 0 {
            for j in 1..gen.len() {
                rem[i + j] = gf256::add(rem[i + j], gf256::mul(gen[j], coef));
            }
        }
    }
    let mut out = vec![0u8; n];
    out[..msg.len()].copy_from_slice(msg);
    out[msg.len()..].copy_from_slice(&rem[msg.len()..]);
    out
}

fn rs256_syndromes(code: &[u8], nsym: usize) -> (Vec<u8>, bool) {
    let mut s = vec![0u8; nsym];
    let mut any = false;
    for i in 0..nsym {
        s[i] = rs256_eval(code, gf256::exp(i));
        any |= s[i] != 0;
    }
    (s, any)
}

fn rs256_berlekamp_massey(synd: &[u8]) -> Vec<u8> {
    let nsym = synd.len();
    let mut sigma = vec![1u8];
    let mut prev = vec![1u8];
    let mut l = 0usize;
    let mut m = 1usize;
    let mut b = 1u8;
    for n in 0..nsym {
        let mut d = synd[n];
        for i in 1..=l.min(sigma.len().saturating_sub(1)) {
            d = gf256::add(d, gf256::mul(sigma[i], synd[n - i]));
        }
        if d == 0 {
            m += 1;
        } else {
            let coef = gf256::div(d, b);
            let mut shifted = vec![0u8; m];
            shifted.extend_from_slice(&prev);
            let update = |sigma: &mut Vec<u8>| {
                for (i, &c) in shifted.iter().enumerate() {
                    if i < sigma.len() {
                        sigma[i] = gf256::add(sigma[i], gf256::mul(coef, c));
                    } else {
                        sigma.push(gf256::mul(coef, c));
                    }
                }
            };
            if 2 * l <= n {
                let old = sigma.clone();
                update(&mut sigma);
                l = n + 1 - l;
                prev = old;
                b = d;
                m = 1;
            } else {
                update(&mut sigma);
                m += 1;
            }
        }
    }
    sigma.reverse();
    sigma
}

fn rs256_find_positions(sigma_hi: &[u8], n: usize, t: usize) -> Option<Vec<usize>> {
    let errs = sigma_hi.len() - 1;
    if errs == 0 {
        return Some(vec![]);
    }
    if errs > t {
        return None;
    }
    let mut pos = Vec::with_capacity(errs);
    for i in 0..255usize {
        if rs256_eval(sigma_hi, gf256::exp(i)) == 0 {
            let j = (255 - i) % 255;
            if j >= n {
                return None;
            }
            pos.push(n - 1 - j);
        }
    }
    if pos.len() != errs {
        return None;
    }
    Some(pos)
}

fn rs256_correct(code: &mut [u8], synd: &[u8], positions: &[usize]) {
    let n = code.len();
    let nsym = synd.len();
    let xs: Vec<u8> = positions
        .iter()
        .map(|&p| gf256::exp((n - 1 - p) % 255))
        .collect();

    // Lambda(x) младшим коэффициентом вперёд из корней (1 - X·x) => [1, X].
    let mut lambda = vec![1u8];
    for &x in &xs {
        let a = lambda.clone();
        let bb = [1u8, x];
        let mut out = vec![0u8; a.len() + 1];
        for (i, &ai) in a.iter().enumerate() {
            for (jj, &bj) in bb.iter().enumerate() {
                out[i + jj] = gf256::add(out[i + jj], gf256::mul(ai, bj));
            }
        }
        lambda = out;
    }
    let mut omega = vec![0u8; nsym];
    for i in 0..nsym {
        for j in 0..=i.min(lambda.len() - 1) {
            omega[i] = gf256::add(omega[i], gf256::mul(synd[i - j], lambda[j]));
        }
    }
    let mut deriv = vec![0u8; lambda.len().saturating_sub(1)];
    let mut i = 1;
    while i < lambda.len() {
        deriv[i - 1] = lambda[i];
        i += 2;
    }
    let eval_lo = |p: &[u8], x: u8| -> u8 {
        let mut y = 0u8;
        for &c in p.iter().rev() {
            y = gf256::add(gf256::mul(y, x), c);
        }
        y
    };
    for (k, &p) in positions.iter().enumerate() {
        let x_inv = gf256::inv(xs[k]);
        let num = eval_lo(&omega, x_inv);
        let den = eval_lo(&deriv, x_inv);
        if den == 0 {
            continue;
        }
        let mag = gf256::mul(xs[k], gf256::div(num, den));
        code[p] = gf256::add(code[p], mag);
    }
}

/// Декодирует слово RS(256) на месте (`nsym` проверочных). Возвращает число
/// исправленных байт или [`Rs256Error::TooManyErrors`]. Позиции ошибок неизвестны
/// (исправляет до `nsym/2`).
pub fn rs256_decode(code: &mut [u8], nsym: usize) -> Result<usize, Rs256Error> {
    let (synd, has) = rs256_syndromes(code, nsym);
    if !has {
        return Ok(0);
    }
    let sigma = rs256_berlekamp_massey(&synd);
    let t = nsym / 2;
    let positions =
        rs256_find_positions(&sigma, code.len(), t).ok_or(Rs256Error::TooManyErrors)?;
    if positions.is_empty() {
        return Err(Rs256Error::TooManyErrors);
    }
    rs256_correct(code, &synd, &positions);
    let (_, still) = rs256_syndromes(code, nsym);
    if still {
        return Err(Rs256Error::TooManyErrors);
    }
    Ok(positions.len())
}

#[cfg(test)]
mod experimental_tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn sample_header() -> FrameHeader {
        FrameHeader::new(0xDEAD_BEEF, 0x0012_3456, 7)
    }

    fn sample_ti() -> TransferInfo {
        TransferInfo {
            transfer_length: 1_000_000,
            symbol_size: 256,
            k: 4000,
            checksum: 0x1234_5678,
        }
    }

    /// Раскладки блоков покрывают все 55 строк без остатка.
    #[test]
    fn block_layout_covers_all_rows() {
        for lay in [BlockLayout::PER_ROW_CRC8, BlockLayout::PER_2ROW_CRC16] {
            assert_eq!(lay.blocks().iter().sum::<usize>(), PAYLOAD_ROWS);
        }
        assert_eq!(BlockLayout::PER_ROW_CRC8.blocks().len(), 55);
        assert_eq!(BlockLayout::PER_2ROW_CRC16.blocks().len(), 28); // 27*2 + 1
    }

    /// Roundtrip build_frame_blocks→parse для обеих раскладок и всех bpc:
    /// все блоки валидны, заголовок+TI восстанавливаются, символы точны.
    #[test]
    fn block_build_parse_roundtrip() {
        for lay in [BlockLayout::PER_ROW_CRC8, BlockLayout::PER_2ROW_CRC16] {
            for bpc in 1..=7u32 {
                let cap = lay.frame_byte_capacity(bpc);
                let sym_len = cap - (FRAME_HEADER_LEN + TRANSFER_INFO_LEN);
                let mut rng = Lcg(0xABCD ^ bpc as u64 ^ (lay.crc_bits as u64) << 8);
                let sym: Vec<u8> = (0..sym_len).map(|_| rng.next() as u8).collect();
                let cells = build_frame_blocks(&sample_header(), Some(&sample_ti()), &sym, bpc, &lay);
                assert_eq!(cells.len(), PAYLOAD_CELLS);
                for &c in &cells {
                    assert!((c as u32) < (1 << bpc));
                }
                let parsed = parse_frame_blocks(&cells, bpc, &lay);
                assert!(parsed.blocks_ok.iter().all(|&ok| ok), "bpc={bpc} crc={}", lay.crc_bits);
                let h = parsed.header.expect("заголовок");
                assert_eq!(h.esi, sample_header().esi);
                assert_eq!(parsed.transfer_info, Some(sample_ti()));

                // склейка салваджа символов
                let hl = FRAME_HEADER_LEN + TRANSFER_INFO_LEN;
                let mut recon = vec![0u8; sym_len];
                for (off, bytes) in &parsed.salvaged {
                    for (j, &b) in bytes.iter().enumerate() {
                        let g = off + j;
                        if g >= hl && g - hl < recon.len() {
                            recon[g - hl] = b;
                        }
                    }
                }
                assert_eq!(recon, sym, "символы bpc={bpc}");
            }
        }
    }

    /// Одна перевёрнутая клетка убивает РОВНО один блок (per-row CRC-8).
    #[test]
    fn block_single_flip_kills_one_block() {
        let lay = BlockLayout::PER_ROW_CRC8;
        let bpc = 3u32;
        let cap = lay.frame_byte_capacity(bpc);
        let sym: Vec<u8> = (0..cap - FRAME_HEADER_LEN).map(|i| i as u8).collect();
        let cells = build_frame_blocks(&sample_header(), None, &sym, bpc, &lay);
        // блок b = строки [b, b+1); клетка внутри него
        for b in [0usize, 7, 27, 54] {
            let victim = b * PAYLOAD_COLS + 3;
            let mut broken = cells.clone();
            broken[victim] ^= 1;
            let parsed = parse_frame_blocks(&broken, bpc, &lay);
            let dead: Vec<usize> = (0..parsed.blocks_ok.len())
                .filter(|&i| !parsed.blocks_ok[i])
                .collect();
            assert_eq!(dead, vec![b], "перевёрнута клетка блока {b}");
        }
    }

    /// Порча строки заголовка убивает заголовок; порча дальней строки — нет.
    #[test]
    fn block_header_survives_far_corruption() {
        let lay = BlockLayout::PER_ROW_CRC8;
        let bpc = 5u32; // заголовок 12 B влезает в строку 0 (34 B/строку)
        let cap = lay.frame_byte_capacity(bpc);
        let sym: Vec<u8> = (0..cap - FRAME_HEADER_LEN).map(|i| (i * 3) as u8).collect();
        let cells = build_frame_blocks(&sample_header(), None, &sym, bpc, &lay);
        // порча строки 40 — заголовок цел
        let mut broken = cells.clone();
        broken[40 * PAYLOAD_COLS + 1] ^= 1;
        assert!(parse_frame_blocks(&broken, bpc, &lay).header.is_some());
        // порча строки 0 — заголовок мёртв
        let mut broken0 = cells.clone();
        broken0[1] ^= 1;
        assert!(parse_frame_blocks(&broken0, bpc, &lay).header.is_none());
    }

    #[test]
    fn parse_frame_blocks_never_panics() {
        let mut rng = Lcg(0x5A5A_1234);
        for lay in [BlockLayout::PER_ROW_CRC8, BlockLayout::PER_2ROW_CRC16] {
            for _ in 0..500 {
                let len = (rng.below(4000)) as usize;
                let cells: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
                let bpc = 1 + (rng.below(7)) as u32;
                let _ = parse_frame_blocks(&cells, bpc, &lay);
            }
        }
    }

    #[test]
    fn gf256_field_axioms() {
        for a in 1u16..256 {
            let a = a as u8;
            assert_eq!(gf256::mul(a, gf256::inv(a)), 1, "a·a⁻¹≠1 a={a}");
        }
        // exp/log обход
        for p in 0..255usize {
            let e = gf256::exp(p);
            assert_ne!(e, 0);
        }
    }

    fn rs256_msg(rng: &mut Lcg, k: usize) -> Vec<u8> {
        (0..k).map(|_| rng.next() as u8).collect()
    }

    #[test]
    fn rs256_clean_roundtrip() {
        let mut rng = Lcg(1);
        for &(k, nsym) in &[(223usize, 32usize), (191, 64), (47, 16)] {
            for _ in 0..50 {
                let msg = rs256_msg(&mut rng, k);
                let mut code = rs256_encode(&msg, nsym);
                assert_eq!(code.len(), k + nsym);
                assert_eq!(rs256_decode(&mut code, nsym), Ok(0));
                assert_eq!(&code[..k], &msg[..]);
            }
        }
    }

    #[test]
    fn rs256_corrects_up_to_t_errors() {
        let mut rng = Lcg(7);
        for &(k, nsym) in &[(223usize, 32usize), (191, 64)] {
            let t = nsym / 2;
            for round in 0..120 {
                let msg = rs256_msg(&mut rng, k);
                let clean = rs256_encode(&msg, nsym);
                let nerr = 1 + (round % t);
                let mut code = clean.clone();
                let mut used = alloc::vec![false; code.len()];
                let mut placed = 0;
                while placed < nerr {
                    let p = rng.below(code.len() as u64) as usize;
                    if !used[p] {
                        used[p] = true;
                        let d = 1 + rng.below(255) as u8;
                        code[p] = gf256::add(code[p], d);
                        placed += 1;
                    }
                }
                let fixed = rs256_decode(&mut code, nsym).expect("≤t ошибок исправимы");
                assert_eq!(fixed, nerr, "k={k} nsym={nsym} round={round}");
                assert_eq!(code, clean);
            }
        }
    }

    /// Сплошной burst длиной t исправляется (RS корректит t в любых позициях).
    #[test]
    fn rs256_corrects_contiguous_burst() {
        let mut rng = Lcg(9);
        let (k, nsym) = (223usize, 32usize);
        let t = nsym / 2;
        let clean = rs256_encode(&rs256_msg(&mut rng, k), nsym);
        for start in [0usize, 50, 200, k + nsym - t] {
            let mut code = clean.clone();
            for p in start..(start + t).min(code.len()) {
                code[p] = gf256::add(code[p], 1 + rng.below(255) as u8);
            }
            let _ = rs256_decode(&mut code, nsym).expect("burst t исправим");
            assert_eq!(code, clean, "start={start}");
        }
    }
}
