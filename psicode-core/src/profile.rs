//! Калибровочный профиль канала «монитор -> камера».
//!
//! Телефон измеряет канал по одному тестовому кадру, упаковывает результат в
//! 80-битный payload (72 бита полей + CRC-8), кодирует двумя перемежёнными
//! RS(16,8) над GF(32) и показывает человеку 32 символа Crockford Base32
//! (160 бит, 8 групп по 4):
//!
//! ```text
//! XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX
//! ```
//!
//! Человек вводит их на передатчике. RS гарантированно исправляет любые
//! 4 опечатки, а благодаря перемежению — и до 8 подряд идущих (две соседние
//! группы целиком); CRC-8 страхует от мискоррекции при большем числе ошибок.
//!
//! Раскладка 72 бит (по порядку записи, старшие биты первыми):
//!
//! | поле                | бит | физический смысл                                  |
//! |---------------------|-----|---------------------------------------------------|
//! | version             |  4  | версия формата (сейчас 1)                         |
//! | cell_size_px_m2     |  6  | сторона ячейки в px монитора, 2..=65 (хранится -2)|
//! | frame_hold_m1       |  4  | удержание кадра в периодах 60 Гц, 1..=16 (-1)     |
//! | luma_bits_m1        |  2  | бит/ячейку яркостью, 1..=4 (-1)                   |
//! | chroma_mode         |  3  | 0=моно .. см. ChromaMode                          |
//! | gamma_g_q           |  6  | гамма G: 1.500 + 0.025*q                          |
//! | gamma_r_delta_q     |  4  | сдвиг гаммы R от G: (q-8)*0.025                   |
//! | gamma_b_delta_q     |  4  | сдвиг гаммы B от G: (q-8)*0.025                   |
//! | white_level_q       |  4  | амплитуда белого: 55 + 3*q %                      |
//! | black_level_q       |  4  | подъём чёрного: q %                               |
//! | noise_sigma_q       |  5  | сигма шума, лог-шкала: 0.25 * 2^(q/4) град.       |
//! | mtf_limit_px_m1     |  5  | мельчайшая разрешимая полоса, px, 1..=32 (-1)     |
//! | torn_frames_q       |  4  | доля рваных кадров, лог-шкала (см. torn_pct)      |
//! | crosstalk_rg_q      |  4  | утечка R<->G, q * 2 %                             |
//! | crosstalk_gb_q      |  4  | утечка G<->B, q * 2 %                             |
//! | quiet_zone          |  2  | пресет тихой зоны                                 |
//! | fec_overhead        |  3  | пресет избыточности RaptorQ                       |
//! | reserved            |  4  | нули; поле для будущих версий                     |
//! | ----- итого         |  72 |                                                   |
//! | crc8                |  8  | CRC-8 (poly 0x07) по 9 байтам полей               |

use crate::base32;
use crate::bits::{BitReader, BitWriter};
use crate::rs;
use alloc::string::String;

pub const CODE_SYMBOLS: usize = rs::CODE_LEN; // 32
pub const CODE_CHARS_GROUPED: usize = CODE_SYMBOLS + CODE_SYMBOLS / base32::GROUP - 1; // 39 с дефисами

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaMode {
    /// только яркость (канал G / luma)
    Mono = 0,
    /// + 1 бит в хроме (2 уровня на R-B оси)
    Chroma1 = 1,
    /// + 2 бита в хроме
    Chroma2 = 2,
    /// + 3 бита в хроме
    Chroma3 = 3,
    /// только зелёный субпиксель, R и B выключены (борьба с хром. аберрацией)
    GreenOnly = 4,
}

impl ChromaMode {
    fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Mono,
            1 => Self::Chroma1,
            2 => Self::Chroma2,
            3 => Self::Chroma3,
            4 => Self::GreenOnly,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibProfile {
    pub version: u8,
    pub cell_size_px: u8,      // 2..=65
    pub frame_hold_periods: u8, // 1..=16
    pub luma_bits: u8,         // 1..=4
    pub chroma_mode: ChromaMode,
    pub gamma_g_q: u8,      // 0..=63
    pub gamma_r_delta_q: u8, // 0..=15, центр 8
    pub gamma_b_delta_q: u8, // 0..=15, центр 8
    pub white_level_q: u8,  // 0..=15
    pub black_level_q: u8,  // 0..=15
    pub noise_sigma_q: u8,  // 0..=31
    pub mtf_limit_px: u8,   // 1..=32
    pub torn_frames_q: u8,  // 0..=15
    pub crosstalk_rg_q: u8, // 0..=15
    pub crosstalk_gb_q: u8, // 0..=15
    pub quiet_zone: u8,     // 0..=3
    pub fec_overhead: u8,   // 0..=7
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileError {
    /// недопустимый символ во вводе (позиция в строке)
    BadChar(usize),
    /// длина не равна 32 символам
    BadLength(usize),
    /// RS не смог исправить
    Uncorrectable,
    /// CRC не сошёлся после RS-декодирования (вероятная мискоррекция)
    CrcMismatch,
    /// неизвестная версия формата
    BadVersion(u8),
    /// поле вне допустимого диапазона
    BadField(&'static str),
}

impl core::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProfileError::BadChar(i) => write!(f, "invalid character at position {i}"),
            ProfileError::BadLength(n) => write!(f, "expected 32 symbols, got {n}"),
            ProfileError::Uncorrectable => f.write_str("too many errors: RS could not correct"),
            ProfileError::CrcMismatch => f.write_str("CRC mismatch after RS decode (miscorrection)"),
            ProfileError::BadVersion(v) => write!(f, "unsupported format version {v}"),
            ProfileError::BadField(name) => write!(f, "field out of range: {name}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProfileError {}

fn crc8(bytes: &[u8]) -> u8 {
    // CRC-8, poly 0x07, init 0x00
    let mut crc = 0u8;
    for &b in bytes {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 { (crc << 1) ^ 0x07 } else { crc << 1 };
        }
    }
    crc
}

impl CalibProfile {
    pub const VERSION: u8 = 1;

    // --- физические значения из квантованных полей ---

    pub fn gamma_g(&self) -> f32 {
        1.5 + 0.025 * self.gamma_g_q as f32
    }
    pub fn gamma_r(&self) -> f32 {
        self.gamma_g() + 0.025 * (self.gamma_r_delta_q as f32 - 8.0)
    }
    pub fn gamma_b(&self) -> f32 {
        self.gamma_g() + 0.025 * (self.gamma_b_delta_q as f32 - 8.0)
    }
    pub fn white_level_pct(&self) -> u8 {
        55 + 3 * self.white_level_q
    }
    pub fn black_level_pct(&self) -> u8 {
        self.black_level_q
    }
    /// сигма шума в градациях серого (0..255): 0.25 * 2^(q/4), т.е. 0.25..~54
    ///
    /// Требует трансцендентной математики (`powf`), поэтому доступно только с
    /// фичей `std`. В no_std пользуйтесь сырым полем `noise_sigma_q`.
    #[cfg(feature = "std")]
    pub fn noise_sigma(&self) -> f32 {
        0.25 * (2f32).powf(self.noise_sigma_q as f32 / 4.0)
    }
    /// доля рваных кадров, %: 0 -> 0%, иначе 0.1 * 2^(q-1) %, потолок 100
    ///
    /// Требует трансцендентной математики (`powi`/`min`), поэтому доступно
    /// только с фичей `std`. В no_std пользуйтесь сырым полем `torn_frames_q`.
    #[cfg(feature = "std")]
    pub fn torn_pct(&self) -> f32 {
        if self.torn_frames_q == 0 {
            0.0
        } else {
            (0.1 * (2f32).powi(self.torn_frames_q as i32 - 1)).min(100.0)
        }
    }
    pub fn crosstalk_rg_pct(&self) -> u8 {
        self.crosstalk_rg_q * 2
    }
    pub fn crosstalk_gb_pct(&self) -> u8 {
        self.crosstalk_gb_q * 2
    }
    /// эффективный FPS полезных кадров при мониторе 60 Гц
    /// (обёртка над [`effective_fps_at`](Self::effective_fps_at) для 60 Гц).
    pub fn effective_fps(&self) -> f32 {
        self.effective_fps_at(60)
    }
    /// эффективный FPS полезных кадров при мониторе `hz` Гц:
    /// каждый кадр держится `frame_hold_periods` периодов обновления.
    pub fn effective_fps_at(&self, hz: u32) -> f32 {
        hz as f32 / self.frame_hold_periods as f32
    }
    /// ширина тихой зоны в ячейках (§3.1): пресеты 0..3 -> 2,4,6,8 ячеек.
    pub fn quiet_zone_cells(&self) -> u8 {
        2 * (self.quiet_zone + 1)
    }
    /// число бит хромы на ячейку для текущего `chroma_mode` (§5.1):
    /// Mono и GreenOnly несут 0 бит хромы, Chroma1..3 — 1..3 бита.
    pub fn chroma_bits(&self) -> u8 {
        match self.chroma_mode {
            ChromaMode::Mono | ChromaMode::GreenOnly => 0,
            ChromaMode::Chroma1 => 1,
            ChromaMode::Chroma2 => 2,
            ChromaMode::Chroma3 => 3,
        }
    }
    /// Период вставки repair-символов в источник, в исходных символах (§6.1).
    ///
    /// Поле `fec_overhead` задаёт поведение потока RaptorQ:
    /// * `0` — источник ×1, затем бесконечный repair; регулярного интервала
    ///   вставки нет -> `None`.
    /// * `1..=7` — repair подмешивается каждые `2^fec_overhead` исходных
    ///   символов -> `Some(2^fec_overhead)`.
    pub fn repair_interval_source_symbols(&self) -> Option<u32> {
        match self.fec_overhead {
            0 => None,
            v => Some(1u32 << v),
        }
    }

    fn validate(&self) -> Result<(), ProfileError> {
        if self.version != Self::VERSION {
            return Err(ProfileError::BadVersion(self.version));
        }
        if !(2..=65).contains(&self.cell_size_px) {
            return Err(ProfileError::BadField("cell_size_px"));
        }
        if !(1..=16).contains(&self.frame_hold_periods) {
            return Err(ProfileError::BadField("frame_hold_periods"));
        }
        if !(1..=4).contains(&self.luma_bits) {
            return Err(ProfileError::BadField("luma_bits"));
        }
        if self.gamma_g_q > 63
            || self.gamma_r_delta_q > 15
            || self.gamma_b_delta_q > 15
            || self.white_level_q > 15
            || self.black_level_q > 15
            || self.noise_sigma_q > 31
            || self.torn_frames_q > 15
            || self.crosstalk_rg_q > 15
            || self.crosstalk_gb_q > 15
            || self.quiet_zone > 3
            || self.fec_overhead > 7
        {
            return Err(ProfileError::BadField("quantized field out of range"));
        }
        if !(1..=32).contains(&self.mtf_limit_px) {
            return Err(ProfileError::BadField("mtf_limit_px"));
        }
        Ok(())
    }

    /// 72 бита полей + пустое место под CRC, в старших битах u128
    fn pack_fields(&self) -> u128 {
        let mut w = BitWriter::new();
        w.write(self.version as u32, 4);
        w.write((self.cell_size_px - 2) as u32, 6);
        w.write((self.frame_hold_periods - 1) as u32, 4);
        w.write((self.luma_bits - 1) as u32, 2);
        w.write(self.chroma_mode as u32, 3);
        w.write(self.gamma_g_q as u32, 6);
        w.write(self.gamma_r_delta_q as u32, 4);
        w.write(self.gamma_b_delta_q as u32, 4);
        w.write(self.white_level_q as u32, 4);
        w.write(self.black_level_q as u32, 4);
        w.write(self.noise_sigma_q as u32, 5);
        w.write((self.mtf_limit_px - 1) as u32, 5);
        w.write(self.torn_frames_q as u32, 4);
        w.write(self.crosstalk_rg_q as u32, 4);
        w.write(self.crosstalk_gb_q as u32, 4);
        w.write(self.quiet_zone as u32, 2);
        w.write(self.fec_overhead as u32, 3);
        w.write(0, 4); // reserved
        w.write(0, 8); // место CRC, заполним ниже
        w.finish()
    }

    /// payload = 72 бита полей + CRC-8 в младшем байте (итого 80 бит)
    fn to_payload(self) -> u128 {
        let fields = self.pack_fields();
        let field_bytes = fields_to_bytes(fields);
        let crc = crc8(&field_bytes);
        fields | crc as u128
    }

    fn from_payload(p: u128) -> Result<Self, ProfileError> {
        let crc_got = (p & 0xFF) as u8;
        let fields = p & !0xFFu128;
        let crc_want = crc8(&fields_to_bytes(fields));
        if crc_got != crc_want {
            return Err(ProfileError::CrcMismatch);
        }

        let mut r = BitReader::new(p);
        let version = r.read(4) as u8;
        if version != Self::VERSION {
            return Err(ProfileError::BadVersion(version));
        }
        let cell_size_px = r.read(6) as u8 + 2;
        let frame_hold_periods = r.read(4) as u8 + 1;
        let luma_bits = r.read(2) as u8 + 1;
        let chroma_raw = r.read(3);
        let chroma_mode =
            ChromaMode::from_raw(chroma_raw).ok_or(ProfileError::BadField("chroma_mode"))?;
        let gamma_g_q = r.read(6) as u8;
        let gamma_r_delta_q = r.read(4) as u8;
        let gamma_b_delta_q = r.read(4) as u8;
        let white_level_q = r.read(4) as u8;
        let black_level_q = r.read(4) as u8;
        let noise_sigma_q = r.read(5) as u8;
        let mtf_limit_px = r.read(5) as u8 + 1;
        let torn_frames_q = r.read(4) as u8;
        let crosstalk_rg_q = r.read(4) as u8;
        let crosstalk_gb_q = r.read(4) as u8;
        let quiet_zone = r.read(2) as u8;
        let fec_overhead = r.read(3) as u8;
        let _reserved = r.read(4); // 4 бита, игнорируем

        let p = Self {
            version,
            cell_size_px,
            frame_hold_periods,
            luma_bits,
            chroma_mode,
            gamma_g_q,
            gamma_r_delta_q,
            gamma_b_delta_q,
            white_level_q,
            black_level_q,
            noise_sigma_q,
            mtf_limit_px,
            torn_frames_q,
            crosstalk_rg_q,
            crosstalk_gb_q,
            quiet_zone,
            fec_overhead,
        };
        p.validate()?;
        Ok(p)
    }

    /// Профиль -> "XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX"
    pub fn encode_string(&self) -> Result<String, ProfileError> {
        let mut buf = [0u8; CODE_CHARS_GROUPED];
        self.encode_into(&mut buf)?;
        // buf гарантированно ASCII (алфавит + дефисы)
        Ok(String::from(
            core::str::from_utf8(&buf).expect("Base32 output is ASCII"),
        ))
    }

    /// Кодирование без аллокаций: пишет 39 ASCII-байт "XXXX-...-XXXX" в буфер.
    pub fn encode_into(&self, out: &mut [u8; CODE_CHARS_GROUPED]) -> Result<(), ProfileError> {
        self.validate()?;
        let payload = self.to_payload();
        let msg = crate::bits::payload_to_symbols(payload);
        let code = rs::encode_pair(&msg);
        let n = base32::format_grouped_into(&code, out);
        debug_assert_eq!(n, CODE_CHARS_GROUPED);
        Ok(())
    }

    /// Строка от человека -> профиль. Возвращает (профиль, число исправленных опечаток).
    ///
    /// Разбор входа идёт без аллокаций (в буфер фиксированного размера); память
    /// выделяет только внутренний RS-декодер.
    pub fn decode_string(input: &str) -> Result<(Self, usize), ProfileError> {
        let mut code = [0u8; rs::CODE_LEN];
        let n = base32::parse_into(input, &mut code).map_err(|e| match e {
            base32::ParseError::BadChar(i) => ProfileError::BadChar(i),
            // символов больше буфера (>32): длина заведомо неверна
            base32::ParseError::TooLong => ProfileError::BadLength(CODE_SYMBOLS + 1),
        })?;
        if n != CODE_SYMBOLS {
            return Err(ProfileError::BadLength(n));
        }
        let fixed = rs::decode_pair(&mut code).map_err(|_| ProfileError::Uncorrectable)?;
        let msg = rs::extract_message(&code);
        let payload = crate::bits::symbols_to_payload(&msg);
        let profile = Self::from_payload(payload)?;
        Ok((profile, fixed))
    }
}

/// 72 старших бита payload -> 9 байт big-endian (для CRC)
fn fields_to_bytes(fields: u128) -> [u8; 9] {
    let v = fields >> 8; // отбрасываем байт CRC
    let mut out = [0u8; 9];
    for (i, b) in out.iter_mut().enumerate() {
        *b = ((v >> (72 - 8 * (i + 1))) & 0xFF) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CalibProfile {
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: 16,
            frame_hold_periods: 6,
            luma_bits: 3,
            chroma_mode: ChromaMode::Chroma2,
            gamma_g_q: 28, // 2.2
            gamma_r_delta_q: 8,
            gamma_b_delta_q: 10,
            white_level_q: 15, // 100%
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

    #[test]
    fn string_roundtrip() {
        let p = sample();
        let s = p.encode_string().unwrap();
        assert_eq!(s.len(), CODE_CHARS_GROUPED, "{s}");
        let (q, fixed) = CalibProfile::decode_string(&s).unwrap();
        assert_eq!(fixed, 0);
        assert_eq!(p, q);
    }

    #[test]
    fn physical_values() {
        let p = sample();
        assert!((p.gamma_g() - 2.2).abs() < 1e-6);
        assert!((p.gamma_b() - 2.25).abs() < 1e-6);
        assert_eq!(p.white_level_pct(), 100);
        assert!((p.effective_fps() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn helper_fields() {
        let p = sample();
        // 60 Гц -> 10 fps (hold 6); произвольная частота обобщается
        assert!((p.effective_fps_at(120) - 20.0).abs() < 1e-6);
        assert!((p.effective_fps_at(60) - p.effective_fps()).abs() < 1e-6);
        // quiet_zone=1 -> 4 ячейки (§3.1)
        assert_eq!(p.quiet_zone_cells(), 4);
        // Chroma2 -> 2 бита хромы (§5.1)
        assert_eq!(p.chroma_bits(), 2);
        // fec_overhead=2 -> repair каждые 4 исходных символа (§6.1)
        assert_eq!(p.repair_interval_source_symbols(), Some(4));
        // граничные пресеты хромы/тихой зоны/fec
        let mut q = p;
        q.chroma_mode = ChromaMode::GreenOnly;
        assert_eq!(q.chroma_bits(), 0);
        q.chroma_mode = ChromaMode::Mono;
        assert_eq!(q.chroma_bits(), 0);
        q.chroma_mode = ChromaMode::Chroma3;
        assert_eq!(q.chroma_bits(), 3);
        q.quiet_zone = 3;
        assert_eq!(q.quiet_zone_cells(), 8);
        q.fec_overhead = 0;
        assert_eq!(q.repair_interval_source_symbols(), None);
        q.fec_overhead = 7;
        assert_eq!(q.repair_interval_source_symbols(), Some(128));
    }

    /// Замороженный формат: точная эталонная строка из §7.4 SPEC. Падение этого
    /// теста означает, что проводной формат изменился (несовместимость).
    const REFERENCE_CODE: &str = "26E2-BM46-VHH8-B6R3-8XP4-HBNK-PJCD-GHF7";

    #[test]
    fn frozen_wire_format() {
        let p = sample();
        // 1. кодирование эталонного профиля даёт точную эталонную строку
        assert_eq!(p.encode_string().unwrap(), REFERENCE_CODE);
        // 2. канонический декод возвращает тот же профиль без исправлений
        let (q, fixed) = CalibProfile::decode_string(REFERENCE_CODE).unwrap();
        assert_eq!(fixed, 0);
        assert_eq!(p, q);
        // 3. нижний регистр + мусорные пробелы (таб, перевод строки, NBSP,
        //    ведущие/хвостовые) декодируются в тот же профиль
        let mangled = "  26e2 bm46\tvhh8\nb6r3\u{00A0}8xp4-hbnk pjcd ghf7  ";
        let (r, fixed2) = CalibProfile::decode_string(mangled).unwrap();
        assert_eq!(fixed2, 0);
        assert_eq!(p, r);
    }

    #[test]
    fn encode_into_matches_encode_string() {
        let p = sample();
        let mut buf = [0u8; CODE_CHARS_GROUPED];
        p.encode_into(&mut buf).unwrap();
        assert_eq!(&buf, REFERENCE_CODE.as_bytes());
    }

    #[test]
    fn encode_rejects_wrong_version() {
        let mut p = sample();
        p.version = 2;
        assert_eq!(p.encode_string(), Err(ProfileError::BadVersion(2)));
        let mut buf = [0u8; CODE_CHARS_GROUPED];
        assert_eq!(p.encode_into(&mut buf), Err(ProfileError::BadVersion(2)));
    }

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

    /// Широкий пул символов: алфавит, конфузаблы, разделители, юникод-пробелы,
    /// мусор и управляющие — всё, чем человек может «промахнуться».
    const FUZZ_POOL: &[char] = &[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
        'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'a', 'o', 'i', 's',
        'z', 'O', 'I', 'S', 'Z', '-', '\u{2013}', '\u{2014}', ' ', '\t', '\n', '\r', '\u{00A0}',
        '\u{2003}', '\u{3000}', '*', '#', '@', '.', '/', '+', '=', '?', 'é', 'ß', 'ψ', '中', '\0',
    ];

    fn fuzz_decode(iterations: usize) {
        let mut rng = XorShift64(0x1234_5678_9ABC_DEF0);
        for _ in 0..iterations {
            let len = (rng.next() % 65) as usize; // 0..=64
            let mut s = String::new();
            for _ in 0..len {
                s.push(FUZZ_POOL[(rng.next() as usize) % FUZZ_POOL.len()]);
            }
            // Контракт: путь декодирования НИКОГДА не паникует; любой исход —
            // строго Ok или Err (определённая ошибка). При случайном Ok профиль
            // обязан быть самосогласованным (перекодируется в валидную строку).
            if let Ok((profile, _)) = CalibProfile::decode_string(&s) {
                let re = profile.encode_string().expect("decoded profile must re-encode");
                let (again, _) = CalibProfile::decode_string(&re).expect("re-decode");
                assert_eq!(profile, again);
            }
        }
    }

    #[test]
    fn decode_never_panics_on_random_input() {
        // держим быстрым для обычного `cargo test`
        fuzz_decode(100_000);
    }

    #[test]
    #[ignore = "тяжёлый прогон; запускать через `cargo test -- --ignored`"]
    fn decode_never_panics_on_random_input_heavy() {
        fuzz_decode(2_000_000);
    }

    #[test]
    fn tolerates_four_typos_and_sloppy_input() {
        let p = sample();
        let s = p.encode_string().unwrap();
        // портим 4 символа (не дефисы): подменяем на символ с гарантированно
        // другим значением с учётом подстановок парсера
        let mut chars: Vec<char> = s.chars().collect();
        let mut broken = 0;
        for i in [0usize, 6, 21, 38] {
            let orig = chars[i];
            assert_ne!(orig, '-');
            let orig_val = crate::base32::char_to_symbol(orig).unwrap();
            let subst_val = (orig_val + 16) % 32;
            chars[i] = crate::base32::symbol_to_char(subst_val);
            broken += 1;
        }
        // и вводим неряшливо: нижний регистр, пробелы вместо дефисов
        let sloppy: String = chars
            .iter()
            .map(|&c| if c == '-' { ' ' } else { c.to_ascii_lowercase() })
            .collect();
        let (q, fixed) = CalibProfile::decode_string(&sloppy).unwrap();
        assert_eq!(fixed, broken);
        assert_eq!(p, q);
    }

    #[test]
    fn recovers_every_adjacent_group_pair() {
        // Для каждой соседней пары групп (0-1, 1-2, ..., 6-7) целиком портим
        // обе группы (8 символов подряд): перемежение 4/4 делит серию между
        // словами A и B, и код обязан восстановиться, исправив ровно 8 символов.
        let p = sample();
        let s = p.encode_string().unwrap();
        for g in 0..7usize {
            let mut chars: Vec<char> = s.chars().collect();
            for grp in [g, g + 1] {
                for k in 0..base32::GROUP {
                    let idx = grp * (base32::GROUP + 1) + k; // +1 на дефис между группами
                    let orig = chars[idx];
                    assert_ne!(orig, '-', "pair {g}: idx {idx}");
                    let v = base32::char_to_symbol(orig).unwrap();
                    chars[idx] = base32::symbol_to_char((v + 16) % 32);
                }
            }
            let garbled: String = chars.into_iter().collect();
            let (q, fixed) = CalibProfile::decode_string(&garbled).unwrap();
            assert_eq!(fixed, 8, "pair {g},{}", g + 1);
            assert_eq!(p, q, "pair {g},{}", g + 1);
        }
    }

    #[test]
    fn rejects_wrong_length_and_garbage() {
        assert!(matches!(
            CalibProfile::decode_string("ABCDE"),
            Err(ProfileError::BadLength(5))
        ));
        assert!(matches!(
            CalibProfile::decode_string("ABCD-ABCD-ABCD-ABCD-ABCD-ABCD-ABCD-ABC*"),
            Err(ProfileError::BadChar(_))
        ));
    }

    #[test]
    fn heavy_corruption_never_slips_through_silently() {
        // 9+ опечаток: минимум одно RS-слово получает >=5 ошибок — вне гарантии.
        // Допустимые исходы — любая честная ошибка (Uncorrectable / CrcMismatch /
        // BadVersion / BadField). Недопустимый — "успех" с другим профилем.
        let p = sample();
        let clean = p.encode_string().unwrap();
        let alphabet: Vec<char> = (0..32).map(base32::symbol_to_char).collect();
        let mut slipped = 0;
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rand = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as usize) % n
        };
        for _ in 0..300 {
            let mut chars: Vec<char> = clean.chars().collect();
            let positions: Vec<usize> =
                (0..chars.len()).filter(|&i| chars[i] != '-').collect();
            let mut chosen = std::collections::HashSet::new();
            while chosen.len() < 9 {
                chosen.insert(positions[rand(positions.len())]);
            }
            for &i in &chosen {
                loop {
                    let c = alphabet[rand(32)];
                    if c != chars[i] {
                        chars[i] = c;
                        break;
                    }
                }
            }
            let s: String = chars.into_iter().collect();
            if let Ok((q, _)) = CalibProfile::decode_string(&s) {
                if q != p {
                    slipped += 1;
                }
            }
        }
        assert_eq!(slipped, 0, "corrupted code decoded as different profile");
    }
}
