//! Калибровочный профиль канала «монитор -> камера».
//!
//! Телефон измеряет канал по одному тестовому кадру, упаковывает результат в
//! 120-битный payload (112 бит полей + CRC-8), кодирует двумя перемежёнными
//! RS(20,12) над GF(32) и показывает человеку 40 символов Crockford Base32
//! (200 бит, 8 групп по 5):
//!
//! ```text
//! XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX
//! ```
//!
//! Человек вводит их на передатчике. RS гарантированно исправляет любые
//! 4 опечатки, а благодаря перемежению — и до 8 подряд идущих (целая
//! испорченная группа из 5 — с запасом); CRC-8 страхует от мискоррекции при
//! большем числе ошибок.
//!
//! Раскладка 112 бит (по порядку записи, старшие биты первыми):
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
//! | reserved            | 44  | нули; поле для будущих версий                     |
//! | ----- итого         | 112 |                                                   |
//! | crc8                |  8  | CRC-8 (poly 0x07) по 14 байтам полей              |

use crate::base32;
use crate::bits::{BitReader, BitWriter};
use crate::rs;

pub const CODE_SYMBOLS: usize = rs::CODE_LEN; // 40
pub const CODE_CHARS_GROUPED: usize = CODE_SYMBOLS + CODE_SYMBOLS / base32::GROUP - 1; // 47 с дефисами

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
    /// длина не равна 40 символам
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
    pub fn noise_sigma(&self) -> f32 {
        0.25 * (2f32).powf(self.noise_sigma_q as f32 / 4.0)
    }
    /// доля рваных кадров, %: 0 -> 0%, иначе 0.1 * 2^(q-1) %, потолок 100
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
    pub fn effective_fps(&self) -> f32 {
        60.0 / self.frame_hold_periods as f32
    }

    fn validate(&self) -> Result<(), ProfileError> {
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

    /// 112 бит полей (без CRC), в старших битах u128
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
        w.write(0, 44); // reserved
        w.write(0, 8); // место CRC, заполним ниже
        w.finish()
    }

    /// payload = 112 бит полей + CRC-8 в младшем байте
    fn to_payload(&self) -> u128 {
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
        let _reserved = r.read(32) as u64 | ((r.read(12) as u64) << 32); // 44 бита, игнорируем

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

    /// Профиль -> "XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX"
    pub fn encode_string(&self) -> Result<String, ProfileError> {
        self.validate()?;
        let payload = self.to_payload();
        let msg = crate::bits::payload_to_symbols(payload);
        let code = rs::encode_pair(&msg);
        Ok(base32::format_grouped(&code))
    }

    /// Строка от человека -> профиль. Возвращает (профиль, число исправленных опечаток).
    pub fn decode_string(input: &str) -> Result<(Self, usize), ProfileError> {
        let symbols = base32::parse(input).map_err(ProfileError::BadChar)?;
        if symbols.len() != CODE_SYMBOLS {
            return Err(ProfileError::BadLength(symbols.len()));
        }
        let mut code = [0u8; rs::CODE_LEN];
        code.copy_from_slice(&symbols);
        let fixed = rs::decode_pair(&mut code).map_err(|_| ProfileError::Uncorrectable)?;
        let msg = rs::extract_message(&code);
        let payload = crate::bits::symbols_to_payload(&msg);
        let profile = Self::from_payload(payload)?;
        Ok((profile, fixed))
    }
}

/// 112 старших бит u128 -> 14 байт big-endian (для CRC)
fn fields_to_bytes(fields: u128) -> [u8; 14] {
    let mut out = [0u8; 14];
    for (i, b) in out.iter_mut().enumerate() {
        *b = ((fields >> (112 - 8 * (i + 1) + 8)) & 0xFF) as u8;
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
    fn tolerates_four_typos_and_sloppy_input() {
        let p = sample();
        let s = p.encode_string().unwrap();
        // портим 4 символа (не дефисы): подменяем на символ с гарантированно
        // другим значением с учётом подстановок парсера
        let mut chars: Vec<char> = s.chars().collect();
        let mut broken = 0;
        for i in [0usize, 13, 27, 43] {
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
    fn tolerates_fully_garbled_group() {
        // вторая группа (символы 6..=10 в строке) испорчена целиком: 5 подряд
        // опечаток делятся перемежением 3/2 между словами A и B
        let p = sample();
        let s = p.encode_string().unwrap();
        let mut chars: Vec<char> = s.chars().collect();
        for i in 6..=10usize {
            let orig = chars[i];
            assert_ne!(orig, '-');
            let orig_val = crate::base32::char_to_symbol(orig).unwrap();
            chars[i] = crate::base32::symbol_to_char((orig_val + 16) % 32);
        }
        let garbled: String = chars.into_iter().collect();
        let (q, fixed) = CalibProfile::decode_string(&garbled).unwrap();
        assert_eq!(fixed, 5);
        assert_eq!(p, q);
    }

    #[test]
    fn rejects_wrong_length_and_garbage() {
        assert!(matches!(
            CalibProfile::decode_string("ABCDE"),
            Err(ProfileError::BadLength(5))
        ));
        assert!(matches!(
            CalibProfile::decode_string("ABCDE-ABCDE-ABCDE-ABCDE-ABCDE-ABCDE-ABCDE-ABCD*"),
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
