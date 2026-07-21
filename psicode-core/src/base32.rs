//! Base32 с алфавитом eck-экосистемы: исключены I, O, S, Z.
//!
//! Парсер терпим к человеку: регистр не важен, `o/O -> 0`, `i/I -> 1`,
//! `s/S -> 5`, `z/Z -> 2`; дефисы, en/em-тире и ЛЮБОЙ пробельный символ
//! (пробел, таб, перевод строки, NBSP, юникод-пробелы) игнорируются.

use alloc::{string::String, vec::Vec};

pub const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKLMNPQRTUVWXY";

/// Ошибка терпимого разбора в буфер фиксированного размера.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// Недопустимый символ (позиция в исходной строке).
    BadChar(usize),
    /// Значимых символов больше, чем помещается в выходной буфер.
    TooLong,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::BadChar(i) => write!(f, "invalid character at position {i}"),
            ParseError::TooLong => f.write_str("more symbols than the output buffer holds"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

pub fn symbol_to_char(s: u8) -> char {
    debug_assert!(s < 32);
    ALPHABET[s as usize] as char
}

pub fn char_to_symbol(c: char) -> Option<u8> {
    let c = c.to_ascii_uppercase();
    match c {
        'O' => Some(0),
        'I' => Some(1),
        'S' => Some(5),
        'Z' => Some(2),
        _ => ALPHABET.iter().position(|&a| a as char == c).map(|p| p as u8),
    }
}

/// Разделитель, который человек может вставить: дефис, en/em-тире или любой
/// пробельный символ (включая таб, перевод строки, NBSP и прочие юникод-пробелы).
#[inline]
fn is_separator(c: char) -> bool {
    c == '-' || c == '\u{2013}' || c == '\u{2014}' || c.is_whitespace()
}

/// Строка -> символы; разделители пропускаются, мусорные символы -> Err с позицией.
pub fn parse(input: &str) -> Result<Vec<u8>, usize> {
    let mut out = Vec::new();
    for (i, c) in input.chars().enumerate() {
        if is_separator(c) {
            continue;
        }
        match char_to_symbol(c) {
            Some(s) => out.push(s),
            None => return Err(i),
        }
    }
    Ok(out)
}

/// Терпимый разбор во внешний буфер без аллокаций.
/// Возвращает число записанных символов либо ошибку.
pub fn parse_into(input: &str, out: &mut [u8]) -> Result<usize, ParseError> {
    let mut n = 0;
    for (i, c) in input.chars().enumerate() {
        if is_separator(c) {
            continue;
        }
        match char_to_symbol(c) {
            Some(s) => {
                if n >= out.len() {
                    return Err(ParseError::TooLong);
                }
                out[n] = s;
                n += 1;
            }
            None => return Err(ParseError::BadChar(i)),
        }
    }
    Ok(n)
}

/// Размер группы при отображении человеку.
pub const GROUP: usize = 4;

/// Символы -> строка группами по 4 через дефис: XXXX-XXXX-...
pub fn format_grouped(symbols: &[u8]) -> String {
    let mut s = String::with_capacity(symbols.len() + symbols.len() / GROUP);
    for (i, &sym) in symbols.iter().enumerate() {
        if i > 0 && i % GROUP == 0 {
            s.push('-');
        }
        s.push(symbol_to_char(sym));
    }
    s
}

/// То же, что [`format_grouped`], но пишет ASCII-байты во внешний буфер без
/// аллокаций. Возвращает число записанных байт. Буфер должен вмещать
/// `symbols.len() + symbols.len()/GROUP` байт (для 32 символов — 39).
pub fn format_grouped_into(symbols: &[u8], out: &mut [u8]) -> usize {
    let mut n = 0;
    for (i, &sym) in symbols.iter().enumerate() {
        if i > 0 && i % GROUP == 0 {
            out[n] = b'-';
            n += 1;
        }
        out[n] = symbol_to_char(sym) as u8;
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_substitutions() {
        assert_eq!(char_to_symbol('o'), Some(0));
        assert_eq!(char_to_symbol('O'), Some(0));
        assert_eq!(char_to_symbol('i'), Some(1));
        assert_eq!(char_to_symbol('s'), Some(5));
        assert_eq!(char_to_symbol('z'), Some(2));
        assert_eq!(char_to_symbol('a'), Some(10));
        assert_eq!(char_to_symbol('L'), Some(20)); // L входит в алфавит
        assert_eq!(char_to_symbol('*'), None);
    }

    #[test]
    fn roundtrip_and_grouping() {
        let syms: Vec<u8> = (0..32).map(|i| (i * 7 % 32) as u8).collect();
        let s = format_grouped(&syms);
        assert_eq!(s.split('-').count(), 8);
        assert!(s.split('-').all(|g| g.len() == GROUP));
        assert_eq!(parse(&s).unwrap(), syms);
        assert_eq!(parse(&s.to_lowercase()).unwrap(), syms);
    }
}
