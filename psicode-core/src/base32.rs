//! Base32 с алфавитом eck-экосистемы: исключены I, O, S, Z.
//!
//! Парсер терпим к человеку: регистр не важен, `o/O -> 0`, `i/I -> 1`,
//! `s/S -> 5`, `z/Z -> 2`, дефисы и пробелы игнорируются.

pub const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKLMNPQRTUVWXY";

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

/// Строка -> символы; разделители пропускаются, мусорные символы -> Err с позицией.
pub fn parse(input: &str) -> Result<Vec<u8>, usize> {
    let mut out = Vec::new();
    for (i, c) in input.chars().enumerate() {
        if c == '-' || c == ' ' || c == '\u{2013}' || c == '\u{2014}' {
            continue;
        }
        match char_to_symbol(c) {
            Some(s) => out.push(s),
            None => return Err(i),
        }
    }
    Ok(out)
}

/// Размер группы при отображении человеку.
pub const GROUP: usize = 5;

/// Символы -> строка группами по 5 через дефис: XXXXX-XXXXX-...
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
        let syms: Vec<u8> = (0..40).map(|i| (i * 7 % 32) as u8).collect();
        let s = format_grouped(&syms);
        assert_eq!(s.split('-').count(), 8);
        assert!(s.split('-').all(|g| g.len() == GROUP));
        assert_eq!(parse(&s).unwrap(), syms);
        assert_eq!(parse(&s.to_lowercase()).unwrap(), syms);
    }
}
