//! Вывод: P6 PPM (свой писатель, без зависимостей) и форматирование markdown.

use crate::image::Image;
use std::io;
use std::path::Path;

/// Записать бинарный P6 PPM: заголовок `P6\n{w} {h}\n255\n` + сырые RGB-байты.
/// `pixels` — ровно `w·h` элементов построчно.
pub fn write_ppm(path: &Path, w: usize, h: usize, pixels: &[[u8; 3]]) -> io::Result<()> {
    assert_eq!(pixels.len(), w * h, "ppm: размер буфера не совпал с w·h");
    let mut buf = Vec::with_capacity(32 + w * h * 3);
    buf.extend_from_slice(format!("P6\n{w} {h}\n255\n").as_bytes());
    for px in pixels {
        buf.extend_from_slice(px);
    }
    std::fs::write(path, &buf)
}

/// Линейное изображение [0,1] -> drive-байты через ОБРАТНУЮ гамму канала
/// (для человеческого глаза: чистый канал так восстанавливает исходный кадр).
pub fn image_to_drive(img: &Image, gammas: [f64; 3]) -> Vec<[u8; 3]> {
    img.data
        .iter()
        .map(|p| {
            let mut out = [0u8; 3];
            for c in 0..3 {
                let lin = (p[c] as f64).clamp(0.0, 1.0);
                let drive = lin.powf(1.0 / gammas[c]);
                out[c] = (drive * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            out
        })
        .collect()
}

/// Число с 4 значащими цифрами (для ячеек таблиц SER). Ноль -> "0".
pub fn sig4(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    let dec = (3 - exp).clamp(0, 12) as usize;
    format!("{x:.dec$}")
}

/// Строка markdown-таблицы: `| {label} | {каждое значение} |`.
pub fn table_row(label: &str, cells: &[String]) -> String {
    let mut s = String::from("| ");
    s.push_str(label);
    for c in cells {
        s.push_str(" | ");
        s.push_str(c);
    }
    s.push_str(" |");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_header_and_size_are_correct() {
        let w = 3;
        let h = 2;
        let pixels = vec![[1u8, 2, 3]; w * h];
        let path = std::env::temp_dir().join("psicode_sim_ppm_test.ppm");
        write_ppm(&path, w, h, &pixels).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let header = b"P6\n3 2\n255\n";
        assert!(bytes.starts_with(header), "bad header");
        // заголовок + w·h·3 байта данных
        assert_eq!(bytes.len(), header.len() + w * h * 3);
        // первый пиксель данных
        assert_eq!(&bytes[header.len()..header.len() + 3], &[1, 2, 3]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sig4_gives_four_significant_digits() {
        assert_eq!(sig4(0.0), "0");
        assert_eq!(sig4(0.5), "0.5000");
        assert_eq!(sig4(0.1234567), "0.1235");
        assert_eq!(sig4(0.001234567), "0.001235");
    }

    #[test]
    fn drive_roundtrips_through_gamma() {
        // линейное значение, поднятое в 1/γ и упакованное в байт, близко к
        // исходному drive^γ (проверяем середину шкалы)
        let gammas = [2.2, 2.2, 2.2];
        let lin = (128.0f64 / 255.0).powf(2.2) as f32;
        let img = Image::filled(1, 1, [lin, lin, lin]);
        let drive = image_to_drive(&img, gammas);
        // обратно должно получиться ~128
        assert!((drive[0][0] as i32 - 128).abs() <= 1);
    }
}
