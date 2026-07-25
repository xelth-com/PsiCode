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

/// Пропустить пробелы и комментарии `# ... \n` начиная с позиции `i`.
fn skip_ws_comments(data: &[u8], i: &mut usize) {
    loop {
        while *i < data.len() && data[*i].is_ascii_whitespace() {
            *i += 1;
        }
        if *i < data.len() && data[*i] == b'#' {
            while *i < data.len() && data[*i] != b'\n' {
                *i += 1;
            }
        } else {
            break;
        }
    }
}

/// Прочитать целое (с ведущими пробелами/комментариями). Курсор — за последней
/// цифрой. `None`, если цифр нет.
fn read_uint(data: &[u8], i: &mut usize) -> Option<usize> {
    skip_ws_comments(data, i);
    let start = *i;
    while *i < data.len() && data[*i].is_ascii_digit() {
        *i += 1;
    }
    if *i == start {
        return None;
    }
    let mut v = 0usize;
    for &b in &data[start..*i] {
        v = v * 10 + (b - b'0') as usize;
    }
    Some(v)
}

/// Прочитать бинарный P6 PPM с диска: `P6`, w, h, 255, затем сырые RGB-триплеты.
/// Терпит пробелы и комментарии `#` в заголовке. Возвращает (w, h, пиксели).
pub fn read_ppm(path: &Path) -> io::Result<(usize, usize, Vec<[u8; 3]>)> {
    let data = std::fs::read(path)?;
    let err = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());

    let mut i = 0usize;
    skip_ws_comments(&data, &mut i);
    if data.get(i) != Some(&b'P') || data.get(i + 1) != Some(&b'6') {
        return Err(err("ppm: не P6"));
    }
    i += 2;

    let w = read_uint(&data, &mut i).ok_or_else(|| err("ppm: нет ширины"))?;
    let h = read_uint(&data, &mut i).ok_or_else(|| err("ppm: нет высоты"))?;
    let maxval = read_uint(&data, &mut i).ok_or_else(|| err("ppm: нет maxval"))?;
    if maxval != 255 {
        return Err(err("ppm: maxval != 255"));
    }
    // ровно один пробельный байт-разделитель перед бинарными данными
    if i >= data.len() || !data[i].is_ascii_whitespace() {
        return Err(err("ppm: нет разделителя перед данными"));
    }
    i += 1;

    let need = w * h * 3;
    if data.len() - i < need {
        return Err(err("ppm: данные обрезаны"));
    }
    let body = &data[i..i + need];
    let mut px = Vec::with_capacity(w * h);
    for chunk in body.chunks_exact(3) {
        px.push([chunk[0], chunk[1], chunk[2]]);
    }
    Ok((w, h, px))
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
    fn ppm_write_then_read_roundtrips() {
        let w = 5;
        let h = 3;
        let mut pixels = Vec::with_capacity(w * h);
        for i in 0..(w * h) {
            pixels.push([(i * 3) as u8, (i * 3 + 1) as u8, (i * 3 + 2) as u8]);
        }
        let path = std::env::temp_dir().join("psicode_sim_ppm_roundtrip.ppm");
        write_ppm(&path, w, h, &pixels).unwrap();

        let (rw, rh, rpix) = read_ppm(&path).unwrap();
        assert_eq!(rw, w);
        assert_eq!(rh, h);
        assert_eq!(rpix, pixels);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ppm_reader_tolerates_comments() {
        // заголовок с комментарием и лишними пробелами
        let path = std::env::temp_dir().join("psicode_sim_ppm_comment.ppm");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"P6\n# psicode-sim comment\n2  1\n255\n");
        bytes.extend_from_slice(&[10, 20, 30, 40, 50, 60]);
        std::fs::write(&path, &bytes).unwrap();

        let (w, h, pix) = read_ppm(&path).unwrap();
        assert_eq!((w, h), (2, 1));
        assert_eq!(pix, vec![[10, 20, 30], [40, 50, 60]]);

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
