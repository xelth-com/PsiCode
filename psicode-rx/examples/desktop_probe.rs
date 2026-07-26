//! Прогон ТЕЛЕФОННОГО тракта (RxSession: даунскейл-детекция + capped detect +
//! YUV) на настольном PPM-снимке: диагностика без телефона в петле.
//! usage: desktop_probe <file.ppm> [cell]

use psicode_rx::{RxSession, tx_default_profile};
use std::fs;

/// Минимальный P6-ридер (как в psicode-sim/report.rs).
fn read_ppm(path: &str) -> (usize, usize, Vec<[u8; 3]>) {
    let data = fs::read(path).expect("read ppm");
    let mut pos = 0usize;
    let mut token = || {
        while pos < data.len() && (data[pos] as char).is_whitespace() {
            pos += 1;
        }
        if pos < data.len() && data[pos] == b'#' {
            while pos < data.len() && data[pos] != b'\n' {
                pos += 1;
            }
        }
        let start = pos;
        while pos < data.len() && !(data[pos] as char).is_whitespace() {
            pos += 1;
        }
        String::from_utf8_lossy(&data[start..pos]).into_owned()
    };
    assert_eq!(token(), "P6");
    let w: usize = token().parse().unwrap();
    let h: usize = token().parse().unwrap();
    let _max: usize = token().parse().unwrap();
    let px_start = pos + 1;
    let mut out = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let o = px_start + 3 * i;
        out.push([data[o], data[o + 1], data[o + 2]]);
    }
    (w, h, out)
}

/// RGB -> BT.601 limited-range YUV420 (двухплоскостный, uv_pixel_stride=1).
fn rgb_to_yuv420(w: usize, h: usize, rgb: &[[u8; 3]]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; w * h];
    let cw = w / 2;
    let ch = h / 2;
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..h {
        for i in 0..w {
            let [r, g, b] = rgb[j * w + i];
            let (r, g, b) = (r as f64, g as f64, b as f64);
            let yy = 16.0 + (65.738 * r + 129.057 * g + 25.064 * b) / 256.0;
            y[j * w + i] = yy.round().clamp(0.0, 255.0) as u8;
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            let [r, g, b] = rgb[(j * 2) * w + i * 2];
            let (r, g, b) = (r as f64, g as f64, b as f64);
            let uu = 128.0 + (-37.945 * r - 74.494 * g + 112.439 * b) / 256.0;
            let vv = 128.0 + (112.439 * r - 94.154 * g - 18.285 * b) / 256.0;
            u[j * cw + i] = uu.round().clamp(0.0, 255.0) as u8;
            v[j * cw + i] = vv.round().clamp(0.0, 255.0) as u8;
        }
    }
    (y, u, v)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: desktop_probe <file.ppm|dump-prefix> [cell]");
    let cell: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    // режим YUV-дампов: префиксы через запятую -> одна сессия, кадры по очереди
    if !path.ends_with(".ppm") {
        let mut p = tx_default_profile();
        p.cell_size_px = cell;
        let mut s = RxSession::new(p);
        for prefix in path.split(',') {
            let meta = std::fs::read_to_string(format!("{prefix}.meta")).expect("meta");
            let m: Vec<usize> = meta.split_whitespace().map(|t| t.parse().unwrap()).collect();
            let (w, h, ys, uvs, uvp) = (m[0], m[1], m[2], m[3], m[4]);
            let y = std::fs::read(format!("{prefix}.y")).unwrap();
            let u = std::fs::read(format!("{prefix}.u")).unwrap();
            let v = std::fs::read(format!("{prefix}.v")).unwrap();
            for i in 0..2 {
                let t0 = std::time::Instant::now();
                let st = s.process_frame_yuv(&y, &u, &v, w, h, ys, uvs, uvp);
                println!("  [{prefix} #{i}] {:.0}ms {}", t0.elapsed().as_secs_f64()*1000.0, st.to_json());
            }
        }
        return;
    }
    let (w, h, rgb) = read_ppm(path);
    println!("probe: {path} ({w}x{h}), cell {cell}");
    let (y, u, v) = rgb_to_yuv420(w, h, &rgb);
    let mut p = tx_default_profile();
    p.cell_size_px = cell;
    let mut s = RxSession::new(p);
    // Один И ТОТ ЖЕ кадр N раз в ОДНУ сессию: кадр 1 -> ЗАХВАТ (acquire),
    // кадры 2..N -> СЛЕЖЕНИЕ (track) от геометрии прошлого кадра. Стоимость
    // acquire (раз на лок) заметно выше track — видно по мс.
    for i in 0..3 {
        let t0 = std::time::Instant::now();
        let st = s.process_frame_yuv(&y, &u, &v, w, h, w, w / 2, 1);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let kind = if i == 0 { "acquire" } else { "track  " };
        println!("  [{i}] {kind} {ms:6.0} ms  {}", st.to_json());
    }
}
