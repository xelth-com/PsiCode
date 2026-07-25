//! Тракт канала и одна попытка Monte Carlo.
//!
//! Порядок стадий (§9.1 SPEC): drive u8 -> emitted linear (v/255)^γ -> warp в
//! сетку камеры (inverse-map + билинейка) -> сепарабельный гауссов блюр σ ->
//! кросстолк -> усиление/смещение -> аддитивный гауссов шум -> clamp [0,1].
//! Дальше genie-`map` = [`Geometry::forward`], genie-`sample` = билинейка по
//! финальному изображению отдаются в `demod_symbol`.

use crate::channel::{ChannelParams, Geometry};
use crate::image::Image;
use crate::rng::Rng;
use psicode_core::symbol::{self, Frame};
use psicode_core::CalibProfile;

/// drive-кадр -> emitted linear per channel через 256-элементные LUT гамм.
/// (drive ∈ 0..255 целочисленный, так что гамма считается таблицей.)
pub fn emit_linear(frame: &Frame, gammas: [f64; 3]) -> Image {
    let mut lut = [[0.0f32; 256]; 3];
    for (c, &g) in gammas.iter().enumerate() {
        for v in 0..256 {
            lut[c][v] = (v as f64 / 255.0).powf(g) as f32;
        }
    }
    let n = frame.size_px;
    let mut img = Image::new(n, n);
    for (dst, px) in img.data.iter_mut().zip(frame.rgb.iter()) {
        *dst = [
            lut[0][px[0] as usize],
            lut[1][px[1] as usize],
            lut[2][px[2] as usize],
        ];
    }
    img
}

/// drive-байты с диска -> сенсорно-линейное изображение: linear = (d/255)^γ
/// поканально (LUT). Обращает запись PPM (`image_to_drive` возводил в 1/γ):
/// это и есть «идеальный сенсор» для readback уже квантованных файлов.
pub fn drive_to_linear(drive: &[[u8; 3]], w: usize, h: usize, gammas: [f64; 3]) -> Image {
    let mut lut = [[0.0f32; 256]; 3];
    for (c, &g) in gammas.iter().enumerate() {
        for v in 0..256 {
            lut[c][v] = (v as f64 / 255.0).powf(g) as f32;
        }
    }
    let mut img = Image::new(w, h);
    for (dst, d) in img.data.iter_mut().zip(drive.iter()) {
        *dst = [lut[0][d[0] as usize], lut[1][d[1] as usize], lut[2][d[2] as usize]];
    }
    img
}

/// Warp дисплейного изображения в сетку камеры обратным отображением: для
/// каждого выходного пикселя берём координату символа через [`Geometry::inverse`]
/// и билинейно сэмплируем источник.
pub fn warp(disp: &Image, geom: &Geometry) -> Image {
    let mut out = Image::new(geom.out_w, geom.out_h);
    for y in 0..geom.out_h {
        for x in 0..geom.out_w {
            let (u, v) = geom.inverse(x as f64, y as f64);
            out.set(x, y, disp.sample(u, v));
        }
    }
    out
}

/// Нормированное 1-D гауссово ядро, радиус ceil(3σ) (сумма весов = 1).
/// При σ ≤ 0 — вырожденное ядро [1.0] (тождество).
pub fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let r = (3.0 * sigma).ceil() as usize;
    let mut k = Vec::with_capacity(2 * r + 1);
    let mut sum = 0.0;
    for i in 0..=(2 * r) {
        let x = i as f64 - r as f64;
        let w = (-(x * x) / (2.0 * sigma * sigma)).exp();
        k.push(w);
        sum += w;
    }
    for w in &mut k {
        *w /= sum;
    }
    k
}

#[inline]
fn clamp_idx(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// Сепарабельный гауссов блюр σ (в px камеры), clamp-to-edge на границах.
/// При σ ≤ 0 возвращает копию без изменений.
pub fn blur(img: &Image, sigma: f64) -> Image {
    if sigma <= 0.0 {
        return img.clone();
    }
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as isize;

    // горизонтальный проход
    let mut tmp = Image::new(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            let mut acc = [0.0f64; 3];
            for (ki, &w) in k.iter().enumerate() {
                let sx = clamp_idx(x as isize + ki as isize - r, img.w);
                let p = img.at(sx, y);
                for c in 0..3 {
                    acc[c] += w * p[c] as f64;
                }
            }
            tmp.set(x, y, [acc[0] as f32, acc[1] as f32, acc[2] as f32]);
        }
    }

    // вертикальный проход
    let mut out = Image::new(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            let mut acc = [0.0f64; 3];
            for (ki, &w) in k.iter().enumerate() {
                let sy = clamp_idx(y as isize + ki as isize - r, img.h);
                let p = tmp.at(x, sy);
                for c in 0..3 {
                    acc[c] += w * p[c] as f64;
                }
            }
            out.set(x, y, [acc[0] as f32, acc[1] as f32, acc[2] as f32]);
        }
    }
    out
}

/// Перекрёстные помехи каналов. Матрица симметрична и построчно нормирована
/// (сумма каждой строки = 1, так что ровный серый остаётся серым):
///
/// ```text
/// [ 1-x_rg        x_rg           0      ] [R]
/// [ x_rg          1-x_rg-x_gb    x_gb   ] [G]
/// [ 0             x_gb           1-x_gb ] [B]
/// ```
///
/// x_rg связывает R<->G, x_gb — G<->B; прямой R<->B связи нет.
pub fn crosstalk(img: &mut Image, x_rg: f64, x_gb: f64) {
    if x_rg == 0.0 && x_gb == 0.0 {
        return;
    }
    for p in &mut img.data {
        let r = p[0] as f64;
        let g = p[1] as f64;
        let b = p[2] as f64;
        let nr = (1.0 - x_rg) * r + x_rg * g;
        let ng = x_rg * r + (1.0 - x_rg - x_gb) * g + x_gb * b;
        let nb = x_gb * g + (1.0 - x_gb) * b;
        *p = [nr as f32, ng as f32, nb as f32];
    }
}

/// Поканальное усиление/смещение: out = gain·in + offset.
pub fn gain_offset(img: &mut Image, gain: [f64; 3], offset: [f64; 3]) {
    if gain == [1.0; 3] && offset == [0.0; 3] {
        return;
    }
    for p in &mut img.data {
        for c in 0..3 {
            p[c] = (gain[c] * p[c] as f64 + offset[c]) as f32;
        }
    }
}

/// Аддитивный гауссов шум σ (доля полной шкалы), из детерминированного ГПСЧ.
pub fn add_noise(img: &mut Image, sigma: f64, rng: &mut Rng) {
    if sigma <= 0.0 {
        return;
    }
    for p in &mut img.data {
        for c in 0..3 {
            p[c] = (p[c] as f64 + rng.gaussian() * sigma) as f32;
        }
    }
}

/// Зажать все каналы в [0, 1].
pub fn clamp01(img: &mut Image) {
    for p in &mut img.data {
        for c in 0..3 {
            p[c] = p[c].clamp(0.0, 1.0);
        }
    }
}

/// Полный тракт канала: кадр -> финальное изображение камеры + геометрия
/// проекции (для genie-map демода). Шум берётся из переданного ГПСЧ.
pub fn apply_channel(frame: &Frame, ch: &ChannelParams, rng: &mut Rng) -> (Image, Geometry) {
    let disp = emit_linear(frame, ch.gammas);
    let scale = ch.px_per_cell / ch.cell_size_px;
    let geom = Geometry::new(scale, ch.homography, frame.size_px);
    let mut img = warp(&disp, &geom);
    img = blur(&img, ch.blur_sigma_px);
    crosstalk(&mut img, ch.crosstalk_rg, ch.crosstalk_gb);
    gain_offset(&mut img, ch.gain, ch.offset);
    add_noise(&mut img, ch.noise_sigma, rng);
    clamp01(&mut img);
    (img, geom)
}

/// Итог одной попытки: сколько клеток/бит переданы неверно.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrialResult {
    pub wrong_cells: usize,
    pub total_cells: usize,
    pub wrong_bits: u32,
    pub total_bits: u32,
}

impl TrialResult {
    /// Symbol error rate: доля неверных клеток.
    pub fn ser(&self) -> f64 {
        if self.total_cells == 0 {
            0.0
        } else {
            self.wrong_cells as f64 / self.total_cells as f64
        }
    }
    /// Bit error rate по битам клеточных символов (Грей-код: XOR младших
    /// bits_per_cell бит = число ошибочных бит).
    /// (Считается в каждой попытке; в таблицы BENCHMARKS §1–2 пока не выводится.)
    #[allow(dead_code)]
    pub fn ber(&self) -> f64 {
        if self.total_bits == 0 {
            0.0
        } else {
            self.wrong_bits as f64 / self.total_bits as f64
        }
    }
}

/// Одна попытка Monte Carlo: случайные клеточные символы -> render -> канал ->
/// demod -> сравнение. Сид выводится из (point_idx, trial_idx) — воспроизводимо.
pub fn run_trial(
    p: &CalibProfile,
    ch: &ChannelParams,
    point_idx: usize,
    trial_idx: usize,
) -> TrialResult {
    let mut rng = Rng::new(crate::rng::seed_for(point_idx, trial_idx));
    let bpc = symbol::bits_per_cell(p);
    let n_levels = 1u32 << bpc;
    let n_cells = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;

    // случайные клеточные символы, равномерно по 2^bits_per_cell
    let cells: Vec<u8> = (0..n_cells)
        .map(|_| rng.next_u32_below(n_levels) as u8)
        .collect();

    let frame = symbol::render_symbol(p, &cells);
    let (img, geom) = apply_channel(&frame, ch, &mut rng);

    let map = |u: f64, v: f64| geom.forward(u, v);
    let sample = |x: f64, y: f64| img.sample(x, y);
    let out = symbol::demod_symbol(p, &map, &sample);

    let mut res = TrialResult {
        total_cells: n_cells,
        ..Default::default()
    };
    let mask = n_levels - 1;
    for (&sent, &got) in cells.iter().zip(out.iter()) {
        if sent != got {
            res.wrong_cells += 1;
        }
        let diff = (sent ^ got) as u32 & mask;
        res.wrong_bits += diff.count_ones();
        res.total_bits += bpc;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_of_impulse_conserves_mass() {
        // импульс в центре, вдали от краёв -> масса сохраняется (ядро сумм.=1)
        let n = 41;
        let mut img = Image::new(n, n);
        img.set(n / 2, n / 2, [1.0, 1.0, 1.0]);
        let out = blur(&img, 2.0);
        let mass: f64 = out.data.iter().map(|p| p[0] as f64).sum();
        assert!((mass - 1.0).abs() < 1e-4, "mass {mass}");
    }

    #[test]
    fn blur_second_moment_matches_kernel_variance() {
        // второй момент размытого импульса вдоль оси = дисперсия 1-D ядра,
        // а та в пределах усечения ≈ σ².
        let sigma = 2.0;
        let n = 61;
        let mut img = Image::new(n, n);
        let cx = n / 2;
        let cy = n / 2;
        img.set(cx, cy, [1.0, 0.0, 0.0]);
        let out = blur(&img, sigma);

        let mut sum = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for y in 0..n {
            for x in 0..n {
                let w = out.at(x, y)[0] as f64;
                sum += w;
                let dx = x as f64 - cx as f64;
                let dy = y as f64 - cy as f64;
                var_x += w * dx * dx;
                var_y += w * dy * dy;
            }
        }
        var_x /= sum;
        var_y /= sum;

        // дисперсия дискретного ядра
        let k = gaussian_kernel(sigma);
        let r = (k.len() / 2) as f64;
        let kvar: f64 = k
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let d = i as f64 - r;
                w * d * d
            })
            .sum();

        // второй момент изображения совпадает с дисперсией ядра
        assert!((var_x - kvar).abs() < 1e-6, "var_x {var_x} vs kvar {kvar}");
        assert!((var_y - kvar).abs() < 1e-6, "var_y {var_y} vs kvar {kvar}");
        // и та близка к σ² (усечение на 3σ занижает не более чем на ~3%)
        assert!(
            (kvar - sigma * sigma).abs() / (sigma * sigma) < 0.05,
            "kvar {kvar} vs sigma^2 {}",
            sigma * sigma
        );
    }

    #[test]
    fn warp_identity_reproduces_image() {
        // масштаб 1 + единичная гомография -> warp воспроизводит вход
        let n = 32;
        let mut img = Image::new(n, n);
        let mut rng = Rng::new(7);
        for p in &mut img.data {
            let v = rng.next_f64() as f32;
            *p = [v, v * 0.5, 1.0 - v];
        }
        let geom = Geometry::new(1.0, crate::channel::IDENTITY, n);
        let out = warp(&img, &geom);
        // сравниваем внутренность (края тождественны тоже, но держим запас)
        let mut max_diff = 0.0f32;
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                for c in 0..3 {
                    let d = (out.at(x, y)[c] - img.at(x, y)[c]).abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
        assert!(max_diff < 1e-6, "max_diff {max_diff}");
    }

    #[test]
    fn crosstalk_preserves_flat_gray() {
        // ровный серый должен остаться собой (строки нормированы)
        let mut img = Image::filled(4, 4, [0.5, 0.5, 0.5]);
        crosstalk(&mut img, 0.06, 0.08);
        for p in &img.data {
            for c in 0..3 {
                assert!((p[c] - 0.5).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn trial_result_rates() {
        // 1 неверная клетка из 4, 3 ошибочных бита из 20 -> SER 0.25, BER 0.15
        let r = TrialResult {
            wrong_cells: 1,
            total_cells: 4,
            wrong_bits: 3,
            total_bits: 20,
        };
        assert!((r.ser() - 0.25).abs() < 1e-12);
        assert!((r.ber() - 0.15).abs() < 1e-12);
        // пустой результат не делит на ноль
        let z = TrialResult::default();
        assert_eq!(z.ser(), 0.0);
        assert_eq!(z.ber(), 0.0);
    }

    #[test]
    fn crosstalk_mixes_pure_channel() {
        // чистый R: часть уходит в G, в B напрямую — ничего
        let mut img = Image::filled(1, 1, [1.0, 0.0, 0.0]);
        crosstalk(&mut img, 0.1, 0.2);
        let p = img.data[0];
        assert!((p[0] - 0.9).abs() < 1e-6, "R {}", p[0]);
        assert!((p[1] - 0.1).abs() < 1e-6, "G {}", p[1]);
        assert!(p[2].abs() < 1e-6, "B {}", p[2]);
    }
}

/// Сквозные тесты тракта: render (psicode-core) -> канал -> demod. Активны с
/// момента интеграции symbol.rs (до неё падали на `todo!()` и держались под
/// #[ignore]). Идеальный канал обязан давать SER = 0.
#[cfg(test)]
mod e2e {
    use super::*;
    use crate::channel::ChannelParams;

    fn reference_profile() -> CalibProfile {
        use psicode_core::ChromaMode;
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: 16,
            frame_hold_periods: 6,
            luma_bits: 3,
            chroma_mode: ChromaMode::Chroma2,
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

    #[test]
    fn clean_channel_gives_zero_ser() {
        let p = reference_profile();
        let ch = ChannelParams::clean(&p); // σ=0, без шума, identity, px/cell=8
        let r = run_trial(&p, &ch, 0, 0);
        assert_eq!(r.wrong_cells, 0, "SER={}", r.ser());
    }

    #[test]
    fn gain_offset_clean_channel_gives_zero_ser() {
        let p = reference_profile();
        let mut ch = ChannelParams::clean(&p);
        // усиление/смещение при прочем чистом канале не должны портить решение
        ch.gain = [0.8; 3];
        ch.offset = [0.05; 3];
        let r = run_trial(&p, &ch, 0, 0);
        assert_eq!(r.wrong_cells, 0, "SER={}", r.ser());
    }
}
