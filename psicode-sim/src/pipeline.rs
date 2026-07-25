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
use psicode_core::detect::{detect_symbol, frame_map};
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

/// Сепарабельный гауссов блюр С РАЗНОЙ σ по каналам — модель хроматической
/// аберрации (§5.1). Каждый канал сворачивается своим ядром; при трёх равных
/// σ результат побитно совпадает с [`blur`] (та же сумма в том же порядке).
/// clamp-to-edge на границах; все σ ≤ 0 — тождество.
pub fn blur_rgb(img: &Image, sigma: [f64; 3]) -> Image {
    if sigma.iter().all(|&s| s <= 0.0) {
        return img.clone();
    }
    // изотропный случай (три равные σ) — общий путь развёрток: один проход,
    // побитно тот же результат, что и раньше (сохраняем числа BENCHMARKS §1–2).
    if sigma[0] == sigma[1] && sigma[1] == sigma[2] {
        return blur(img, sigma[0]);
    }
    let kernels = [
        gaussian_kernel(sigma[0]),
        gaussian_kernel(sigma[1]),
        gaussian_kernel(sigma[2]),
    ];
    let radii = [
        (kernels[0].len() / 2) as isize,
        (kernels[1].len() / 2) as isize,
        (kernels[2].len() / 2) as isize,
    ];

    // горизонтальный проход (каждый канал — своим ядром)
    let mut tmp = Image::new(img.w, img.h);
    for y in 0..img.h {
        for x in 0..img.w {
            let mut acc = [0.0f64; 3];
            for c in 0..3 {
                let (k, r) = (&kernels[c], radii[c]);
                for (ki, &w) in k.iter().enumerate() {
                    let sx = clamp_idx(x as isize + ki as isize - r, img.w);
                    acc[c] += w * img.at(sx, y)[c] as f64;
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
            for c in 0..3 {
                let (k, r) = (&kernels[c], radii[c]);
                for (ki, &w) in k.iter().enumerate() {
                    let sy = clamp_idx(y as isize + ki as isize - r, img.h);
                    acc[c] += w * tmp.at(x, sy)[c] as f64;
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
    img = blur_rgb(&img, ch.blur_sigma_rgb);
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

/// Розыгрыш случайных клеточных символов, равномерно по 2^bits_per_cell.
/// Порядок вызовов ГПСЧ фиксирован — воспроизводимость развёрток.
fn draw_random_cells(p: &CalibProfile, rng: &mut Rng) -> Vec<u8> {
    let bpc = symbol::bits_per_cell(p);
    let n_levels = 1u32 << bpc;
    let n_cells = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
    (0..n_cells)
        .map(|_| rng.next_u32_below(n_levels) as u8)
        .collect()
}

/// Ядро всех попыток: render(`p_tx`) -> канал `ch` (шум из `rng`) -> demod(`p_rx`).
/// Разделение передатчика и декодера нужно для развёрток рассогласования
/// калибровки (§7.3): приёмник декодирует чужим профилем. Обычно p_tx == p_rx.
fn channel_roundtrip(
    p_tx: &CalibProfile,
    p_rx: &CalibProfile,
    ch: &ChannelParams,
    cells: &[u8],
    rng: &mut Rng,
) -> Vec<u8> {
    let frame = symbol::render_symbol(p_tx, cells);
    let (img, geom) = apply_channel(&frame, ch, rng);
    let map = |u: f64, v: f64| geom.forward(u, v);
    let sample = |x: f64, y: f64| img.sample(x, y);
    symbol::demod_symbol(p_rx, &map, &sample)
}

/// Одна попытка Monte Carlo с сырыми данными: (отправленные клетки,
/// демодулированные клетки). Сид выводится из (point_idx, trial_idx). Общая
/// точка входа для SER-развёрток, полос-goodput и раздельного luma/chroma учёта.
pub fn run_trial_io(
    p: &CalibProfile,
    ch: &ChannelParams,
    point_idx: usize,
    trial_idx: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng::new(crate::rng::seed_for(point_idx, trial_idx));
    let cells = draw_random_cells(p, &mut rng);
    let got = channel_roundtrip(p, p, ch, &cells, &mut rng);
    (cells, got)
}

/// Как [`run_trial_io`], но передатчик/канал знают `p_tx`, а ДЕКОДЕР — `p_rx`
/// (рассогласование калибровки §7.3: стойкость к устаревшим/неверным гаммам).
/// `p_tx` и `p_rx` обязаны иметь одинаковые luma_bits/chroma_mode (общий алфавит).
pub fn run_trial_decode_io(
    p_tx: &CalibProfile,
    p_rx: &CalibProfile,
    ch: &ChannelParams,
    point_idx: usize,
    trial_idx: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng::new(crate::rng::seed_for(point_idx, trial_idx));
    let cells = draw_random_cells(p_tx, &mut rng);
    let got = channel_roundtrip(p_tx, p_rx, ch, &cells, &mut rng);
    (cells, got)
}

/// Одна попытка Monte Carlo: случайные клеточные символы -> render -> канал ->
/// demod -> сравнение. Сид выводится из (point_idx, trial_idx) — воспроизводимо.
pub fn run_trial(
    p: &CalibProfile,
    ch: &ChannelParams,
    point_idx: usize,
    trial_idx: usize,
) -> TrialResult {
    let (sent, got) = run_trial_io(p, ch, point_idx, trial_idx);
    let bpc = symbol::bits_per_cell(p);
    let mask = (1u32 << bpc) - 1;
    let mut res = TrialResult {
        total_cells: sent.len(),
        ..Default::default()
    };
    for (&s, &g) in sent.iter().zip(got.iter()) {
        if s != g {
            res.wrong_cells += 1;
        }
        let diff = (s ^ g) as u32 & mask;
        res.wrong_bits += diff.count_ones();
        res.total_bits += bpc;
    }
    res
}

/// Раскладка 55 payload-строк на 8 последовательных полос (§6.2): 7·7 + 6.
pub const STRIPE_ROWS: [usize; 8] = [7, 7, 7, 7, 7, 7, 7, 6];

/// Модель выживания полос (§6.2 per-stripe CRC-16): полоса «жива» тогда и только
/// тогда, когда ВСЕ её клетки декодированы точно — одна ошибка внутри роняет
/// целую полосу. Возвращает (уцелевшие payload-БИТЫ кадра, «кадр мёртв целиком»).
/// Кадр мёртв ⇔ ни одна полоса не выжила (это и есть FER §6.3).
pub fn surviving_payload_bits(sent: &[u8], got: &[u8], bpc: u32) -> (usize, bool) {
    let cols = symbol::PAYLOAD_COLS;
    let mut surviving = 0usize;
    let mut any_alive = false;
    let mut row0 = 0usize;
    for &rows in &STRIPE_ROWS {
        let start = row0 * cols;
        let end = (row0 + rows) * cols;
        let alive = sent[start..end]
            .iter()
            .zip(&got[start..end])
            .all(|(a, b)| a == b);
        if alive {
            surviving += rows * cols * bpc as usize;
            any_alive = true;
        }
        row0 += rows;
    }
    (surviving, !any_alive)
}

/// Число клеток с ошибкой именно в ХРОМА-части символа (младшие chroma_bits).
/// При chroma_bits = 0 (Mono/GreenOnly) всегда 0. Изолирует R−B-тракт от luma.
/// (Задействован тестами свойств §5.1; в бинаре развёрток не используется.)
#[allow(dead_code)]
pub fn count_chroma_wrong(sent: &[u8], got: &[u8], chroma_bits: u32) -> usize {
    if chroma_bits == 0 {
        return 0;
    }
    let mask = (1u32 << chroma_bits) - 1;
    sent.iter()
        .zip(got.iter())
        .filter(|(&a, &b)| (a as u32 & mask) != (b as u32 & mask))
        .count()
}

/// SER по клеткам: доля несовпавших символов.
fn cell_ser(sent: &[u8], got: &[u8]) -> f64 {
    let wrong = sent.iter().zip(got).filter(|(a, b)| a != b).count();
    wrong as f64 / sent.len() as f64
}

/// Итог попытки «честного приёмника»: один и тот же снимок демодулируется ДВАЖДЫ
/// — genie-геометрией (известная гомография тракта) и РЕАЛЬНОЙ детекцией ЗЧ-рамки
/// (§3.2, `psicode_core::detect`). Потеря кадра (detect вернул Err) — это
/// `detected_ser = None`; вызывающий трактует её как SER = 1.0 (кадр потерян).
#[derive(Clone, Copy, Debug)]
pub struct DetectTrial {
    /// SER при genie-геометрии (совпадает с [`run_trial`]) — опорная точка.
    pub genie_ser: f64,
    /// SER при демодуляции по ДЕТЕКТИРОВАННОЙ рамке; None — детекция не удалась.
    pub detected_ser: Option<f64>,
    /// Детекция вернула Ok.
    pub detect_ok: bool,
    /// Score детекции (средняя корреляция сторон, [0,1]); 0 при неудаче.
    pub score: f64,
}

/// Одна попытка честного приёмника: render -> канал -> (genie-demod ‖ detect+demod).
///
/// Детекция работает по ЛИНЕЙНОЙ яркости — берём G-канал снимка как есть (выход
/// канала уже в линейном свете). Соглашение о центрах пикселей у `detect` —
/// (i+0.5, j+0.5) (см. `sample_luma` в core), а у нашего [`Image::sample`] —
/// целочисленное; поэтому детектированную карту сэмплируем со сдвигом −0.5 по
/// обеим осям, иначе получим систематический полупиксельный промах.
pub fn run_trial_detect(
    p: &CalibProfile,
    ch: &ChannelParams,
    point_idx: usize,
    trial_idx: usize,
) -> DetectTrial {
    let mut rng = Rng::new(crate::rng::seed_for(point_idx, trial_idx));
    let cells = draw_random_cells(p, &mut rng);
    let frame = symbol::render_symbol(p, &cells);
    let (img, geom) = apply_channel(&frame, ch, &mut rng);

    // genie-путь: известная гомография тракта, целочисленные центры Image::sample.
    let genie = {
        let map = |u: f64, v: f64| geom.forward(u, v);
        let sample = |x: f64, y: f64| img.sample(x, y);
        symbol::demod_symbol(p, &map, &sample)
    };
    let genie_ser = cell_ser(&cells, &genie);

    // detect-путь: G-плоскость снимка -> детекция ЗЧ-рамки -> frame_map -> demod.
    let g_plane: Vec<f32> = img.data.iter().map(|px| px[1]).collect();
    let (detected_ser, detect_ok, score) = match detect_symbol(img.w, img.h, &g_plane) {
        Ok(d) => {
            let map = frame_map(p, &d);
            // −0.5: перевод (i+0.5)-центров detect в целочисленные Image::sample.
            let sample = |x: f64, y: f64| img.sample(x - 0.5, y - 0.5);
            let got = symbol::demod_symbol(p, &map, &sample);
            (Some(cell_ser(&cells, &got)), true, d.score)
        }
        Err(_) => (None, false, 0.0),
    };

    DetectTrial {
        genie_ser,
        detected_ser,
        detect_ok,
        score,
    }
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

    // --- честный приёмник: реальная детекция ЗЧ-рамки (§3.2) ---

    /// Чистый канал: детекция обязана найти кадр, поворот 0, SER_detected = 0.
    #[test]
    fn detect_clean_channel_zero_ser() {
        let p = reference_profile();
        let mut ch = ChannelParams::clean(&p);
        ch.px_per_cell = 8.0;

        // прямая детекция того же снимка — проверяем поворот 0.
        let mut rng = Rng::new(crate::rng::seed_for(0, 0));
        let cells = draw_random_cells(&p, &mut rng);
        let frame = symbol::render_symbol(&p, &cells);
        let (img, _geom) = apply_channel(&frame, &ch, &mut rng);
        let g: Vec<f32> = img.data.iter().map(|px| px[1]).collect();
        let d = detect_symbol(img.w, img.h, &g).expect("детекция чистого кадра");
        assert_eq!(d.rotation_quadrants, 0, "поворот чистого кадра != 0");

        // полный путь честного приёмника: SER_detected == 0.
        let r = run_trial_detect(&p, &ch, 0, 0);
        assert!(r.detect_ok, "детекция провалилась на чистом канале");
        assert_eq!(r.detected_ser, Some(0.0), "SER_detected != 0 (score {})", r.score);
        assert_eq!(r.genie_ser, 0.0, "genie SER != 0 на чистом канале");
    }

    /// Мягкая перспектива (keystone) при σ=0.5 и нулевом шуме: детекция должна
    /// сработать, а detected SER — почти не отставать от genie (≤ 5 п.п.).
    #[test]
    fn detect_perspective_close_to_genie() {
        let p = reference_profile();
        let mut ch = ChannelParams::clean(&p); // низкий (нулевой) шум
        ch.px_per_cell = 8.0;
        ch.set_blur(0.5);
        ch.homography = crate::channel::mild_perspective(crate::channel::symbol_size_px(&p));
        let r = run_trial_detect(&p, &ch, 42, 0);
        assert!(r.detect_ok, "детекция перспективы провалилась (score {})", r.score);
        let det = r.detected_ser.expect("detected SER");
        assert!(
            det - r.genie_ser <= 0.05,
            "detected SER {det} сильно хуже genie {}",
            r.genie_ser
        );
    }

    /// Вырожденный снимок (равномерно серый): детекция обязана вернуть Err без
    /// паники, а соглашение «потеря кадра -> SER 1.0» — дать ровно 1.0.
    #[test]
    fn detect_degenerate_gray_is_lost_frame() {
        let img = Image::filled(300, 300, [0.5, 0.5, 0.5]);
        let g: Vec<f32> = img.data.iter().map(|px| px[1]).collect();
        assert!(
            detect_symbol(img.w, img.h, &g).is_err(),
            "серое поле принято за кадр"
        );
        let lost: Option<f64> = None; // detected_ser при неудаче
        assert_eq!(lost.unwrap_or(1.0), 1.0);
    }
}

/// Валидация ТЕОРЕТИЧЕСКИХ утверждений §5.1 на симулированной физике: Im-слепота
/// камерной яркости (свойство a), усреднение ядра хромы под аберрацией (b) и
/// НЕ описанная в спеке утечка «край яркости -> хрома» при K_R ≠ K_B. Плюс
/// проверка модели выживания полос §6.2. Всё детерминировано (свои сиды).
#[cfg(test)]
mod spec_claims {
    use super::*;
    use crate::channel::ChannelParams;
    use psicode_core::ChromaMode;

    /// Эталонный профиль §7.4, но с настраиваемыми luma_bits/chroma_mode.
    fn profile(luma_bits: u8, chroma_mode: ChromaMode) -> CalibProfile {
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: 16,
            frame_hold_periods: 6,
            luma_bits,
            chroma_mode,
            gamma_g_q: 28,       // γ_G = 2.200
            gamma_r_delta_q: 8,  // γ_R = 2.200
            gamma_b_delta_q: 10, // γ_B = 2.250
            white_level_q: 15,   // белый 100%
            black_level_q: 2,    // чёрный 2%
            noise_sigma_q: 12,
            mtf_limit_px: 6,
            torn_frames_q: 5,
            crosstalk_rg_q: 3,
            crosstalk_gb_q: 4,
            quiet_zone: 1,
            fec_overhead: 2,
        }
    }

    /// b -> Грей-код (в core приватно, повторяем локально).
    fn b2g(b: u32) -> u32 {
        b ^ (b >> 1)
    }

    /// Реплика формулы амплитуд §5.1 v0 (в core приватна): (A_L, A_C) в drive.
    fn amplitudes(p: &CalibProfile) -> (f64, f64) {
        let white = (255.0 * p.white_level_pct() as f64 / 100.0).round();
        let black = (255.0 * p.black_level_pct() as f64 / 100.0).round();
        let usable = ((white - 128.0).min(128.0 - black)).max(0.0);
        match p.chroma_mode {
            ChromaMode::Mono | ChromaMode::GreenOnly => (usable, 0.0),
            _ => (0.7 * usable, 0.3 * usable),
        }
    }

    /// Средняя камерная яркость Y = 0.3R+0.6G+0.1B по центрам payload-клеток
    /// снимка (линейный свет [0,1] на выходе канала).
    fn mean_payload_luma_linear(p: &CalibProfile, img: &Image, geom: &Geometry) -> f64 {
        let quiet = p.quiet_zone_cells() as usize;
        let cell = p.cell_size_px as usize;
        let (mut sum, mut n) = (0.0f64, 0usize);
        for pr in 0..symbol::PAYLOAD_ROWS {
            let cy = symbol::RING + 1 + pr;
            for pc in 0..symbol::PAYLOAD_COLS {
                let cx = symbol::RING + pc;
                let u = ((quiet + cx) * cell) as f64 + cell as f64 / 2.0;
                let v = ((quiet + cy) * cell) as f64 + cell as f64 / 2.0;
                let (x, y) = geom.forward(u, v);
                let s = img.sample(x, y);
                sum += 0.3 * s[0] as f64 + 0.6 * s[1] as f64 + 0.1 * s[2] as f64;
                n += 1;
            }
        }
        sum / n as f64
    }

    /// Средняя яркость Y по payload в DRIVE-домене (кадр до канала, значения 0..255).
    fn mean_payload_luma_drive(p: &CalibProfile, frame: &symbol::Frame) -> f64 {
        let quiet = p.quiet_zone_cells() as usize;
        let cell = p.cell_size_px as usize;
        let sz = frame.size_px;
        let (mut sum, mut n) = (0.0f64, 0usize);
        for pr in 0..symbol::PAYLOAD_ROWS {
            let cy = symbol::RING + 1 + pr;
            for pc in 0..symbol::PAYLOAD_COLS {
                let cx = symbol::RING + pc;
                let px = (quiet + cx) * cell + cell / 2;
                let py = (quiet + cy) * cell + cell / 2;
                let d = frame.rgb[py * sz + px];
                sum += 0.3 * d[0] as f64 + 0.6 * d[1] as f64 + 0.1 * d[2] as f64;
                n += 1;
            }
        }
        sum / n as f64
    }

    /// МНК-наклон y по x (cov/var).
    fn slope(xs: &[f64], ys: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for (&x, &y) in xs.iter().zip(ys) {
            num += (x - mx) * (y - my);
            den += (x - mx) * (x - mx);
        }
        num / den
    }

    /// Заливает ВЕСЬ payload одной парой (luma-уровень, chroma-уровень) и строит кадр.
    fn uniform_frame(p: &CalibProfile, b_l: u32, b_c: u32) -> symbol::Frame {
        let cb = p.chroma_bits() as u32;
        let sym = ((b2g(b_l) << cb) | b2g(b_c)) as u8;
        let cells = vec![sym; symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS];
        symbol::render_symbol(p, &cells)
    }

    /// Попытка с ФИКСИРОВАННОЙ luma (Re const по кадру) и случайной хромой —
    /// изолирует R−B-тракт. Возвращает (отправленные, демодулированные) клетки.
    fn trial_fixed_luma(
        p: &CalibProfile,
        ch: &ChannelParams,
        point: usize,
        trial: usize,
        luma_gray: u32,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut rng = Rng::new(crate::rng::seed_for(point, trial));
        let cb = p.chroma_bits() as u32;
        let c_levels = 1u32 << cb;
        let n = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
        let cells: Vec<u8> = (0..n)
            .map(|_| {
                let cg = if c_levels > 1 {
                    rng.next_u32_below(c_levels)
                } else {
                    0
                };
                ((luma_gray << cb) | cg) as u8
            })
            .collect();
        let got = channel_roundtrip(p, p, ch, &cells, &mut rng);
        (cells, got)
    }

    /// §5.1 свойство (a): камерная яркость Y = 0.3R+0.6G+0.1B почти слепа к Im.
    /// При постоянной Re и переменной Im теоретическая утечка = 0.2·A_C·Im
    /// (0.3·(+A_C·Im) + 0.1·(−A_C·Im)). В drive-домене коэффициент ровно 0.2;
    /// в линейном свете гамма-кривизна слегка его сдвигает — измеряем и то и то.
    #[test]
    fn property_a_luma_is_nearly_im_blind() {
        // luma_bits=4 даёт уровень b_l=8 c Re≈+0.067 — близко к середине шкалы,
        // где локальный наклон гаммы ≈ компенсирует ослабление у рабочей точки.
        let p = profile(4, ChromaMode::Chroma3); // 8 уровней Im для регрессии
        let ch = ChannelParams::clean(&p); // matched γ, без блюра/шума/кросстолка
        let (_a_l, a_c) = amplitudes(&p);
        let c_levels = 1u32 << p.chroma_bits();
        let b_l = (1u32 << p.luma_bits) / 2; // рабочая точка около середины

        let mut xs = Vec::new(); // A_C·Im / 255  (доля драйва)
        let mut y_lin = Vec::new();
        let mut y_drv = Vec::new();
        for b_c in 0..c_levels {
            let frame = uniform_frame(&p, b_l, b_c);
            let mut rng = Rng::new(1);
            let (img, geom) = apply_channel(&frame, &ch, &mut rng);
            let im = -1.0 + 2.0 * b_c as f64 / (c_levels - 1) as f64;
            xs.push(a_c * im / 255.0);
            y_lin.push(mean_payload_luma_linear(&p, &img, &geom));
            y_drv.push(mean_payload_luma_drive(&p, &frame) / 255.0);
        }

        // drive-домен: алгебра точная -> наклон ровно 0.2 (кроме квантизации).
        let k_drive = slope(&xs, &y_drv);
        assert!(
            (k_drive - 0.2).abs() < 0.01,
            "drive-домен: наклон утечки Im->Y = {k_drive}, ждём 0.2"
        );

        // линейный свет («камерная luma»): гамма сдвигает коэффициент. Проверяем
        // окрестность 0.2 и печатаем точное значение (это находка, а не 0.2).
        let k_lin = slope(&xs, &y_lin);
        eprintln!("1a: Im->Y leakage  drive={k_drive:.4}  linear={k_lin:.4} (theory 0.2)");
        assert!(
            (k_lin - 0.2).abs() < 0.06,
            "линейный свет: наклон утечки Im->Y = {k_lin}, вне окрестности 0.2"
        );
    }

    /// §5.1 свойство (b): при постоянной Re эффективное ядро хромы = (K_R+K_B)/2,
    /// поэтому аберрация (σ_R=1.3, σ_G=1.0, σ_B=1.5) декодирует хрому как общий
    /// блюр средней σ=1.4. Сравниваем хрома-SER, база — чистый канал + только блюр.
    #[test]
    fn property_b_aberration_chroma_matches_average_kernel() {
        // Chroma3 (тесные уровни Im) + умеренный блюр -> хрома-SER измерима,
        // а не вырожденно нулевая: эквивалентность ядер проверяется по-настоящему.
        let p = profile(3, ChromaMode::Chroma3);
        let nominal = 1.0;
        let mut aberr = ChannelParams::clean(&p);
        aberr.px_per_cell = 8.0;
        aberr.set_aberration(nominal); // σ_R=1.3, σ_G=1.0, σ_B=1.5
        let mut common = ChannelParams::clean(&p);
        common.px_per_cell = 8.0;
        common.set_blur(1.4 * nominal); // среднее (σ_R+σ_B)/2 = 1.4

        let luma_gray = b2g((1u32 << p.luma_bits) / 2); // Re фиксирована
        let cb = p.chroma_bits() as u32;
        let seeds = 10;
        let (mut ab_w, mut cm_w, mut total) = (0usize, 0usize, 0usize);
        for t in 0..seeds {
            let (s1, g1) = trial_fixed_luma(&p, &aberr, 5100, t, luma_gray);
            let (s2, g2) = trial_fixed_luma(&p, &common, 5100, t, luma_gray);
            ab_w += count_chroma_wrong(&s1, &g1, cb);
            cm_w += count_chroma_wrong(&s2, &g2, cb);
            total += s1.len();
        }
        let ser_ab = ab_w as f64 / total as f64;
        let ser_cm = cm_w as f64 / total as f64;
        eprintln!(
            "1b: chroma-SER  aberration(σ_R={:.2},σ_B={:.2})={ser_ab:.5}  common(σ={:.2})={ser_cm:.5}  |Δ|={:.5}",
            aberr.blur_sigma_rgb[0],
            aberr.blur_sigma_rgb[2],
            common.blur_sigma_rgb[0],
            (ser_ab - ser_cm).abs()
        );
        // держатся в пределах ~1.5 pp: свойство (b) выполняется. Малый остаток
        // систематичен — смесь (K_R+K_B)/2 чуть острее гаусса σ=1.4, поэтому
        // аберрация декодирует хрому даже слегка лучше согласованного среднего.
        assert!(
            (ser_ab - ser_cm).abs() < 0.025,
            "аберрация не эквивалентна среднему ядру: {ser_ab} vs {ser_cm}"
        );
    }

    /// НЕ в спеке: при K_R ≠ K_B и ПЕРЕМЕННОЙ Re член (K_R−K_B)∗(A_L·Re) больше
    /// не зануляется и течёт в ось R−B на краях яркости. Меряем зазор хрома-SER
    /// между аберрацией и согласованным средним блюром (то же (K_R+K_B)/2, но
    /// K_R=K_B) — это цена аберрации при живой luma. Репортим величину.
    #[test]
    fn luma_edges_leak_into_chroma_under_aberration() {
        let p = profile(3, ChromaMode::Chroma2);
        let mut aberr = ChannelParams::clean(&p);
        aberr.px_per_cell = 8.0;
        aberr.set_aberration(1.0);
        let mut common = ChannelParams::clean(&p);
        common.px_per_cell = 8.0;
        common.set_blur(1.4);

        let cb = p.chroma_bits() as u32;
        let seeds = 8;
        let (mut ab_w, mut cm_w, mut total) = (0usize, 0usize, 0usize);
        for t in 0..seeds {
            // случайные Re И Im -> есть края яркости
            let (s1, g1) = run_trial_io(&p, &aberr, 6100, t);
            let (s2, g2) = run_trial_io(&p, &common, 6100, t);
            ab_w += count_chroma_wrong(&s1, &g1, cb);
            cm_w += count_chroma_wrong(&s2, &g2, cb);
            total += s1.len();
        }
        let ser_ab = ab_w as f64 / total as f64;
        let ser_cm = cm_w as f64 / total as f64;
        let gap = ser_ab - ser_cm;
        eprintln!(
            "1c: chroma-SER(varying Re)  aberration={ser_ab:.5}  common={ser_cm:.5}  gap={gap:+.5}"
        );
        // аберрация с живой luma не должна быть ЗАМЕТНО лучше согласованного
        // среднего — фиксируем факт утечки (зазор ≳ 0), величину печатаем.
        assert!(gap > -0.01, "неожиданно: аберрация лучше среднего на {gap}");
    }

    /// Модель выживания полос §6.2: одна ошибка роняет всю полосу; кадр мёртв,
    /// если мертвы все 8 полос.
    #[test]
    fn stripe_survival_model() {
        let cols = symbol::PAYLOAD_COLS;
        let rows = symbol::PAYLOAD_ROWS;
        let bpc = 5u32;
        let sent = vec![0u8; cols * rows];

        // всё верно -> все полосы живы -> полный payload
        let (bits, dead) = surviving_payload_bits(&sent, &sent, bpc);
        assert_eq!(bits, rows * cols * bpc as usize);
        assert!(!dead);

        // ошибка в строке 0 роняет первую полосу (7 строк)
        let mut got = sent.clone();
        got[0] = 1;
        let (bits1, dead1) = surviving_payload_bits(&sent, &got, bpc);
        assert_eq!(bits1, (rows - STRIPE_ROWS[0]) * cols * bpc as usize);
        assert!(!dead1);

        // по ошибке в каждой полосе -> кадр мёртв целиком
        let mut all = sent.clone();
        let mut r0 = 0usize;
        for &rs in &STRIPE_ROWS {
            all[r0 * cols] = 1;
            r0 += rs;
        }
        let (bits2, dead2) = surviving_payload_bits(&sent, &all, bpc);
        assert_eq!(bits2, 0);
        assert!(dead2);
    }
}
