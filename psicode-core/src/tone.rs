//! Самокалибровка тон-кривой приёмника по референсной строке (§3.4).
//!
//! На реальной камере сквозная кривая «дисплей(~2.2) -> JPEG/сенсор(~sRGB)»
//! неизвестна. Демодулятору ([`crate::symbol::demod_symbol`]) для нормировки
//! §3.4 нужна верная ФОРМА кривой на канал — дальше он сам снимет gain/offset
//! по якорям K/W. Здесь мы оцениваем пер-канальную гамму линеаризации по
//! НЕЙТРАЛЬНЫМ клеткам референсной строки (K, W и 6 серых ступеней), минимизируя
//! невязку восстановления drive — ту же операцию, что делает демодулятор.
//!
//! На живом снимке это снизило SER 0.34 -> 0.09 — самокалибровка ОБЯЗАТЕЛЬНА для
//! реальных камер (жёсткая CAMERA_GAMMA=2.2 давала SER 0.34).
//!
//! Модуль под фичей `std` (нужна вещественная математика: `powf`).
//! Соглашения `map`/`sample` — те же, что у [`crate::symbol::demod_symbol`],
//! только `sample` отдаёт СЫРОЙ (до гаммы) нормированный RGB [0,1] снимка.

use crate::profile::CalibProfile;
use crate::symbol;
use alloc::vec::Vec;

/// Приближение обратной гаммы камеры/дисплея по умолчанию (§3.4): применяется,
/// когда точек нейтральной лесенки не хватает для устойчивой оценки.
pub const DEFAULT_GAMMA: f64 = 2.2;

/// Период референсного паттерна §3.4 в клетках: `K W R G B C M Y K W` + 6 серых.
const REF_PERIOD: usize = 16;
/// Число ступеней серой лесенки в референсном паттерне (§3.4).
const REF_GRAY_STEPS: usize = 6;

/// Оценивает пер-канальную гамму линеаризации по референсной строке (§3.4).
/// На канал подбираем γ так, чтобы (raw)^γ была максимально ЛИНЕЙНА (drive)^γ_проф
/// по НЕЙТРАЛЬНЫМ клеткам строки (K, W и 6 серых ступеней) — т.е. форма кривой
/// совпала с той, что ждёт демодулятор (он затем сам снимет gain/offset по K/W).
/// Заменяет жёсткую CAMERA_GAMMA самокалибровкой на кадр.
///
/// `map(u, v)` — координаты плоскости символа (display-px, тихая зона включена) ->
/// координаты снимка; `sample(x, y)` — СЫРОЙ нормированный RGB [0,1] снимка (до
/// гаммы) в этой точке. Возвращает [γr, γg, γb]; при нехватке точек нейтральной
/// лесенки (<3 уровней) — [`DEFAULT_GAMMA`] на все каналы.
pub fn estimate_channel_gammas(
    p: &CalibProfile,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    sample: &dyn Fn(f64, f64) -> [f32; 3],
) -> [f64; 3] {
    let quiet = p.quiet_zone_cells() as usize;
    let cell = p.cell_size_px as usize;
    let white = (255.0 * p.white_level_pct() as f64 / 100.0).round();
    let black = (255.0 * p.black_level_pct() as f64 / 100.0).round();
    // нейтральный drive клетки строки по индексу периода (None — цветная).
    let neutral = |idx: usize| -> Option<f64> {
        match idx {
            0 | 8 => Some(black),
            1 | 9 => Some(white),
            10..=15 => {
                let step = (idx - 10) as f64;
                Some(black + (white - black) * step / (REF_GRAY_STEPS - 1) as f64)
            }
            _ => None,
        }
    };
    // сэмпл клетки (cx,cy) канона: 4 точки ±cell/4 от центра (как sample_cell).
    let cell_raw = |cx: usize, cy: usize| -> [f64; 3] {
        let u = ((quiet + cx) * cell) as f64 + cell as f64 / 2.0;
        let v = ((quiet + cy) * cell) as f64 + cell as f64 / 2.0;
        let d = cell as f64 / 4.0;
        let mut acc = [0.0f64; 3];
        for &(sx, sy) in &[(-d, -d), (-d, d), (d, -d), (d, d)] {
            let (x, y) = map(u + sx, v + sy);
            let s = sample(x, y);
            for c in 0..3 {
                acc[c] += s[c] as f64;
            }
        }
        for c in 0..3 {
            acc[c] /= 4.0;
        }
        acc
    };
    // собираем (drive01, raw[c]) по нейтральным клеткам, ГРУППИРУЯ по уровню
    // drive (усредняем повторы периода -> чистая лесенка из ~6 точек).
    let mut levels: Vec<(f64, [f64; 3], usize)> = Vec::new();
    for ic in 0..symbol::INTERIOR {
        if let Some(d) = neutral(ic % REF_PERIOD) {
            let d01 = d / 255.0;
            let s = cell_raw(symbol::RING + ic, symbol::RING);
            match levels.iter_mut().find(|e| (e.0 - d01).abs() < 1e-6) {
                Some(e) => {
                    for c in 0..3 {
                        e.1[c] += s[c];
                    }
                    e.2 += 1;
                }
                None => levels.push((d01, s, 1)),
            }
        }
    }
    if levels.len() < 3 {
        return [DEFAULT_GAMMA; 3];
    }
    levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    for e in &mut levels {
        for c in 0..3 {
            e.1[c] /= e.2 as f64;
        }
    }
    let pts: Vec<(f64, [f64; 3])> = levels.iter().map(|e| (e.0, e.1)).collect();
    let gprof = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
    fit_linearisation_gamma(&pts, &gprof)
}

/// Ядро оценки пер-канальной гаммы ЛИНЕАРИЗАЦИИ по нейтральной лесенке.
///
/// `levels` — точки `(drive01, сырой RGB)`, ОТСОРТИРОВАННЫЕ по drive и уже
/// усреднённые по повторам; `gprof` — гаммы профиля (что ждёт демодулятор).
/// Возвращает `[γr, γg, γb]`: показатель, которым надо возвести СЫРОЙ отсчёт,
/// чтобы получить линейный свет.
///
/// Вынесено из [`estimate_channel_gammas`] без изменения поведения, чтобы
/// внутриполосный калибровочный кадр ([`crate::calframe`]) считал гамму ТЕМ ЖЕ
/// оценщиком, но по КРУПНЫМ нейтральным плиткам, а не по односкеточной
/// референсной строке §3.4 — иначе head-to-head сравнивал бы и оценщики тоже.
pub fn fit_linearisation_gamma(levels: &[(f64, [f64; 3])], gprof: &[f64; 3]) -> [f64; 3] {
    if levels.len() < 3 {
        return [DEFAULT_GAMMA; 3];
    }
    let mut out = [DEFAULT_GAMMA; 3];
    // Крайние уровни (чёрный/белый) — якоря a,b, как у демодулятора (§3.4).
    let (dk, sk_all) = (levels[0].0, levels[0].1);
    let (dw, sw_all) = (levels[levels.len() - 1].0, levels[levels.len() - 1].1);
    for c in 0..3 {
        let gp = gprof[c];
        let (xk, xw) = (dk.powf(gp), dw.powf(gp));
        // подбираем γ ∈ [1,6] по МИНИМУМУ невязки восстановления drive: строим
        // сенсорную модель s=(raw)^γ, снимаем a,b по чёрн/бел, инвертируем как
        // демодулятор d̂=((s−b)/a)^(1/γ_проф) и штрафуем |d̂−d| по всем ступеням.
        let mut best_g = DEFAULT_GAMMA;
        let mut best_res = f64::MAX;
        let mut g = 1.0;
        while g <= 6.0 + 1e-9 {
            let sk = sk_all[c].max(1e-6).powf(g);
            let sw = sw_all[c].max(1e-6).powf(g);
            let denom = xw - xk;
            if denom.abs() > 1e-12 {
                let a = (sw - sk) / denom;
                let b = sk - a * xk;
                if a.abs() > 1e-12 {
                    let mut res = 0.0;
                    for e in levels {
                        let s = e.1[c].max(1e-6).powf(g);
                        let base = ((s - b) / a).max(0.0);
                        let dhat = base.powf(1.0 / gp);
                        res += (dhat - e.0).powi(2);
                    }
                    if res < best_res {
                        best_res = res;
                        best_g = g;
                    }
                }
            }
            g += 0.05;
        }
        out[c] = best_g;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ChromaMode;
    use crate::symbol::render_symbol;
    use alloc::vec;

    fn prof() -> CalibProfile {
        CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: 16,
            frame_hold_periods: 6,
            luma_bits: 2,
            chroma_mode: ChromaMode::Chroma1,
            gamma_g_q: 28,       // γ_G = 2.20
            gamma_r_delta_q: 8,  // γ_R = 2.20
            gamma_b_delta_q: 10, // γ_B = 2.25
            white_level_q: 15,
            black_level_q: 2,
            noise_sigma_q: 0,
            mtf_limit_px: 6,
            torn_frames_q: 0,
            crosstalk_rg_q: 0,
            crosstalk_gb_q: 0,
            quiet_zone: 1,
            fec_overhead: 0,
        }
    }

    /// Идеальный канал raw = drive/255 (γ_true = 1): подбор обязан вернуть РОВНО
    /// профильную гамму на канал (при g_true=1 минимум невязки — g = γ_проф),
    /// причём γ_R/γ_G/γ_B кратны 0.05 и лежат на сетке поиска. Это подтверждает,
    /// что математика §3.4 переехала в core без искажений.
    #[test]
    fn recovers_profile_gamma_on_identity_channel() {
        let p = prof();
        let cells = vec![0u8; symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS];
        let frame = render_symbol(&p, &cells);
        let size = frame.size_px;
        let rgb = frame.rgb.clone();
        // sample = raw drive/255 через ближайший пиксель (клетки — сплошные блоки,
        // поэтому центр±cell/4 всегда внутри одной клетки -> точное значение).
        let sample = move |x: f64, y: f64| -> [f32; 3] {
            let xi = (x.max(0.0) as usize).min(size - 1);
            let yi = (y.max(0.0) as usize).min(size - 1);
            let d = rgb[yi * size + xi];
            [
                d[0] as f32 / 255.0,
                d[1] as f32 / 255.0,
                d[2] as f32 / 255.0,
            ]
        };
        let map = |u: f64, v: f64| (u, v);
        let cg = estimate_channel_gammas(&p, &map, &sample);
        let want = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
        for c in 0..3 {
            assert!(
                (cg[c] - want[c]).abs() < 0.05,
                "канал {c}: got γ={:.3}, want γ_проф={:.3}",
                cg[c],
                want[c]
            );
        }
    }
}
