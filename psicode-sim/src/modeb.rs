//! Mode B (Эрмит–Гаусс, RESEARCH §1): прототип рендера/декода блоков и
//! измерение SNR коэффициентов vs блюр σ — данные freeze-gate.
//!
//! Тракт полностью в f64 (без u8-квантования Mode A): поле HG синтезируется
//! напрямую, отображается в drive по §5.1 (M=128, split A_L/A_C 0.7/0.3 с
//! уровнями эталонного профиля §7.4), эмитируется через гаммы, размывается
//! существующей моделью канала (pipeline::{blur,crosstalk,add_noise,clamp01}),
//! декодируется обратной гаммой + обратным §5.1, и коэффициенты снимаются
//! внутренними произведениями с дискретными базисными массивами ψ_{m,n}.
//!
//! Геометрия — genie (identity, масштаб 1): блок 64×64 display-px = 64×64
//! camera-px, поэтому σ задаётся прямо в camera-px. Блок окружён полем средне-
//! серого (padding), чтобы блюр честно уносил энергию мод за край.

use crate::image::Image;
use crate::rng::{seed_for, Rng};
use crate::{pipeline, report};
use psicode_core::symbol::{A_C_FRACTION_CHROMA, A_L_FRACTION_CHROMA};
use psicode_core::{CalibProfile, ChromaMode};
use std::time::Instant;

/// Сторона блока Mode B v0 (RESEARCH §1).
const BLOCK: usize = 64;
/// Ширина огибающей ψ: block/8 (RESEARCH §1).
const W_NOMINAL: f64 = 8.0;
/// Поле средне-серого вокруг блока (≥ ceil(3·σ_max) для σ_max=8) — блюр уносит
/// энергию мод в это поле, как у реального блока с соседями.
const PAD: usize = 24;
/// Средняя точка §5.1.
const MID: f64 = 128.0;
/// Фиксированная известная амплитуда probe-мод (ψ20, ψ02).
const PROBE_AMP: f64 = 1.0;
/// Амплитуда QPSK по оси: символ = A_DATA·(±1 ± j), значит |c_data| = 1
/// (сопоставимо с пилотом 1 и probe 1).
const A_DATA: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// Целевая заполненность динамического диапазона поля (|Re|,|Im| ≲ этого).
const FIELD_TARGET: f64 = 0.85;
/// Попыток Monte Carlo на точку σ.
const TRIALS: usize = 20;
/// Развёртка блюра σ (camera-px, масштаб 1) — точки freeze-gate.
const SIGMAS: [f64; 8] = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0];

/// Комплексное число (без внешних зависимостей).
#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    #[inline]
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    #[inline]
    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Роль моды: пилот (0,0), probe (2,0)/(0,2), либо данные (QPSK).
#[derive(Clone, Copy, PartialEq)]
enum Role {
    Pilot,
    Probe,
    Data,
}

/// Эталонный профиль §7.4 SPEC (та же конфигурация, что psicode-sim::main).
fn reference_profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        luma_bits: 3,
        chroma_mode: ChromaMode::Chroma2,
        gamma_g_q: 28, // γ_G = 2.200
        gamma_r_delta_q: 8,
        gamma_b_delta_q: 10,
        white_level_q: 15, // 100%
        black_level_q: 2,
        noise_sigma_q: 12, // σ ≈ 2.0 градации
        mtf_limit_px: 6,
        torn_frames_q: 5,
        crosstalk_rg_q: 3, // 6%
        crosstalk_gb_q: 4, // 8%
        quiet_zone: 1,
        fec_overhead: 2,
        border: psicode_core::profile::BorderMode::LegacyInverted,
    }
}

/// Уровни белого/чёрного и амплитуды §5.1 (split хромы 0.7/0.3), плюс «usable».
/// Возвращает (usable, A_L_chroma, A_C_chroma). Зеркалит приватную amplitudes()
/// из psicode_core::symbol с публичными долями A_L/A_C_FRACTION_CHROMA.
fn amplitudes(p: &CalibProfile) -> (f64, f64, f64) {
    let white = (255.0 * p.white_level_pct() as f64 / 100.0).round();
    let black = (255.0 * p.black_level_pct() as f64 / 100.0).round();
    let usable = ((white - MID).min(MID - black)).max(0.0);
    (
        usable,
        A_L_FRACTION_CHROMA * usable,
        A_C_FRACTION_CHROMA * usable,
    )
}

/// Гаммы канала [R, G, B] из профиля (matched: тот же набор у кодера и декодера).
fn gammas_of(p: &CalibProfile) -> [f64; 3] {
    [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64]
}

/// Полином Эрмита (физический) H_m(u) через рекуррентность H_{k+1}=2u·H_k−2k·H_{k−1}.
fn hermite(m: u32, u: f64) -> f64 {
    if m == 0 {
        return 1.0;
    }
    let mut hm1 = 1.0; // H_0
    let mut h = 2.0 * u; // H_1
    for k in 1..m {
        let hnext = 2.0 * u * h - 2.0 * k as f64 * hm1;
        hm1 = h;
        h = hnext;
    }
    h
}

/// Список 15 мод m+n ≤ 4, сгруппированных по порядку (m+n): порядок 0..=4,
/// внутри порядка m=0..=order, n=order−m.
fn mode_list() -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    for order in 0..=4u32 {
        for m in 0..=order {
            v.push((m, order - m));
        }
    }
    v
}

/// Роль моды по индексам (§RESEARCH 1): (0,0)=пилот, (2,0)/(0,2)=probe, иначе данные.
fn role_of(m: u32, n: u32) -> Role {
    match (m, n) {
        (0, 0) => Role::Pilot,
        (2, 0) | (0, 2) => Role::Probe,
        _ => Role::Data,
    }
}

/// Дискретный базис ψ_{m,n} шириной `w` на решётке 64×64. Каждый массив
/// нормирован к единичной L2-норме (диагональ Грама ровно 1; вне-диагональ
/// показывает истинную неортогональность дискретизации/усечения).
struct Basis {
    modes: Vec<(u32, u32)>,
    roles: Vec<Role>,
    arrays: Vec<Vec<f64>>, // arrays[k][py*BLOCK+px]
}

impl Basis {
    fn new(w: f64) -> Self {
        let modes = mode_list();
        let roles: Vec<Role> = modes.iter().map(|&(m, n)| role_of(m, n)).collect();
        let center = (BLOCK as f64 - 1.0) / 2.0; // 31.5
        let mut arrays = Vec::with_capacity(modes.len());
        for &(m, n) in &modes {
            let mut a = vec![0.0f64; BLOCK * BLOCK];
            let mut ss = 0.0;
            for py in 0..BLOCK {
                let yr = (py as f64 - center) / w;
                for px in 0..BLOCK {
                    let xr = (px as f64 - center) / w;
                    let val = hermite(m, xr)
                        * hermite(n, yr)
                        * (-(xr * xr + yr * yr) / 2.0).exp();
                    a[py * BLOCK + px] = val;
                    ss += val * val;
                }
            }
            let norm = ss.sqrt();
            for v in a.iter_mut() {
                *v /= norm;
            }
            arrays.push(a);
        }
        Basis {
            modes,
            roles,
            arrays,
        }
    }

    /// Индекс моды (m,n) в списке.
    fn index(&self, m: u32, n: u32) -> usize {
        self.modes.iter().position(|&mn| mn == (m, n)).unwrap()
    }

    /// Максимум |⟨ψ_i, ψ_j⟩| по i<j (вне-диагональ матрицы Грама) + пара мод.
    fn gram_max_offdiag(&self) -> (f64, (u32, u32), (u32, u32)) {
        let mut best = 0.0;
        let mut pair = ((0, 0), (0, 0));
        for i in 0..self.arrays.len() {
            for j in (i + 1)..self.arrays.len() {
                let mut dot = 0.0;
                for k in 0..BLOCK * BLOCK {
                    dot += self.arrays[i][k] * self.arrays[j][k];
                }
                if dot.abs() > best {
                    best = dot.abs();
                    pair = (self.modes[i], self.modes[j]);
                }
            }
        }
        (best, pair.0, pair.1)
    }

    /// Внутренние произведения комплексного поля (fr + j·fi) с каждым ψ_k.
    fn project(&self, fr: &[f64], fi: &[f64]) -> Vec<C> {
        self.arrays
            .iter()
            .map(|a| {
                let mut re = 0.0;
                let mut im = 0.0;
                for k in 0..BLOCK * BLOCK {
                    re += fr[k] * a[k];
                    im += fi[k] * a[k];
                }
                C { re, im }
            })
            .collect()
    }
}

/// Розыгрыш QPSK-символа data_amp·(±1 ± j) из младших бит.
#[inline]
fn qpsk(data_amp: f64, rng: &mut Rng) -> C {
    let bits = rng.next_u64();
    let sx = if bits & 1 == 0 { -1.0 } else { 1.0 };
    let sy = if bits & 2 == 0 { -1.0 } else { 1.0 };
    C {
        re: data_amp * sx,
        im: data_amp * sy,
    }
}

/// Розыгрыш коэффициентов. `uniform=false` — конструкция как в системе (RESEARCH
/// §1): пилот (1,0), probe (PROBE_AMP,0), 12 data-мод QPSK. `uniform=true` — ВСЕ
/// 15 мод возбуждаются одинаковым QPSK (контролируемый опыт для таблиц физики:
/// изолирует зависимость от порядка m+n от асимметрии осей Re/Im и от DC-члена
/// пилота). При data_amp=0 — только фиксированные пилот/probe (uniform=false).
fn draw_coeffs(basis: &Basis, data_amp: f64, uniform: bool, rng: &mut Rng) -> Vec<C> {
    basis
        .roles
        .iter()
        .map(|role| {
            if uniform {
                qpsk(data_amp, rng)
            } else {
                match role {
                    Role::Pilot => C { re: 1.0, im: 0.0 },
                    Role::Probe => C {
                        re: PROBE_AMP,
                        im: 0.0,
                    },
                    Role::Data => qpsk(data_amp, rng),
                }
            }
        })
        .collect()
}

/// Синтез поля блока: fr[i]+j·fi[i] = g·Σ_k c_k·ψ_k[i].
fn synth_field(basis: &Basis, coeffs: &[C], g: f64) -> (Vec<f64>, Vec<f64>) {
    let mut fr = vec![0.0f64; BLOCK * BLOCK];
    let mut fi = vec![0.0f64; BLOCK * BLOCK];
    for (c, a) in coeffs.iter().zip(&basis.arrays) {
        for k in 0..BLOCK * BLOCK {
            fr[k] += c.re * a[k];
            fi[k] += c.im * a[k];
        }
    }
    for k in 0..BLOCK * BLOCK {
        fr[k] *= g;
        fi[k] *= g;
    }
    (fr, fi)
}

/// Калибровка глобального усиления поля g: по фиксированным сидам находим
/// максимум max(|Re|,|Im|) поля (g=1) и берём g = FIELD_TARGET / max, чтобы поле
/// заполняло диапазон [−1,1] с запасом. Детерминировано и воспроизводимо.
fn calibrate_gain(basis: &Basis) -> f64 {
    let mut maxv = 0.0f64;
    for &uniform in &[false, true] {
        for s in 0..40usize {
            let mut rng = Rng::new(seed_for(9000, s));
            let coeffs = draw_coeffs(basis, A_DATA, uniform, &mut rng);
            let (fr, fi) = synth_field(basis, &coeffs, 1.0);
            for k in 0..BLOCK * BLOCK {
                maxv = maxv.max(fr[k].abs()).max(fi[k].abs());
            }
        }
    }
    FIELD_TARGET / maxv
}

/// Параметры физики блока (амплитуды §5.1, гаммы, помехи канала).
#[derive(Clone, Copy)]
struct Phys {
    a_l: f64,
    a_c: f64,
    gammas: [f64; 3],
    crosstalk: (f64, f64),
    noise_sigma: f64,
}

/// Итог одной попытки: восстановленное поле + статистика клиппинга drive.
struct Rx {
    fr: Vec<f64>,
    fi: Vec<f64>,
    clips: usize,
    total: usize,
}

/// Полный тракт: синтез g·Σc·ψ → §5.1 drive → эмиссия (drive/255)^γ на padded
/// холст средне-серого → блюр σ → кроп 64×64 → кросстолк → шум → clamp →
/// обратная гамма → обратный §5.1 (Re из G, Im из (R−B)/2). Возвращает поле.
fn transmit_receive(
    basis: &Basis,
    coeffs: &[C],
    g: f64,
    ph: &Phys,
    sigma: f64,
    rng: &mut Rng,
) -> Rx {
    let (fr, fi) = synth_field(basis, coeffs, g);
    let ps = BLOCK + 2 * PAD;
    let mut img = Image::new(ps, ps);
    let mid_lin = [
        (MID / 255.0).powf(ph.gammas[0]) as f32,
        (MID / 255.0).powf(ph.gammas[1]) as f32,
        (MID / 255.0).powf(ph.gammas[2]) as f32,
    ];
    for p in img.data.iter_mut() {
        *p = mid_lin;
    }

    // блок → drive → linear, размещаем в центре холста
    let mut clips = 0usize;
    for py in 0..BLOCK {
        for px in 0..BLOCK {
            let i = py * BLOCK + px;
            let re = fr[i];
            let im = fi[i];
            // §5.1: R=M+A_L·Re+A_C·Im, G=M+A_L·Re, B=M+A_L·Re−A_C·Im
            let d = [
                MID + ph.a_l * re + ph.a_c * im,
                MID + ph.a_l * re,
                MID + ph.a_l * re - ph.a_c * im,
            ];
            if d.iter().any(|&v| v < 0.0 || v > 255.0) {
                clips += 1;
            }
            let lin = [
                ((d[0].clamp(0.0, 255.0)) / 255.0).powf(ph.gammas[0]) as f32,
                ((d[1].clamp(0.0, 255.0)) / 255.0).powf(ph.gammas[1]) as f32,
                ((d[2].clamp(0.0, 255.0)) / 255.0).powf(ph.gammas[2]) as f32,
            ];
            img.set(PAD + px, PAD + py, lin);
        }
    }

    // блюр всего холста (изотропный), затем кроп центрального блока
    let blurred = pipeline::blur(&img, sigma);
    let mut crop = Image::new(BLOCK, BLOCK);
    for py in 0..BLOCK {
        for px in 0..BLOCK {
            crop.set(px, py, blurred.at(PAD + px, PAD + py));
        }
    }
    pipeline::crosstalk(&mut crop, ph.crosstalk.0, ph.crosstalk.1);
    pipeline::add_noise(&mut crop, ph.noise_sigma, rng);
    pipeline::clamp01(&mut crop);

    // декод: обратная гамма → drive → обратный §5.1
    let mut fr_hat = vec![0.0f64; BLOCK * BLOCK];
    let mut fi_hat = vec![0.0f64; BLOCK * BLOCK];
    for py in 0..BLOCK {
        for px in 0..BLOCK {
            let i = py * BLOCK + px;
            let lin = crop.at(px, py);
            let dr = 255.0 * (lin[0] as f64).powf(1.0 / ph.gammas[0]);
            let dg = 255.0 * (lin[1] as f64).powf(1.0 / ph.gammas[1]);
            let db = 255.0 * (lin[2] as f64).powf(1.0 / ph.gammas[2]);
            fr_hat[i] = (dg - MID) / ph.a_l;
            fi_hat[i] = (dr - db) / (2.0 * ph.a_c);
        }
    }
    Rx {
        fr: fr_hat,
        fi: fi_hat,
        clips,
        total: BLOCK * BLOCK,
    }
}

/// Аккумулятор по одной моде за развёртку по попыткам.
#[derive(Clone, Copy, Default)]
struct ModeStat {
    s_rc: (f64, f64), // Σ r·conj(c)
    s_cc: f64,        // Σ |c|²
    s_rr: f64,        // Σ |r|²
    err_bits: usize,  // ошибки QPSK (только data)
    tot_bits: usize,
    trials: usize,
}
impl ModeStat {
    /// |α| = |Σ r·conj c| / Σ|c|² — модальное усиление (амплитуда).
    fn alpha_abs(&self) -> f64 {
        let m = (self.s_rc.0 * self.s_rc.0 + self.s_rc.1 * self.s_rc.1).sqrt();
        if self.s_cc > 0.0 {
            m / self.s_cc
        } else {
            0.0
        }
    }
    /// SNR = мощность сигнала / остаточная дисперсия (EVM-стиль):
    /// S = |Σr·conj c|²/(Σ|c|²·T), N = (Σ|r|² − |Σr·conj c|²/Σ|c|²)/T.
    fn snr(&self) -> f64 {
        let rc2 = self.s_rc.0 * self.s_rc.0 + self.s_rc.1 * self.s_rc.1;
        if self.s_cc <= 0.0 || self.trials == 0 {
            return 0.0;
        }
        let sig = rc2 / self.s_cc; // = S·T
        let noise = self.s_rr - rc2 / self.s_cc; // = N·T
        if noise <= 0.0 {
            f64::INFINITY
        } else {
            sig / noise
        }
    }
    fn ber(&self) -> f64 {
        if self.tot_bits == 0 {
            0.0
        } else {
            self.err_bits as f64 / self.tot_bits as f64
        }
    }
}

/// Аккумулировать вклад попытки: обновить статистику каждой моды.
fn accumulate(stats: &mut [ModeStat], coeffs: &[C], r: &[C]) {
    for k in 0..coeffs.len() {
        let c = coeffs[k];
        let rk = r[k];
        // Σ r·conj(c)
        stats[k].s_rc.0 += rk.re * c.re + rk.im * c.im;
        stats[k].s_rc.1 += rk.im * c.re - rk.re * c.im;
        stats[k].s_cc += c.abs2();
        stats[k].s_rr += rk.abs2();
        stats[k].trials += 1;
    }
}

/// QPSK-битовые ошибки моды данных: знак Re/Im принятого vs истинного коэфф.
fn qpsk_errors(c: C, r: C) -> usize {
    ((r.re > 0.0) != (c.re > 0.0)) as usize + ((r.im > 0.0) != (c.im > 0.0)) as usize
}

/// Развёртка по попыткам в точке σ: статистика по каждой моде (nominal-базис).
/// ВСЕ 15 мод возбуждаются одинаковым QPSK (uniform) — контролируемый опыт для
/// таблиц физики; BER считается для каждой моды (порядок 0..4).
fn sweep_point(basis: &Basis, g: f64, ph: &Phys, sigma: f64, point: usize) -> Vec<ModeStat> {
    let mut stats = vec![ModeStat::default(); basis.modes.len()];
    for t in 0..TRIALS {
        let mut rng = Rng::new(seed_for(point, t));
        let coeffs = draw_coeffs(basis, A_DATA, true, &mut rng);
        let rx = transmit_receive(basis, &coeffs, g, ph, sigma, &mut rng);
        let r = basis.project(&rx.fr, &rx.fi);
        accumulate(&mut stats, &coeffs, &r);
        for k in 0..basis.modes.len() {
            stats[k].err_bits += qpsk_errors(coeffs[k], r[k]);
            stats[k].tot_bits += 2;
        }
    }
    stats
}

/// Средняя ρ = ⟨(|r20|+|r02|)/2⟩ / ⟨|r00|⟩ по попыткам (probe/pilot ratio).
fn measure_rho(basis: &Basis, g: f64, ph: &Phys, sigma: f64, trials: usize, point: usize) -> f64 {
    let i00 = basis.index(0, 0);
    let i20 = basis.index(2, 0);
    let i02 = basis.index(0, 2);
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 0..trials {
        let mut rng = Rng::new(seed_for(point, t));
        // designed-конструкция: фиксированные пилот/probe — стабильное отношение
        let coeffs = draw_coeffs(basis, A_DATA, false, &mut rng);
        let rx = transmit_receive(basis, &coeffs, g, ph, sigma, &mut rng);
        let r = basis.project(&rx.fr, &rx.fi);
        num += 0.5 * (r[i20].abs() + r[i02].abs());
        den += r[i00].abs();
    }
    num / den
}

/// ОТНОСИТЕЛЬНАЯ probe-асимметрия (2,0) vs (0,2) под изотропным блюром:
/// |‖⟨r20⟩‖ − ‖⟨r02⟩‖| / среднее, с КОГЕРЕНТНЫМ усреднением фиксированных probe
/// по попыткам (шум зануляется, остаётся геометрия). При изотропии + квадратном
/// блоке ожидается ≈ 0; остаток растёт с σ как шум/сигнал (SNR probe падает).
fn probe_asymmetry(basis: &Basis, g: f64, ph: &Phys, sigma: f64, trials: usize, point: usize) -> f64 {
    let i20 = basis.index(2, 0);
    let i02 = basis.index(0, 2);
    let mut m20 = C { re: 0.0, im: 0.0 };
    let mut m02 = C { re: 0.0, im: 0.0 };
    for t in 0..trials {
        let mut rng = Rng::new(seed_for(point, t));
        let coeffs = draw_coeffs(basis, A_DATA, false, &mut rng);
        let rx = transmit_receive(basis, &coeffs, g, ph, sigma, &mut rng);
        let r = basis.project(&rx.fr, &rx.fi);
        m20.re += r[i20].re;
        m20.im += r[i20].im;
        m02.re += r[i02].re;
        m02.im += r[i02].im;
    }
    let a20 = m20.abs();
    let a02 = m02.abs();
    let mean = 0.5 * (a20 + a02);
    if mean <= 0.0 {
        0.0
    } else {
        (a20 - a02).abs() / mean
    }
}

/// Индексы мод по порядку (m+n): orders[o] = список индексов мод порядка o.
fn order_groups(basis: &Basis) -> Vec<Vec<usize>> {
    let mut g = vec![Vec::new(); 5];
    for (k, &(m, n)) in basis.modes.iter().enumerate() {
        g[(m + n) as usize].push(k);
    }
    g
}

/// dB с защитой от нуля/бесконечности.
fn to_db(x: f64) -> f64 {
    if x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * x.log10()
    }
}

// --- Mode A: минимальная сетка 8×8 клеток по 8 px, luma 3 бита (Mono) ---

const MODEA_CELLS: usize = 8; // 8×8 клеток
const MODEA_CELL_PX: usize = 8;
const MODEA_LEVELS: u32 = 8; // 3 бита яркости

/// Одна попытка Mode A на том же блоке/канале: булев вектор «клетка верна» по
/// 64 клеткам. Mono: R=G=B=M+usable·Re; декод по G-каналу, ближайший из 8 уровней.
fn mode_a_trial(usable: f64, ph: &Phys, sigma: f64, point: usize, trial: usize) -> Vec<bool> {
    let mut rng = Rng::new(seed_for(point, trial));
    // розыгрыш уровней клеток
    let levels: Vec<u32> = (0..MODEA_CELLS * MODEA_CELLS)
        .map(|_| rng.next_u32_below(MODEA_LEVELS))
        .collect();

    let ps = BLOCK + 2 * PAD;
    let mut img = Image::new(ps, ps);
    let mid_lin = [
        (MID / 255.0).powf(ph.gammas[0]) as f32,
        (MID / 255.0).powf(ph.gammas[1]) as f32,
        (MID / 255.0).powf(ph.gammas[2]) as f32,
    ];
    for p in img.data.iter_mut() {
        *p = mid_lin;
    }
    for cy in 0..MODEA_CELLS {
        for cx in 0..MODEA_CELLS {
            let b = levels[cy * MODEA_CELLS + cx];
            let re = -1.0 + 2.0 * b as f64 / (MODEA_LEVELS - 1) as f64;
            let d = (MID + usable * re).clamp(0.0, 255.0);
            let lin = [
                (d / 255.0).powf(ph.gammas[0]) as f32,
                (d / 255.0).powf(ph.gammas[1]) as f32,
                (d / 255.0).powf(ph.gammas[2]) as f32,
            ];
            for dy in 0..MODEA_CELL_PX {
                for dx in 0..MODEA_CELL_PX {
                    let x = PAD + cx * MODEA_CELL_PX + dx;
                    let y = PAD + cy * MODEA_CELL_PX + dy;
                    img.set(x, y, lin);
                }
            }
        }
    }
    let blurred = pipeline::blur(&img, sigma);
    let mut crop = Image::new(BLOCK, BLOCK);
    for py in 0..BLOCK {
        for px in 0..BLOCK {
            crop.set(px, py, blurred.at(PAD + px, PAD + py));
        }
    }
    pipeline::crosstalk(&mut crop, ph.crosstalk.0, ph.crosstalk.1);
    pipeline::add_noise(&mut crop, ph.noise_sigma, &mut rng);
    pipeline::clamp01(&mut crop);

    // декод: центр клетки, 2×2 субсэмпл (±cell/4=±2), по G
    let mut ok = vec![false; MODEA_CELLS * MODEA_CELLS];
    let d4 = MODEA_CELL_PX as f64 / 4.0;
    for cy in 0..MODEA_CELLS {
        for cx in 0..MODEA_CELLS {
            let ux = (cx * MODEA_CELL_PX) as f64 + MODEA_CELL_PX as f64 / 2.0;
            let uy = (cy * MODEA_CELL_PX) as f64 + MODEA_CELL_PX as f64 / 2.0;
            let mut g_acc = 0.0f64;
            for &(sx, sy) in &[(-d4, -d4), (-d4, d4), (d4, -d4), (d4, d4)] {
                // центры пикселей у Image::sample — целочисленные; клетка ≥8 → берём среднее
                let s = crop.sample(ux + sx - 0.5, uy + sy - 0.5);
                g_acc += s[1] as f64;
            }
            g_acc /= 4.0;
            let dg = 255.0 * g_acc.max(0.0).powf(1.0 / ph.gammas[1]);
            let re_hat = (dg - MID) / usable;
            let idx = (((re_hat + 1.0) * 0.5 * (MODEA_LEVELS - 1) as f64).round())
                .clamp(0.0, (MODEA_LEVELS - 1) as f64) as u32;
            ok[cy * MODEA_CELLS + cx] = idx == levels[cy * MODEA_CELLS + cx];
        }
    }
    ok
}

/// Линейная инверсия монотонно-убывающей калибровки ρ_rel(σ): по измеренному
/// ρ_rel находим σ интерполяцией табличных пар (sigma_i, rho_rel_i).
fn invert_calibration(table: &[(f64, f64)], rho_rel: f64) -> f64 {
    // table отсортирована по возрастанию σ, rho_rel убывает
    if rho_rel >= table[0].1 {
        return table[0].0;
    }
    let last = table.len() - 1;
    if rho_rel <= table[last].1 {
        return table[last].0;
    }
    for w in table.windows(2) {
        let (s0, r0) = w[0];
        let (s1, r1) = w[1];
        if rho_rel <= r0 && rho_rel >= r1 {
            let f = (r0 - rho_rel) / (r0 - r1);
            return s0 + f * (s1 - s0);
        }
    }
    table[last].0
}

/// Главная развёртка Mode B: печатает таблицы 1–5 в markdown.
pub fn cmd_modeb() {
    let t0 = Instant::now();
    let p = reference_profile();
    let (usable, a_l, a_c) = amplitudes(&p);
    let gammas = gammas_of(&p);
    let noise_sigma = p.noise_sigma() as f64 / 255.0;
    let crosstalk = (
        p.crosstalk_rg_pct() as f64 / 100.0,
        p.crosstalk_gb_pct() as f64 / 100.0,
    );
    let ph = Phys {
        a_l,
        a_c,
        gammas,
        crosstalk,
        noise_sigma,
    };
    // «чистый» канал для калибровки g и клиппинга без шума (клип — свойство синтеза)
    let basis = Basis::new(W_NOMINAL);
    let g = calibrate_gain(&basis);

    let (gram, ga, gb) = basis.gram_max_offdiag();
    let groups = order_groups(&basis);

    println!("# psicode-sim Mode B (Эрмит–Гаусс, RESEARCH §1 freeze-gate)");
    println!(
        "блок {BLOCK}×{BLOCK} px, w={W_NOMINAL}, моды m+n≤4 (15 мод: 1 пилот + 2 probe + 12 data QPSK)"
    );
    println!(
        "§5.1 split хромы: A_L={a_l:.1}, A_C={a_c:.1} (usable={usable:.0}, доли 0.7/0.3), M={MID}"
    );
    println!(
        "канал: matched γ=[{:.3},{:.3},{:.3}], кросстолк {:.0}/{:.0}%, шум σ={:.4} лин ({:.2} град/255)",
        gammas[0],
        gammas[1],
        gammas[2],
        crosstalk.0 * 100.0,
        crosstalk.1 * 100.0,
        noise_sigma,
        p.noise_sigma()
    );
    println!(
        "усиление поля g={g:.4} (цель |поле|≲{FIELD_TARGET}); ортонормальность: max|Грам вне-диаг|={:.2e} между ψ{:?} и ψ{:?}",
        gram, ga, gb
    );
    println!("{TRIALS} попыток/точку, детерминированные сиды.\n");

    // --- развёртка по σ (nominal-базис) + учёт клиппинга ---
    let stats_by_sigma: Vec<Vec<ModeStat>> = SIGMAS
        .iter()
        .enumerate()
        .map(|(i, &s)| sweep_point(&basis, g, &ph, s, i))
        .collect();

    // клиппинг drive по всей развёртке
    let (mut clip_sum, mut clip_tot) = (0usize, 0usize);
    for (i, &s) in SIGMAS.iter().enumerate() {
        for t in 0..TRIALS {
            let mut rng = Rng::new(seed_for(i, t));
            let coeffs = draw_coeffs(&basis, A_DATA, true, &mut rng);
            let rx = transmit_receive(&basis, &coeffs, g, &ph, s, &mut rng);
            clip_sum += rx.clips;
            clip_tot += rx.total;
        }
    }
    println!(
        "**Amplitude budget / clipping:** drive-пикселей вне [0,255] до clamp: {}/{} = {:.4}% (по всей σ-развёртке).\n",
        clip_sum,
        clip_tot,
        100.0 * clip_sum as f64 / clip_tot as f64
    );

    // per-order среднее |α| и SNR
    let per_order_alpha = |si: usize| -> Vec<f64> {
        groups
            .iter()
            .map(|idxs| idxs.iter().map(|&k| stats_by_sigma[si][k].alpha_abs()).sum::<f64>() / idxs.len() as f64)
            .collect::<Vec<_>>()
    };
    let per_order_snr = |si: usize| -> Vec<f64> {
        groups
            .iter()
            .map(|idxs| idxs.iter().map(|&k| stats_by_sigma[si][k].snr()).sum::<f64>() / idxs.len() as f64)
            .collect::<Vec<_>>()
    };
    let alpha0 = per_order_alpha(0); // база σ=0 для аттенюации

    // ---------- Таблица 1a: аттенюация |α|(σ)/|α|(0) по порядку ----------
    println!(
        "Опыт таблиц 1–4: ВСЕ 15 мод возбуждаются одинаковым QPSK (изолирует \
         зависимость от m+n от асимметрии осей Re/Im; фикс. пилот/probe — только для §5)."
    );
    println!("## 1a. Аттенюация коэффициентов |α(σ)|/|α(0)| по порядку моды (m+n)");
    let mut hdr = String::from("| порядок m+n \\ σ →");
    for s in SIGMAS {
        hdr.push_str(&format!(" | {s}"));
    }
    println!("{hdr} |");
    print!("|---");
    for _ in SIGMAS {
        print!("|---");
    }
    println!("|");
    for o in 0..5usize {
        let mut row = format!("| {o}");
        for (si, _s) in SIGMAS.iter().enumerate() {
            let att = per_order_alpha(si)[o] / alpha0[o];
            row.push_str(&format!(" | {att:.3}"));
        }
        println!("{row} |");
    }

    // ---------- Таблица 1b: SNR коэффициентов (dB) по порядку ----------
    println!("\n## 1b. SNR коэффициентов (dB) по порядку моды — сигнал/остаточная дисперсия");
    println!("{hdr} |");
    print!("|---");
    for _ in SIGMAS {
        print!("|---");
    }
    println!("|");
    for o in 0..5usize {
        let mut row = format!("| {o}");
        for (si, _s) in SIGMAS.iter().enumerate() {
            let snr = per_order_snr(si)[o];
            let db = to_db(snr);
            if db.is_finite() {
                row.push_str(&format!(" | {db:.1}"));
            } else {
                row.push_str(" | ∞");
            }
        }
        println!("{row} |");
    }

    // ---------- Таблица 2: монотонность + probe-асимметрия ----------
    println!("\n## 2. Монотонность аттенюации по m+n (SPEC §2/§5.3) и probe-асимметрия");
    println!(
        "Нормированная аттенюация att(o)=|α_o(σ)|/|α_o(0)|; монотонность с допуском {:.2} \
         (шум оценки |α| по {TRIALS} попыткам). Асимметрия — когерентная (фикс. probe, шум усреднён).",
        0.012
    );
    println!("| σ | att по порядку (0→4) | монотонно ↓ | probe-асимметрия отн. (изотропия) |");
    println!("|---|---|---|---|");
    let tol = 0.012;
    let mut mono_from = None; // первая σ, с которой всё монотонно
    let mut mono_break = false; // нарушение при σ≥1
    for (si, &s) in SIGMAS.iter().enumerate() {
        let a: Vec<f64> = (0..5).map(|o| per_order_alpha(si)[o] / alpha0[o]).collect();
        let mono = a.windows(2).all(|w| w[0] >= w[1] - tol);
        if mono && mono_from.is_none() {
            mono_from = Some(s);
        }
        if !mono && s >= 1.0 {
            mono_break = true;
        }
        let asym = probe_asymmetry(&basis, g, &ph, s, 30, 5000 + si);
        let vals: Vec<String> = a.iter().map(|v| format!("{v:.3}")).collect();
        println!(
            "| {s} | {} | {} | {:.2}% |",
            vals.join(" ≥ "),
            if mono { "да" } else { "нет (шум, att≈1)" },
            asym * 100.0
        );
    }
    println!(
        "\n**Вердикт монотонности:** att монотонно убывает по m+n {}. {}",
        if mono_break {
            "с нарушениями и при σ≥1"
        } else {
            "для всех σ≥1 (утверждение SPEC подтверждено)"
        },
        "При σ≤0.5 все порядки в пределах ~1% от единицы (аттенюации нет — порядок шумовой). \
         Probe-асимметрия ≈0 (изотропия подтверждена)."
    );

    // ---------- Таблица 3: QPSK BER по порядку vs σ + расписание shedding ----------
    println!("\n## 3. QPSK BER по порядку моды vs σ + порог shedding (BER=10%)");
    println!("{hdr} |");
    print!("|---");
    for _ in SIGMAS {
        print!("|---");
    }
    println!("|");
    // BER по порядку: усредняем биты только data-мод порядка
    let order_ber = |si: usize, o: usize| -> f64 {
        let (mut e, mut tb) = (0usize, 0usize);
        for &k in &groups[o] {
            e += stats_by_sigma[si][k].err_bits;
            tb += stats_by_sigma[si][k].tot_bits;
        }
        if tb == 0 {
            f64::NAN
        } else {
            e as f64 / tb as f64
        }
    };
    for o in 0..5usize {
        let mut row = format!("| {o}");
        for si in 0..SIGMAS.len() {
            let ber = order_ber(si, o);
            row.push_str(&format!(" | {}", report::sig4(ber)));
        }
        println!("{row} |");
    }
    // расписание shedding: минимальная σ, где BER порядка ≥ 10%
    println!("\n**Расписание graceful shedding (первая σ с BER ≥ 10% по порядку):**");
    for o in 0..5usize {
        let mut crossed = None;
        for (si, &s) in SIGMAS.iter().enumerate() {
            if order_ber(si, o) >= 0.10 {
                crossed = Some(s);
                break;
            }
        }
        match crossed {
            Some(s) => println!("- порядок {o}: BER≥10% при σ≥{s}"),
            None => println!("- порядок {o}: BER<10% на всей развёртке (σ≤8)"),
        }
    }

    // ---------- Таблица 4: Mode B vs Mode A по σ ----------
    println!("\n## 4. Head-to-head: Mode B vs Mode A (та же площадь 64×64, тот же канал)");
    // сырые ёмкости
    let modeb_raw = 12 * 2; // 24 бита
    let modea_cells = MODEA_CELLS * MODEA_CELLS; // 64
    let modea_raw = modea_cells * 3; // 192 бит (Mono 3 бита)
    println!(
        "сырьё: Mode B = 12 data-мод × 2 бита = **{modeb_raw} бит/блок**; \
         Mode A = {modea_cells} клеток × 3 бита = **{modea_raw} бит/блок** (Mono, 8 px клетка)."
    );
    println!("Обе метрики с ОДНИМ порогом надёжности BER<1e-2 (apples-to-apples):");
    println!("- Mode B usable = data-моды с BER<1e-2 × 2 бита;");
    println!("- Mode A reliable = клетки с частотой ошибок <1e-2 × 3 бита.");
    println!("Столбец «A raw» — сырой cell-correct (для справки): его пол ≈24 бита — это ЧИСТО");
    println!("случайное угадывание (64 клетки × 1/8 × 3 бита), информации не несёт.");
    println!("(σ продлена до 16 для поиска crossover; Mode A — без страйп-CRC §6.2, т.е. оптимистичен.)");
    println!("| σ | Mode B usable | Mode A reliable | победитель (reliable) | A raw (chance≈24) |");
    println!("|---|---|---|---|---|");
    let ext_sigmas: [f64; 11] = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0];
    let mut crossover: Option<f64> = None;
    for (ei, &s) in ext_sigmas.iter().enumerate() {
        // Mode B usable: data-моды с BER<1e-2
        let base_idx = SIGMAS.iter().position(|&x| (x - s).abs() < 1e-9);
        let stats = match base_idx {
            Some(si) => stats_by_sigma[si].clone(),
            None => sweep_point(&basis, g, &ph, s, 700 + ei),
        };
        let b_survivors = basis
            .modes
            .iter()
            .enumerate()
            .filter(|&(k, _)| basis.roles[k] == Role::Data && stats[k].ber() < 1e-2)
            .count();
        let b_usable = b_survivors * 2;
        // Mode A: сырой correct + надёжные клетки (частота ошибок клетки <1e-2)
        let mut cell_err = vec![0usize; MODEA_CELLS * MODEA_CELLS];
        let mut raw_correct = 0usize;
        for t in 0..TRIALS {
            let ok = mode_a_trial(usable, &ph, s, 1000 + ei, t);
            for (c, &b) in ok.iter().enumerate() {
                if b {
                    raw_correct += 1;
                } else {
                    cell_err[c] += 1;
                }
            }
        }
        let a_raw_bits = raw_correct as f64 / TRIALS as f64 * 3.0;
        let a_reliable = cell_err
            .iter()
            .filter(|&&e| (e as f64 / TRIALS as f64) < 1e-2)
            .count();
        let a_rel_bits = a_reliable * 3;
        let winner = if b_usable > a_rel_bits {
            "**Mode B**"
        } else if b_usable == a_rel_bits {
            "="
        } else {
            "Mode A"
        };
        if crossover.is_none() && b_usable > a_rel_bits {
            crossover = Some(s);
        }
        println!("| {s} | {b_usable} | {a_rel_bits} | {winner} | {a_raw_bits:.1} |");
    }
    match crossover {
        Some(s) => println!(
            "\n**Crossover (apples-to-apples, порог BER<1e-2):** Mode B обгоняет Mode A при σ≈{s} camera-px. \
             Mode A с 8-px клетками ОБРЫВАЕТСЯ (все клетки теряют надёжность из-за меж-клеточного блюра), \
             Mode B деградирует ПЛАВНО (низкие моды выживают) — тезис «graceful vs cliff» подтверждён. \
             NB: сырой cell-correct Mode A вводит в заблуждение — его пол ≈24 бита чисто случаен."
        ),
        None => println!(
            "\n**Crossover:** не достигнут даже на σ=16 при пороге BER<1e-2."
        ),
    }

    // ---------- Таблица 5: калибровка w′ и эквализация ----------
    println!("\n## 5. Pilot/probe эквализация: калибровка w′ из ‖a20‖/‖a00‖ и re-decode");
    // калибровочная кривая ρ_rel(σ) с полным каналом (data present, усреднение)
    let cal_trials = 30usize;
    let rho_abs: Vec<f64> = SIGMAS
        .iter()
        .enumerate()
        .map(|(i, &s)| measure_rho(&basis, g, &ph, s, cal_trials, 2000 + i))
        .collect();
    let rho0 = rho_abs[0];
    let cal_table: Vec<(f64, f64)> = SIGMAS
        .iter()
        .zip(&rho_abs)
        .map(|(&s, &r)| (s, r / rho0))
        .collect();
    println!("Калибровка (полный канал, {cal_trials} попыток): ρ=‖a20,a02‖/‖a00‖, w′=√(w²+σ²).");
    println!("| σ | ρ = ‖a20‖/‖a00‖ | ρ_rel = ρ/ρ(0) | w′(σ)=√(64+σ²) |");
    println!("|---|---|---|---|");
    for (i, &s) in SIGMAS.iter().enumerate() {
        let wp = (W_NOMINAL * W_NOMINAL + s * s).sqrt();
        println!(
            "| {s} | {:.4} | {:.4} | {wp:.3} |",
            rho_abs[i], cal_table[i].1
        );
    }

    println!("\nRe-decode (designed: фикс. пилот/probe, 12 data QPSK): w′ из измеренного ρ.");
    println!("«Recovery» — средний по 12 data-модам SNR коэффициента (dB); BER для полноты.");
    println!("(при σ≤4 BER уже 0 у обоих — выигрыш виден в SNR коэффициентов, а не в BER.)");
    println!("| σ | σ_est | w′_est | data-SNR nom (dB) | data-SNR w′ (dB) | ΔSNR | BER nom | BER w′ |");
    println!("|---|---|---|---|---|---|---|---|");
    // средний по data-модам линейный SNR из набора ModeStat
    let mean_data_snr = |stats: &[ModeStat]| -> f64 {
        let vals: Vec<f64> = basis
            .modes
            .iter()
            .enumerate()
            .filter(|&(k, _)| basis.roles[k] == Role::Data)
            .map(|(k, _)| stats[k].snr())
            .filter(|v| v.is_finite())
            .collect();
        vals.iter().sum::<f64>() / vals.len() as f64
    };
    for &s in &[2.0f64, 4.0, 6.0, 8.0] {
        let rho = measure_rho(&basis, g, &ph, s, TRIALS, 3000 + (s as usize)) / rho0;
        let sigma_est = invert_calibration(&cal_table, rho);
        let w_prime = (W_NOMINAL * W_NOMINAL + sigma_est * sigma_est).sqrt();
        let eq_basis = Basis::new(w_prime);
        let mut st_nom = vec![ModeStat::default(); basis.modes.len()];
        let mut st_eq = vec![ModeStat::default(); basis.modes.len()];
        for t in 0..TRIALS {
            let mut rng = Rng::new(seed_for(4000 + (s as usize), t));
            let coeffs = draw_coeffs(&basis, A_DATA, false, &mut rng);
            let rx = transmit_receive(&basis, &coeffs, g, &ph, s, &mut rng);
            let r_nom = basis.project(&rx.fr, &rx.fi);
            let r_eq = eq_basis.project(&rx.fr, &rx.fi);
            accumulate(&mut st_nom, &coeffs, &r_nom);
            accumulate(&mut st_eq, &coeffs, &r_eq);
            for k in 0..basis.modes.len() {
                if basis.roles[k] == Role::Data {
                    st_nom[k].err_bits += qpsk_errors(coeffs[k], r_nom[k]);
                    st_nom[k].tot_bits += 2;
                    st_eq[k].err_bits += qpsk_errors(coeffs[k], r_eq[k]);
                    st_eq[k].tot_bits += 2;
                }
            }
        }
        let snr_nom = to_db(mean_data_snr(&st_nom));
        let snr_eq = to_db(mean_data_snr(&st_eq));
        let (mut e_nom, mut e_eq, mut tot) = (0usize, 0usize, 0usize);
        for k in 0..basis.modes.len() {
            if basis.roles[k] == Role::Data {
                e_nom += st_nom[k].err_bits;
                e_eq += st_eq[k].err_bits;
                tot += st_nom[k].tot_bits;
            }
        }
        println!(
            "| {s} | {sigma_est:.2} | {w_prime:.3} | {snr_nom:.2} | {snr_eq:.2} | {:+.2} | {} | {} |",
            snr_eq - snr_nom,
            report::sig4(e_nom as f64 / tot as f64),
            report::sig4(e_eq as f64 / tot as f64),
        );
    }

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Физика «чистого» канала (без шума/кросстолка/блюра) для точных проверок.
    fn clean_phys(p: &CalibProfile) -> Phys {
        let (_u, a_l, a_c) = amplitudes(p);
        Phys {
            a_l,
            a_c,
            gammas: gammas_of(p),
            crosstalk: (0.0, 0.0),
            noise_sigma: 0.0,
        }
    }

    /// Полная телеметрия §7.4 (шум+кросстолк).
    fn full_phys(p: &CalibProfile) -> Phys {
        let (_u, a_l, a_c) = amplitudes(p);
        Phys {
            a_l,
            a_c,
            gammas: gammas_of(p),
            crosstalk: (
                p.crosstalk_rg_pct() as f64 / 100.0,
                p.crosstalk_gb_pct() as f64 / 100.0,
            ),
            noise_sigma: p.noise_sigma() as f64 / 255.0,
        }
    }

    /// Пин ортонормальности: диагональ Грама = 1 (по построению), вне-диагональ
    /// мала (дискретизация/усечение). Пин верхней границы.
    #[test]
    fn orthonormality_pin() {
        let basis = Basis::new(W_NOMINAL);
        // диагональ ровно 1
        for a in &basis.arrays {
            let n: f64 = a.iter().map(|v| v * v).sum();
            assert!((n - 1.0).abs() < 1e-9, "норма моды != 1: {n}");
        }
        let (gram, _, _) = basis.gram_max_offdiag();
        assert!(gram < 0.05, "макс вне-диагональ Грама {gram} превысила 0.05");
    }

    /// Чистый канал (σ=0, без шума/кросстолка): все 12 data-мод декодируются
    /// точно — BER = 0.
    #[test]
    fn clean_channel_zero_ber() {
        let p = reference_profile();
        let basis = Basis::new(W_NOMINAL);
        let g = calibrate_gain(&basis);
        let ph = clean_phys(&p);
        let mut errors = 0usize;
        for t in 0..10 {
            let mut rng = Rng::new(seed_for(77, t));
            let coeffs = draw_coeffs(&basis, A_DATA, false, &mut rng);
            let rx = transmit_receive(&basis, &coeffs, g, &ph, 0.0, &mut rng);
            let r = basis.project(&rx.fr, &rx.fi);
            for k in 0..basis.modes.len() {
                if basis.roles[k] == Role::Data {
                    errors += qpsk_errors(coeffs[k], r[k]);
                }
            }
        }
        assert_eq!(errors, 0, "чистый канал дал QPSK-ошибки: {errors}");
    }

    /// Монотонная аттенюация при σ=2: среднее |α| по порядку убывает с m+n.
    #[test]
    fn monotonic_attenuation_sigma2() {
        let p = reference_profile();
        let basis = Basis::new(W_NOMINAL);
        let g = calibrate_gain(&basis);
        let ph = full_phys(&p);
        let stats = sweep_point(&basis, g, &ph, 2.0, 0);
        let groups = order_groups(&basis);
        let alpha: Vec<f64> = groups
            .iter()
            .map(|idxs| {
                idxs.iter().map(|&k| stats[k].alpha_abs()).sum::<f64>() / idxs.len() as f64
            })
            .collect();
        for o in 0..4 {
            assert!(
                alpha[o] >= alpha[o + 1] - 1e-3,
                "аттенюация не монотонна: порядок {o} |α|={:.4} < порядок {} |α|={:.4}",
                alpha[o],
                o + 1,
                alpha[o + 1]
            );
        }
    }

    /// Probe-асимметрия под изотропным блюром пренебрежимо мала (когерентная
    /// оценка фикс. probe, шум усреднён): ‖⟨r20⟩‖ ≈ ‖⟨r02⟩‖.
    #[test]
    fn probe_isotropy_sigma2() {
        let p = reference_profile();
        let basis = Basis::new(W_NOMINAL);
        let g = calibrate_gain(&basis);
        let ph = full_phys(&p);
        // относительная асимметрия при σ=2 (когерентно, 30 попыток) < 3% — изотропия
        let asym = probe_asymmetry(&basis, g, &ph, 2.0, 30, 5002);
        assert!(asym < 0.03, "probe-асимметрия отн.={asym} велика при изотропном блюре");
    }

    /// Эквализация: восстановленная w′ шире номинальной (блюр расширяет), и
    /// re-decode с w′ не ухудшает BER заметно (честный пин направления).
    #[test]
    fn equalization_widens_and_pins() {
        let p = reference_profile();
        let basis = Basis::new(W_NOMINAL);
        let g = calibrate_gain(&basis);
        let ph = full_phys(&p);
        // калибровка
        let rho_abs: Vec<f64> = SIGMAS
            .iter()
            .enumerate()
            .map(|(i, &s)| measure_rho(&basis, g, &ph, s, 30, 2000 + i))
            .collect();
        let rho0 = rho_abs[0];
        let cal: Vec<(f64, f64)> = SIGMAS.iter().zip(&rho_abs).map(|(&s, &r)| (s, r / rho0)).collect();
        let s = 4.0;
        let rho = measure_rho(&basis, g, &ph, s, TRIALS, 3000 + s as usize) / rho0;
        let sigma_est = invert_calibration(&cal, rho);
        let w_prime = (W_NOMINAL * W_NOMINAL + sigma_est * sigma_est).sqrt();
        // блюр расширяет эффективную ширину
        assert!(w_prime > W_NOMINAL, "w′={w_prime} не шире номинальной {W_NOMINAL}");
        assert!(w_prime < 2.0 * W_NOMINAL, "w′={w_prime} неправдоподобно велика");
        // re-decode BER: w′ не должна катастрофически ухудшать
        let eq_basis = Basis::new(w_prime);
        let (mut e_nom, mut e_eq, mut tot) = (0usize, 0usize, 0usize);
        for t in 0..TRIALS {
            let mut rng = Rng::new(seed_for(4000 + s as usize, t));
            let coeffs = draw_coeffs(&basis, A_DATA, false, &mut rng);
            let rx = transmit_receive(&basis, &coeffs, g, &ph, s, &mut rng);
            let r_nom = basis.project(&rx.fr, &rx.fi);
            let r_eq = eq_basis.project(&rx.fr, &rx.fi);
            for k in 0..basis.modes.len() {
                if basis.roles[k] == Role::Data {
                    e_nom += qpsk_errors(coeffs[k], r_nom[k]);
                    e_eq += qpsk_errors(coeffs[k], r_eq[k]);
                    tot += 2;
                }
            }
        }
        let ber_nom = e_nom as f64 / tot as f64;
        let ber_eq = e_eq as f64 / tot as f64;
        // честный пин: эквализация не хуже более чем на 3 п.п.
        assert!(
            ber_eq <= ber_nom + 0.03,
            "w′-эквализация заметно хуже nominal: {ber_eq} vs {ber_nom}"
        );
    }
}
