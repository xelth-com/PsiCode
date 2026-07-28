//! [ДИАГНОСТИКА] Где на самом деле лежит пол размера клетки.
//!
//! Передатчик отказывается рисовать клетку меньше 8 display-px, ссылаясь на
//! `symbol::sample_cell` («ниже 8 теряется 2×2 субсэмплинг»). Ссылка неверна по
//! построению: `sample_cell` смотрит на `profile.cell_size_px` (у приёмника это
//! всегда 16), а НЕ на клетку показа передатчика. Значит пол ничем не обоснован
//! и его надо ЗАМЕРИТЬ.
//!
//! # Что разделяется
//!
//! Две РАЗНЫЕ величины, которые живой стенд склеивает (у A22 камерных px на
//! клетку ≈ 1.06 × клетки показа, поэтому в ночной матрице они неразличимы):
//!
//! * **display px/клетку** `D` — свободный параметр передатчика. Влияет только
//!   через апертуру пикселя панели (≈ 1 display px), потому что растеризация
//!   клетки ТОЧНАЯ: клетка — сплошной квадрат ровно `D×D` px при любом `D ≥ 1`.
//! * **камерных px/клетку** `ppc = D · m` — задаётся стендом (расстояние,
//!   объектив, сенсор). Расфокус оптики σ фиксирован в КАМЕРНЫХ px, поэтому
//!   именно `σ/ppc` определяет межклеточную интерференцию.
//!
//! Развёртка `sep` гоняет `D` при ФИКСИРОВАННОМ `ppc` (меняя увеличение `m`) —
//! это и есть контрольный опыт: если кривая плоская по `D`, клетка показа не
//! ограничивает вообще, и пол передатчика должен уйти к геометрическому
//! минимуму рендера.
//!
//! # Канал
//!
//! Замеренный тракт «монитор -> Galaxy A22» (см. RESEARCH): поканальная гамма
//! тракта 2.0/2.2/3.2, расфокус σ ≈ 2 камерных px, ISP-НЧ по цветности σ ≈ 3.2
//! камерных px, поле освещённости 0.62…0.86, шум в КОДАХ камеры и
//! УРОВНЕ-ЗАВИСИМЫЙ (чёрное 2.12, белое 1.51 кода из 255 — обратно дробовому).
//! Шум масштабируется `--noise` , потому что A22 снимал на ISO ≈ 5000, а 1.79
//! кода намерены на Note 10 Lite при ISO 100…250.
//!
//! Тракт СОБИРАЕТСЯ в кодах камеры именно потому, что приёмник читает 8-битные
//! коды и линеаризует их сам: размывать надо СВЕТ, а шуметь — КОД.
//!
//! запуск: `cargo run --release -p psicode-sim -- cellfloor <sep|floor|aperture|anchor>`

use crate::image::Image;
use crate::pipeline::{blur, crosstalk, surviving_payload_bits};
use crate::rng::{seed_for, Rng};
use psicode_core::profile::BorderMode;
use psicode_core::symbol::{
    self, demod_symbol_isi, demod_symbol_local, demod_symbol_local_isi, IsiConfig,
};
use psicode_core::{CalibProfile, ChromaMode};

// ---------------------------------------------------------------------------
// Канал
// ---------------------------------------------------------------------------

/// Замеренный тракт «дисплей -> A22».
#[derive(Clone, Copy, Debug)]
pub struct Live {
    /// Поканальная гамма тракта (та, которую приёмник оценивает по реф-строке).
    pub gamma: [f64; 3],
    /// Расфокус оптики, КАМЕРНЫХ px (одинаков для всех размеров клетки).
    pub sigma_opt: f64,
    /// Суммарная НЧ по цветности ISP, камерных px (≥ `sigma_opt`).
    ///
    /// σ ≈ 3.2 из сводки ночного прогона относится к ЗЧ-РАМКЕ в цветности (ей
    /// нужна точность в доли клетки по всей стороне) и для нагрузки оказалась
    /// завышенной: при 3.2 сим не читает ConstLuma1 даже на 12.8 камерных
    /// px/клетку, тогда как живой A22 читает на 8.5. Подгонка по трём живым
    /// точкам (8.46/10.72/12.77 камерных px, ISI выкл и вкл) даёт 2.2 — при нём
    /// сим воспроизводит рабочую точку (0.67/0.96 против живых 0.68/0.85
    /// условных) и остаётся КОНСЕРВАТИВНЫМ ниже неё (0.04 против живых 0.68 на
    /// 8.46). Пол, снятый на таком канале, — оценка СВЕРХУ.
    pub sigma_chroma: f64,
    /// Множитель шума кодов относительно замера Note 10 Lite.
    pub noise_scale: f64,
    /// Поле освещённости 0.62…0.86.
    pub field: bool,
    /// Кросстолк каналов из профиля (6 % R<->G, 8 % G<->B).
    pub crosstalk: bool,
    /// ПОКАДРОВЫЙ разброс расфокуса (доля от `sigma_opt`): автофокус дышит,
    /// стол/рука вибрируют, экспозиция гуляет. Без него переход «работает ->
    /// не работает» в симе получается ступенькой, а живьём он растянут.
    pub sigma_jit: f64,
    /// Ошибка геометрии захвата: сдвиг карты приёмника, σ в КАМЕРНЫХ px.
    /// Захват локализует рамку с точностью долей камерного пикселя, и эта
    /// точность НЕ уменьшается вместе с клеткой — в долях клетки она растёт.
    pub jitter_px: f64,
}

impl Default for Live {
    fn default() -> Self {
        Live {
            gamma: [2.0, 2.2, 3.2],
            sigma_opt: 2.0,
            sigma_chroma: 2.2,
            noise_scale: 2.6,
            field: true,
            crosstalk: true,
            jitter_px: 0.0,
            sigma_jit: 0.15,
        }
    }
}

/// Шум кода камеры на уровне `code` (0..255), в кодах: замер per-cell
/// одиночного снимка — чёрное 2.12, белое 1.51 (уровне-зависимый и ОБРАТНЫЙ
/// дробовому, т.е. ISP давит шум в светах).
fn code_sigma(code: f64) -> f64 {
    2.12 + (1.51 - 2.12) * (code / 255.0).clamp(0.0, 1.0)
}

/// Снимок: 8-битные коды камеры + геометрия (камерных px на клетку и начало).
pub struct Capture {
    pub code: Vec<[u8; 3]>,
    pub w: usize,
    pub h: usize,
    /// Камерных px на клетку символа.
    pub ppc: f64,
    /// Непрерывная камерная координата левого верхнего угла символа.
    pub ox: f64,
    pub oy: f64,
}

impl Capture {
    /// Билинейный сырой отсчёт [0,1] по ИНДЕКСНОЙ координате (как у приёмника).
    #[inline]
    pub fn raw(&self, x: f64, y: f64) -> [f32; 3] {
        let xc = x.clamp(0.0, (self.w - 1) as f64);
        let yc = y.clamp(0.0, (self.h - 1) as f64);
        let x0 = xc.floor() as usize;
        let y0 = yc.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = (xc - x0 as f64) as f32;
        let fy = (yc - y0 as f64) as f32;
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            let a = self.code[y0 * self.w + x0][c] as f32 * (1.0 - fx)
                + self.code[y0 * self.w + x1][c] as f32 * fx;
            let b = self.code[y1 * self.w + x0][c] as f32 * (1.0 - fx)
                + self.code[y1 * self.w + x1][c] as f32 * fx;
            o[c] = (a * (1.0 - fy) + b * fy) / 255.0;
        }
        o
    }
}

/// Полный тракт: рендер профилем `p_tx` (его `cell_size_px` — клетка ПОКАЗА) ->
/// свет дисплея -> варп с увеличением `mag` -> расфокус -> НЧ цветности ->
/// кросстолк -> поле освещённости -> тон камеры -> шум кода -> квантование.
pub fn capture(
    p_tx: &CalibProfile,
    cells: &[u8],
    counter: u8,
    mag: f64,
    ch: &Live,
    rng: &mut Rng,
) -> Capture {
    let frame = symbol::render_symbol_counter(p_tx, cells, counter);
    let n = frame.size_px;
    let d = p_tx.cell_size_px as f64;
    let ppc = d * mag;

    // 1. свет дисплея (линейный): L = (drive/255)^γ
    let mut lut = [[0.0f32; 256]; 3];
    for (c, &g) in ch.gamma.iter().enumerate() {
        for v in 0..256 {
            lut[c][v] = (v as f64 / 255.0).powf(g) as f32;
        }
    }
    let mut disp = Image::new(n, n);
    for (dst, px) in disp.data.iter_mut().zip(frame.rgb.iter()) {
        *dst = [
            lut[0][px[0] as usize],
            lut[1][px[1] as usize],
            lut[2][px[2] as usize],
        ];
    }

    // 2. варп в сетку камеры. Соглашение: непрерывная координата, пиксель с
    //    индексом i покрывает [i, i+1), поэтому центр индекса i равен i+0.5.
    let pad = (6.0 * ch.sigma_chroma).max(8.0).ceil();
    let ox = pad + rng.next_f64();
    let oy = pad + rng.next_f64();
    let w = ((n as f64) * mag + 2.0 * pad).ceil() as usize + 2;
    let h = w;
    let bg = [
        (0.5f64).powf(ch.gamma[0]) as f32,
        (0.5f64).powf(ch.gamma[1]) as f32,
        (0.5f64).powf(ch.gamma[2]) as f32,
    ];
    let mut img = Image::new(w, h);
    for y in 0..h {
        let vc = ((y as f64 + 0.5) - oy) / mag;
        for x in 0..w {
            let uc = ((x as f64 + 0.5) - ox) / mag;
            let px = if uc < 0.0 || vc < 0.0 || uc >= n as f64 || vc >= n as f64 {
                bg
            } else {
                disp.sample(uc - 0.5, vc - 0.5)
            };
            img.set(x, y, px);
        }
    }

    // 3. расфокус оптики (все каналы) и ДОПОЛНИТЕЛЬНАЯ НЧ по цветности ISP.
    let kf = if ch.sigma_jit > 0.0 {
        (1.0 + ch.sigma_jit * rng.gaussian()).clamp(0.4, 2.0)
    } else {
        1.0
    };
    let s_opt = ch.sigma_opt * kf;
    let s_chr = ch.sigma_chroma * kf;
    img = blur(&img, s_opt);
    let extra = (s_chr * s_chr - s_opt * s_opt).max(0.0);
    if extra > 1e-6 {
        chroma_lowpass(&mut img, extra.sqrt());
    }

    // 4. кросстолк каналов
    if ch.crosstalk {
        crosstalk(&mut img, 0.06, 0.08);
    }

    // 5. поле освещённости + 6. тон камеры + 7. шум кода + квантование
    let mut code = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let f = if ch.field {
                let dx = x as f64 / w as f64 - 0.45;
                let dy = y as f64 / h as f64 - 0.40;
                0.62 + 0.24 * (-(dx * dx + dy * dy) * 4.0).exp()
            } else {
                1.0
            };
            let p = img.at(x, y);
            for c in 0..3 {
                let l = (p[c] as f64 * f).max(0.0);
                let v = 255.0 * l.powf(1.0 / ch.gamma[c]);
                let v = v + rng.gaussian() * code_sigma(v) * ch.noise_scale;
                code[y * w + x][c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Capture { code, w, h, ppc, ox, oy }
}

/// ДОПОЛНИТЕЛЬНАЯ низкочастотная фильтрация ЦВЕТНОСТИ (ISP): яркость остаётся,
/// цветоразностные плоскости размываются ещё на σ.
fn chroma_lowpass(img: &mut Image, sigma: f64) {
    let n = img.w * img.h;
    let mut y = vec![0.0f32; n];
    for (i, p) in img.data.iter().enumerate() {
        y[i] = 0.25 * p[0] + 0.5 * p[1] + 0.25 * p[2];
    }
    let mut cd = Image::new(img.w, img.h);
    for i in 0..n {
        let p = img.data[i];
        cd.data[i] = [p[0] - y[i], p[1] - y[i], p[2] - y[i]];
    }
    let cd = blur(&cd, sigma);
    for i in 0..n {
        let c = cd.data[i];
        img.data[i] = [y[i] + c[0], y[i] + c[1], y[i] + c[2]];
    }
}

// ---------------------------------------------------------------------------
// Приёмник
// ---------------------------------------------------------------------------

/// Форма апертуры отсчёта клетки. Реализуется СНАРУЖИ `sample_cell`: `map`
/// защёлкивает координату на центр клетки (все четыре тапа §symbol сливаются в
/// один), а интегрирование по апертуре делает `sample` уже в камерных px.
/// `Quad2x2`, взятая через ту же защёлку, воспроизводит производственный путь.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Aperture {
    /// Производственная: 4 тапа в ±клетка/4, билинейка.
    Quad2x2,
    /// Один билинейный отсчёт в центре клетки.
    Point,
    /// Равномерное среднее по квадрату шириной `f` клеток (k×k тапов).
    Box(f64),
    /// Гауссово взвешивание с σ = `f` клеток (обрезка 2σ, 9×9 тапов).
    Gauss(f64),
    /// 3×3 тапа в ±клетка/3.
    Grid3,
    /// 4 тапа в ±`f` клетки — параметрическое обобщение производственной схемы
    /// (`f = 0.25` даёт её ровно). `f = 0.1443 = 0.25/√3` — двухточечная
    /// квадратура Гаусса–Лежандра для бокса шириной полклетки, то есть ТОТ ЖЕ
    /// интеграл, что `Box(0.5)`, но за 4 отсчёта вместо 49.
    Quad(f64),
}

impl Aperture {
    fn name(self) -> String {
        match self {
            Aperture::Quad2x2 => "2x2 ±1/4 (сейчас)".into(),
            Aperture::Point => "точка".into(),
            Aperture::Box(f) => format!("бокс {f:.2} клетки"),
            Aperture::Gauss(f) => format!("гаусс σ={f:.2} кл"),
            Aperture::Grid3 => "3x3 ±1/3".into(),
            Aperture::Quad(f) => format!("2x2 ±{f:.4}"),
        }
    }

    /// Тапы апертуры в долях клетки: (dx, dy, вес). Веса нормированы.
    fn taps(self) -> Vec<(f64, f64, f64)> {
        match self {
            Aperture::Quad2x2 => vec![
                (-0.25, -0.25, 0.25),
                (-0.25, 0.25, 0.25),
                (0.25, -0.25, 0.25),
                (0.25, 0.25, 0.25),
            ],
            Aperture::Point => vec![(0.0, 0.0, 1.0)],
            Aperture::Quad(f) => vec![
                (-f, -f, 0.25),
                (-f, f, 0.25),
                (f, -f, 0.25),
                (f, f, 0.25),
            ],
            Aperture::Grid3 => {
                let mut v = Vec::new();
                for iy in -1..=1 {
                    for ix in -1..=1 {
                        v.push((ix as f64 / 3.0, iy as f64 / 3.0, 1.0 / 9.0));
                    }
                }
                v
            }
            Aperture::Box(f) => {
                let k = 7usize;
                let mut v = Vec::new();
                for iy in 0..k {
                    for ix in 0..k {
                        let dx = (ix as f64 + 0.5) / k as f64 - 0.5;
                        let dy = (iy as f64 + 0.5) / k as f64 - 0.5;
                        v.push((dx * f, dy * f, 1.0 / (k * k) as f64));
                    }
                }
                v
            }
            Aperture::Gauss(s) => {
                let k = 9usize;
                let mut v = Vec::new();
                let mut sum = 0.0;
                for iy in 0..k {
                    for ix in 0..k {
                        let dx = ((ix as f64) - (k as f64 - 1.0) / 2.0) * (4.0 * s / (k as f64 - 1.0));
                        let dy = ((iy as f64) - (k as f64 - 1.0) / 2.0) * (4.0 * s / (k as f64 - 1.0));
                        let wgt = (-(dx * dx + dy * dy) / (2.0 * s * s)).exp();
                        sum += wgt;
                        v.push((dx, dy, wgt));
                    }
                }
                for t in &mut v {
                    t.2 /= sum;
                }
                v
            }
        }
    }
}

/// Итог одной попытки приёма.
#[derive(Clone, Copy, Default, Debug)]
pub struct Shot {
    pub quality: [f64; 2],
    pub wrong: usize,
    pub total: usize,
    pub alive_stripes: usize,
    pub surviving_bits: usize,
    pub total_bits: usize,
}

/// Приём одного снимка ЖИВЫМ трактом приёмника (genie-геометрия, чтобы
/// отделить демодуляцию от захвата) с заданной апертурой.
#[allow(clippy::too_many_arguments)]
pub fn receive(
    p_rx: &CalibProfile,
    cap: &Capture,
    sent: &[u8],
    isi: bool,
    ap: Aperture,
    gamma: [f64; 3],
    jitter: (f64, f64),
) -> Shot {
    let cell_rx = p_rx.cell_size_px as f64;
    let taps = ap.taps();
    let gf = [gamma[0] as f32, gamma[1] as f32, gamma[2] as f32];

    // map: защёлка на центр клетки -> камерная ИНДЕКСНАЯ координата.
    let map = |u: f64, v: f64| {
        let cu = (u / cell_rx).floor() + 0.5;
        let cv = (v / cell_rx).floor() + 0.5;
        (
            cap.ox + cu * cap.ppc - 0.5 + jitter.0,
            cap.oy + cv * cap.ppc - 0.5 + jitter.1,
        )
    };
    let raw_sample = |x: f64, y: f64| -> [f32; 3] {
        let mut acc = [0.0f32; 3];
        for &(dx, dy, wgt) in &taps {
            let p = cap.raw(x + dx * cap.ppc, y + dy * cap.ppc);
            for c in 0..3 {
                acc[c] += p[c] * wgt as f32;
            }
        }
        acc
    };
    let lin_sample = |x: f64, y: f64| -> [f32; 3] {
        let r = raw_sample(x, y);
        [
            r[0].max(0.0).powf(gf[0]),
            r[1].max(0.0).powf(gf[1]),
            r[2].max(0.0).powf(gf[2]),
        ]
    };

    let mono = p_rx.luma_bits == 1 && p_rx.chroma_bits() == 0;
    let mut quality = [f64::NAN; 2];
    let got = if isi {
        let cfg = IsiConfig::default();
        if mono {
            demod_symbol_local_isi(p_rx, &map, &raw_sample, &cfg).cells
        } else {
            let r = demod_symbol_isi(p_rx, &map, &lin_sample, None, &cfg);
            quality = r.quality;
            r.cells
        }
    } else if mono {
        demod_symbol_local(p_rx, &map, &raw_sample)
    } else {
        let cfg = IsiConfig { kernel: Some([psicode_core::isi::IsiKernel::identity(1); 3]), ..IsiConfig::default() };
        let r = demod_symbol_isi(p_rx, &map, &lin_sample, None, &cfg);
        quality = r.quality;
        r.base_cells
    };

    let bpc = symbol::bits_per_cell(p_rx);
    let wrong = sent.iter().zip(&got).filter(|(a, b)| a != b).count();
    let (bits, _dead) = surviving_payload_bits(sent, &got, bpc);
    let alive = alive_stripe_count(sent, &got);
    Shot {
        quality,
        wrong,
        total: sent.len(),
        alive_stripes: alive,
        surviving_bits: bits,
        total_bits: sent.len() * bpc as usize,
    }
}

/// Сколько из восьми страйпов §6.2 прошли бы CRC (ни одной ошибочной клетки).
fn alive_stripe_count(sent: &[u8], got: &[u8]) -> usize {
    let cols = symbol::PAYLOAD_COLS;
    let mut row0 = 0usize;
    let mut n = 0usize;
    for &rows in &crate::pipeline::STRIPE_ROWS {
        let (a, b) = (row0 * cols, (row0 + rows) * cols);
        if sent[a..b].iter().zip(&got[a..b]).all(|(x, y)| x == y) {
            n += 1;
        }
        row0 += rows;
    }
    n
}

// ---------------------------------------------------------------------------
// Профили и прогон точки
// ---------------------------------------------------------------------------

/// Профиль передатчика ночной матрицы: рамка v1, 1 бит яркости (`v1`) или
/// постоянная яркость 2 бита (`v1c`).
pub fn tx_profile(chroma: bool, cell: usize) -> CalibProfile {
    let mut p = CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: cell as u8,
        frame_hold_periods: 6,
        luma_bits: 1,
        chroma_mode: ChromaMode::Mono,
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
        border: BorderMode::ExtrudedStrips,
    };
    if chroma {
        p.chroma_mode = ChromaMode::ConstLuma1;
    }
    p
}

/// Профиль ПРИЁМНИКА: тот же, но `cell_size_px` = 16, как в живом приёмнике
/// (он не знает клетки показа и не обязан её знать).
pub fn rx_profile(chroma: bool) -> CalibProfile {
    tx_profile(chroma, 16)
}

/// Усреднённый результат точки развёртки.
#[derive(Clone, Copy, Default, Debug)]
pub struct Point {
    pub ser: f64,
    /// Доля кадров, где живы ВСЕ восемь страйпов.
    pub full: f64,
    /// Среднее число живых страйпов.
    pub stripes: f64,
    /// Доля дошедших payload-бит (goodput с учётом выживания страйпов).
    pub goodput: f64,
    /// Доставленных бит на клетку.
    pub bits_per_cell: f64,
    /// Доставленных бит на 1000 display-px² полотна.
    pub bits_per_kpx: f64,
    /// Разделение осей созвездия в σ (только семейство постоянной яркости).
    pub quality: [f64; 2],
}

/// Прогон одной точки: `trials` кадров.
#[allow(clippy::too_many_arguments)]
pub fn run_point(
    chroma: bool,
    d: usize,
    mag: f64,
    isi: bool,
    ap: Aperture,
    ch: &Live,
    trials: usize,
    point_idx: usize,
) -> Point {
    let p_tx = tx_profile(chroma, d);
    let p_rx = rx_profile(chroma);
    let bpc = symbol::bits_per_cell(&p_rx);
    let n_cells = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
    let levels = 1u32 << bpc;

    let mut acc = Point::default();
    for t in 0..trials {
        let mut rng = Rng::new(seed_for(point_idx, t));
        let cells: Vec<u8> = (0..n_cells)
            .map(|_| rng.next_u32_below(levels) as u8)
            .collect();
        let cap = capture(&p_tx, &cells, (t & 0xff) as u8, mag, ch, &mut rng);
        let jit = if ch.jitter_px > 0.0 {
            (
                rng.gaussian() * ch.jitter_px,
                rng.gaussian() * ch.jitter_px,
            )
        } else {
            (0.0, 0.0)
        };
        let sh = receive(&p_rx, &cap, &cells, isi, ap, ch.gamma, jit);
        acc.ser += sh.wrong as f64 / sh.total as f64;
        acc.full += (sh.alive_stripes == 8) as u8 as f64;
        acc.stripes += sh.alive_stripes as f64;
        acc.goodput += sh.surviving_bits as f64 / sh.total_bits as f64;
        if sh.quality[0].is_finite() {
            acc.quality[0] += sh.quality[0];
            acc.quality[1] += sh.quality[1];
        }
    }
    let n = trials as f64;
    acc.ser /= n;
    acc.full /= n;
    acc.stripes /= n;
    acc.goodput /= n;
    acc.quality[0] /= n;
    acc.quality[1] /= n;
    acc.bits_per_cell = acc.goodput * bpc as f64;
    let side_px = (symbol::GRID * d) as f64;
    acc.bits_per_kpx = acc.goodput * (n_cells * bpc as usize) as f64 / (side_px * side_px) * 1000.0;
    acc
}

// ---------------------------------------------------------------------------
// Развёртки
// ---------------------------------------------------------------------------

/// Живые опорные точки A22 (ночная матрица, `results.md`): клетка показа ->
/// (камерных px/клетку, доля кадров с восемью страйпами, ISI).
const ANCHORS: &[(usize, f64, f64, bool)] = &[
    (8, 8.46, 478.0 / 812.0, false),
    (8, 8.46, 532.0 / 796.0, true),
    (10, 10.72, 691.0 / 869.0, false),
    (10, 10.57, 898.0 / 1186.0, true),
    (12, 12.77, 551.0 / 729.0, false),
    (14, 15.03, 455.0 / 683.0, false),
];

pub fn cmd(args: &[String]) {
    let what = args.first().map(|s| s.as_str()).unwrap_or("floor");
    let trials: usize = args
        .iter()
        .position(|a| a == "--trials")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let noise: f64 = args
        .iter()
        .position(|a| a == "--noise")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(Live::default().noise_scale);
    let sigma: f64 = args
        .iter()
        .position(|a| a == "--sigma")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(Live::default().sigma_opt);
    let csigma: f64 = args
        .iter()
        .position(|a| a == "--csigma")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(Live::default().sigma_chroma);
    let jitter: f64 = args
        .iter()
        .position(|a| a == "--jitter")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let sjit: f64 = args
        .iter()
        .position(|a| a == "--sjit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(Live::default().sigma_jit);
    let plain = args.iter().any(|a| a == "--plain");
    let ch = Live {
        noise_scale: noise,
        sigma_opt: sigma,
        sigma_chroma: csigma,
        field: !plain,
        crosstalk: !plain,
        jitter_px: jitter,
        sigma_jit: sjit,
        ..Default::default()
    };

    match what {
        "anchor" => cmd_anchor(&ch, trials),
        "probe" => cmd_probe(&ch, trials, args),
        "sep" => cmd_sep(&ch, trials),
        "floor" => cmd_floor(&ch, trials),
        "aperture" => cmd_aperture(&ch, trials),
        other => eprintln!("неизвестная развёртка {other}; sep|floor|aperture|anchor|probe"),
    }
}

/// Одна точка по требованию: `probe --chroma 1 --d 8 --ppc 8.46 --isi 0`.
fn cmd_probe(ch: &Live, trials: usize, args: &[String]) {
    let num = |k: &str, d: f64| -> f64 {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let chroma = num("--chroma", 1.0) != 0.0;
    let d = num("--d", 8.0) as usize;
    let ppc = num("--ppc", d as f64 * 1.06);
    let isi = num("--isi", 0.0) != 0.0;
    let ap = match args
        .iter()
        .position(|a| a == "--ap")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
    {
        Some("point") => Aperture::Point,
        Some("box05") => Aperture::Box(0.5),
        Some("box10") => Aperture::Box(1.0),
        Some("grid3") => Aperture::Grid3,
        Some("g025") => Aperture::Gauss(0.25),
        Some("g035") => Aperture::Gauss(0.35),
        Some(other) if other.starts_with("q") => {
            Aperture::Quad(other[1..].parse().unwrap_or(0.25))
        }
        _ => Aperture::Quad2x2,
    };
    let r = run_point(chroma, d, ppc / d as f64, isi, ap, ch, trials, 7777);
    println!(
        "chroma={} D={d} ppc={ppc:.2} isi={} ap={} noise={:.2} sigma={:.2} csigma={:.2} jitter={:.2} | SER {:.3e} full {:.3} stripes {:.2} goodput {:.4}",
        chroma as u8, isi as u8, ap.name(), ch.noise_scale, ch.sigma_opt, ch.sigma_chroma, ch.jitter_px,
        r.ser, r.full, r.stripes, r.goodput
    );
    println!("   разделение осей: Re {:.2}σ Im {:.2}σ", r.quality[0], r.quality[1]);
}

/// Калибровка канала по живым опорным точкам A22.
fn cmd_anchor(ch: &Live, trials: usize) {
    println!("# cellfloor anchor — сим против ночной матрицы A22 (v1c, 2 бит/клетку)");
    println!("шум ×{:.2}, σ_опт {:.2} камерных px\n", ch.noise_scale, ch.sigma_opt);
    println!("| клетка | ppc | ISI | живые 8/8 | сим 8/8 | сим SER | сим goodput |");
    println!("|---|---|---|---|---|---|---|");
    for (i, &(d, ppc, live, isi)) in ANCHORS.iter().enumerate() {
        let mag = ppc / d as f64;
        let r = run_point(true, d, mag, isi, Aperture::Quad2x2, ch, trials, 900 + i);
        println!(
            "| {d} | {ppc:.2} | {} | {:.3} | {:.3} | {:.5} | {:.3} |",
            if isi { "вкл" } else { "выкл" },
            live,
            r.full,
            r.ser,
            r.goodput
        );
    }
}

/// РАЗДЕЛЕНИЕ: клетка показа при ФИКСИРОВАННЫХ камерных px/клетку.
fn cmd_sep(ch: &Live, trials: usize) {
    println!("# cellfloor sep — клетка показа D при ФИКСИРОВАННЫХ камерных px/клетку");
    println!(
        "шум ×{:.2}, σ_опт {:.2} px. Если строка плоская по D — клетка показа не ограничивает.\n",
        ch.noise_scale, ch.sigma_opt
    );
    let ds = [3usize, 4, 5, 6, 8, 10, 12, 16];
    for &chroma in &[false, true] {
        for &isi in &[false, true] {
            println!(
                "## {} , ISI {}",
                if chroma { "2 бита (ConstLuma1)" } else { "1 бит (Mono)" },
                if isi { "вкл" } else { "выкл" }
            );
            println!("| ppc \\ D | {} |", ds.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" | "));
            println!("|---{}|", "|---".repeat(ds.len()));
            for &ppc in &[5.5f64, 7.0, 8.5, 10.5] {
                let mut row = Vec::new();
                for (j, &d) in ds.iter().enumerate() {
                    let mag = ppc / d as f64;
                    let idx = 1000 + (chroma as usize) * 400 + (isi as usize) * 200 + (ppc as usize) * 10 + j;
                    let r = run_point(chroma, d, mag, isi, Aperture::Quad2x2, ch, trials, idx);
                    row.push(format!("{:.3}", r.full));
                }
                println!("| {ppc:.1} | {} |", row.join(" | "));
            }
            println!();
        }
    }
}

/// ПОЛ: goodput против камерных px/клетку при естественном увеличении стенда.
fn cmd_floor(ch: &Live, trials: usize) {
    println!("# cellfloor floor — goodput против камерных px/клетку (увеличение стенда A22 ×1.06)");
    println!("шум ×{:.2}, σ_опт {:.2} камерных px. D — клетка показа, ppc = 1.06·D.\n", ch.noise_scale, ch.sigma_opt);
    let mag = 1.06;
    println!("| режим | ISI | D | ppc | σ/ppc | SER | 8/8 | страйпов | goodput | бит/клетку | бит/1000px² |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for &chroma in &[false, true] {
        for &isi in &[false, true] {
            for (j, &d) in [3usize, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16].iter().enumerate() {
                let idx = 2000 + (chroma as usize) * 100 + (isi as usize) * 50 + j;
                let r = run_point(chroma, d, mag, isi, Aperture::Quad2x2, ch, trials, idx);
                println!(
                    "| {} | {} | {d} | {:.1} | {:.3} | {:.5} | {:.3} | {:.2} | {:.3} | {:.3} | {:.2} |",
                    if chroma { "2 бита" } else { "1 бит" },
                    if isi { "вкл" } else { "выкл" },
                    d as f64 * mag,
                    ch.sigma_opt / (d as f64 * mag),
                    r.ser,
                    r.full,
                    r.stripes,
                    r.goodput,
                    r.bits_per_cell,
                    r.bits_per_kpx
                );
            }
        }
    }
}

/// АПЕРТУРА: какая форма отсчёта клетки выигрывает и на каких ppc.
fn cmd_aperture(ch: &Live, trials: usize) {
    println!("# cellfloor aperture — форма апертуры отсчёта клетки");
    println!("шум ×{:.2}, σ_опт {:.2} камерных px, увеличение стенда ×1.06\n", ch.noise_scale, ch.sigma_opt);
    let aps = [
        Aperture::Quad2x2,
        Aperture::Point,
        Aperture::Box(0.5),
        Aperture::Box(1.0),
        Aperture::Grid3,
        Aperture::Gauss(0.25),
        Aperture::Gauss(0.35),
    ];
    let ds = [4usize, 5, 6, 8, 10, 12];
    for &chroma in &[false, true] {
        for &isi in &[false, true] {
            println!(
                "## {}, ISI {} — доля кадров 8/8",
                if chroma { "2 бита (ConstLuma1)" } else { "1 бит (Mono)" },
                if isi { "вкл" } else { "выкл" }
            );
            println!("| апертура \\ D | {} |", ds.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" | "));
            println!("|---{}|", "|---".repeat(ds.len()));
            for (ai, &ap) in aps.iter().enumerate() {
                let mut row = Vec::new();
                for (j, &d) in ds.iter().enumerate() {
                    let idx = 3000 + (chroma as usize) * 400 + (isi as usize) * 200 + ai * 20 + j;
                    let r = run_point(chroma, d, 1.06, isi, ap, ch, trials, idx);
                    row.push(format!("{:.3}", r.full));
                }
                println!("| {} | {} |", ap.name(), row.join(" | "));
            }
            println!();
        }
    }
}
