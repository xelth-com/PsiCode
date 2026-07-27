//! [ИЗМЕРЕНИЕ] Пер-клеточный шум ОДИНОЧНОГО снимка по серии Y-дампов СТАТИЧНОГО
//! символа (экран не менялся, телефон на штативе). Всё, что гуляет от кадра к
//! кадру, — это канал: шум сенсора + ISP + джиттер геометрии.
//!
//! usage: cell_noise <dump-dir> [x0,y0,w,h] [max-frames]
//!   каталог содержит dump0.meta и dump{N}.y (сырая плоскость Y, rowStride из meta);
//!   необязательный кроп — как в examples/live_ser.rs: холодная грубая стадия
//!   детекции на ЗАГРОМОЖДЁННОМ кадре (символ в окне поверх редактора/терминала)
//!   промахивается, кроп вокруг символа решает. Кроп применяется ОДИНАКОВО ко
//!   всем кадрам, поэтому на измеряемые величины он не влияет.
//!
//! Что считается (по шагам задачи):
//!   1. геометрия на КАЖДОМ кадре -> разброс четырёх углов кольца (RMS/пик, px);
//!   2. значение каждой payload-клетки (57×55) в каждом кадре — ровно тем
//!      сэмплированием, что у демодулятора (§5.2: центр + 2×2 субсэмпл ±cell/4);
//!   3. σ клетки по кадрам: сырые коды 0..255 и НОРМИРОВАННЫЕ на размах
//!      «белое−чёрное» × 255, раздельно по чёрным/белым, до и после снятия
//!      глобального сдвига кадра (мерцание экспозиции/подсветки);
//!   4. фиксированный паттерн (PRNU-подобный): разброс ВРЕМЕННЫХ средних клеток
//!      после вычитания локального (радиус 12 клеток) среднего ПО СВОЕМУ КЛАССУ,
//!      в % от размаха; отдельно — с довычетом ISI-тренда по соседям;
//!   5. сколько из (3) объясняется джиттером геометрии: аналитически (градиент ×
//!      измеренное смещение) и ЭМПИРИЧЕСКИ (та же серия, но геометрия заморожена).

use psicode_core::detect::{self, Detection};
use psicode_core::symbol::{self, PAYLOAD_COLS, PAYLOAD_ROWS, RING};
use psicode_rx::tx_default_profile;
use std::fs;

/// Сторона клеточной сетки символа как f64 (габарит гомографии детекции).
const G: f64 = symbol::GRID as f64;
/// Шаг центральной разности при оценке пространственного градиента, камерных px.
const GRAD_STEP: f64 = 0.5;
/// Радиус окна локального среднего (клетки) — как `LOCAL_COARSE_RADIUS` в
/// `symbol::demod_symbol_local`.
const COARSE_RADIUS: i32 = 12;
/// Порог отбраковки кадра-выброса по геометрии, в КЛЕТКАХ: смещение угла меньше
/// четверти клетки — это джиттер (его и меряем), больше — уже другой лок
/// выравнивателя (иной локальный оптимум), такой кадр в статистику не берём.
const OUTLIER_CELLS: f64 = 0.25;
/// Число payload-клеток символа.
const NCELL: usize = PAYLOAD_COLS * PAYLOAD_ROWS;

// ---------------------------------------------------------------------------
// Плоскость Y
// ---------------------------------------------------------------------------

/// Сырая плоскость яркости одного дампа (8 бит, rowStride из meta).
struct Plane {
    y: Vec<u8>,
    w: usize,
    h: usize,
    stride: usize,
}

impl Plane {
    #[inline]
    fn at(&self, x: usize, y: usize) -> f64 {
        self.y[y * self.stride + x] as f64
    }

    /// Билинейная выборка Y в КОДОВЫХ единицах 0..255 с зажимом к краю. Тот же
    /// интерполятор, что у приёмника (session::bilinear_rgb), но по одной Y.
    fn bilinear(&self, x: f64, y: f64) -> f64 {
        let xc = x.clamp(0.0, (self.w - 1) as f64);
        let yc = y.clamp(0.0, (self.h - 1) as f64);
        let x0 = xc.floor() as usize;
        let y0 = yc.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = xc - x0 as f64;
        let fy = yc - y0 as f64;
        let a = self.at(x0, y0) * (1.0 - fx) + self.at(x1, y0) * fx;
        let b = self.at(x0, y1) * (1.0 - fx) + self.at(x1, y1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Плоскость нормированной яркости [0,1] для детекции (как `YuvFrame::y_norm`).
    fn luma_f32(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.w * self.h];
        for j in 0..self.h {
            for i in 0..self.w {
                v[j * self.w + i] = self.y[j * self.stride + i] as f32 / 255.0;
            }
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Геометрия
// ---------------------------------------------------------------------------

/// Применение гомографии (клеточные координаты -> px снимка). Копия внутренней
/// `detect::apply_h` (та не экспортирована).
fn apply_h(h: &[[f64; 3]; 3], u: f64, v: f64) -> (f64, f64) {
    let d = h[2][0] * u + h[2][1] * v + h[2][2];
    let inv = if d.abs() < 1e-12 { 0.0 } else { 1.0 / d };
    (
        (h[0][0] * u + h[0][1] * v + h[0][2]) * inv,
        (h[1][0] * u + h[1][1] * v + h[1][2]) * inv,
    )
}

/// Внешние углы кольца [tl, tr, br, bl] из гомографии (так же, как их достаёт
/// `detect::track_symbol`).
fn corners_of(h: &[[f64; 3]; 3]) -> [(f64, f64); 4] {
    [
        apply_h(h, 0.0, 0.0),
        apply_h(h, G, 0.0),
        apply_h(h, G, G),
        apply_h(h, 0.0, G),
    ]
}

/// Гомография «клеточный квадрат [0,G]² -> четырёхугольник [tl,tr,br,bl]»
/// (формулы Хекберта для единичного квадрата, затем масштаб 1/G по u,v).
/// Нужна, чтобы собрать СРЕДНЮЮ геометрию серии из усреднённых углов.
fn quad_to_h(c: &[(f64, f64); 4]) -> [[f64; 3]; 3] {
    let (x0, y0) = c[0];
    let (x1, y1) = c[1];
    let (x2, y2) = c[2];
    let (x3, y3) = c[3];
    let dx1 = x1 - x2;
    let dx2 = x3 - x2;
    let dx3 = x0 - x1 + x2 - x3;
    let dy1 = y1 - y2;
    let dy2 = y3 - y2;
    let dy3 = y0 - y1 + y2 - y3;
    let (a, b, cc, d, e, f, g, hh);
    let den = dx1 * dy2 - dx2 * dy1;
    if dx3.abs() < 1e-12 && dy3.abs() < 1e-12 || den.abs() < 1e-12 {
        // аффинный вырожденный случай
        g = 0.0;
        hh = 0.0;
        a = x1 - x0;
        b = x3 - x0;
        cc = x0;
        d = y1 - y0;
        e = y3 - y0;
        f = y0;
    } else {
        g = (dx3 * dy2 - dx2 * dy3) / den;
        hh = (dx1 * dy3 - dx3 * dy1) / den;
        a = x1 - x0 + g * x1;
        b = x3 - x0 + hh * x3;
        cc = x0;
        d = y1 - y0 + g * y1;
        e = y3 - y0 + hh * y3;
        f = y0;
    }
    // подстановка u = U/G, v = V/G: делим первые два столбца на G.
    [
        [a / G, b / G, cc],
        [d / G, e / G, f],
        [g / G, hh / G, 1.0],
    ]
}

// ---------------------------------------------------------------------------
// Сэмплирование клеток — ровно как в symbol::sample_cell (§5.2)
// ---------------------------------------------------------------------------

/// Центр клетки (cx, cy) сетки GRID в display-px плоскости Frame.
#[inline]
fn cell_center_uv(quiet: usize, cell: usize, cx: usize, cy: usize) -> (f64, f64) {
    (
        ((quiet + cx) * cell) as f64 + cell as f64 / 2.0,
        ((quiet + cy) * cell) as f64 + cell as f64 / 2.0,
    )
}

/// Значение клетки в кодах Y 0..255 по правилу демодулятора: среднее 2×2
/// субсэмплов в центр ± cell/4 (cell ≥ 8, §5.2 MUST). `(sx, sy)` — добавочный
/// сдвиг ТОЧЕК ВЫБОРКИ в камерных px (0,0 — штатно; ненулевой нужен градиенту).
fn cell_value(
    pl: &Plane,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    quiet: usize,
    cell: usize,
    cx: usize,
    cy: usize,
    sx: f64,
    sy: f64,
) -> f64 {
    let (u, v) = cell_center_uv(quiet, cell, cx, cy);
    let d = cell as f64 / 4.0;
    let mut acc = 0.0;
    for &(ox, oy) in &[(-d, -d), (-d, d), (d, -d), (d, d)] {
        let (x, y) = map(u + ox, v + oy);
        acc += pl.bilinear(x + sx, y + sy);
    }
    acc / 4.0
}

/// Сетка payload-значений (57×55, растровый порядок) одного кадра.
fn payload_grid(
    pl: &Plane,
    map: &dyn Fn(f64, f64) -> (f64, f64),
    quiet: usize,
    cell: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; NCELL];
    for pr in 0..PAYLOAD_ROWS {
        for pc in 0..PAYLOAD_COLS {
            out[pr * PAYLOAD_COLS + pc] =
                cell_value(pl, map, quiet, cell, RING + pc, RING + 1 + pr, 0.0, 0.0);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Статистика
// ---------------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Выборочное СКО (делитель n−1): несмещённая оценка σ по конечной серии кадров.
fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    quantile(&s, 0.5)
}

/// Распределение величины по клеткам с масштабом `scale` (для нормировки).
fn print_dist(label: &str, v: &[f64], scale: f64) {
    let mut s: Vec<f64> = v.iter().map(|x| x * scale).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "    {label:<34} mean {:7.3}  med {:7.3}  p10 {:7.3}  p90 {:7.3}  max {:7.3}   n={}",
        mean(&s),
        quantile(&s, 0.5),
        quantile(&s, 0.10),
        quantile(&s, 0.90),
        quantile(&s, 1.0),
        s.len()
    );
}

/// МНК: β = argmin ‖Xβ − y‖ через нормальные уравнения (Гаусс–Жордан с частичным
/// выбором и лёгким гребнем против вырождения). Возвращает (β, остатки).
fn lstsq(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let p = x[0].len();
    let mut a = vec![vec![0.0f64; p + 1]; p];
    for (row, &yy) in x.iter().zip(y) {
        for r in 0..p {
            for k in 0..p {
                a[r][k] += row[r] * row[k];
            }
            a[r][p] += row[r] * yy;
        }
    }
    let tr: f64 = (0..p).map(|r| a[r][r]).sum();
    for (r, row) in a.iter_mut().enumerate() {
        row[r] += 1e-9 * tr / p as f64; // гребень: страхует вырожденные столбцы
    }
    for col in 0..p {
        let piv = (col..p)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-12 {
            continue;
        }
        for k in col..=p {
            a[col][k] /= d;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let m = a[r][col];
            if m == 0.0 {
                continue;
            }
            for k in col..=p {
                a[r][k] -= m * a[col][k];
            }
        }
    }
    let beta: Vec<f64> = (0..p).map(|r| a[r][p]).collect();
    let res: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(row, &yy)| yy - row.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>())
        .collect();
    (beta, res)
}

/// Квадратичное среднее.
fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: cell_noise <dump-dir> [x0,y0,w,h] [max-frames]");
    let crop: Option<[usize; 4]> = args.get(2).filter(|s| s.contains(',')).map(|s| {
        let c: Vec<usize> = s.split(',').map(|t| t.parse().expect("кроп: числа")).collect();
        [c[0], c[1], c[2], c[3]]
    });
    let max_frames: usize = args
        .iter()
        .skip(2)
        .find(|s| !s.contains(','))
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    // --- meta + загрузка плоскостей Y ---
    let meta = fs::read_to_string(format!("{dir}/dump0.meta")).expect("dump0.meta");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|t| t.parse().expect("meta: числа"))
        .collect();
    let (fw, fh, stride) = (m[0], m[1], m[2]);
    let (cx0, cy0, w, h) = match crop {
        Some([x0, y0, cw, ch]) => (x0, y0, cw.min(fw - x0), ch.min(fh - y0)),
        None => (0, 0, fw, fh),
    };
    let mut planes: Vec<Plane> = Vec::new();
    let mut i = 0usize;
    while i < max_frames {
        let path = format!("{dir}/dump{i}.y");
        let Ok(buf) = fs::read(&path) else { break };
        assert!(
            buf.len() >= fh * stride,
            "{path}: {} байт, ожидалось >= {}",
            buf.len(),
            fh * stride
        );
        // кроп -> плотная плоскость (stride = ширина кропа); без кропа буфер
        // берётся как есть с исходным rowStride.
        let (y, st) = if crop.is_some() {
            let mut v = vec![0u8; w * h];
            for j in 0..h {
                let s = (cy0 + j) * stride + cx0;
                v[j * w..(j + 1) * w].copy_from_slice(&buf[s..s + w]);
            }
            (v, w)
        } else {
            (buf, stride)
        };
        planes.push(Plane {
            y,
            w,
            h,
            stride: st,
        });
        i += 1;
    }
    let nf = planes.len();
    assert!(nf >= 2, "нужно >= 2 кадров, найдено {nf}");
    println!("=== cell_noise: {nf} кадров {fw}x{fh} (rowStride {stride}) из {dir} ===");
    if let Some(c) = crop {
        println!("кроп: {},{} {}x{} -> рабочая плоскость {w}x{h}", c[0], c[1], w, h);
    }
    println!();

    // Сырая межкадровая разность ПО ПИКСЕЛЯМ (до всякой детекции): показывает,
    // одинаково ли ведут себя все переходы кадр->кадр. Резкий выброс на одном
    // переходе = сцена реально дёрнулась (или кадр рваный), а не шум.
    {
        let (x0, y0, x1, y1) = (w / 4, h / 4, 3 * w / 4, 3 * h / 4);
        let d: Vec<String> = (0..nf - 1)
            .map(|f| {
                let mut s = 0.0f64;
                let mut cnt = 0usize;
                for j in (y0..y1).step_by(2) {
                    for i in (x0..x1).step_by(2) {
                        let dd = planes[f + 1].at(i, j) - planes[f].at(i, j);
                        s += dd * dd;
                        cnt += 1;
                    }
                }
                format!("{:.2}", (s / cnt as f64).sqrt())
            })
            .collect();
        println!("RMS разности соседних кадров (центр кадра, коды): {}", d.join(" "));
    }

    let p = tx_default_profile();
    let quiet = p.quiet_zone_cells() as usize;
    let cellpx = p.cell_size_px as usize;
    println!(
        "профиль приёма: luma_bits {}, chroma {:?}, cell_size_px {}, quiet {} клеток",
        p.luma_bits, p.chroma_mode, cellpx, quiet
    );

    // =======================================================================
    // ШАГ 1. Геометрия на каждом кадре.
    // =======================================================================
    // Захват один раз на кадре 0 полным офлайн-детектом (максимум качества
    // геометрии) + самоуточнение итерациями track по тому же кадру — это тот же
    // рецепт, что в examples/live_ser.rs. Дальше КАЖДЫЙ кадр выравнивается
    // НЕЗАВИСИМО от ОДНОГО И ТОГО ЖЕ семени det0 (а не цепочкой кадр-от-кадра):
    // так разброс углов отражает только содержимое кадра, без случайного
    // блуждания оценки. Цепочка (реальный путь RxSession) считается отдельно
    // ниже для сравнения.
    let luma0 = planes[0].luma_f32();
    let mut det0 = detect::detect_symbol(w, h, &luma0)
        .or_else(|_| detect::detect_symbol_acquire(w, h, &luma0))
        .expect("детекция на кадре 0 не удалась");
    for _ in 0..10 {
        match detect::track_symbol(w, h, &luma0, &det0) {
            Ok(d) if d.score > det0.score + 1e-4 => det0 = d,
            _ => break,
        }
    }
    println!(
        "семя (кадр 0): score {:.4}, rot {}",
        det0.score, det0.rotation_quadrants
    );
    // самопроверка quad_to_h: пересборка гомографии из её же углов даёт ту же карту.
    {
        let hh = quad_to_h(&corners_of(&det0.homography));
        let mut worst = 0.0f64;
        for &(u, v) in &[(0.5, 0.5), (30.5, 30.5), (60.5, 0.5), (12.0, 47.0)] {
            let a = apply_h(&det0.homography, u, v);
            let b = apply_h(&hh, u, v);
            worst = worst.max(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt());
        }
        println!("самопроверка quad_to_h: макс. невязка {worst:.2e} px");
    }

    // Независимое выравнивание каждого кадра от ОБЩЕГО семени, ДО СХОДИМОСТИ
    // (те же 10 итераций, что применены к семени). Одинаковый рецепт для всех
    // кадров обязателен: один шаг track давал бы кадру 0 привилегию (он уже
    // сидит в своём оптимуме). ДВА ПРОХОДА: семя кадра 0 может застрять в
    // худшем локальном оптимуме выравнивателя (наблюдалось: score 0.9936 против
    // 0.9954+ у остальных, углы на 11 px в стороне), поэтому вторым проходом
    // пересеиваем ЛУЧШИМ по score локом первого прохода.
    let mut lumas: Vec<Vec<f32>> = Vec::with_capacity(nf);
    for pl in &planes {
        lumas.push(pl.luma_f32());
    }
    let align_all = |seed: &Detection| -> Vec<Option<Detection>> {
        (0..nf)
            .map(|f| {
                let mut d = detect::track_symbol(w, h, &lumas[f], seed).ok();
                for _ in 0..10 {
                    let Some(cur) = d.as_ref() else { break };
                    match detect::track_symbol(w, h, &lumas[f], cur) {
                        Ok(d2) if d2.score > cur.score + 1e-4 => d = Some(d2),
                        _ => break,
                    }
                }
                d
            })
            .collect()
    };
    let pass1 = align_all(&det0);
    let best = pass1
        .iter()
        .flatten()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .map(|d| Detection {
            homography: d.homography,
            rotation_quadrants: d.rotation_quadrants,
            score: d.score,
        })
        .expect("ни один кадр не выровнялся");
    println!("пересев вторым проходом: лучший score прохода 1 = {:.4}", best.score);
    let dets = align_all(&best);
    let det0 = best;
    let failed: Vec<usize> = (0..nf).filter(|&f| dets[f].is_none()).collect();

    // цепочка (как в RxSession: кадр N ведётся от геометрии кадра N−1)
    let mut chain: Vec<Option<Detection>> = Vec::with_capacity(nf);
    {
        let mut prev = Detection {
            homography: det0.homography,
            rotation_quadrants: det0.rotation_quadrants,
            score: det0.score,
        };
        for f in 0..nf {
            match detect::track_symbol(w, h, &lumas[f], &prev) {
                Ok(d) => {
                    prev = Detection {
                        homography: d.homography,
                        rotation_quadrants: d.rotation_quadrants,
                        score: d.score,
                    };
                    chain.push(Some(d));
                }
                Err(_) => chain.push(None),
            }
        }
    }
    drop(lumas);

    // углы по кадрам + отбраковка выбросов по робастному z-скору (MAD)
    let idx_ok: Vec<usize> = (0..nf).filter(|&f| dets[f].is_some()).collect();
    let corners: Vec<[(f64, f64); 4]> = idx_ok
        .iter()
        .map(|&f| corners_of(&dets[f].as_ref().unwrap().homography))
        .collect();
    // медиана по каждому углу
    let mut med_corner = [(0.0f64, 0.0f64); 4];
    for k in 0..4 {
        let xs: Vec<f64> = corners.iter().map(|c| c[k].0).collect();
        let ys: Vec<f64> = corners.iter().map(|c| c[k].1).collect();
        med_corner[k] = (median(&xs), median(&ys));
    }
    let dev: Vec<f64> = corners
        .iter()
        .map(|c| {
            (0..4)
                .map(|k| {
                    ((c[k].0 - med_corner[k].0).powi(2) + (c[k].1 - med_corner[k].1).powi(2)).sqrt()
                })
                .fold(0.0f64, f64::max)
        })
        .collect();
    // предварительная оценка px/клетку по медианным углам (нужна порогу выброса)
    let ppc_est = {
        let d = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        (d(med_corner[1], med_corner[0])
            + d(med_corner[2], med_corner[3])
            + d(med_corner[3], med_corner[0])
            + d(med_corner[2], med_corner[1]))
            / (4.0 * G)
    };
    // порог можно переопределить (CELL_NOISE_OUTLIER_CELLS=99 -> ничего не
    // отбрасывать) — для проверки устойчивости вывода к отбраковке.
    let outlier_cells = std::env::var("CELL_NOISE_OUTLIER_CELLS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(OUTLIER_CELLS);
    let outlier_px = outlier_cells * ppc_est;
    let mut keep: Vec<usize> = Vec::new();
    let mut dropped: Vec<(usize, f64)> = Vec::new();
    for (j, &f) in idx_ok.iter().enumerate() {
        if dev[j] > outlier_px {
            dropped.push((f, dev[j]));
        } else {
            keep.push(f);
        }
    }
    println!("\n--- ШАГ 1: геометрия ---");
    println!("детекция провалилась на кадрах: {failed:?}");
    println!(
        "отклонение углов от медианы по кадрам, px: {}",
        idx_ok
            .iter()
            .zip(&dev)
            .map(|(f, d)| format!("{f}:{d:.3}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "порог выброса {outlier_cells} клетки = {outlier_px:.2} px -> отброшено {dropped:?}"
    );
    println!("оставлено кадров: {} из {nf}", keep.len());
    let n = keep.len();
    assert!(n >= 2, "после отбраковки осталось {n} кадров");

    // разброс углов относительно СРЕДНЕГО положения (по оставленным кадрам)
    let kc: Vec<[(f64, f64); 4]> = keep
        .iter()
        .map(|&f| corners_of(&dets[f].as_ref().unwrap().homography))
        .collect();
    let mut mean_corner = [(0.0f64, 0.0f64); 4];
    for k in 0..4 {
        mean_corner[k] = (
            mean(&kc.iter().map(|c| c[k].0).collect::<Vec<_>>()),
            mean(&kc.iter().map(|c| c[k].1).collect::<Vec<_>>()),
        );
    }
    let names = ["tl", "tr", "br", "bl"];
    let mut all_r: Vec<f64> = Vec::new();
    for k in 0..4 {
        let r: Vec<f64> = kc
            .iter()
            .map(|c| {
                ((c[k].0 - mean_corner[k].0).powi(2) + (c[k].1 - mean_corner[k].1).powi(2)).sqrt()
            })
            .collect();
        let dx: Vec<f64> = kc.iter().map(|c| c[k].0 - mean_corner[k].0).collect();
        let dy: Vec<f64> = kc.iter().map(|c| c[k].1 - mean_corner[k].1).collect();
        println!(
            "  угол {}: RMS {:.3} px (x {:.3} / y {:.3}), пик {:.3} px, среднее ({:.2}, {:.2})",
            names[k],
            rms(&r),
            rms(&dx),
            rms(&dy),
            r.iter().cloned().fold(0.0, f64::max),
            mean_corner[k].0,
            mean_corner[k].1
        );
        all_r.extend(r);
    }
    println!(
        "  ИТОГО по 4 углам: RMS {:.3} px, пик {:.3} px",
        rms(&all_r),
        all_r.iter().cloned().fold(0.0, f64::max)
    );
    // то же для ЦЕПНОГО трекинга (операционный путь RxSession)
    {
        let cc: Vec<[(f64, f64); 4]> = keep
            .iter()
            .filter(|&&f| chain[f].is_some())
            .map(|&f| corners_of(&chain[f].as_ref().unwrap().homography))
            .collect();
        if cc.len() >= 2 {
            let mut r_all: Vec<f64> = Vec::new();
            for k in 0..4 {
                let mx = mean(&cc.iter().map(|c| c[k].0).collect::<Vec<_>>());
                let my = mean(&cc.iter().map(|c| c[k].1).collect::<Vec<_>>());
                r_all.extend(
                    cc.iter()
                        .map(|c| ((c[k].0 - mx).powi(2) + (c[k].1 - my).powi(2)).sqrt()),
                );
            }
            println!(
                "  цепной трекинг (путь RxSession): RMS {:.3} px, пик {:.3} px",
                rms(&r_all),
                r_all.iter().cloned().fold(0.0, f64::max)
            );
        }
    }
    // px на клетку в полном разрешении
    let ppc = {
        let map0 = detect::frame_map(&p, dets[keep[0]].as_ref().unwrap());
        let (u0, v0) = cell_center_uv(quiet, cellpx, RING + 28, RING + 28);
        let a = map0(u0, v0);
        let b = map0(u0 + cellpx as f64, v0);
        ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
    };
    let scores: Vec<f64> = keep
        .iter()
        .map(|&f| dets[f].as_ref().unwrap().score)
        .collect();
    println!(
        "  px/клетку {ppc:.2}; score детекции: min {:.4} med {:.4} max {:.4}",
        scores.iter().cloned().fold(f64::MAX, f64::min),
        median(&scores),
        scores.iter().cloned().fold(0.0, f64::max)
    );

    // СРЕДНЯЯ геометрия серии (замороженная) — для шага 5.
    let mean_h = quad_to_h(&mean_corner);
    let det_mean = Detection {
        homography: mean_h,
        rotation_quadrants: det0.rotation_quadrants,
        score: det0.score,
    };

    // =======================================================================
    // ШАГ 2. Значения payload-клеток во всех кадрах.
    // =======================================================================
    // A — геометрия СВОЯ у каждого кадра (операционный режим демодулятора);
    // B — геометрия ЗАМОРОЖЕНА на средней (снимает джиттер оценки геометрии);
    // C — цепной трекинг (буквальный путь RxSession).
    let map_mean = detect::frame_map(&p, &det_mean);
    let mut va: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut vb: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut vc: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut bits: Vec<Vec<u8>> = Vec::with_capacity(n);
    for &f in &keep {
        let d = dets[f].as_ref().unwrap();
        let map_f = detect::frame_map(&p, d);
        va.push(payload_grid(&planes[f], &map_f, quiet, cellpx));
        vb.push(payload_grid(&planes[f], &map_mean, quiet, cellpx));
        if let Some(dc) = chain[f].as_ref() {
            let map_c = detect::frame_map(&p, dc);
            vc.push(payload_grid(&planes[f], &map_c, quiet, cellpx));
        }
        // демодуляция ровно продакшн-путём (локальный двухмасштабный порог);
        // sample отдаёт Y во все три канала — Mono читает канал G.
        let pl = &planes[f];
        let samp = |x: f64, y: f64| -> [f32; 3] {
            let g = (pl.bilinear(x, y) / 255.0) as f32;
            [g, g, g]
        };
        bits.push(symbol::demod_symbol_local(&p, &map_f, &samp));
    }

    // ВАЛИДНОСТЬ ГЕОМЕТРИИ: символ на экране — `psicode-tx single`, то есть
    // ДЕТЕРМИНИРОВАННАЯ нагрузка splitmix64 (тот же сид, что в tx frames.rs и в
    // live_ser --single). Сравнение с ней доказывает, что сетка выборки села на
    // клетки точно (сдвиг на клетку или иная ротация дали бы SER ~0.5).
    {
        let bpc = symbol::bits_per_cell(&p);
        let mask: u16 = if bpc >= 16 { u16::MAX } else { (1u16 << bpc) - 1 };
        let mut state = 0x0D15_EA5E_5EED_1234u64;
        let truth: Vec<u8> = (0..NCELL)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (((z ^ (z >> 31)) >> 24) as u16 & mask) as u8
            })
            .collect();
        let errs: Vec<usize> = bits
            .iter()
            .map(|b| b.iter().zip(&truth).filter(|(a, c)| a != c).count())
            .collect();
        println!(
            "\n--- валидность: SER против эталонной нагрузки tx single ---\n  \
             ошибок на кадр: min {} max {} (из {NCELL}) -> SER {:.5}..{:.5}",
            errs.iter().min().unwrap(),
            errs.iter().max().unwrap(),
            *errs.iter().min().unwrap() as f64 / NCELL as f64,
            *errs.iter().max().unwrap() as f64 / NCELL as f64
        );
    }

    // проверка «экран статичен»: биты обязаны совпадать во всех кадрах
    println!("\n--- проверка: демодулированный битовый узор по кадрам ---");
    let mut total_dis = 0usize;
    for (j, b) in bits.iter().enumerate() {
        let d = b.iter().zip(&bits[0]).filter(|(a, c)| a != c).count();
        total_dis += d;
        if d != 0 {
            println!("  кадр {} (dump{}): расхождений с кадром 0: {d}", j, keep[j]);
        }
    }
    println!(
        "  суммарно расхождений с кадром 0: {total_dis} (из {} сравнений)",
        (n - 1) * NCELL
    );
    // классы клеток — по большинству голосов (при total_dis = 0 это просто кадр 0)
    let mut cls = vec![0u8; NCELL];
    for c in 0..NCELL {
        let ones = bits.iter().filter(|b| b[c] == 1).count();
        cls[c] = (2 * ones > n) as u8;
    }
    let nw = cls.iter().filter(|&&c| c == 1).count();
    println!("  белых клеток {nw}, чёрных {}", NCELL - nw);

    // =======================================================================
    // ШАГ 3. σ клетки по кадрам.
    // =======================================================================
    // temporal σ для набора значений vals[кадр][клетка]
    let per_cell_sd = |vals: &Vec<Vec<f64>>| -> Vec<f64> {
        (0..NCELL)
            .map(|c| sd(&vals.iter().map(|fr| fr[c]).collect::<Vec<_>>()))
            .collect()
    };
    // снятие ГЛОБАЛЬНОГО сдвига кадра (среднее по всем payload-клеткам кадра)
    let demeaned = |vals: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        vals.iter()
            .map(|fr| {
                let m = mean(fr);
                fr.iter().map(|x| x - m).collect()
            })
            .collect()
    };
    // снятие ПОКЛАССОВОГО среднего кадра (сдвиг + «дыхание» контраста)
    let declassed = |vals: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        vals.iter()
            .map(|fr| {
                let mk: Vec<f64> = (0..NCELL).filter(|&c| cls[c] == 0).map(|c| fr[c]).collect();
                let mw: Vec<f64> = (0..NCELL).filter(|&c| cls[c] == 1).map(|c| fr[c]).collect();
                let (mk, mw) = (mean(&mk), mean(&mw));
                (0..NCELL)
                    .map(|c| fr[c] - if cls[c] == 1 { mw } else { mk })
                    .collect()
            })
            .collect()
    };

    let sd_a = per_cell_sd(&va);
    let sd_a_dm = per_cell_sd(&demeaned(&va));
    let sd_a_dc = per_cell_sd(&declassed(&va));
    let sd_b = per_cell_sd(&vb);
    let sd_b_dm = per_cell_sd(&demeaned(&vb));

    // уровни и размах
    let cellmean: Vec<f64> = (0..NCELL)
        .map(|c| mean(&va.iter().map(|fr| fr[c]).collect::<Vec<_>>()))
        .collect();
    let black_mean = mean(
        &(0..NCELL)
            .filter(|&c| cls[c] == 0)
            .map(|c| cellmean[c])
            .collect::<Vec<_>>(),
    );
    let white_mean = mean(
        &(0..NCELL)
            .filter(|&c| cls[c] == 1)
            .map(|c| cellmean[c])
            .collect::<Vec<_>>(),
    );
    let swing = white_mean - black_mean;
    let norm = 255.0 / swing; // коды 0..255 -> «полная шкала 255»

    println!("\n--- ШАГ 3: уровни и σ ---");
    println!(
        "  средняя ЧЁРНАЯ клетка {black_mean:.2}, средняя БЕЛАЯ {white_mean:.2}, \
         размах {swing:.2} кода -> нормировка ×{norm:.4}"
    );
    // насыщение внутри символа (кадр 0)
    {
        let c0 = corners_of(&dets[keep[0]].as_ref().unwrap().homography);
        let x0 = c0.iter().map(|c| c.0).fold(f64::MAX, f64::min).max(0.0) as usize;
        let x1 = (c0.iter().map(|c| c.0).fold(0.0, f64::max) as usize).min(w - 1);
        let y0 = c0.iter().map(|c| c.1).fold(f64::MAX, f64::min).max(0.0) as usize;
        let y1 = (c0.iter().map(|c| c.1).fold(0.0, f64::max) as usize).min(h - 1);
        let (mut hi, mut lo, mut tot) = (0usize, 0usize, 0usize);
        for j in y0..=y1 {
            for i in x0..=x1 {
                let v = planes[keep[0]].at(i, j);
                tot += 1;
                if v >= 255.0 {
                    hi += 1;
                }
                if v <= 0.0 {
                    lo += 1;
                }
            }
        }
        println!(
            "  насыщение в bbox символа (кадр {}): Y=255 {:.4}% ({hi}), Y=0 {:.4}% ({lo}), всего {tot} px",
            keep[0],
            100.0 * hi as f64 / tot as f64,
            100.0 * lo as f64 / tot as f64
        );
    }

    let sel = |v: &Vec<f64>, want: u8| -> Vec<f64> {
        (0..NCELL)
            .filter(|&c| cls[c] == want)
            .map(|c| v[c])
            .collect()
    };
    for (tag, s, sdm, sdc) in [
        ("A (своя геометрия кадра)", &sd_a, &sd_a_dm, Some(&sd_a_dc)),
        ("B (геометрия заморожена)", &sd_b, &sd_b_dm, None),
    ] {
        println!("\n  === набор {tag} ===");
        println!("  -- СЫРЫЕ коды 0..255 --");
        print_dist("все клетки", s, 1.0);
        print_dist("чёрные", &sel(s, 0), 1.0);
        print_dist("белые", &sel(s, 1), 1.0);
        println!("  -- НОРМИРОВАНО на размах (кодов/255 полной шкалы) --");
        print_dist("все клетки", s, norm);
        print_dist("чёрные", &sel(s, 0), norm);
        print_dist("белые", &sel(s, 1), norm);
        println!("  -- НОРМИРОВАНО, после снятия глоб. среднего кадра --");
        print_dist("все клетки", sdm, norm);
        print_dist("чёрные", &sel(sdm, 0), norm);
        print_dist("белые", &sel(sdm, 1), norm);
        if let Some(sdc) = sdc {
            println!("  -- НОРМИРОВАНО, после снятия ПОКЛАССОВЫХ средних кадра --");
            print_dist("все клетки", sdc, norm);
            print_dist("чёрные", &sel(sdc, 0), norm);
            print_dist("белые", &sel(sdc, 1), norm);
        }
    }
    if vc.len() >= 2 {
        let sd_c = per_cell_sd(&vc);
        println!("\n  === набор C (цепной трекинг, путь RxSession) ===");
        print_dist("все клетки, нормировано", &sd_c, norm);
    }
    // =======================================================================
    // ДИАГНОСТИКА НЕЗАВИСИМОСТИ КАДРОВ (главный способ, которым это измерение
    // может обмануть): ISP телефона почти наверняка делает ВРЕМЕННОЕ шумо-
    // подавление. Тогда соседние кадры статичной сцены — НЕ независимые снимки,
    // разброс по серии занижен, и σ одиночного снимка на самом деле больше.
    // Признак — положительная автокорреляция остатка на лаге 1: для чистого
    // белого шума ρ1 ≈ 0 и σ(разности соседних кадров) = √2·σ.
    {
        println!("\n--- диагностика: независимы ли кадры (временное шумоподавление ISP?) ---");
        for (tag, vv) in [
            ("сырые", &va),
            ("минус глоб. среднее кадра", &demeaned(&va)),
            ("минус поклассовые средние кадра", &declassed(&va)),
        ] {
            let mut rho = vec![0.0f64; NCELL];
            let mut ratio = vec![0.0f64; NCELL];
            for c in 0..NCELL {
                let x: Vec<f64> = (0..n).map(|f| vv[f][c]).collect();
                let m = mean(&x);
                let num: f64 = (0..n - 1).map(|f| (x[f] - m) * (x[f + 1] - m)).sum();
                let den: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
                rho[c] = if den > 1e-12 { num / den } else { 0.0 };
                let diffs: Vec<f64> = (0..n - 1).map(|f| x[f + 1] - x[f]).collect();
                let s = sd(&x);
                ratio[c] = if s > 1e-9 {
                    rms(&diffs) / (s * std::f64::consts::SQRT_2)
                } else {
                    1.0
                };
            }
            println!(
                "  [{tag}] ρ1: все {:+.3} чёрные {:+.3} белые {:+.3} (белый шум ~{:+.3}) | \
                 σ(Δсоседних)/(√2σ): все {:.3} чёрные {:.3} белые {:.3} (белый шум 1.000)",
                mean(&rho),
                mean(&sel(&rho, 0)),
                mean(&sel(&rho, 1)),
                -1.0 / (n - 1) as f64,
                mean(&ratio),
                mean(&sel(&ratio, 0)),
                mean(&sel(&ratio, 1))
            );
        }
    }

    // Пер-ПИКСЕЛЬНАЯ σ по времени внутри символа (без усреднения по клетке):
    // сэмплирование клетки усредняет 4 билинейных отсчёта (до 16 пикселей), что
    // само по себе давит шум. Это число показывает, сколько выигрыша даёт именно
    // усреднение, и что видит любой алгоритм, работающий по пикселям.
    {
        let c0 = corners_of(&dets[keep[0]].as_ref().unwrap().homography);
        let x0 = (c0.iter().map(|c| c.0).fold(f64::MAX, f64::min).max(0.0) as usize) + 20;
        let x1 = ((c0.iter().map(|c| c.0).fold(0.0, f64::max) as usize).min(w - 1)).saturating_sub(20);
        let y0 = (c0.iter().map(|c| c.1).fold(f64::MAX, f64::min).max(0.0) as usize) + 20;
        let y1 = ((c0.iter().map(|c| c.1).fold(0.0, f64::max) as usize).min(h - 1)).saturating_sub(20);
        let (mut sdark, mut nd, mut sbright, mut nb) = (0.0f64, 0usize, 0.0f64, 0usize);
        let mut step = 0usize;
        for j in (y0..=y1).step_by(3) {
            for i in (x0..=x1).step_by(3) {
                let series: Vec<f64> = keep.iter().map(|&f| planes[f].at(i, j)).collect();
                let m = mean(&series);
                let s = sd(&series);
                step += 1;
                if m < black_mean + 0.25 * swing {
                    sdark += s * s;
                    nd += 1;
                } else if m > white_mean - 0.25 * swing {
                    sbright += s * s;
                    nb += 1;
                }
            }
        }
        let _ = step;
        println!(
            "  пер-ПИКСЕЛЬНАЯ σ (RMS) внутри символа: тёмные {:.3} кода = {:.3} норм. (n={nd}), \
             светлые {:.3} кода = {:.3} норм. (n={nb})",
            (sdark / nd.max(1) as f64).sqrt(),
            (sdark / nd.max(1) as f64).sqrt() * norm,
            (sbright / nb.max(1) as f64).sqrt(),
            (sbright / nb.max(1) as f64).sqrt() * norm
        );
    }

    // глобальный сдвиг кадра сам по себе (мерцание экспозиции)
    {
        let gm: Vec<f64> = va.iter().map(|fr| mean(fr)).collect();
        let bk: Vec<f64> = va
            .iter()
            .map(|fr| mean(&(0..NCELL).filter(|&c| cls[c] == 0).map(|c| fr[c]).collect::<Vec<_>>()))
            .collect();
        let wt: Vec<f64> = va
            .iter()
            .map(|fr| mean(&(0..NCELL).filter(|&c| cls[c] == 1).map(|c| fr[c]).collect::<Vec<_>>()))
            .collect();
        let sw: Vec<f64> = (0..n).map(|f| wt[f] - bk[f]).collect();
        println!(
            "\n  мерцание кадра: σ(глоб. среднего) {:.3} кода = {:.3} норм.; \
             σ(средн. чёрного) {:.3}; σ(средн. белого) {:.3}; σ(размаха) {:.3} кода ({:.2}%)",
            sd(&gm),
            sd(&gm) * norm,
            sd(&bk),
            sd(&wt),
            sd(&sw),
            100.0 * sd(&sw) / swing
        );
    }

    // =======================================================================
    // ШАГ 4. Фиксированный паттерн (PRNU-подобный).
    // =======================================================================
    // Локальное среднее радиуса COARSE_RADIUS по клеткам ТОГО ЖЕ класса (смешивать
    // классы нельзя: разница чёрное/белое на порядок больше искомого остатка).
    let mut resid = vec![0.0f64; NCELL];
    for pr in 0..PAYLOAD_ROWS as i32 {
        for pc in 0..PAYLOAD_COLS as i32 {
            let c = (pr as usize) * PAYLOAD_COLS + pc as usize;
            let mut s = 0.0;
            let mut cnt = 0usize;
            for dr in -COARSE_RADIUS..=COARSE_RADIUS {
                for dc in -COARSE_RADIUS..=COARSE_RADIUS {
                    let (r, cc2) = (pr + dr, pc + dc);
                    if r < 0 || r >= PAYLOAD_ROWS as i32 || cc2 < 0 || cc2 >= PAYLOAD_COLS as i32 {
                        continue;
                    }
                    let k = (r as usize) * PAYLOAD_COLS + cc2 as usize;
                    if cls[k] == cls[c] {
                        s += cellmean[k];
                        cnt += 1;
                    }
                }
            }
            resid[c] = cellmean[c] - s / cnt.max(1) as f64;
        }
    }
    // доля дисперсии остатка, наведённая ВРЕМЕННЫМ шумом (среднее по 24 кадрам):
    // var(среднего) = σ²/n.
    let noise_var_in_mean = |want: u8| -> f64 {
        mean(&sel(&sd_a, want).iter().map(|s| s * s / n as f64).collect::<Vec<_>>())
    };
    println!("\n--- ШАГ 4: фиксированный паттерн ---");
    for want in [0u8, 1] {
        let r = sel(&resid, want);
        let raw = sd(&r);
        let corr = (raw * raw - noise_var_in_mean(want)).max(0.0).sqrt();
        println!(
            "  {}: разброс остатка {:.3} кода = {:.3}% размаха; за вычетом временного шума {:.3} кода = {:.3}% размаха",
            if want == 1 { "белые" } else { "чёрные" },
            raw,
            100.0 * raw / swing,
            corr,
            100.0 * corr / swing
        );
    }
    // ISI. Остаток выше — НЕ чистый PRNU: при ~13 px/клетку блюр камеры/оптики
    // затягивает в клетку соседей, и одиночная чёрная клетка среди белых читается
    // заметно светлее чёрной в чёрном блоке. Это ДЕТЕРМИНИРОВАННЫЙ, зависящий от
    // узора эффект (его можно выравнивать), а не фиксированный паттерн сенсора.
    // Снимаем его линейной моделью по ВСЕЙ окрестности 5×5 (24 бита соседей +
    // константа); что останется — верхняя оценка настоящего PRNU-подобного
    // разброса. За краем payload сосед неизвестен -> нейтральные 0.5.
    {
        let mut offs: Vec<(i32, i32)> = Vec::new();
        for dr in -2..=2i32 {
            for dc in -2..=2i32 {
                if (dr, dc) != (0, 0) {
                    offs.push((dr, dc));
                }
            }
        }
        let feats = |c: usize| -> Vec<f64> {
            let pr = (c / PAYLOAD_COLS) as i32;
            let pc = (c % PAYLOAD_COLS) as i32;
            let mut v = Vec::with_capacity(offs.len() + 1);
            v.push(1.0);
            for &(dr, dc) in &offs {
                let (r, cc2) = (pr + dr, pc + dc);
                v.push(
                    if r < 0 || r >= PAYLOAD_ROWS as i32 || cc2 < 0 || cc2 >= PAYLOAD_COLS as i32 {
                        0.5
                    } else {
                        cls[(r as usize) * PAYLOAD_COLS + cc2 as usize] as f64
                    },
                );
            }
            v
        };
        // СУБПИКСЕЛЬНАЯ ФАЗА. 12.98 камерных px на клетку — не целое число,
        // поэтому центр каждой клетки садится на СВОЮ фазу относительно пиксельной
        // сетки камеры (и, через масштаб, сетки экрана). Это даёт систематическую,
        // повторяющуюся от кадра к кадру добавку, неотличимую от «фиксированного
        // паттерна», но по природе — сэмплирование/муар, а не PRNU сенсора.
        // Проверяем гармониками фазы (1-я и 2-я) по обеим осям.
        let phase = |c: usize| -> Vec<f64> {
            let (u, v) = cell_center_uv(
                quiet,
                cellpx,
                RING + c % PAYLOAD_COLS,
                RING + 1 + c / PAYLOAD_COLS,
            );
            let (x, y) = map_mean(u, v);
            let (fx, fy) = (x - x.floor(), y - y.floor());
            let t = std::f64::consts::TAU;
            vec![
                (t * fx).cos(),
                (t * fx).sin(),
                (t * fy).cos(),
                (t * fy).sin(),
                (2.0 * t * fx).cos(),
                (2.0 * t * fx).sin(),
                (2.0 * t * fy).cos(),
                (2.0 * t * fy).sin(),
            ]
        };
        for want in [0u8, 1] {
            let idx: Vec<usize> = (0..NCELL).filter(|&c| cls[c] == want).collect();
            let yv: Vec<f64> = idx.iter().map(|&c| resid[c]).collect();
            let sst: f64 = yv.iter().map(|v| v * v).sum();
            let name = if want == 1 { "белые" } else { "чёрные" };
            for (tag, with_phase) in [("ISI 5×5", false), ("ISI 5×5 + субпиксельная фаза", true)] {
                let x: Vec<Vec<f64>> = idx
                    .iter()
                    .map(|&c| {
                        let mut v = feats(c);
                        if with_phase {
                            v.extend(phase(c));
                        }
                        v
                    })
                    .collect();
                let pnum = x[0].len();
                let (beta, res) = lstsq(&x, &yv);
                let sse: f64 = res.iter().map(|r| r * r).sum();
                let raw = (sse / (idx.len() - pnum) as f64).sqrt();
                let corr = (raw * raw - noise_var_in_mean(want)).max(0.0).sqrt();
                let r2 = 1.0 - sse / sst;
                let (bi, bv) = beta
                    .iter()
                    .enumerate()
                    .take(offs.len() + 1)
                    .skip(1)
                    .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                    .unwrap();
                println!(
                    "  {name}: после довычета {tag} (R² {:.3}, сильнейший сосед {:?} {:+.2} кода): \
                     {:.3} кода = {:.3}% размаха; за вычетом временного шума {:.3}% размаха",
                    r2,
                    offs[bi - 1],
                    bv,
                    raw,
                    100.0 * raw / swing,
                    100.0 * corr / swing
                );
            }
        }
    }

    // =======================================================================
    // ШАГ 5. Сколько из шага 3 — джиттер геометрии.
    // =======================================================================
    // (а) аналитически: предсказание Δзначения = grad · смещение точки выборки.
    let mut gx = vec![0.0f64; NCELL];
    let mut gy = vec![0.0f64; NCELL];
    for &f in &keep {
        for pr in 0..PAYLOAD_ROWS {
            for pc in 0..PAYLOAD_COLS {
                let (cx, cy) = (RING + pc, RING + 1 + pr);
                let c = pr * PAYLOAD_COLS + pc;
                let px = cell_value(&planes[f], &map_mean, quiet, cellpx, cx, cy, GRAD_STEP, 0.0);
                let mx = cell_value(&planes[f], &map_mean, quiet, cellpx, cx, cy, -GRAD_STEP, 0.0);
                let py = cell_value(&planes[f], &map_mean, quiet, cellpx, cx, cy, 0.0, GRAD_STEP);
                let my = cell_value(&planes[f], &map_mean, quiet, cellpx, cx, cy, 0.0, -GRAD_STEP);
                gx[c] += (px - mx) / (2.0 * GRAD_STEP);
                gy[c] += (py - my) / (2.0 * GRAD_STEP);
            }
        }
    }
    for c in 0..NCELL {
        gx[c] /= n as f64;
        gy[c] /= n as f64;
    }
    // смещение центра клетки в каждом кадре относительно средней геометрии
    let mut pred_sd = vec![0.0f64; NCELL];
    let mut disp_rms = vec![0.0f64; NCELL];
    {
        let maps: Vec<_> = keep
            .iter()
            .map(|&f| detect::frame_map(&p, dets[f].as_ref().unwrap()))
            .collect();
        for pr in 0..PAYLOAD_ROWS {
            for pc in 0..PAYLOAD_COLS {
                let c = pr * PAYLOAD_COLS + pc;
                let (u, v) = cell_center_uv(quiet, cellpx, RING + pc, RING + 1 + pr);
                let (bx, by) = map_mean(u, v);
                let mut preds = Vec::with_capacity(n);
                let mut ds = Vec::with_capacity(n);
                for mp in &maps {
                    let (x, y) = mp(u, v);
                    let (dx, dy) = (x - bx, y - by);
                    preds.push(gx[c] * dx + gy[c] * dy);
                    ds.push((dx * dx + dy * dy).sqrt());
                }
                pred_sd[c] = sd(&preds);
                disp_rms[c] = rms(&ds);
            }
        }
    }
    let var = |v: &Vec<f64>| mean(&v.iter().map(|x| x * x).collect::<Vec<_>>());
    println!("\n--- ШАГ 5: вклад джиттера геометрии ---");
    print_dist("|grad| клетки, кодов/px", &(0..NCELL).map(|c| (gx[c]*gx[c]+gy[c]*gy[c]).sqrt()).collect::<Vec<_>>(), 1.0);
    print_dist("смещение центра клетки, px", &disp_rms, 1.0);
    print_dist("предсказанная σ_геом (норм.)", &pred_sd, norm);
    print_dist("наблюдённая σ_A (норм.)", &sd_a, norm);
    println!(
        "  доля дисперсии: <σ_геом²>/<σ_A²> = {:.3}  (медиана отношения σ_геом/σ_A = {:.3})",
        var(&pred_sd) / var(&sd_a),
        median(&(0..NCELL).map(|c| pred_sd[c] / sd_a[c].max(1e-9)).collect::<Vec<_>>())
    );
    println!(
        "  ЭМПИРИЧЕСКИ: <σ_A²> = {:.4}, <σ_B²> (геометрия заморожена) = {:.4} \
         -> σ_A {:.3} vs σ_B {:.3} нормированных кодов",
        var(&sd_a),
        var(&sd_b),
        var(&sd_a).sqrt() * norm,
        var(&sd_b).sqrt() * norm
    );
    println!(
        "  σ_A после снятия глоб. среднего: {:.3}; σ_B после снятия глоб. среднего: {:.3}",
        var(&sd_a_dm).sqrt() * norm,
        var(&sd_b_dm).sqrt() * norm
    );

    // =======================================================================
    // ВЕРДИКТ
    // =======================================================================
    let headline = median(&sd_a) * norm;
    let verdict = if headline < 5.0 {
        "НИЖЕ 5"
    } else if headline <= 8.0 {
        "МЕЖДУ 5 И 8"
    } else {
        "ВЫШЕ 8"
    };
    println!(
        "\n=== ВЕРДИКТ: медианная нормированная σ одиночного снимка = {headline:.2} кодов/255 \
         полной шкалы -> {verdict} ===",
    );
    println!(
        "    (среднее по клеткам {:.2}; RMS {:.2}; после снятия глобального сдвига кадра {:.2})",
        mean(&sd_a) * norm,
        var(&sd_a).sqrt() * norm,
        median(&sd_a_dm) * norm
    );
}
