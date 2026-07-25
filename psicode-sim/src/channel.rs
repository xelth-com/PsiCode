//! Параметры канала «монитор -> камера» и геометрия проекции.
//!
//! `ChannelParams` описывает всю физику L0 (§3, §5.1, §9.1 SPEC): гаммы,
//! масштаб (px камеры на клетку монитора), гомографию, дефокус-блюр,
//! перекрёстные помехи каналов, усиление/смещение и шум. Значения по
//! умолчанию берутся ИЗ телеметрии эталонного профиля (§7.4), т.е. «правда
//! канала = то, что намерила калибровка».
//!
//! `Geometry` — производная от (масштаб, гомография, размер символа): прямое
//! отображение плоскости символа в координаты снимка (для genie-map демода) и
//! обратное (для warp).

use psicode_core::{symbol, CalibProfile};

/// Единичная 3×3 матрица (гомография по умолчанию).
pub const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Сторона отрендеренного символа в display-пикселях (тихая зона включена):
/// `(GRID + 2·quiet_cells) · cell_size_px`.
pub fn symbol_size_px(p: &CalibProfile) -> usize {
    let quiet = p.quiet_zone_cells() as usize;
    (symbol::GRID + 2 * quiet) * p.cell_size_px as usize
}

/// Полный набор параметров канала.
#[derive(Clone, Debug)]
pub struct ChannelParams {
    /// Гаммы дисплея по каналам [R, G, B]: emitted = (drive/255)^γ.
    pub gammas: [f64; 3],
    /// Пикселей КАМЕРЫ на одну КЛЕТКУ монитора. Геометрический масштаб warp
    /// равен `px_per_cell / cell_size_px` (cell_size_px — display-px на клетку).
    pub px_per_cell: f64,
    /// Display-px на клетку (из профиля) — вторая половина масштаба.
    pub cell_size_px: f64,
    /// Гомография 3×3 в display-пиксельных координатах (по умолчанию единичная).
    pub homography: [[f64; 3]; 3],
    /// Сигма гауссова дефокус-блюра ПО КАНАЛАМ [R, G, B], в пикселях КАМЕРЫ
    /// (0 — без блюра). Изотропный дефокус — три равные σ ([`ChannelParams::set_blur`]);
    /// продольная хроматическая аберрация — разные σ ([`ChannelParams::set_aberration`]).
    pub blur_sigma_rgb: [f64; 3],
    /// Доля перекрёстной помехи R<->G (0..1).
    pub crosstalk_rg: f64,
    /// Доля перекрёстной помехи G<->B (0..1).
    pub crosstalk_gb: f64,
    /// Усиление по каналам [R, G, B]: out = gain·in + offset.
    pub gain: [f64; 3],
    /// Смещение (чёрный уровень) по каналам, доля полной шкалы.
    pub offset: [f64; 3],
    /// Сигма аддитивного гауссова шума, ДОЛЯ полной шкалы (полная шкала = 1.0).
    pub noise_sigma: f64,
}

impl ChannelParams {
    /// Эталонный «правдивый» канал из телеметрии профиля (§7.4): гаммы, кросстолк
    /// и шум — как их намерил приёмник. Геометрия нейтральная: единичная
    /// гомография, без блюра, px_per_cell = 8 (базовая точка развёрток).
    ///
    /// Единицы шума: `noise_sigma()` профиля даёт σ в градациях серого (0..255);
    /// переводим в долю полной шкалы делением на 255 и трактуем как σ в линейном
    /// домене (модельное упрощение — калибровка меряет шум в drive-градациях).
    pub fn from_profile(p: &CalibProfile) -> Self {
        ChannelParams {
            gammas: [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64],
            px_per_cell: 8.0,
            cell_size_px: p.cell_size_px as f64,
            homography: IDENTITY,
            blur_sigma_rgb: [0.0; 3],
            crosstalk_rg: p.crosstalk_rg_pct() as f64 / 100.0,
            crosstalk_gb: p.crosstalk_gb_pct() as f64 / 100.0,
            gain: [1.0; 3],
            offset: [0.0; 3],
            noise_sigma: p.noise_sigma() as f64 / 255.0,
        }
    }

    /// Идеальный канал: согласованные гаммы, никаких искажений (нет блюра,
    /// кросстолка, шума; единичная гомография, gain=1, offset=0, px_per_cell=8).
    /// На таком канале SER обязан быть 0 (проверка сквозного тракта).
    /// (Задействован сквозными тестами e2e; в бинаре развёрток не используется.)
    #[allow(dead_code)]
    pub fn clean(p: &CalibProfile) -> Self {
        ChannelParams {
            gammas: [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64],
            px_per_cell: 8.0,
            cell_size_px: p.cell_size_px as f64,
            homography: IDENTITY,
            blur_sigma_rgb: [0.0; 3],
            crosstalk_rg: 0.0,
            crosstalk_gb: 0.0,
            gain: [1.0; 3],
            offset: [0.0; 3],
            noise_sigma: 0.0,
        }
    }

    /// Изотропный дефокус: одинаковая σ на все три канала. Замена прежнему
    /// скалярному полю `blur_sigma_px` — держит инвариант `blur_sigma_rgb`.
    pub fn set_blur(&mut self, sigma: f64) {
        self.blur_sigma_rgb = [sigma; 3];
    }

    /// Пресет продольной хроматической аберрации (§5.1): σ_R = 1.3·σ, σ_G = σ,
    /// σ_B = 1.5·σ. Красный и синий фокусируются в разных плоскостях, зелёный
    /// посередине — типичная картина недорогого объектива. Средняя σ = 1.4·σ_G,
    /// что и есть ядро R−B-хромы (K_R+K_B)/2 из свойства (b) §5.1.
    /// (Задействован тестами свойств §5.1; в бинаре развёрток не используется.)
    #[allow(dead_code)]
    pub fn set_aberration(&mut self, sigma: f64) {
        self.blur_sigma_rgb = [1.3 * sigma, sigma, 1.5 * sigma];
    }
}

/// Мягкий перспективный пресет: единичная гомография с малыми членами
/// перспективы (лёгкий «keystone»). `size_px` — сторона символа: члены
/// нормируются на неё, чтобы искажение не зависело от разрешения.
pub fn mild_perspective(size_px: usize) -> [[f64; 3]; 3] {
    let s = size_px as f64;
    // в нормированных координатах n=coord/S матрица [[1,0,0],[0,1,0],[kx,ky,1]];
    // перевод в пиксельные координаты даёт члены kx/S, ky/S в нижней строке.
    let kx = 0.12;
    let ky = 0.08;
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [kx / s, ky / s, 1.0]]
}

/// Применить гомографию к точке (u, v) в однородных координатах.
#[inline]
fn apply_h(m: &[[f64; 3]; 3], u: f64, v: f64) -> (f64, f64) {
    let x = m[0][0] * u + m[0][1] * v + m[0][2];
    let y = m[1][0] * u + m[1][1] * v + m[1][2];
    let w = m[2][0] * u + m[2][1] * v + m[2][2];
    (x / w, y / w)
}

/// Обратная 3×3 матрица через присоединённую (кофакторы). Для наших
/// почти-единичных гомографий определитель заведомо ненулевой.
fn mat3_inverse(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let inv_det = 1.0 / det;

    [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]
}

/// Геометрия проекции: прямое (символ->снимок) и обратное (снимок->символ)
/// отображения. Прямое = масштаб ∘ гомография (сначала гомография, потом
/// умножение на масштаб), со сдвигом начала так, чтобы bbox спроецированного
/// символа начинался в (0, 0). Выходной размер = bbox спроецированных углов.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub scale: f64,
    pub h: [[f64; 3]; 3],
    h_inv: [[f64; 3]; 3],
    ox: f64,
    oy: f64,
    pub out_w: usize,
    pub out_h: usize,
}

impl Geometry {
    /// Построить геометрию для масштаба, гомографии и стороны символа (px).
    pub fn new(scale: f64, h: [[f64; 3]; 3], size_px: usize) -> Self {
        let h_inv = mat3_inverse(&h);
        let s = size_px as f64;
        let corners = [(0.0, 0.0), (s, 0.0), (0.0, s), (s, s)];
        let mut minx = f64::INFINITY;
        let mut miny = f64::INFINITY;
        let mut maxx = f64::NEG_INFINITY;
        let mut maxy = f64::NEG_INFINITY;
        for &(u, v) in &corners {
            let (uh, vh) = apply_h(&h, u, v);
            let rx = scale * uh;
            let ry = scale * vh;
            minx = minx.min(rx);
            maxx = maxx.max(rx);
            miny = miny.min(ry);
            maxy = maxy.max(ry);
        }
        let out_w = (maxx - minx).ceil().max(1.0) as usize;
        let out_h = (maxy - miny).ceil().max(1.0) as usize;
        Geometry { scale, h, h_inv, ox: minx, oy: miny, out_w, out_h }
    }

    /// Прямое отображение: координаты плоскости символа (display-px) ->
    /// координаты снимка (camera-px). Это ровно genie-`map` для демода.
    #[inline]
    pub fn forward(&self, u: f64, v: f64) -> (f64, f64) {
        let (uh, vh) = apply_h(&self.h, u, v);
        (self.scale * uh - self.ox, self.scale * vh - self.oy)
    }

    /// Обратное отображение: координаты снимка -> координаты плоскости символа
    /// (для inverse-map warp).
    #[inline]
    pub fn inverse(&self, x: f64, y: f64) -> (f64, f64) {
        let uh = (x + self.ox) / self.scale;
        let vh = (y + self.oy) / self.scale;
        apply_h(&self.h_inv, uh, vh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_geometry_is_lossless_roundtrip() {
        let g = Geometry::new(1.0, IDENTITY, 100);
        assert_eq!(g.out_w, 100);
        assert_eq!(g.out_h, 100);
        let (x, y) = g.forward(37.5, 62.25);
        assert!((x - 37.5).abs() < 1e-9);
        assert!((y - 62.25).abs() < 1e-9);
        let (u, v) = g.inverse(x, y);
        assert!((u - 37.5).abs() < 1e-9);
        assert!((v - 62.25).abs() < 1e-9);
    }

    #[test]
    fn scale_half_shrinks_output() {
        let g = Geometry::new(0.5, IDENTITY, 200);
        assert_eq!(g.out_w, 100);
        let (x, _) = g.forward(100.0, 0.0);
        assert!((x - 50.0).abs() < 1e-9);
    }

    #[test]
    fn matrix_inverse_roundtrips_forward_inverse() {
        // мягкая перспектива: forward∘inverse ≈ identity на снимке
        let h = mild_perspective(300);
        let g = Geometry::new(0.5, h, 300);
        for &(x, y) in &[(10.0, 20.0), (80.0, 130.0), (5.0, 5.0)] {
            let (u, v) = g.inverse(x, y);
            let (x2, y2) = g.forward(u, v);
            assert!((x - x2).abs() < 1e-6, "x {x} vs {x2}");
            assert!((y - y2).abs() < 1e-6, "y {y} vs {y2}");
        }
    }
}
