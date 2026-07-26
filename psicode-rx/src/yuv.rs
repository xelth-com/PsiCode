//! Разбор кадра Camera2 `YUV_420_888` и преобразование в RGB (§8).
//!
//! Camera2 отдаёт три плоскости с произвольными шагами: Y полного разрешения
//! (`y_stride` байт/строку), U и V субдискретизированы 2×2 (`uv_stride` байт/
//! строку, `uv_pixel_stride` байт между соседними отсчётами хромы). Встречаются
//! оба варианта упаковки хромы: планарный (`uv_pixel_stride == 1`, отдельные
//! плотные плоскости U и V, стиль I420/YV12) и полупланарный (`uv_pixel_stride
//! == 2`, U и V чередуются в одном буфере, стиль NV12/NV21 — плоскости `u`/`v`
//! указывают на первый байт своего компонента).
//!
//! Цвет: ограниченный диапазон (studio swing) BT.601 — умолчание Camera2.
//! Все обращения к буферам с зажимом/`get`-защитой: битый кадр (кривые шаги,
//! обрезанные буферы) НИКОГДА не паникует, а даёт нейтральный отсчёт.

/// Заимствованный кадр YUV_420_888 (владения буферами нет).
pub struct YuvFrame<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub y_stride: usize,
    pub uv_stride: usize,
    pub uv_pixel_stride: usize,
}

impl<'a> YuvFrame<'a> {
    /// Нормированная яркость [0,1] пикселя (x,y) для ДЕТЕКЦИИ (§3.2): детекция
    /// перцентильно-корреляционная и монотонно-инвариантна, поэтому берём Y как
    /// есть (без гаммы). Выход за буфер -> 0.
    #[inline]
    pub fn y_norm(&self, x: usize, y: usize) -> f32 {
        let idx = y * self.y_stride + x;
        self.y.get(idx).copied().unwrap_or(0) as f32 / 255.0
    }

    /// Сырой (сенсорно-линейный ДО тон-кривой) RGB [0,1] пикселя (x,y) через
    /// ближайший отсчёт хромы. BT.601 limited-range. Защита от выхода за буфер:
    /// Y -> 0, хрома -> нейтраль 128.
    #[inline]
    pub fn rgb_at(&self, x: usize, y: usize) -> [f32; 3] {
        let yy = self.y.get(y * self.y_stride + x).copied().unwrap_or(0);
        let cx = x >> 1;
        let cy = y >> 1;
        let ci = cy * self.uv_stride + cx * self.uv_pixel_stride;
        let uu = self.u.get(ci).copied().unwrap_or(128);
        let vv = self.v.get(ci).copied().unwrap_or(128);
        yuv601_limited(yy, uu, vv)
    }
}

/// BT.601 limited-range YCbCr -> RGB [0,1] (studio swing, умолчание Camera2).
///
/// Y ∈ [16,235], Cb/Cr ∈ [16,240] с центром 128. Коэффициенты:
///   R = 1.164·(Y−16)                    + 1.596·(Cr−128)
///   G = 1.164·(Y−16) − 0.392·(Cb−128) − 0.813·(Cr−128)
///   B = 1.164·(Y−16) + 2.017·(Cb−128)
/// Результат зажат в [0,1].
#[inline]
pub fn yuv601_limited(y: u8, u: u8, v: u8) -> [f32; 3] {
    let c = y as f32 - 16.0;
    let d = u as f32 - 128.0; // Cb
    let e = v as f32 - 128.0; // Cr
    let r = 1.164 * c + 1.596 * e;
    let g = 1.164 * c - 0.392 * d - 0.813 * e;
    let b = 1.164 * c + 2.017 * d;
    [
        (r / 255.0).clamp(0.0, 1.0),
        (g / 255.0).clamp(0.0, 1.0),
        (b / 255.0).clamp(0.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_gray_roundtrips_channels_equal() {
        // серый: Cb=Cr=128 -> R=G=B, и близко к линейной шкале Y.
        for &yv in &[16u8, 128, 235] {
            let rgb = yuv601_limited(yv, 128, 128);
            assert!((rgb[0] - rgb[1]).abs() < 1e-4, "R==G для серого");
            assert!((rgb[1] - rgb[2]).abs() < 1e-4, "G==B для серого");
        }
        // чёрный низ и белый верх шкалы limited-range
        assert!(yuv601_limited(16, 128, 128)[1] < 0.02);
        assert!(yuv601_limited(235, 128, 128)[1] > 0.98);
    }

    #[test]
    fn out_of_bounds_never_panics() {
        let yf = YuvFrame {
            y: &[10u8; 4],
            u: &[128u8; 1],
            v: &[128u8; 1],
            w: 100,
            h: 100,
            y_stride: 100,
            uv_stride: 50,
            uv_pixel_stride: 1,
        };
        // далеко за пределами крошечных буферов -> нейтральный отсчёт, без паники
        let _ = yf.rgb_at(99, 99);
        let _ = yf.y_norm(99, 99);
    }
}
