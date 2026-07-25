//! Растровое изображение канала: линейный (сенсорный) RGB [0,1], построчно.
//!
//! Единственный тип пикселя во всём симуляторе — `[f32; 3]` в линейном свете
//! (не gamma-домен). Координатное соглашение: центр пикселя (i, j) лежит в
//! целочисленной точке (i, j); сэмплирование вне границ — clamp-to-edge.

/// Линейное RGB-изображение [0,1], `w × h`, пиксели построчно (row-major).
#[derive(Clone, Debug)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    /// `h·w` пикселей, `data[y*w + x] = [R, G, B]`, линейный свет [0,1].
    pub data: Vec<[f32; 3]>,
}

impl Image {
    /// Чёрное изображение `w × h`.
    pub fn new(w: usize, h: usize) -> Self {
        Image { w, h, data: vec![[0.0; 3]; w * h] }
    }

    /// Изображение `w × h`, целиком залитое одним цветом.
    /// (Конструктор общего назначения; в бинаре пока задействован только тестами.)
    #[allow(dead_code)]
    pub fn filled(w: usize, h: usize, color: [f32; 3]) -> Self {
        Image { w, h, data: vec![color; w * h] }
    }

    /// Пиксель (x, y) без интерполяции (x, y — целочисленный индекс в границах).
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> [f32; 3] {
        self.data[y * self.w + x]
    }

    /// Записать пиксель (x, y).
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: [f32; 3]) {
        self.data[y * self.w + x] = v;
    }

    /// Билинейная выборка в дробной точке (x, y). Центр пикселя (i, j) — в
    /// целой точке (i, j). Вне изображения координаты зажимаются к краю
    /// (clamp-to-edge), поэтому функция определена на всей плоскости.
    pub fn sample(&self, x: f64, y: f64) -> [f32; 3] {
        if self.w == 0 || self.h == 0 {
            return [0.0; 3];
        }
        // зажимаем координату к [0, w-1] / [0, h-1] — выборка за краем берёт край
        let xf = x.clamp(0.0, (self.w - 1) as f64);
        let yf = y.clamp(0.0, (self.h - 1) as f64);
        let x0 = xf.floor() as usize;
        let y0 = yf.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = (xf - x0 as f64) as f32;
        let ty = (yf - y0 as f64) as f32;

        let p00 = self.at(x0, y0);
        let p10 = self.at(x1, y0);
        let p01 = self.at(x0, y1);
        let p11 = self.at(x1, y1);

        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let top = p00[c] + (p10[c] - p00[c]) * tx;
            let bot = p01[c] + (p11[c] - p01[c]) * tx;
            out[c] = top + (bot - top) * ty;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_hits_pixel_centers_exactly() {
        // 2x2 с различимыми пикселями: выборка в целых точках = сами пиксели
        let mut img = Image::new(2, 2);
        img.set(0, 0, [0.1, 0.0, 0.0]);
        img.set(1, 0, [0.2, 0.0, 0.0]);
        img.set(0, 1, [0.3, 0.0, 0.0]);
        img.set(1, 1, [0.4, 0.0, 0.0]);
        assert!((img.sample(0.0, 0.0)[0] - 0.1).abs() < 1e-6);
        assert!((img.sample(1.0, 0.0)[0] - 0.2).abs() < 1e-6);
        assert!((img.sample(1.0, 1.0)[0] - 0.4).abs() < 1e-6);
        // середина между всеми четырьмя — среднее
        assert!((img.sample(0.5, 0.5)[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn sample_clamps_to_edge() {
        let mut img = Image::new(2, 2);
        img.set(0, 0, [0.5, 0.0, 0.0]);
        img.set(1, 0, [0.9, 0.0, 0.0]);
        // далеко за левым краем -> угловой пиксель (0,0)
        assert!((img.sample(-10.0, 0.0)[0] - 0.5).abs() < 1e-6);
        // далеко за правым -> (1,0)
        assert!((img.sample(100.0, 0.0)[0] - 0.9).abs() < 1e-6);
    }
}
