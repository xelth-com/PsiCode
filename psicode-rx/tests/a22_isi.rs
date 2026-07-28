//! Закрепляющий тест на СОХРАНЁННЫХ снимках Galaxy A22 (цветной набор).
//!
//! Пиннит главный результат выравнивателя межклеточной интерференции: на
//! кадрах, где созвездие не схлопнуто, КЭШИРОВАННОЕ (медианное по кадрам) ядро
//! поднимает выживаемость страйпов с 9/32 до 29/32.
//!
//! Заодно пиннит ГЕЙТ КАЧЕСТВА: `IsiDemod::quality` на этих шести снимках даёт
//! 1.55 и 1.70 для двух кадров со схлопнутым созвездием против 2.19…2.57 для
//! четырёх остальных, то есть разделяет их порогом 1.8 с запасом — и делает это
//! БЕЗ эталона, по собственным решениям кадра.
//!
//! Числа 9/32 и 29/32 сняты ПРОДАКШН-путём `symbol::demod_symbol_isi`. Офлайновый
//! разбор в `examples/isi_eq.rs` даёт 10/32 и 30/32: он обращает гамму профилем
//! в отдельной ветке и расходится на одном страйпе. Авторитетны здешние.
//!
//! Снимки лежат вне репозитория (сырые кадры камеры, десятки мегабайт), поэтому
//! путь берётся из переменной окружения `PSICODE_A22_DUMPS`. Без неё тест
//! ПРОПУСКАЕТСЯ — так он не ломает сборку там, где данных нет, но остаётся
//! исполняемым там, где они есть.
//!
//! ```text
//! PSICODE_A22_DUMPS=<каталог с dump0.meta и dump{0..5}.{y,u,v}> cargo test -p psicode-rx
//! ```

use psicode_core::detect::{self, Detection};
use psicode_core::isi::IsiKernel;
use psicode_core::l3;
use psicode_core::symbol::{self, IsiConfig, ISI_QUALITY_MIN};
use psicode_core::tone;
use psicode_rx::tx_chromatic_profile;
use psicode_rx::yuv::YuvFrame;
use std::fs;

/// Кроп вокруг символа: холодная детекция на полном загромождённом кадре
/// (символ в окне поверх редактора и терминала) промахивается. Кроп одинаков
/// для всех кадров, поэтому на измеряемые величины не влияет.
const CROP: [usize; 4] = [700, 0, 900, 860];

struct Dump {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    w: usize,
    h: usize,
    uv_stride: usize,
    uv_px: usize,
}

impl Dump {
    fn frame(&self) -> YuvFrame<'_> {
        YuvFrame {
            y: &self.y,
            u: &self.u,
            v: &self.v,
            w: self.w,
            h: self.h,
            y_stride: self.w,
            uv_stride: self.uv_stride,
            uv_pixel_stride: self.uv_px,
        }
    }

    fn raw(&self, x: f64, y: f64) -> [f32; 3] {
        let xc = x.clamp(0.0, (self.w - 1) as f64);
        let yc = y.clamp(0.0, (self.h - 1) as f64);
        let (x0, y0) = (xc.floor() as usize, yc.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = ((xc - x0 as f64) as f32, (yc - y0 as f64) as f32);
        let fr = self.frame();
        let (s00, s10, s01, s11) = (
            fr.rgb_at(x0, y0),
            fr.rgb_at(x1, y0),
            fr.rgb_at(x0, y1),
            fr.rgb_at(x1, y1),
        );
        let mut o = [0.0f32; 3];
        for c in 0..3 {
            let a = s00[c] * (1.0 - fx) + s10[c] * fx;
            let b = s01[c] * (1.0 - fx) + s11[c] * fx;
            o[c] = a * (1.0 - fy) + b * fy;
        }
        o
    }
}

fn load(dir: &str) -> Vec<Dump> {
    let meta = fs::read_to_string(format!("{dir}/dump0.meta")).expect("dump0.meta");
    let m: Vec<usize> = meta
        .split_whitespace()
        .map(|t| t.parse().expect("meta"))
        .collect();
    let (fw, fh, y_stride, uv_stride, uv_px) = (m[0], m[1], m[2], m[3], m[4]);
    let [x0, y0, cw0, ch0] = CROP;
    let (x0, y0) = (x0 & !1, y0 & !1);
    let (cw, ch) = ((cw0.min(fw - x0)) & !1, (ch0.min(fh - y0)) & !1);
    let mut out = Vec::new();
    let mut i = 0usize;
    loop {
        let Ok(yb) = fs::read(format!("{dir}/dump{i}.y")) else {
            break;
        };
        let ub = fs::read(format!("{dir}/dump{i}.u")).unwrap_or_default();
        let vb = fs::read(format!("{dir}/dump{i}.v")).unwrap_or_default();
        let mut ny = vec![0u8; cw * ch];
        for j in 0..ch {
            let s = (y0 + j) * y_stride + x0;
            ny[j * cw..(j + 1) * cw].copy_from_slice(&yb[s..s + cw]);
        }
        let cs = if uv_px == 2 { cw } else { cw / 2 };
        let (mut nu, mut nv) = (vec![128u8; cs * (ch / 2)], vec![128u8; cs * (ch / 2)]);
        for j in 0..ch / 2 {
            for i2 in 0..cw / 2 {
                let s = (y0 / 2 + j) * uv_stride + (x0 / 2 + i2) * uv_px;
                if s < ub.len() {
                    nu[j * cs + i2 * uv_px] = ub[s];
                }
                if s < vb.len() {
                    nv[j * cs + i2 * uv_px] = vb[s];
                }
            }
        }
        out.push(Dump {
            y: ny,
            u: nu,
            v: nv,
            w: cw,
            h: ch,
            uv_stride: cs,
            uv_px,
        });
        i += 1;
    }
    out
}

/// Поотсчётная медиана набора ядер — то же, что делает `KernelPool::working`.
fn median(ks: &[[IsiKernel; 3]]) -> [IsiKernel; 3] {
    let mut out = [IsiKernel::identity(ks[0][0].radius); 3];
    for (c, o) in out.iter_mut().enumerate() {
        let r = ks[0][c].radius;
        let side = 2 * r + 1;
        let mut k = IsiKernel::identity(r);
        for dr in -(r as i32)..=(r as i32) {
            for dc in -(r as i32)..=(r as i32) {
                let mut v: Vec<f64> = ks.iter().map(|h| h[c].tap(dr, dc)).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                k.taps[((dr + r as i32) as usize) * side + (dc + r as i32) as usize] =
                    v[v.len() / 2];
            }
        }
        k.taps[r * side + r] = 1.0;
        *o = k;
    }
    out
}

#[test]
fn cached_kernel_reproduces_a22_stripe_survival() {
    let Ok(dir) = std::env::var("PSICODE_A22_DUMPS") else {
        eprintln!("PSICODE_A22_DUMPS не задана — тест пропущен");
        return;
    };
    let dumps = load(&dir);
    assert!(dumps.len() >= 6, "нужно >= 6 снимков, найдено {}", dumps.len());
    let p = tx_chromatic_profile();
    let bpc = symbol::bits_per_cell(&p);
    let cfg = IsiConfig::default();

    // общее семя детекции с кадра 0, дальше каждый кадр выравнивается независимо
    let luma0: Vec<f32> = {
        let d = &dumps[0];
        let fr = d.frame();
        (0..d.w * d.h).map(|i| fr.y_norm(i % d.w, i / d.w)).collect()
    };
    let (w, h) = (dumps[0].w, dumps[0].h);
    let mut seed = detect::detect_symbol(w, h, &luma0)
        .or_else(|_| detect::detect_symbol_acquire(w, h, &luma0))
        .expect("детекция на кадре 0");
    for _ in 0..10 {
        match detect::track_symbol(w, h, &luma0, &seed) {
            Ok(d) if d.score > seed.score + 1e-4 => seed = d,
            _ => break,
        }
    }
    let dets: Vec<Option<Detection>> = dumps
        .iter()
        .map(|dp| {
            let fr = dp.frame();
            let l: Vec<f32> = (0..dp.w * dp.h)
                .map(|i| fr.y_norm(i % dp.w, i / dp.w))
                .collect();
            let mut d = detect::track_symbol(dp.w, dp.h, &l, &seed).ok();
            for _ in 0..10 {
                let Some(cur) = d.as_ref() else { break };
                match detect::track_symbol(dp.w, dp.h, &l, cur) {
                    Ok(n) if n.score > cur.score + 1e-4 => d = Some(n),
                    _ => break,
                }
            }
            d
        })
        .collect();

    // ПРОХОД 1: покадровые ядра + гейт качества (как в RxSession)
    let mut pool: Vec<[IsiKernel; 3]> = Vec::new();
    let mut clean: Vec<usize> = Vec::new();
    let mut base_alive = 0usize;
    for (fi, dp) in dumps.iter().enumerate() {
        let Some(d) = dets[fi].as_ref() else { continue };
        let map = detect::frame_map(&p, d);
        let raw = |x: f64, y: f64| dp.raw(x, y);
        let g = tone::estimate_channel_gammas(&p, &map, &raw);
        let lin = |x: f64, y: f64| -> [f32; 3] {
            let s = raw(x, y);
            [
                (s[0] as f64).max(0.0).powf(g[0]) as f32,
                (s[1] as f64).max(0.0).powf(g[1]) as f32,
                (s[2] as f64).max(0.0).powf(g[2]) as f32,
            ]
        };
        let r = symbol::demod_symbol_isi(&p, &map, &lin, None, &cfg);
        eprintln!(
            "[a22] кадр {fi}: quality {:.2}, сила ядра {:.4}, cell_mtf {:.3}",
            r.quality[0],
            r.kernels[1].strength(),
            symbol::cell_scale_mtf(&p, &map, &raw)
        );
        if r.quality[0] >= ISI_QUALITY_MIN {
            clean.push(fi);
            pool.push(r.kernels);
            base_alive += l3::parse_frame(&r.base_cells, bpc)
                .stripes_ok
                .iter()
                .filter(|&&b| b)
                .count();
        }
    }
    assert_eq!(
        clean.len(),
        4,
        "гейт качества обязан оставить ровно 4 кадра из 6, оставил {clean:?}"
    );

    // ПРОХОД 2: медианное (кэшированное) ядро на те же кадры
    let med = median(&pool);
    let mut cfg_cached = cfg;
    cfg_cached.kernel = Some(med);
    let mut alive = 0usize;
    for &fi in &clean {
        let dp = &dumps[fi];
        let d = dets[fi].as_ref().unwrap();
        let map = detect::frame_map(&p, d);
        let raw = |x: f64, y: f64| dp.raw(x, y);
        let g = tone::estimate_channel_gammas(&p, &map, &raw);
        let lin = |x: f64, y: f64| -> [f32; 3] {
            let s = raw(x, y);
            [
                (s[0] as f64).max(0.0).powf(g[0]) as f32,
                (s[1] as f64).max(0.0).powf(g[1]) as f32,
                (s[2] as f64).max(0.0).powf(g[2]) as f32,
            ]
        };
        let r = symbol::demod_symbol_isi(&p, &map, &lin, None, &cfg_cached);
        let ok = l3::parse_frame(&r.cells, bpc)
            .stripes_ok
            .iter()
            .filter(|&&b| b)
            .count();
        alive += ok;
        eprintln!("[a22] кадр {fi}: страйпов с кэшированным ядром {ok}/8");
    }
    eprintln!("[a22] ИТОГО: база {base_alive}/32 -> кэшированное ядро {alive}/32");
    assert!(
        base_alive <= 12,
        "база должна давать ~9/32, а дала {base_alive}"
    );
    assert!(
        alive >= 29,
        "кэшированное ядро должно давать >= 29/32, а дало {alive}"
    );
}
