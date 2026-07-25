//! psicode-sim: симулятор канала «монитор -> камера» и Monte Carlo-развёртки SER.
//!
//! Физика L0/§9.1 SPEC: гамма, гомография, дефокус-блюр, кросстолк, усиление/
//! смещение, шум. Плюс развёртки, заполняющие BENCHMARKS.md §1–2. Зависимостей,
//! кроме psicode-core, нет — весь ГПСЧ, геометрия и PPM свои.
//!
//! Подкоманды:
//!   sweep        — три развёртки (blur σ; px/клетку; шум) эталонным профилем
//!   dump <dir>   — P6 PPM: чистый кадр + канал при σ=1 и σ=4 (глазами посмотреть)

mod channel;
mod image;
mod pipeline;
mod report;
mod rng;

use channel::{ChannelParams, Geometry, IDENTITY};
use pipeline::{apply_channel, drive_to_linear, run_trial};
use psicode_core::{symbol, CalibProfile, ChromaMode};
use rng::{seed_for, Rng};
use std::path::Path;
use std::time::Instant;

/// Эталонный профиль §7.4 SPEC — конфигурация симулятора по умолчанию.
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
        quiet_zone: 1,     // 4 клетки
        fec_overhead: 2,
    }
}

const TRIALS: usize = 15;

/// Средний SER по TRIALS попыткам в точке развёртки `point_idx`.
fn mean_ser(p: &CalibProfile, ch: &ChannelParams, point_idx: usize) -> f64 {
    let mut acc = 0.0;
    for t in 0..TRIALS {
        acc += run_trial(p, ch, point_idx, t).ser();
    }
    acc / TRIALS as f64
}

fn cmd_sweep() {
    let p = reference_profile();
    let base = ChannelParams::from_profile(&p);
    println!("# psicode-sim sweeps");
    println!(
        "профиль §7.4: cell {} px, luma {} бит, chroma {} бит, γ_G {:.3}; \
         канал: кросстолк {}%/{}%, шум σ {:.2} град/255",
        p.cell_size_px,
        p.luma_bits,
        p.chroma_bits(),
        p.gamma_g(),
        p.crosstalk_rg_pct(),
        p.crosstalk_gb_pct(),
        p.noise_sigma(),
    );
    println!("{TRIALS} попыток на точку\n");

    let t0 = Instant::now();
    let mut point = 0usize;

    // (a) SER vs blur σ, px_per_cell = 8  (BENCHMARKS §1)
    let sigmas = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0];
    println!("## 1. SER vs blur σ (px/cell = 8)");
    println!("| source | blur σ (px) → | 0.5 | 1 | 2 | 4 | 6 | 8 |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &s in &sigmas {
        let mut ch = base.clone();
        ch.px_per_cell = 8.0;
        ch.blur_sigma_px = s;
        cells.push(report::sig4(mean_ser(&p, &ch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    // (b) SER vs px/cell, σ = 1  (BENCHMARKS §2)
    let ppcs = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5];
    println!("\n## 2. SER vs px/cell (blur σ = 1)");
    println!("| source | px/cell → | 8 | 6 | 4 | 3 | 2 | 1.5 |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &ppc in &ppcs {
        let mut ch = base.clone();
        ch.px_per_cell = ppc;
        ch.blur_sigma_px = 1.0;
        cells.push(report::sig4(mean_ser(&p, &ch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    // (c) бонус: SER vs множитель шума, σ = 1, px/cell = 8
    let base_noise = base.noise_sigma;
    let mults = [1.0, 2.0, 4.0, 8.0];
    println!("\n## 3. (бонус) SER vs множитель шума (σ_blur = 1, px/cell = 8)");
    println!("| source | noise ×base → | 1 | 2 | 4 | 8 |");
    println!("|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &m in &mults {
        let mut ch = base.clone();
        ch.px_per_cell = 8.0;
        ch.blur_sigma_px = 1.0;
        ch.noise_sigma = base_noise * m;
        cells.push(report::sig4(mean_ser(&p, &ch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

const DUMP_DIR_DEFAULT: &str = "psicode-sim-dump";

/// Что за файл дампа: `clean.ppm` (сырые drive-байты, геометрия 1:1) либо
/// выход канала `channel_*.ppm` (после `image_to_drive`, геометрия из `ch`).
enum DumpKind {
    Clean,
    Channel(ChannelParams),
}

struct DumpSpec {
    name: String,
    kind: DumpKind,
}

/// Розыгрыш клеток дампа: тот же порядок вызовов ГПСЧ, что в тракте попытки,
/// поэтому при одинаковом сиде клетки совпадают. Общий для dump и readback.
fn draw_cells(p: &CalibProfile, rng: &mut Rng) -> Vec<u8> {
    let bpc = symbol::bits_per_cell(p);
    let n_levels = 1u32 << bpc;
    let n_cells = symbol::PAYLOAD_COLS * symbol::PAYLOAD_ROWS;
    (0..n_cells)
        .map(|_| rng.next_u32_below(n_levels) as u8)
        .collect()
}

/// Детерминированный список файлов дампа (порядок фиксирует и последовательность
/// расхода ГПСЧ на шум: clean без шума, затем s1, s4, перспектива). ГПСЧ не
/// трогает — параметры канала выводятся только из профиля.
fn dump_specs(p: &CalibProfile) -> Vec<DumpSpec> {
    let mut v = Vec::new();
    v.push(DumpSpec { name: "clean.ppm".into(), kind: DumpKind::Clean });
    for &s in &[1.0f64, 4.0] {
        let mut ch = ChannelParams::from_profile(p);
        ch.px_per_cell = 8.0;
        ch.blur_sigma_px = s;
        v.push(DumpSpec {
            name: format!("channel_s{}.ppm", s as u32),
            kind: DumpKind::Channel(ch),
        });
    }
    let mut ch = ChannelParams::from_profile(p);
    ch.px_per_cell = 8.0;
    ch.blur_sigma_px = 1.0;
    ch.homography = channel::mild_perspective(channel::symbol_size_px(p));
    v.push(DumpSpec {
        name: "channel_perspective.ppm".into(),
        kind: DumpKind::Channel(ch),
    });
    v
}

fn cmd_dump(dir: &Path) {
    let p = reference_profile();
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("не создать {}: {e}", dir.display());
        std::process::exit(1);
    }

    // фиксированный сид -> воспроизводимый кадр и шум
    let mut rng = Rng::new(seed_for(0, 0));
    let cells = draw_cells(&p, &mut rng);
    let frame = symbol::render_symbol(&p, &cells);

    for spec in dump_specs(&p) {
        let path = dir.join(&spec.name);
        match spec.kind {
            // чистый кадр: drive-байты как есть (шум не расходуется)
            DumpKind::Clean => {
                report::write_ppm(&path, frame.size_px, frame.size_px, &frame.rgb).unwrap();
            }
            // выход канала: шум берётся из общего ГПСЧ, drive для человеческого глаза
            DumpKind::Channel(ch) => {
                let (img, _geom) = apply_channel(&frame, &ch, &mut rng);
                let drive = report::image_to_drive(&img, ch.gammas);
                report::write_ppm(&path, img.w, img.h, &drive).unwrap();
            }
        }
        println!("записан {}", path.display());
    }
}

/// Сравнение демодулированных клеток с эталоном: (неверных клеток, ошибочных
/// бит, всего бит). Биты — младшие bits_per_cell (Грей-код), как в run_trial.
fn compare_cells(sent: &[u8], got: &[u8], bpc: u32) -> (usize, u32, u32) {
    let mask = (1u32 << bpc) - 1;
    let mut wrong_cells = 0;
    let mut wrong_bits = 0;
    let mut total_bits = 0;
    for (&a, &b) in sent.iter().zip(got.iter()) {
        if a != b {
            wrong_cells += 1;
        }
        wrong_bits += ((a ^ b) as u32 & mask).count_ones();
        total_bits += bpc;
    }
    (wrong_cells, wrong_bits, total_bits)
}

/// Декодирование ИЗ ФАЙЛОВ на диске: сквозь 8-битное квантование и обратную
/// гамму, которой писался PPM. Если файлов нет — сначала пишет дамп.
fn cmd_readback(dir: &Path) {
    let p = reference_profile();
    let specs = dump_specs(&p);

    // нет хотя бы одного файла -> сгенерировать дамп (байт-в-байт тот же)
    let missing = specs.iter().any(|s| !dir.join(&s.name).exists());
    if missing {
        println!("(файлы отсутствуют — генерирую дамп в {})", dir.display());
        cmd_dump(dir);
        println!();
    }

    // эталонные клетки: тот же сид и тот же розыгрыш, что в дампе
    let mut rng = Rng::new(seed_for(0, 0));
    let cells = draw_cells(&p, &mut rng);
    let size_px = channel::symbol_size_px(&p);
    let bpc = symbol::bits_per_cell(&p);
    let total = cells.len();

    println!("# readback ({total} клеток payload)");
    println!("| file | correct/total | SER | BER |");
    println!("|---|---|---|---|");

    for spec in &specs {
        let path = dir.join(&spec.name);
        let (w, h, drive) = match report::read_ppm(&path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{}: ошибка чтения: {e}", spec.name);
                continue;
            }
        };

        // реконструкция сенсорно-линейного и геометрии ровно как при записи
        let (img, geom) = match &spec.kind {
            // clean: идеальный сенсор из drive-байт, символ-плоскость 1:1
            DumpKind::Clean => {
                let gammas = [p.gamma_r() as f64, p.gamma_g() as f64, p.gamma_b() as f64];
                let img = drive_to_linear(&drive, w, h, gammas);
                let geom = Geometry::new(1.0, IDENTITY, size_px);
                (img, geom)
            }
            // channel: обращаем ту же гамму и повторяем ту же Geometry, что писала файл
            DumpKind::Channel(ch) => {
                let img = drive_to_linear(&drive, w, h, ch.gammas);
                let scale = ch.px_per_cell / ch.cell_size_px;
                let geom = Geometry::new(scale, ch.homography, size_px);
                (img, geom)
            }
        };

        // геометрия файла должна совпасть с ожидаемой (иначе рассинхрон записи)
        if geom.out_w != w || geom.out_h != h {
            eprintln!(
                "{}: размер {w}x{h} не совпал с геометрией {}x{}",
                spec.name, geom.out_w, geom.out_h
            );
        }

        let map = |u: f64, v: f64| geom.forward(u, v);
        let sample = |x: f64, y: f64| img.sample(x, y);
        let out = symbol::demod_symbol(&p, &map, &sample);

        let (wrong, wrong_bits, total_bits) = compare_cells(&cells, &out, bpc);
        let correct = total - wrong;
        let ser = wrong as f64 / total as f64;
        let ber = wrong_bits as f64 / total_bits as f64;
        println!(
            "| {} | {correct}/{total} | {} | {} |",
            spec.name,
            report::sig4(ser),
            report::sig4(ber),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("sweep") => cmd_sweep(),
        Some("dump") => {
            let dir = args.get(2).map(String::as_str).unwrap_or(DUMP_DIR_DEFAULT);
            cmd_dump(Path::new(dir));
        }
        Some("readback") => {
            let dir = args.get(2).map(String::as_str).unwrap_or(DUMP_DIR_DEFAULT);
            cmd_readback(Path::new(dir));
        }
        _ => {
            eprintln!("usage: psicode-sim <sweep | dump [dir] | readback [dir]>");
            std::process::exit(2);
        }
    }
}
