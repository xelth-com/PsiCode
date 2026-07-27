//! psicode-sim: симулятор канала «монитор -> камера» и Monte Carlo-развёртки SER.
//!
//! Физика L0/§9.1 SPEC: гамма, гомография, дефокус-блюр, кросстолк, усиление/
//! смещение, шум. Плюс развёртки, заполняющие BENCHMARKS.md §1–2. Зависимостей,
//! кроме psicode-core, нет — весь ГПСЧ, геометрия и PPM свои.
//!
//! Подкоманды:
//!   sweep        — развёртки SER: blur σ; px/клетку; шум; рассогласование
//!                  калибровки Δγ и белого уровня (§7.3) — эталонным профилем
//!   dump <dir>   — P6 PPM: чистый кадр + канал при σ=1 и σ=4 (глазами посмотреть)
//!   readback [dir] — декод из PPM-файлов (через 8-битное квантование)
//!   goodput      — модель goodput с выживанием полос (§6.2) по пространству
//!                  конфигураций (luma_bits × chroma × blur σ)
//!   exp          — экспериментальные студии для решений по спеке: ISP temporal
//!                  mixing (SER/выживаемость vs α) и детектируемость рамки/маяков
//!                  под блюром (matched-filter SNR)

mod channel;
mod exp;
mod finder;
mod framed;
mod gprobe;
mod image;
mod l3live;
mod live;
mod modeb;
mod pipeline;
mod probe;
mod report;
mod rng;
mod tprobe;
mod transfer;

use channel::{ChannelParams, Geometry, IDENTITY};
use pipeline::{
    apply_channel, drive_to_linear, run_trial, run_trial_decode_io, run_trial_detect,
    run_trial_io, surviving_payload_bits,
};
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
        border: psicode_core::profile::BorderMode::LegacyInverted,
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
        ch.set_blur(s);
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
        ch.set_blur(1.0);
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
        ch.set_blur(1.0);
        ch.noise_sigma = base_noise * m;
        cells.push(report::sig4(mean_ser(&p, &ch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    // (d) SER vs рассогласование гаммы Δγ (§7.3): канал держит ИСТИННЫЕ гаммы
    // профиля, а декодер применяет профиль со сдвигом ВСЕХ ТРЁХ γ на Δγ (худший
    // случай устаревшей/ошибочной калибровки). σ_blur = 1, px/cell = 8.
    let mut ch_mismatch = base.clone();
    ch_mismatch.px_per_cell = 8.0;
    ch_mismatch.set_blur(1.0);
    let gamma_dq = [-8i32, -4, 0, 4, 8]; // Δγ = 0.025·dq = ∓0.2 … ±0.2
    println!("\n## 4. SER vs рассогласование гаммы Δγ (σ_blur = 1, px/cell = 8)");
    println!("| source | Δγ (все каналы) → | −0.2 | −0.1 | 0 | +0.1 | +0.2 |");
    println!("|---|---|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &dq in &gamma_dq {
        let p_rx = with_gamma_shift(&p, dq);
        cells.push(report::sig4(mean_ser_decode(&p, &p_rx, &ch_mismatch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    // (e) SER vs недооценка белого уровня декодером (бонус). Эталон уже на 100%
    // (потолок поля), поэтому мисматч содержателен только вниз: декодер думает,
    // что белый ниже истины -> плывут якоря нормировки §3.4 и амплитуды §5.1.
    let white_dq = [0i32, -1, -2, -3, -4]; // Δwhite = 3·dq % = 0 … −12%
    println!("\n## 5. (бонус) SER vs недооценка белого уровня (σ_blur = 1, px/cell = 8)");
    println!("| source | Δ white → | 0 | −3% | −6% | −9% | −12% |");
    println!("|---|---|---|---|---|---|---|");
    let mut cells = Vec::new();
    for &dq in &white_dq {
        let p_rx = with_white_shift(&p, dq);
        cells.push(report::sig4(mean_ser_decode(&p, &p_rx, &ch_mismatch, point)));
        point += 1;
    }
    println!("{}", report::table_row("sim | SER", &cells));

    // (f) ЧЕСТНЫЙ ПРИЁМНИК: реальная детекция ЗЧ-рамки (§3.2) вместо genie-
    // геометрии. Те же точки/сиды, что §1/§2, поэтому строка «genie SER» здесь
    // совпадает с §1/§2 (сверка). Соглашение: detect Err ⇒ кадр потерян ⇒ SER=1.0.
    println!("\n## 6. Detected vs genie: SER vs blur σ (px/cell = 8)");
    println!("| metric \\ σ (px) → | 0.5 | 1 | 2 | 4 | 6 | 8 |");
    println!("|---|---|---|---|---|---|---|");
    let aggs: Vec<DetectAgg> = sigmas
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut ch = base.clone();
            ch.px_per_cell = 8.0;
            ch.set_blur(s);
            detect_point(&p, &ch, i) // точка i == точка §1
        })
        .collect();
    print_detect_rows(&aggs);

    println!("\n## 7. Detected vs genie: SER vs px/cell (blur σ = 1)");
    println!("| metric \\ px/cell → | 8 | 6 | 4 | 3 | 2 | 1.5 |");
    println!("|---|---|---|---|---|---|---|");
    let aggs: Vec<DetectAgg> = ppcs
        .iter()
        .enumerate()
        .map(|(i, &ppc)| {
            let mut ch = base.clone();
            ch.px_per_cell = ppc;
            ch.set_blur(1.0);
            detect_point(&p, &ch, 6 + i) // точка 6+i == точка §2
        })
        .collect();
    print_detect_rows(&aggs);

    // (g) перспектива — случай, где детекции приходится реально работать.
    println!("\n## 8. Detected vs genie: mild perspective (keystone, σ=1, px/cell=8)");
    let mut ch_persp = base.clone();
    ch_persp.px_per_cell = 8.0;
    ch_persp.set_blur(1.0);
    ch_persp.homography = channel::mild_perspective(channel::symbol_size_px(&p));
    let a = detect_point(&p, &ch_persp, 300);
    println!("| case | genie SER | detected SER | detect success | mean score |");
    println!("|---|---|---|---|---|");
    println!(
        "| sim mild_perspective | {} | {} | {:.2} | {:.3} |",
        report::sig4(a.genie),
        report::sig4(a.detected),
        a.success,
        a.score
    );

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

/// Агрегат честного приёмника по точке развёртки.
struct DetectAgg {
    genie: f64,
    detected: f64,
    success: f64,
    score: f64,
}

/// Средние по TRIALS попыткам: genie-SER, detected-SER (потеря кадра ⇒ 1.0),
/// доля успешных детекций, средний score ПО УСПЕШНЫМ детекциям (0, если их нет).
fn detect_point(p: &CalibProfile, ch: &ChannelParams, point: usize) -> DetectAgg {
    let mut genie = 0.0;
    let mut detected = 0.0;
    let mut ok = 0usize;
    let mut score = 0.0;
    for t in 0..TRIALS {
        let r = run_trial_detect(p, ch, point, t);
        genie += r.genie_ser;
        detected += r.detected_ser.unwrap_or(1.0); // detect Err -> кадр потерян
        if r.detect_ok {
            ok += 1;
            score += r.score;
        }
    }
    let n = TRIALS as f64;
    DetectAgg {
        genie: genie / n,
        detected: detected / n,
        success: ok as f64 / n,
        score: if ok > 0 { score / ok as f64 } else { 0.0 },
    }
}

/// Печать 4 строк таблицы честного приёмника (genie/detected SER, успех, score).
fn print_detect_rows(aggs: &[DetectAgg]) {
    let genie: Vec<String> = aggs.iter().map(|a| report::sig4(a.genie)).collect();
    let det: Vec<String> = aggs.iter().map(|a| report::sig4(a.detected)).collect();
    let succ: Vec<String> = aggs.iter().map(|a| format!("{:.2}", a.success)).collect();
    let score: Vec<String> = aggs.iter().map(|a| format!("{:.3}", a.score)).collect();
    println!("{}", report::table_row("sim | genie SER", &genie));
    println!("{}", report::table_row("sim | detected SER", &det));
    println!("{}", report::table_row("sim | detect success", &succ));
    println!("{}", report::table_row("sim | mean score", &score));
}

/// Профиль-декодер со сдвигом всех трёх гамм на Δγ = 0.025·dq (меняем только
/// gamma_g_q; r_delta/b_delta постоянны -> γ_R, γ_G, γ_B сдвигаются синхронно).
fn with_gamma_shift(p: &CalibProfile, dq: i32) -> CalibProfile {
    let mut q = *p;
    q.gamma_g_q = (p.gamma_g_q as i32 + dq).clamp(0, 63) as u8;
    q
}

/// Профиль-декодер со сдвигом белого уровня на Δwhite = 3·dq % (white_level_q).
fn with_white_shift(p: &CalibProfile, dq: i32) -> CalibProfile {
    let mut q = *p;
    q.white_level_q = (p.white_level_q as i32 + dq).clamp(0, 15) as u8;
    q
}

/// Средний SER по TRIALS попыткам при РАССОГЛАСОВАННОМ декодере: передатчик и
/// канал знают `p_tx`, приёмник декодирует `p_rx` (§7.3).
fn mean_ser_decode(
    p_tx: &CalibProfile,
    p_rx: &CalibProfile,
    ch: &ChannelParams,
    point_idx: usize,
) -> f64 {
    let mut acc = 0.0;
    for t in 0..TRIALS {
        let (sent, got) = run_trial_decode_io(p_tx, p_rx, ch, point_idx, t);
        let wrong = sent.iter().zip(&got).filter(|(a, b)| a != b).count();
        acc += wrong as f64 / sent.len() as f64;
    }
    acc / TRIALS as f64
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
        ch.set_blur(s);
        v.push(DumpSpec {
            name: format!("channel_s{}.ppm", s as u32),
            kind: DumpKind::Channel(ch),
        });
    }
    let mut ch = ChannelParams::from_profile(p);
    ch.px_per_cell = 8.0;
    ch.set_blur(1.0);
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

// --- goodput: модель полезной пропускной с выживанием полос (§6.2/§6.3) ---

const GP_TRIALS: usize = 15;
const GP_FPS: f64 = 10.0; // 60 Гц / hold 6 периодов (§6.3)
const GP_FEC_KEEP: f64 = 0.8; // fec_overhead=2: 1 repair / 4 source -> ×4/5 (§6.1)
const GP_HEADER_SHARE: f64 = 0.015; // ~30 B заголовка / ~1959 B payload (§6.2)

/// Профиль эталона с подменёнными luma_bits и chroma_mode (одна точка конфига).
fn config_profile(base: &CalibProfile, luma: u8, chroma: ChromaMode) -> CalibProfile {
    let mut p = *base;
    p.luma_bits = luma;
    p.chroma_mode = chroma;
    p
}

/// Короткая метка хромо-режима для таблиц.
fn chroma_short(m: ChromaMode) -> &'static str {
    match m {
        ChromaMode::Mono => "Mono",
        ChromaMode::Chroma1 => "C1",
        ChromaMode::Chroma2 => "C2",
        ChromaMode::Chroma3 => "C3",
        ChromaMode::GreenOnly => "G",
        ChromaMode::ConstLuma1 => "CL1",
        ChromaMode::ConstLuma2 => "CL2",
        ChromaMode::ConstLuma3 => "CL3",
    }
}

/// (goodput kbit/s, сырой SER, FER) по GP_TRIALS попыткам в точке `point`.
/// goodput = среднее_уцелевших_бит/кадр · fps · FEC_keep · (1 − header_share).
/// «Уцелевшие биты» — сумма по ПОЛОСАМ, декодированным целиком без ошибок (§6.2):
/// одна ошибочная клетка убивает всю полосу — тот самый обрыв, что прячет сырой SER.
fn eval_goodput(p: &CalibProfile, ch: &ChannelParams, point: usize) -> (f64, f64, f64) {
    let bpc = symbol::bits_per_cell(p);
    let mut surv_bits = 0.0f64;
    let mut wrong = 0usize;
    let mut total_cells = 0usize;
    let mut dead_frames = 0usize;
    for t in 0..GP_TRIALS {
        let (sent, got) = run_trial_io(p, ch, point, t);
        let (bits, dead) = surviving_payload_bits(&sent, &got, bpc);
        surv_bits += bits as f64;
        if dead {
            dead_frames += 1;
        }
        wrong += sent.iter().zip(&got).filter(|(a, b)| a != b).count();
        total_cells += sent.len();
    }
    let avg_bits = surv_bits / GP_TRIALS as f64;
    let goodput = avg_bits * GP_FPS * GP_FEC_KEEP * (1.0 - GP_HEADER_SHARE) / 1000.0;
    let ser = wrong as f64 / total_cells as f64;
    let fer = dead_frames as f64 / GP_TRIALS as f64;
    (goodput, ser, fer)
}

fn cmd_goodput() {
    let base = reference_profile();
    let lumas = [1u8, 2, 3, 4];
    let chromas = [ChromaMode::Mono, ChromaMode::Chroma1, ChromaMode::Chroma2];
    let sigmas = [0.5f64, 1.0, 2.0];

    println!("# psicode-sim goodput (модель выживания полос §6.2)");
    println!(
        "goodput = сред_уцелевшие_биты/кадр · fps({GP_FPS}) · FEC_keep({GP_FEC_KEEP}) · \
         (1−header {GP_HEADER_SHARE})"
    );
    println!(
        "полос 8 (строки {:?}); полоса гибнет от ЛЮБОЙ ошибки внутри; FER = все полосы мертвы.",
        pipeline::STRIPE_ROWS
    );
    println!(
        "канал: телеметрия §7.4 (кросстолк 6/8%, шум σ2 град, matched γ), px/клетку 8, \
         {GP_TRIALS} попыток/точку."
    );
    println!("ячейка таблицы = goodput_kbit/s · сырой_SER\n");

    let t0 = Instant::now();
    let mut point = 0usize;
    let mut best_at_s2: (f64, String) = (f64::MIN, String::new());

    for &sigma in &sigmas {
        println!("## σ = {sigma}");
        println!("| luma \\ chroma | Mono | Chroma1 | Chroma2 |");
        println!("|---|---|---|---|");
        let mut best: (f64, String) = (f64::MIN, String::new());
        for &luma in &lumas {
            let mut cells = Vec::new();
            for &chroma in &chromas {
                let p = config_profile(&base, luma, chroma);
                let mut ch = ChannelParams::from_profile(&p);
                ch.px_per_cell = 8.0;
                ch.set_blur(sigma);
                let (gp, ser, _fer) = eval_goodput(&p, &ch, point);
                point += 1;
                cells.push(format!("{gp:.1} · {}", report::sig4(ser)));
                let bpc = symbol::bits_per_cell(&p);
                let label = format!("luma {luma}+{} ({bpc}b)", chroma_short(chroma));
                if gp > best.0 {
                    best = (gp, label.clone());
                }
                if (sigma - 2.0).abs() < 1e-9 && gp > best_at_s2.0 {
                    best_at_s2 = (gp, label);
                }
            }
            println!("{}", report::table_row(&format!("luma {luma}"), &cells));
        }
        println!("**argmax σ={sigma}:** {} → {:.1} kbit/s\n", best.1, best.0);
    }

    // BENCHMARKS §4 «clean channel»: на идеальном канале SER=0 у ЛЮБОГО конфига
    // -> все полосы живы -> goodput монотонен по битам/клетку, значит argmax —
    // максимум бит/клетку (luma 4 + Chroma2 = 6 бит). Считаем только его.
    let (luma_c, chroma_c) = (4u8, ChromaMode::Chroma2);
    let p_clean = config_profile(&base, luma_c, chroma_c);
    let ch_clean = ChannelParams::clean(&p_clean);
    let (gp_clean, ser_clean, _fer) = eval_goodput(&p_clean, &ch_clean, point);
    let bpc_clean = symbol::bits_per_cell(&p_clean);
    debug_assert_eq!(ser_clean, 0.0, "чистый канал обязан давать SER=0");

    println!("## для BENCHMARKS.md §4 (partial decode OFF)");
    println!(
        "- clean channel : luma {luma_c}+{} ({bpc_clean}b, SER {}) → {:.1} kbit/s",
        chroma_short(chroma_c),
        report::sig4(ser_clean),
        gp_clean
    );
    println!(
        "- blur σ = 2 px : {} → {:.1} kbit/s",
        best_at_s2.1, best_at_s2.0
    );
    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

// --- framed: L3-фрейминг сквозь канал, FER (§2/§6.3) и torn-goodput (§4/§6.3) ---

const FER_TRIALS: usize = 15;
/// Попыток на среднее байт-салваджа (чистые кадры / рваные кадры).
const CLEAN_TRIALS: usize = 15;
const TORN_TRIALS: usize = 40;
/// goodput = байты · 8 · fps(10) · FEC_keep(0.8) / 1000. Заголовок исключён
/// естественно (реальные байты l3), а не плоской скидкой 1.5% — поэтому p=0
/// сходится к ~148 kbit/s строки «clean channel» §4.
const TORN_GP_K: f64 = 8.0 * 10.0 * 0.8 / 1000.0;

/// FER по FER_TRIALS попыткам: доля потерянных кадров (страйп-заголовок мёртв
/// ИЛИ все страйпы мертвы), без разрыва.
fn mean_fer(p: &CalibProfile, ch: &ChannelParams, point: usize) -> f64 {
    let mut lost = 0usize;
    for t in 0..FER_TRIALS {
        if framed::run_framed_trial(p, ch, 0.0, point, t).frame_lost {
            lost += 1;
        }
    }
    lost as f64 / FER_TRIALS as f64
}

/// Среднее число полезных байт (off, on) по `trials` попыткам при фиксированной
/// вероятности разрыва `torn_prob` (0 = всегда чисто, 1 = всегда рвём).
fn mean_useful(
    p: &CalibProfile,
    ch: &ChannelParams,
    torn_prob: f64,
    point: usize,
    trials: usize,
) -> (f64, f64) {
    let mut off = 0usize;
    let mut on = 0usize;
    for t in 0..trials {
        let r = framed::run_framed_trial(p, ch, torn_prob, point, t);
        off += r.useful_off;
        on += r.useful_on;
    }
    (off as f64 / trials as f64, on as f64 / trials as f64)
}

fn cmd_framed() {
    let t0 = Instant::now();
    println!("# psicode-sim framed (L3 §6.2–6.3: FER + torn-frame partial decode)");

    // --- BENCHMARKS §2: FER vs px/cell (σ=1, genie, без разрыва) ---
    // тот же эталонный профиль (luma3+Chroma2, 5 бит), что строка SER §2.
    let p_ref = reference_profile();
    let base = ChannelParams::from_profile(&p_ref);
    let ppcs = [8.0, 6.0, 4.0, 3.0, 2.0, 1.5];
    println!("\n## BENCHMARKS §2 — FER vs px/cell (blur σ=1, эталон {}b/клетку)", symbol::bits_per_cell(&p_ref));
    println!("FER: кадр потерян = страйп-заголовок (0) мёртв ИЛИ все 8 страйпов мертвы.");
    println!("| source | px/cell → | 8 | 6 | 4 | 3 | 2 | 1.5 |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut cells = Vec::new();
    for (i, &ppc) in ppcs.iter().enumerate() {
        let mut ch = base.clone();
        ch.px_per_cell = ppc;
        ch.set_blur(1.0);
        cells.push(report::sig4(mean_fer(&p_ref, &ch, 500 + i)));
    }
    println!("{}", report::table_row("sim | FER", &cells));

    // --- BENCHMARKS §4: torn-frame goodput (partial decode OFF vs ON) ---
    // конфиг luma4+Chroma2 (6 бит) на ЧИСТОМ канале — изолируем разрыв и держим
    // сопоставимость с baseline-строкой «clean channel» (§4). На σ=1 этот 6-бит
    // конфиг даёт SER≈0.07 -> все страйпы гибнут (обрыв §6), поэтому демонстрация
    // выигрыша от разрыва идёт на чистом канале (blur покрыт другими строками §4).
    let mut p_gp = reference_profile();
    p_gp.luma_bits = 4;
    p_gp.chroma_mode = ChromaMode::Chroma2;
    let ch_clean = ChannelParams::clean(&p_gp);
    let bpc = symbol::bits_per_cell(&p_gp);
    println!("\n## BENCHMARKS §4 — torn-frame goodput (luma4+Chroma2 = {bpc}b, чистый канал)");
    println!(
        "goodput = сред_салвадж_байт · 8 · fps(10) · FEC_keep(0.8) / 1000. p_torn — известный\n\
         параметр, входит аналитически; Monte-Carlo усредняет только ПОЛОЖЕНИЕ разрыва\n\
         ({CLEAN_TRIALS} чистых + {TORN_TRIALS} рваных попыток)."
    );
    // средние: чистый кадр (=полный салвадж) и рваный кадр (частичный салвадж).
    let (clean_off, _clean_on) = mean_useful(&p_gp, &ch_clean, 0.0, 600, CLEAN_TRIALS);
    let (_torn_off, torn_on) = mean_useful(&p_gp, &ch_clean, 1.0, 610, TORN_TRIALS);
    println!(
        "// сред. салвадж: чистый кадр {clean_off:.0} байт, рваный кадр ON {torn_on:.0} байт (OFF 0)"
    );
    println!("| p_torn | goodput OFF | goodput ON | gain ON/OFF |");
    println!("|---|---|---|---|");
    let mut rows = Vec::new();
    for &pt in &[0.0f64, 0.20, 0.50] {
        // OFF: рваный кадр теряется целиком -> вклад только чистых кадров.
        let off = (1.0 - pt) * clean_off * TORN_GP_K;
        // ON: чистые кадры полностью + рваные частично.
        let on = ((1.0 - pt) * clean_off + pt * torn_on) * TORN_GP_K;
        let gain = if off > 0.0 { (on / off - 1.0) * 100.0 } else { 0.0 };
        println!(
            "| {:.0}% | {:.1} | {:.1} | {:+.1}% |",
            pt * 100.0,
            off,
            on,
            gain
        );
        rows.push((pt, off, on, gain));
    }

    // сводка для вердикта против §6.3 «+20–30%».
    println!("\n### verdict vs SPEC §6.3 «+20–30% goodput под тяжёлым разрывом»");
    for &(pt, off, on, gain) in &rows {
        if pt > 0.0 {
            println!("- p_torn {:.0}%: ON/OFF = {:+.1}% ({:.1} -> {:.1} kbit/s)", pt * 100.0, gain, off, on);
        }
    }

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("sweep") => cmd_sweep(),
        Some("framed") => cmd_framed(),
        Some("l3live") => l3live::cmd_l3live(),
        Some("modeb") => modeb::cmd_modeb(),
        Some("transfer") => transfer::cmd_transfer(),
        Some("live") => {
            let path = args.get(2).map(String::as_str).unwrap_or_else(|| {
                eprintln!("usage: psicode-sim live <file.ppm> [cell]");
                std::process::exit(2);
            });
            let cell = args.get(3).and_then(|s| s.parse::<u8>().ok());
            live::cmd_live(path, cell);
        }
        Some("dump") => {
            let dir = args.get(2).map(String::as_str).unwrap_or(DUMP_DIR_DEFAULT);
            cmd_dump(Path::new(dir));
        }
        Some("readback") => {
            let dir = args.get(2).map(String::as_str).unwrap_or(DUMP_DIR_DEFAULT);
            cmd_readback(Path::new(dir));
        }
        Some("goodput") => cmd_goodput(),
        Some("exp") => exp::cmd_exp(),
        Some("gprobe") => gprobe::cmd_gprobe(),
        Some("probe") => probe::cmd_probe(),
        Some("tprobe") => tprobe::cmd_tprobe(),
        Some("finder") => finder::cmd_finder(),
        _ => {
            eprintln!(
                "usage: psicode-sim <sweep | dump [dir] | readback [dir] | goodput | framed | l3live | modeb | transfer | exp | probe | tprobe>"
            );
            std::process::exit(2);
        }
    }
}
