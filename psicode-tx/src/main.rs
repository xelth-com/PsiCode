//! psicode-tx — передатчик PsiCode для Windows 11 (SPEC §8).
//!
//! Тонкая оболочка winit/softbuffer над ЧИСТЫМ контентом кадров из [`frames`].
//! GUI занимается только окном, презентацией, таймингом и вводом; весь контент
//! кадров (паттерн §4, потоковые кадры §6, счётчики §3.3/§4.5) строится в
//! [`frames`] отдельными юнит-тестируемыми функциями без окна.
//!
//! # Режимы (CLI, без clap — только std::env::args)
//!
//! ```text
//! psicode-tx calibrate [--cell N] [--smoke]
//!     Калибровочный паттерн §4 (фаза 1). SPACE -> фаза 2 (§4.5, анимированный
//!     счётчик кадров сверху/снизу для оценки rolling-shutter). C или закрытие
//!     окна -> консольный запрос кода профиля с телефона.
//!
//! psicode-tx stream <file> [--code C] [--cell N] [--chroma] [--calib] [--smoke]
//!     Стриминг файла кадрами L3 (§6.1/§6.2). Профиль: --code, иначе
//!     psicode-tx.profile из cwd, иначе эталон §6/§7.4. Цикл бесконечен, Esc — выход.
//!
//! psicode-tx single [--counter N] [--cell N] [--chroma] [--smoke]
//!     Один статический кадр с детерминированной псевдослучайной нагрузкой —
//!     навести камеру, проверить детекцию ЗЧ на живом экране.
//!
//! psicode-tx swatch [--n 1..16] [--sweep] [--cell N] [--smoke]
//!     [ДИАГНОСТИКА] Лестница масштабов §5.1-CL (см. psicode_core::swatch):
//!     штатная геометрия символа, но payload заменён на n×n КРУПНЫХ блоков
//!     известной цветности. Кадр анимирован (счётчик §3.3 несёт его номер),
//!     поэтому анализатор восстанавливает истину из ОДНОГО снимка.
//! ```
//!
//! `--cell N` подменяет размер клетки для ПОКАЗА (см. `frames::Streamer::render`:
//! `cell_size_px` профиля — то, что предполагает ПРИЁМНИК; на экране мы рисуем
//! 1 display-px на Frame-px, умноженный на целочисленный зум окна).
//!
//! `--chroma` переводит профиль в семейство ПОСТОЯННОЙ ЯРКОСТИ (§5.1-CL,
//! `ChromaMode::ConstLuma1` + 1 бит действительной оси = 2 бит/клетку):
//! комплексный символ живёт целиком в ЦВЕТНОСТИ при постоянной сумме R+G+B, и
//! поле освещённости приёмника сокращается в поклеточном отношении каналов.
//! Флаг применяется ПОВЕРХ разрешённого профиля (--code / файл / эталон), то
//! есть меняет только отображение цвета, не трогая геометрию и тайминг.
//!
//! `--smoke` автозакрывает окно после ~20 кадров (для CI и быстрой проверки).
//!
//! # Тайминг (§6.3) — предупреждение
//!
//! Удержание кадра = `frame_hold_periods × период_обновления_монитора`. Частота
//! берётся из winit (`refresh_rate_millihertz`), иначе 60 Гц. Пейсинг — по
//! `Instant` через `ControlFlow::WaitUntil`; это НЕ синхронизировано с настоящим
//! vsync, поэтому фактическое удержание дрожит на ±1 период. Точность пейсинга
//! обязана быть перепроверена на живом канале (см. отчёт).

mod frames;

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::{Duration, Instant};

use frames::{
    calibration_counter_frame, calibration_frame, chromatic_profile, reference_profile,
    single_frame, swatch_frame, to_const_luma, RgbFrame, Streamer, SYMBOLS_PER_FRAME,
};
use psicode_core::swatch;
use psicode_core::CalibProfile;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Смоук-режим: автозакрытие после стольких кадров.
const SMOKE_FRAMES: u64 = 20;
/// Файл нормализованного кода профиля рядом с exe/cwd.
const PROFILE_FILE: &str = "psicode-tx.profile";
/// Средне-серый фон окна (§3.1), softbuffer-формат 0x00RRGGBB.
const BG: u32 = 0x0080_8080;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match parse_args(&args) {
        Ok(cmd) => run(cmd),
        Err(msg) => {
            eprintln!("{msg}\n");
            print_usage();
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Разобранная команда CLI.
enum Command {
    Calibrate { cell: usize, smoke: bool },
    Stream { file: String, code: Option<String>, cell: Option<usize>, chroma: bool, calib: bool, smoke: bool },
    Single { counter: u8, cell: Option<usize>, chroma: bool, smoke: bool },
    Swatch { n: usize, sweep: bool, cell: Option<usize>, smoke: bool },
}

fn print_usage() {
    eprintln!(
        "psicode-tx — передатчик PsiCode (Windows)\n\
         \n\
         Использование:\n\
         \x20 psicode-tx calibrate [--cell N] [--smoke]\n\
         \x20 psicode-tx stream <file> [--code C] [--cell N] [--chroma] [--calib] [--smoke]\n\
         \x20 psicode-tx single [--counter N] [--cell N] [--chroma] [--smoke]\n\
         \x20 psicode-tx swatch [--n 1..16] [--sweep] [--cell N] [--smoke]\n\
         \n\
         \x20 swatch:   [ДИАГНОСТИКА] лестница масштабов — n×n крупных однородных\n\
         \x20           блоков вместо payload; --sweep даёт плотную развёртку всей\n\
         \x20           плоскости цветности за 256 кадров.\n\
         \x20 --chroma: отображение ПОСТОЯННОЙ ЯРКОСТИ §5.1-CL (ConstLuma1,\n\
         \x20           2 бит/клетку) — символ целиком в цветности, R+G+B ≡ const.\n\
         \x20 --calib:  вплетать ВНУТРИПОЛОСНЫЕ калибровочные кадры §4-IB по\n\
         \x20           расписанию удвоения 0,1,2,4,...,128 и далее каждые 128\n\
         \x20           (0.78 % в установившемся режиме). Приёмник снимает с них\n\
         \x20           матрицу 3×3, гамму и σ блюра вместо односкеточной\n\
         \x20           референсной строки §3.4.\n"
    );
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    let mode = args.get(1).ok_or_else(|| "не указан режим".to_string())?;

    // общий разбор флагов начиная с индекса `start`
    let mut cell: Option<usize> = None;
    let mut code: Option<String> = None;
    let mut counter: u8 = 0;
    let mut chroma = false;
    let mut calib = false;
    let mut smoke = false;
    let mut n = 1usize;
    let mut sweep = false;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--cell" => {
                let v = args.get(i + 1).ok_or("--cell требует число")?;
                cell = Some(v.parse().map_err(|_| format!("--cell: не число: {v}"))?);
                i += 2;
            }
            "--code" => {
                let v = args.get(i + 1).ok_or("--code требует строку")?;
                code = Some(v.clone());
                i += 2;
            }
            "--counter" => {
                let v = args.get(i + 1).ok_or("--counter требует число")?;
                let cv: u64 = v.parse().map_err(|_| format!("--counter: не число: {v}"))?;
                counter = (cv & 0xFF) as u8;
                i += 2;
            }
            "--n" => {
                let v = args.get(i + 1).ok_or("--n требует число")?;
                let k: usize = v.parse().map_err(|_| format!("--n: не число: {v}"))?;
                if !(1..=swatch::SWATCH_MAX_N).contains(&k) {
                    return Err(format!(
                        "--n: ожидалось 1..={}, дано {k}",
                        swatch::SWATCH_MAX_N
                    ));
                }
                n = k;
                i += 2;
            }
            "--sweep" => {
                sweep = true;
                i += 1;
            }
            "--chroma" => {
                chroma = true;
                i += 1;
            }
            "--calib" => {
                calib = true;
                i += 1;
            }
            "--smoke" => {
                smoke = true;
                i += 1;
            }
            other if other.starts_with("--") => {
                return Err(format!("неизвестный флаг: {other}"));
            }
            _ => {
                positional.push(args[i].clone());
                i += 1;
            }
        }
    }

    match mode.as_str() {
        "calibrate" => Ok(Command::Calibrate {
            cell: cell.unwrap_or(12),
            smoke,
        }),
        "stream" => {
            let file = positional
                .into_iter()
                .next()
                .ok_or("stream требует <file>")?;
            Ok(Command::Stream { file, code, cell, chroma, calib, smoke })
        }
        "single" => Ok(Command::Single { counter, cell, chroma, smoke }),
        "swatch" => Ok(Command::Swatch { n, sweep, cell, smoke }),
        other => Err(format!("неизвестный режим: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Диспетчер режимов
// ---------------------------------------------------------------------------

fn run(cmd: Command) {
    match cmd {
        Command::Calibrate { cell, smoke } => run_calibrate(cell, smoke),
        Command::Stream { file, code, cell, chroma, calib, smoke } => {
            run_stream(file, code, cell, chroma, calib, smoke)
        }
        Command::Single { counter, cell, chroma, smoke } => {
            run_single(counter, cell, chroma, smoke)
        }
        Command::Swatch { n, sweep, cell, smoke } => run_swatch(n, sweep, cell, smoke),
    }
}

fn run_calibrate(cell: usize, smoke: bool) {
    let cell = cell.max(1);
    println!("psicode-tx calibrate — паттерн §4, cell={cell} px. SPACE: фаза 2 (§4.5). C/закрыть: ввод кода.");
    // Окно вмещает фазу 2 (паттерн + две полосы счётчика по 2 клетки).
    let band_h = 2 * cell;
    let w = frames_grid_side(cell);
    let h = w + 2 * band_h;
    let source = calibration_frame(cell);
    let mode = ModeState::Calibrate { phase: 1, frame_no: 0 };
    let mut app = App::new(mode, source, cell, cell, w, h, smoke, !smoke, "PsiCode TX — calibrate");
    launch(&mut app);
    // Консольный обратный канал (§4/§7): после закрытия окна — ввод кода.
    if app.console_after && !app.window_failed {
        prompt_profile_code();
    }
    if app.window_failed {
        report_no_window();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_stream(
    file: String,
    code: Option<String>,
    cell: Option<usize>,
    chroma: bool,
    calib: bool,
    smoke: bool,
) {
    let bytes = match std::fs::read(&file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Не удалось прочитать файл {file}: {e}");
            std::process::exit(1);
        }
    };
    if bytes.is_empty() {
        eprintln!("Файл {file} пуст — нечего передавать.");
        std::process::exit(1);
    }

    let profile = apply_chroma(resolve_profile(code.as_deref()), chroma);
    let session_id = make_session_id();
    let streamer = Streamer::new(&bytes, profile, session_id).with_calibration(calib);
    let display_cell = cell.unwrap_or(profile.cell_size_px as usize).max(1);

    // Печать «на старте» (§8): K, symbol_size, кадров/проход, оценка goodput.
    println!("psicode-tx stream — файл {file} ({} байт)", bytes.len());
    println!(
        "профиль: cell {} px, hold {} период(ов), luma {} + chroma {} бит = {} бит/клетку, quiet {}, fec {}",
        profile.cell_size_px,
        profile.frame_hold_periods,
        profile.luma_bits,
        profile.chroma_bits(),
        streamer.bpc(),
        profile.quiet_zone_cells(),
        profile.fec_overhead,
    );
    println!(
        "session_id 0x{:08X}; K={} source-символов; symbol_size={} байт; {} символов/кадр; кадров/проход={}",
        session_id,
        streamer.k(),
        streamer.symbol_size(),
        SYMBOLS_PER_FRAME,
        streamer.frames_per_pass(),
    );
    if streamer.calibration_enabled() {
        // §4-IB.3: расписание удвоения. Печатаем ФАКТИЧЕСКИЕ накладные расходы
        // на длине этой передачи, а не только асимптоту.
        use psicode_core::calframe;
        let n = (streamer.frames_per_pass() as u64).max(1);
        println!(
            "калибровочные кадры §4-IB: ВКЛ (0,1,2,4,...,128 далее каждые 128); накладные расходы {:.2} % на первых {n} кадрах (расписание front-loaded: 8 кадров до 128-го), {:.2} % в установившемся режиме; худшее ожидание позднего приёмника {} кадров",
            100.0 * streamer.calibration_overhead(n),
            100.0 / calframe::CALIB_PERIOD as f64,
            calframe::worst_join_wait(4 * calframe::CALIB_PERIOD),
        );
    }
    // goodput при 60 Гц как baseline (уточняется по факту в resumed()).
    let est60 = goodput_kbps(bytes.len(), streamer.frames_per_pass(), profile.frame_hold_periods, 60.0);
    println!("оценка goodput @60 Гц: {est60:.1} kbit/s (уточняется по частоте монитора); Esc — выход.");

    let side = (61 + 2 * profile.quiet_zone_cells() as usize) * display_cell;
    let source = streamer.render(0, Some(display_cell));
    let mode = ModeState::Stream {
        streamer,
        seq: 0,
        file_len: bytes.len(),
        hold_periods: profile.frame_hold_periods,
    };
    let mut app = App::new(mode, source, display_cell, display_cell, side, side, smoke, false, "PsiCode TX — stream");
    launch(&mut app);
    if app.window_failed {
        report_no_window();
    }
}

fn run_single(counter: u8, cell: Option<usize>, chroma: bool, smoke: bool) {
    let profile = if chroma {
        announce_const_luma(chromatic_profile())
    } else {
        reference_profile()
    };
    let display_cell = cell.unwrap_or(profile.cell_size_px as usize).max(1);
    println!("psicode-tx single — статический кадр, counter={counter}, cell={display_cell} px. Esc — выход.");
    let source = single_frame(&profile, counter, Some(display_cell));
    let side = (61 + 2 * profile.quiet_zone_cells() as usize) * display_cell;
    let mut app = App::new(ModeState::Single, source, display_cell, display_cell, side, side, smoke, false, "PsiCode TX — single");
    launch(&mut app);
    if app.window_failed {
        report_no_window();
    }
}

/// [ДИАГНОСТИКА] Лестница масштабов §5.1-CL (`psicode_core::swatch`).
///
/// Геометрия символа — ШТАТНАЯ (детекция §3.2 и калибровка §3.4 приёмника
/// работают без правок), payload заменён на `n×n` крупных однородных блоков
/// известной цветности, разделённых средне-серым рвом. Кадр анимирован с тем же
/// удержанием, что и stream, а строка счётчика §3.3 несёт его номер — поэтому
/// анализатору хватает ОДНОГО снимка, чтобы точно знать, что было послано.
fn run_swatch(n: usize, sweep: bool, cell: Option<usize>, smoke: bool) {
    let profile = chromatic_profile();
    let display_cell = cell.unwrap_or(profile.cell_size_px as usize).max(1);
    let nb = n * n;
    println!(
        "psicode-tx swatch — {n}×{n} = {nb} блок(ов), режим {}, cell {display_cell} px, \
         удержание {} период(ов). Esc — выход.",
        if sweep {
            "РАЗВЁРТКА плоскости цветности (256 кадров на полный проход)"
        } else {
            "углы созвездия ConstLuma1 (алфавит payload)"
        },
        profile.frame_hold_periods,
    );
    // размер блока в клетках/пикселях — главное число лестницы
    let (pc0, pr0, bc, br) = swatch::swatch_rect(n, 0, swatch::SWATCH_GUARD);
    // фактический ров (на мелких ступенях он сжимается — см. swatch_rect)
    let gap = 2 * pc0.min(pr0);
    println!(
        "блок ≈ {bc}×{br} клеток = {}×{} px; между блоками {gap} клеток ({} px)",
        bc * display_cell,
        br * display_cell,
        gap * display_cell,
    );
    let source = swatch_frame(&profile, n, 0, sweep, Some(display_cell));
    let side = (61 + 2 * profile.quiet_zone_cells() as usize) * display_cell;
    let mode = ModeState::Swatch {
        profile,
        frame: 0,
        n,
        sweep,
        hold_periods: profile.frame_hold_periods,
    };
    let mut app = App::new(
        mode,
        source,
        display_cell,
        display_cell,
        side,
        side,
        smoke,
        false,
        "PsiCode TX — swatch",
    );
    launch(&mut app);
    if app.window_failed {
        report_no_window();
    }
}

/// Сторона квадрата калибровочного паттерна в px: 61·cell.
fn frames_grid_side(cell: usize) -> usize {
    61 * cell
}

/// `--chroma`: перевести профиль в режим постоянной яркости §5.1-CL. Флаг
/// применяется ПОВЕРХ разрешённого профиля, поэтому меняет только отображение
/// цвета (и bits_per_cell), не трогая геометрию/тайминг/FEC.
fn apply_chroma(p: CalibProfile, chroma: bool) -> CalibProfile {
    if chroma {
        announce_const_luma(to_const_luma(p))
    } else {
        p
    }
}

/// Печать того, что именно сделал `--chroma` (профиль возвращается как есть).
fn announce_const_luma(p: CalibProfile) -> CalibProfile {
    println!(
        "--chroma: отображение §5.1-CL {:?} — Re на оси 2G−R−B ({} бит), Im на оси R−B ({} бит), \
         R+G+B ≡ const, всего {} бит/клетку",
        p.chroma_mode,
        p.luma_bits,
        p.chroma_bits(),
        p.luma_bits as u32 + p.chroma_bits() as u32,
    );
    p
}

/// Профиль для стриминга: --code -> файл psicode-tx.profile -> эталон §6/§7.4.
fn resolve_profile(code: Option<&str>) -> CalibProfile {
    if let Some(c) = code {
        match CalibProfile::decode_string(c) {
            Ok((p, fixed)) => {
                if fixed > 0 {
                    println!("--code: исправлено опечаток RS: {fixed}");
                }
                return p;
            }
            Err(e) => {
                eprintln!("--code недействителен: {e}");
                std::process::exit(1);
            }
        }
    }
    if let Ok(text) = std::fs::read_to_string(PROFILE_FILE) {
        let line = text.lines().next().unwrap_or("").trim();
        if !line.is_empty() {
            match CalibProfile::decode_string(line) {
                Ok((p, fixed)) => {
                    println!("профиль из {PROFILE_FILE} (исправлено опечаток: {fixed})");
                    return p;
                }
                Err(e) => {
                    eprintln!("{PROFILE_FILE} нечитаем ({e}) — беру эталонный профиль §6/§7.4");
                }
            }
        }
    }
    println!("профиль: эталонный §6/§7.4 (luma 2 + Chroma1, cell 16, hold 6, fec 2)");
    reference_profile()
}

/// Оценка goodput одного прохода (без потерь): file_bits / (кадров·hold_сек).
fn goodput_kbps(file_len: usize, frames_per_pass: usize, hold_periods: u8, refresh_hz: f64) -> f64 {
    let hold_s = hold_periods as f64 / refresh_hz.max(1.0);
    let total_s = frames_per_pass as f64 * hold_s;
    if total_s <= 0.0 {
        0.0
    } else {
        (file_len as f64 * 8.0) / total_s / 1000.0
    }
}

/// session_id (§6.2): смесь наносекунд SystemTime и PID через splitmix64-финализатор.
/// Случаен на запуск, постоянен внутри передачи.
fn make_session_id() -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let mut z = nanos ^ pid.rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

/// Консольный ввод кода профиля (§4/§7): терпимый парсер, печать прескрипций и
/// числа исправленных опечаток, сохранение НОРМАЛИЗОВАННОГО кода в файл.
fn prompt_profile_code() {
    use std::io::Write;
    loop {
        print!("Введите код профиля с телефона: ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        let n = std::io::stdin().read_line(&mut line).unwrap_or(0);
        if n == 0 {
            println!("\n(конец ввода) — пропуск.");
            return;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            println!("Пустая строка — пропуск.");
            return;
        }
        match CalibProfile::decode_string(trimmed) {
            Ok((p, fixed)) => {
                print_prescriptions(&p, fixed);
                match p.encode_string() {
                    Ok(norm) => match std::fs::write(PROFILE_FILE, format!("{norm}\n")) {
                        Ok(()) => println!("Нормализованный код сохранён в {PROFILE_FILE}: {norm}"),
                        Err(e) => eprintln!("Не удалось записать {PROFILE_FILE}: {e}"),
                    },
                    Err(e) => eprintln!("Профиль не кодируется обратно: {e}"),
                }
                return;
            }
            Err(e) => {
                eprintln!("Неверный код: {e}. Повторите (пустая строка — пропуск).");
            }
        }
    }
}

/// Печать декодированных ПРЕСКРИПЦИЙ профиля (§7.3 поля 2–5, 16–17) + опечатки.
fn print_prescriptions(p: &CalibProfile, fixed: usize) {
    println!("Код принят. Исправлено опечаток RS: {fixed}.");
    println!("Прескрипции профиля (то, что задаёт поведение передатчика):");
    println!("  cell_size_px       = {} px", p.cell_size_px);
    println!("  frame_hold_periods = {} период(ов) обновления", p.frame_hold_periods);
    println!("  luma_bits          = {}", p.luma_bits);
    println!("  chroma_mode        = {:?} ({} бит хромы)", p.chroma_mode, p.chroma_bits());
    println!("  quiet_zone         = {} ({} клеток)", p.quiet_zone, p.quiet_zone_cells());
    println!("  fec_overhead       = {}", p.fec_overhead);
    println!(
        "  -> {} бит/клетку, эффективный FPS @60 Гц = {:.1}",
        p.luma_bits as u32 + p.chroma_bits() as u32,
        p.effective_fps()
    );
}

fn report_no_window() {
    eprintln!(
        "Окно открыть не удалось (нет доступного дисплея/сессии). Контент кадров \
         проверяется юнит-тестами frames.rs без окна: `cargo test -p psicode-tx`."
    );
}

// ---------------------------------------------------------------------------
// Оболочка окна (winit 0.30 + softbuffer 0.4)
// ---------------------------------------------------------------------------

/// Состояние конкретного режима для логики продвижения кадров (advance).
enum ModeState {
    Calibrate {
        phase: u8,
        frame_no: u64,
    },
    Stream {
        streamer: Streamer,
        seq: u64,
        file_len: usize,
        hold_periods: u8,
    },
    Single,
    /// [ДИАГНОСТИКА] Лестница масштабов: анимированный swatch-паттерн.
    Swatch {
        profile: CalibProfile,
        frame: u64,
        n: usize,
        sweep: bool,
        hold_periods: u8,
    },
}

struct App {
    mode: ModeState,
    /// Текущий кадр для блита (обновляется в advance/rebuild).
    source: RgbFrame,
    /// Размер клетки калибровки (для фаз 1/2).
    cell: usize,
    /// Размер клетки показа для стрима/single.
    display_cell: usize,
    /// Желаемый начальный размер окна.
    init_w: usize,
    init_h: usize,
    smoke: bool,
    /// Для calibrate: после закрытия окна запросить код профиля в консоли.
    console_after: bool,
    title: &'static str,

    window: Option<Rc<Window>>,
    surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
    /// Момент следующего продвижения кадра (пейсинг §6.3).
    next_frame: Instant,
    /// Период между продвижениями (hold для стрима, обновление для остального).
    period: Duration,
    /// Частота монитора, Гц (для печати и периода).
    refresh_hz: f64,
    frames_drawn: u64,
    printed_refresh: bool,
    window_failed: bool,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mode: ModeState,
        source: RgbFrame,
        cell: usize,
        display_cell: usize,
        init_w: usize,
        init_h: usize,
        smoke: bool,
        console_after: bool,
        title: &'static str,
    ) -> Self {
        App {
            mode,
            source,
            cell,
            display_cell,
            init_w,
            init_h,
            smoke,
            console_after,
            title,
            window: None,
            surface: None,
            next_frame: Instant::now(),
            period: Duration::from_secs_f64(1.0 / 60.0),
            refresh_hz: 60.0,
            frames_drawn: 0,
            printed_refresh: false,
            window_failed: false,
        }
    }

    /// Продвинуть состояние на один тик и перестроить `source`. Для статических
    /// режимов (single, фаза 1) контент не меняется — только счётчик тиков.
    fn advance(&mut self) {
        // Отдельные поля берём напрямую, чтобы не конфликтовать с &mut self.mode.
        let cell = self.cell;
        let display_cell = self.display_cell;
        let new_source: Option<RgbFrame> = match &mut self.mode {
            ModeState::Calibrate { phase, frame_no } => {
                if *phase == 2 {
                    *frame_no = frame_no.wrapping_add(1);
                    Some(calibration_counter_frame(cell, *frame_no))
                } else {
                    None // фаза 1 статична
                }
            }
            ModeState::Stream { streamer, seq, .. } => {
                *seq = seq.wrapping_add(1);
                Some(streamer.render(*seq, Some(display_cell)))
            }
            ModeState::Single => None,
            ModeState::Swatch { profile, frame, n, sweep, .. } => {
                *frame = frame.wrapping_add(1);
                Some(swatch_frame(profile, *n, *frame, *sweep, Some(display_cell)))
            }
        };
        if let Some(s) = new_source {
            self.source = s;
        }
    }

    /// Перестроить `source` из текущего состояния БЕЗ продвижения (после SPACE).
    fn rebuild_source(&mut self) {
        let cell = self.cell;
        if let ModeState::Calibrate { phase, frame_no } = &self.mode {
            self.source = if *phase == 2 {
                calibration_counter_frame(cell, *frame_no)
            } else {
                calibration_frame(cell)
            };
        }
    }

    /// Блит `source` в буфер окна: фон средне-серый, целочисленный зум, по центру.
    fn present(&mut self) {
        let (window, surface) = match (&self.window, &mut self.surface) {
            (Some(w), Some(s)) => (w, s),
            _ => return,
        };
        let size = window.inner_size();
        let (win_w, win_h) = (size.width as usize, size.height as usize);
        let (nw, nh) = match (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
            (Some(w), Some(h)) => (w, h),
            _ => return, // минимизировано
        };
        if surface.resize(nw, nh).is_err() {
            return;
        }
        let mut buffer = match surface.buffer_mut() {
            Ok(b) => b,
            Err(_) => return,
        };
        blit(&mut buffer, win_w, win_h, &self.source);
        let _ = buffer.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // повторный resume — окно уже есть
        }
        let attrs = Window::default_attributes()
            .with_title(self.title)
            .with_inner_size(PhysicalSize::new(self.init_w as u32, self.init_h as u32));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Rc::new(w),
            Err(e) => {
                eprintln!("create_window: {e}");
                self.window_failed = true;
                event_loop.exit();
                return;
            }
        };
        let context = match softbuffer::Context::new(window.clone()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("softbuffer::Context: {e}");
                self.window_failed = true;
                event_loop.exit();
                return;
            }
        };
        let surface = match softbuffer::Surface::new(&context, window.clone()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("softbuffer::Surface: {e}");
                self.window_failed = true;
                event_loop.exit();
                return;
            }
        };

        // Частота монитора -> период пейсинга (§6.3).
        self.refresh_hz = window
            .current_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
            .map(|mhz| mhz as f64 / 1000.0)
            .filter(|hz| *hz > 1.0)
            .unwrap_or(60.0);
        let refresh_period = Duration::from_secs_f64(1.0 / self.refresh_hz);
        self.period = match &self.mode {
            ModeState::Stream { hold_periods, .. }
            | ModeState::Swatch { hold_periods, .. } => refresh_period * (*hold_periods as u32),
            _ => refresh_period, // калибровочный счётчик/статика — темп обновления
        };
        self.next_frame = Instant::now() + self.period;

        if !self.printed_refresh {
            self.printed_refresh = true;
            self.print_refresh_info();
        }

        self.window = Some(window.clone());
        self.surface = Some(surface);
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.present();
                self.frames_drawn += 1;
                if self.smoke && self.frames_drawn >= SMOKE_FRAMES {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match code {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::Space => {
                    if let ModeState::Calibrate { phase, .. } = &mut self.mode {
                        *phase = if *phase == 1 { 2 } else { 1 };
                        let ph = *phase;
                        self.rebuild_source();
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                        println!("calibrate: фаза {ph}");
                    }
                }
                KeyCode::KeyC => {
                    if matches!(self.mode, ModeState::Calibrate { .. }) {
                        event_loop.exit();
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            return;
        }
        let now = Instant::now();
        if now >= self.next_frame {
            self.advance();
            // Ставим следующий дедлайн от now (без «навёрстывания» пропущенных
            // тиков — иначе после залипания система устроит шторм кадров).
            self.next_frame = now + self.period;
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}

impl App {
    /// Печать фактической частоты/удержания/goodput после определения монитора.
    fn print_refresh_info(&self) {
        if let ModeState::Stream { file_len, hold_periods, streamer, .. } = &self.mode {
            let hold_ms = 1000.0 * *hold_periods as f64 / self.refresh_hz;
            let gp = goodput_kbps(*file_len, streamer.frames_per_pass(), *hold_periods, self.refresh_hz);
            println!(
                "монитор {:.2} Гц; удержание кадра {:.1} мс ({} период(ов)); goodput ≈ {:.1} kbit/s",
                self.refresh_hz, hold_ms, hold_periods, gp
            );
        } else {
            println!("монитор {:.2} Гц", self.refresh_hz);
        }
    }
}

/// Запуск цикла событий над `app`. Блокирует до выхода.
fn launch(app: &mut App) {
    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            eprintln!("EventLoop::new: {e}");
            app.window_failed = true;
            return;
        }
    };
    event_loop.set_control_flow(ControlFlow::Wait);
    if let Err(e) = event_loop.run_app(app) {
        eprintln!("run_app: {e}");
        app.window_failed = true;
    }
}

/// Целочисленный зум, центрирование, ближайший сосед, средне-серый леттербокс.
/// `buf` — softbuffer 0x00RRGGBB длиной `win_w·win_h`.
fn blit(buf: &mut [u32], win_w: usize, win_h: usize, src: &RgbFrame) {
    for p in buf.iter_mut() {
        *p = BG;
    }
    if src.w == 0 || src.h == 0 || win_w == 0 || win_h == 0 {
        return;
    }
    let zoom = (win_w / src.w).min(win_h / src.h).max(1);
    let draw_w = (src.w * zoom).min(win_w);
    let draw_h = (src.h * zoom).min(win_h);
    let off_x = (win_w - draw_w) / 2;
    let off_y = (win_h - draw_h) / 2;
    for dy in 0..draw_h {
        let sy = dy / zoom;
        if sy >= src.h {
            break;
        }
        let row_src = sy * src.w;
        let row_dst = (off_y + dy) * win_w + off_x;
        for dx in 0..draw_w {
            let sx = dx / zoom;
            if sx >= src.w {
                break;
            }
            let c = src.px[row_src + sx];
            buf[row_dst + dx] = ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32;
        }
    }
}
