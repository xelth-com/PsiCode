//! Приёмная сессия (§8): поток кадров камеры -> состояние Searching/Locked/Done.
//!
//! На каждый кадр: YUV -> плоскость Y для ДЕТЕКЦИИ ЗЧ-рамки (§3.2) -> гомография
//! -> демодуляция RGB c пер-канальной самокалибровкой тона (§3.4, core::tone) ->
//! разбор L3 (§6.2) -> фильтр session_id -> XOR-фонтан (§9.2) -> CRC-32C. Зеркалит
//! приёмную сторону psicode-sim/src/transfer.rs (порядок ESI, атрибуция рваного
//! снимка §6.3), но ест кадры камеры вместо синтетических.
//!
//! Чистый Rust, десктоп-тестируемый: JNI (jni_glue) — тонкая обёртка над этим.
//!
//! # Производительность (§8)
//!
//! Детекция дорога и масштабируется с числом пикселей (связные компоненты карты
//! активности, выравнивание кольца). Бюджет ~200–300 мс/кадр на среднем телефоне
//! при 1920×1080. Поэтому детекция идёт по УМЕНЬШЕННОЙ плоскости Y (сторона <=
//! [`DETECT_MAX_DIM`]), а демодуляция — по ПОЛНОМУ разрешению через гомографию,
//! отмасштабированную обратно. Клеточная точность выравнивания масштабно-
//! инвариантна, поэтому уменьшение почти не влияет на геометрию демода, зато
//! ограничивает и дорогой фолбэк детекции (`detect_symbol` — чужой модуль, его
//! фолбэк отдельно не вызвать, поэтому бюджет держим через входное разрешение).

use std::collections::HashMap;

use psicode_core::calframe::{self, CalibCache};
use psicode_core::detect::{detect_symbol_acquire, frame_map, track_symbol, Detection};
use psicode_core::fountain::FountainDecoder;
use psicode_core::l3::{self, FrameHeader, ParsedFrame};
use psicode_core::symbol::{
    self, demod_symbol, demod_symbol_local, demod_symbol_matrix, read_counters,
};
use psicode_core::tone;
use psicode_core::{CalibProfile, ChromaMode};

use crate::yuv::YuvFrame;

/// Отладка приёмного тракта (переменная окружения PSICODE_RX_DEBUG).
fn rx_dbg() -> bool {
    std::env::var_os("PSICODE_RX_DEBUG").is_some()
}

/// Размер encoding-символа как в psicode-tx: (ёмкость кадра − заголовок 12 −
/// TransferInfo 14) / 8. Зависит от bits_per_cell профиля (3bpc -> 141,
/// 1bpc -> 42). ВНИМАНИЕ: psicode-sim/transfer.rs исторически использует 140 —
/// сим самосогласован, а живой тракт обязан совпадать с psicode-tx.
fn symbol_size_for(bpc: u32) -> usize {
    let cap: usize = l3::STRIPE_ROWS
        .iter()
        .map(|&r| (r * l3::PAYLOAD_COLS * bpc as usize - 16) / 8)
        .sum();
    (cap - l3::FRAME_HEADER_LEN - l3::TRANSFER_INFO_LEN) / SYMBOLS_PER_FRAME
}
/// Число символов в кадре (фиксировано — упрощает разметку ESI и §6.3).
const SYMBOLS_PER_FRAME: usize = 8;
/// Вплетать repair после каждых 4 source (§6.1, fec_overhead=2).
const REPAIR_EVERY: u32 = 4;

/// Максимальная сторона плоскости Y для детекции. Стоимость align'а (и захвата,
/// и трекинга) НЕ зависит от размера кадра (фикс. число точек кольца через
/// гомографию) — уменьшение НЕ экономит время align'а, зато режет px/клетку и
/// портит лок (на 640px реальный снимок доходил лишь до corr 0.43 < SCORE_MIN).
/// Поэтому детектим по сути в полном разрешении (кап только для огромных фото,
/// чтобы ограничить единственную размер-зависимую часть — coarse_candidates).
const DETECT_MAX_DIM: usize = 4096;

/// Профиль tx по умолчанию (§8): σ=1-оптимум BENCHMARKS §6 — luma 2 + Chroma1 =
/// 3 бит/клетку. Идентичен transfer.rs/live.rs. `cell_size_px` — display-масштаб;
/// в демодуляции он СОКРАЩАЕТСЯ (frame_map переводит px->клетки, demod клетки->px
/// тем же cell), важно лишь cell >= 8 (порог 2×2-субсэмпла клетки, §5.2).
/// Профиль ПОСТОЯННОЙ ЯРКОСТИ (§5.1-CL) — зеркало `psicode-tx::frames::chromatic_profile`.
/// 2 бит/клетку: 1 бит на ось `2G−R−B`, 1 бит на ось `R−B`. Обращение делит на
/// измеренную поклеточную сумму каналов, поэтому поле освещённости приёмника
/// сокращается внутри клетки и не требует локального порога.
pub fn tx_chromatic_profile() -> CalibProfile {
    let mut p = tx_default_profile();
    p.luma_bits = 1;
    p.chroma_mode = ChromaMode::ConstLuma1;
    p
}

pub fn tx_default_profile() -> CalibProfile {
    CalibProfile {
        version: CalibProfile::VERSION,
        cell_size_px: 16,
        frame_hold_periods: 6,
        // v0 live: 1 бит/клетку — живой канал телефона даёт эффективный блюр
        // ~σ2, где BENCHMARKS §6 предписывает luma1+Mono (SER 0 в симе).
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
    }
}

/// Состояние приёмной сессии (§8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RxState {
    /// Ещё не поймали ни одной сессии передачи (нет валидного заголовка).
    Searching,
    /// Сессия захвачена (декодер инициализирован по TransferInfo), идёт набор.
    Locked,
    /// Объект собран и подтверждён CRC-32C.
    Done,
}

/// Итог обработки одного кадра (§8). Готов к сериализации в JSON для UI.
#[derive(Debug, Clone, Copy)]
pub struct FrameStatus {
    /// Найдена ли ЗЧ-рамка на кадре.
    pub detected: bool,
    /// Score детекции [0,1] (0, если не найдено).
    pub score: f64,
    /// Поворот кадра относительно канона в четвертях (0..3).
    pub rotation: u8,
    /// Оценка px на клетку в ПОЛНОМ разрешении (0, если не найдено).
    pub px_per_cell: f64,
    /// Сколько из 8 страйпов кадра прошли CRC-16 (§6.2).
    pub stripes_ok: u8,
    /// Новых (ранее неизвестных) encoding-символов принято этим кадром.
    pub symbols_new: usize,
    /// K source-символов из TransferInfo (0, пока TI не получен).
    pub k: u32,
    /// Всего уникальных encoding-символов, принятых декодером к этому моменту.
    pub symbols_have: usize,
    /// Собран ли объект (state == Done).
    pub done: bool,
    /// Совпал ли CRC-32C собранного объекта (§6.2).
    pub crc_ok: bool,
    /// Кадр опознан как ВНУТРИПОЛОСНЫЙ КАЛИБРОВОЧНЫЙ (§4-IB) — данных не несёт.
    pub calibration: bool,
    /// Возраст кэша калибровки в кадрах (0, если только что обновлён).
    pub calib_age: u32,
    /// Кэш калибровки устарел (по возрасту или по деградации декодирования):
    /// приёмник ЖДЁТ следующего запланированного калибровочного кадра.
    pub calib_stale: bool,
}

impl FrameStatus {
    /// Пустой статус «ничего не найдено» с текущим прогрессом декодера.
    fn empty(k: u32, symbols_have: usize, state: RxState) -> Self {
        FrameStatus {
            detected: false,
            score: 0.0,
            rotation: 0,
            px_per_cell: 0.0,
            stripes_ok: 0,
            symbols_new: 0,
            k,
            symbols_have,
            done: state == RxState::Done,
            crc_ok: state == RxState::Done,
            calibration: false,
            calib_age: 0,
            calib_stale: true,
        }
    }

    /// Ручная сериализация в JSON (без serde) — ровно поля [`FrameStatus`].
    pub fn to_json(&self) -> String {
        format!(
            "{{\"detected\":{},\"score\":{:.4},\"rotation\":{},\"px_per_cell\":{:.2},\
             \"stripes_ok\":{},\"symbols_new\":{},\"k\":{},\"symbols_have\":{},\
             \"done\":{},\"crc_ok\":{},\"calibration\":{},\"calib_age\":{},\"calib_stale\":{}}}",
            self.detected,
            self.score,
            self.rotation,
            self.px_per_cell,
            self.stripes_ok,
            self.symbols_new,
            self.k,
            self.symbols_have,
            self.done,
            self.crc_ok,
            self.calibration,
            self.calib_age,
            self.calib_stale,
        )
    }
}

/// Приёмная сессия (§8). Держит профиль, кэш тон-гамм, декодер фонтана и разметку
/// ESI. `process_frame_yuv` НИКОГДА не паникует на кривом входе.
pub struct RxSession {
    profile: CalibProfile,
    /// bits_per_cell профиля (кэш).
    bpc: u32,
    /// Кэш пер-канальной тон-гаммы (§3.4): оценивается на локе (захвате),
    /// сбрасывается при потере лока (re-lock -> переоценка).
    gammas: Option<[f64; 3]>,
    /// Геометрия прошлого кадра в УМЕНЬШЕННЫХ координатах детекции (§8): пока
    /// есть — идём дешёвым `track_symbol`; None — Searching, нужен захват.
    last_detection: Option<Detection>,
    /// Текущая сессия передачи (первый увиденный session_id побеждает, §6.2).
    session_id: Option<u32>,
    /// Декодер фонтана (создаётся при первом TransferInfo текущей сессии).
    decoder: Option<FountainDecoder>,
    /// K source-символов (из TransferInfo).
    k: u32,
    /// Поток ESI (§6.1), воспроизведённый из K; индекс -> ESI.
    emit: Vec<u32>,
    /// ESI -> первая позиция в потоке (для атрибуции кадра по заголовку.esi).
    pos: HashMap<u32, usize>,
    /// Кумулятивные байт-смещения 8 страйпов (§6.2).
    cum: [usize; 9],
    /// Состояние сессии.
    state: RxState,
    /// Собранный объект (ждёт `take_result`).
    result: Option<Vec<u8>>,
    /// Кэш калибровки по внутриполосным калибровочным кадрам (§4-IB.4).
    /// Пока он валиден, §3.4 берётся ОТТУДА, а односкеточная референсная строка
    /// снимка не читается вовсе.
    calib: CalibCache,
    /// Обрабатывать ли внутриполосные калибровочные кадры (§4-IB). Включено;
    /// путь v0 при пустом/устаревшем кэше остаётся байт-в-байт прежним.
    use_calib: bool,
}

impl RxSession {
    /// Новая сессия с заданным профилем tx.
    pub fn new(profile: CalibProfile) -> Self {
        let bpc = symbol::bits_per_cell(&profile);
        RxSession {
            profile,
            bpc,
            gammas: None,
            last_detection: None,
            session_id: None,
            decoder: None,
            k: 0,
            emit: Vec::new(),
            pos: HashMap::new(),
            cum: [0usize; 9],
            state: RxState::Searching,
            result: None,
            calib: CalibCache::new(),
            use_calib: true,
        }
    }

    /// Включить/выключить обработку внутриполосных калибровочных кадров (§4-IB).
    /// Выключение возвращает приёмник РОВНО на путь v0 (референсная строка §3.4).
    pub fn set_use_calibration_frames(&mut self, on: bool) {
        self.use_calib = on;
    }

    /// Действующая калибровка канала (`None` — нет или устарела).
    pub fn calibration(&self) -> Option<&psicode_core::calframe::CalibEstimate> {
        self.calib.get()
    }

    /// Новая сессия с профилем tx по умолчанию и заданным display-размером клетки.
    /// `cell_px` — как рендерит tx; в демоде он сокращается, важно лишь >= 8.
    /// Значение < 8 заменяется на 12 (live tx-умолчание).
    pub fn with_cell(cell_px: u8) -> Self {
        Self::with_cell_mode(cell_px, false)
    }

    /// То же, но с выбором семейства отображения: `chromatic = true` -> §5.1-CL
    /// (постоянная яркость, 2 бит/клетку), иначе живой 1-битный монохром.
    /// Должно совпадать с тем, что рендерит передатчик (`psicode-tx --chroma`).
    pub fn with_cell_mode(cell_px: u8, chromatic: bool) -> Self {
        let mut p = if chromatic {
            tx_chromatic_profile()
        } else {
            tx_default_profile()
        };
        p.cell_size_px = if cell_px >= 8 { cell_px } else { 12 };
        Self::new(p)
    }

    /// Текущее состояние сессии.
    pub fn state(&self) -> RxState {
        self.state
    }

    /// Забрать собранный объект РОВНО один раз (потом `None`).
    pub fn take_result(&mut self) -> Option<Vec<u8>> {
        self.result.take()
    }

    /// Прогресс: (K, уникальных принятых символов).
    fn progress(&self) -> (u32, usize) {
        let have = self.decoder.as_ref().map_or(0, |d| d.n_seen());
        (self.k, have)
    }

    /// Обработать один кадр камеры YUV_420_888 (§8). Возвращает статус кадра.
    /// НИКОГДА не паникует на кривых размерах/шагах/буферах.
    #[allow(clippy::too_many_arguments)]
    pub fn process_frame_yuv(
        &mut self,
        y: &[u8],
        u: &[u8],
        v: &[u8],
        w: usize,
        h: usize,
        y_stride: usize,
        uv_stride: usize,
        uv_pixel_stride: usize,
    ) -> FrameStatus {
        let (k, have) = self.progress();
        if w == 0 || h == 0 || y_stride < w {
            return FrameStatus::empty(k, have, self.state);
        }
        let yf = YuvFrame {
            y,
            u,
            v,
            w,
            h,
            y_stride,
            uv_stride,
            uv_pixel_stride,
        };

        // --- детекция по УМЕНЬШЕННОЙ плоскости Y (бюджет §8) ---
        let ds = detect_downscale(w, h);
        let dw = (w / ds).max(1);
        let dh = (h / ds).max(1);
        let mut luma = vec![0.0f32; dw * dh];
        let half = ds / 2;
        for j in 0..dh {
            let sy = (j * ds + half).min(h - 1);
            for i in 0..dw {
                let sx = (i * ds + half).min(w - 1);
                luma[j * dw + i] = yf.y_norm(sx, sy);
            }
        }
        // Acquire/track split (§8): Locked -> дешёвый track от прошлой геометрии;
        // Searching -> ограниченный acquire. Track/acquire в УМЕНЬШЕННЫХ координатах.
        let det = if let Some(prev) = self.last_detection.as_ref() {
            match track_symbol(dw, dh, &luma, prev) {
                Ok(d) => d,
                Err(_) => {
                    // потеря слежения -> сброс лока и тона; следующий кадр захватит.
                    self.last_detection = None;
                    self.gammas = None;
                    return FrameStatus::empty(k, have, self.state);
                }
            }
        } else {
            match detect_symbol_acquire(dw, dh, &luma) {
                Ok(d) => d,
                Err(_) => {
                    self.gammas = None;
                    return FrameStatus::empty(k, have, self.state);
                }
            }
        };
        let det_score = det.score;
        let det_rotation = det.rotation_quadrants;

        // гомография детекции работает в уменьшенных px; демод — в полных, поэтому
        // выход base-карты домножаем на ds.
        let base = frame_map(&self.profile, &det);
        let dsf = ds as f64;
        let map = |u: f64, v: f64| {
            let (x, y) = base(u, v);
            (x * dsf, y * dsf)
        };

        // px/клетку в ПОЛНОМ разрешении: расстояние между центрами клеток (30,30)
        // и (31,30) в координатах снимка.
        let px_per_cell = {
            let cell = self.profile.cell_size_px as f64;
            let quiet = self.profile.quiet_zone_cells() as f64;
            let cc = |cx: f64, cy: f64| {
                let uu = (quiet + cx) * cell + cell / 2.0;
                let vv = (quiet + cy) * cell + cell / 2.0;
                map(uu, vv)
            };
            let (ax, ay) = cc(symbol::PAYLOAD_COLS as f64 / 2.0, symbol::PAYLOAD_ROWS as f64 / 2.0);
            let (bx, by) = cc(
                symbol::PAYLOAD_COLS as f64 / 2.0 + 1.0,
                symbol::PAYLOAD_ROWS as f64 / 2.0,
            );
            ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt()
        };

        // --- сэмплеры RGB по ПОЛНОМУ разрешению (билинейка над YUV on-demand) ---
        // Сырой (до гаммы) сэмпл для самокалибровки тона; демод-сэмпл добавляет
        // пер-канальную гамму. On-demand над YUV (а не предвычисленный RGB-план)
        // держит бюджет: ~тысячи сэмплов/кадр вместо гаммы по всем 2M пикселям.
        let raw_sample = |x: f64, y: f64| -> [f32; 3] { bilinear_rgb(&yf, x, y) };

        // --- внутриполосный КАЛИБРОВОЧНЫЙ кадр (§4-IB)? ---
        // Проверка идёт ДО демодуляции и до всякой калибровки: маркер бинарный,
        // luma-only и снимается со второй разности по вертикали (§4-IB.2),
        // поэтому не требует ни гаммы, ни развязки каналов.
        if self.use_calib && calframe::is_calibration_frame(&self.profile, &map, &raw_sample) {
            let est = calframe::estimate_from_frame(&self.profile, &map, &raw_sample);
            if est.ok {
                // калибровка перекрывает и тон-гамму: 5 КРУПНЫХ нейтральных
                // плиток против 8 односкеточных клеток референсной строки.
                self.gammas = Some(est.gammas);
            }
            self.calib.accept(est);
            self.last_detection = Some(det);
            let (k, have) = self.progress();
            return FrameStatus {
                detected: true,
                score: det_score,
                rotation: det_rotation,
                px_per_cell,
                stripes_ok: 0,
                symbols_new: 0,
                k,
                symbols_have: have,
                done: self.state == RxState::Done,
                crc_ok: self.state == RxState::Done,
                calibration: true,
                calib_age: 0,
                calib_stale: self.calib.stale(),
            };
        }

        // Тон-гамма: оценка на первом локе, дальше кэш (сброшен при потере лока).
        // Живая калибровка §4-IB переживает потерю лока — она свойство пары
        // «дисплей+камера», а не геометрии, — поэтому после ре-лока гамма
        // берётся из неё, а не переоценивается по референсной строке.
        if self.gammas.is_none() {
            self.gammas = Some(match self.calib.get() {
                Some(est) => est.gammas,
                None => tone::estimate_channel_gammas(&self.profile, &map, &raw_sample),
            });
        }
        let cg = self.gammas.unwrap_or([tone::DEFAULT_GAMMA; 3]);
        let cgf = [cg[0] as f32, cg[1] as f32, cg[2] as f32];
        let lin_sample = |x: f64, y: f64| -> [f32; 3] {
            let r = bilinear_rgb(&yf, x, y);
            [
                r[0].max(0.0).powf(cgf[0]),
                r[1].max(0.0).powf(cgf[1]),
                r[2].max(0.0).powf(cgf[2]),
            ]
        };

        // --- демод + разбор L3 ---
        // 1-битная яркость (Mono/GreenOnly): ЛОКАЛЬНЫЙ двухмасштабный порог
        // (symbol::demod_symbol_local) — на живом телефоне SER 0.13 -> ~0.0004
        // против глобального порога, т.к. фон яркости гуляет 0.62..0.86 по кадру.
        // Порог монотонен -> берём СЫРОЙ зелёный (raw_sample), гамма не нужна.
        // Прочие режимы (многоуровневая яркость/хрома) — прежний demod_symbol.
        // Многоуровневые/хромо-режимы: если кэш калибровки (§4-IB.4) жив, берём
        // матрицу развязки §3.4 ИЗ НЕГО и НЕ читаем односкеточную референсную
        // строку снимка вовсе. Устарел — молча падаем на путь v0 и ждём
        // следующего запланированного калибровочного кадра (обратного канала
        // нет, запрашивать нечего).
        let cells_rx = if self.profile.luma_bits == 1 && self.profile.chroma_bits() == 0 {
            demod_symbol_local(&self.profile, &map, &raw_sample)
        } else if let Some(est) = self.calib.get() {
            demod_symbol_matrix(&self.profile, &map, &lin_sample, &est.matrix)
        } else {
            demod_symbol(&self.profile, &map, &lin_sample)
        };
        let parsed = l3::parse_frame(&cells_rx, self.bpc);
        let (c_start, c_end) = read_counters(&self.profile, &map, &lin_sample);
        let stripes_ok = parsed.stripes_ok.iter().filter(|&&ok| ok).count() as u8;

        // --- сессия + фонтан ---
        let mut symbols_new = 0usize;
        if let Some(hdr) = parsed.header {
            // фильтр session_id: первый увиденный побеждает (§6.2).
            let sid = *self.session_id.get_or_insert(hdr.session_id);
            if hdr.session_id == sid {
                // инициализация декодера по TransferInfo (первый TI-кадр сессии).
                if self.decoder.is_none() {
                    if let Some(ti) = parsed.transfer_info {
                        self.init_decoder(ti.k, ti.symbol_size as usize, ti.transfer_length as usize, ti.checksum);
                    }
                }
                if self.decoder.is_some() {
                    self.ensure_emit_covers(hdr.esi);
                    let (_, useful) = self.feed_frame(&parsed, &hdr, c_start, c_end);
                    symbols_new = useful;
                    self.try_complete();
                }
            }
        }

        // сохраняем геометрию (в УМЕНЬШЕННЫХ координатах) для трекинга следующего кадра.
        self.last_detection = Some(det);

        // Контроль устаревания (§4-IB.4): возраст +1 и EWMA доли валидных
        // страйпов. Просевшее здоровье = сигнал «канал уехал», и приёмник просто
        // ЖДЁТ ближайшего калибровочного кадра расписания.
        self.calib.observe_payload(stripes_ok, 8);

        let (k, have) = self.progress();
        FrameStatus {
            detected: true,
            score: det_score,
            rotation: det_rotation,
            px_per_cell,
            stripes_ok,
            symbols_new,
            k,
            symbols_have: have,
            done: self.state == RxState::Done,
            crc_ok: self.state == RxState::Done,
            calibration: false,
            calib_age: self.calib.age().min(u32::MAX as u64) as u32,
            calib_stale: self.calib.stale(),
        }
    }

    /// Дорастить разметку ESI до `esi` (§6.1): tx крутится бесконечно, и после
    /// систематической фазы поток — чистый repair с инкрементом ESI, поэтому
    /// продолжение таблицы тривиально. Без этого приёмник, подключившийся к
    /// давно идущей передаче, молча выбрасывал КАЖДЫЙ кадр (esi за горизонтом).
    fn ensure_emit_covers(&mut self, esi: u32) {
        if self.k == 0 || esi < self.k || self.pos.contains_key(&esi) {
            return;
        }
        let mut last = match self.emit.last() {
            Some(&e) => e,
            None => return,
        };
        if last < self.k {
            return; // хвост ещё в систематической фазе — esi не отсюда
        }
        let target = esi.saturating_add(SYMBOLS_PER_FRAME as u32 * 64);
        while last < target {
            last += 1;
            self.emit.push(last);
            let i = self.emit.len() - 1;
            self.pos.entry(last).or_insert(i);
        }
    }

    /// Инициализировать декодер и разметку ESI по параметрам TransferInfo (§6.2).
    fn init_decoder(&mut self, k: u32, symbol_size: usize, transfer_length: usize, checksum: u32) {
        self.k = k;
        // поток ESI с запасом: cap отправителя ~10·K символов (§9.2) + запас на
        // N+1-lookahead рваного снимка. Неизвестные ESI просто не попадут в pos.
        let n_frames = (10 * k as usize).div_ceil(SYMBOLS_PER_FRAME) + 64;
        self.emit = emission_order(k, n_frames);
        self.pos = HashMap::with_capacity(self.emit.len());
        for (i, &e) in self.emit.iter().enumerate() {
            self.pos.entry(e).or_insert(i);
        }
        self.cum = stripe_cum(self.bpc);
        self.decoder = Some(FountainDecoder::new(k, symbol_size, transfer_length, checksum));
        if self.state == RxState::Searching {
            self.state = RxState::Locked;
        }
    }

    /// Скормить символы разобранного кадра фонтану с атрибуцией §6.3.
    fn feed_frame(
        &mut self,
        parsed: &ParsedFrame,
        hdr: &FrameHeader,
        c_start: u8,
        c_end: u8,
    ) -> (usize, usize) {
        // разъединяем заимствования полей self (decoder mut, emit/pos/cum immut).
        let symbol_size = symbol_size_for(self.bpc);
        let RxSession {
            decoder,
            emit,
            pos,
            cum,
            ..
        } = self;
        let decoder = match decoder.as_mut() {
            Some(d) => d,
            None => return (0, 0),
        };
        process_frame(decoder, emit, pos, parsed, hdr, c_start, c_end, cum, symbol_size)
    }

    /// Попытка завершить сборку; при успехе фиксирует результат и Done.
    fn try_complete(&mut self) {
        if self.state == RxState::Done {
            return;
        }
        if let Some(decoder) = self.decoder.as_mut() {
            if decoder.n_seen() >= decoder.k() {
                if let Some(obj) = decoder.try_finish() {
                    self.result = Some(obj);
                    self.state = RxState::Done;
                }
            }
        }
    }
}

/// Целочисленный фактор уменьшения плоскости детекции: наибольшая сторона <=
/// [`DETECT_MAX_DIM`]. Не меньше 1.
fn detect_downscale(w: usize, h: usize) -> usize {
    let m = w.max(h);
    if m <= DETECT_MAX_DIM {
        1
    } else {
        m.div_ceil(DETECT_MAX_DIM)
    }
}

/// Билинейный сэмпл сырого RGB [0,1] по кадру YUV с зажимом к краю.
#[inline]
fn bilinear_rgb(yf: &YuvFrame, x: f64, y: f64) -> [f32; 3] {
    let xc = x.clamp(0.0, (yf.w - 1) as f64);
    let yc = y.clamp(0.0, (yf.h - 1) as f64);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(yf.w - 1);
    let y1 = (y0 + 1).min(yf.h - 1);
    let fx = (xc - x0 as f64) as f32;
    let fy = (yc - y0 as f64) as f32;
    let c00 = yf.rgb_at(x0, y0);
    let c10 = yf.rgb_at(x1, y0);
    let c01 = yf.rgb_at(x0, y1);
    let c11 = yf.rgb_at(x1, y1);
    let mut o = [0.0f32; 3];
    for c in 0..3 {
        let a = c00[c] * (1.0 - fx) + c10[c] * fx;
        let b = c01[c] * (1.0 - fx) + c11[c] * fx;
        o[c] = a * (1.0 - fy) + b * fy;
    }
    o
}

// ---------------------------------------------------------------------------
// Приёмная сторона фонтана — зеркалит psicode-sim/src/transfer.rs (§6.1/§6.3).
// Порядок ESI и атрибуция рваного снимка ОБЯЗАНЫ совпадать с передатчиком.
// ---------------------------------------------------------------------------

/// Порядок ESI потока (§6.1): source 0..K−1 с вплетением repair каждые
/// `REPAIR_EVERY` source; после систематического прохода — только repair.
/// Генерирует ровно `n_frames · SYMBOLS_PER_FRAME` элементов.
fn emission_order(k: u32, n_frames: usize) -> Vec<u32> {
    let total = n_frames * SYMBOLS_PER_FRAME;
    let mut v = Vec::with_capacity(total);
    let mut src = 0u32;
    let mut rep = k;
    let mut since = 0u32;
    while v.len() < total {
        if src < k {
            v.push(src);
            src += 1;
            since += 1;
            if since == REPAIR_EVERY {
                if v.len() < total {
                    v.push(rep);
                    rep += 1;
                }
                since = 0;
            }
        } else {
            v.push(rep);
            rep += 1;
        }
    }
    v.truncate(total);
    v
}

/// Кумулятивные байт-смещения 8 страйпов при `bpc` (§6.2).
fn stripe_cum(bpc: u32) -> [usize; 9] {
    let mut cum = [0usize; 9];
    for i in 0..8 {
        let cap = (l3::STRIPE_ROWS[i] * l3::PAYLOAD_COLS * bpc as usize - 16) / 8;
        cum[i + 1] = cum[i] + cap;
    }
    cum
}

/// Восстановить символы кадра из выбранных страйпов и скормить фонтану.
/// `include[i]` — учитывать ли страйп i; `j0` — emission-индекс первого символа
/// кадра; `header_len` — длина шапки потока. Возврат (получено, из них новых).
#[allow(clippy::too_many_arguments)]
fn recover_and_feed(
    decoder: &mut FountainDecoder,
    emit: &[u32],
    j0: usize,
    header_len: usize,
    symbol_size: usize,
    stripe_data: &[Option<Vec<u8>>; 8],
    include: &[bool; 8],
    cum: &[usize; 9],
) -> (usize, usize) {
    let total = cum[8];
    let mut stream = vec![0u8; total];
    let mut known = vec![false; total];
    for i in 0..8 {
        if include[i] {
            if let Some(data) = &stripe_data[i] {
                let off = cum[i];
                for (b, &val) in data.iter().enumerate() {
                    if off + b < total {
                        stream[off + b] = val;
                        known[off + b] = true;
                    }
                }
            }
        }
    }
    let (mut recv, mut useful) = (0usize, 0usize);
    let mut cov_dbg = String::new();
    for kk in 0..SYMBOLS_PER_FRAME {
        let s = header_len + kk * symbol_size;
        let e = s + symbol_size;
        if e > total {
            break;
        }
        if rx_dbg() {
            let cov = known[s..e].iter().filter(|&&x| x).count();
            cov_dbg.push_str(&format!(" s{kk}:{cov}/{symbol_size}"));
        }
        if known[s..e].iter().all(|&x| x) {
            let ei = j0 + kk;
            if ei < emit.len() {
                recv += 1;
                if decoder.push(emit[ei], &stream[s..e]) {
                    useful += 1;
                }
            }
        }
    }
    if rx_dbg() {
        std::eprintln!("[rx] recover j0={j0} hlen={header_len}:{cov_dbg} -> recv {recv} useful {useful}");
    }
    (recv, useful)
}

/// Разобрать один принятый кадр и скормить его символы фонтану с корректной
/// разметкой ESI и атрибуцией рваного снимка (§6.3). Возврат (получено, новых).
#[allow(clippy::too_many_arguments)]
fn process_frame(
    decoder: &mut FountainDecoder,
    emit: &[u32],
    pos: &HashMap<u32, usize>,
    parsed: &ParsedFrame,
    h: &FrameHeader,
    c_start: u8,
    c_end: u8,
    cum: &[usize; 9],
    symbol_size: usize,
) -> (usize, usize) {
    // индекс кадра N из заголовка (emit[seq·8] уникален -> позиция = seq·8).
    let j0 = match pos.get(&h.esi) {
        Some(&j) => j,
        None => return (0, 0),
    };
    let n_idx = j0 / SYMBOLS_PER_FRAME;

    // данные по страйпам, разложенные по индексу страйпа.
    let mut stripe_data: [Option<Vec<u8>>; 8] =
        [None, None, None, None, None, None, None, None];
    for (off, bytes) in &parsed.salvaged {
        for i in 0..8 {
            if cum[i] == *off {
                stripe_data[i] = Some(bytes.clone());
                break;
            }
        }
    }
    let valid = parsed.stripes_ok;
    let hlen_n = if h.has_transfer_info() {
        l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN
    } else {
        l3::FRAME_HEADER_LEN
    };

    let exp_clean = (n_idx & 0xFF) as u8;
    let exp_torn = ((n_idx + 1) & 0xFF) as u8;

    let (mut recv, mut useful) = (0usize, 0usize);
    if c_start == exp_clean && c_end == exp_clean {
        // не рвано: все CRC-валидные страйпы принадлежат кадру N.
        let (r, u) = recover_and_feed(decoder, emit, j0, hlen_n, symbol_size, &stripe_data, &valid, cum);
        recv += r;
        useful += u;
    } else if c_start == exp_torn && c_end == exp_torn {
        // рвано (§6.3): верхний прогон -> N, нижний -> N+1.
        let mut top_end = 0usize;
        while top_end < 8 && valid[top_end] {
            top_end += 1;
        }
        let mut inc_top = [false; 8];
        for b in inc_top.iter_mut().take(top_end) {
            *b = true;
        }
        let (r, u) = recover_and_feed(decoder, emit, j0, hlen_n, symbol_size, &stripe_data, &inc_top, cum);
        recv += r;
        useful += u;

        let mut bot_start = 8usize;
        while bot_start > 0 && valid[bot_start - 1] {
            bot_start -= 1;
        }
        if bot_start > top_end {
            let j0_np1 = (n_idx + 1) * SYMBOLS_PER_FRAME;
            if j0_np1 + SYMBOLS_PER_FRAME <= emit.len() {
                let hlen_np1 = if (n_idx + 1) % 8 == 0 {
                    l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN
                } else {
                    l3::FRAME_HEADER_LEN
                };
                let mut inc_bot = [false; 8];
                for b in inc_bot.iter_mut().take(8).skip(bot_start) {
                    *b = true;
                }
                let (r, u) =
                    recover_and_feed(decoder, emit, j0_np1, hlen_np1, symbol_size, &stripe_data, &inc_bot, cum);
                recv += r;
                useful += u;
            }
        }
    } else {
        // неоднозначно (шумный счётчик): берём только непрерывный верхний прогон.
        let mut top_end = 0usize;
        while top_end < 8 && valid[top_end] {
            top_end += 1;
        }
        let mut inc_top = [false; 8];
        for b in inc_top.iter_mut().take(top_end) {
            *b = true;
        }
        let (r, u) = recover_and_feed(decoder, emit, j0, hlen_n, symbol_size, &stripe_data, &inc_top, cum);
        recv += r;
        useful += u;
    }
    (recv, useful)
}
