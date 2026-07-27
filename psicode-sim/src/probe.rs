//! Подкоманда `probe` — ёмкость ОДНОГО крупного патча как функция способа
//! укладки комплексного символа z = Re + i·Im в цвет.
//!
//! Вопрос, на который отвечает модуль: SPEC §5.1 кладёт Re на ЯРКОСТНУЮ ось
//! (G), а Im — на ось R−B. Гипотеза владельца: это неверно, потому что яркость
//! — ровно та ось, по которой бьют ошибка экспозиции и поле освещённости, тогда
//! как ЦВЕТНОСТЬ (направление вектора RGB) инвариантна и к блюру (блюр сохраняет
//! локальные средние), и к экспозиции (глобальный скаляр).
//!
//! Эксперимент НЕ трогает ни core, ни rx/tx, ни спеку: это изолированная
//! студия поверх `image`/`pipeline`/`channel`/`rng`.
//!
//! # Геометрия
//! Никакой сетки 57×55 и никакого ЗЧ-кольца: считаем геометрию «genie» (позиции
//! патчей известны точно). Полезная область — N×N КРУПНЫХ квадратных патчей со
//! стороной W display-px. Рядом (выше) — блок 4×4 референсного паттерна §3.4
//! (`K W R G B C M Y K W` + 6-ступенчатая серая лесенка), по которому приёмник
//! калибрует гамму / чёрный / белый / кросстолк ровно так же, как настоящий.
//! Обе области окружены нейтральным полем шириной ≥ 6σ.
//!
//! # Четыре отображения
//! * **M1** — текущая SPEC §5.1: `G = M + A_L·x`, `R,B = M + A_L·x ± A_C·y`.
//! * **M2** — постоянная яркость: сумма драйвов R+G+B ≡ S, z целиком в цветности.
//! * **M3** — только яркость (контроль): `R = G = B = M + usable·x`.
//! * **M4** — M2 с гауссовой апертурой (σ_apod = W/4) и согласованным фильтром.
//!
//! # Тракт канала (что добавлено к `ChannelParams`)
//! `drive -> линейная яркость (d/255)^γ_disp -> ПОЛЕ ОСВЕЩЁННОСТИ (диагональный
//! наклон 0.62..0.86) -> блюр σ -> ОШИБКА ЭКСПОЗИЦИИ (глобальный скаляр
//! U[0.8,1.25] на попытку) -> кросстолк -> тон-кривая ISP raw = L^(1/γ_isp)
//! -> шум -> клип -> 8 бит`. Поле и экспозиция — новые импейрменты, их нет в
//! базовом `ChannelParams`; именно они и есть предмет спора M1 vs M2.
//!
//! # Метрика (ПРОКСИ, не Шеннон)
//! `bits_axis = log2(2 / (k·σ_axis))`, k = 3.3 — грубо 1e-3 вероятности ошибки
//! на крайних уровнях PAM-созвездия на оси длиной 2 ([-1,+1]). σ_axis — СКО
//! рассеяния ПОСЛЕ вычитания систематического смещения; смещение печатается
//! отдельно (его приёмник в принципе может выкалибровать, шум — нет).

use crate::channel::{ChannelParams, IDENTITY};
use crate::image::Image;
use crate::pipeline::{blur, crosstalk};
use crate::rng::{seed_for, Rng};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Константы
// ---------------------------------------------------------------------------

/// Средняя точка динамического диапазона (§5.1).
const MID: f64 = 128.0;
/// Белый уровень эталонного профиля §7.4: white_level_pct = 100% -> 255.
const WHITE255: f64 = 255.0;
/// Чёрный уровень эталонного профиля §7.4: black_level_pct = 2% -> 5.
const BLACK255: f64 = 5.0;
/// Гамма ДИСПЛЕЯ (drive -> линейная яркость), семантика `ChannelParams::gammas`.
const DISPLAY_GAMMA: f64 = 2.2;
/// Доли `usable` под яркость/хрому в §5.1 (psicode_core::symbol::A_*_FRACTION_CHROMA).
const A_L_FRACTION: f64 = 0.7;
const A_C_FRACTION: f64 = 0.3;
/// Коэффициент запаса в метрике бит/ось.
const K_PAM: f64 = 3.3;
/// Поле освещённости: множитель по диагонали кадра (замер на Samsung A22).
const ILLUM_LO: f64 = 0.62;
const ILLUM_HI: f64 = 0.86;
/// Сторона «кадра камеры» для режима поля с ФИКСИРОВАННОЙ физической
/// протяжённостью: наклон 0.62..0.86 растянут ровно на столько px, а канвас
/// центрируется внутри. Нужно там, где W меняется: при поле, привязанном к
/// канвасу, крупные W автоматически получают более пологий градиент на пиксель,
/// и сравнение бит-на-пиксель между разными W становится нечестным.
const FIELD_FRAME_PX: f64 = 1400.0;
/// Ошибка экспозиции: глобальный скаляр, равномерно на попытку.
const EXP_LO: f64 = 0.80;
const EXP_HI: f64 = 1.25;
/// Гаммы ISP, замеренные на Samsung A22 (raw = L^(1/γ)).
const GAMMAS_SAMSUNG: [f64; 3] = [3.8, 4.7, 5.7];
/// Согласованные гаммы (ISP в точности обращает дисплей -> raw ∝ drive).
const GAMMAS_MATCHED: [f64; 3] = [DISPLAY_GAMMA; 3];
/// σ шума в ЗАХВАЧЕННОМ (после тон-кривой) кадре, доля полной шкалы.
/// Эталонный профиль §7.4: noise_sigma_q = 12 -> σ ≈ 2.0 градации -> 2/255.
const NOISE_SIGMA: f64 = 2.0 / 255.0;
/// Кросстолк эталонного профиля §7.4: 6% R<->G, 8% G<->B.
const XTALK_RG: f64 = 0.06;
const XTALK_GB: f64 = 0.08;
/// Порог насыщения: сэмплы выше него выбрасываются из калибровки поканально.
const SAT: f64 = 0.99;

// ---------------------------------------------------------------------------
// Отображения
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mapping {
    /// SPEC §5.1 (базовая линия).
    M1,
    /// Постоянная яркость, z целиком в цветности.
    M2,
    /// Только яркость (контроль).
    M3,
    /// M2 + гауссова апертура и согласованный фильтр.
    M4,
    /// SPEC §5.1 + РАСПРЕДЕЛЁННЫЕ ПИЛОТЫ: решётка известных нейтральных патчей
    /// внутри полезной области, по ней приёмник подгоняет и снимает
    /// низкопорядковое поле освещённости ДО демодуляции. Площадь пилотов
    /// вычитается из полезной (см. `Cfg::pilot_overhead`).
    M1P,
}

impl Mapping {
    fn all() -> [Mapping; 4] {
        [Mapping::M1, Mapping::M2, Mapping::M3, Mapping::M4]
    }
    /// Те же четыре плюс арм M1' (для таблиц, где он участвует).
    fn all_with_pilots() -> [Mapping; 5] {
        [Mapping::M1, Mapping::M1P, Mapping::M2, Mapping::M3, Mapping::M4]
    }
    fn name(self) -> &'static str {
        match self {
            Mapping::M1 => "M1 spec§5.1",
            Mapping::M2 => "M2 const-lum",
            Mapping::M3 => "M3 luma-only",
            Mapping::M4 => "M4 const-lum+gauss",
            Mapping::M1P => "M1' spec§5.1+pilots",
        }
    }
    /// Нейтральный drive (z = 0) — им же залито охранное поле.
    fn neutral_drive(self) -> f64 {
        match self {
            Mapping::M1 | Mapping::M3 | Mapping::M1P => MID,
            Mapping::M2 | Mapping::M4 => m2().u,
        }
    }
    /// Несёт ли отображение мнимую ось.
    fn has_im(self) -> bool {
        self != Mapping::M3
    }
    /// Нужна ли гауссова апертура (согласованный фильтр вместо плоского среднего).
    fn apodized(self) -> bool {
        self == Mapping::M4
    }
}

/// Амплитуды §5.1: usable = min(white − M, M − black), A_L = 0.7·usable,
/// A_C = 0.3·usable.
fn m1_amps() -> (f64, f64, f64) {
    let usable = (WHITE255 - MID).min(MID - BLACK255).max(0.0);
    (usable, A_L_FRACTION * usable, A_C_FRACTION * usable)
}

/// Параметры M2, решающие задачу «максимум созвездия при drive ∈ [black, white]
/// для всех |z| ≤ 1».
#[derive(Clone, Copy, Debug)]
struct M2Params {
    /// Нейтральная точка на канал, u = S/3.
    u: f64,
    /// Максимальное отклонение drive от u на границе диска.
    amp: f64,
    b: f64,
    c: f64,
    s: f64,
}

/// Вывод параметров M2.
///
/// Прямое отображение: `R = u(1 − b·x + c·y)`, `G = u(1 + 2b·x)`,
/// `B = u(1 − b·x − c·y)`, u = S/3, сумма ≡ S по построению.
///
/// Векторы отклонений каналов в плоскости (x, y):
/// `G: (2b, 0)`, `R: (−b, c)`, `B: (−b, −c)`. На единичном ДИСКЕ модуль
/// отклонения канала равен норме его вектора. Нормы равны ⇔ `c = √3·b`; при
/// этом три вектора образуют равносторонний треугольник (120°), все нормы = 2b,
/// и ограничение гамута сводится к ОДНОМУ радиусу ρ = 2b.
///
/// Тот же выбор `c = √3·b` уравнивает шум по осям: из обращения (см. ниже)
/// σ_x ∝ √6/(2Sb), σ_y ∝ 3√2/(2Sc); равенство ⇔ c = √3·b.
///
/// Обозначив A = u·2b (свинг drive на канал), ограничения превращаются в
/// `u + A ≤ white`, `u − A ≥ black`; максимум A = min(white−u, u−black)
/// достигается при u = (white + black)/2, где A = (white − black)/2.
fn m2() -> M2Params {
    let u = 0.5 * (WHITE255 + BLACK255);
    let amp = (WHITE255 - u).min(u - BLACK255);
    let b = amp / (2.0 * u);
    let c = 3f64.sqrt() * b;
    M2Params { u, amp, b, c, s: 3.0 * u }
}

/// Прямое отображение z -> тройка drive (без квантования и без клампа).
fn drive_for(m: Mapping, x: f64, y: f64) -> [f64; 3] {
    match m {
        Mapping::M1 | Mapping::M1P => {
            let (_, a_l, a_c) = m1_amps();
            [MID + a_l * x + a_c * y, MID + a_l * x, MID + a_l * x - a_c * y]
        }
        Mapping::M3 => {
            let (usable, _, _) = m1_amps();
            let v = MID + usable * x;
            [v, v, v]
        }
        Mapping::M2 | Mapping::M4 => {
            let p = m2();
            [
                p.u * (1.0 - p.b * x + p.c * y),
                p.u * (1.0 + 2.0 * p.b * x),
                p.u * (1.0 - p.b * x - p.c * y),
            ]
        }
    }
}

/// Обратное отображение: тройка ОЦЕНЁННЫХ drive -> ẑ.
///
/// Проверка алгебры M2 (ТЗ содержало ошибку в множителе 3/2 в обеих формулах):
/// с u = S/3,
/// `2G − R − B = 2u(1+2bx) − u(1−bx+cy) − u(1−bx−cy) = 6u·b·x = 2S·b·x`
/// ⇒ **x = (2G − R − B)/(2·S·b)** (в ТЗ было /(3·S·b));
/// `R − B = 2u·c·y = (2S/3)·c·y`
/// ⇒ **y = 3(R − B)/(2·S·c)** (в ТЗ было /(S·c)).
///
/// S берётся ИЗМЕРЕННОЙ суммой R+G+B, а не номиналом: общий множитель λ на всех
/// трёх драйвах (а именно им становится любой скаляр освещённости/экспозиции
/// после обращения гаммы дисплея) сокращается в отношении ТОЧНО. В этом и есть
/// весь смысл «постоянной яркости».
fn z_from_drive(m: Mapping, d: [f64; 3]) -> (f64, f64) {
    match m {
        Mapping::M1 | Mapping::M1P => {
            let (_, a_l, a_c) = m1_amps();
            ((d[1] - MID) / a_l, (d[0] - d[2]) / (2.0 * a_c))
        }
        Mapping::M3 => {
            let (usable, _, _) = m1_amps();
            // все три канала несут один и тот же сигнал -> усредняем (честный
            // монохромный контроль: он бы так и делал).
            (((d[0] + d[1] + d[2]) / 3.0 - MID) / usable, 0.0)
        }
        Mapping::M2 | Mapping::M4 => {
            let p = m2();
            let s = (d[0] + d[1] + d[2]).max(1e-9);
            (
                (2.0 * d[1] - d[0] - d[2]) / (2.0 * s * p.b),
                3.0 * (d[0] - d[2]) / (2.0 * s * p.c),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Референсный паттерн §3.4 (локальная копия private `symbol::ref_pattern`)
// ---------------------------------------------------------------------------

/// Число ступеней серой лесенки §3.4.
const REF_GRAY_STEPS: usize = 6;
/// Число клеток референсного паттерна §3.4.
const REF_N: usize = 16;
/// Сторона референсного блока в клетках. 16 записей паттерна кладутся в
/// квадрант 4×4 и ЗЕРКАЛЯТСЯ в остальные три: каждая запись получает 4 копии в
/// точках, симметричных относительно центра блока. Центроид копий любой записи
/// — центр блока, поэтому ЛЮБОЕ линейное поле освещённости сокращается в
/// среднем по копиям ТОЧНО. Без этого градиент поля ВНУТРИ референса
/// портит подгонку гаммы/кросстолка и бьёт по всем отображениям сразу.
const REF_TILE: usize = 8;

/// Индекс записи паттерна §3.4 в клетке (row, col) блока со стороной `tile`
/// клеток: `tile = 8` — зеркальная раскладка, `tile = 4` — наивная (одна копия
/// на запись, используется только диагностической строкой абляции).
fn ref_entry_at(tile: usize, row: usize, col: usize) -> usize {
    if tile == REF_TILE {
        let r0 = row.min(tile - 1 - row);
        let c0 = col.min(tile - 1 - col);
        r0 * 4 + c0
    } else {
        row * 4 + col
    }
}

/// Клетка референсного паттерна по индексу 0..16: `K W R G B C M Y K W` + 6 серых.
fn ref_drives(i: usize) -> [f64; 3] {
    let (k, w) = (BLACK255, WHITE255);
    match i {
        0 => [k, k, k],
        1 => [w, w, w],
        2 => [w, k, k],
        3 => [k, w, k],
        4 => [k, k, w],
        5 => [k, w, w],
        6 => [w, k, w],
        7 => [w, w, k],
        8 => [k, k, k],
        9 => [w, w, w],
        _ => {
            let step = (i - 10) as f64;
            let d = k + (w - k) * step / (REF_GRAY_STEPS - 1) as f64;
            [d, d, d]
        }
    }
}

/// Нейтральная ли клетка паттерна (K, W и серая лесенка).
fn ref_is_neutral(i: usize) -> bool {
    matches!(i, 0 | 1 | 8 | 9 | 10..=15)
}

// ---------------------------------------------------------------------------
// Конфигурация точки развёртки
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Cfg {
    mapping: Mapping,
    /// Сторона патча, display-px.
    w: usize,
    /// Патчей на сторону.
    n: usize,
    /// Поле освещённости включено.
    illum: bool,
    /// Ошибка экспозиции включена.
    exposure: bool,
    /// Гаммы ISP камеры (raw = L^(1/γ)).
    isp: [f64; 3],
    trials: usize,
    /// 8-битное квантование drive НА ПЕРЕДАЧЕ и raw НА ПРИЁМЕ.
    quant: bool,
    /// Блюр / кросстолк / шум / гамма дисплея — из штатного `ChannelParams`.
    ch: ChannelParams,
    /// Зеркальный (8×8) референсный блок вместо наивного 4×4.
    ref_sym: bool,
    /// Полный размах хроматического дифференциала поля между крайними
    /// каналами по кадру (доля). 0 — ахроматическое поле, как раньше.
    chroma_tilt: f64,
    /// Какая из двух ортогональных хроматических мод наклоняется.
    chroma_mode: ChromaMode,
    /// Поле привязано к ФИКСИРОВАННОМУ кадру `FIELD_FRAME_PX`, а не к канвасу.
    field_fixed_frame: bool,
    /// Период решётки пилотов в патчах (0 — пилотов нет). Активен только у M1'.
    pilot_period: usize,
    /// Порядок полинома поля, подгоняемого по пилотам: 1 — аффинный, 2 — биквадрат.
    pilot_order: usize,
    /// База сида (одна на все отображения в одной точке развёртки).
    seed_point: usize,
}

impl Cfg {
    fn sigma(&self) -> f64 {
        self.ch.blur_sigma_rgb[1]
    }
    fn ref_tile(&self) -> usize {
        if self.ref_sym {
            REF_TILE
        } else {
            4
        }
    }
    /// Пилотная ли позиция сетки (только у M1' и при заданном периоде).
    fn is_pilot(&self, i: usize, j: usize) -> bool {
        self.mapping == Mapping::M1P
            && self.pilot_period > 0
            && i % self.pilot_period == 0
            && j % self.pilot_period == 0
    }
    /// Доля позиций сетки, отданная под пилоты (площадные накладные расходы).
    fn pilot_overhead(&self) -> f64 {
        if self.mapping != Mapping::M1P || self.pilot_period == 0 {
            return 0.0;
        }
        let mut pilots = 0usize;
        for j in 0..self.n {
            for i in 0..self.n {
                if self.is_pilot(i, j) {
                    pilots += 1;
                }
            }
        }
        pilots as f64 / (self.n * self.n) as f64
    }
    /// Базовая конфигурация: W=64, N=2, σ=2, оба импейрмента, гаммы Samsung.
    fn base(mapping: Mapping, trials: usize, n: usize) -> Self {
        let mut ch = ChannelParams {
            gammas: [DISPLAY_GAMMA; 3],
            px_per_cell: 8.0,
            cell_size_px: 8.0,
            homography: IDENTITY,
            blur_sigma_rgb: [0.0; 3],
            crosstalk_rg: XTALK_RG,
            crosstalk_gb: XTALK_GB,
            gain: [1.0; 3],
            offset: [0.0; 3],
            noise_sigma: NOISE_SIGMA,
        };
        ch.set_blur(2.0);
        Cfg {
            mapping,
            w: 64,
            n,
            illum: true,
            exposure: true,
            isp: GAMMAS_SAMSUNG,
            trials,
            quant: true,
            ch,
            ref_sym: true,
            chroma_tilt: 0.0,
            chroma_mode: ChromaMode::Rb,
            field_fixed_frame: false,
            pilot_period: 3,
            pilot_order: 1,
            seed_point: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Геометрия сцены
// ---------------------------------------------------------------------------

struct Layout {
    /// Охранное поле вокруг каждого блока, ≥ 6σ.
    margin: usize,
    /// Сторона клетки референсного блока.
    r_cell: usize,
    /// Сторона референсного блока в клетках.
    r_tile: usize,
    canvas_w: usize,
    canvas_h: usize,
    /// Канвас-координаты начала подкартинки референса.
    ref_off: (usize, usize),
    /// Канвас-координаты начала подкартинки полезной области.
    pay_off: (usize, usize),
    ref_size: usize,
    pay_size: usize,
}

impl Layout {
    fn new(cfg: &Cfg) -> Self {
        let sigma = cfg.sigma();
        let margin = ((6.0 * sigma).ceil() as usize).max(8);
        // Референсные клетки НАМЕРЕННО крупные: качество калибровки — не предмет
        // измерения, поэтому клетка ≥ 6σ, а сэмплируется её центральная треть.
        let r_cell = ((6.0 * sigma).ceil() as usize).max(32);
        let r_tile = cfg.ref_tile();
        let pay_block = cfg.n * cfg.w;
        let ref_block = r_tile * r_cell;
        let wide = pay_block.max(ref_block);
        let canvas_w = wide + 2 * margin;
        let canvas_h = 3 * margin + ref_block + pay_block;
        let ref_x0 = margin + (wide - ref_block) / 2;
        let pay_x0 = margin + (wide - pay_block) / 2;
        let pay_y0 = 2 * margin + ref_block;
        Layout {
            margin,
            r_cell,
            r_tile,
            canvas_w,
            canvas_h,
            ref_off: (ref_x0 - margin, 0),
            pay_off: (pay_x0 - margin, pay_y0 - margin),
            ref_size: ref_block + 2 * margin,
            pay_size: pay_block + 2 * margin,
        }
    }
}

/// Нормированная координата вдоль наклона поля, 0..1.
fn field_t(lay: &Layout, cfg: &Cfg, cx: usize, cy: usize) -> f64 {
    let (tx, ty) = if cfg.field_fixed_frame {
        let ox = 0.5 * (FIELD_FRAME_PX - lay.canvas_w as f64);
        let oy = 0.5 * (FIELD_FRAME_PX - lay.canvas_h as f64);
        (
            ((cx as f64 + ox) / (FIELD_FRAME_PX - 1.0)).clamp(0.0, 1.0),
            ((cy as f64 + oy) / (FIELD_FRAME_PX - 1.0)).clamp(0.0, 1.0),
        )
    } else {
        (
            cx as f64 / (lay.canvas_w.max(2) - 1) as f64,
            cy as f64 / (lay.canvas_h.max(2) - 1) as f64,
        )
    };
    0.5 * (tx + ty)
}

/// Какая ХРОМАТИЧЕСКАЯ мода поля наклоняется по кадру.
///
/// Пространство поканальных усилений трёхмерно: одна ахроматическая ось (общий
/// множитель) и ДВЕ хроматические. Моды выбраны так, чтобы совпасть с осями
/// обращения M2: иначе легко проверить только ту, к которой M2 неуязвим по
/// построению, и принять это за общий вывод.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChromaMode {
    /// R↔B антисимметрично, G неподвижен. Ложится ровно на ось `R−B`, то есть
    /// на МНИМУЮ ось M2; действительная ось `2G−R−B` к этой моде нечувствительна
    /// ТОЧНО (её коэффициенты при R и B равны).
    Rb,
    /// G против (R+B)/2. Ложится ровно на ось `2G−R−B`, то есть на
    /// ДЕЙСТВИТЕЛЬНУЮ ось M2 — вторая, «неудобная» для M2 мода.
    G,
}

/// Диагональный наклон поля освещённости, ПОКАНАЛЬНО.
///
/// Ахроматическая часть — как раньше: `g = LO + (HI−LO)·t`. Хроматический
/// наклон (модель ухода цвета по углу обзора ЖК) добавляется одной из двух
/// ОРТОГОНАЛЬНЫХ мод; в обеих `d` — ПОЛНЫЙ размах усиления между крайними
/// каналами по кадру, а сумма поканальных отклонений нулевая (мода ортогональна
/// ахроматической):
/// * `Rb`: `m_R = 1 + (d/2)(t−½)`, `m_G = 1`, `m_B = 1 − (d/2)(t−½)`;
/// * `G`:  `m_G = 1 + (2d/3)(t−½)`, `m_R = m_B = 1 − (d/3)(t−½)`.
///
/// При d = 0 воспроизводится прежнее ахроматическое поле ПОБИТНО.
fn illum_at(lay: &Layout, cfg: &Cfg, cx: usize, cy: usize) -> [f64; 3] {
    let t = field_t(lay, cfg, cx, cy);
    let g = ILLUM_LO + (ILLUM_HI - ILLUM_LO) * t;
    if cfg.chroma_tilt == 0.0 {
        return [g; 3];
    }
    let s = cfg.chroma_tilt * (t - 0.5);
    match cfg.chroma_mode {
        ChromaMode::Rb => [g * (1.0 + 0.5 * s), g, g * (1.0 - 0.5 * s)],
        ChromaMode::G => {
            let lo = 1.0 - s / 3.0;
            [g * lo, g * (1.0 + 2.0 * s / 3.0), g * lo]
        }
    }
}

/// drive -> линейная яркость дисплея.
#[inline]
fn lin_of_drive(d: f64, gamma: f64, quant: bool) -> f32 {
    let mut v = d.clamp(0.0, 255.0);
    if quant {
        v = v.round();
    }
    ((v / 255.0).powf(gamma)) as f32
}

/// Центральное окно доли 1/`den` внутри стороны `size`: (смещение, ширина).
fn center_window(size: usize, den: usize, min: usize) -> (usize, usize) {
    let ww = (size / den).max(min).min(size);
    ((size - ww) / 2, ww)
}

// ---------------------------------------------------------------------------
// Рендер
// ---------------------------------------------------------------------------

/// Подкартинка референсного блока в ЛИНЕЙНОЙ яркости, поле освещённости уже
/// применено (оно живёт на излучении, до оптики).
fn render_ref(cfg: &Cfg, lay: &Layout) -> Image {
    let n = lay.ref_size;
    let mut img = Image::new(n, n);
    let neutral = lin_of_drive(cfg.mapping.neutral_drive(), DISPLAY_GAMMA, cfg.quant);
    for px in img.data.iter_mut() {
        *px = [neutral; 3];
    }
    for row in 0..lay.r_tile {
        for col in 0..lay.r_tile {
            let d = ref_drives(ref_entry_at(lay.r_tile, row, col));
            let l = [
                lin_of_drive(d[0], DISPLAY_GAMMA, cfg.quant),
                lin_of_drive(d[1], DISPLAY_GAMMA, cfg.quant),
                lin_of_drive(d[2], DISPLAY_GAMMA, cfg.quant),
            ];
            let x0 = lay.margin + col * lay.r_cell;
            let y0 = lay.margin + row * lay.r_cell;
            for dy in 0..lay.r_cell {
                for dx in 0..lay.r_cell {
                    img.set(x0 + dx, y0 + dy, l);
                }
            }
        }
    }
    apply_illum(&mut img, cfg, lay, lay.ref_off);
    img
}

/// Подкартинка полезной области в ЛИНЕЙНОЙ яркости (поле освещённости внутри).
///
/// Для M4 апертура применяется в ЛИНЕЙНОЙ ЯРКОСТИ (а не в drive): канал линеен
/// именно там, и только так утверждение «блюр гауссианы = известный скаляр»
/// вообще может быть верным. Передатчик это умеет: он знает свою гамму.
fn render_payload(cfg: &Cfg, lay: &Layout, zs: &[(f64, f64)]) -> Image {
    let n = lay.pay_size;
    let w = cfg.w;
    let mut img = Image::new(n, n);
    let neutral = lin_of_drive(cfg.mapping.neutral_drive(), DISPLAY_GAMMA, cfg.quant);
    for px in img.data.iter_mut() {
        *px = [neutral; 3];
    }
    let sa = w as f64 / 4.0;
    let cen = (w - 1) as f64 / 2.0;
    for pj in 0..cfg.n {
        for pi in 0..cfg.n {
            // пилот несёт ИЗВЕСТНЫЙ символ z = 0 (нейтральный патч)
            let (x, y) = if cfg.is_pilot(pi, pj) {
                (0.0, 0.0)
            } else {
                zs[pj * cfg.n + pi]
            };
            let d = drive_for(cfg.mapping, x, y);
            let x0 = lay.margin + pi * w;
            let y0 = lay.margin + pj * w;
            if cfg.mapping.apodized() {
                // цель в линейной яркости, потом интерполяция к нейтрали
                let lt = [
                    lin_of_drive(d[0], DISPLAY_GAMMA, false),
                    lin_of_drive(d[1], DISPLAY_GAMMA, false),
                    lin_of_drive(d[2], DISPLAY_GAMMA, false),
                ];
                for dy in 0..w {
                    for dx in 0..w {
                        let rx = dx as f64 - cen;
                        let ry = dy as f64 - cen;
                        let a = (-(rx * rx + ry * ry) / (2.0 * sa * sa)).exp() as f32;
                        let mut v = [0.0f32; 3];
                        for c in 0..3 {
                            let mut l = neutral + a * (lt[c] - neutral);
                            if cfg.quant {
                                // передатчик всё равно выдаёт 8-битный drive
                                let dr = (255.0 * (l.max(0.0) as f64).powf(1.0 / DISPLAY_GAMMA))
                                    .round()
                                    .clamp(0.0, 255.0);
                                l = ((dr / 255.0).powf(DISPLAY_GAMMA)) as f32;
                            }
                            v[c] = l;
                        }
                        img.set(x0 + dx, y0 + dy, v);
                    }
                }
            } else {
                let l = [
                    lin_of_drive(d[0], DISPLAY_GAMMA, cfg.quant),
                    lin_of_drive(d[1], DISPLAY_GAMMA, cfg.quant),
                    lin_of_drive(d[2], DISPLAY_GAMMA, cfg.quant),
                ];
                for dy in 0..w {
                    for dx in 0..w {
                        img.set(x0 + dx, y0 + dy, l);
                    }
                }
            }
        }
    }
    apply_illum(&mut img, cfg, lay, lay.pay_off);
    img
}

/// Умножить линейную яркость на поле освещённости (канвас-координаты).
fn apply_illum(img: &mut Image, cfg: &Cfg, lay: &Layout, off: (usize, usize)) {
    if !cfg.illum {
        return;
    }
    for y in 0..img.h {
        for x in 0..img.w {
            let g = illum_at(lay, cfg, off.0 + x, off.1 + y);
            let p = img.at(x, y);
            img.set(
                x,
                y,
                [
                    p[0] * g[0] as f32,
                    p[1] * g[1] as f32,
                    p[2] * g[2] as f32,
                ],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Сенсор: экспозиция -> тон-кривая ISP -> шум -> клип -> 8 бит
// ---------------------------------------------------------------------------

#[inline]
fn capture_px(lin: [f32; 3], exposure: f64, cfg: &Cfg, rng: &mut Rng) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for c in 0..3 {
        let l = (lin[c] as f64 * exposure).max(0.0);
        let mut raw = l.powf(1.0 / cfg.isp[c]).min(1.0);
        if cfg.ch.noise_sigma > 0.0 {
            raw += rng.gaussian() * cfg.ch.noise_sigma;
        }
        raw = raw.clamp(0.0, 1.0);
        if cfg.quant {
            raw = (raw * 255.0).round() / 255.0;
        }
        out[c] = raw;
    }
    out
}

// ---------------------------------------------------------------------------
// Калибровка приёмника по референсному блоку
// ---------------------------------------------------------------------------

/// Оценка сенсорной модели: показатель тон-кривой на канал + 3×3 развязка
/// (кросстолк + поканальные усиления) + смещение.
///
/// ВНИМАНИЕ (расхождение с ТЗ): `psicode_core::tone::estimate_channel_gammas`
/// жёстко привязан к геометрии символа (RING/quiet/INTERIOR/cell_size), поэтому
/// здесь реализована ЛОКАЛЬНАЯ МНК-подгонка γ + белый/чёрный + кросстолк по тем
/// же 16 клеткам §3.4 — по духу та же операция, но по геометрии `probe`.
struct Calib {
    p: [f64; 3],
    minv: [[f64; 3]; 3],
    off: [f64; 3],
    /// LUT raw(0..255) -> raw^p на канал (валиден только при quant = true).
    lut: Vec<[f64; 3]>,
}

impl Calib {
    /// raw -> линейная яркость в шкале референсного блока.
    #[inline]
    fn linearize(&self, raw: [f64; 3], quant: bool) -> [f64; 3] {
        let y = if quant {
            let i0 = (raw[0] * 255.0).round().clamp(0.0, 255.0) as usize;
            let i1 = (raw[1] * 255.0).round().clamp(0.0, 255.0) as usize;
            let i2 = (raw[2] * 255.0).round().clamp(0.0, 255.0) as usize;
            [
                self.lut[i0][0] - self.off[0],
                self.lut[i1][1] - self.off[1],
                self.lut[i2][2] - self.off[2],
            ]
        } else {
            [
                raw[0].max(0.0).powf(self.p[0]) - self.off[0],
                raw[1].max(0.0).powf(self.p[1]) - self.off[1],
                raw[2].max(0.0).powf(self.p[2]) - self.off[2],
            ]
        };
        let m = &self.minv;
        [
            m[0][0] * y[0] + m[0][1] * y[1] + m[0][2] * y[2],
            m[1][0] * y[0] + m[1][1] * y[1] + m[1][2] * y[2],
            m[2][0] * y[0] + m[2][1] * y[1] + m[2][2] * y[2],
        ]
    }
}

/// Аффинная регрессия y = a·t + b по точкам; возвращает (a, b, 1 − R²).
fn affine_fit(t: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = t.len() as f64;
    let (mut st, mut sy, mut stt, mut sty) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..t.len() {
        st += t[i];
        sy += y[i];
        stt += t[i] * t[i];
        sty += t[i] * y[i];
    }
    let den = n * stt - st * st;
    if den.abs() < 1e-18 {
        return (0.0, sy / n, 1.0);
    }
    let a = (n * sty - st * sy) / den;
    let b = (sy - a * st) / n;
    let ybar = sy / n;
    let (mut sse, mut sst) = (0.0, 0.0);
    for i in 0..t.len() {
        let r = y[i] - (a * t[i] + b);
        sse += r * r;
        sst += (y[i] - ybar) * (y[i] - ybar);
    }
    // масштабно-инвариантная невязка: минимизация чистой SSE поощряла бы малые p
    let rel = if sst > 1e-18 { sse / sst } else { 1.0 };
    (a, b, rel)
}

/// Гаусс с частичным выбором главного элемента для системы n×n (n ≤ 4).
fn solve_lin(mut a: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        rhs.swap(col, piv);
        let d = a[col][col];
        for c in col..n {
            a[col][c] /= d;
        }
        rhs[col] /= d;
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    Some(rhs)
}

/// Базис полинома поля: порядок 1 — аффинный (1, x, y), порядок 2 —
/// биквадратный (1, x, y, x², xy, y²). Координаты нормированы в [-1, 1].
fn poly_basis(order: usize, x: f64, y: f64) -> Vec<f64> {
    if order >= 2 {
        vec![1.0, x, y, x * x, x * y, y * y]
    } else {
        vec![1.0, x, y]
    }
}

/// МНК-подгонка полинома поля по точкам (x, y, значение). Возвращает
/// коэффициенты в базисе `poly_basis`. `None`, если точек меньше числа
/// параметров или система вырождена.
fn fit_poly(order: usize, pts: &[(f64, f64, f64)]) -> Option<Vec<f64>> {
    let k = poly_basis(order, 0.0, 0.0).len();
    if pts.len() < k {
        return None;
    }
    let mut ata = vec![vec![0.0f64; k]; k];
    let mut atb = vec![0.0f64; k];
    for &(x, y, v) in pts {
        let b = poly_basis(order, x, y);
        for a in 0..k {
            for c in 0..k {
                ata[a][c] += b[a] * b[c];
            }
            atb[a] += b[a] * v;
        }
    }
    for a in 0..k {
        ata[a][a] += 1e-9;
    }
    solve_lin(ata, atb)
}

/// Значение подогнанного полинома в точке.
fn poly_eval(order: usize, coef: &[f64], x: f64, y: f64) -> f64 {
    poly_basis(order, x, y)
        .iter()
        .zip(coef.iter())
        .map(|(b, c)| b * c)
        .sum()
}

/// Обратная 3×3 (локальная копия: `channel::mat3_inverse` приватна).
fn inv3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-14 {
        return None;
    }
    let i = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * i,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * i,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * i,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * i,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * i,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * i,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * i,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * i,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * i,
        ],
    ])
}

/// Подогнать сенсорную модель по 16 средним raw-значениям клеток §3.4.
fn calibrate(cells: &[[f64; 3]; REF_N], quant: bool) -> Calib {
    // истинные линейные значения клеток (нормированные drive^γ_disp)
    let mut lref = [[0.0f64; 3]; REF_N];
    for i in 0..REF_N {
        let d = ref_drives(i);
        for c in 0..3 {
            lref[i][c] = (d[c] / 255.0).powf(DISPLAY_GAMMA);
        }
    }

    // --- шаг 1: показатель тон-кривой на канал по НЕЙТРАЛЬНОЙ лесенке ---
    let mut p = [DISPLAY_GAMMA; 3];
    let mut diag = [(1.0f64, 0.0f64); 3]; // запасной поканальный аффинный фит
    for c in 0..3 {
        let mut tt = Vec::new();
        let mut rr = Vec::new();
        for i in 0..REF_N {
            if ref_is_neutral(i) && cells[i][c] < SAT {
                tt.push(lref[i][c]);
                rr.push(cells[i][c]);
            }
        }
        if tt.len() < 3 {
            continue;
        }
        let eval = |g: f64| -> (f64, f64, f64) {
            let ys: Vec<f64> = rr.iter().map(|&v| v.max(0.0).powf(g)).collect();
            affine_fit(&tt, &ys)
        };
        // грубая сетка 1.0..8.0, затем уточнение ±0.06 с шагом 0.002
        let mut best = (DISPLAY_GAMMA, f64::MAX, 1.0, 0.0);
        let mut g = 1.0;
        while g <= 8.0 + 1e-9 {
            let (a, b, res) = eval(g);
            if res < best.1 {
                best = (g, res, a, b);
            }
            g += 0.05;
        }
        let g0 = best.0;
        let mut g = (g0 - 0.06).max(0.5);
        while g <= g0 + 0.06 + 1e-9 {
            let (a, b, res) = eval(g);
            if res < best.1 {
                best = (g, res, a, b);
            }
            g += 0.002;
        }
        p[c] = best.0;
        diag[c] = (best.2, best.3);
    }

    // --- шаг 2: 3×3 + смещение по ВСЕМ клеткам (цветные дают кросстолк) ---
    let mut mrows = [[0.0f64; 3]; 3];
    let mut off = [0.0f64; 3];
    let mut ok = true;
    for c in 0..3 {
        // нормальные уравнения 4×4 для регрессии y_c ~ [Lr, Lg, Lb, 1]
        let mut ata = vec![vec![0.0f64; 4]; 4];
        let mut atb = vec![0.0f64; 4];
        let mut used = 0;
        for i in 0..REF_N {
            if cells[i][c] >= SAT {
                continue; // насыщенные сэмплы выбрасываем ПОКАНАЛЬНО
            }
            let y = cells[i][c].max(0.0).powf(p[c]);
            let row = [lref[i][0], lref[i][1], lref[i][2], 1.0];
            for a in 0..4 {
                for b in 0..4 {
                    ata[a][b] += row[a] * row[b];
                }
                atb[a] += row[a] * y;
            }
            used += 1;
        }
        for a in 0..4 {
            ata[a][a] += 1e-9; // гребень против вырождения при клиппинге
        }
        match (used >= 4).then(|| solve_lin(ata, atb)).flatten() {
            Some(sol) => {
                mrows[c] = [sol[0], sol[1], sol[2]];
                off[c] = sol[3];
            }
            None => {
                ok = false;
                break;
            }
        }
    }
    let minv = if ok { inv3(&mrows) } else { None };
    let (minv, off) = match minv {
        Some(mi) => (mi, off),
        None => {
            // запасной путь: диагональ по нейтральной лесенке, без развязки
            let mut mi = [[0.0f64; 3]; 3];
            let mut o = [0.0f64; 3];
            for c in 0..3 {
                mi[c][c] = 1.0 / if diag[c].0.abs() < 1e-12 { 1e-12 } else { diag[c].0 };
                o[c] = diag[c].1;
            }
            (mi, o)
        }
    };

    let mut lut = Vec::new();
    if quant {
        lut.reserve(256);
        for v in 0..256 {
            let r = v as f64 / 255.0;
            lut.push([r.powf(p[0]), r.powf(p[1]), r.powf(p[2])]);
        }
    }
    Calib { p, minv, off, lut }
}

// ---------------------------------------------------------------------------
// Согласованный фильтр для M4
// ---------------------------------------------------------------------------

/// Шаблон гауссовой апертуры и его ИЗВЕСТНАЯ блюр-аттенюация.
///
/// Оценка амплитуды строится на ЦЕНТРИРОВАННОМ шаблоне `w0 = w − mean(w)`:
/// тогда неизвестный локальный нейтральный уровень (он же неизвестный масштаб
/// освещённости) выпадает ТОЧНО, а не приближённо.
///
/// Нормировка (расхождение с ТЗ, п. (a)): множитель `σ_a²/(σ_a²+σ_b²)` из ТЗ —
/// это аттенюация ПИКА гауссианы. Для оценки «взвешенное среднее / МНК-амплитуда»
/// правильный множитель другой: на бесконечной плоскости он равен
/// `2σ_a²/(2σ_a² + σ_b²)`. Но окно КОНЕЧНО (W×W при σ_a = W/4), поэтому здесь
/// аттенюация считается ЧИСЛЕННО — сворачиваем усечённый шаблон тем же ядром и
/// берём `A = Σ w0·(a⊛G)`. Приёмнику для этого нужны только σ_a и σ_b, обе
/// «известны» (σ_b в реальности оценивалась бы по референсу — это поблажка M4).
struct MatchedFilter {
    /// Центрированные веса, длина w·w.
    w0: Vec<f64>,
    /// Σ w0·t — нормировка амплитуды.
    a_norm: f64,
    /// Среднее размытого шаблона по окну.
    t_mean: f64,
    npx: f64,
}

impl MatchedFilter {
    fn new(w: usize, sigma: f64) -> Self {
        let sa = w as f64 / 4.0;
        let cen = (w - 1) as f64 / 2.0;
        let mut a = vec![0.0f64; w * w];
        for dy in 0..w {
            for dx in 0..w {
                let rx = dx as f64 - cen;
                let ry = dy as f64 - cen;
                a[dy * w + dx] = (-(rx * rx + ry * ry) / (2.0 * sa * sa)).exp();
            }
        }
        let abar = a.iter().sum::<f64>() / (w * w) as f64;
        let w0: Vec<f64> = a.iter().map(|v| v - abar).collect();

        // t = (усечённый шаблон) ⊛ G_σ, вычисляем численно через штатный блюр
        let t: Vec<f64> = if sigma > 0.0 {
            let pad = ((3.0 * sigma).ceil() as usize) + 2;
            let n = w + 2 * pad;
            let mut img = Image::new(n, n);
            for dy in 0..w {
                for dx in 0..w {
                    let v = a[dy * w + dx] as f32;
                    img.set(pad + dx, pad + dy, [v, v, v]);
                }
            }
            let b = blur(&img, sigma);
            let mut out = vec![0.0f64; w * w];
            for dy in 0..w {
                for dx in 0..w {
                    out[dy * w + dx] = b.at(pad + dx, pad + dy)[0] as f64;
                }
            }
            out
        } else {
            a.clone()
        };
        let a_norm = w0.iter().zip(t.iter()).map(|(x, y)| x * y).sum::<f64>();
        let t_mean = t.iter().sum::<f64>() / (w * w) as f64;
        MatchedFilter { w0, a_norm, t_mean, npx: (w * w) as f64 }
    }
}

// ---------------------------------------------------------------------------
// Один прогон конфигурации
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Acc {
    n: f64,
    sx: f64,
    sxx: f64,
    sy: f64,
    syy: f64,
}

impl Acc {
    fn push(&mut self, ex: f64, ey: f64) {
        self.n += 1.0;
        self.sx += ex;
        self.sxx += ex * ex;
        self.sy += ey;
        self.syy += ey * ey;
    }
}

#[derive(Clone, Copy)]
struct Stats {
    bias_x: f64,
    sd_x: f64,
    bias_y: f64,
    sd_y: f64,
    has_y: bool,
    max_abs: f64,
    /// Сколько (патч × попытка) вошло в статистику.
    samples: usize,
    /// Доля площади сетки, отданная под пилоты (0 для всех арм, кроме M1').
    overhead: f64,
}

impl Stats {
    fn from(acc: &Acc, has_y: bool, max_abs: f64, overhead: f64) -> Self {
        let n = acc.n.max(1.0);
        let bias_x = acc.sx / n;
        let bias_y = acc.sy / n;
        Stats {
            bias_x,
            sd_x: (acc.sxx / n - bias_x * bias_x).max(0.0).sqrt(),
            bias_y,
            sd_y: (acc.syy / n - bias_y * bias_y).max(0.0).sqrt(),
            has_y,
            max_abs,
            samples: acc.n as usize,
            overhead,
        }
    }
}

/// Итог прогона: статистика по всем полезным патчам, отдельно по ВНУТРЕННИМ
/// (полностью окружённым восемью соседями) и отдельно по краевым.
struct RunOut {
    all: Stats,
    interior: Stats,
    edge: Stats,
    /// Корреляция Пирсона ошибки e_x между СОСЕДНИМИ внутренними патчами.
    /// Ключевая проверка достоверности столбца «бит на пиксель»: метрика бит
    /// считает ошибку независимой на патч, поэтому суммировать биты по патчам
    /// законно ТОЛЬКО при ρ ≈ 0. При большом σ/W ошибка — это межсимвольная
    /// интерференция, то есть ДАННЫЕ СОСЕДЕЙ; она коррелирована, и сумма бит по
    /// патчам превращается в многократный учёт одной и той же информации.
    rho_adj: f64,
}

impl Stats {
    fn bits_x(&self) -> f64 {
        bits_of(self.sd_x)
    }
    fn bits_y(&self) -> f64 {
        if self.has_y {
            bits_of(self.sd_y)
        } else {
            0.0
        }
    }
    fn bits(&self) -> f64 {
        self.bits_x() + self.bits_y()
    }
    /// Те же биты, но по ПОЛНОЙ СКО (смещение НЕ вычтено) — сколько остаётся,
    /// если приёмник смещение выкалибровать не может.
    fn bits_rms(&self) -> f64 {
        let rx = (self.bias_x * self.bias_x + self.sd_x * self.sd_x).sqrt();
        let ry = (self.bias_y * self.bias_y + self.sd_y * self.sd_y).sqrt();
        bits_of(rx) + if self.has_y { bits_of(ry) } else { 0.0 }
    }
    /// `bits_rms`, уценённые на площадь, занятую пилотами: столько бит в
    /// среднем несёт ПОЗИЦИЯ СЕТКИ, а не только полезный патч.
    fn bits_net(&self) -> f64 {
        self.bits_rms() * (1.0 - self.overhead)
    }
    /// Бит на 1000 display-px площади сетки (по `bits_net`) — величина, которая
    /// на самом деле выбирает размер клетки.
    fn bits_per_kpx(&self, w: usize) -> f64 {
        self.bits_net() * 1000.0 / (w * w) as f64
    }
}

/// bits = log2(2 / (k·σ)), зажато в [0, 24].
fn bits_of(sd: f64) -> f64 {
    if sd <= 0.0 {
        return 24.0;
    }
    (2.0 / (K_PAM * sd)).log2().clamp(0.0, 24.0)
}

/// Равномерная точка единичного ДИСКА (r = √u, θ = 2πv — без отбраковки,
/// расход ГПСЧ фиксирован).
fn draw_disk(rng: &mut Rng) -> (f64, f64) {
    let r = rng.next_f64().sqrt();
    let th = 2.0 * core::f64::consts::PI * rng.next_f64();
    (r * th.cos(), r * th.sin())
}

fn run_config(cfg: &Cfg) -> RunOut {
    let lay = Layout::new(cfg);

    // --- референс: рендер/блюр/кросстолк не зависят от попытки -> один раз ---
    let ref_lin = render_ref(cfg, &lay);
    let mut ref_b = blur(&ref_lin, cfg.sigma());
    crosstalk(&mut ref_b, cfg.ch.crosstalk_rg, cfg.ch.crosstalk_gb);
    // Пиксели окна КАЖДОЙ записи паттерна: все копии записи складываются в один
    // список, так что их простое среднее = среднее по копиям (копий поровну).
    let (roff, rww) = center_window(lay.r_cell, 3, 4);
    let mut ref_win: Vec<Vec<[f32; 3]>> = vec![Vec::new(); REF_N];
    for row in 0..lay.r_tile {
        for col in 0..lay.r_tile {
            let e = ref_entry_at(lay.r_tile, row, col);
            let x0 = lay.margin + col * lay.r_cell + roff;
            let y0 = lay.margin + row * lay.r_cell + roff;
            for dy in 0..rww {
                for dx in 0..rww {
                    ref_win[e].push(ref_b.at(x0 + dx, y0 + dy));
                }
            }
        }
    }

    let mf = if cfg.mapping.apodized() {
        Some(MatchedFilter::new(cfg.w, cfg.sigma()))
    } else {
        None
    };
    let (poff, pww) = center_window(cfg.w, 2, 2);
    let npatch = cfg.n * cfg.n;

    let mut acc = Acc::default();
    let mut acc_in = Acc::default();
    let mut acc_ed = Acc::default();
    let mut max_abs = 0.0f64;
    let mut pair_sum = 0.0f64;
    let mut pair_n = 0.0f64;

    for t in 0..cfg.trials {
        let mut rng = Rng::new(seed_for(cfg.seed_point, t));
        // порядок расхода ГПСЧ фиксирован: экспозиция -> z -> шум референса ->
        // шум полезной области. Одинаковый seed_point у всех отображений даёт
        // им ОДНИ И ТЕ ЖЕ z и экспозицию (снижение дисперсии сравнения).
        let exposure = if cfg.exposure {
            EXP_LO + (EXP_HI - EXP_LO) * rng.next_f64()
        } else {
            1.0
        };
        let zs: Vec<(f64, f64)> = (0..npatch).map(|_| draw_disk(&mut rng)).collect();

        // --- калибровка по референсу ---
        let mut cells = [[0.0f64; 3]; REF_N];
        for i in 0..REF_N {
            let mut acc3 = [0.0f64; 3];
            for px in &ref_win[i] {
                let r = capture_px(*px, exposure, cfg, &mut rng);
                for c in 0..3 {
                    acc3[c] += r[c];
                }
            }
            let k = ref_win[i].len() as f64;
            for c in 0..3 {
                cells[i][c] = acc3[c] / k;
            }
        }
        let cal = calibrate(&cells, cfg.quant);

        // --- полезная область ---
        let pay_lin = render_payload(cfg, &lay, &zs);
        let mut pay = blur(&pay_lin, cfg.sigma());
        crosstalk(&mut pay, cfg.ch.crosstalk_rg, cfg.ch.crosstalk_gb);

        // Оценка тройки drive для КАЖДОЙ позиции сетки (включая пилотные).
        let mut dhats: Vec<[f64; 3]> = Vec::with_capacity(npatch);
        for pj in 0..cfg.n {
            for pi in 0..cfg.n {
                let x0 = lay.margin + pi * cfg.w;
                let y0 = lay.margin + pj * cfg.w;
                let dhat = match &mf {
                    // --- M4: согласованный фильтр по всему патчу ---
                    Some(f) => {
                        let mut sw = [0.0f64; 3];
                        let mut sl = [0.0f64; 3];
                        for dy in 0..cfg.w {
                            for dx in 0..cfg.w {
                                let raw =
                                    capture_px(pay.at(x0 + dx, y0 + dy), exposure, cfg, &mut rng);
                                let l = cal.linearize(raw, cfg.quant);
                                let w = f.w0[dy * cfg.w + dx];
                                for c in 0..3 {
                                    sw[c] += w * l[c];
                                    sl[c] += l[c];
                                }
                            }
                        }
                        // ΔL̂ = G·ΔL (общий неизвестный скаляр G терпим),
                        // G·L_n восстанавливаем из среднего минус вклад шаблона.
                        let mut dl = [0.0f64; 3];
                        let mut ml = [0.0f64; 3];
                        for c in 0..3 {
                            dl[c] = sw[c] / f.a_norm;
                            ml[c] = sl[c] / f.npx;
                        }
                        let gln =
                            (0..3).map(|c| ml[c] - f.t_mean * dl[c]).sum::<f64>() / 3.0;
                        let mut d = [0.0f64; 3];
                        for c in 0..3 {
                            d[c] = 255.0 * (gln + dl[c]).max(0.0).powf(1.0 / DISPLAY_GAMMA);
                        }
                        d
                    }
                    // --- M1/M2/M3: плоское среднее по внутренности патча ---
                    None => {
                        let mut sl = [0.0f64; 3];
                        for dy in 0..pww {
                            for dx in 0..pww {
                                let raw = capture_px(
                                    pay.at(x0 + poff + dx, y0 + poff + dy),
                                    exposure,
                                    cfg,
                                    &mut rng,
                                );
                                let l = cal.linearize(raw, cfg.quant);
                                for c in 0..3 {
                                    sl[c] += l[c];
                                }
                            }
                        }
                        let k = (pww * pww) as f64;
                        let mut d = [0.0f64; 3];
                        for c in 0..3 {
                            d[c] = 255.0 * (sl[c] / k).max(0.0).powf(1.0 / DISPLAY_GAMMA);
                        }
                        d
                    }
                };
                dhats.push(dhat);
            }
        }

        // --- M1': подгонка поля освещённости по пилотам и его снятие ---
        // Пилот несёт известный нейтральный drive MID, поэтому d̂_c/MID — это
        // прямо остаточный ПОКАНАЛЬНЫЙ множитель поля в точке пилота (то, что
        // калибровка по референсу снять не смогла: она знает поле только в
        // СВОЕЙ точке). Подгоняем его полиномом низкого порядка по позиции
        // патча и делим. Порядок 1 снимает и ахроматическую, и ХРОМАТИЧЕСКУЮ
        // линейную составляющую — коэффициенты подгоняются на канал независимо.
        let mut field: Option<[Vec<f64>; 3]> = None;
        if cfg.pilot_period > 0 && cfg.mapping == Mapping::M1P {
            let half = 0.5 * (cfg.n as f64 - 1.0);
            let norm = |v: usize| (v as f64 - half) / half.max(1.0);
            let mut pts: [Vec<(f64, f64, f64)>; 3] =
                [Vec::new(), Vec::new(), Vec::new()];
            for pj in 0..cfg.n {
                for pi in 0..cfg.n {
                    if !cfg.is_pilot(pi, pj) {
                        continue;
                    }
                    let d = dhats[pj * cfg.n + pi];
                    for c in 0..3 {
                        pts[c].push((norm(pi), norm(pj), d[c] / MID));
                    }
                }
            }
            let f0 = fit_poly(cfg.pilot_order, &pts[0]);
            let f1 = fit_poly(cfg.pilot_order, &pts[1]);
            let f2 = fit_poly(cfg.pilot_order, &pts[2]);
            if let (Some(a), Some(b), Some(c)) = (f0, f1, f2) {
                field = Some([a, b, c]);
            }
        }

        // --- статистика по НЕпилотным позициям ---
        let half = 0.5 * (cfg.n as f64 - 1.0);
        let norm = |v: usize| (v as f64 - half) / half.max(1.0);
        let mut exs = vec![f64::NAN; npatch];
        for pj in 0..cfg.n {
            for pi in 0..cfg.n {
                if cfg.is_pilot(pi, pj) {
                    continue;
                }
                let idx = pj * cfg.n + pi;
                let mut dhat = dhats[idx];
                if let Some(f) = &field {
                    for c in 0..3 {
                        let g = poly_eval(cfg.pilot_order, &f[c], norm(pi), norm(pj));
                        if g.abs() > 1e-6 {
                            dhat[c] /= g;
                        }
                    }
                }
                let (xh, yh) = z_from_drive(cfg.mapping, dhat);
                let (x, y) = zs[idx];
                let ex = xh - x;
                let ey = if cfg.mapping.has_im() { yh - y } else { 0.0 };
                acc.push(ex, ey);
                if pi > 0 && pi + 1 < cfg.n && pj > 0 && pj + 1 < cfg.n {
                    acc_in.push(ex, ey);
                    exs[idx] = ex;
                } else {
                    acc_ed.push(ex, ey);
                }
                max_abs = max_abs.max(ex.abs()).max(ey.abs());
            }
        }

        // корреляция ошибок соседних внутренних патчей (по обеим осям сетки)
        for pj in 0..cfg.n {
            for pi in 0..cfg.n {
                let a = exs[pj * cfg.n + pi];
                if a.is_nan() {
                    continue;
                }
                for (di, dj) in [(1usize, 0usize), (0, 1)] {
                    let (qi, qj) = (pi + di, pj + dj);
                    if qi >= cfg.n || qj >= cfg.n {
                        continue;
                    }
                    let b = exs[qj * cfg.n + qi];
                    if b.is_nan() {
                        continue;
                    }
                    pair_sum += a * b;
                    pair_n += 1.0;
                }
            }
        }
    }

    let has_y = cfg.mapping.has_im();
    let ov = cfg.pilot_overhead();
    let interior = Stats::from(&acc_in, has_y, max_abs, ov);
    let var_in = interior.sd_x * interior.sd_x;
    let rho_adj = if pair_n > 0.0 && var_in > 1e-18 {
        ((pair_sum / pair_n) - interior.bias_x * interior.bias_x) / var_in
    } else {
        0.0
    };
    RunOut {
        all: Stats::from(&acc, has_y, max_abs, ov),
        interior,
        edge: Stats::from(&acc_ed, has_y, max_abs, ov),
        rho_adj,
    }
}

// ---------------------------------------------------------------------------
// Печать
// ---------------------------------------------------------------------------

fn table_head(first: &str) {
    println!(
        "| {first} | mapping | bias_x | sd_x | bias_y | sd_y | bits_x | bits_y | bits/patch | bits_rms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
}

fn table_row(first: &str, m: Mapping, s: &Stats) {
    let (by, sy, bity) = if s.has_y {
        (
            format!("{:+.4}", s.bias_y),
            format!("{:.4}", s.sd_y),
            format!("{:.2}", s.bits_y()),
        )
    } else {
        ("n/a".to_string(), "n/a".to_string(), "0".to_string())
    };
    println!(
        "| {} | {} | {:+.4} | {:.4} | {} | {} | {:.2} | {} | {:.2} | {:.2} |",
        first,
        m.name(),
        s.bias_x,
        s.sd_x,
        by,
        sy,
        s.bits_x(),
        bity,
        s.bits(),
        s.bits_rms()
    );
}

// ---------------------------------------------------------------------------
// Проверка алгебры (sanity gate)
// ---------------------------------------------------------------------------

fn sanity_gate(n: usize) {
    println!("## 0. Sanity gate — прямое/обратное отображение");
    println!();
    println!(
        "Канал: σ=0, шум=0, поле выкл, экспозиция выкл, гаммы согласованы (2.2), \
         кросстолк ОСТАВЛЕН (6%/8%) — развязка обязана его снять точно. \
         W=64, N={n}, 20 попыток."
    );
    println!();
    println!("| mapping | quant | max |e| | sd_x | sd_y |");
    println!("|---|---|---|---|---|");
    for &q in &[false, true] {
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, 20, n);
            cfg.ch.set_blur(0.0);
            cfg.ch.noise_sigma = 0.0;
            cfg.illum = false;
            cfg.exposure = false;
            cfg.isp = GAMMAS_MATCHED;
            cfg.quant = q;
            cfg.seed_point = 1;
            let s = run_config(&cfg).all;
            println!(
                "| {} | {} | {:.2e} | {:.2e} | {} |",
                m.name(),
                if q { "8 бит" } else { "выкл" },
                s.max_abs,
                s.sd_x,
                if s.has_y {
                    format!("{:.2e}", s.sd_y)
                } else {
                    "n/a".to_string()
                }
            );
        }
    }
    println!();
    println!(
        "Строки «quant = выкл» — чистая проверка алгебры (порог ТЗ ~1e-3). \
         Строки «8 бит» — та же точка с 8-битным квантованием drive НА ПЕРЕДАЧЕ \
         и raw НА ПРИЁМЕ: это неустранимый пол, он же действует во всех развёртках."
    );
    println!();
}

// ---------------------------------------------------------------------------
// Развёртки
// ---------------------------------------------------------------------------

/// W из ТЗ плюс W = 4 — точка добавлена, чтобы ЗАСЕЧЬ обвал: на {8..128} при
/// σ = 2 обвала ещё не видно.
const WS: [usize; 6] = [4, 8, 16, 32, 64, 128];
const SIGMAS: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

fn sweep_w(trials: usize, n: usize) {
    println!("## 1. Закон σ/W — σ_blur = 2.0 px, W ∈ {{4,8,16,32,64,128}}, импейрменты ВКЛ");
    println!();
    table_head("W");
    for (i, &w) in WS.iter().enumerate() {
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, trials, n);
            cfg.w = w;
            cfg.seed_point = 1000 + i;
            let s = run_config(&cfg).all;
            table_row(&format!("{w} (σ/W={:.3})", 2.0 / w as f64), m, &s);
        }
    }
    println!();
}

fn sweep_blur(trials: usize, n: usize) {
    println!("## 2. Блюр при фиксированном W = 64, импейрменты ВКЛ");
    println!();
    table_head("σ_blur");
    for (i, &sg) in SIGMAS.iter().enumerate() {
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, trials, n);
            cfg.ch.set_blur(sg);
            cfg.seed_point = 2000 + i;
            let s = run_config(&cfg).all;
            table_row(&format!("{sg} (σ/W={:.3})", sg / 64.0), m, &s);
        }
    }
    println!();
}

fn sweep_ablation(trials: usize, n: usize) {
    println!("## 3. Абляция импейрментов — W = 64, σ_blur = 2");
    println!();
    table_head("impair");
    let combos: [(&str, bool, bool); 4] = [
        ("clean", false, false),
        ("illum", true, false),
        ("expo", false, true),
        ("both", true, true),
    ];
    for (i, &(label, il, ex)) in combos.iter().enumerate() {
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, trials, n);
            cfg.illum = il;
            cfg.exposure = ex;
            cfg.seed_point = 3000 + i;
            let s = run_config(&cfg).all;
            table_row(label, m, &s);
        }
    }
    println!();

    // Диагностика: сколько из ущерба «illum» — это НЕ отображение, а градиент
    // поля ВНУТРИ самого референсного блока (он портит подгонку γ/кросстолка
    // и бьёт по всем отображениям сразу). Сверху — штатный зеркальный блок 8×8
    // (линейное поле сокращается точно), снизу — наивный 4×4.
    println!("### 3b. Диагностика: вклад геометрии референса (только поле, W=64, σ=2)");
    println!();
    table_head("ref-блок");
    for (i, &(label, sym)) in [("8×8 зеркальный", true), ("4×4 наивный", false)]
        .iter()
        .enumerate()
    {
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, trials, n);
            cfg.illum = true;
            cfg.exposure = false;
            cfg.ref_sym = sym;
            cfg.seed_point = 3100 + i;
            let s = run_config(&cfg).all;
            table_row(label, m, &s);
        }
    }
    println!();
}

fn sweep_gamma(trials: usize, n: usize) {
    println!("## 4. Гаммы ISP — W = 64, σ_blur = 2, импейрменты ВКЛ");
    println!();
    table_head("γ_ISP");
    let sets: [(&str, [f64; 3]); 2] =
        [("Samsung 3.8/4.7/5.7", GAMMAS_SAMSUNG), ("matched 2.2", GAMMAS_MATCHED)];
    for (i, &(label, g)) in sets.iter().enumerate() {
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, trials, n);
            cfg.isp = g;
            cfg.seed_point = 4000 + i;
            let s = run_config(&cfg).all;
            table_row(label, m, &s);
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Развёртка 5 — ХРОМАТИЧЕСКОЕ поле освещённости
// ---------------------------------------------------------------------------

/// Полный размах хроматического дифференциала между крайними каналами, доля.
const CHROMA_TILTS: [f64; 5] = [0.0, 0.01, 0.03, 0.06, 0.12];

fn sweep_chromatic(trials: usize, n: usize) {
    println!("## 5. ХРОМАТИЧЕСКОЕ поле освещённости — W = 64, σ = 2, поле ВКЛ, экспозиция ВКЛ");
    println!();
    println!(
        "Наклон поля становится поканальным: `m_R = 1 + (d/2)(t−½)`, `m_G = 1`, \
         `m_B = 1 − (d/2)(t−½)`, то есть отношение усилений **R:B меняется на d по кадру** \
         (модель ухода цвета ЖК по углу обзора). d = 0 воспроизводит ахроматическое поле \
         развёртки 3 побитно. Арм **M1'** — тот же §5.1, но с решёткой пилотов внутри \
         полезной области (период 3) и аффинной подгонкой поля ПОКАНАЛЬНО; его биты уже \
         уценены на площадь пилотов."
    );
    println!();
    table_head("d (R:B)");
    for (i, &d) in CHROMA_TILTS.iter().enumerate() {
        for m in Mapping::all_with_pilots() {
            let mut cfg = Cfg::base(m, trials, if m == Mapping::M1P { 7 } else { n });
            cfg.chroma_tilt = d;
            cfg.seed_point = 5000 + i;
            let s = run_config(&cfg).all;
            table_row(&format!("{:.0}%", d * 100.0), m, &s);
        }
    }
    println!();
    println!(
        "> M1' считается на сетке 7×7 (9 пилотов из 49, накладные 18.4%), остальные армы — \
         на сетке {n}×{n}; сравнивать по `bits_rms`/`bits_net`."
    );
    println!();

    // Вторая, ОРТОГОНАЛЬНАЯ хроматическая мода. Мода R↔B выше ложится ровно на
    // мнимую ось M2, а действительная ось `2G−R−B` к ней нечувствительна ТОЧНО:
    // проверять только её значило бы проверять ту моду, к которой M2 неуязвим
    // по построению, и выдавать это за общий вывод. Мода «G против (R+B)/2»
    // бьёт именно по действительной оси M2.
    println!("### 5b. Вторая хроматическая мода: G против (R+B)/2 — бьёт по ДЕЙСТВИТЕЛЬНОЙ оси M2");
    println!();
    table_head("d (G:RB)");
    for (i, &d) in CHROMA_TILTS.iter().enumerate() {
        for m in Mapping::all_with_pilots() {
            let mut cfg = Cfg::base(m, trials, if m == Mapping::M1P { 7 } else { n });
            cfg.chroma_tilt = d;
            cfg.chroma_mode = ChromaMode::G;
            cfg.seed_point = 5100 + i;
            let s = run_config(&cfg).all;
            table_row(&format!("{:.0}%", d * 100.0), m, &s);
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Развёртка 6 — плотная сетка: только ВНУТРЕННИЕ патчи + бит на пиксель
// ---------------------------------------------------------------------------

const WS_DENSE: [usize; 9] = [2, 3, 4, 6, 8, 16, 32, 64, 128];
/// Сетка 5×5 -> внутренний блок 3×3 патчей, у каждого ВСЕ 8 соседей настоящие.
const DENSE_N: usize = 5;

fn dense_head() {
    println!(
        "| W | mapping | зона | патчей | bias_x | sd_x | bias_y | sd_y | bits | bits_rms | bits/kpx | ρ_adj |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
}

fn dense_row(w: usize, m: Mapping, zone: &str, s: &Stats, rho: Option<f64>) {
    if s.samples == 0 {
        return;
    }
    let (by, sy) = if s.has_y {
        (format!("{:+.4}", s.bias_y), format!("{:.4}", s.sd_y))
    } else {
        ("n/a".to_string(), "n/a".to_string())
    };
    let r = match rho {
        Some(v) => format!("{v:+.2}"),
        None => "—".to_string(),
    };
    println!(
        "| {} | {} | {} | {} | {:+.4} | {:.4} | {} | {} | {:.2} | {:.2} | {:.3} | {} |",
        w,
        m.name(),
        zone,
        s.samples,
        s.bias_x,
        s.sd_x,
        by,
        sy,
        s.bits(),
        s.bits_rms(),
        s.bits_per_kpx(w),
        r
    );
}

fn sweep_dense(trials: usize) {
    println!(
        "## 6. Плотная сетка {DENSE_N}×{DENSE_N} — ВНУТРЕННИЕ патчи (все 8 соседей настоящие), σ = 2, импейрменты ВКЛ"
    );
    println!();
    println!(
        "Внутренний блок — 3×3 патча с индексами 1..{}, у каждого из них восемь настоящих \
         соседей; краевые 16 патчей показаны отдельно. **Поле здесь привязано к \
         ФИКСИРОВАННОМУ кадру {FIELD_FRAME_PX:.0} px**, а не к канвасу: иначе крупные W \
         автоматически получали бы более пологий градиент на пиксель и сравнение \
         бит-на-пиксель между W было бы подтасовано (в развёртках 1–4 поле привязано к \
         канвасу, поэтому их числа с этой таблицей напрямую не сравнивать).",
        DENSE_N - 2
    );
    println!();
    println!("`bits/kpx` = бит на 1000 display-px площади ПОЗИЦИИ СЕТКИ, посчитано по `bits_rms`.");
    println!();
    dense_head();
    for (i, &w) in WS_DENSE.iter().enumerate() {
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, trials, DENSE_N);
            cfg.w = w;
            cfg.field_fixed_frame = true;
            cfg.seed_point = 6000 + i;
            let r = run_config(&cfg);
            dense_row(w, m, "interior", &r.interior, Some(r.rho_adj));
            dense_row(w, m, "edge", &r.edge, None);
        }
    }
    println!();
    println!(
        "> **ρ_adj — столбец-предохранитель.** Метрика бит считает ошибку НЕЗАВИСИМОЙ на \
         патч, поэтому складывать биты по патчам (а именно это и делает `bits/kpx`) \
         законно только при ρ_adj ≈ 0. Там, где ρ_adj заметно отличается от нуля, ошибка \
         — это межсимвольная интерференция, то есть данные соседей: она коррелирована, и \
         `bits/kpx` многократно учитывает одну и ту же информацию. Читать столбец \
         бит-на-пиксель как «мельче клетка — лучше» в этой зоне НЕЛЬЗЯ."
    );
    println!();
}

// ---------------------------------------------------------------------------
// Развёртка 7 — можно ли спасти M1 распределёнными пилотами
// ---------------------------------------------------------------------------

fn sweep_pilots(trials: usize) {
    println!("## 7. Спасают ли M1 распределённые пилоты — сетка 7×7, W = 64, σ = 2");
    println!();
    println!(
        "M1' = SPEC §5.1 + решётка известных нейтральных патчей с периодом 3 (позиции \
         i%3=0 и j%3=0): 9 пилотов из 49, **накладные расходы по площади 18.4%**. Приёмник \
         берёт d̂/128 в пилотах как остаточный ПОКАНАЛЬНЫЙ множитель поля, подгоняет его \
         полиномом порядка 1 (аффинный) или 2 (биквадратный) по позиции патча и делит. \
         Столбец `bits_net` = `bits_rms` × (1 − 0.184) — то есть биты на ПОЗИЦИЮ СЕТКИ, \
         с честно вычтенной площадью пилотов. Все армы на одной сетке 7×7, статистика по \
         внутренним патчам."
    );
    println!();
    println!(
        "| поле | mapping | bias_x | sd_x | bias_y | sd_y | bits_rms | bits_net | накладные |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let cases: [(&str, f64); 3] = [
        ("ахром. (d=0)", 0.0),
        ("хром. d=6%", 0.06),
        ("хром. d=12%", 0.12),
    ];
    // варианты решётки пилотов: (период, порядок полинома). Период 3 — плотная
    // решётка 9 пилотов; период 6 — только 4 угловых пилота (вдвое дешевле по
    // площади, но аффинный фит на пределе: 4 точки на 3 параметра).
    let pilots: [(usize, usize); 3] = [(3, 1), (3, 2), (6, 1)];
    for (i, &(label, d)) in cases.iter().enumerate() {
        for m in Mapping::all_with_pilots() {
            for &(period, order) in &pilots {
                if m != Mapping::M1P && (period, order) != (3, 1) {
                    continue; // решётка имеет смысл только для арма с пилотами
                }
                let mut cfg = Cfg::base(m, trials, 7);
                cfg.chroma_tilt = d;
                cfg.pilot_period = period;
                cfg.pilot_order = order;
                cfg.seed_point = 7000 + i;
                let s = run_config(&cfg).interior;
                let name = if m == Mapping::M1P {
                    format!("{} (период {period}, порядок {order})", m.name())
                } else {
                    m.name().to_string()
                };
                let (by, sy) = if s.has_y {
                    (format!("{:+.4}", s.bias_y), format!("{:.4}", s.sd_y))
                } else {
                    ("n/a".to_string(), "n/a".to_string())
                };
                println!(
                    "| {} | {} | {:+.4} | {:.4} | {} | {} | {:.2} | {:.2} | {:.1}% |",
                    label,
                    name,
                    s.bias_x,
                    s.sd_x,
                    by,
                    sy,
                    s.bits_rms(),
                    s.bits_net(),
                    s.overhead * 100.0
                );
            }
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Точка входа
// ---------------------------------------------------------------------------

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn arg_f64(args: &[String], key: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn arg_flag(args: &[String], key: &str, default: bool) -> bool {
    match args
        .iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
    {
        Some(v) => matches!(v.as_str(), "1" | "on" | "true" | "yes"),
        None => default,
    }
}

pub fn cmd_probe() {
    let args: Vec<String> = std::env::args().collect();
    let trials = arg_usize(&args, "--trials", 200);
    let n = arg_usize(&args, "--grid", 2).max(1);
    let t0 = Instant::now();

    let p = m2();
    let (usable, a_l, a_c) = m1_amps();
    println!("# probe — ёмкость одного крупного патча vs способ укладки z в цвет");
    println!();
    println!(
        "Уровни профиля §7.4: white = {WHITE255}, black = {BLACK255}, M = {MID}, \
         usable = {usable}. γ_disp = {DISPLAY_GAMMA}, шум σ = {:.4} полной шкалы, \
         кросстолк {}%/{}%.",
        NOISE_SIGMA,
        (XTALK_RG * 100.0) as u32,
        (XTALK_GB * 100.0) as u32
    );
    println!();
    println!("**M1** (SPEC §5.1): A_L = {a_l:.1}, A_C = {a_c:.1}.");
    println!(
        "**M2/M4** (постоянная яркость, вывод в `m2()`): u = S/3 = {:.1}, S = {:.1}, \
         b = {:.6}, c = √3·b = {:.6}, свинг drive на канал A = u·2b = {:.1} \
         (то есть каналы ходят ровно от black={BLACK255} до white={WHITE255}).",
        p.u, p.s, p.b, p.c, p.amp
    );
    println!(
        "**M3** (контроль): A_L_full = usable = {usable}, ось Im не используется."
    );
    println!();
    println!("> Обращение M2 выведено заново; формулы ТЗ содержали ошибку в множителе 3/2:");
    println!("> `x = (2G − R − B)/(2·S·b)` и `y = 3(R − B)/(2·S·c)`, причём **S берётся");
    println!("> измеренной суммой R+G+B** — иначе инвариантность к скаляру теряется.");
    println!();
    println!("### Метрика (ПРОКСИ, не пропускная способность Шеннона)");
    println!();
    println!(
        "`bits_axis = log2(2 / (k·σ_axis))`, k = {K_PAM}. Допущения: (1) ошибка на оси \
         гауссова и однородна по созвездию; (2) оси независимы; (3) k = 3.3 соответствует \
         ~1e-3 вероятности ошибки на КРАЙНИХ уровнях PAM на оси длиной 2; (4) σ_axis — \
         это СКО ПОСЛЕ вычитания смещения, поэтому систематическое смещение в биты НЕ \
         входит и печатается отдельным столбцом. Там, где |bias| > sd, реальная история \
         — в столбце bias, а не в битах."
    );
    println!();
    println!(
        "Импейрменты сверх `ChannelParams`: поле освещённости {ILLUM_LO}..{ILLUM_HI} \
         (диагональный наклон, на излучение до оптики) и ошибка экспозиции \
         U[{EXP_LO}, {EXP_HI}] (глобальный скаляр на попытку, тоже до тон-кривой). \
         Референсный блок §3.4 стоит ВЫШЕ полезной области, поэтому калибровка снимает \
         усиление в СВОЁЙ точке поля, а не в точке патчей — это и есть измеряемый эффект."
    );
    println!();
    println!("Попыток на точку: {trials}; патчей на попытку: {}.", n * n);
    println!();
    println!("### Где модель может вводить в заблуждение");
    println!();
    println!(
        "1. **Поле освещённости здесь АХРОМАТИЧНО** — один и тот же множитель на все три \
         канала. Именно это и делает M2 неуязвимым. Реальный увод ЦВЕТА по углу обзора \
         ЖК-панели (или неоднородность подсветки по спектру) бьёт по цветности НАПРЯМУЮ, \
         и здесь он НЕ измерен. Это главный непроверенный риск для вывода в пользу M2."
    );
    println!(
        "2. **Тон-кривая ISP смоделирована точной степенной функцией**, а оценщик \
         подбирает ровно показатель степени — то есть обращает её точно. Поэтому \
         развёртка 4 (гаммы) почти ничего не показывает. У настоящего ISP S-образная \
         кривая плюс локальный тон-маппинг, которые одной степенью не описываются."
    );
    println!(
        "3. **Ошибка экспозиции здесь — строго глобальный скаляр**, а его калибровка по \
         референсу снимает ТОЧНО (что развёртка 3 и показывает). Реальный автоэкспозамер \
         меняет ещё и шумовой режим, и точку клиппинга покадрово."
    );
    println!(
        "4. **Гамма дисплея взята одинаковой по каналам ({DISPLAY_GAMMA}).** Инвариантность \
         M2 держится на том, что скаляр яркости превращается в ОБЩИЙ множитель драйвов, \
         а это верно лишь при равных γ каналов. Разброс профиля §7.4 (γ_B на +0.05) даёт \
         дифференциальный множитель ~0.08% при перепаде поля 1.08 -> вклад в ŷ ~5e-4, \
         то есть пренебрежимо; но при большем разбросе γ эффект растёт."
    );
    println!(
        "5. **Метрика — прокси**, а не пропускная способность: она не знает ни про FEC, \
         ни про то, что оси могут быть коррелированы, ни про негауссовы хвосты."
    );
    println!();

    if args.iter().any(|a| a == "--single") {
        let w = arg_usize(&args, "--w", 64);
        let sg = arg_f64(&args, "--sigma", 2.0);
        let il = arg_flag(&args, "--illum", true);
        let ex = arg_flag(&args, "--exposure", true);
        let g = match args
            .iter()
            .position(|a| a == "--gammas")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
        {
            Some("matched") => GAMMAS_MATCHED,
            _ => GAMMAS_SAMSUNG,
        };
        println!("## single: W={w}, σ={sg}, illum={il}, exposure={ex}, γ={g:?}");
        println!();
        table_head("cfg");
        for m in Mapping::all() {
            let mut cfg = Cfg::base(m, trials, n);
            cfg.w = w;
            cfg.ch.set_blur(sg);
            cfg.illum = il;
            cfg.exposure = ex;
            cfg.isp = g;
            cfg.seed_point = 9000;
            let s = run_config(&cfg).all;
            table_row("single", m, &s);
        }
        println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
        return;
    }

    sanity_gate(n);
    sweep_w(trials, n);
    sweep_blur(trials, n);
    sweep_ablation(trials, n);
    sweep_gamma(trials, n);
    sweep_chromatic(trials, n);
    sweep_dense(trials);
    sweep_pilots(trials);

    println!("всего {:.2} c", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Прямое и обратное отображения обязаны быть точно обратны друг другу
    /// (алгебра, без канала) — для всех четырёх режимов.
    #[test]
    fn forward_inverse_roundtrip() {
        for m in Mapping::all() {
            for &(x, y) in &[
                (0.0, 0.0),
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (0.6, -0.8),
                (-0.5, 0.5),
            ] {
                let d = drive_for(m, x, y);
                let (xh, yh) = z_from_drive(m, d);
                assert!((xh - x).abs() < 1e-9, "{m:?} x {x} -> {xh}");
                if m.has_im() {
                    assert!((yh - y).abs() < 1e-9, "{m:?} y {y} -> {yh}");
                }
            }
        }
    }

    /// M2 держит сумму драйвов постоянной и не выходит из гамута на диске.
    #[test]
    fn m2_constant_sum_and_gamut() {
        let p = m2();
        let steps = 64;
        for i in 0..steps {
            let th = 2.0 * core::f64::consts::PI * i as f64 / steps as f64;
            for &r in &[0.0, 0.5, 1.0] {
                let (x, y) = (r * th.cos(), r * th.sin());
                let d = drive_for(Mapping::M2, x, y);
                let s = d[0] + d[1] + d[2];
                assert!((s - p.s).abs() < 1e-9, "sum {s} != {}", p.s);
                for c in 0..3 {
                    assert!(
                        d[c] >= BLACK255 - 1e-9 && d[c] <= WHITE255 + 1e-9,
                        "drive {} вне гамута",
                        d[c]
                    );
                }
            }
        }
    }

    /// M2 инвариантен к общему множителю на драйвах (то, во что превращается
    /// скаляр освещённости после обращения гаммы дисплея).
    #[test]
    fn m2_invariant_to_common_drive_scale() {
        let (x, y) = (0.4, -0.7);
        let d = drive_for(Mapping::M2, x, y);
        for &lam in &[0.7, 1.0, 1.35] {
            let ds = [d[0] * lam, d[1] * lam, d[2] * lam];
            let (xh, yh) = z_from_drive(Mapping::M2, ds);
            assert!((xh - x).abs() < 1e-9, "λ={lam} x {x} -> {xh}");
            assert!((yh - y).abs() < 1e-9, "λ={lam} y {y} -> {yh}");
        }
    }

    /// M1, наоборот, к общему множителю НЕ инвариантен по действительной оси —
    /// ровно это и есть проверяемая гипотеза.
    #[test]
    fn m1_real_axis_is_scale_sensitive() {
        let (x, y) = (0.4, -0.7);
        let d = drive_for(Mapping::M1, x, y);
        let lam = 1.1;
        let ds = [d[0] * lam, d[1] * lam, d[2] * lam];
        let (xh, _) = z_from_drive(Mapping::M1, ds);
        assert!((xh - x).abs() > 0.1, "M1 неожиданно устойчив: {x} -> {xh}");
    }

    /// Референсный паттерн: 16 клеток, 6 различных нейтральных уровней,
    /// K и W на своих местах.
    #[test]
    fn ref_pattern_shape() {
        assert_eq!(ref_drives(0), [BLACK255; 3]);
        assert_eq!(ref_drives(1), [WHITE255; 3]);
        assert_eq!(ref_drives(10), [BLACK255; 3]);
        assert_eq!(ref_drives(15), [WHITE255; 3]);
        let mut lv: Vec<f64> = (0..REF_N)
            .filter(|&i| ref_is_neutral(i))
            .map(|i| ref_drives(i)[0])
            .collect();
        lv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        lv.dedup();
        assert_eq!(lv.len(), REF_GRAY_STEPS);
    }

    /// Согласованный фильтр без блюра — несмещённая оценка амплитуды шаблона.
    #[test]
    fn matched_filter_unbiased_without_blur() {
        let w = 32;
        let f = MatchedFilter::new(w, 0.0);
        let sa = w as f64 / 4.0;
        let cen = (w - 1) as f64 / 2.0;
        let amp = 0.37;
        let base = 0.11;
        let (mut sw, mut sl) = (0.0, 0.0);
        for dy in 0..w {
            for dx in 0..w {
                let rx = dx as f64 - cen;
                let ry = dy as f64 - cen;
                let a = (-(rx * rx + ry * ry) / (2.0 * sa * sa)).exp();
                let v = base + amp * a;
                sw += f.w0[dy * w + dx] * v;
                sl += v;
            }
        }
        let dl = sw / f.a_norm;
        let ml = sl / f.npx;
        assert!((dl - amp).abs() < 1e-9, "амплитуда {dl} != {amp}");
        assert!((ml - f.t_mean * dl - base).abs() < 1e-9, "уровень восстановлен неверно");
    }

    /// Калибровка обязана точно снимать кросстолк и согласованную гамму.
    #[test]
    fn calibration_inverts_crosstalk() {
        let (xrg, xgb) = (0.06, 0.08);
        let isp = [3.8, 4.7, 5.7];
        let mix = |l: [f64; 3]| -> [f64; 3] {
            [
                (1.0 - xrg) * l[0] + xrg * l[1],
                xrg * l[0] + (1.0 - xrg - xgb) * l[1] + xgb * l[2],
                xgb * l[1] + (1.0 - xgb) * l[2],
            ]
        };
        let mut cells = [[0.0f64; 3]; REF_N];
        for i in 0..REF_N {
            let d = ref_drives(i);
            let l = [
                (d[0] / 255.0).powf(DISPLAY_GAMMA),
                (d[1] / 255.0).powf(DISPLAY_GAMMA),
                (d[2] / 255.0).powf(DISPLAY_GAMMA),
            ];
            let m = mix(l);
            for c in 0..3 {
                cells[i][c] = m[c].powf(1.0 / isp[c]);
            }
        }
        let cal = calibrate(&cells, false);
        for c in 0..3 {
            assert!((cal.p[c] - isp[c]).abs() < 0.01, "γ{c} {} != {}", cal.p[c], isp[c]);
        }
        // произвольный цвет должен восстановиться
        let d = [200.0f64, 90.0, 40.0];
        let l = [
            (d[0] / 255.0).powf(DISPLAY_GAMMA),
            (d[1] / 255.0).powf(DISPLAY_GAMMA),
            (d[2] / 255.0).powf(DISPLAY_GAMMA),
        ];
        let m = mix(l);
        let raw = [
            m[0].powf(1.0 / isp[0]),
            m[1].powf(1.0 / isp[1]),
            m[2].powf(1.0 / isp[2]),
        ];
        let back = cal.linearize(raw, false);
        for c in 0..3 {
            assert!((back[c] - l[c]).abs() < 1e-4, "канал {c}: {} != {}", back[c], l[c]);
        }
    }
}
