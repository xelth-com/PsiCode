//! Захват символа ПО КОРРЕЛЯЦИИ ЗЧ-рамки (§3.2/§8): без тихой зоны и без блоба.
//!
//! # Зачем это вместо [`crate::detect::coarse_candidates`]
//!
//! Прежний захват искал символ как ПЯТНО: карта градиентной энергии → бокс-блюр
//! → порог → двойная эрозия → связные компоненты → выпуклая оболочка →
//! min-area rect. Работало это ровно потому, что вокруг символа был ров тихой
//! зоны; на реальном экране, где рядом терминал и редактор, компонента символа
//! слипается с загромождением, и захват промахивается — отсюда обязательный
//! ручной кроп в `examples/cell_noise.rs` и `examples/swatch_diag.rs`.
//!
//! Тихая зона — костыль для детектора, который цепляется за КОНТРАСТ С ФОНОМ.
//! У нас есть ЗЧ-рамка, чей корреляционный пик локализует границу сам, и мы
//! платили полем за механизм, который и так лучше. Здесь ключ — корреляция, а не
//! пятно, поэтому тихая зона не нужна.
//!
//! # Алгоритм
//!
//! Масштаб и наклон неизвестны, и это дорого: длинная корреляция ЛОМАЕТСЯ от
//! малой ошибки масштаба (на N отсчётах ошибка δ уводит хвост на N·δ клеток,
//! то есть δ ≲ 1/N — при N = 61 это 1.6 %). Ставить лестницу масштабов с шагом
//! 1.6 % нельзя: их выйдут десятки. Поэтому лестница ГРУБАЯ, а длина растёт:
//!
//! 1. **Затравка.** Окно [`SEED_TAPS`] клеток, ЦЕНТРИРОВАННОЕ (пивот в середине
//!    окна — это вдвое ослабляет и масштабный, и угловой снос по сравнению с
//!    префиксом). Допуск по масштабу ±1/(SEED_TAPS/2), по углу — столько же в
//!    тангенсе, отсюда шаг лестницы [`SCALE_RATIO`] и шаг угла.
//!    Сканируются ДВЕ оси (θ и θ+90°) и ОБА направления обхода: этого хватает,
//!    чтобы поймать канонически-верхнюю сторону при любом из четырёх поворотов.
//! 2. **Отращивание.** Затравка растёт SEED_TAPS → 2·SEED_TAPS → N, и на каждой
//!    длине (масштаб, угол, положение) доводятся мелкой локальной сеткой.
//! 3. **Проверка.** Из стороны собирается четырёхугольник, все четыре стороны
//!    коррелируются со всеми четырьмя корнями, ориентация берётся из матрицы
//!    ([`crate::zcborder::orientation_from_matrix`]).
//!
//! Стоимость затравки НЕ зависит от разрешения кадра линейно: число стартов на
//! гипотезу — это число КЛЕТОК, укладывающихся в кадр, то есть ~(W·H)/s², и
//! мелкие масштабы стоят дороже крупных.
//!
//! # ЗЧ-последовательность — палиндром, и это меняет захват
//!
//! При нечётно-длинном соглашении `z[n] = exp(−jπ·q·n·(n+1)/N)` выполняется
//! ТОЖДЕСТВЕННО `z[N−1−n] = z[n]`: при `m = N−1−n` имеем
//! `m(m+1) = N² − 2Nn − N + n + n² ≡ n(n+1) (mod 2N)`, поскольку `2Nn ≡ 0` и
//! `N(N−1) ≡ 0` (N нечётно, значит N−1 чётно). Проверено численно для
//! N ∈ {31, 37, 61} и всех корней.
//!
//! Следствия, каждое существенное:
//!
//! * коррелировать с ОБРАЩЁННЫМ шаблоном бессмысленно — это тот же самый тест;
//! * одно совпадение на линии не говорит, в какую сторону идёт обход, а значит
//!   и с какой стороны линии лежит символ: обе гипотезы приходится нести до
//!   проверки по остальным сторонам;
//! * **поворот на 180° одной стороной не различить в принципе.** Различает его
//!   только НАЗНАЧЕНИЕ КОРНЕЙ: при повороте верх↔низ (корни 3↔4) и
//!   лево↔право (1↔2). То есть пер-осевые пары корней не украшение, а несущая
//!   конструкция — без них ориентация была бы двузначной.

use crate::zcborder::{corr_complex, side_reference, BorderSpec, Carrier, RING};
use alloc::vec::Vec;

/// Отладочный вывод стадий захвата (переменная окружения PSICODE_ACQ_DEBUG).
fn dbg_on() -> bool {
    std::env::var_os("PSICODE_ACQ_DEBUG").is_some()
}

/// Длина затравочного окна в клетках.
///
/// Компромисс: короче — шире допуск по масштабу/углу (дешевле лестница), но выше
/// ложная тревога; длиннее — наоборот. 16 даёт допуск ±12 % по масштабу и ±3.6°
/// по углу при вероятности ложной затравки ~2·10⁻³ на гипотезу.
const SEED_TAPS: usize = 16;
/// Шаг лестницы масштабов (отношение соседних). Держится под допуском затравки.
const SCALE_RATIO: f64 = 1.12;
/// Шаг сетки углов, градусы. Тоже под допуском затравки.
const ANGLE_STEP_DEG: f64 = 3.5;
/// Полудиапазон наклона в плоскости, градусы.
const TILT_MAX_DEG: f64 = 10.5;
/// Сколько затравок несём в отращивание (после подавления немаксимумов).
///
/// Верная затравка практически всегда в первой десятке по корреляции (на живых
/// снимках она и есть максимум), а каждая лишняя стоит полного отращивания с
/// доводкой по четырём сторонам. 48 -> 32 заметно срезает время захвата, не меняя
/// ни одного исхода на замеренных сценах.
const MAX_SEEDS: usize = 32;
/// Порог затравочной корреляции.
const SEED_MIN: f64 = 0.55;
/// Порог принятия итогового захвата (средняя корреляция четырёх сторон).
///
/// ИЗМЕРЕНО на живых кадрах, а не выбрано круглым числом. Распределения:
/// * символ ЕСТЬ (рамка v1, реальная сцена): 0.951..0.968;
/// * символа НЕТ (реальные экраны с Excel/терминалом): 0.206..0.268.
///
/// Порог ставится в середину зазора по логарифму: 0.55 даёт запас 2.1× снизу
/// от истинных и 2.05× сверху от мусора. Прежние 0.40 отстояли от мусора всего
/// на 0.13 и на живом телефоне были пробиты (score 0.4359 на кадре без символа).
pub const ACQUIRE_MIN: f64 = 0.55;
/// Порог для КАЖДОЙ стороны в отдельности.
///
/// Среднее по четырём сторонам — плохая одиночная статистика: «одна сильная
/// сторона + три шумовых» даёт ровно 0.25·1.0 + 0.75·0.25 ≈ 0.44, то есть
/// садится точно на старый порог. У настоящей рамки v1 стороны РОВНЫЕ
/// (0.94..0.97), у мусора минимальная сторона 0.09..0.27.
pub const SIDE_MIN: f64 = 0.55;
/// Минимальное структурное отношение полосы ([`Acquisition::strip_ratio`]).
///
/// ИЗМЕРЕНО на живых кадрах:
/// * символ ЕСТЬ (рамка v1): 8.6..13.0;
/// * символа НЕТ (реальные экраны): 1.05..1.92.
///
/// Это самый сильный из трёх признаков, и понятно почему: он проверяет ровно то,
/// что рамку ОПРЕДЕЛЯЕТ, — одну и ту же последовательность на двух соседних
/// рядах и её ОТСУТСТВИЕ на клетку наружу и на клетку вглубь. Текст и таблицы
/// такой структуры не имеют ни при каком масштабе, поэтому у них отношение ~1.
/// Порог 3.0 стоит в зазоре: 1.6× над мусором, 2.9× под истинными.
pub const STRIP_RATIO_MIN: f64 = 3.0;
/// Минимальный отрыв лучшей ориентации от второй.
pub const ORIENT_MARGIN: f64 = 0.10;
/// Глубина щупа внутри полосы, в клетках от внешнего края.
///
/// 1.0 — граница двух ОДИНАКОВЫХ рядов экструдированной полосы, то есть центр
/// двухклеточной однородной ленты: максимум удержания под расфокусом. Для
/// legacy-рамки (внутреннее кольцо — инверсия внешнего) здесь ровно серое, и
/// щуп обязан стоять на 0.5 — в центре ВНЕШНЕГО кольца.
pub const PROBE_DEPTH_STRIP: f64 = 1.0;
/// Глубина щупа для legacy-рамки v0 (центр внешнего кольца).
pub const PROBE_DEPTH_LEGACY: f64 = 0.5;

/// Поле, по которому идёт корреляция: одноканальное (яркость) или двухканальное
/// (цветность как комплексная величина — тогда `im` задано).
pub struct Field<'a> {
    pub w: usize,
    pub h: usize,
    pub re: &'a [f32],
    pub im: Option<&'a [f32]>,
}

impl Field<'_> {
    #[inline]
    fn at(&self, x: f64, y: f64) -> (f64, f64) {
        // ближайший сосед: затравке хватает (≥6 px/клетку), а это втрое дешевле
        // билинейки, и вся стоимость захвата сидит именно здесь.
        if x < 0.0 || y < 0.0 {
            return (0.0, 0.0);
        }
        let xi = x as usize;
        let yi = y as usize;
        if xi >= self.w || yi >= self.h {
            return (0.0, 0.0);
        }
        let k = yi * self.w + xi;
        (
            self.re[k] as f64,
            self.im.map_or(0.0, |p| p[k] as f64),
        )
    }

    #[inline]
    fn inside(&self, x: f64, y: f64) -> bool {
        x >= 0.0 && y >= 0.0 && x < self.w as f64 && y < self.h as f64
    }

    /// Билинейная выборка — для отращивания и проверки, где точность важнее.
    fn bilin(&self, x: f64, y: f64) -> (f64, f64) {
        let fx = (x - 0.5).clamp(0.0, (self.w - 1) as f64);
        let fy = (y - 0.5).clamp(0.0, (self.h - 1) as f64);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let g = |ix: usize, iy: usize| {
            let k = iy * self.w + ix;
            (self.re[k] as f64, self.im.map_or(0.0, |p| p[k] as f64))
        };
        let mix = |a: (f64, f64), b: (f64, f64), t: f64| {
            (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
        };
        let top = mix(g(x0, y0), g(x1, y0), tx);
        let bot = mix(g(x0, y1), g(x1, y1), tx);
        mix(top, bot, ty)
    }
}

/// Подобие «каноническая клеточная сетка → пиксели снимка»:
/// `P(cu, cv) = origin + cu·s·û + cv·s·n̂`, где n̂ — û, повёрнутый на 90° по
/// часовой (в экранных координатах с осью y вниз), то есть ВНУТРЬ символа от
/// верхней стороны.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub origin: (f64, f64),
    /// px на клетку.
    pub scale: f64,
    /// Направление обхода канонической ВЕРХНЕЙ стороны, радианы.
    pub theta: f64,
}

impl Placement {
    #[inline]
    fn axes(&self) -> ((f64, f64), (f64, f64)) {
        let (c, s) = (self.theta.cos(), self.theta.sin());
        ((c, s), (-s, c))
    }

    /// Канонические клеточные координаты → пиксели снимка.
    #[inline]
    pub fn map(&self, cu: f64, cv: f64) -> (f64, f64) {
        let (u, n) = self.axes();
        (
            self.origin.0 + self.scale * (cu * u.0 + cv * n.0),
            self.origin.1 + self.scale * (cu * u.1 + cv * n.1),
        )
    }

    /// Четыре КАНОНИЧЕСКИХ угла [tl, tr, br, bl] в пикселях снимка.
    pub fn corners(&self, n: usize) -> [(f64, f64); 4] {
        let g = n as f64;
        [
            self.map(0.0, 0.0),
            self.map(g, 0.0),
            self.map(g, g),
            self.map(0.0, g),
        ]
    }
}

/// Итог захвата.
#[derive(Debug, Clone, Copy)]
pub struct Acquisition {
    /// Углы в порядке ЧТЕНИЯ СНИМКА [tl, tr, br, bl] — как их ждёт
    /// [`crate::detect`].
    pub corners: [(f64, f64); 4],
    /// На сколько четвертей повёрнут снимок относительно канона.
    pub rotation_quadrants: u8,
    /// Средняя корреляция четырёх сторон при найденной ориентации, [0, 1].
    pub score: f64,
    /// Отрыв лучшей ориентации от второй.
    pub margin: f64,
    /// Оценка px на клетку.
    pub px_per_cell: f64,
    /// Найденная раскладка — вход для дешёвого [`track`] на следующем кадре.
    pub place: Placement,
    /// Корреляция КАЖДОЙ стороны по отдельности (в каноническом порядке).
    ///
    /// Среднее скрывает форму отказа: ложное срабатывание на тексте обычно даёт
    /// одну сильную сторону и три слабых, а настоящий символ — четыре ровных.
    pub sides: [f64; 4],
    /// СТРУКТУРНАЯ проверка полосы: во сколько раз корреляция ВНУТРИ полосы
    /// сильнее корреляции на клетку НАРУЖУ и на клетку ВГЛУБЬ от неё.
    ///
    /// Определяющее свойство рамки v1 — одна и та же последовательность на двух
    /// соседних рядах и НЕ на соседях этих рядов. Загромождение (текст, таблицы)
    /// такой структуры не имеет, поэтому отношение у него около единицы, а у
    /// настоящей рамки — заметно больше.
    pub strip_ratio: f64,
}

/// Затравка: гипотеза о положении КАНОНИЧЕСКОЙ ВЕРХНЕЙ стороны.
#[derive(Debug, Clone, Copy)]
struct Seed {
    score: f64,
    place: Placement,
}

/// Позиция клетки `i` стороны на глубине щупа, в канонических коорд. сетки.
#[inline]
fn top_probe(i: usize, depth: f64) -> (f64, f64) {
    (i as f64 + 0.5, depth)
}

/// Нормированная корреляция набора отсчётов с эталоном (комплексная, со снятием
/// среднего) — тонкая обёртка, чтобы не аллоцировать в горячем цикле.
#[inline]
fn corr_slice(got: &[(f64, f64)], want: &[(f64, f64)]) -> f64 {
    corr_complex(got, want)
}

/// Настройки захвата. Стоимость затравки ~ (число клеток в кадре) × (масштабов)
/// × (углов), поэтому сужение диапазонов — главный рычаг производительности.
#[derive(Debug, Clone, Copy)]
pub struct AcquireOpts {
    /// Диапазон поиска масштаба, px на клетку.
    pub px_per_cell: (f64, f64),
    /// Полудиапазон наклона в плоскости, градусы.
    pub tilt_max_deg: f64,
    /// Шаг сетки углов, градусы.
    pub angle_step_deg: f64,
    /// Отношение соседних масштабов лестницы.
    pub scale_ratio: f64,
    /// Глубина щупа внутри полосы, клетки от внешнего края.
    pub probe_depth: f64,
    /// Порог затравочной корреляции.
    pub seed_min: f64,
    /// Сколько потоков пустить на затравочный скан; `None` — по числу ядер.
    /// `Some(1)` принудительно возвращает однопоточный путь (для замеров).
    pub threads: Option<usize>,
    /// Приёмочный гейт: (средняя корреляция, минимальная сторона, отношение полосы).
    ///
    /// Пороги — свойство РАМКИ, а не алгоритма: у полосы v1 стороны ровные и
    /// гейт можно ставить высоко, у кольца v0 под реальным расфокусом стороны
    /// заведомо неровные (замерено 0.99 / 0.19 / 0.33 / 0.37 на одном символе),
    /// и тот же гейт отверг бы настоящий символ. Поэтому это параметр вызова.
    pub gate: (f64, f64, f64),
}

impl Default for AcquireOpts {
    fn default() -> Self {
        AcquireOpts {
            px_per_cell: DEFAULT_PX_PER_CELL,
            tilt_max_deg: TILT_MAX_DEG,
            angle_step_deg: ANGLE_STEP_DEG,
            scale_ratio: SCALE_RATIO,
            probe_depth: PROBE_DEPTH_STRIP,
            seed_min: SEED_MIN,
            threads: None,
            gate: (ACQUIRE_MIN, SIDE_MIN, STRIP_RATIO_MIN),
        }
    }
}

/// Захват символа по корреляции ЗЧ-рамки. `field` — поле корреляции (яркость
/// или цветность как комплексная величина).
pub fn acquire(spec: &BorderSpec, field: &Field, opts: &AcquireOpts) -> Option<Acquisition> {
    acquire_inner(spec, field, opts).filter(|a| accepted(a, opts.gate))
}

fn acquire_inner(spec: &BorderSpec, field: &Field, opts: &AcquireOpts) -> Option<Acquisition> {
    let seeds = seed_scan(spec, field, opts);
    // Отращивание и проверка независимы по затравкам, поэтому раскладываются по
    // потокам так же, как затравочный скан. После распараллеливания скана именно
    // эта стадия стала узким местом (у затравок разная судьба: часть отсеивается
    // сразу, часть доходит до полной 4-сторонней проверки), поэтому чередующийся
    // шаг здесь тоже важнее блочного.
    let want = opts
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |v| v.get()));
    let nthreads = want.clamp(1, MAX_THREADS).min(seeds.len().max(1));
    let best = if nthreads <= 1 {
        best_of_seeds(spec, field, opts, &seeds, 0, 1)
    } else {
        let seeds = &seeds;
        std::thread::scope(|sc| {
            let handles: Vec<_> = (0..nthreads)
                .map(|t| sc.spawn(move || best_of_seeds(spec, field, opts, seeds, t, nthreads)))
                .collect();
            let mut acc: Option<Acquisition> = None;
            for h in handles {
                if let Ok(Some(a)) = h.join() {
                    acc = Some(pick_better(acc, a));
                }
            }
            acc
        })
    };
    best
}

/// Проходит ли гипотеза приёмочный гейт (§3.2).
///
/// Гейт СОСТАВНОЙ, и это существенно. Средняя корреляция четырёх сторон одна
/// по себе плохо отделяет символ от загромождённого экрана: текст и таблицы
/// дают одну-две сильные стороны, среднее подтягивается, и порог по нему
/// приходится ставить так низко, что мимо проходит мусор (замер на живом
/// телефоне: score 0.4359 при пороге 0.40 на кадре, где символа НЕ БЫЛО).
/// Поэтому проверяются ещё две вещи, которых у мусора нет по построению:
/// РОВНОСТЬ сторон и СТРУКТУРА полосы (см. [`Acquisition::strip_ratio`]).
pub fn accepted(a: &Acquisition, gate: (f64, f64, f64)) -> bool {
    a.score >= gate.0
        && a.margin >= ORIENT_MARGIN
        && a.sides.iter().cloned().fold(f64::INFINITY, f64::min) >= gate.1
        && a.strip_ratio >= gate.2
}

/// ДИАГНОСТИКА: лучшая гипотеза БЕЗ приёмочного гейта.
///
/// Существует ради измерения ЗАЗОРА между «символ есть» и «символа нет»: порог
/// имеет смысл только как зазор, и увидеть его можно лишь глядя на отвергнутое.
pub fn acquire_best_unfiltered(
    spec: &BorderSpec,
    field: &Field,
    opts: &AcquireOpts,
) -> Option<Acquisition> {
    acquire_inner(spec, field, opts)
}

/// Лучшая гипотеза среди затравок `seeds[t], seeds[t+step], ...`.
fn best_of_seeds(
    spec: &BorderSpec,
    field: &Field,
    opts: &AcquireOpts,
    seeds: &[Seed],
    t: usize,
    step: usize,
) -> Option<Acquisition> {
    let mut best: Option<Acquisition> = None;
    let mut k = t;
    while k < seeds.len() {
        if let Some(grown) = grow(spec, field, opts.probe_depth, seeds[k]) {
            if let Some(a) = verify(spec, field, opts.probe_depth, &grown) {
                best = Some(pick_better(best, a));
            }
        }
        k += step;
    }
    best
}

/// Выбор лучшей из двух гипотез с ДЕТЕРМИНИРОВАННЫМ доразрешением ничьих:
/// иначе число потоков могло бы менять исход на равных score.
fn pick_better(cur: Option<Acquisition>, a: Acquisition) -> Acquisition {
    match cur {
        None => a,
        Some(b) => {
            if a.score > b.score || (a.score == b.score && key(&a.place) < key(&b.place)) {
                a
            } else {
                b
            }
        }
    }
}

/// СЛЕЖЕНИЕ: дешёвая доводка от раскладки предыдущего кадра.
///
/// Захват стоит сотни миллисекунд, потому что перебирает масштаб, угол и
/// положение. На соседнем кадре всё это уже известно с точностью до дрожания
/// руки, поэтому достаточно того же совместного паттерн-поиска по четырём
/// сторонам, что завершает захват ([`refine_all`]): 4 × (N−2) щупов на
/// вычисление цели, десятки вычислений — единицы миллисекунд.
///
/// `None` — доводка не удержала порог; вызывающий обязан перезахватить.
pub fn track(
    spec: &BorderSpec,
    field: &Field,
    opts: &AcquireOpts,
    prev: &Placement,
) -> Option<Acquisition> {
    // Сперва МАЛЕНЬКИЙ ИСЧЕРПЫВАЮЩИЙ поиск трансляции, потом мелкая доводка.
    //
    // Одним только жадным покоординатным спуском тут не обойтись: корреляция
    // вдоль стороны периодична с периодом в клетку, поэтому у цели есть рябь, и
    // спуск из точки прошлого кадра садится в соседний локальный максимум в
    // полклетки от истины (замерено: сдвиг (3, 2) px восстанавливался как
    // (5.0, 0.0)). Сетка ±1 клетка с шагом 0.25 накрывает межкадровое дрожание
    // целиком и выбирает ПРАВИЛЬНЫЙ бассейн — 81 вычисление цели, единицы
    // миллисекунд, всё ещё на два порядка дешевле захвата.
    let mut best = *prev;
    let mut bs = f64::MIN;
    for gy in -4i32..=4 {
        for gx in -4i32..=4 {
            let cand = Placement {
                origin: (
                    prev.origin.0 + gx as f64 * 0.25 * prev.scale,
                    prev.origin.1 + gy as f64 * 0.25 * prev.scale,
                ),
                ..*prev
            };
            let v = side_score(spec, field, opts.probe_depth, &cand);
            if v > bs {
                bs = v;
                best = cand;
            }
        }
    }
    let place = refine_all(spec, field, opts.probe_depth, best, REFINE_STEP_TRACK);
    verify(spec, field, opts.probe_depth, &place).filter(|a| accepted(a, opts.gate))
}

/// Средняя корреляция четырёх сторон при данной раскладке — цель доводки.
fn side_score(spec: &BorderSpec, field: &Field, probe_depth: f64, p: &Placement) -> f64 {
    let n = spec.n;
    let mut acc = 0.0;
    for j in 0..4 {
        let pr = side_probes(spec, j, probe_depth, n);
        let got: Vec<(f64, f64)> = pr
            .iter()
            .map(|&(cu, cv, _)| {
                let (x, y) = p.map(cu, cv);
                field.bilin(x, y)
            })
            .collect();
        let want: Vec<(f64, f64)> = pr.iter().map(|&(_, _, w)| w).collect();
        acc += corr_slice(&got, &want);
    }
    acc / 4.0
}

// ---------------------------------------------------------------------------
// 1. затравочное сканирование
// ---------------------------------------------------------------------------

fn seed_scan(spec: &BorderSpec, field: &Field, opts: &AcquireOpts) -> Vec<Seed> {
    let n = spec.n;
    let probe_depth = opts.probe_depth;
    // затравочный шаблон: окно SEED_TAPS клеток в СЕРЕДИНЕ канонической верхней
    // стороны (пивот по центру окна — вдвое меньший снос по масштабу и углу).
    let i0 = n / 2 - SEED_TAPS / 2;
    let refs = side_reference(spec, 0);
    let tmpl: Vec<(f64, f64)> = (0..SEED_TAPS)
        .map(|k| {
            let want = refs
                .iter()
                .find(|(i, _, _)| *i == i0 + k)
                .expect("окно внутри собственного диапазона стороны");
            (want.1, want.2)
        })
        .collect();

    // Перечисляем ГИПОТЕЗЫ (масштаб, угол) заранее: так их можно разложить по
    // потокам, а порядок обхода перестаёт влиять на результат.
    let mut hyps: Vec<(f64, f64)> = Vec::new();
    let mut scale = opts.px_per_cell.0;
    while scale <= opts.px_per_cell.1 * 1.0001 {
        let steps = (opts.tilt_max_deg / opts.angle_step_deg).floor() as i32;
        for ai in -steps..=steps {
            let theta0 = (ai as f64 * opts.angle_step_deg).to_radians();
            // две оси: θ и θ+90° — вместе с обоими направлениями обхода они
            // покрывают все четыре поворота снимка.
            for axis in 0..2 {
                hyps.push((scale, theta0 + axis as f64 * core::f64::consts::FRAC_PI_2));
            }
        }
        scale *= opts.scale_ratio;
    }
    let mut out = scan_hypotheses(field, &hyps, probe_depth, &tmpl, i0, n, opts);

    // подавление немаксимумов: затравки, стоящие ближе половины окна и с близким
    // масштабом, — одна и та же гипотеза.
    // Сортировка с ДЕТЕРМИНИРОВАННЫМ доразрешением: при равных score порядок
    // должен зависеть только от геометрии, иначе число потоков меняло бы, какая
    // из равных затравок переживёт подавление немаксимумов.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(key(&a.place).partial_cmp(&key(&b.place)).unwrap_or(core::cmp::Ordering::Equal))
    });
    let mut keep: Vec<Seed> = Vec::new();
    for s in out {
        // Подавление немаксимумов идёт в пространстве РАСКЛАДКИ, а не «где-то
        // рядом на картинке». Затравка — это (origin, масштаб, угол) целиком, и
        // две затравки на ОДНОЙ линии, различающиеся тем, какому индексу
        // последовательности сопоставлен отсчёт, дают origin, разнесённые на
        // целое число клеток ВДОЛЬ стороны. Это РАЗНЫЕ гипотезы: у одной лаг
        // нулевой, у другой — целая клетка, и отращивание лаг не вытянет.
        //
        // Дедупликация по «середине стороны» их сливала (радиус в полокна ≈ 8
        // клеток), и какая из них выживет, решал порядок сортировки — на
        // бесшумной сцене, где десятки затравок имеют score ровно 1.0, это
        // означало лотерею и потерю поворота на 180°.
        //
        // Направление обхода тоже входит в ключ: последовательность палиндромна,
        // поэтому две гипотезы одной линии различаются ровно им (см. док модуля).
        let dup = keep.iter().any(|k| {
            let d = ((s.place.origin.0 - k.place.origin.0).powi(2)
                + (s.place.origin.1 - k.place.origin.1).powi(2))
            .sqrt();
            let dth = (s.place.theta - k.place.theta).abs() % core::f64::consts::TAU;
            let dth = dth.min(core::f64::consts::TAU - dth);
            d < 0.75 * s.place.scale
                && (s.place.scale / k.place.scale - 1.0).abs() < 0.1
                && dth < 0.1
        });
        if !dup {
            keep.push(s);
        }
        if keep.len() >= MAX_SEEDS {
            break;
        }
    }
    if dbg_on() {
        std::eprintln!("[acq] затравок после НМП: {}", keep.len());
        for s in keep.iter().take(6) {
            std::eprintln!(
                "[acq]   seed {:.3} o=({:.1},{:.1}) s={:.2} th={:.1}°",
                s.score, s.place.origin.0, s.place.origin.1, s.place.scale,
                s.place.theta.to_degrees()
            );
        }
    }
    keep
}

/// Ключ детерминированного доразрешения ничьих (только геометрия).
fn key(p: &Placement) -> f64 {
    p.origin.0 * 4096.0 + p.origin.1 * 16.0 + p.theta
}

/// Верхняя граница числа рабочих потоков захвата.
///
/// Захват идёт на телефоне вперемешку с камерой и ISP; забирать все ядра
/// невежливо и на практике не ускоряет — упирается в память.
const MAX_THREADS: usize = 8;

/// Прогон всех гипотез (масштаб, угол). Гипотезы независимы, поэтому раскладка
/// по потокам тривиальна; чередующийся шаг (`step_by`) выравнивает нагрузку —
/// стоимость гипотезы ~1/масштаб², и подряд идущие сильно разнятся.
///
/// `psicode-core` — no_std + alloc, но модуль захвата и так собирается только
/// под фичей `std` (нужна вещественная математика), поэтому потоки здесь
/// доступны безусловно; при `available_parallelism() == 1` остаётся ровно
/// прежний однопоточный путь.
#[allow(clippy::too_many_arguments)]
fn scan_hypotheses(
    field: &Field,
    hyps: &[(f64, f64)],
    probe_depth: f64,
    tmpl: &[(f64, f64)],
    i0: usize,
    n: usize,
    opts: &AcquireOpts,
) -> Vec<Seed> {
    let want = opts
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |v| v.get()));
    let nthreads = want.clamp(1, MAX_THREADS).min(hyps.len().max(1));
    if nthreads <= 1 {
        let mut out = Vec::new();
        for &(scale, theta) in hyps {
            scan_axis(field, scale, theta, probe_depth, tmpl, i0, n, opts.seed_min, &mut out);
        }
        return out;
    }
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..nthreads)
            .map(|t| {
                sc.spawn(move || {
                    let mut local = Vec::new();
                    let mut k = t;
                    while k < hyps.len() {
                        let (scale, theta) = hyps[k];
                        scan_axis(
                            field, scale, theta, probe_depth, tmpl, i0, n, opts.seed_min,
                            &mut local,
                        );
                        k += nthreads;
                    }
                    local
                })
            })
            .collect();
        let mut out = Vec::new();
        for h in handles {
            if let Ok(v) = h.join() {
                out.extend(v);
            }
        }
        out
    })
}

#[allow(clippy::too_many_arguments)]
fn scan_axis(
    field: &Field,
    scale: f64,
    theta: f64,
    probe_depth: f64,
    tmpl: &[(f64, f64)],
    i0: usize,
    n: usize,
    seed_min: f64,
    out: &mut Vec<Seed>,
) {
    let (c, s) = (theta.cos(), theta.sin());
    let u = (c, s);
    let nv = (-s, c); // внутрь символа от верхней стороны
    // габариты кадра в координатах (вдоль û, вдоль n̂)
    let corners = [
        (0.0, 0.0),
        (field.w as f64, 0.0),
        (field.w as f64, field.h as f64),
        (0.0, field.h as f64),
    ];
    let (mut a0, mut a1, mut b0, mut b1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in corners {
        let a = p.0 * u.0 + p.1 * u.1;
        let b = p.0 * nv.0 + p.1 * nv.1;
        a0 = a0.min(a);
        a1 = a1.max(a);
        b0 = b0.min(b);
        b1 = b1.max(b);
    }
    // Шаг вдоль линии — ПОЛКЛЕТКИ (нужен по фазе), между линиями — клетка:
    // лента толщиной RING однородна по глубине, промах в полклетки безвреден.
    let da = 0.5 * scale;
    let db = scale;
    let taps = tmpl.len();
    let nlines = ((b1 - b0) / db).floor() as i64;
    let nsamp = ((a1 - a0) / da).floor() as i64;
    // окно занимает taps отсчётов с ШАГОМ 2 по массиву полуклеточных проб
    let span_idx = 2 * (taps - 1);
    if nlines <= 0 || nsamp <= span_idx as i64 {
        return;
    }

    // Шаблон со снятым средним и его норма — считаются ОДИН раз на скан.
    // Так корреляция каждого старта сводится к одному проходу без аллокаций:
    // поскольку Σ t_c = 0, то Σ (s − s̄)·conj(t_c) = Σ s·conj(t_c).
    let inv_t = 1.0 / taps as f64;
    let (mut tr, mut ti) = (0.0, 0.0);
    for &(r, i) in tmpl {
        tr += r * inv_t;
        ti += i * inv_t;
    }
    let tc: Vec<(f64, f64)> = tmpl.iter().map(|&(r, i)| (r - tr, i - ti)).collect();
    let tnorm = tc.iter().map(|&(r, i)| r * r + i * i).sum::<f64>().sqrt();
    if tnorm < 1e-12 {
        return;
    }

    // Буфер полуклеточных проб вдоль ОДНОЙ линии: собирается один раз, затем
    // окно скользит по нему с шагом 1 (полклетки) и шагом отсчётов 2. Раньше
    // каждый старт пересэмплировал свои taps точек — это было в 16 раз дороже.
    let mut buf: Vec<(f64, f64)> = Vec::with_capacity(nsamp as usize + 1);
    let mut ok: Vec<bool> = Vec::with_capacity(nsamp as usize + 1);
    for li in 0..=nlines {
        let b = b0 + li as f64 * db;
        buf.clear();
        ok.clear();
        for mi in 0..=nsamp {
            let a = a0 + mi as f64 * da;
            let (x, y) = (a * u.0 + b * nv.0, a * u.1 + b * nv.1);
            if field.inside(x, y) {
                buf.push(field.at(x, y));
                ok.push(true);
            } else {
                buf.push((0.0, 0.0));
                ok.push(false);
            }
        }
        for m in 0..=(nsamp as usize - span_idx) {
            // окно целиком в кадре?
            if !ok[m] || !ok[m + span_idx] {
                continue;
            }
            let (mut sr, mut si, mut saa) = (0.0f64, 0.0f64, 0.0f64);
            let (mut xr, mut xi) = (0.0f64, 0.0f64);
            for k in 0..taps {
                let v = buf[m + 2 * k];
                let t = tc[k];
                sr += v.0;
                si += v.1;
                saa += v.0 * v.0 + v.1 * v.1;
                // Σ s·conj(t_c)
                xr += v.0 * t.0 + v.1 * t.1;
                xi += v.1 * t.0 - v.0 * t.1;
            }
            let var = saa - (sr * sr + si * si) * inv_t;
            if var < 1e-12 {
                continue;
            }
            let cf = (xr * xr + xi * xi).sqrt() / (var.sqrt() * tnorm);
            if cf < seed_min {
                continue;
            }
            let p0 = {
                let a = a0 + m as f64 * da;
                (a * u.0 + b * nv.0, a * u.1 + b * nv.1)
            };
            // ЗЧ-последовательность — ПАЛИНДРОМ (см. док модуля), поэтому одно
            // совпадение задаёт ДВЕ равноправные гипотезы о направлении обхода,
            // а с ним и о том, с какой стороны линии лежит символ. Обе идут
            // дальше; арбитрирует их `verify` по корням остальных сторон.
            //
            // Гипотеза A: сторона идёт вдоль +û, отсчёт k — это индекс i0 + k,
            // значит точка p0 — щуп индекса i0.
            out.push(Seed {
                score: cf,
                place: place_from_probe(p0, scale, theta, i0, probe_depth),
            });
            // Гипотеза B: сторона идёт вдоль −û. Тогда отсчёт k — индекс j0 − k,
            // и совпадение с шаблоном по палиндрому даёт z[j0−k] = z[N−1−j0+k] =
            // z[i0+k], откуда j0 = N−1−i0. То есть p0 — щуп индекса N−1−i0
            // (а НЕ конец окна: смещение на N−1−2·i0−taps+1 клеток, при N=61 и
            // i0=22 это целая клетка, и ошибка здесь стоила бы всего поворота).
            out.push(Seed {
                score: cf,
                place: place_from_probe(
                    p0,
                    scale,
                    theta + core::f64::consts::PI,
                    n - 1 - i0,
                    probe_depth,
                ),
            });
        }
    }
}

/// Восстанавливает [`Placement`] из «щуп клетки `i0` стоит в точке `p`».
fn place_from_probe(
    p: (f64, f64),
    scale: f64,
    theta: f64,
    i0: usize,
    probe_depth: f64,
) -> Placement {
    let (c, s) = (theta.cos(), theta.sin());
    let u = (c, s);
    let nv = (-s, c);
    let (cu, cv) = top_probe(i0, probe_depth);
    // origin = p − scale·(cu·û + cv·n̂)
    Placement {
        origin: (
            p.0 - scale * (cu * u.0 + cv * nv.0),
            p.1 - scale * (cu * u.1 + cv * nv.1),
        ),
        scale,
        theta,
    }
}

// ---------------------------------------------------------------------------
// 2. отращивание
// ---------------------------------------------------------------------------

/// Доводит гипотезу, наращивая длину коррелируемого участка.
///
/// На каждой длине допуск по масштабу ~1/(L/2), поэтому шаг сетки берётся вдвое
/// мельче допуска, а диапазон — на весь допуск предыдущей ступени.
fn grow(spec: &BorderSpec, field: &Field, probe_depth: f64, seed: Seed) -> Option<Placement> {
    let n = spec.n;
    let refs = side_reference(spec, 0);
    let mut place = seed.place;
    let mut len = SEED_TAPS;
    loop {
        let next = (len * 2).min(n - 2);
        // окно длины `next`, центрированное на середине собственного диапазона
        let mid = refs.len() / 2;
        let lo = mid.saturating_sub(next / 2);
        let hi = (lo + next).min(refs.len());
        let win: Vec<(usize, f64, f64)> = refs[lo..hi].to_vec();
        place = refine(field, probe_depth, place, &win, next)?;
        len = next;
        if next >= n - 2 {
            break;
        }
    }
    // Совместная доводка по ЧЕТЫРЁМ сторонам — не роскошь, а необходимость.
    //
    // Экструдированная полоса толщиной RING однородна ПОПЕРЁК, поэтому
    // корреляция ОДНОЙ стороны почти не чувствует поперечного смещения в
    // пределах ±1 клетки: у пика широкая полка. Это обратная сторона той самой
    // устойчивости к расфокусу, ради которой полоса и введена. Пока подгонка
    // идёт по одной стороне, поперечная координата недоопределена, и остаточные
    // полклетки превращаются в ПРОДОЛЬНЫЙ лаг для двух перпендикулярных сторон,
    // убивая их корреляцию (замерено: верх/низ 1.000, право 0.040).
    //
    // Четыре стороны фиксируют её совместно: продольная координата одной пары —
    // это поперечная координата другой. Никаких дополнительных щупов (наружу от
    // рамки или внутрь, в payload) для этого не нужно.
    Some(refine_all(spec, field, probe_depth, place, REFINE_STEP_ACQUIRE))
}

/// Начальный шаг доводки при ЗАХВАТЕ, в клетках.
///
/// Ровно полутолщина полосы: поперечная координата после подгонки по одной
/// стороне определена лишь с точностью ±1 клетка (полоса однородна по глубине),
/// и шаг обязан этот разброс перекрывать.
const REFINE_STEP_ACQUIRE: f64 = 1.0;
/// Начальный шаг доводки при СЛЕЖЕНИИ, в клетках.
///
/// Здесь поперечная координата УЖЕ известна с прошлого кадра, и крупный шаг не
/// нужен, а ВРЕДЕН: клетка — это ровно период последовательности вдоль стороны,
/// поэтому шаг в клетку способен перескочить в соседнее выравнивание, которое
/// коррелирует почти так же хорошо. Полклетки перекрывают дрожание руки между
/// кадрами (при 10 fps это единицы пикселей) и перескочить не дают.
const REFINE_STEP_TRACK: f64 = 0.5;

/// Доводка по всем четырём сторонам: паттерн-поиск по (origin, масштаб, угол),
/// максимизирующий среднюю корреляцию сторон. `coarse` — начальный шаг в клетках.
fn refine_all(
    spec: &BorderSpec,
    field: &Field,
    probe_depth: f64,
    start: Placement,
    coarse: f64,
) -> Placement {
    let n = spec.n;
    let probes: Vec<Vec<(f64, f64, (f64, f64))>> =
        (0..4).map(|k| side_probes(spec, k, probe_depth, n)).collect();
    let eval = |p: &Placement| -> f64 {
        let mut acc = 0.0;
        for j in 0..4 {
            let got: Vec<(f64, f64)> = probes[j]
                .iter()
                .map(|&(cu, cv, _)| {
                    let (x, y) = p.map(cu, cv);
                    field.bilin(x, y)
                })
                .collect();
            let want: Vec<(f64, f64)> = probes[j].iter().map(|&(_, _, w)| w).collect();
            acc += corr_slice(&got, &want);
        }
        acc / 4.0
    };
    let mut best = start;
    let mut bs = eval(&best);
    // Лестница шагов от `coarse` вниз до сотых долей клетки.
    let ladder = [coarse, coarse * 0.5, coarse * 0.25, coarse * 0.1, coarse * 0.04];
    for &step in &ladder {
        let d = step * best.scale;
        let ds = step * 0.01; // относительный шаг масштаба
        let dt = step * 0.004; // шаг угла, радианы
        let mut guard = 0;
        loop {
            guard += 1;
            let mut improved = false;
            let cands = [
                Placement { origin: (best.origin.0 + d, best.origin.1), ..best },
                Placement { origin: (best.origin.0 - d, best.origin.1), ..best },
                Placement { origin: (best.origin.0, best.origin.1 + d), ..best },
                Placement { origin: (best.origin.0, best.origin.1 - d), ..best },
                Placement { scale: best.scale * (1.0 + ds), ..best },
                Placement { scale: best.scale * (1.0 - ds), ..best },
                Placement { theta: best.theta + dt, ..best },
                Placement { theta: best.theta - dt, ..best },
            ];
            for c in cands {
                let v = eval(&c);
                if v > bs + 1e-7 {
                    bs = v;
                    best = c;
                    improved = true;
                }
            }
            if !improved || guard >= 24 {
                break;
            }
        }
    }
    best
}

/// Локальная сетка по (масштаб, угол, положение) максимизирующая корреляцию окна.
fn refine(
    field: &Field,
    probe_depth: f64,
    start: Placement,
    win: &[(usize, f64, f64)],
    len: usize,
) -> Option<Placement> {
    let want: Vec<(f64, f64)> = win.iter().map(|&(_, re, im)| (re, im)).collect();
    let eval = |p: &Placement| -> f64 {
        let got: Vec<(f64, f64)> = win
            .iter()
            .map(|&(i, _, _)| {
                let (cu, cv) = top_probe(i, probe_depth);
                let (x, y) = p.map(cu, cv);
                field.bilin(x, y)
            })
            .collect();
        corr_slice(&got, &want)
    };
    // допуск текущей длины: снос на конце ≤ полклетки при пивоте по центру
    let tol = 1.0 / (len as f64 / 2.0);
    let mut best = start;
    let mut bs = eval(&best);
    // грубая сетка по масштабу и углу, затем спуск по положению
    for si in -2i32..=2 {
        let sc = best.scale * (1.0 + si as f64 * tol * 0.5);
        for ti in -2i32..=2 {
            let th = best.theta + ti as f64 * tol * 0.5;
            let cand = Placement {
                origin: start.origin,
                scale: sc,
                theta: th,
            };
            // положение подгоняем так, чтобы центр окна остался на месте
            let mid = win[win.len() / 2].0;
            let (cu, cv) = top_probe(mid, probe_depth);
            let anchor = start.map(cu, cv);
            let moved = cand.map(cu, cv);
            let cand = Placement {
                origin: (
                    cand.origin.0 + anchor.0 - moved.0,
                    cand.origin.1 + anchor.1 - moved.1,
                ),
                ..cand
            };
            let v = eval(&cand);
            if v > bs {
                bs = v;
                best = cand;
            }
        }
    }
    // спуск по положению (полклетки → десятая клетки)
    for &step in &[0.5f64, 0.2, 0.08] {
        let d = step * best.scale;
        let mut improved = true;
        let mut guard = 0;
        while improved && guard < 8 {
            improved = false;
            guard += 1;
            for &(dx, dy) in &[(d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)] {
                let cand = Placement {
                    origin: (best.origin.0 + dx, best.origin.1 + dy),
                    ..best
                };
                let v = eval(&cand);
                if v > bs + 1e-6 {
                    bs = v;
                    best = cand;
                    improved = true;
                }
            }
        }
    }
    Some(best)
}

// ---------------------------------------------------------------------------
// 3. проверка и ориентация
// ---------------------------------------------------------------------------

/// Щупы стороны `side` для ДОВОДКИ и ПРОВЕРКИ: позиции и ожидаемые значения.
///
/// Для экструдированной полосы читаются ОБА её ряда (глубины `probe_depth ∓ 0.5`,
/// то есть центры клеток), а не одна серединная линия. Это не про SNR:
///
/// * полоса однородна ПОПЕРЁК, поэтому корреляция одной серединной линии почти
///   не чувствует поперечного сдвига в пределах ±1 клетки — плато;
/// * из-за плато масштаб оказывался определён лишь до ~0.3 %, а это ±0.2 клетки
///   набега на 61 клетку, и вместе с поперечным сдвигом крайние строки payload
///   уезжали за границу клетки: замерено — рамка коррелирует на 1.000, а
///   payload читается с 28 % ошибок;
/// * два ряда РАЗНОСЯТ эти степени свободы: сдвиг внутрь ухудшает дальний ряд
///   (он попадает в payload), наружу — ближний (он уходит из символа).
///
/// Для рамки v0 глубина одна: там внутреннее кольцо — ИНВЕРСИЯ внешнего, и
/// второй ряд нёс бы противоположный знак.
fn side_probes(
    spec: &BorderSpec,
    side: usize,
    probe_depth: f64,
    n: usize,
) -> Vec<(f64, f64, (f64, f64))> {
    // Полоса v1 (толщина RING, оба ряда одинаковы) допускает и второй ряд, и
    // приграничные щупы. Кольцо v0 — одна клетка толщиной, и под реальным
    // расфокусом приграничный щуп попадает в размытый край: замерено, что с
    // тангенциальными смещениями захват v0 на живых снимках падает с 18/28 до
    // 8/28. Поэтому для v0 остаётся ОДИН центральный щуп, как и было.
    let strip = spec.carrier == Carrier::ComplexChroma || probe_depth >= 0.99;
    let depths: &[f64] = if strip { &[-0.5, 0.5] } else { &[0.0] };
    let tang: &[f64] = if strip { TANG } else { &[0.0] };
    let mut out = Vec::with_capacity(depths.len() * tang.len() * (n - 2));
    for (i, re, im) in side_reference(spec, side) {
        for &d in depths {
            for &t in tang {
                let (cu, cv) = side_probe_at(side, i as f64 + t, probe_depth + d, n);
                out.push((cu, cv, (re, im)));
            }
        }
    }
    out
}

/// Тангенциальные смещения щупа ВДОЛЬ стороны, в клетках.
///
/// Без них геометрия определена лишь с точностью до ПОЛКЛЕТКИ: щуп в центре
/// клетки даёт ровно её значение при любом сдвиге в пределах ±0.5, поэтому у
/// цели ПЛОСКОЕ плато, и жадный поиск останавливается где угодно на нём. На
/// реальном снимке плато сглаживает расфокус, но опираться на расфокус как на
/// источник точности нельзя: на резком снимке (крупная клетка, хорошая оптика)
/// плато возвращается, и замер это показал — рамка коррелирует на 1.000, а
/// payload читается с 28 % ошибок из-за 0.27 клетки сдвига.
///
/// Щупы у КРАЁВ клетки (±0.3) переходят в соседнюю, как только рассогласование
/// превышает 0.2 клетки, и тем сужают плато с ±0.5 до ±0.2. Тот же приём
/// использует выравниватель рамки v0 (`detect::TANG`).
const TANG: &[f64] = &[-0.3, 0.0, 0.3];

/// Как [`side_probe`], но позиция вдоль стороны — ДРОБНАЯ (в клетках от начала).
#[inline]
fn side_probe_at(side: usize, along_cells: f64, depth: f64, n: usize) -> (f64, f64) {
    let a = along_cells + 0.5;
    let g = n as f64;
    match side & 3 {
        0 => (a, depth),
        1 => (g - depth, a),
        2 => (g - a, g - depth),
        _ => (depth, g - a),
    }
}

fn verify(
    spec: &BorderSpec,
    field: &Field,
    probe_depth: f64,
    place: &Placement,
) -> Option<Acquisition> {
    let n = spec.n;
    // Матрица: сторона снимка j (в КАНОНИЧЕСКОЙ раскладке place) против эталона
    // канонической стороны k. `place` уже привязан к канонической верхней
    // стороне, поэтому «сторона снимка j» здесь — просто j-я сторона квадрата.
    let mut m = [[0.0f64; 4]; 4];
    for j in 0..4 {
        let pr = side_probes(spec, j, probe_depth, n);
        let got: Vec<(f64, f64)> = pr
            .iter()
            .map(|&(cu, cv, _)| {
                let (x, y) = place.map(cu, cv);
                field.bilin(x, y)
            })
            .collect();
        for k in 0..4 {
            // эталон берётся у СТОРОНЫ k, но на тех же щупах (та же длина)
            let kr = side_probes(spec, k, probe_depth, n);
            let want: Vec<(f64, f64)> = kr.iter().map(|&(_, _, w)| w).collect();
            m[j][k] = corr_slice(&got, &want);
        }
    }
    // `place` привязан к КАНОНИЧЕСКОЙ верхней стороне, поэтому сторона j этой
    // раскладки обязана нести эталон канонической стороны j: верная гипотеза
    // даёт r = 0, а всё прочее означает, что затравка села на чужую сторону или
    // на алиас — такую гипотезу отбрасываем, не пытаясь «починить» поворотом.
    let (r, score, margin) = crate::zcborder::orientation_from_matrix(&m);
    // Структурная проверка: та же корреляция, но щупы сдвинуты на клетку НАРУЖУ
    // (depth − 1) и на клетку ВГЛУБЬ (depth + 1). У настоящей полосы там уже не
    // её последовательность, поэтому корреляция падает.
    let mut off = 0.0f64;
    for j in 0..4 {
        // Смещение на ДВЕ клетки, не на одну: полоса занимает глубины 0..RING,
        // а щупы берутся парой `depth ∓ 0.5`, поэтому при смещении на клетку
        // половина «внешних» щупов остаётся ВНУТРИ полосы и отношение вырождается
        // (замерено: у настоящего символа 1.4, у мусора 1.06..1.54 — не
        // разделяет вовсе). При смещении на две обе пары уходят наружу.
        for d in [-2.0f64, 2.0] {
            let pr = side_probes(spec, j, probe_depth + d, n);
            let got: Vec<(f64, f64)> = pr
                .iter()
                .map(|&(cu, cv, _)| {
                    let (x, y) = place.map(cu, cv);
                    field.bilin(x, y)
                })
                .collect();
            let want: Vec<(f64, f64)> = pr.iter().map(|&(_, _, w)| w).collect();
            off += corr_slice(&got, &want);
        }
    }
    off /= 8.0;
    let strip_ratio = if off > 1e-6 { score / off } else { f64::INFINITY };
    if dbg_on() {
        std::eprintln!(
            "[acq] verify o=({:.2},{:.2}) s={:.4} th={:.3}° диагональ [{:.3} {:.3} {:.3} {:.3}] r={r} score {score:.3}",
            place.origin.0, place.origin.1,
            place.scale,
            place.theta.to_degrees(),
            m[0][0], m[1][1], m[2][2], m[3][3]
        );
    }
    if r != 0 {
        return None;
    }
    // Углы: place задаёт КАНОНИЧЕСКИЕ углы. Порядок чтения снимка — тот, где
    // первым идёт угол с минимальной суммой координат.
    let canon = place.corners(n);
    let mut tl = 0usize;
    for k in 1..4 {
        if canon[k].0 + canon[k].1 < canon[tl].0 + canon[tl].1 {
            tl = k;
        }
    }
    let mut corners = [(0.0, 0.0); 4];
    for j in 0..4 {
        corners[j] = canon[(j + tl) % 4];
    }
    // снимок повёрнут на r четвертей; согласовано с detect::finalize_detection,
    // который читает ZC_ROOTS[(side + 4 − r) % 4].
    let rot = ((4 - tl) % 4) as u8;
    let mut sides = [0.0f64; 4];
    for (j, sd) in sides.iter_mut().enumerate() {
        *sd = m[j][j];
    }
    Some(Acquisition {
        corners,
        rotation_quadrants: rot,
        score,
        margin,
        px_per_cell: place.scale,
        place: *place,
        sides,
        strip_ratio,
    })
}

/// Диапазон px/клетку по умолчанию для живого приёмника (§8).
pub const DEFAULT_PX_PER_CELL: (f64, f64) = (6.0, 20.0);

/// Толщина рамки, экспортируется для симметрии с [`crate::zcborder`].
pub const BORDER_RING: usize = RING;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zcborder::{render_cells, Carrier};
    use alloc::vec;

    struct XorShift64(u64);
    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Синтетическая сцена: символ, вкомпонованный в ЗАГРОМОЖДЁННОЕ поле, БЕЗ
    /// тихой зоны, с заданными масштабом, наклоном и поворотом.
    #[allow(clippy::too_many_arguments)]
    fn scene(
        spec: &BorderSpec,
        w: usize,
        h: usize,
        cell: f64,
        tilt_deg: f64,
        rot_q: usize,
        at: (f64, f64),
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = spec.n;
        let mut rng = XorShift64(seed | 1);
        // клеточная карта: рамка + случайный payload
        let border = render_cells(spec);
        let mut grid = vec![(0.0f64, 0.0f64); n * n];
        for (k, g) in grid.iter_mut().enumerate() {
            *g = match border[k] {
                Some(v) => v,
                None => {
                    let a = rng.unit() * core::f64::consts::TAU;
                    (a.cos(), a.sin())
                }
            };
        }
        // поворот сетки на rot_q четвертей ПО ЧАСОВОЙ
        for _ in 0..rot_q {
            let mut out = vec![(0.0f64, 0.0f64); n * n];
            for y in 0..n {
                for x in 0..n {
                    out[y * n + x] = grid[(n - 1 - x) * n + y];
                }
            }
            grid = out;
        }
        // фон: полосатое загромождение («титул», «строки редактора») + градиент
        let mut re = vec![0.0f32; w * h];
        let mut im = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let bg = 0.3 + 0.4 * (x as f64 / w as f64);
                let text = if (y / 9) % 3 == 0 && (x / 7) % 2 == 0 { 0.6 } else { -0.5 };
                re[y * w + x] = (bg + text) as f32;
                im[y * w + x] = (0.2 * (y as f64 / h as f64) - 0.1) as f32;
            }
        }
        // вкладываем символ с наклоном
        let th = tilt_deg.to_radians();
        let (c, s) = (th.cos(), th.sin());
        for y in 0..h {
            for x in 0..w {
                // обратное преобразование пикселя в клеточные координаты
                let dx = x as f64 - at.0;
                let dy = y as f64 - at.1;
                let cu = (dx * c + dy * s) / cell;
                let cv = (-dx * s + dy * c) / cell;
                if cu < 0.0 || cv < 0.0 {
                    continue;
                }
                let (iu, iv) = (cu as usize, cv as usize);
                if iu >= n || iv >= n {
                    continue;
                }
                let v = grid[iv * n + iu];
                re[y * w + x] = v.0 as f32;
                im[y * w + x] = v.1 as f32;
            }
        }
        (re, im)
    }

    /// Захват находит символ в загромождённой сцене БЕЗ тихой зоны, при всех
    /// четырёх поворотах, и восстанавливает геометрию с субклеточной точностью.
    #[test]
    fn acquires_in_clutter_without_quiet_zone() {
        let spec = BorderSpec {
            n: 61,
            roots: [3, 1, 4, 2],
            carrier: Carrier::ComplexChroma,
        };
        let (w, h) = (900usize, 700usize);
        let cell = 10.0f64;
        for rot in 0..4usize {
            let at = (120.0, 90.0);
            let (re, im) = scene(&spec, w, h, cell, 0.0, rot, at, 0xACC0_0000 + rot as u64);
            let f = Field {
                w,
                h,
                re: &re,
                im: Some(&im),
            };
            let opts = AcquireOpts {
                px_per_cell: (9.0, 11.5),
                tilt_max_deg: 3.5,
                ..Default::default()
            };
            let a = acquire(&spec, &f, &opts)
                .unwrap_or_else(|| panic!("поворот {rot}: символ не найден"));
            eprintln!(
                "[acq] поворот {rot}: score {:.3} margin {:.3} px/клетку {:.2} rot_q {}",
                a.score, a.margin, a.px_per_cell, a.rotation_quadrants
            );
            assert!(a.score > 0.8, "поворот {rot}: score {:.3}", a.score);
            assert!(
                (a.px_per_cell - cell).abs() < 0.3,
                "поворот {rot}: масштаб {:.3}",
                a.px_per_cell
            );
        }
    }

    /// Распараллеливание НЕ должно менять исход: захват на 1 и на 8 потоках
    /// обязан дать побитово ту же геометрию.
    ///
    /// Это не педантизм: затравки на чистой сцене сплошь имеют score ровно 1.0,
    /// и без детерминированного доразрешения ничьих (и без подавления
    /// немаксимумов в пространстве раскладки) число потоков решало бы, какая из
    /// равных гипотез выживет.
    #[test]
    fn threading_does_not_change_the_answer() {
        let spec = BorderSpec { n: 61, roots: [3, 1, 4, 2], carrier: Carrier::ComplexChroma };
        let (w, h) = (900usize, 700usize);
        for rot in 0..4usize {
            let (re, im) = scene(&spec, w, h, 10.0, 0.0, rot, (120.0, 90.0), 0xACC0_0000 + rot as u64);
            let f = Field { w, h, re: &re, im: Some(&im) };
            let base = AcquireOpts {
                px_per_cell: (9.0, 11.5),
                tilt_max_deg: 3.5,
                ..Default::default()
            };
            let one = acquire(&spec, &f, &AcquireOpts { threads: Some(1), ..base })
                .expect("однопоточный захват");
            let many = acquire(&spec, &f, &AcquireOpts { threads: Some(8), ..base })
                .expect("многопоточный захват");
            assert_eq!(one.rotation_quadrants, many.rotation_quadrants, "поворот {rot}");
            assert_eq!(one.score.to_bits(), many.score.to_bits(), "score, поворот {rot}");
            for k in 0..4 {
                assert_eq!(
                    one.corners[k].0.to_bits(), many.corners[k].0.to_bits(),
                    "угол {k}.x, поворот {rot}"
                );
                assert_eq!(
                    one.corners[k].1.to_bits(), many.corners[k].1.to_bits(),
                    "угол {k}.y, поворот {rot}"
                );
            }
        }
    }

    /// Слежение подхватывает раскладку прошлого кадра и обязано отдать
    /// геометрию, по которой payload ЧИТАЕТСЯ ТОЧНО.
    ///
    /// Проверяется именно декодирование, а не смещение углов: цель доводки —
    /// корреляция рамки, и требовать от неё восстановления сдвига до пикселя
    /// значило бы фиксировать в тесте поведение оптимизатора, а не свойство
    /// системы. Значимо ровно одно — читается ли кадр.
    #[test]
    fn track_yields_decodable_geometry_on_the_next_frame() {
        use crate::profile::{BorderMode, CalibProfile, ChromaMode};
        use crate::symbol::{self, PAYLOAD_COLS, PAYLOAD_ROWS};

        let p = CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: 12,
            frame_hold_periods: 6,
            luma_bits: 1,
            chroma_mode: ChromaMode::Mono,
            gamma_g_q: 28,
            gamma_r_delta_q: 8,
            gamma_b_delta_q: 10,
            white_level_q: 15,
            black_level_q: 2,
            noise_sigma_q: 12,
            mtf_limit_px: 6,
            torn_frames_q: 0,
            crosstalk_rg_q: 0,
            crosstalk_gb_q: 0,
            quiet_zone: 1,
            fec_overhead: 2,
            border: BorderMode::ExtrudedStrips,
        };
        let spec = crate::zcborder::spec_for(&p).expect("v1");
        let mut rng = XorShift64(0x0BAD_F00D_1234_5678);
        let cells: Vec<u8> = (0..PAYLOAD_COLS * PAYLOAD_ROWS)
            .map(|_| (rng.next() & 1) as u8)
            .collect();
        let frame = symbol::render_symbol(&p, &cells);
        let sz = frame.size_px;
        let cw = 1000usize;
        let opts = AcquireOpts { px_per_cell: (8.0, 17.0), ..Default::default() };

        // кадр N и кадр N+1, символ сместился на (5, 3) px (дрожание руки)
        let build = |ox: usize, oy: usize| -> Vec<f32> {
            let mut img = vec![0.0f32; cw * cw];
            for y in 0..cw {
                for x in 0..cw {
                    img[y * cw + x] =
                        if (y / 11) % 3 == 0 && (x / 8) % 2 == 0 { 0.85 } else { 0.12 };
                }
            }
            let gg = p.gamma_g() as f64;
            for y in 0..sz {
                for x in 0..sz {
                    let d = frame.rgb[y * sz + x][1] as f64 / 255.0;
                    img[(oy + y) * cw + ox + x] = d.powf(gg) as f32;
                }
            }
            img
        };
        let img0 = build(140, 130);
        let img1 = build(145, 133);

        let f0 = Field { w: cw, h: cw, re: &img0, im: None };
        let a0 = acquire(&spec, &f0, &opts).expect("захват кадра N");
        let f1 = Field { w: cw, h: cw, re: &img1, im: None };
        let a1 = track(&spec, &f1, &opts, &a0.place).expect("слежение на кадре N+1");
        assert!(a1.score > 0.9, "слежение: score {:.3}", a1.score);
        assert_eq!(a1.rotation_quadrants, 0);

        // главное: по геометрии слежения payload читается точно
        let det = crate::detect::detection_from_corners(&a1.corners, a1.rotation_quadrants, a1.score)
            .expect("невырожденная гомография");
        let map = crate::detect::frame_map(&p, &det);
        let sample = |x: f64, y: f64| -> [f32; 3] {
            let fx = (x - 0.5).clamp(0.0, (cw - 1) as f64);
            let fy = (y - 0.5).clamp(0.0, (cw - 1) as f64);
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(cw - 1), (y0 + 1).min(cw - 1));
            let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
            let g = |ix: usize, iy: usize| img1[iy * cw + ix] as f64;
            let top = g(x0, y0) + (g(x1, y0) - g(x0, y0)) * tx;
            let bot = g(x0, y1) + (g(x1, y1) - g(x0, y1)) * tx;
            let v = (top + (bot - top) * ty) as f32;
            [v, v, v]
        };
        let got = symbol::demod_symbol_local(&p, &map, &sample);
        let errs = got.iter().zip(&cells).filter(|(a, b)| a != b).count();
        eprintln!(
            "[track] a0 o=({:.2},{:.2}) s={:.4} th={:.4}°  a1 o=({:.2},{:.2}) s={:.4} th={:.4}°               истина o=(145,133) s=12  score {:.3} ошибок {errs}/{}",
            a0.place.origin.0, a0.place.origin.1, a0.place.scale, a0.place.theta.to_degrees(),
            a1.place.origin.0, a1.place.origin.1, a1.place.scale, a1.place.theta.to_degrees(),
            a1.score, cells.len()
        );
        assert_eq!(errs, 0, "по геометрии слежения payload прочитан неточно");
    }

    /// Негатив: то же загромождение БЕЗ символа не должно давать захвата.
    #[test]
    fn clutter_alone_is_not_acquired() {
        let spec = BorderSpec {
            n: 61,
            roots: [3, 1, 4, 2],
            carrier: Carrier::ComplexChroma,
        };
        let (w, h) = (700usize, 560usize);
        let mut re = vec![0.0f32; w * h];
        let mut im = vec![0.0f32; w * h];
        let mut rng = XorShift64(0x5EED_1234);
        for y in 0..h {
            for x in 0..w {
                let text = if (y / 9) % 3 == 0 && (x / 7) % 2 == 0 { 0.6 } else { -0.5 };
                re[y * w + x] = (text + 0.1 * rng.unit()) as f32;
                im[y * w + x] = (0.1 * rng.unit()) as f32;
            }
        }
        let f = Field {
            w,
            h,
            re: &re,
            im: Some(&im),
        };
        let opts = AcquireOpts {
            px_per_cell: (9.0, 11.5),
            tilt_max_deg: 3.5,
            ..Default::default()
        };
        let got = acquire(&spec, &f, &opts);
        assert!(
            got.is_none(),
            "загромождение без символа принято за захват: score {:?}",
            got.map(|a| a.score)
        );
        // И ЗАПАС до гейта, а не просто «не прошло»: порог имеет смысл только
        // как зазор. На живых кадрах без символа лучший кандидат набирает
        // 0.206..0.268 при пороге 0.55, отношение полосы 1.05..1.92 при пороге
        // 3.0. Регресс, съевший запас, провалит этот тест ДО того, как отказ
        // всплывёт на телефоне (а он всплыл: score 0.4359 при пороге 0.40).
        let raw = acquire_best_unfiltered(&spec, &f, &opts).expect("кандидат есть всегда");
        assert!(
            raw.score < 0.5 * opts.gate.0 + 0.5,
            "лучший кандидат на мусоре {:.4} подобрался к порогу {:.2}",
            raw.score,
            opts.gate.0
        );
        assert!(
            raw.strip_ratio < opts.gate.2,
            "структура полосы у мусора {:.2} дошла до порога {:.2}",
            raw.strip_ratio,
            opts.gate.2
        );
    }

    /// СКВОЗНОЙ круг: рендер символа v1 -> ИЗМЕРЕННЫЙ канал -> захват -> демод,
    /// точное восстановление payload при всех четырёх поворотах.
    ///
    /// Канал ровно тот, что снят с живого телефона (см. отчёт):
    /// шум 1.79 кода из 255, поле освещённости 0.62..0.86 по кадру,
    /// расфокус σ = 2 камерных px. Символ рендерится штатным
    /// [`crate::symbol::render_symbol`] с `BorderMode::ExtrudedStrips`, то есть
    /// БЕЗ тихой зоны, и вкладывается в загромождённое поле.
    #[test]
    fn v1_round_trip_through_measured_channel() {
        use crate::profile::{BorderMode, CalibProfile, ChromaMode};
        use crate::symbol::{self, PAYLOAD_COLS, PAYLOAD_ROWS};

        // живой профиль приёмника: 1 бит/клетку, Mono, рамка v1
        let mut p = CalibProfile {
            version: CalibProfile::VERSION,
            cell_size_px: 12,
            frame_hold_periods: 6,
            luma_bits: 1,
            chroma_mode: ChromaMode::Mono,
            gamma_g_q: 28,
            gamma_r_delta_q: 8,
            gamma_b_delta_q: 10,
            white_level_q: 15,
            black_level_q: 2,
            noise_sigma_q: 12,
            mtf_limit_px: 6,
            torn_frames_q: 0,
            crosstalk_rg_q: 0,
            crosstalk_gb_q: 0,
            quiet_zone: 1,
            fec_overhead: 2,
            border: BorderMode::ExtrudedStrips,
        };
        p.border = BorderMode::ExtrudedStrips;
        // тихой зоны в v1 нет — это и есть проверяемое свойство формата
        assert_eq!(p.quiet_zone_cells(), 0);

        let mut rng = XorShift64(0x1234_5678_9ABC_DEF0);
        let cells: Vec<u8> = (0..PAYLOAD_COLS * PAYLOAD_ROWS)
            .map(|_| (rng.next() & 1) as u8)
            .collect();
        let frame = symbol::render_symbol(&p, &cells);
        let sz = frame.size_px;
        assert_eq!(sz, 61 * 12, "полотно v1 = 61 клетка, без тихой зоны");

        // холст с загромождением, символ вложен со смещением
        let cw = 1000usize;
        let (ox, oy) = (140usize, 130usize);
        assert!(ox + sz <= cw && oy + sz <= cw);
        for rot in 0..4u8 {
            let mut img = vec![0.0f32; cw * cw];
            for y in 0..cw {
                for x in 0..cw {
                    // «терминал»: строки текста + плавный фон
                    let text = if (y / 11) % 3 == 0 && (x / 8) % 2 == 0 { 0.85 } else { 0.12 };
                    img[y * cw + x] = text as f32;
                }
            }
            // гамма дисплея -> линейная яркость, как это видит камера
            let gg = p.gamma_g() as f64;
            for y in 0..sz {
                for x in 0..sz {
                    let d = frame.rgb[y * sz + x][1] as f64 / 255.0;
                    img[(oy + y) * cw + ox + x] = d.powf(gg) as f32;
                }
            }
            // поворот холста на rot четвертей по часовой
            for _ in 0..rot {
                let mut out = vec![0.0f32; cw * cw];
                for y in 0..cw {
                    for x in 0..cw {
                        out[y * cw + x] = img[(cw - 1 - x) * cw + y];
                    }
                }
                img = out;
            }
            // поле освещённости 0.62..0.86 (блик в центре + виньетка)
            for y in 0..cw {
                for x in 0..cw {
                    let dx = (x as f64 / cw as f64) - 0.45;
                    let dy = (y as f64 / cw as f64) - 0.40;
                    let f = 0.62 + 0.24 * (-(dx * dx + dy * dy) * 4.0).exp();
                    img[y * cw + x] *= f as f32;
                }
            }
            // расфокус σ = 2 камерных px (сепарабельный гауссиан)
            let sigma = 2.0f64;
            let r = (3.0 * sigma).ceil() as isize;
            let kk: Vec<f64> = (-r..=r)
                .map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp())
                .collect();
            let ks: f64 = kk.iter().sum();
            let kk: Vec<f64> = kk.iter().map(|v| v / ks).collect();
            let cl = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
            let mut tmp = img.clone();
            for y in 0..cw {
                for x in 0..cw {
                    let mut a = 0.0;
                    for (t, &kv) in kk.iter().enumerate() {
                        a += kv * img[y * cw + cl(x as isize + t as isize - r, cw)] as f64;
                    }
                    tmp[y * cw + x] = a as f32;
                }
            }
            for y in 0..cw {
                for x in 0..cw {
                    let mut a = 0.0;
                    for (t, &kv) in kk.iter().enumerate() {
                        a += kv * tmp[cl(y as isize + t as isize - r, cw) * cw + x] as f64;
                    }
                    img[y * cw + x] = a as f32;
                }
            }
            // сенсорный шум 1.79 кода из 255
            let mut nrng = XorShift64(0x9E37_79B9_7F4A_7C15u64 ^ (rot as u64 + 1));
            for v in img.iter_mut() {
                let g = {
                    let u1 = (nrng.unit()).max(1e-12);
                    let u2 = nrng.unit();
                    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
                };
                *v = (*v as f64 + g * (1.79 / 255.0)).clamp(0.0, 1.0) as f32;
            }

            // --- захват ---
            let spec = crate::zcborder::spec_for(&p).expect("v1 -> есть spec");
            let f = Field { w: cw, h: cw, re: &img, im: None };
            let opts = AcquireOpts {
                px_per_cell: (8.0, 17.0),
                ..Default::default()
            };
            let a = acquire(&spec, &f, &opts)
                .unwrap_or_else(|| panic!("поворот {rot}: символ не найден"));
            assert_eq!(a.rotation_quadrants, rot, "поворот определён неверно");

            // --- демодуляция через мост detect ---
            let det = crate::detect::detection_from_corners(&a.corners, a.rotation_quadrants, a.score)
                .expect("невырожденная гомография");
            let map = crate::detect::frame_map(&p, &det);
            let sample = |x: f64, y: f64| -> [f32; 3] {
                let fx = (x - 0.5).clamp(0.0, (cw - 1) as f64);
                let fy = (y - 0.5).clamp(0.0, (cw - 1) as f64);
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (x1, y1) = ((x0 + 1).min(cw - 1), (y0 + 1).min(cw - 1));
                let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
                let g = |ix: usize, iy: usize| img[iy * cw + ix] as f64;
                let top = g(x0, y0) + (g(x1, y0) - g(x0, y0)) * tx;
                let bot = g(x0, y1) + (g(x1, y1) - g(x0, y1)) * tx;
                let v = (top + (bot - top) * ty) as f32;
                [v, v, v]
            };
            let got = symbol::demod_symbol_local(&p, &map, &sample);
            let errs = got.iter().zip(&cells).filter(|(a, b)| a != b).count();
            eprintln!(
                "[e2e] поворот {rot}: score {:.3} px/клетку {:.2} ошибок {errs}/{}",
                a.score,
                a.px_per_cell,
                cells.len()
            );
            assert_eq!(errs, 0, "поворот {rot}: payload восстановлен неточно");
        }
    }

    /// Диагностика: корреляция каждой стороны при ТОЧНОЙ (ground-truth) геометрии.
    /// Разделяет «сцена нарисована не так» и «алгоритм встал не туда».
    #[test]
    fn ground_truth_placement_correlates_on_all_sides() {
        let spec = BorderSpec { n: 61, roots: [3, 1, 4, 2], carrier: Carrier::ComplexChroma };
        let (w, h) = (900usize, 700usize);
        let cell = 10.0f64;
        let at = (120.0, 90.0);
        let (re, im) = scene(&spec, w, h, cell, 0.0, 0, at, 0xACC0_0000);
        let f = Field { w, h, re: &re, im: Some(&im) };
        let place = Placement { origin: at, scale: cell, theta: 0.0 };
        for depth in [0.5f64, 1.0] {
            let mut d = [0.0f64; 4];
            for j in 0..4 {
                let got: Vec<(f64, f64)> = (2..spec.n)
                    .map(|i| {
                        let (cu, cv) = side_probe_at(j, i as f64, depth, spec.n);
                        let (x, y) = place.map(cu, cv);
                        f.bilin(x, y)
                    })
                    .collect();
                let want: Vec<(f64, f64)> = side_reference(&spec, j)
                    .iter().map(|&(_, r, i2)| (r, i2)).collect();
                d[j] = corr_complex(&got, &want);
            }
            eprintln!("[diag] depth {depth}: стороны [{:.3} {:.3} {:.3} {:.3}]", d[0], d[1], d[2], d[3]);
        }
    }
}
