//! Оценка L3-кандидатов против live-grade SER (§6.2, подкоманда `l3live`).
//!
//! Живой канал даёт SER клетки 1–13% даже здоровым (замеры на статических
//! дампах: bpc=1 Mono, SER ≈ 12–15%). Текущий страйп §6.2 — 399 клеток под одним
//! CRC-16 — выживает с вероятностью (1−p)^399 ≈ 5·10⁻⁶ при p=3%: заголовок не
//! читается НИКОГДА, TransferInfo (несёт k) недоступна, поток стоит (k=0).
//!
//! Здесь количественно сравниваются кандидаты (аддитивные кодеки l3 + приёмный
//! алгоритм V3) на канале «SER клетки p, i.i.d. ЛИБО кластерами (burst 3–8)»:
//!   V1a — CRC-8 на строку (57 клеток), V1b — CRC-16 на 2 строки (114);
//!   V2a — RS(255,223) t=16 по байтам, V2b — RS(255,191) t=32;
//!   V3  — голосование повторов на приёмнике (V ∈ {3,5,9}) — БЕЗ провода;
//!   V1a+V3, V2a+V3 — комбинации.
//!
//! i.i.d. считается точными замкнутыми формами (выживание блока = (1−p)^cells;
//! RS — хвост биномиального по байтовым ошибкам; голосование — биномиальное
//! p_voted). Кластерный режим — Monte Carlo реальными кодеками l3 (детермин. сид).
//!
//! Метрики: (1) время до первого читаемого header+TI (кадров) — это разблокирует
//! k; (2) множитель полезной пропускной vs идеал (накладные · выживание · голоса).

use crate::rng::{seed_for, Rng};
use psicode_core::l3::{self, BlockLayout};

const COLS: usize = l3::PAYLOAD_COLS; // 57
const ROWS: usize = l3::PAYLOAD_ROWS; // 55
const NCELLS: usize = l3::PAYLOAD_CELLS; // 3135

/// Точки SER канала (доля клеток), совпадают с live-замерами.
const PS: [f64; 5] = [0.005, 0.01, 0.03, 0.08, 0.13];

/// Идеальная байт-ёмкость кадра (все клетки — данные, без накладных/потерь).
fn ideal_bytes(bpc: u32) -> f64 {
    (NCELLS * bpc as usize / 8) as f64
}

/// Кадров на один проход всех K символов (для перевода голосов в кадры/секунды).
/// База 94 при bpc=1 (бриф); payload/кадр ∝ bpc ⇒ кадров/проход ∝ 1/bpc.
fn frames_per_pass(bpc: u32) -> f64 {
    (94.0 / bpc as f64).round().max(1.0)
}

// ------------------------ замкнутые формы (i.i.d.) --------------------------

/// Выживание блока из `cells` клеток при SER `p`: ни одной ошибочной клетки.
fn survive(cells: usize, p: f64) -> f64 {
    (1.0 - p).powi(cells as i32)
}

/// Эффективный SER клетки после мажоритарного голосования V (нечётного) копий:
/// вероятность, что БОЛЬШИНСТВО голосов ошибочно. Для bpc=1 точно (ошибка всегда
/// = один и тот же флип); для bpc>1 — консервативная верхняя оценка (ошибочные
/// голоса редко совпадают по значению, поэтому реально ещё ниже).
fn p_voted(p: f64, v: usize) -> f64 {
    let need = v / 2 + 1; // > V/2
    let mut acc = 0.0;
    for i in need..=v {
        acc += binom_pmf(v, i, p);
    }
    acc
}

/// C(n,k) p^k (1-p)^(n-k). n мало для голосования; для RS используется
/// binom_le с устойчивой рекуррентой.
fn binom_pmf(n: usize, k: usize, p: f64) -> f64 {
    let mut c = 1.0f64;
    for i in 0..k {
        c *= (n - i) as f64 / (i + 1) as f64;
    }
    c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}

/// P(Binomial(n,p) ≤ k) устойчивой рекуррентой term_i = term_{i-1}·(n-i+1)/i·p/(1-p).
fn binom_le(n: usize, k: usize, p: f64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    let mut term = (1.0 - p).powi(n as i32); // i = 0
    let mut acc = term;
    let r = p / (1.0 - p);
    for i in 1..=k.min(n) {
        term *= (n - i + 1) as f64 / i as f64 * r;
        acc += term;
    }
    acc.min(1.0)
}

/// Пост-голосовой SER при доле систематических ошибок ρ (одинаковы КАЖДЫЙ проход).
///
/// Эмпирика соседней сессии: на СТАТИЧЕСКОМ стенде ошибки клеток систематичны
/// (тот же кадр + та же камера ⇒ идентичные ошибки на каждом снимке) — простое
/// голосование их НЕ чинит. Живой дрожащий кадр даёт субпиксельный дрейф +
/// независимый сенсорный шум ⇒ частичную декорреляцию. Модель: клетка ошибочна
/// систематически с вероятностью ρ·p (ошибочна во всех V проходах ⇒ большинство
/// ошибочно всегда), иначе — независимо с p_ind так, что маргинальный SER ≈ p.
fn p_voted_systematic(p: f64, v: usize, rho: f64) -> f64 {
    let p_sys = (rho * p).min(1.0);
    let p_ind = if p_sys >= 1.0 { 0.0 } else { (p - p_sys) / (1.0 - p_sys) };
    p_sys + (1.0 - p_sys) * p_voted(p_ind, v)
}

/// Блоки варианта как (клеток, байт данных). CUR = 8 страйпов §6.2.
fn cur_blocks(bpc: u32) -> Vec<(usize, usize)> {
    l3::STRIPE_ROWS
        .iter()
        .map(|&r| {
            let cells = r * COLS;
            (cells, (cells * bpc as usize - 16) / 8)
        })
        .collect()
}

/// Блоки V1 (per-row / per-2row) как (клеток, байт данных).
fn v1_blocks(bpc: u32, lay: &BlockLayout) -> Vec<(usize, usize)> {
    lay.blocks()
        .iter()
        .map(|&r| (r * COLS, lay.block_data_bytes(r, bpc)))
        .collect()
}

/// Клеток в ведущих блоках, покрывающих первые `need` байт (все обязаны быть
/// чистыми, чтобы прочитать заголовок [+TI]).
fn cover_cells(blocks: &[(usize, usize)], need: usize) -> usize {
    let mut acc = 0usize;
    let mut cells = 0usize;
    for &(c, d) in blocks {
        cells += c;
        acc += d;
        if acc >= need {
            break;
        }
    }
    cells
}

/// Множитель goodput для блочного варианта: Σ выживание·байты / идеал / голоса.
fn goodput_blocks(blocks: &[(usize, usize)], pe: f64, bpc: u32, votes: usize) -> f64 {
    let clean: f64 = blocks
        .iter()
        .map(|&(c, d)| survive(c, pe) * d as f64)
        .sum();
    clean / ideal_bytes(bpc) / votes as f64
}

/// (выживание header+TI, множитель goodput) для V2 RS(255, 255-nsym) при SER pe.
fn v2_metrics(bpc: u32, nsym: usize, pe: f64, votes: usize) -> (f64, f64) {
    let cells_per_byte = 8.0 / bpc as f64;
    let b_p = 1.0 - (1.0 - pe).powf(cells_per_byte); // вероятность битого байта
    let n = 255usize;
    let t = nsym / 2;
    let p_cw = binom_le(n, t, b_p); // слово декодируется
    let rate = (n - nsym) as f64 / n as f64;
    let d = (ideal_bytes(bpc) / n as f64).ceil().max(1.0) as usize; // слов/кадр
    let hdr_cw = (l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN).min(d); // слов под header+TI
    let hdr_surv = p_cw.powi(hdr_cw as i32);
    let goodput = rate * p_cw / votes as f64;
    (hdr_surv, goodput)
}

/// Описание строки таблицы для одной точки p.
struct Row {
    name: &'static str,
    /// множитель goodput
    gp: f64,
    /// ожидаемое число кадров до первого читаемого header+TI
    frames: f64,
}

/// Все варианты в точке (p, bpc). `voting` варианты используют p_voted.
fn eval_point(p: f64, bpc: u32) -> Vec<Row> {
    let fpp = frames_per_pass(bpc);
    let ti_need = l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN; // 26 байт
    let cur = cur_blocks(bpc);
    let v1a = v1_blocks(bpc, &BlockLayout::PER_ROW_CRC8);
    let v1b = v1_blocks(bpc, &BlockLayout::PER_2ROW_CRC16);

    // не-голосующие: header+TI встречается каждый 8-й кадр ⇒ E[кадров]=8/s.
    let non_vote = |hdr: f64| -> f64 {
        if hdr <= 0.0 {
            f64::INFINITY
        } else {
            8.0 / hdr
        }
    };
    // голосующие: нужно V проходов (V голосов); дальше near-certain.
    let vote_frames = |hdr_voted: f64, v: usize| -> f64 {
        if hdr_voted <= 0.0 {
            f64::INFINITY
        } else {
            fpp * ((v as f64 - 1.0) + 1.0 / hdr_voted)
        }
    };

    let mut rows = Vec::new();

    // CUR
    {
        let hdr = survive(cover_cells(&cur, ti_need), p);
        rows.push(Row { name: "CUR §6.2", gp: goodput_blocks(&cur, p, bpc, 1), frames: non_vote(hdr) });
    }
    // V1a per-row CRC-8
    {
        let hdr = survive(cover_cells(&v1a, ti_need), p);
        rows.push(Row { name: "V1a row/CRC8", gp: goodput_blocks(&v1a, p, bpc, 1), frames: non_vote(hdr) });
    }
    // V1b per-2row CRC-16
    {
        let hdr = survive(cover_cells(&v1b, ti_need), p);
        rows.push(Row { name: "V1b 2row/CRC16", gp: goodput_blocks(&v1b, p, bpc, 1), frames: non_vote(hdr) });
    }
    // V2a RS(255,223)
    {
        let (hdr, gp) = v2_metrics(bpc, 32, p, 1);
        rows.push(Row { name: "V2a RS223 t16", gp, frames: non_vote(hdr) });
    }
    // V2b RS(255,191)
    {
        let (hdr, gp) = v2_metrics(bpc, 64, p, 1);
        rows.push(Row { name: "V2b RS191 t32", gp, frames: non_vote(hdr) });
    }
    // V3 голосование поверх CUR §6.2, V ∈ {3,5,9}
    for &v in &[3usize, 5, 9] {
        let pv = p_voted(p, v);
        let hdr = survive(cover_cells(&cur, ti_need), pv);
        let name = match v { 3 => "V3 vote×3", 5 => "V3 vote×5", _ => "V3 vote×9" };
        rows.push(Row { name, gp: goodput_blocks(&cur, pv, bpc, v), frames: vote_frames(hdr, v) });
    }
    // V1a+V3 (голосование + мелкие блоки), V=5
    {
        let v = 5;
        let pv = p_voted(p, v);
        let hdr = survive(cover_cells(&v1a, ti_need), pv);
        rows.push(Row { name: "V1a+V3 ×5", gp: goodput_blocks(&v1a, pv, bpc, v), frames: vote_frames(hdr, v) });
    }
    // V2a+V3, V=5
    {
        let v = 5;
        let pv = p_voted(p, v);
        let (hdr, gp) = v2_metrics(bpc, 32, pv, v);
        rows.push(Row { name: "V2a+V3 ×5", gp, frames: vote_frames(hdr, v) });
    }
    rows
}

/// Формат числа кадров (∞ / экспонента / целое).
fn fmt_frames(f: f64) -> String {
    if !f.is_finite() || f > 1e12 {
        "∞".to_string()
    } else if f >= 1e5 {
        format!("{:.0e}", f)
    } else if f >= 100.0 {
        format!("{:.0}", f)
    } else {
        format!("{:.1}", f)
    }
}

fn fmt_secs(frames: f64) -> String {
    if !frames.is_finite() || frames > 1e12 {
        "∞".to_string()
    } else {
        let s = frames / 10.0; // 10 fps
        if s >= 3600.0 {
            format!("{:.1}ч", s / 3600.0)
        } else if s >= 60.0 {
            format!("{:.1}м", s / 60.0)
        } else {
            format!("{:.1}с", s)
        }
    }
}

fn fmt_gp(g: f64) -> String {
    if g <= 0.0 || g < 1e-6 {
        "~0".to_string()
    } else {
        format!("{:.3}", g)
    }
}

fn print_analytic(bpc: u32) {
    let names = eval_point(PS[0], bpc)
        .iter()
        .map(|r| r.name)
        .collect::<Vec<_>>();
    let cols: Vec<Vec<Row>> = PS.iter().map(|&p| eval_point(p, bpc)).collect();

    println!("\n## Аналитика i.i.d., bpc={bpc} (идеал {} байт/кадр, кадров/проход {})",
        ideal_bytes(bpc) as usize, frames_per_pass(bpc) as usize);

    // A1: множитель goodput
    println!("\n### множитель goodput (payload_frac · выживание · 1/голоса)");
    print!("| вариант |");
    for &p in &PS { print!(" {:.1}% |", p * 100.0); }
    println!();
    print!("|---|"); for _ in &PS { print!("---|"); } println!();
    for (ri, name) in names.iter().enumerate() {
        print!("| {name} |");
        for c in &cols { print!(" {} |", fmt_gp(c[ri].gp)); }
        println!();
    }

    // A2: время до первого header+TI (кадры)
    println!("\n### E[кадров до первого читаемого header+TI]  (=разблокировка k)");
    print!("| вариант |");
    for &p in &PS { print!(" {:.1}% |", p * 100.0); }
    println!();
    print!("|---|"); for _ in &PS { print!("---|"); } println!();
    for (ri, name) in names.iter().enumerate() {
        print!("| {name} |");
        for c in &cols { print!(" {} |", fmt_frames(c[ri].frames)); }
        println!();
    }
    // A2': то же в секундах @10fps на p=3% и 8%
    println!("\n### то же — секунды @10fps (p=3% / p=8%)");
    print!("| вариант | 3% | 8% |\n|---|---|---|\n");
    let i3 = PS.iter().position(|&p| (p - 0.03).abs() < 1e-9).unwrap();
    let i8 = PS.iter().position(|&p| (p - 0.08).abs() < 1e-9).unwrap();
    for (ri, name) in names.iter().enumerate() {
        println!("| {name} | {} | {} |", fmt_secs(cols[i3][ri].frames), fmt_secs(cols[i8][ri].frames));
    }
}

// ------------------------------ Monte Carlo ---------------------------------

/// i.i.d. порча клеток: каждая клетка с вероятностью p меняется на ДРУГОЕ значение.
fn corrupt_iid(cells: &mut [u8], p: f64, bpc: u32, rng: &mut Rng) {
    let alpha = 1u32 << bpc;
    for c in cells.iter_mut() {
        if rng.next_f64() < p {
            flip_cell(c, alpha, rng);
        }
    }
}

/// Кластерная порча: burst 3–8 подряд идущих клеток, средняя доля ≈ p.
fn corrupt_clustered(cells: &mut [u8], p: f64, bpc: u32, rng: &mut Rng) {
    let n = cells.len();
    let alpha = 1u32 << bpc;
    let target = (p * n as f64).round() as usize;
    if target == 0 {
        return;
    }
    let mut errored = vec![false; n];
    let mut count = 0usize;
    let mut guard = 0usize;
    while count < target && guard < target * 6 + 32 {
        guard += 1;
        let len = 3 + rng.next_u32_below(6) as usize; // 3..=8
        let start = rng.next_u32_below(n as u32) as usize;
        for k in 0..len {
            let idx = start + k;
            if idx >= n {
                break;
            }
            if !errored[idx] {
                errored[idx] = true;
                count += 1;
            }
        }
    }
    for (i, c) in cells.iter_mut().enumerate() {
        if errored[i] {
            flip_cell(c, alpha, rng);
        }
    }
}

#[inline]
fn flip_cell(c: &mut u8, alpha: u32, rng: &mut Rng) {
    if alpha == 2 {
        *c ^= 1;
    } else {
        let mut nv = rng.next_u32_below(alpha) as u8;
        if nv == *c {
            nv = ((nv as u32 + 1) % alpha) as u8;
        }
        *c = nv;
    }
}

/// Мажоритарное голосование по клеткам из V проходов (tie → значение прохода 0).
fn vote(passes: &[Vec<u8>], bpc: u32) -> Vec<u8> {
    let alpha = 1usize << bpc;
    let n = passes[0].len();
    let mut out = vec![0u8; n];
    let mut counts = vec![0u16; alpha];
    for i in 0..n {
        for cnt in counts.iter_mut() {
            *cnt = 0;
        }
        for pass in passes {
            counts[pass[i] as usize] += 1;
        }
        // максимум; при равенстве берём значение первого прохода (детерминизм)
        let first = passes[0][i] as usize;
        let mut best = first;
        let mut best_c = counts[first];
        for (v, &c) in counts.iter().enumerate() {
            if c > best_c {
                best_c = c;
                best = v;
            }
        }
        out[i] = best as u8;
    }
    out
}

/// Тестовый заголовок+TI для MC (TI всегда, худший регион).
fn mc_header() -> l3::FrameHeader {
    l3::FrameHeader::new(0x5153_4930, 0x0012_3456, 8)
}
fn mc_ti() -> l3::TransferInfo {
    l3::TransferInfo { transfer_length: 1_000_000, symbol_size: 256, k: 4000, checksum: 0xC0FF_EE00 }
}

#[derive(Clone, Copy)]
enum Codec {
    Cur,
    Block(BlockLayout),
}

/// (header+TI прочитан?, чистых байт данных) одного разбора.
fn measure(codec: Codec, cells: &[u8], bpc: u32) -> (bool, usize) {
    match codec {
        Codec::Cur => {
            let pr = l3::parse_frame(cells, bpc);
            let hdr = pr.header.is_some() && pr.transfer_info.is_some();
            let bytes: usize = pr.salvaged.iter().map(|(_, b)| b.len()).sum();
            (hdr, bytes)
        }
        Codec::Block(lay) => {
            let pr = l3::parse_frame_blocks(cells, bpc, &lay);
            let hdr = pr.header.is_some() && pr.transfer_info.is_some();
            let bytes: usize = pr.salvaged.iter().map(|(_, b)| b.len()).sum();
            (hdr, bytes)
        }
    }
}

/// Построить клетки кадра выбранным кодеком с заголовком+TI и заполнением.
fn build(codec: Codec, bpc: u32) -> Vec<u8> {
    let cap = match codec {
        Codec::Cur => l3::frame_byte_capacity(bpc),
        Codec::Block(lay) => lay.frame_byte_capacity(bpc),
    };
    let hl = l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN;
    let sym: Vec<u8> = (0..cap.saturating_sub(hl)).map(|i| (i * 31 + 7) as u8).collect();
    match codec {
        Codec::Cur => l3::build_frame(&mc_header(), Some(&mc_ti()), &sym, bpc),
        Codec::Block(lay) => l3::build_frame_blocks(&mc_header(), Some(&mc_ti()), &sym, bpc, &lay),
    }
}

/// MC точка: (доля прочитанных header+TI, средний множитель goodput).
/// `votes`=1 — одиночный кадр; иначе голосование V проходов.
#[allow(clippy::too_many_arguments)]
fn mc_point(codec: Codec, bpc: u32, p: f64, votes: usize, clustered: bool, trials: usize, point: usize) -> (f64, f64) {
    let clean = build(codec, bpc);
    let ideal = ideal_bytes(bpc);
    let mut hdr_ok = 0usize;
    let mut bytes_acc = 0.0f64;
    for t in 0..trials {
        let mut rng = Rng::new(seed_for(point, t));
        let voted = if votes <= 1 {
            let mut rx = clean.clone();
            if clustered { corrupt_clustered(&mut rx, p, bpc, &mut rng); } else { corrupt_iid(&mut rx, p, bpc, &mut rng); }
            rx
        } else {
            let passes: Vec<Vec<u8>> = (0..votes)
                .map(|_| {
                    let mut rx = clean.clone();
                    if clustered { corrupt_clustered(&mut rx, p, bpc, &mut rng); } else { corrupt_iid(&mut rx, p, bpc, &mut rng); }
                    rx
                })
                .collect();
            vote(&passes, bpc)
        };
        let (h, bytes) = measure(codec, &voted, bpc);
        if h {
            hdr_ok += 1;
        }
        bytes_acc += bytes as f64;
    }
    let hdr_rate = hdr_ok as f64 / trials as f64;
    let gp = bytes_acc / trials as f64 / ideal / votes as f64;
    (hdr_rate, gp)
}

fn print_mc(bpc: u32) {
    let trials = 4000usize;
    let l1 = BlockLayout::PER_ROW_CRC8;
    let l2 = BlockLayout::PER_2ROW_CRC16;
    // (метка, кодек, голоса)
    let variants: [(&str, Codec, usize); 6] = [
        ("CUR §6.2", Codec::Cur, 1),
        ("V1a row/CRC8", Codec::Block(l1), 1),
        ("V1b 2row/CRC16", Codec::Block(l2), 1),
        ("V3 vote×5", Codec::Cur, 5),
        ("V1a+V3 ×5", Codec::Block(l1), 5),
        ("V3 vote×9", Codec::Cur, 9),
    ];

    println!("\n## Кластерные ошибки (burst 3–8), Monte Carlo реальными кодеками, bpc={bpc}");
    println!("{trials} кадров/точку; голоса — независимая раскладка burst каждый проход.");
    println!("| вариант | p=3% hdr | p=3% gp | p=8% hdr | p=8% gp | (i.i.d. свер. p=3% hdr/gp) |");
    println!("|---|---|---|---|---|---|");
    let mut point = 9000usize;
    for (name, codec, votes) in variants {
        let (h3, g3) = mc_point(codec, bpc, 0.03, votes, true, trials, point); point += 1;
        let (h8, g8) = mc_point(codec, bpc, 0.08, votes, true, trials, point); point += 1;
        let (h3i, g3i) = mc_point(codec, bpc, 0.03, votes, false, trials, point); point += 1;
        println!(
            "| {name} | {:.3} | {} | {:.3} | {} | {:.3} / {} |",
            h3, fmt_gp(g3), h8, fmt_gp(g8), h3i, fmt_gp(g3i)
        );
    }
}

fn print_pvoted() {
    println!("\n## p_voted(p, V) — SER клетки после мажоритарного голосования (bpc=1 точно)");
    println!("| p \\ V | 3 | 5 | 9 |");
    println!("|---|---|---|---|");
    for &p in &PS {
        println!(
            "| {:.1}% | {:.2e} | {:.2e} | {:.2e} |",
            p * 100.0, p_voted(p, 3), p_voted(p, 5), p_voted(p, 9)
        );
    }
}

/// Чувствительность V3 к систематической корреляции ошибок между проходами.
fn print_systematic() {
    let rhos = [0.0, 0.25, 0.5, 1.0];
    let ti_need = l3::FRAME_HEADER_LEN + l3::TRANSFER_INFO_LEN;
    let cur = cur_blocks(1);
    let v1a = v1_blocks(1, &BlockLayout::PER_ROW_CRC8);
    let cur_cells = cover_cells(&cur, ti_need); // 399 (страйп 0)
    let v1a_cells = cover_cells(&v1a, ti_need); // 285 (5 строк при bpc=1)

    println!("\n## Риск V3: голосование при систематической корреляции ошибок (bpc=1)");
    println!(
        "ρ = доля систематической (одинаковой каждый проход) ошибки. ρ=0 — идеал (мой основной\n\
         режим); ρ=1 — статический стенд (соседняя сессия: голосование бесполезно). Ячейка =\n\
         eff_SER · выживание header+TI (страйп CUR {cur_cells} кл / блок V1a {v1a_cells} кл)."
    );
    println!("| p, V \\ ρ | 0.0 | 0.25 | 0.5 | 1.0 |");
    println!("|---|---|---|---|---|");
    for &(p, v) in &[(0.03, 5usize), (0.03, 9), (0.08, 5), (0.08, 9)] {
        print!("| p={:.0}% V={v} |", p * 100.0);
        for &rho in &rhos {
            let eff = p_voted_systematic(p, v, rho);
            let s_cur = survive(cur_cells, eff);
            let s_v1a = survive(v1a_cells, eff);
            print!(" {:.1e}·{:.2}/{:.2} |", eff, s_cur, s_v1a);
        }
        println!();
    }
    println!("вывод: при ρ→1 голосование вырождается (eff→p) — rx ДОЛЖЕН сперва убрать");
    println!("систематический пол демодуляцией (лок. порог соседней сессии: 13%→0.04%),");
    println!("и/или обеспечить дрейф кадра; V1 (мелкий CRC) устойчив без допущения независимости.");
}

pub fn cmd_l3live() {
    let t0 = std::time::Instant::now();
    println!("# psicode-sim l3live — L3-варианты против live-grade SER (§6.2)");
    println!(
        "сетка {COLS}×{ROWS}={NCELLS} клеток; страйпы §6.2 {:?}; live bpc=1 (Mono, SER 12–15% на дампах).",
        l3::STRIPE_ROWS
    );
    println!("KPI: (1) кадров до первого header+TI = разблокировка k; (2) множитель goodput vs идеал.");

    print_pvoted();
    print_analytic(1); // живой рабочий режим
    print_analytic(5); // эталонный профиль (для контраста по bpc)
    print_systematic();
    print_mc(1);

    println!("\nвсего {:.2} c", t0.elapsed().as_secs_f64());
}
