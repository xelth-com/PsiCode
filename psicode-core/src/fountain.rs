//! Временный XOR-фонтан (§9.2, EXPERIMENTAL): систематические символы + repair
//! как XOR детерминированных псевдослучайных подмножеств (сид = ESI); декодер —
//! peeling + гауссово исключение над GF(2). Заменяется RaptorQ (§6.1) до
//! заморозки. `no_std` + `alloc`, БЕЗ вещественной математики.
//!
//! # Схема кода (v0)
//!
//! Источник длиной `L` байт режется на `K = ceil(L / symbol_size)` source-символов
//! (последний добивается нулями). Код систематический: `symbol(esi)` при `esi < K`
//! отдаёт source-символ `esi` как есть; при `esi ≥ K` отдаёт repair-символ — XOR
//! детерминированного псевдослучайного подмножества source-символов, чья степень
//! и состав выведены из ГПСЧ splitmix64, засеянного самим `esi`. Приёмник,
//! получив тот же `esi`, воспроизводит то же подмножество — сид общий.
//!
//! ## Распределение степеней repair-символа (задокументировано, под тюнинг)
//!
//! Степень `d` (число source-символов в XOR) берётся из усечённой «солитоно-
//! подобной» таблицы по одному ГПСЧ-броску `r = rng % 100`:
//!
//! | диапазон `r` | доля | степень `d` |
//! |---|---|---|
//! | 0..9   | 10 % | 1 |
//! | 10..59 | 50 % | 2 |
//! | 60..79 | 20 % | 3 |
//! | 80..94 | 15 % | 4..8 (ещё бросок) |
//! | 95..99 | 5 %  | max(2, K/2) — «тяжёлый» символ, лечит остаточный ранг |
//!
//! Малые степени (1–3) кормят peeling; редкие большие степени поднимают ранг,
//! когда peeling застревает и включается GF(2)-исключение. Измеренный средний
//! overhead ε (см. тест `overhead_epsilon_pinned`) при потерях 10–50 % ≤ 0.25.
//!
//! # Декодер
//!
//! `push(esi, bytes)` строит уравнение (набор source-индексов + XOR-значение),
//! сокращает его по уже решённым символам и запускает peeling (решение уравнений
//! степени 1 с подстановкой). Когда peeling застревает, `try_finish` собирает
//! оставшиеся уравнения в матрицу над GF(2) и решает гауссовым исключением;
//! кандидат-решение принимается ТОЛЬКО если CRC-32C совпал (§6.2) — иначе декодер
//! не «завершается» ложно. Мусор/дубликаты/противоречия не роняют декодер.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// CRC-32C (Castagnoli, §6.2 TransferInfo checksum)
// ---------------------------------------------------------------------------

/// CRC-32C (полином Castagnoli 0x1EDC6F41, рефлектированный, init 0xFFFFFFFF,
/// финальный XOR 0xFFFFFFFF) — стандартный CRC-32C для контрольной суммы объекта
/// §6.2. Табличная реализация не нужна: побитовый вариант с рефлектированным
/// полиномом 0x82F63B78 (= reverse(0x1EDC6F41)). `no_std`, без float.
///
/// Тест-вектор: `crc32c(b"123456789") == 0xE306_9283`.
pub fn crc32c(data: &[u8]) -> u32 {
    // рефлектированный полином CRC-32C
    const POLY: u32 = 0x82F6_3B78;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            // маска = 0xFFFFFFFF если младший бит взведён, иначе 0
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (POLY & mask);
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// ГПСЧ и выбор repair-подмножества (общий для кодера и декодера)
// ---------------------------------------------------------------------------

/// splitmix64 — детерминированный ГПСЧ, засеваемый ESI. Без внешних зависимостей.
struct SplitMix {
    state: u64,
}

impl SplitMix {
    #[inline]
    fn new(seed: u64) -> Self {
        SplitMix { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Равномерное целое из [0, n). При n = 0 возвращает 0.
    #[inline]
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as u32
    }
}

/// Степень repair-символа из усечённой солитоно-подобной таблицы (см. модульдок).
/// Только целочисленная арифметика. `d` зажимается в 1..=K.
fn sample_degree(rng: &mut SplitMix, k: u32) -> u32 {
    if k <= 1 {
        return 1;
    }
    let r = (rng.next_u64() % 100) as u32;
    let d = if r < 10 {
        1
    } else if r < 60 {
        2
    } else if r < 80 {
        3
    } else if r < 95 {
        4 + rng.below(5) // 4..=8
    } else {
        core::cmp::max(2, k / 2)
    };
    core::cmp::min(d, k)
}

/// Детерминированный набор source-индексов repair-символа `esi` при `K` источниках.
/// Кодер и декодер вызывают одну и ту же функцию с тем же `esi` — состав совпадает.
/// Индексы различны; порядок не важен (участвуют только в XOR).
pub fn repair_neighbors(esi: u32, k: u32) -> Vec<u32> {
    if k == 0 {
        return Vec::new();
    }
    // хорошо перемешанный сид из ESI (splitmix-константы)
    let seed = (esi as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ 0xD1B5_4A32_D192_ED03;
    let mut rng = SplitMix::new(seed);
    rng.next_u64(); // разогрев
    let d = sample_degree(&mut rng, k);
    let mut chosen: Vec<u32> = Vec::with_capacity(d as usize);
    while (chosen.len() as u32) < d {
        let idx = rng.below(k);
        if !chosen.contains(&idx) {
            chosen.push(idx);
        }
    }
    chosen
}

// ---------------------------------------------------------------------------
// Кодер
// ---------------------------------------------------------------------------

/// Систематический XOR-фонтан-кодер (§9.2).
pub struct FountainEncoder {
    symbol_size: usize,
    k: u32,
    transfer_length: usize,
    /// Источник, добитый нулями до `K · symbol_size`.
    data: Vec<u8>,
}

impl FountainEncoder {
    /// `data` — исходный объект, `symbol_size` — размер encoding-символа (байт).
    /// `K = ceil(len / symbol_size)`; последний source-символ добивается нулями.
    pub fn new(data: &[u8], symbol_size: usize) -> Self {
        assert!(symbol_size > 0, "symbol_size должен быть > 0");
        let transfer_length = data.len();
        let k = if data.is_empty() {
            0
        } else {
            data.len().div_ceil(symbol_size) as u32
        };
        let mut padded = data.to_vec();
        padded.resize(k as usize * symbol_size, 0);
        FountainEncoder {
            symbol_size,
            k,
            transfer_length,
            data: padded,
        }
    }

    /// Число source-символов K.
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Полная длина объекта (байт).
    pub fn transfer_length(&self) -> usize {
        self.transfer_length
    }

    /// Размер символа (байт).
    pub fn symbol_size(&self) -> usize {
        self.symbol_size
    }

    /// Encoding-символ с номером `esi`: систематический source (`esi < K`) или
    /// repair — XOR псевдослучайного подмножества (`esi ≥ K`). Всегда `symbol_size`
    /// байт.
    pub fn symbol(&self, esi: u32) -> Vec<u8> {
        let ss = self.symbol_size;
        if esi < self.k {
            let s = esi as usize * ss;
            self.data[s..s + ss].to_vec()
        } else {
            let mut out = vec![0u8; ss];
            for &idx in &repair_neighbors(esi, self.k) {
                let s = idx as usize * ss;
                xor_into(&mut out, &self.data[s..s + ss]);
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Декодер
// ---------------------------------------------------------------------------

/// Уравнение декодера: набор ещё НЕ решённых source-индексов и XOR-значение
/// (уже сокращённое по решённым символам).
struct Equation {
    nb: Vec<u32>,
    val: Vec<u8>,
}

/// Систематический XOR-фонтан-декодер (§9.2): peeling + GF(2)-исключение,
/// приёмка решения по CRC-32C. Не паникует на мусоре/дубликатах/противоречиях.
pub struct FountainDecoder {
    k: usize,
    symbol_size: usize,
    transfer_length: usize,
    checksum: u32,
    /// Решённые source-символы (None — ещё не решён).
    solved: Vec<Option<Vec<u8>>>,
    n_solved: usize,
    /// Активные уравнения (ссылаются только на НЕ решённые символы — инвариант).
    eqs: Vec<Equation>,
    /// Уже виденные ESI (защита от дубликатов).
    seen: BTreeSet<u32>,
}

/// XOR `src` в `dst` побайтно (равные длины).
#[inline]
fn xor_into(dst: &mut [u8], src: &[u8]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

impl FountainDecoder {
    /// Новый декодер по параметрам §6.2 TransferInfo. `checksum` — CRC-32C объекта.
    pub fn new(k: u32, symbol_size: usize, transfer_length: usize, checksum: u32) -> Self {
        let k = k as usize;
        FountainDecoder {
            k,
            symbol_size,
            transfer_length,
            checksum,
            solved: (0..k).map(|_| None).collect(),
            n_solved: 0,
            eqs: Vec::new(),
            seen: BTreeSet::new(),
        }
    }

    /// Число source-символов K.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Сколько source-символов уже решено.
    pub fn n_solved(&self) -> usize {
        self.n_solved
    }

    /// Число принятых (уникальных) encoding-символов.
    pub fn n_seen(&self) -> usize {
        self.seen.len()
    }

    /// Все ли source-символы решены (без проверки CRC).
    pub fn is_solved(&self) -> bool {
        self.n_solved == self.k && self.k > 0
    }

    /// Принять encoding-символ. Возвращает `true`, если символ нёс новую информацию
    /// (не дубликат, корректной длины, не сократился в тождество). Никогда не паникует.
    pub fn push(&mut self, esi: u32, bytes: &[u8]) -> bool {
        if self.k == 0 || bytes.len() != self.symbol_size {
            return false;
        }
        if self.seen.contains(&esi) {
            return false;
        }
        self.seen.insert(esi);

        // соседи: source -> {esi}; repair -> псевдослучайное подмножество.
        let mut nb: Vec<u32> = if (esi as usize) < self.k {
            vec![esi]
        } else {
            repair_neighbors(esi, self.k as u32)
        };
        nb.retain(|&i| (i as usize) < self.k);
        if nb.is_empty() {
            return false;
        }

        // сокращение по уже решённым символам
        let mut val = bytes.to_vec();
        let mut i = 0;
        while i < nb.len() {
            let idx = nb[i] as usize;
            if let Some(sv) = &self.solved[idx] {
                xor_into(&mut val, sv);
                nb.swap_remove(i);
            } else {
                i += 1;
            }
        }
        if nb.is_empty() {
            // уравнение полностью определено решёнными символами — тождество,
            // новой информации нет (значение val должно быть нулём при согласии).
            return false;
        }

        self.eqs.push(Equation { nb, val });
        self.peel();
        true
    }

    /// Peeling: пока есть уравнение степени 1 — решаем его source-символ и
    /// подставляем во все прочие уравнения.
    fn peel(&mut self) {
        loop {
            let mut found = None;
            for (ei, e) in self.eqs.iter().enumerate() {
                if e.nb.len() == 1 {
                    found = Some(ei);
                    break;
                }
            }
            let ei = match found {
                Some(x) => x,
                None => break,
            };
            let e = self.eqs.swap_remove(ei);
            let idx = e.nb[0] as usize;
            if self.solved[idx].is_some() {
                continue; // уже решён другим уравнением — отбрасываем
            }
            self.solved[idx] = Some(e.val.clone());
            self.n_solved += 1;
            let sv = e.val;

            let mut j = 0;
            while j < self.eqs.len() {
                if let Some(p) = self.eqs[j].nb.iter().position(|&x| x as usize == idx) {
                    xor_into(&mut self.eqs[j].val, &sv);
                    self.eqs[j].nb.swap_remove(p);
                    if self.eqs[j].nb.is_empty() {
                        self.eqs.swap_remove(j);
                        continue;
                    }
                }
                j += 1;
            }
        }
    }

    /// Попытка завершить декодирование. Возвращает объект ТОЛЬКО если все source-
    /// символы решены И CRC-32C совпал. Если peeling не закрыл систему — пробует
    /// гауссово исключение над GF(2); решение фиксируется лишь при совпадении CRC
    /// (иначе декодер не «завершается» ложно). Никогда не паникует.
    pub fn try_finish(&mut self) -> Option<Vec<u8>> {
        if self.k == 0 {
            return None;
        }
        if self.n_solved == self.k {
            return self.assemble_and_check(&[]);
        }
        // peeling застрял — решаем остаток над GF(2)
        let sol = self.solve_ge()?;
        let out = self.assemble_and_check(&sol);
        if out.is_some() {
            // фиксируем решение только после успешной сверки CRC
            for (idx, v) in sol {
                if self.solved[idx].is_none() {
                    self.solved[idx] = Some(v);
                    self.n_solved += 1;
                }
            }
            self.eqs.clear();
        }
        out
    }

    /// Собрать объект из решённых символов, наложив дополнительные пары
    /// `(idx, value)` из GF(2)-решения; усечь до `transfer_length`; вернуть
    /// `Some(data)` при совпадении CRC-32C, иначе `None`.
    fn assemble_and_check(&self, extra: &[(usize, Vec<u8>)]) -> Option<Vec<u8>> {
        let mut data = vec![0u8; self.k * self.symbol_size];
        let mut have = vec![false; self.k];
        for i in 0..self.k {
            if let Some(v) = &self.solved[i] {
                data[i * self.symbol_size..(i + 1) * self.symbol_size].copy_from_slice(v);
                have[i] = true;
            }
        }
        for (idx, v) in extra {
            let i = *idx;
            data[i * self.symbol_size..(i + 1) * self.symbol_size].copy_from_slice(v);
            have[i] = true;
        }
        if !have.iter().all(|&h| h) {
            return None;
        }
        data.truncate(self.transfer_length);
        if crc32c(&data) == self.checksum {
            Some(data)
        } else {
            None
        }
    }

    /// Гауссово исключение над GF(2) по буферизованным уравнениям. Возвращает
    /// значения ВСЕХ ещё не решённых source-символов, если система полного ранга;
    /// иначе `None`. Ничего не фиксирует в `self` (решение проверяется вызывающим).
    fn solve_ge(&self) -> Option<Vec<(usize, Vec<u8>)>> {
        // столбцы = не решённые source-индексы
        let unknowns: Vec<usize> = (0..self.k).filter(|&i| self.solved[i].is_none()).collect();
        let m = unknowns.len();
        if m == 0 {
            return Some(Vec::new());
        }
        let mut col_of = vec![usize::MAX; self.k];
        for (c, &idx) in unknowns.iter().enumerate() {
            col_of[idx] = c;
        }
        let words = m.div_ceil(64);

        // строки матрицы из уравнений (ссылаются только на не решённые -> инвариант)
        let mut bits: Vec<Vec<u64>> = Vec::with_capacity(self.eqs.len());
        let mut vals: Vec<Vec<u8>> = Vec::with_capacity(self.eqs.len());
        for e in &self.eqs {
            let mut row = vec![0u64; words];
            for &nb in &e.nb {
                let c = col_of[nb as usize];
                // c == MAX не должно случаться (инвариант), но не паникуем
                if c != usize::MAX {
                    row[c / 64] ^= 1u64 << (c % 64);
                }
            }
            bits.push(row);
            vals.push(e.val.clone());
        }

        // Гаусс–Жордан: приводим к RREF; при полном ранге каждая опорная строка
        // остаётся с единственным битом -> её значение = значение неизвестного.
        let n = bits.len();
        let mut r = 0usize;
        let mut pivot_row_of_col = vec![usize::MAX; m];
        for col in 0..m {
            if r >= n {
                break;
            }
            let w = col / 64;
            let bmask = 1u64 << (col % 64);
            let mut sel = usize::MAX;
            for i in r..n {
                if bits[i][w] & bmask != 0 {
                    sel = i;
                    break;
                }
            }
            if sel == usize::MAX {
                continue; // свободный столбец на этом этапе
            }
            bits.swap(r, sel);
            vals.swap(r, sel);
            // исключаем col из ВСЕХ прочих строк (полный Гаусс–Жордан)
            for i in 0..n {
                if i != r && (bits[i][w] & bmask != 0) {
                    for wi in 0..words {
                        let x = bits[r][wi];
                        bits[i][wi] ^= x;
                    }
                    let (pivot_val, target_val) = if i < r {
                        let (a, b) = vals.split_at_mut(r);
                        (&b[0], &mut a[i])
                    } else {
                        let (a, b) = vals.split_at_mut(i);
                        (&a[r], &mut b[0])
                    };
                    xor_into(target_val, pivot_val);
                }
            }
            pivot_row_of_col[col] = r;
            r += 1;
        }
        if r < m {
            return None; // недоопределено — полный ранг не набран
        }
        let mut out = Vec::with_capacity(m);
        for col in 0..m {
            let pr = pivot_row_of_col[col];
            out.push((unknowns[col], vals[pr].clone()));
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Детерминированный псевдослучайный поток байт (splitmix64) для источников.
    fn pseudo_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix::new(seed ^ 0x1234_5678_9ABC_DEF0);
        (0..len).map(|_| (rng.next_u64() >> 24) as u8).collect()
    }

    #[test]
    fn crc32c_test_vector() {
        // канонический тест-вектор CRC-32C
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        // пустой вход -> финальный XOR обнуляет: !0xFFFFFFFF = 0
        assert_eq!(crc32c(b""), 0x0000_0000);
    }

    #[test]
    fn systematic_roundtrip_exact_k() {
        // ровно K систематических символов -> точная сборка без repair.
        for &(len, ss) in &[(1000usize, 100usize), (1023, 64), (140, 140), (7, 4)] {
            let data = pseudo_bytes(len, len as u64);
            let enc = FountainEncoder::new(&data, ss);
            let k = enc.k();
            let mut dec = FountainDecoder::new(k, ss, len, crc32c(&data));
            for esi in 0..k {
                let sym = enc.symbol(esi);
                assert_eq!(sym.len(), ss);
                dec.push(esi, &sym);
            }
            let out = dec.try_finish().expect("систематический декод");
            assert_eq!(out, data, "len={len} ss={ss}");
        }
    }

    #[test]
    fn crc_mismatch_blocks_completion() {
        // тот же K символов, но CRC подсунут неверный -> декодер не завершается.
        let data = pseudo_bytes(500, 42);
        let enc = FountainEncoder::new(&data, 50);
        let k = enc.k();
        let mut dec = FountainDecoder::new(k, 50, 500, crc32c(&data) ^ 0x1);
        for esi in 0..k {
            let s = enc.symbol(esi);
            dec.push(esi, &s);
        }
        assert!(dec.is_solved(), "символы решены");
        assert!(dec.try_finish().is_none(), "неверный CRC не должен пройти");
    }

    #[test]
    fn duplicate_and_garbage_tolerance() {
        let data = pseudo_bytes(800, 7);
        let enc = FountainEncoder::new(&data, 80);
        let k = enc.k();

        // (1) дубликаты и мусор неверной длины: без паники, без новой информации.
        let mut dec = FountainDecoder::new(k, 80, 800, crc32c(&data));
        let s0 = enc.symbol(0);
        assert!(dec.push(0, &s0));
        assert!(!dec.push(0, &s0), "дубликат не новая информация");
        assert!(!dec.push(1, &[0u8; 3]), "мусор неверной длины отвергается");
        assert!(!dec.push(1, &[0u8; 200]), "мусор неверной длины отвергается");

        // (2) конфликтующий repair-мусор, ПРИНЯТЫЙ после решения его источников,
        //     сокращается в тождество и отбрасывается — корректный декод не страдает.
        let mut dec = FountainDecoder::new(k, 80, 800, crc32c(&data));
        for esi in 0..k {
            dec.push(esi, &enc.symbol(esi)); // все source решены систематически
        }
        let garbage = vec![0xABu8; 80];
        dec.push(k, &garbage); // repair-мусор поверх уже решённой системы
        assert_eq!(dec.try_finish(), Some(data.clone()), "мусор поверх решённого безвреден");

        // (3) КОНТРАКТ на произвольный мусор: декодер НИКОГДА не паникует и НИКОГДА
        //     не возвращает объект, отличный от истинного (CRC-32C — страж §6.2).
        //     Мусорный repair до решения источников может «отравить» декод — тогда
        //     try_finish обязан вернуть None, но не чужие данные.
        let mut r = SplitMix::new(0xDEAD_BEEF);
        for _ in 0..64 {
            let mut dec = FountainDecoder::new(k, 80, 800, crc32c(&data));
            for esi in 0..(k + 8) {
                let use_garbage = r.next_u64() & 1 == 0;
                let sym = if use_garbage {
                    (0..80).map(|_| (r.next_u64() >> 24) as u8).collect::<Vec<u8>>()
                } else {
                    enc.symbol(esi)
                };
                dec.push(esi as u32, &sym);
                let out = dec.try_finish();
                assert!(
                    out.is_none() || out.as_deref() == Some(&data[..]),
                    "мусор привёл к ложному объекту"
                );
            }
        }
    }

    #[test]
    fn empty_decoder_never_completes() {
        let mut dec = FountainDecoder::new(0, 10, 0, crc32c(b""));
        assert!(dec.try_finish().is_none());
        assert!(!dec.push(0, &[0u8; 10]));
    }

    #[test]
    fn peeling_and_ge_recover_from_loss() {
        // теряем ~30% source, добираем repair -> peeling+GE должны восстановить.
        let data = pseudo_bytes(2000, 99);
        let ss = 100;
        let enc = FountainEncoder::new(&data, ss);
        let k = enc.k();
        let mut dec = FountainDecoder::new(k, ss, data.len(), crc32c(&data));

        let mut rng = SplitMix::new(0xBEEF);
        // систематический проход с потерей 30%
        for esi in 0..k {
            if rng.below(100) >= 30 {
                dec.push(esi, &enc.symbol(esi));
            }
        }
        // поток repair, пока не соберётся
        let mut esi = k;
        let mut done = None;
        while esi < k + 10 * k && done.is_none() {
            dec.push(esi, &enc.symbol(esi));
            done = dec.try_finish();
            esi += 1;
        }
        assert_eq!(done, Some(data), "peeling+GE восстановили из потерь");
    }

    /// PIN среднего overhead ε при потерях 10–50 %: репортим и держим ε ≤ 0.25.
    /// ε = (полезных_символов_на_момент_декода / K) − 1.
    #[test]
    fn overhead_epsilon_pinned() {
        let ss = 64;
        let mut sum_eps = 0.0f64;
        let seeds = 24u64;
        let mut worst = 0.0f64;
        for seed in 0..seeds {
            let mut r = SplitMix::new(0xA11CE ^ seed.wrapping_mul(0x9E37_79B9));
            // случайная длина -> K в разумном диапазоне
            let k_target = 60 + (r.next_u64() % 160) as usize; // 60..219 source
            let len = k_target * ss - (r.next_u64() % ss as u64) as usize; // не кратно ss
            let data = pseudo_bytes(len, seed);
            let enc = FountainEncoder::new(&data, ss);
            let k = enc.k();
            let loss = 10 + (r.next_u64() % 41) as u32; // 10..50 %

            let mut dec = FountainDecoder::new(k, ss, data.len(), crc32c(&data));
            let mut useful = 0usize;
            let mut done = false;
            // перемежаем source и repair детерминированным потоком §6.1-подобно,
            // сбрасывая случайные символы; считаем полезные push.
            let mut esi = 0u32;
            let cap = 20 * k as u32;
            while esi < cap && !done {
                let keep = (r.next_u64() % 100) as u32 >= loss;
                if keep {
                    if dec.push(esi, &enc.symbol(esi)) {
                        useful += 1;
                    }
                    if dec.try_finish().is_some() {
                        done = true;
                    }
                }
                esi += 1;
            }
            assert!(done, "seed {seed}: не декодировалось (K={k}, loss={loss}%)");
            let eps = useful as f64 / k as f64 - 1.0;
            sum_eps += eps;
            if eps > worst {
                worst = eps;
            }
        }
        let mean_eps = sum_eps / seeds as f64;
        // печатаем измеренное значение (число — часть поставки)
        eprintln!(
            "fountain overhead: mean ε = {mean_eps:.4}, worst ε = {worst:.4} (потери 10–50%, {seeds} сидов)"
        );
        assert!(mean_eps <= 0.25, "средний overhead ε={mean_eps:.4} > 0.25");
    }
}
