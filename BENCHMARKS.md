# PsiCode Benchmarks

Filled first by `psicode-sim` Monte Carlo sweeps, then validated by live
measurements (SPEC §9). Empty cell = not measured yet. Every row states its
configuration; default configuration unless noted:

> Mode A, 1080p, cell 16 px, luma 3 bit, chroma 2 bit (Chroma2), quiet zone
> 4 cells, frame hold 6 periods @ 60 Hz, reference vector profile of SPEC §7.4.

## 1. Symbol error rate vs. blur σ

Gaussian blur applied in sim; live equivalent measured via `mtf_limit_px`.

| source | blur σ (px) → | 0.5 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| sim | SER | | | | | | |
| live | SER | | | | | | |

## 2. SER / FER vs. distance

Distance normalized as camera pixels per display cell.

| source | px/cell → | 8 | 6 | 4 | 3 | 2 | 1.5 |
|---|---|---|---|---|---|---|---|
| sim | SER | | | | | | |
| sim | FER | | | | | | |
| live | SER | | | | | | |
| live | FER | | | | | | |

## 3. SER / FER vs. viewing angle

| source | angle (°) → | 0 | 15 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|
| sim | SER | | | | | |
| sim | FER | | | | | |
| live | SER | | | | | |
| live | FER | | | | | |

## 4. Goodput

End-to-end goodput (kbit/s) after framing, FEC and loss, per channel
condition; includes torn-frame partial decoding (SPEC §6.3) on/off.

| source | condition | goodput, partial decode OFF | goodput, partial decode ON |
|---|---|---|---|
| sim | clean channel | | |
| sim | blur σ = 2 px | | |
| sim | 20 % torn frames | | |
| sim | 50 % torn frames | | |
| live | best display/phone pair | | |
| live | worst display/phone pair | | |

## 5. Device matrix (live)

≥ 3 display/phone pairs (SPEC §9.3).

| display | camera | mtf_limit_px | torn % | crosstalk R↔G / G↔B (%) | max goodput |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
