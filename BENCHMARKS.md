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
| sim | SER | 0.0034 | 0.0700 | 0.7100 | 0.9333 | 0.9599 | 0.9653 |
| live | SER | | | | | | |

> sim v0 (2026-07-22): genie-aided geometry (known homography, no ZC
> detection yet); channel = telemetry truth of the reference profile
> (crosstalk 6 %/8 %, sensor noise σ = 2.0 gray levels, matched gammas);
> px/cell = 8; 15 frames/point, deterministic seeds. SER saturation ≈ 0.97
> = chance level for 5-bit cells. Bonus noise sweep (σ_blur = 1): noise
> ×1/×2/×4/×8 → SER 0.074 / 0.103 / 0.185 / 0.369.

## 2. SER / FER vs. distance

Distance normalized as camera pixels per display cell.

| source | px/cell → | 8 | 6 | 4 | 3 | 2 | 1.5 |
|---|---|---|---|---|---|---|---|
| sim | SER | 0.0688 | 0.4718 | 0.7204 | 0.8632 | 0.9389 | 0.9256 |
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
