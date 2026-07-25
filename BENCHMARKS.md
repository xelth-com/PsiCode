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

### Detected vs genie geometry

Honest receiver path (`psicode-sim sweep`, §6–8): the same post-channel frame
is demodulated twice — with **genie** geometry (known homography of the tract)
and with **real ZC-frame detection** (`psicode-core::detect`, §3.2) run on the
linear G-plane of the snapshot. Detection failure ⇒ frame lost ⇒ that trial's
SER counts as 1.0.

| metric \ blur σ (px) | 0.5 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| genie SER | 0.0034 | 0.070 | 0.710 | 0.933 | 0.960 | 0.965 |
| detected SER | 0.0008 | 0.284 | 0.735 | 0.931 | 0.959 | 0.998 |
| detect success | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.07 |
| mean score | 1.00 | 1.00 | 0.995 | 0.908 | 0.710 | 0.631 |

> sim v0 (2026-07-25): G-plane luma into `detect_symbol`, px/cell = 8, 15
> frames/point, deterministic seeds; detect failure ⇒ frame lost (SER = 1.0).
> Detection is **robust** — it recovers frame + orientation through σ ≤ 6 and
> every px/cell tested down to 1.5, collapsing only at σ = 8 (7 % success). But
> its estimated geometry costs SER: at the σ = 1 operating point detected SER
> ≈ 4× genie (0.284 vs 0.070), and mild perspective (§8) gives detected 0.478
> vs genie 0.250 (detection 100 %). The genie tables (§1–2) overstate the real
> detected-geometry operating envelope.

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
| sim | clean channel | 148.2 | |
| sim | blur σ = 2 px | 24.7 | |
| sim | 20 % torn frames | | |
| sim | 50 % torn frames | | |
| live | best display/phone pair | | |
| live | worst display/phone pair | | |

> sim v0 (2026-07-25), `psicode-sim goodput`: goodput = surviving-stripe bits
> (§6.2 per-stripe CRC) × 10 fps × 0.8 FEC × (1 − 0.015 header). Each row uses
> the **optimum (luma_bits × chroma) config for that condition** (clean →
> luma 4 + Chroma2, 6 bit/cell; blur σ = 2 → luma 1 + Mono, 1 bit/cell — see §6
> for the full config sweep and the argmax per σ). Channel = telemetry truth of
> the §7.4 profile, px/cell = 8, 15 frames/point. Partial-decode (torn-frame,
> §6.3) model not built yet ⇒ the ON column is empty.

## 5. Device matrix (live)

≥ 3 display/phone pairs (SPEC §9.3).

| display | camera | mtf_limit_px | torn % | crosstalk R↔G / G↔B (%) | max goodput |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

## 6. Goodput vs (luma_bits × chroma) — sim

Config-space sweep by `psicode-sim goodput`. **Stripe-survival model** (§6.2):
the 55 payload rows split into 8 stripes (7 × 7 rows + 6 rows, × 57 cols); a
stripe delivers its bits only if **all** its cells decode exactly (per-stripe
CRC), so a single bad cell drops the whole stripe. Frame FER = all 8 stripes
dead. This is the cliff that raw SER hides.

```
goodput = mean_surviving_bits_per_frame · fps(10) · FEC_keep(0.8) · (1 − header 0.015)
```

Channel = telemetry truth of the §7.4 profile (crosstalk 6 %/8 %, sensor noise
σ = 2 gray, matched gammas), px/cell = 8, blur σ as noted, 15 frames/point,
deterministic seeds. Cell = **goodput kbit/s · raw SER**.

### σ = 0.5 px

| luma \ chroma | Mono | Chroma1 | Chroma2 |
|---|---|---|---|
| luma 1 | 24.7 · 0 | 49.4 · 0 | 0.0 · 0.0141 |
| luma 2 | 49.4 · 0 | 74.1 · 0 | 5.0 · 0.0067 |
| luma 3 | 8.1 · 0.0063 | **98.0 · 2.1e−5** | 25.6 · 0.0040 |
| luma 4 | 0.0 · 0.0593 | 3.0 · 0.0095 | 1.1 · 0.0147 |

argmax: **luma 3 + Chroma1 (4 bit/cell) → 98.0 kbit/s**

### σ = 1 px

| luma \ chroma | Mono | Chroma1 | Chroma2 |
|---|---|---|---|
| luma 1 | 24.7 · 0 | 49.4 · 0 | 0.0 · 0.0892 |
| luma 2 | 4.6 · 0.0113 | **74.1 · 0** | 0.0 · 0.0340 |
| luma 3 | 0.0 · 0.1137 | 0.0 · 0.0427 | 0.0 · 0.0673 |
| luma 4 | 0.0 · 0.2495 | 0.0 · 0.2141 | 0.0 · 0.2511 |

argmax: **luma 2 + Chroma1 (3 bit/cell) → 74.1 kbit/s**

### σ = 2 px

| luma \ chroma | Mono | Chroma1 | Chroma2 |
|---|---|---|---|
| luma 1 | **24.7 · 0** | 0.0 · 0.1707 | 0.0 · 0.3393 |
| luma 2 | 0.0 · 0.1905 | 0.0 · 0.3353 | 0.0 · 0.4578 |
| luma 3 | 0.0 · 0.4694 | 0.0 · 0.6451 | 0.0 · 0.6918 |
| luma 4 | 0.0 · 0.7071 | 0.0 · 0.8274 | 0.0 · 0.8641 |

argmax: **luma 1 + Mono (1 bit/cell) → 24.7 kbit/s**

### Verdicts

* **Fewer bits/cell win as blur grows** — optimum bits/cell = 4 (σ 0.5) → 3
  (σ 1) → 1 (σ 2), all **below the reference profile's 5 bit/cell**
  (hypothesis 2 confirmed). The reference config (luma 3 + Chroma2, 5 bit)
  yields **0 goodput at σ ≥ 1**: raw SER 0.067 at σ 1 shreds every stripe.
* **The stripe cliff needs raw SER ≲ 0.1 %** (hypothesis 1 confirmed, and
  then some). A 399-cell stripe survives with probability (1 − SER)^399, so
  SER = 0.67 % already leaves ~5 % of stripes alive (luma 2 + Chroma2 at σ 0.5
  → only 5.0 of a possible ~99 kbit/s), and SER ≥ 1 % collapses goodput to ~0.
  Only SER ≈ 0 (≲ 1e−3) sustains goodput — the raw-SER tables (§1–2) badly
  overstate the usable operating envelope.
* **Chroma can beat Mono at equal-or-more bits** — e.g. luma 3 + Chroma1
  (4 bit) delivers 98.0 kbit/s at σ 0.5 while luma 3 + Mono (3 bit) delivers
  only 8.1: Mono spends the full luma amplitude, pushing extreme levels near
  the white clip where sensor noise saturates and errors spike; the chroma
  split (A_L = 0.7·usable) keeps luma off the rail.

## 7. Calibration sensitivity (sim)

Decoder uses deliberately wrong profile values; channel stays at truth.
σ_blur = 1, px/cell = 8, reference §7.4 profile, 15 frames/point.

| Δγ (all channels) | −0.2 | −0.1 | 0 | +0.1 | +0.2 |
|---|---|---|---|---|---|
| SER | 0.297 | 0.130 | 0.064 | 0.053 | 0.051 |

| Δwhite_level | 0 | −3 % | −6 % | −9 % | −12 % |
|---|---|---|---|---|---|
| SER | 0.068 | 0.128 | 0.301 | 0.558 | 0.700 |

> Asymmetric γ penalty: underestimating gamma is ~4.6× worse than
> overestimating — prescriptions can safely guard-band γ upward.
> white_level is the highest-leverage single field measured so far.
