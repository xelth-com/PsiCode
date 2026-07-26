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
| detected SER | 0.0003 | 0.098 | 0.768 | 0.942 | 0.962 | 0.995 |
| detect success | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.13 |
| mean score | 1.000 | 1.000 | 0.999 | 0.935 | 0.763 | 0.576 |

> sim v0 (2026-07-25, after sub-cell ring fine-alignment): G-plane luma into
> `detect_symbol`, px/cell = 8, 15 frames/point, deterministic seeds; detect
> failure ⇒ frame lost (SER = 1.0). Detection recovers frame + orientation
> 100 % through σ ≤ 6 and every px/cell down to 1.5, collapsing only at σ = 8.
> Geometry cost after fine-alignment: **1.4× genie at σ = 1** (0.098 vs 0.070;
> was 4× before), *better* than genie at σ = 0.5 and px/cell = 6, and mild
> perspective (§8) 0.303 vs genie 0.250. History: the pre-refinement penalty
> came from per-side 1-D lags saturating under blur; fixed by full-ring 2-D
> Pearson alignment sampled near cell edges (±0.35), where blur creates
> gradient. Cost ≈ +3.7 ms/frame.

## 2. SER / FER vs. distance

Distance normalized as camera pixels per display cell.

| source | px/cell → | 8 | 6 | 4 | 3 | 2 | 1.5 |
|---|---|---|---|---|---|---|---|
| sim | SER | 0.0688 | 0.4718 | 0.7204 | 0.8632 | 0.9389 | 0.9256 |
| sim | FER | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| live | SER | | | | | | |
| live | FER | | | | | | |

> sim v0 (2026-07-25), `psicode-sim framed`: **FER** = fraction of frames lost,
> where a frame is lost iff the header stripe (stripe 0) fails its CRC-16 **or**
> all 8 stripes fail (§6.2 per-stripe CRC; L3 `parse_frame`). Reference profile
> (luma 3 + Chroma2, 5 bit/cell), genie geometry, σ = 1, no tearing, 15
> frames/point. FER = 1.000 at **every** px/cell: at σ = 1 the reference config's
> raw SER ≈ 0.07 shreds every 399-cell stripe (the §6 stripe cliff), so even at
> px/cell 8 no whole stripe — hence no header — survives. FER < 1 needs the
> robust low-bit configs of §6 (raw SER ≲ 1e−3), not the 5-bit reference config.

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
| sim | clean channel | 148.2 | 148.2 |
| sim | blur σ = 2 px | 24.7 | 24.7 |
| sim | 20 % torn frames | 118.7 | 142.6 |
| sim | 50 % torn frames | 74.2 | 133.7 |
| live | best display/phone pair | | |
| live | worst display/phone pair | | |

> sim v0 (2026-07-25), `psicode-sim goodput`: goodput = surviving-stripe bits
> (§6.2 per-stripe CRC) × 10 fps × 0.8 FEC × (1 − 0.015 header). Each row uses
> the **optimum (luma_bits × chroma) config for that condition** (clean →
> luma 4 + Chroma2, 6 bit/cell; blur σ = 2 → luma 1 + Mono, 1 bit/cell — see §6
> for the full config sweep and the argmax per σ). Channel = telemetry truth of
> the §7.4 profile, px/cell = 8, 15 frames/point. On the clean / blur rows there
> are **no torn frames**, so partial-decode is a no-op ⇒ ON = OFF.
>
> **Torn-frame rows** (`psicode-sim framed`, 2026-07-25): real L3 framing —
> `build_frame` lays FrameHeader [+ TransferInfo] + encoding-symbol bytes across
> the 8 stripes, each closed by CRC-16/CCITT; `parse_frame` recovers per-stripe
> validity. A rolling-shutter tear (§6.3) is composed **before** the channel:
> rows [0, t) from rendered frame N, rows [t, H) from frame N+1, tear row t drawn
> inside a random payload cell-row so the straddling stripe always fails CRC and
> marks the seam. Config = luma 4 + Chroma2 (6 bit/cell) on a **clean channel**
> (isolates tearing, matching the clean-channel baseline; on a σ = 1 channel the
> 6-bit config's stripe cliff would zero both columns, so blur is left to the
> rows above). Goodput = mean salvaged **symbol** bytes (header excluded via real
> L3) × 8 × 10 fps × 0.8 FEC / 1000; p_torn is a known parameter entering
> analytically, Monte-Carlo averages only the tear **position** (15 clean + 40
> torn draws). Attribution v0: top CRC-valid run → frame N (stripe-0 header),
> bottom run → frame N+1 (ESI predicted from N per §6.1), detected by counter-row
> ≠ header; OFF discards any torn capture, ON salvages both runs. If the header
> stripe (0) lands on the seam the capture is unattributable ⇒ 0 in both (real
> ~13 % of torn captures — the honest cap on ON's gain).
>
> **Verdict vs §6.3 "+20–30 % goodput under heavy tearing":** measured ON/OFF
> gain = **+20.1 %** at 20 % torn (148.4 → clean; 118.7 → 142.6) and **+80.2 %**
> at 50 % torn (74.2 → 133.7). So the spec's +20–30 % band is accurate for
> *mild-to-moderate* tearing (~20 %) but **understates** the payoff under *heavy*
> tearing: at 50 % torn the salvage nearly **doubles** goodput (OFF loses half of
> all captures; ON keeps ~7/8 of each torn frame from two frames at once). p = 0
> reproduces the 148.2 clean baseline, confirming the real-header accounting is
> consistent with the flat-1.5 % header model of the rows above.

## 5. Device matrix (live)

≥ 3 display/phone pairs (SPEC §9.3).

| display | camera | mtf_limit_px | torn % | crosstalk R↔G / G↔B (%) | max goodput |
|---|---|---|---|---|---|
| 1080p dev display (cell 12 px) | Samsung Galaxy A22 5G, camera-app JPEG | — | — | — | n/a (offline single frame; SER 0.057–0.105, BER 0.019–0.035) |
| | | | | | |
| | | | | | |

> **First light, 2026-07-26** (offline path `psicode-sim live <photo.ppm>`):
> hand-held photos of `psicode-tx single --cell 12`, decoded by the standard
> tract (ZC detection → demod) at 26–32 camera px/cell. 3/3 frames detected
> with correct rotation (incl. one 90° portrait shot) and correct frame
> counter; 2/3 fully decoded at SER 0.094/0.105 (BER 0.031/0.035). The phone's
> tone curve was self-calibrated per channel from the reference-strip gray
> staircase (γ_RGB ≈ 3.8/4.7/5.7 — far from sRGB; assuming γ 2.2 gives SER
> 0.34). Tilt update (same day): rotating-calipers min-area-rect coarse
> corners (convex hull of the activity component) made the ~7°-tilted frame
> the BEST decode (SER 0.058, BER 0.019), and **full uncropped 12 Mpx photos
> now decode end-to-end with no manual framing** (shot1 SER 0.104, shot2
> 0.057). Remaining edge cases: clutter fused to the symbol across a
> narrower-than-blur quiet moat, and one busy-frame partial lock — both
> documented for the Android rx.

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

## 8. End-to-end file transfer (sim)

`psicode-sim transfer`: 20 KB deterministic payload → XOR fountain
(EXPERIMENTAL §9.2, ε_pinned ≈ 1 %) → L3 frames (§6.2, symbol 140 B ×
8/frame, TransferInfo every 8th) → render with frame counter → channel →
**real ZC detection** → demod → stripe salvage (§6.3 on torn captures) →
fountain decode → CRC-32C verify. Config = σ=1 optimum of §6 (luma 2 +
Chroma1, 3 bit/cell), px/cell 8, telemetry-truth channel.

| σ | p_torn | K | frames | recv/lost | overhead ε | goodput | CRC-32C |
|---|---|---|---|---|---|---|---|
| 0.5 | 0 % | 143 | 22 | 176/0 | +0.007 | 72.7 kbit/s | OK |
| 1.0 | 0 % | 143 | 22 | 165/11 | +0.000 | 72.7 kbit/s | OK |
| 0.5 | 20 % | 143 | 33 | 254/10 | +0.014 | 48.5 kbit/s | OK |
| 1.0 | 20 % | 143 | 31 | 225/23 | +0.007 | 51.6 kbit/s | OK |

> First full-stack run (2026-07-25). Measured 72.7 kbit/s at σ=1 matches the
> §6 stripe-survival model's 74.1 kbit/s prediction. At σ=1 the interleaved
> repair symbols absorbed 11 stripe-death losses at zero net overhead
> (decode completed at exactly K useful symbols).
