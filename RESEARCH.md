# PsiCode Research Notes

**Non-normative.** [SPEC.md](SPEC.md) holds only what is frozen or on a
freezing path; this file holds the full text of experimental mechanisms and
the directions being explored around them. Anything here may change or die
without a spec version bump. Measurement results go to
[BENCHMARKS.md](BENCHMARKS.md).

---

## 1. Mode B — Hermite–Gauss payload (full construction)

Referenced from SPEC §5.3 **[EXPERIMENTAL]**.

The payload region (or designated blocks of it) carries a superposition of
2-D Hermite–Gauss functions:

```
ψ_{m,n}(x, y) = H_m(x/w) · H_n(y/w) · exp(−(x²+y²)/(2w²)) / √(2^{m+n} m! n! π w²)
```

* Block size: 64×64 px v0; envelope width `w` = block/8.
* Mode set v0: `m + n ≤ 4` → 15 modes.
* `ψ_{0,0}` is a **pilot** (fixed amplitude 1+0j).
* `ψ_{2,0}` and `ψ_{0,2}` are **channel probes** (fixed amplitude): the
  received ratio ‖a₂₀‖/‖a₀₀‖ estimates blur per block; the receiver derives
  the effective basis width w′ and per-order gain equalization from it.
* Remaining 12 modes carry data as complex coefficients (QPSK on the
  Re/Im color axes v0 ⇒ 24 bits/block raw).
* Decoding: inner product of the corrected complex image with each ψ_{m,n}
  of width w′; orthogonality separates coefficients.
* **Progressive property** (the point of Mode B): under increasing blur,
  coefficient SNR falls monotonically with m+n. A receiver MAY decode only
  the mode subset whose measured SNR clears threshold; the transmitter
  assigns data to modes in significance order.

Mode B parameters move into SPEC (→ DRAFT → STABLE) only after live
measurements of coefficient SNR vs. blur σ on real display/camera pairs.

**Sim freeze-gate data (v0, 2026-07-25, `psicode-sim modeb`):** 64×64 block,
w=8, telemetry-truth channel of the §7.4 profile, genie geometry.

* Discrete-basis orthonormality: max Gram off-diagonal **6.9e-5** — the
  64-px/w=8 grid samples the mode set essentially perfectly.
* Coefficient **attenuation is monotonic in m+n at every σ ≥ 1** (the §2
  claim holds); at σ=8 orders 0→4 attenuate to 0.638/0.469/0.366/0.258/0.219.
  Probe (2,0)/(0,2) isotropy under isotropic blur: 0.5 % at σ=2 — nil.
  Nuance: full-loading coefficient **SNR** is only broadly (not strictly)
  monotonic — gamma nonlinearity penalizes the DC-heavy ψ00; the clean
  monotonic quantity is attenuation.
* **Graceful shedding schedule:** QPSK BER = 0 for orders 0–3 through σ=8;
  order 4 is first to cross 10 % (at σ=8). Everything survives σ ≤ 6.
* **Mode A head-to-head** (same 64×64 area, same channel; Mode A = 64 cells
  8 px × 3 bit Mono = 192 bit raw vs Mode B 24 bit raw): Mode A cliffs
  (153→3→0 reliable bits over σ 1→2), Mode B holds 24 bit through σ=6.
  Crossover ≈ σ 1 for this config (config-dependent; the robust result is
  cliff-vs-graceful). Mode A's raw floor ≈ 24 bit is pure chance level —
  reliability-thresholded comparison is the honest one.
* **Pilot/probe equalization works**: ρ = ‖a20,02‖/‖a00‖ → calibration curve
  → σ_est (σ4→3.87, σ8→8.00), then w′ = √(w² + σ²); re-decoding with w′
  recovers **+1.8 dB (σ2) … +14.8 dB (σ8)** coefficient SNR and halves BER
  at σ=8. The graceful-degradation mechanism is real in v0 already.
* Caveats: blur applied in linear light while §5.1 is a drive-domain map ⇒
  mild γ cross-mode coupling on top of pure eigenmode attenuation;
  w′ = √(w²+σ²) is heuristic (empirically excellent). Live display/camera
  measurement still required before §5.3 freezes.

## 2. Mode order as physical priority for fountain symbols — primary line

The most promising composition of the two graceful mechanisms: map RaptorQ
encoding symbols onto HG modes so that **physical significance order
(m+n) = transport significance order**. Under defocus the channel then
sheds exactly the symbols the fountain code can most afford to lose, and
goodput degrades smoothly with blur instead of cliffing — HG graceful
degradation × fountain graceful reconstruction. Open questions: symbol-to-
mode scheduling across blocks, how the receiver's per-block w′ estimate
should gate which coefficients are handed to transport, and whether repair
symbols belong in low or high modes.

Sim input (2026-07-25, see §1): the shedding order is measured — orders 0–3
survive through σ=6, order 4 drops first — and the w′ probe estimate tracks
true σ within ~5 %, so per-block gating has a reliable signal to work with.
An XOR-fountain interim transport already runs end-to-end in the sim
(`psicode-sim transfer`, ε ≈ 1 %), giving this line a working harness.

## 3. QR code as optional return channel for calibration

When the transmitter device has a camera facing the receiver (e.g. laptop
webcam looking at the phone), the receiver can show the calibration profile
as a QR code instead of / in addition to the typed string. The typed
32-character code (SPEC §7) stays the universal fallback and the only
*required* path; QR is a convenience layer, not a spec dependency.

## 4. Subpixel R/G/B striping for near-field mode — parked

Drive display subpixels independently to triple horizontal resolution at
very short range. Parked: camera demosaicing likely destroys the subpixel
structure at any distance beyond near-contact, so the win exists only in a
regime where capacity is already abundant. Revisit only if a contact-range
use case appears.

## 5. Chroma error budget: gamma dominates aberration (sim finding, 2026-07-25)

Measured in psicode-sim while validating the §5.1 claims (see BENCHMARKS §7):

* Camera-luma Im-leakage: measured 0.201–0.205 vs the theoretical 0.2·A_C —
  §5.1 property (a) holds essentially exactly; gamma curvature is negligible
  at the mid-gray operating point.
* Im kernel averaging, §5.1 property (b): chroma SER under aberration
  (σ_R = 1.3, σ_B = 1.5) vs matched common blur σ = 1.4 differs by 1.4 pp —
  confirmed, residual slightly *favors* aberration.
* The claim the spec does NOT make: with varying luma, K_R ≠ K_B leaks
  (K_R−K_B)∗(A_L·Re) into the R−B chroma axis. Measured: +0.28 pp — real but
  second-order. The dominant chroma-error source at σ = 1 is instead the
  **per-channel gamma-inversion asymmetry** (γ_R ≠ γ_B in the decoder's
  gamma removal): ~12 pp chroma SER even with matched blur.

Implication: chroma robustness work should target **gamma estimation
accuracy** (richer reference-strip anchors, per-channel staircase fitting)
before any optics compensation. Related: calibration-sensitivity sweeps show
underestimating γ is ~4.6× worse than overestimating (asymmetric penalty —
prescriptions could guard-band γ upward), and white_level is high-leverage
(−6 % error ≈ 4× SER) — see BENCHMARKS §7.

## 6. Spatial redundancy / cell replication

Alternative knob to lowering `luma_bits` on noisy channels: keep bit depth
but replicate each cell n× (or scale `cell_size_px`), trading capacity for
SNR by averaging. Worth a sim sweep against the luma_bits knob to see which
buys more goodput per unit capacity lost; could become a profile
prescription if it wins in some regime.
