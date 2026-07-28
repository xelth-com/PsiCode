# PsiCode Specification

**Version 0.2.0-draft · 2026-07-28**

PsiCode (ΨCode) is an open, royalty-free visual code and one-way optical data
link. It transmits data from any display to any camera with **no feedback
channel**, using color-encoded complex fields, Hermite–Gauss mode payloads for
graceful degradation under defocus, a Zadoff–Chu synchronization frame, and
fountain-coded transport.

PsiCode is a barcode specification as much as a link: the static symbol is a
2-D color barcode, and the streaming form is a **3-D barcode whose third
dimension is time** — a symbol that evolves frame by frame on a display.
Barcodes left paper long ago; a code *designed* for emissive, time-varying
screens (rather than a paper code shown on one) is the niche this
specification claims.

This document is published as an open specification and as a **defensive
publication**: the mechanisms described herein are placed in the public domain
as prior art as of the date above. The PsiCode format is free to implement by
anyone, forever. The name "PsiCode" and the Ψ mark identify the format and its
reference implementation ([github.com/xelth-com/PsiCode](https://github.com/xelth-com/PsiCode)).

Reference implementation license: MIT OR Apache-2.0.

### Requirement levels

MUST / SHOULD / MAY are used per RFC 2119.

### Stability markers

| Marker | Meaning |
|---|---|
| **[STABLE]** | Implemented and frozen. Changing it bumps the major spec version. |
| **[DRAFT]** | Designed, parameters concrete, expected to survive live testing with tweaks. |
| **[EXPERIMENTAL]** | Direction is fixed, numbers are not. Will be frozen after channel measurements. |

---

## 1. Architecture overview

PsiCode is layered like a modem, because it is one:

```
┌────────────────────────────────────────────────────────┐
│ L4  Transport      RaptorQ fountain code (RFC 6330)    │  no ACK, ever
├────────────────────────────────────────────────────────┤
│ L3  Framing        frame header, ESI, per-stripe CRC   │
├────────────────────────────────────────────────────────┤
│ L2  Modulation     complex field → RGB;                │
│                    payload: cell grid (Mode A) or      │
│                    Hermite–Gauss modes (Mode B)        │
├────────────────────────────────────────────────────────┤
│ L1  Geometry       Zadoff–Chu sync frame, quiet zone,  │
│                    homography recovery                 │
├────────────────────────────────────────────────────────┤
│ L0  Physics        display → air → lens → sensor       │
└────────────────────────────────────────────────────────┘
```

Plus one out-of-band element: the **calibration profile** — a 32-character
code displayed by the receiver and typed once by a human into the transmitter
(§7). This replaces the feedback channel with a single manual round-trip.

Two operating contexts share every layer above L2:

* **Static PsiCode** — a single printed/displayed symbol (like a QR code).
* **Streaming PsiCode** — a display cycling frames to transfer a file.

---

## 2. Design principles (informative)

1. **Blur is the enemy; treat it in the transform domain.** Defocus is a
   low-pass convolution. Hermite–Gauss functions ψₙ are eigenfunctions of the
   Fourier transform, so a Gaussian-ish blur attenuates modes ordered by index
   n instead of mixing them. Low modes survive; information degrades
   *progressively*, not catastrophically.
2. **Color carries the complex plane.** Incoherent optics transmits only
   non-negative intensity — phase is lost. PsiCode restores a two-dimensional
   signal space by mapping Re and Im to orthogonal color axes. Spatial blur
   acts identically and independently on each color channel and never mixes
   channels, so the complex structure of the field survives arbitrary defocus
   exactly (§5).
3. **No feedback.** The receiver may miss any subset of frames. RaptorQ makes
   frame identity irrelevant: any ≈K·(1+ε) received symbols reconstruct the
   payload.
4. **Everything the receiver can measure per-frame is not configuration.**
   White balance, black level, homography — recovered from in-frame reference
   elements every frame. Only parameters that change *transmitter behavior*
   go into the calibration profile.

---

## 3. L1 Geometry **[DRAFT]**

### 3.1 Canvas and quiet zone

A PsiCode symbol is square, `S × S` display pixels. A quiet zone of uniform
mid-gray (128,128,128) MUST surround the symbol. Quiet zone width presets
(selected by profile field `quiet_zone`):

| value | width |
|---|---|
| 0 | 2 cells |
| 1 | 4 cells (default) |
| 2 | 6 cells |
| 3 | 8 cells |

### 3.2 Zadoff–Chu synchronization frame

The border of the symbol is a binarized Zadoff–Chu pattern used for
detection, localization, and homography recovery.

Base sequence, length `N` (odd), root `q`, gcd(q, N) = 1:

```
z[n] = exp(−j·π·q·n·(n+1)/N),   n = 0 … N−1
```

Two border editions are defined. **v1 is the recommended edition**; v0 is
retained for compatibility with deployed v0 symbols and is not recommended for
new implementations.

#### v0 — double inverted ring **[DEPRECATED]**

* `N = 61`, roots `q = 1…4` on top / right / bottom / left in traversal order.
* Quiet zone of 4 cells on every side; symbol side 69 cells.
* Border thickness 2 cells, the **inner ring inverting the outer**.

The inversion is the defect that motivated v1. Under defocus the two rings sum
towards mid-grey, so the correlation collapses: measured retention at 1-cell
blur is **0.350**, below the 0.45 acceptance used in practice. On real captures
v0 true positives score 0.386–0.455 against a clutter band reaching 0.338 —
i.e. they lie *inside* it, so v0 cannot support a tight acceptance gate at all.

#### v1 — extruded strips, no quiet zone **[DRAFT]**

* `N = 61`, symbol side **61 cells — there is no quiet zone**.
* Four strips of `N × 2`. The sequence runs **along** each side and every
  element is painted **2 cells deep, perpendicular to that side**, so both rows
  of a strip are identical index for index.
* Roots by side: **top 3, right 1, bottom 4, left 2** — the horizontal pair
  {3,4} and the vertical pair {1,2}, the two groups differing by one. The pair
  identifies the axis; which member of the pair identifies the 180° case.
* Corner rule: each side owns the corner it walks into and carries indices
  `2…N`, yielding indices 0 and 1 to the preceding side. This is an exact
  partition — `4·(N−2)·2 = 8N−16` equals the cell count of a thickness-2 ring,
  with no overlap and no gap — and every side gets exactly `N−2` samples.
  At a corner the sides cross at 90°, so a yielded cell at position `i`,
  depth `d`, carries the neighbour's index `N−1−d`, **not** `N−2+i`.

Why extrusion rather than a second concentric ring: two concentric rings have
sides of `N` and `N−2` cells and cannot be aligned index for index, so blur
would mix a sequence with a shifted copy of itself. Extruded strips make blur
*perpendicular to a side* average two identical values — pure gain. Measured
retention at 1-cell blur is **0.702 against v0's 0.350**.

Why no quiet zone: a quiet zone serves detectors that key on contrast against
the background. A ZC border localises its own boundary by correlation, which is
why Aztec Code needs none either. Measured: with the border's outer row directly
adjacent to real captured screen content, scale recovers with **0.0 % error**;
the surround does not enter the fit. Dropping it takes the symbol from 69 to 61
cells, i.e. **+13.1 % linear cell size and +27.9 % area at a fixed footprint**.

**Structural property implementers must know: the ZC sequence is its own
reverse.** For odd `N`, `(N−1−n)(N−n) ≡ n(n+1) (mod 2N)`, hence
`z[N−1−n] = z[n]`. Verified for N ∈ {31, 37, 61}, all roots, complex and
binarised. Three consequences:

* correlating against a reversed template is the *same* test, so a reversal
  dimension in a correlation search is vacuous;
* **a single side cannot determine which way it runs, so 180° rotation is
  unresolvable from one side.** It is resolved *only* by the per-side root
  assignment above, which is therefore load-bearing rather than decorative;
* every acquisition seed must spawn two interior-direction hypotheses.

Binarisation costs measurable margin: cross-root correlation is **0.3524**
against the ideal `1/√N = 0.128`, i.e. 8.8 dB discarded. Adjacent roots are not
worse than distant ones — (3,4) = 0.2470, (2,3) = 0.2618, (1,2) = 0.3524, while
(1,5) = 0.4071. A complex carrier recovers that 8.8 dB in simulation, but see
§6.5 for why it is not used.

Rationale (informative): ZC autocorrelation is used exactly for what it is
good at — 1-D shift estimation along each detected border, which is a
cyclic-shift problem, not a 2-D data-coding problem. The four corners fix the
homography; per-side correlation refines it.

Receiver procedure (normative sketch): correlation search over a scale ladder →
candidate quadrilateral → **joint** four-side refinement → root assignment and
orientation → homography → sub-cell descent.

**Two traps, both consequences of the extrusion, both already encountered:**

* A uniform 2-cell strip is deliberately insensitive to displacement
  perpendicular to itself. Combined with keystone, any geometry error yields
  "two sides at 0.93, two at 0.20". This is fixed **only** by fitting all four
  sides jointly, because one pair's along-axis is the other pair's
  perpendicular. Implementations that fit sides independently will reproduce the
  failure. Read "two good sides" as a symptom, never as partial success.
* Local uniformity is exactly the absence of localisation information. Probing
  only the strip's midline leaves scale under-determined by ~0.3 % — 0.2 cells
  over 61 — and on sharp captures the objective has a flat plateau unless
  tangential probes at ±0.3 cell are used. Those probes must be gated to the
  2-cell strip; they degrade a 1-cell v0 ring.

**Sampling MUST follow the homography, not the side.** Under a homography,
equal steps along a segment are not the image of a uniform cell grid, and the
along-side foreshortening of a side is governed by the perspective coefficient
of the *other* axis. A keystone invisible on one pair of sides is therefore
fatal on the perpendicular pair: measured on a live quad with `h = −0.0762`,
`g = +0.0015`, mid-side lands at 0.480 instead of 0.500 — **1.2 cells of lag**
on the left and right sides against 0.012 on top and bottom, against a
correlation peak one cell wide. Border and payload MUST use the same geometric
model; a border score is not evidence of payload alignment otherwise.

### 3.3 Interior layout

Inside the double ZC ring:

```
┌──────────────────────────────────────────┐
│ ZC ring (2 cells)                        │
│ ┌──────────────────────────────────────┐ │
│ │ reference strip (1 cell tall) §3.4   │ │
│ │ ┌──────────────────────────────────┐ │ │
│ │ │                                  │ │ │
│ │ │        payload region            │ │ │
│ │ │   Mode A grid  or  Mode B modes  │ │ │
│ │ │                                  │ │ │
│ │ └──────────────────────────────────┘ │ │
│ │ frame-counter strip (streaming) §6.3 │ │
│ └──────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### 3.4 Reference strip **[DRAFT]**

One cell-row directly below the top ZC ring, repeated every frame:

`K W R G B C M Y K W` + 6-step gray staircase, then repeating.
(K = black, W = white(at configured white level), primaries, secondaries.)

The receiver MUST derive per-frame from this strip: black/white levels,
a 3×3 color correction matrix, and per-channel gain. No frame is decoded
against stale color state.

**The strip is one cell tall, which is the most fragile spatial scale in the
symbol**, and that is a real limitation rather than a detail. The ISP's chroma
low-pass smears adjacent colour patches into one another before it touches
anything else, so a channel matrix estimated from a one-cell strip is corrupted
in exactly the regime where it matters. Measured, by estimating the same matrix
from patches of different sizes: residual after applying the fitted matrix is
0.170 at cell scale, **0.020 at 120 px**, and 0.173 again at 600 px where the
illumination field takes over. The strip therefore SHOULD be demoted to
bootstrap and fallback, with the primary estimate coming from §4-IB.

#### §4-IB In-band calibration frames **[DRAFT — recommended]**

Whole frames interleaved into the stream, carrying colour patches at a scale
that survives the ISP, on a **doubling schedule** — frames 0, 1, 2, 4, … 128,
then every 128. Overhead is **0.79–1.5 %**, and a receiver joining at frame `t`
waits at most until `2t`, worst case 127 frames.

Patch size **120 px** is not "large enough" but the measured optimum of the table
above. Each patch class MUST appear at multiple **point-symmetric** positions, so
that any linear illumination field cancels exactly in the mean; note this cancels
only the *odd* part of the field, and because a symmetric pair sits at one
radius, colour and radius become perfectly correlated — a field polynomial fitted
from the patches' own residuals is then **not identifiable**. Probe the field
with a neutral moat co-located with each patch instead.

A calibration frame MUST be identifiable **without calibration**, since a CRC'd
header cannot be parsed on an uncalibrated channel. The marker SHOULD be large,
binary and luma-only. The construction used here places `m, ¬m, m` on three rows
and takes the **vertical second difference** `½(L₀+L₂) − L₁` on green, which
cancels any y-linear field identically and turns curvature into a constant
removed by mean subtraction: calibration frames score 0.960–0.9997, payload
frames ≤ 0.31, with zero false positives across a full transfer.

Measured accuracy of the recovered channel matrix, against the one-cell strip:

| σ luma | σ chroma | reference strip | calibration frame |
|---|---|---|---|
| 1 | 1 | 0.0040 | 0.0054 |
| 1 | **3 (measured)** | 0.1952 | **0.0054** — ×36 |
| 2 | 3 | 0.1296 | 0.0067 — ×19 |
| 3 | 10 | 0.2948 | 0.0349 — ×8.4 |

The calibration frame is flat across the whole sweep (0.005–0.035) where the
strip ranges over 0.004–0.476.

**A calibration frame is the wrong instrument for a cell-scale quantity**, and
implementations MUST NOT use it whole to estimate one. Its uniform tiles are the
majority of its cells, and inside a uniform patch the response is `DC(h)·x` for
*any* kernel shape — those cells constrain only the tap sum while numerically
outvoting the informative ones. Measured: estimating an interference kernel from
the unmasked frame collapses to identity (max tap deviation 0.1260, exactly the
largest true tap); masked to the marker band and known surround, 0.0143.

---

## 4. Calibration test pattern **[DRAFT]**

Transmitter mode `--calibrate` displays a single static pattern (≥ 5 s)
containing:

1. Full ZC frame (§3.2) — exercises detection.
2. **Frequency wedge**: vertical stripe pairs at pitches
   64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2 px. Receiver finds the finest pitch
   with Michelson contrast ≥ 0.4 ⇒ `mtf_limit_px`.
3. **Staircases**: 16-step gray, 16-step R, 16-step B ⇒ per-channel gamma
   fit (`gamma_g_q`, `gamma_r_delta_q`, `gamma_b_delta_q`), noise σ per step
   ⇒ `noise_sigma_q` ⇒ recommended `luma_bits`; inter-channel leakage ⇒
   `crosstalk_rg_q`, `crosstalk_gb_q`.
4. **White patch** (≥ 10% of area) ⇒ clipping/blooming ⇒ `white_level_q`.
5. **Animated corner counter**: a stripe whose binary frame number changes
   every display refresh, duplicated top/bottom. Receiver captures 2–3 s and
   measures the fraction of captures containing two different numbers
   (rolling-shutter tearing) ⇒ `torn_frames_q` ⇒ recommended
   `frame_hold_periods`.

The receiver then displays the 32-character profile code (§7). The human
types it into the transmitter. Done.

Reference pattern layout v0 (DRAFT): a 61×61-cell canvas (cell = size/61) —
double ZC ring (§3.2); interior 57×57 cells stacked top-to-bottom as:
frequency wedge, 11 bands × 3 cell-rows (pitches in **absolute display px**,
so `mtf_limit_px` is size-independent); 16-step gray, R-only and B-only
staircases (4 cell-rows each, drive = round(255·i/15)); white patch,
12 cell-rows (≈18 % of area). The animated counter (item 5) runs as a
separate phase; the static pattern covers items 1–4. Estimator note: the
γ_G fit SHOULD correct for crosstalk leakage into G (fit γ_R/γ_B from their
staircases first, then fit G against the leakage basis
u^γG + x_rg·u^γR + x_gb·u^γB) — an uncorrected single-power fit is biased
low by several quantization steps.

**Recalibration** (informative): the profile code MAY be issued again later.
In particular, if a streaming transfer ends with a failing payload checksum
(§6.2), the receiver SHOULD display a fresh profile code whose telemetry was
measured over the entire failed transfer — the whole session becomes the
test pattern. The human types the new code, the transmitter re-adjusts
(larger cells, longer frame hold, fewer bits per cell) and retransmits. This
manual round-trip is the only feedback path in PsiCode.

---

## 5. L2 Modulation

### 5.1 Complex-to-color mapping **[DRAFT]**

Let `f = Re + j·Im` be the normalized complex field value at a pixel,
`Re, Im ∈ [−1, +1]`. With mid-point `M = 128`, luma amplitude `A_L` and
chroma amplitude `A_C` (both derived from profile: white/black levels and
`chroma_mode`):

```
G = M + A_L · Re
R = M + A_L · Re + A_C · Im
B = M + A_L · Re − A_C · Im
```

Decoding (after reference-strip color correction):

```
Re ≈ (G − M) / A_L
Im ≈ (R − B) / (2 · A_C)
```

Properties (informative):

* Spatial blur commutes with this mapping channel-wise: the received field
  is the *convolved complex field*, not a scrambled one.
* Im is antisymmetric in R/B ⇒ (a) luma Y ≈ 0.3R+0.6G+0.1B is nearly
  Im-blind, so Re rides the full-resolution luma path of any camera
  pipeline; (b) the effective Im blur kernel is (K_R+K_B)/2, which
  chromatic aberration pushes *toward* K_G, partially self-compensating.
* The DC offset M is a zero-frequency component — untouched by any blur —
  and is subtracted using the reference strip.

`chroma_mode` values (profile field):

| value | name | meaning |
|---|---|---|
| 0 | Mono | A_C = 0; Im axis unused |
| 1–3 | Chroma1..3 | 1..3 bits of Im resolution |
| 4 | GreenOnly | R = B = M always; luma only, aberration-proof |
| 5–7 | ConstLuma1..3 | §5.1-CL below; 1..3 bits on the Im axis |

#### §5.1-CL Constant-luminance mapping **[DRAFT — recommended]**

The mapping above places `Re` on **absolute luminance**, which is the one axis a
receiver's illumination field destroys. Measured live, luma drifts **0.62 → 0.86
across a single frame**. The constant-luminance form places the complex value
entirely in chromaticity and holds total drive constant:

```
transmit:  R = u·(1 − b·x + c·y)     G = u·(1 + 2b·x)     B = u·(1 − b·x − c·y)
           so R + G + B = 3u = S, identically, for every z = x + jy
receive:   S_meas = R + G + B                    (MEASURED, per cell)
           x = (2G − R − B) / (2·S_meas·b)       Re rides green ↔ magenta
           y = 3·(R − B)    / (2·S_meas·c)       Im rides red ↔ blue
```

**Dividing by the measured per-cell channel sum is the whole mechanism.** Any
multiplicative field λ(x, y) scales all three channels alike and cancels
*identically* in the ratio — per cell, for a field of any spatial complexity, with
no fitting and no pilots. Using the nominal `S` instead of the measured one
destroys the invariance completely; implementations MUST use the measured sum.

Constants derived from the profile's black and white levels; for the reference
profile `u = 130`, `S = 390`, `b = 0.4808`, `c = √3·b = 0.8327`.

`c = √3·b` is not a tuning choice. Under iid channel noise
`Var(2G−R−B) = 6σ²` and `Var(R−B) = 2σ²`, a ratio of `√3`, and the signal swings
`6ub` and `2uc` are in the same ratio at that value — so the two axes carry equal
SNR by construction. It also makes the gamut constraint a single radius: the
feasible region is a regular hexagon whose inscribed circle is the unit disk.

A square Gray lattice must be scaled by **`2/(1+√3) ≈ 0.732`** so its corners sit
exactly on that hexagon; this is exact, not padding.

Measured gain over §5.1: **+5 to +6 bits per patch** under the measured field,
nothing lost on a clean channel, and no crossover up to 12 % per-channel
chromatic tilt in either orthogonal mode.

**The 3×3 channel matrix of §3.4 is mandatory for this mapping.** `2G−R−B` gets
no cancellation from a neutral-preserving channel mix, whereas `R−B` cancels one
identically. A device measured at `B→G = −0.26` therefore reads Im correctly and
Re not at all: BER 0.19–0.25 on Re against 0.004–0.03 on Im, restored to
0.007–0.09 once the matrix is applied.

**Implementation note on axis balance.** An apparent axis imbalance is a symptom
of inter-cell interference (§5.2), not of the channel. Before equalisation the
measured per-channel σ ratio is 2.18 and the SNR-optimal `c/b` is 1.27 — the
*opposite* direction from what the raw margins suggest; after equalisation the
ratio falls to 1.82 and the optimum returns to 1.85, within 7 % of `√3`.
Implementations SHOULD NOT retune `c` to compensate for interference; equalise
instead.

### 5.2 Mode A — cell grid **[DRAFT]**

The MVP payload. The payload region is a grid of `cell_size_px` cells. Each
cell carries one symbol:

* Re quantized to `2^luma_bits` levels (Gray-coded).
* Im quantized per `chroma_mode` (Gray-coded), if enabled.

Cells are sampled at their centers after homography; a 2×2 subsample average
MUST be used when `cell_size_px ≥ 8` camera pixels.

#### Inter-cell interference

Adjacent cells bleed into one another, and at cell scale this — not colour
fidelity, not calibration — is what limits a dense payload. Measured on real
captures by rendering uniform blocks of decreasing size and fitting the residual
after removing the linear channel distortion:

| block | residual | within-block noise |
|---|---|---|
| 636 px | 0.025 / 0.067 | 0.026 / 0.041 |
| 120 px | 0.013 / 0.027 | 0.005 / 0.007 |
| ~2 cells | **0.078 / 0.080** | 0.007 / 0.011 |

At block sizes well above a cell the residual sits at the noise floor; at cell
scale it is **ten times** the noise floor. The axis gains meanwhile barely move
with scale, so the distortion itself is scale-independent — it is the
*separability* of neighbours that fails.

**The interference is deterministic and therefore invertible.** A linear model
over the 5×5 neighbour pattern explains R² = 0.516 of the fixed-pattern spread;
sub-pixel phase and moiré contribute nothing (R² 0.516 → 0.522). Measured kernel
strength — the fraction of a cell's light arriving from its neighbours — is
**0.23–0.27 for a chromatic payload against 0.053 for 1-bit luma**. That fivefold
difference is why a 1-bit mono payload has always worked while a two-axis
chromatic one at the same geometry fails: interference eats *margin*, and noise
converts margin into errors.

Equalisation SHOULD be performed **between channel decoupling and the gamma
inverse**, on the light-linear quantity, because that is where the convolution is
actually linear. Measured on a device where it binds: stripes passing CRC
**10/32 → 30/32**, Re-axis SER 0.0159 → 0.00018. Live, at the cell size where
margin is thin, full-stripe frames go from 55.6 % to 69.9 %; at a comfortable
cell size the difference vanishes, which is the correct behaviour — the
equaliser's own noise gain is ×1.01–1.05 and is wasted where interference does
not bind.

The kernel is a property of the display, optics and ISP, and is **fixed and
cacheable**: across 23 static captures the median per-tap deviation is 0.0006,
and a pooled median kernel outperforms per-frame estimation. Implementations
SHOULD cache it rather than re-estimate it per frame.

Two notes for implementers. Interference is anisotropic — edge-to-diagonal ratio
2–15×, median ≈5×, channel-dependent — but a cross-shaped kernel measures
slightly *worse* than a full one, because with thousands of cells against fifteen
parameters the diagonal taps are estimated stably. And the inverse is
well-conditioned: |H(ω)| ∈ [0.75, 1.25] with no near-zeros and a noise-variance
gain of ×1.008–1.051, so no regularisation is required. This is a weak low-pass,
not a channel with spectral nulls.

### 5.3 Mode B — Hermite–Gauss modes **[EXPERIMENTAL]**

The payload region (or blocks of it) carries data as complex coefficients
of 2-D Hermite–Gauss functions ψ_{m,n} — eigenfunctions of the Fourier
transform, so defocus attenuates coefficients monotonically in mode order
m+n instead of mixing them. Low modes survive; the receiver decodes the
mode subset whose measured SNR clears threshold, and the transmitter
assigns data to modes in significance order. The full construction (block
geometry, pilot and probe modes, equalization, coefficient mapping) and the
research directions built on it live in [RESEARCH.md](RESEARCH.md);
parameters enter this spec (→ DRAFT → STABLE) only after live measurements
of coefficient SNR vs. blur σ on real display/camera pairs.

### 5.4 Static PsiCode symbol **[EXPERIMENTAL]**

A single Mode B symbol with the payload protected by RS over GF(256)
(parameters TBD after capacity measurements). Intended use: short IDs/URLs
readable at extreme defocus where QR fails. Not a QR replacement for
capacity — a complement for robustness.

---

## 6. L3 Framing & L4 Transport (streaming) **[DRAFT]**

### 6.1 Transport

* RaptorQ per RFC 6330. One source block per transfer v0.
* The transmitter cycles encoding symbols indefinitely (systematic symbols
  first, then repair) until stopped. Overhead preset from profile field
  `fec_overhead`:

| value | repair stream behavior |
|---|---|
| 0 | source ×1 then endless repair |
| 1–7 | interleave repair every 2^value source symbols |

### 6.2 Frame layout

Each displayed frame's payload region carries, in raster order:

```
FrameHeader {
  magic:      u16   // 0x03A8 ("Ψ" codepoint 0x03A8)
  version:    u8
  flags:      u8
  session_id: u32   // random per transfer, constant within it
  esi:        u24   // encoding symbol ID of first symbol in frame
  count:      u8    // symbols in this frame
}                   // 12 bytes
TransferInfo (in every 8th frame) {
  transfer_length: u40, symbol_size: u16, K: u24, checksum: u32 (CRC-32C)
}
symbols…  // each stripe of H/8 cell-rows ends with CRC-16/CCITT
```

Bit packing (v0, normative): the 57×55-cell payload region is split into 8
row-stripes (7,7,7,7,7,7,7,6 rows). Within a stripe, cells in raster order
emit `bits_per_cell` bits MSB-first; the **last 16 bits of each stripe** are
CRC-16/CCITT (poly 0x1021, init 0xFFFF, no reflection, no final XOR) over
all preceding bits of that stripe. The data region carries whole bytes only
(⌊(cap−16)/8⌋ bytes MSB-first; leftover bits before the CRC are zero
padding), so **no byte spans a stripe boundary** — stripe salvage (§6.3)
never yields torn bytes. The frame byte stream — FrameHeader ∥
[TransferInfo] ∥ encoding-symbol bytes — fills stripe data regions 0→7
contiguously. All multi-byte fields big-endian; `flags` bit 0 =
TransferInfo present.

Per-stripe CRC lets a torn capture (§6.3) salvage its intact stripes.

`session_id` is drawn at random by the transmitter for each transfer and
never changes within it. The receiver MUST discard symbols whose
`session_id` differs from the current session's; a new `session_id`
observed in ≥ 3 consecutively decoded frames starts a new transfer context.
Rationale (informative): without it, stopping transfer A and starting
transfer B mid-capture feeds mixed symbols to the fountain decoder, which
can converge to garbage that passes size checks.

### 6.3 Timing

* Frame hold time = `frame_hold_periods` × display refresh period.
* Transmitter SHOULD hold each frame ≥ 2 receiver exposure periods
  (calibration measures this; default 6 periods ⇒ 10 fps at 60 Hz).
* The frame-counter strip (§3.3) carries the low 8 bits of the frame
  sequence number, duplicated at strip start and end (v0: 8 black/white
  cells at each end of the row, MSB-first — first cell = bit 7, white = 1;
  the middle of the row stays mid-gray). A counter mismatch
  means the capture is torn: it is **two partial frames** — frame N above
  the tear, frame N+1 below it — and the two counter copies identify both
  numbers.
* A torn capture is not discarded. Per-stripe CRC-16 (§6.2) localizes the
  intact stripes; the receiver SHOULD attribute intact stripes above the
  tear to frame N and below it to frame N+1 and feed both partial symbol
  sets to transport. (The ESI range of frame N+1 is predictable from frame
  N's header, since the transmitter emits ESIs in the deterministic §6.1
  order.) Expected gain under heavy tearing: +20–30 % goodput (informative).

### 6.4 Capacity (informative, v0 targets)

1080p, cell 16 px, Mode A, 3 bits luma + 2 bits chroma, 10 fps:
≈ 100×56 cells × 5 bit × 10/s ≈ 280 kbit/s raw; ≈ 100–150 kbit/s goodput
after framing, FEC and loss. Numbers to be replaced by measurements
(→ [BENCHMARKS.md](BENCHMARKS.md)).

### 6.5 Receiver requirements **[DRAFT]**

Demodulators SHOULD hand soft observations upward — per-symbol level
likelihoods, or at minimum value + confidence — rather than hard bytes;
hard slicing is a lossy compatibility shim. Receivers SHOULD log per frame
the diagnostic vector `FrameQuality {detection, geometry, color,
modulation, integrity}`, each component in [0, 1].

The requirements below were derived from live operation against real phone
cameras, not from theory. Each one cost a debugging session.

#### Exposure and the ISP

* Exposure MUST be pinned to approximately **one display refresh period**
  (16.7–20 ms at 60 Hz). Shorter produces rolling-shutter banding; longer blends
  adjacent transmitter frames.
* The ISP's temporal filters (noise reduction, edge enhancement, video
  stabilisation) MUST be disabled. Even so, residual frame mixing exists; its
  coefficient was back-solved as **α ≈ 0.07**, small enough that it is *not* what
  forces a long frame hold — the exposure window landing on frame boundaries is.
* Rolling shutter **helps** and MUST NOT be designed away: row-time diversity
  gives more chances to catch a clean band. A global shutter measured *more*
  errors, and shortening the symbol makes tearing worse, not better, because
  monitor and camera both scan top to bottom and a short symbol lets the two
  scans track each other.

#### Focus

* Contrast autofocus is **unusable** on a time-varying code: it hunts to infinity
  and sticks. Receivers MUST drive the lens manually and SHOULD cache the working
  position across sessions.
* The sweep range MUST be bounded below. Infinity and one metre are pure lens
  travel; a screen sits at roughly 11–65 cm.
* Sweep steps MUST be counted in **frames, not wall-clock time** — a time-based
  window closes before a failed acquisition returns.
* **The focus objective MUST be measured at cell scale on the payload, never on
  the border.** This is the requirement most easily got wrong. A blur-tolerant
  border is by design insensitive to the defocus that destroys the payload:
  measured live, the v1 border score spans only **0.897–0.943** across a focus
  range over which payload stripes go from 8 to 0. A sweep driven by it
  early-exits into an unreadable lock.

  The recommended objective is a **cell-scale modulation ratio**: over known cells
  whose ideal level differs from both neighbours, take the second difference
  `L(n) − ½(L(n−1) + L(n+1))` and divide by the known white-to-black spread. The
  second difference cancels any linear field identically; the division cancels
  gain and veil. Measured behaviour:

  | σ (cells) | 0.0 | 0.3 | 0.4 | 0.5 | 0.7 | 1.0 | 2.0 |
  |---|---|---|---|---|---|---|---|
  | metric | 1.031 | 1.026 | 0.972 | 0.845 | 0.523 | 0.211 | 0.016 |
  | stripes | 8/8 | 8/8 | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |

  Strictly monotone, dynamic range ~64×, and **still 0.211 where SER is 0.64** —
  a usable gradient deep inside the unreadable region, which is what a sweep
  starting out of focus needs. The readability cliff sits between 0.97 and 0.85;
  target ≥ 0.95.

  Two plausible alternatives were measured and MUST NOT be used alone: kernel
  strength rises then collapses to zero past the cliff, so a sweep driven by it
  hunts into defocus and sees the objective improve; and a self-referential
  decision-margin metric turns back up past σ ≈ 1.0.

* A **temporal-blend detector is a separate instrument** and cannot be the focus
  metric: border and reference row are identical in adjacent transmitter frames,
  so a blend leaves the cell-scale metric untouched while destroying the payload.
  A decision-margin statistic separates them (1.55 and 1.70 on blended frames
  against 2.19–2.57 on clean ones). The pair is the diagnostic — both low means
  defocus; cell-scale metric high with low margin means a blend.

#### Timing and rate

* The camera MUST be polled at least as fast as the transmitter changes frames.
  A receiver polling every 120 ms against a 100 ms frame never samples some
  frames at all; live decode costs 25–36 ms, so a slower poll is pure loss.

#### Carrier choice for the border

The border SHOULD be carried in **luma**, not chromaticity, despite the 8.8 dB of
root separation a complex carrier recovers in simulation. On real hardware a
chroma-carried border produced **0 detections in 30 frames** with focus verified
correct, while the luma carrier gave 100 % on the same rig. Chroma is subsampled
4:2:0 and additionally low-passed by the ISP at σ ≈ 3.2 px against ~2 for luma,
and the border needs sub-cell accuracy. The margin the complex carrier buys is
one the luma carrier does not need: measured orientation margin is 0.87 against
a 0.10 threshold.

---

## 7. Calibration profile code **[STABLE]**

Implemented in `psicode-core`; frozen.

### 7.1 Outer format

32 symbols, Base32 — 160 bits total (20 bytes), displayed in 8 groups of 4:

```
XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX
```

**Alphabet** (values 0…31, excludes I, O, S, Z):

```
0123456789ABCDEFGHJKLMNPQRTUVWXY
```

Input normalization (receiver-of-typing side MUST apply): case-insensitive;
`O→0`, `I→1`, `S→5`, `Z→2`; `-`, space, en/em dash ignored.

### 7.2 Error correction

Two interleaved Reed–Solomon codewords, each RS(16, 8) over GF(32) — rate
exactly 1/2. (An RS codeword over GF(32) cannot exceed 31 symbols; the
32-symbol code is therefore built from two.)

* Field: GF(2⁵), primitive polynomial `x⁵ + x² + 1` (0b100101).
* Generator roots: α⁰ … α⁷ (fcr = 0), systematic encoding; each codeword =
  8 payload symbols ∥ 8 parity symbols (RS(31, 23) shortened by 15).
  Polynomial convention: highest-degree coefficient first; within a
  codeword, index i ↔ term x^(15−i).
* Interleaving: displayed symbol 2i is codeword A symbol i; displayed
  symbol 2i+1 is codeword B symbol i (i = 0…15). Codeword A carries
  payload symbols 0–7, codeword B carries payload symbols 8–15.
* Each codeword corrects ≤ 4 symbol errors ⇒ the full code corrects **any**
  ≤ 4 errors, up to 8 when they split evenly between A and B, and any
  contiguous run of ≤ 8 mistyped symbols (interleaving splits a run 4/4 —
  two adjacent fully garbled 4-symbol groups are recoverable).
* Decoders MUST verify zero syndromes after correction in both codewords
  and MUST verify the payload CRC-8 (§7.3); on either failure the code is
  rejected (no silent miscorrection).
* Decoders MUST NOT crash on arbitrary input: every failure path (wrong
  length, invalid character, uncorrectable errors, CRC mismatch) maps to a
  defined error.

### 7.3 Payload — 80 bits

16 five-bit symbols, big-endian bit order (first symbol = most significant
bits). Bits 0–71 are fields; bits 72–79 are CRC-8 (poly 0x07, init 0x00,
no reflection) computed over the 9 field bytes (big-endian).

| # | field | bits | encoding → physical value |
|---|---|---|---|
| 1 | `version` | 4 | format version; this spec: 1 |
| 2 | `cell_size_px` | 6 | stored−2 → 2…65 px |
| 3 | `frame_hold_periods` | 4 | stored−1 → 1…16 refresh periods |
| 4 | `luma_bits` | 2 | stored−1 → 1…4 bits/cell |
| 5 | `chroma_mode` | 3 | §5.1 table; 5–7 reserved |
| 6 | `gamma_g_q` | 6 | γ_G = 1.500 + 0.025·q |
| 7 | `gamma_r_delta_q` | 4 | γ_R = γ_G + 0.025·(q−8) |
| 8 | `gamma_b_delta_q` | 4 | γ_B = γ_G + 0.025·(q−8) |
| 9 | `white_level_q` | 4 | white = (55 + 3q) % of full drive |
| 10 | `black_level_q` | 4 | black lift = q % |
| 11 | `noise_sigma_q` | 5 | σ = 0.25 · 2^(q/4) gray levels |
| 12 | `mtf_limit_px` | 5 | stored−1 → 1…32 px finest resolvable pitch |
| 13 | `torn_frames_q` | 4 | 0 → 0%; else 0.1 · 2^(q−1) %, cap 100 |
| 14 | `crosstalk_rg_q` | 4 | 2q % |
| 15 | `crosstalk_gb_q` | 4 | 2q % |
| 16 | `quiet_zone` | 2 | §3.1 table |
| 17 | `fec_overhead` | 3 | §6.1 table |
| 18 | `reserved` | 4 | MUST be zero in v1; receivers MUST ignore |
| 19 | `crc8` | 8 | §7.3 |

Fields 6–15 are **telemetry** (receiver measurements); 2–5, 16–17 are
**prescriptions**. v1 transmitters MAY recompute prescriptions from
telemetry; the typed prescriptions are the fallback.

### 7.4 Reference vectors

```
profile: version=1, cell=16, hold=6, luma_bits=3, chroma=Chroma2,
         gamma_g_q=28 (γ=2.200), r_delta=8, b_delta=10, white_q=15 (100%),
         black_q=2, noise_q=12, mtf=6, torn_q=5, xtalk_rg=3, xtalk_gb=4,
         quiet=1, fec=2
code:    26E2-BM46-VHH8-B6R3-8XP4-HBNK-PJCD-GHF7
```

A decoder MUST accept `26e2 bm46 vhh8 b6r3 8xp4 hbnk pjcd ghf7` (case,
spaces) and MUST recover the profile from any 4-symbol corruption of the
code, including two adjacent fully garbled 4-symbol groups.

---

## 8. Reference implementation map

| crate | contents | status |
|---|---|---|
| `psicode-core` | §7 complete: GF(32), 2 × RS(16,8) interleaved, Base32, bit packing, `CalibProfile`; no_std + alloc, fuzz-tested no-panic decode | done |
| `psicode-core` `symbol` | §3 render (ZC gen, ref strip, frame counter), §5.1 color map, §5.2 Mode A mod/demod | done |
| `psicode-core` `detect` | §3.2 ZC detection: quad search, per-side correlation, sub-cell full-ring fine-alignment, orientation, homography | done |
| `psicode-core` `l3` | §6.2 framing: header/TransferInfo, per-stripe CRC-16, cell bit-packing, salvage parser | done |
| `psicode-core` `calibrate` | §4 pattern render + channel estimators (γ, MTF, noise, crosstalk, levels) + prescription heuristic | done |
| `psicode-sim` | channel sim (gamma, homography, per-channel blur, crosstalk, gain/offset, noise, rolling-shutter tearing); SER/FER/goodput/sensitivity/detected-vs-genie sweeps → BENCHMARKS.md | v0 done (workspace: 87 tests) |
| `psicode-core` `fountain` | interim XOR-fountain transport (EXPERIMENTAL, §9.2; ε ≈ 1 %) + CRC-32C; replaced by RaptorQ before freeze | done |
| `psicode-sim` `transfer` / `modeb` | end-to-end file transfer over the full stack (BENCHMARKS §8); Mode B freeze-gate measurements (RESEARCH §1) | done (workspace: 101 tests) |
| `psicode-tx` | Windows 11 transmitter (winit 0.30 + softbuffer 0.4): `calibrate` (§4 pattern + animated counter, console profile-code entry), `stream` (fountain §6.1–6.3), `single` | done (workspace: 107 tests) |
| `psicode-sim` `live` | offline decode of phone photos: full uncropped frames, tilt-robust (rotating-calipers corners), SER 0.057–0.105 (BENCHMARKS §5) | first light 2026-07-26 |
| `psicode-rx` + `psicode-android` | Android receiver: Camera2 (pinned exposure ≈ 1 refresh, decoder-guided focus sweep, ISP filters off), acquire/track detection, local-threshold demod, live L3+fountain | **first live file transfer 2026-07-26 (byte-exact, CRC-32C + SHA256)** |
| `psicode-core::{zcborder, acquire}` | §3.2 border v1 (extruded strips, no quiet zone) + ZC-correlation acquisition | detection on full uncropped frames **18 % → 100 %**; live byte-exact transfers on two devices |
| `psicode-core::{isi, calframe}` | §5.2 inter-cell equalisation; §4-IB in-band calibration frames | stripes **10/32 → 30/32** on a device that had never decoded; channel matrix **×36** more accurate |
| next | RaptorQ per RFC 6330; telemetry to xelth.com; hexagonal payload lattice (sim-validated at +12.8…16.6 % in d′, blocked on L3 cell ordering); live device matrix §9.3 | — |
| `psicode-rx` | Rust core for Android (JNI): detect → homography → demod → RaptorQ | — |
| `psicode-android` | thin Kotlin shell: Camera2 (locked AWB/AE/AF, YUV420 direct) | — |

Non-goals for v0: iOS, encryption (transport is cleartext; wrap your payload),
multi-source-block transfers, printed streaming.

---

## 9. Roadmap to freezing

1. `psicode-sim`: channel simulator (blur, noise, gamma, crosstalk,
   tearing, homography distortion); Monte Carlo SER/FER sweeps fill
   [BENCHMARKS.md](BENCHMARKS.md). The simulator is the iteration loop; the
   live channel is the validation gate.
2. Live channel bring-up: ZC frame + Mode A end-to-end. RaptorQ stays
   deferred until a single-frame Mode A decode works end-to-end in sim;
   interim transport MAY be a simple XOR-fountain, marked EXPERIMENTAL and
   replaced by RaptorQ (§6.1) before any freeze.
3. Measure live: SER vs distance/angle/blur; torn-frame statistics; color
   crosstalk on ≥ 3 display/phone pairs (→ BENCHMARKS.md).
4. Freeze §3, §5.1, §5.2, §6 (→ STABLE), bump to 0.2.
5. Mode B measurement campaign: coefficient SNR vs blur σ, grid vs modes
   (→ [RESEARCH.md](RESEARCH.md)).
6. Freeze §5.3/§5.4, publish 1.0.

---

*Copyright © 2026 xelth.com. This specification may be freely copied and
implemented. "PsiCode" and the Ψ logo are trademarks of xelth.com used to
identify conformant implementations.*
