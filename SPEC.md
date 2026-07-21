# PsiCode Specification

**Version 0.1.0-draft · 2026-07-21**

PsiCode (ΨCode) is an open, royalty-free visual code and one-way optical data
link. It transmits data from any display to any camera with **no feedback
channel**, using color-encoded complex fields, Hermite–Gauss mode payloads for
graceful degradation under defocus, a Zadoff–Chu synchronization frame, and
fountain-coded transport.

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

Frame construction v0:

* `N = 61`, `q = 1` (prime N; 61 cells per side is the v0 symbol size).
* Top border, left→right: cells `n = 0…N−1` colored by binarized phase of
  `z[n]`: white if `arg(z[n]) > 0`, black otherwise.
* Right border, top→bottom: same sequence with root `q = 2` (distinct root ⇒
  bounded cross-correlation ⇒ side identification).
* Bottom border, right→left: root `q = 3`. Left border, bottom→top: root
  `q = 4`. Traversal order gives unambiguous orientation.
* Border thickness: 2 cells (inner ring repeats the outer ring inverted,
  improving edge gradient under blur).

Rationale (informative): ZC autocorrelation is used exactly for what it is
good at — 1-D shift estimation along each detected border after coarse
detection, which is a cyclic-shift problem, not a 2-D data-coding problem.
The four corners fix the homography; per-side correlation refines it.

Receiver procedure (normative sketch): coarse quadrilateral detection →
per-side 1-D resampling → correlation against the four root sequences →
side/orientation assignment → homography → iterative refinement.

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

### 5.2 Mode A — cell grid **[DRAFT]**

The MVP payload. The payload region is a grid of `cell_size_px` cells. Each
cell carries one symbol:

* Re quantized to `2^luma_bits` levels (Gray-coded).
* Im quantized per `chroma_mode` (Gray-coded), if enabled.

Cells are sampled at their centers after homography; a 2×2 subsample average
MUST be used when `cell_size_px ≥ 8` camera pixels.

### 5.3 Mode B — Hermite–Gauss modes **[EXPERIMENTAL]**

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

Mode B parameters will be frozen (→ DRAFT → STABLE) only after live
measurements of coefficient SNR vs. blur σ on real display/camera pairs.

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
  magic:    u16   // 0x03A8 ("Ψ" codepoint 0x03A8)
  version:  u8
  flags:    u8
  esi:      u24   // encoding symbol ID of first symbol in frame
  count:    u8    // symbols in this frame
}
TransferInfo (in every 8th frame) {
  transfer_length: u40, symbol_size: u16, K: u24, checksum: u32 (CRC-32C)
}
symbols…  // each stripe of H/8 cell-rows ends with CRC-16/CCITT
```

Per-stripe CRC lets a torn capture (§6.3) salvage its intact stripes.

### 6.3 Timing

* Frame hold time = `frame_hold_periods` × display refresh period.
* Transmitter SHOULD hold each frame ≥ 2 receiver exposure periods
  (calibration measures this; default 6 periods ⇒ 10 fps at 60 Hz).
* The frame-counter strip (§3.3) carries the low 8 bits of the frame
  sequence number, duplicated at strip start and end; a mismatch marks the
  capture torn.

### 6.4 Capacity (informative, v0 targets)

1080p, cell 16 px, Mode A, 3 bits luma + 2 bits chroma, 10 fps:
≈ 100×56 cells × 5 bit × 10/s ≈ 280 kbit/s raw; ≈ 100–150 kbit/s goodput
after framing, FEC and loss. Numbers to be replaced by measurements.

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
| `psicode-core` | §7 complete: GF(32), 2 × RS(16,8) interleaved, Base32, bit packing, `CalibProfile` | done, 18 tests |
| `psicode-core` (next) | §3 ZC frame gen/detect, §5.1 color map, §5.2 Mode A, simulators | — |
| `psicode-tx` | Windows 11 transmitter: minifb → winit/softbuffer, calibrate & stream modes | — |
| `psicode-rx` | Rust core for Android (JNI): detect → homography → demod → RaptorQ | — |
| `psicode-android` | thin Kotlin shell: Camera2 (locked AWB/AE/AF, YUV420 direct) | — |

Non-goals for v0: iOS, encryption (transport is cleartext; wrap your payload),
multi-source-block transfers, printed streaming.

---

## 9. Roadmap to freezing

1. Live channel bring-up: ZC frame + Mode A + RaptorQ end-to-end.
2. Measure: SER vs distance/angle/blur; torn-frame statistics; color
   crosstalk on ≥ 3 display/phone pairs.
3. Freeze §3, §5.1, §5.2, §6 (→ STABLE), bump to 0.2.
4. Mode B measurement campaign: coefficient SNR vs blur σ, grid vs modes.
5. Freeze §5.3/§5.4, publish 1.0.

---

*Copyright © 2026 xelth.com. This specification may be freely copied and
implemented. "PsiCode" and the Ψ logo are trademarks of xelth.com used to
identify conformant implementations.*
