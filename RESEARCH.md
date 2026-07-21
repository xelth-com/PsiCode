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

## 5. Spatial redundancy / cell replication

Alternative knob to lowering `luma_bits` on noisy channels: keep bit depth
but replicate each cell n× (or scale `cell_size_px`), trading capacity for
SNR by averaging. Worth a sim sweep against the luma_bits knob to see which
buys more goodput per unit capacity lost; could become a profile
prescription if it wins in some regime.
