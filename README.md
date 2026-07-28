# PsiCode (ΨCode)

**An open, royalty-free optical data link — and a barcode that moves.**

A display shows a stream of coded frames. A phone films it. A file comes out the
other end, byte-exact, with no feedback channel of any kind.

```
SPEC.md (25 365 bytes)  ──►  monitor  ──►  phone camera  ──►  file
                             4.4 s · ~46 kbit/s · SHA256 identical
```

---

## Status

Working and measured on real hardware. Not frozen — the specification is at
**0.2.0-draft** and section numbers may still move.

| | |
|---|---|
| live byte-exact transfers | 20+, on two phones |
| throughput | ~46 kbit/s at 2 bits/cell, 1080p display, 60 Hz |
| detection, full uncropped 1920×1080 frame | 100 % |
| devices | Galaxy Note 10 Lite, Galaxy A22 5G |

Every number in this repository comes from a measurement. Things that were tried
and did not survive contact with measurement are recorded too — see
[`.eck/DEAD_ENDS.md`](.eck/DEAD_ENDS.md), which is deliberately as long as the
list of what worked.

---

## What it actually is

Despite the name, there is nothing quantum-mechanical left in it. What we
independently rebuilt is a **modem**, and the correspondence is exact:

| PsiCode | radio |
|---|---|
| a complex number per cell, carried in chromaticity | QAM constellation |
| inter-cell interference equaliser | ISI equaliser |
| 3×3 channel matrix + in-band calibration frames | pilots and channel estimation |
| Zadoff–Chu border | preamble — the same sequence LTE uses, for the same reason |
| XOR fountain over the frame stream | rateless FEC |

An optical QAM modem with a fountain transport, running one-way over a camera.

### The two things that are genuinely new

**Chromaticity as a complex plane with an algebraic invariance.** Every other
colour barcode — Microsoft HCCB, JAB Code (ISO/IEC 23634), HiQ — treats colour as
a discrete palette to be classified. PsiCode treats it as a continuous 2-D vector
and holds total drive constant:

```
transmit:  R = u(1 − b·x + c·y)    G = u(1 + 2b·x)    B = u(1 − b·x − c·y)
receive:   x = (2G − R − B) / (2·S·b),   y = 3(R − B) / (2·S·c),   S = R+G+B measured
```

Because `S` is *measured per cell*, any multiplicative illumination field cancels
**identically** — no fitting, no pilots, no local thresholding, for a field of any
spatial complexity. Measured live, screen luminance drifts 0.62 → 0.86 across a
single frame; this construction is simply blind to it.

**Time as the third dimension.** QR, Data Matrix, Aztec, MaxiCode, JAB and HCCB
are all static images. A code on a screen does not have to be. One consequence
that is hard to reach any other way: **calibration becomes a temporal resource
rather than an area budget.** Interleaved calibration frames cost 0.8 % of the
stream and estimate the channel 36× more accurately than a reference strip that
would cost 8–18 % of the symbol's area.

---

## Quick start

Requires a Rust toolchain. The transmitter runs on Windows (winit/softbuffer);
the receiver core is portable Rust and is packaged for Android via JNI.

```bash
# stream a file from the display
cargo run --release -p psicode-tx -- stream <file> --v1 --chroma --cell 10

#   --v1      border v1: extruded ZC strips, no quiet zone
#   --chroma  payload in chromaticity, 2 bits/cell
#   --calib   interleave in-band calibration frames
```

Receiver (Android app in `psicode-android`, native core in `psicode-rx`):

```bash
adb shell am start -n com.xelth.psicode/.MainActivity \
    --ei border 1        # 0 = v0, 1 = v1 strips
    --ez chroma true     # chromatic payload
    --ei isi 1           # inter-cell equaliser
```

The receiver looks for the border edition it is told to look for — there is no
auto-detection yet.

---

## Layout

| crate | what |
|---|---|
| `psicode-core` | codec: profile, symbol rendering, demodulation, ZC border, acquisition, framing, fountain, calibration. `no_std + alloc` |
| `psicode-tx` | transmitter — a stream of frames in a window |
| `psicode-rx` | receiver — session, YUV, JNI bridge. Desktop-testable |
| `psicode-sim` | Monte Carlo harnesses: channel, scale ladders, temporal channel, geometry studies |
| `psicode-android` | Camera2 app over `libpsicode_rx.so` |

## Documents

| | |
|---|---|
| [`SPEC.md`](SPEC.md) | the normative specification |
| [`RESEARCH.md`](RESEARCH.md) | Hermite–Gauss modes and open research directions |
| [`BENCHMARKS.md`](BENCHMARKS.md) | measurement tables |
| [`.eck/FINDINGS.md`](.eck/FINDINGS.md) | everything measured, with provenance |
| [`.eck/DEAD_ENDS.md`](.eck/DEAD_ENDS.md) | what was tried and rejected, and why |

---

## A few results worth knowing before you implement

Collected here because each one cost a debugging session, and because they are
the kind of thing a specification tends not to tell you.

**Measure focus at cell scale, on the payload — never on the border.** A
blur-tolerant finder pattern is by construction insensitive to the defocus that
destroys the payload. Measured live: the border score spanned 0.897–0.943 across a
focus range over which payload stripes went from 8 to 0, so a sweep driven by it
locks onto an unreadable symbol and reports success.

**Sample the border through the homography, not along the side.** Equal steps
along a segment are not the image of a uniform cell grid, and the along-side
foreshortening of a side is governed by the perspective coefficient of the *other*
axis. A keystone invisible on one pair of sides is fatal on the perpendicular
pair — measured, 1.2 cells of lag against a correlation peak one cell wide.

**A quiet zone is not needed if you have a correlation border.** It exists for
detectors that key on contrast against the background. Dropping it gave +27.9 %
symbol area at a fixed footprint and, with a correlation-based finder, took
detection on full uncropped frames from 18 % to 100 %.

**Inter-cell interference, not colour fidelity, limits a dense payload.** Kernel
strength is 0.23–0.27 for a chromatic payload against 0.053 for 1-bit luma — which
is why a mono payload works at geometries where a two-axis colour one fails. It is
deterministic, and therefore invertible.

---

## Prior art this builds on and against

[Microsoft HCCB](https://www.microsoft.com/en-us/research/project/high-capacity-color-barcodes-hccb/)
(Parikh & Jancke, WACV 2008) — colour triangles with a palette reference, and the
idea of letting the decoder give feedback to the vision stage.
[JAB Code](https://jabcode.org/) (ISO/IEC 23634:2022, Fraunhofer SIT) — colour
palette modules at predefined positions.
[HiQ](https://arxiv.org/abs/1704.06447) — which named *cross-module colour
interference* as a previously undocumented problem, and attacks it with a joint
classifier where we use an equaliser.

None of them move in time.

---

*Copyright © 2026 xelth.com. The specification may be freely copied and
implemented. "PsiCode" and the Ψ logo are trademarks of xelth.com, used to
identify conformant implementations.*
