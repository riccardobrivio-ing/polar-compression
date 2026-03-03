# Polar IQ Compression — Research & Analysis

> **Status:** ❌ Non-competitive vs standard algorithms  
> **Conclusion:** Polar coordinate transformation cannot overcome information-theoretic limits for general-purpose compression of IQ data.  
> **Value:** Rigorous documentation of a negative result, with complete mathematical analysis.

---

## Table of Contents

1. [Motivation](#motivation)
2. [The Core Idea](#the-core-idea)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Algorithm Versions](#algorithm-versions)
6. [Benchmark Results](#benchmark-results)
7. [Why It Doesn't Work — Theory](#why-it-doesnt-work--theory)
8. [What We Learned](#what-we-learned)
9. [Future Directions](#future-directions)
10. [References](#references)

---

## Motivation

RTL-SDR devices capture radio signals as **IQ data** (In-phase / Quadrature), where each sample is a pair of bytes `(I, J)` representing the real and imaginary parts of a complex signal. These pairs have a natural geometric interpretation: they are **points in a 2D plane**.

Standard compressors (gzip, bz2, lzma) treat this data as a flat byte stream, unaware of its structure. The hypothesis behind this project was:

> *If IQ samples are naturally distributed along circles in the (I, Q) plane — as they are in many modulation schemes — then regrouping them by radius before encoding could expose redundancy that standard algorithms miss.*

This is the **polar coordinate compression hypothesis**.

---

## The Core Idea

### Step-by-step intuition

Every pair of consecutive bytes `(x, y)` is treated as a 2D point:

```
r = √(x² + y²)        → distance from origin (radius)
θ = atan2(y, x)        → direction (angle)
r_q = round(r / Δr) × Δr  → quantized radius (circle assignment)
```

Points with a similar radius fall on the **same circle**. Instead of storing `(x, y)` for each point, we store:

- The radius `r` **once per circle**
- Only the angle `θ` **for each point on that circle**

If many points share the same radius — which happens in real FSK/AM/FM signals — the radius is stored only once, and each point only costs `bits_theta` bits instead of 16 bits for the full `(x, y)` pair.

### Why this seemed promising

| Signal type | Distribution | Potential gain |
|---|---|---|
| AM radio | Discrete amplitude levels | Many points per circle → high gain |
| FSK modulation | Fixed symbol amplitudes | Ideal for polar grouping |
| FM radio | Varying amplitude | Moderate gain |
| Noise / random | Uniform distribution | No gain |

---

## Project Structure

```
polar-compression/
│
├── README.md                   ← you are here
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── quantizer.py            ← core math: polar transform + circle grouping
│   ├── encoder.py              ← v1 compression → .polr format
│   ├── decoder.py              ← v1 decompression
│   ├── encoder_v2.py           ← v2: bit-packing + entropy coding
│   ├── decoder_v2.py           ← v2 decompression
│   ├── bitstream.py            ← bit-level I/O for v2
│   └── utils.py                ← file I/O and reporting
│
├── tests/
│   ├── test_roundtrip.py       ← 34 lossless round-trip tests
│   └── test_v2.py              ← v2-specific tests
│
├── benchmarks/
│   ├── compare.py              ← v1 vs gzip / bz2 / lzma
│   └── compare_v1_v2.py        ← v1 vs v2 vs standard algorithms
│
├── visualizations/
│   └── visualize.py            ← circle distribution plots
│
├── examples/
│   └── demo.py                 ← quick demo
│
└── output_graphs/
    ├── 01_random.png
    ├── 02_iq_signal.png
    ├── 03_text.png
    └── 04_delta_r_sweep.png
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/riccardobrivio-ing/polar-compression.git
cd polar-compression

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
python examples/demo.py

# 5. Run all tests (34 should pass)
python -m pytest tests/ -v

# 6. Run benchmarks vs standard algorithms
python benchmarks/compare.py

# 7. Run v1 vs v2 comparison
python benchmarks/compare_v1_v2.py

# 8. Generate distribution plots
python visualizations/visualize.py
```

---

## Algorithm Versions

### v1 — Proof of Concept

**Format:** `.polr` (magic: `POLR`)

The first implementation stores:
- A 17-byte header (dimensions, parameters, flags)
- A **circle table**: one entry `(radius, count)` per unique radius
- **Angle data**: quantized θ for each point
- An **order map**: 4 bytes per point to reconstruct the original sequence

The order map turned out to be the critical flaw: it costs `4 × N` bytes regardless of compression quality, making the format larger than the original for most inputs.

**Result:** 150–400% of original size (worse than no compression at all).

---

### v2 — Three Optimizations

**Format:** `.polr v2` (magic: `PLR2`)

Three targeted improvements over v1:

**1. Order map eliminated (saves ~20,000 bytes per 10 KB)**  
Instead of grouping points by circle and tracking their original positions, v2 processes points in **original order**, storing `(circle_id, θ)` per point. Order is implicit in the sequence.

**2. Bit-packing (saves ~25% on angle data)**  
v1 uses a fixed `uint16` (16 bits) for every θ even when only 12 bits are needed.  
v2 packs exactly `bits_theta` bits per angle using a continuous bitstream, plus `⌈log₂(K)⌉` bits for the circle ID.

**3. Entropy coding — zlib DEFLATE (additional 10–30%)**  
After bit-packing, the entire data section is compressed with `zlib` at maximum level. This exploits residual redundancy in angle distributions on structured signals.

**Result:** Still 100–210% of original size. Better than v1, but not competitive.

---

## Benchmark Results

All benchmarks use 10 KB synthetic datasets. Standard algorithms run in lossless mode; polar v2 runs in lossy mode (best case for polar).

| Dataset | gzip | bz2 | lzma | Polar v2 | Winner |
|---|---|---|---|---|---|
| Pure random | 101% | 102% | 99% | 210% | lzma |
| Repeated text | 3% | 2% | 2% | 95% | bz2 |
| IQ signal, 2 amplitudes | 87% | 82% | 81% | 103% | lzma |
| IQ signal, 36 amplitudes | 95% | 91% | 90% | 148% | lzma |
| Single repeated pair | 0.1% | 0.1% | 0.1% | 18% | bz2 |

> **Polar v2 does not beat standard algorithms on any tested dataset.**  
> The best-case scenario (IQ signal with 2 discrete amplitudes, ideal for polar grouping) still results in 103% — slightly *larger* than the original.

### Where the bytes go (v2 lossy, IQ 2 amplitudes, Δr=2.0)

| Component | v1 | v2 (no zlib) | Saving |
|---|---|---|---|
| Header | 17 | 18 | −1 |
| Circle table | 48 | 24 | +24 |
| Angle data (θ) | 10,000 | 4,375 | +5,625 |
| Order map | 20,000 | 0 | +20,000 |
| **Total** | **30,065** | **4,417** | **+25,648** |
| After zlib | — | ~5,150 | — |

v2 is dramatically better than v1. It is still worse than gzip.

---

## Why It Doesn't Work — Theory

### 1. Shannon entropy is already near-maximum

For uniformly distributed IQ data, each byte carries close to 8 bits of information. bz2 and lzma already exploit virtually all exploitable correlations in byte sequences, reaching 98% of the theoretical compression limit.

```
H(X) ≈ 8 bits/byte   →   minimum compressed size ≈ original size
```

### 2. The polar transform does not reduce entropy

Changing coordinate system does not change the information content of a dataset. If the original distribution has entropy H, the polar-transformed distribution has the same entropy H. The transform is information-preserving by definition.

### 3. Quantization overhead offsets any gain

The discrete grouping into circles (radius quantization with step Δr) introduces a lossy approximation. But even in lossy mode, the overhead from encoding circle IDs and the angle table exceeds the savings from shared radius representation, except in extreme cases (single-amplitude signals) where bz2 already compresses by 99.9%.

### 4. The order reconstruction problem

The key structural insight — that points can be reordered by circle — turns out to require storing the reordering itself, which costs as much as the data being saved (v1). Eliminating this (v2) requires storing the circle ID per point anyway, consuming most of the saved space.

### 5. Why bz2 already wins

bz2 (Burrows-Wheeler Transform + Huffman coding) discovers and exploits patterns at the byte level that are equivalent to — or better than — what polar grouping provides. For IQ signals, the natural correlations between consecutive samples are already captured by BWT's block sorting.

---

## What We Learned

1. **Coordinate transforms do not compress data.** They can expose structure more clearly for analysis, but not for storage.

2. **Negative results are valid results.** The conclusion "this approach does not work, and here is the mathematical reason" is a complete scientific answer.

3. **Benchmark against strong baselines first.** bz2 is surprisingly effective and should be the minimum comparison point before implementing a novel algorithm.

4. **The overhead of metadata always counts.** Any format that encodes structural information (circle tables, order maps, headers) must recover those costs from the compression savings. For small files, metadata dominates.

5. **Lossless round-trip is achievable.** 34/34 tests pass. The algorithm is correct — it simply does not compress better than existing tools.

---

## Future Directions

Despite the negative compression results, the polar representation framework built here has potential applications in adjacent areas:

- **Signal classification:** The circle distribution pattern (number of circles, points per circle) is a feature that characterizes modulation type. An AM signal looks very different from FM noise in polar space.
- **Anomaly detection:** Outlier points far from expected circles in IQ data may indicate interference or signal degradation.
- **Preprocessing for ML:** Polar features (r, θ per sample) as input to classifiers for automatic modulation recognition (AMR).
- **Validation on real RTL-SDR data:** All benchmarks here use synthetic signals. Real hardware captures may have different statistical properties worth exploring.

---

## References

- C. E. Shannon, *A Mathematical Theory of Communication*, Bell System Technical Journal, 1948.
- D. Salomon, *Data Compression: The Complete Reference*, Springer, 4th ed., 2007.
- M. Burrows, D. J. Wheeler, *A Block-Sorting Lossless Data Compression Algorithm*, DEC SRC Research Report, 1994.
- RTL-SDR hardware: [rtl-sdr.com](https://www.rtl-sdr.com)
- IQ signal theory: [dspguide.com](http://www.dspguide.com/ch8.htm)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*This project was developed as a rigorous investigation of a compression hypothesis. The negative result is presented in full, with reproducible benchmarks and a mathematical explanation of the limitations encountered.*
