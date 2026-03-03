"""
compare.py — Confronto compressione polare vs algoritmi standard
=================================================================

Esegui con:  python benchmarks/compare.py

Confronta l'algoritmo polare con gzip, bz2, lzma su diversi dataset:
  1. Dati casuali (worst case — incomprimibili)
  2. Testo ripetuto (best case per algoritmi classici)
  3. Segnale IQ simulato (best case per algoritmo polare)
  4. Mix di dati

Per ogni dataset mostra:
  - Dimensione compressa con ogni algoritmo
  - Rapporto di compressione (compressed / originale)
  - Tempo di compressione
  - Rapporto TEORICO per polare con bit-packing ottimale
"""

import sys
import os
import gzip
import bz2
import lzma
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoder import encode, compression_report
from src.utils import print_report


def benchmark_single(data: bytes, label: str, algo_name: str, compress_fn):
    """Comprimi e misura tempo e dimensione."""
    t0 = time.perf_counter()
    compressed = compress_fn(data)
    elapsed = time.perf_counter() - t0
    ratio = len(compressed) / len(data) if len(data) > 0 else 0
    return {
        'algo': algo_name,
        'original': len(data),
        'compressed': len(compressed),
        'ratio': ratio,
        'time': elapsed,
    }


def run_benchmark(data: bytes, label: str):
    """Esegui il benchmark completo su un dataset."""

    print(f"\n{'━' * 65}")
    print(f"  DATASET: {label}")
    print(f"  Dimensione originale: {len(data):,} byte")
    print(f"{'━' * 65}")
    print(f"  {'Algoritmo':<25} {'Compresso':>10} {'Ratio':>8} {'Tempo':>10}")
    print(f"  {'─' * 55}")

    results = []

    # Algoritmi standard
    for name, fn in [
        ("gzip (livello 9)", lambda d: gzip.compress(d, compresslevel=9)),
        ("bz2", bz2.compress),
        ("lzma", lzma.compress),
    ]:
        r = benchmark_single(data, label, name, fn)
        results.append(r)
        print(f"  {name:<25} {r['compressed']:>10,} {r['ratio']:>8.1%} "
              f"{r['time']:>9.4f}s")

    # Polare lossless
    r = benchmark_single(data, label, "polare lossless",
                         lambda d: encode(d, delta_r=1.0, lossless=True))
    results.append(r)
    print(f"  {'polare lossless':<25} {r['compressed']:>10,} {r['ratio']:>8.1%} "
          f"{r['time']:>9.4f}s")

    # Polare lossy con vari Δr
    for dr in [1.0, 2.0, 5.0]:
        name = f"polare lossy Δr={dr}"
        r = benchmark_single(data, label, name,
                             lambda d, dr=dr: encode(d, delta_r=dr, lossless=False))
        results.append(r)
        print(f"  {name:<25} {r['compressed']:>10,} {r['ratio']:>8.1%} "
              f"{r['time']:>9.4f}s")

    # Combinazione: polare + gzip
    r = benchmark_single(data, label, "polare+gzip",
                         lambda d: gzip.compress(
                             encode(d, delta_r=2.0, lossless=False),
                             compresslevel=9))
    results.append(r)
    print(f"  {'polare+gzip':<25} {r['compressed']:>10,} {r['ratio']:>8.1%} "
          f"{r['time']:>9.4f}s")

    # Ratio TEORICO
    report = compression_report(data, delta_r=2.0, bits_theta=12, lossless=False)
    print(f"  {'─' * 55}")
    print(f"  {'TEORICO (polare Δr=2)':<25} {report['theoretical_bytes']:>10,.0f} "
          f"{report['theoretical_ratio']:>8.1%}")
    print(f"  Cerchi: {report['n_circles']}, "
          f"Media punti/cerchio: {report['avg_points_per_circle']:.1f}")

    return results


def generate_iq_signal(n_bytes: int = 10000,
                       amplitudes: list = None) -> bytes:
    """
    Genera un segnale IQ simulato.

    Un segnale IQ con poche ampiezze discrete è il caso IDEALE
    per la compressione polare: molti punti cadono sugli stessi cerchi.

    amplitudes: lista di ampiezze del segnale.
        Se None, usa [50, 100, 150] (3 livelli di potenza).
    """
    if amplitudes is None:
        amplitudes = [50, 100, 150]

    n_pairs = n_bytes // 2
    data = bytearray()

    for _ in range(n_pairs):
        amp = np.random.choice(amplitudes)
        phase = np.random.uniform(0, np.pi / 2)
        x = int(np.clip(round(amp * np.cos(phase)), 0, 255))
        y = int(np.clip(round(amp * np.sin(phase)), 0, 255))
        data.extend([x, y])

    return bytes(data)


if __name__ == "__main__":

    print("\n" + "═" * 65)
    print("  BENCHMARK COMPRESSIONE POLARE vs STANDARD")
    print("═" * 65)

    # Test 1: dati casuali
    run_benchmark(os.urandom(10000), "Random puro (10 KB) — worst case")

    # Test 2: testo ripetuto
    run_benchmark(b"the quick brown fox jumps " * 400,
                  "Testo ripetuto (10 KB) — best case classico")

    # Test 3: segnale IQ con poche ampiezze (best case polare)
    iq_data = generate_iq_signal(10000, amplitudes=[80, 120])
    run_benchmark(iq_data,
                  "Segnale IQ simulato, 2 ampiezze (10 KB) — best case polare")

    # Test 4: segnale IQ con molte ampiezze
    iq_data_many = generate_iq_signal(10000, amplitudes=list(range(20, 200, 5)))
    run_benchmark(iq_data_many,
                  "Segnale IQ simulato, 36 ampiezze (10 KB)")

    # Test 5: byte ripetuti (coppia identica)
    run_benchmark(bytes([100, 100] * 5000),
                  "Coppia (100,100) ripetuta (10 KB) — 1 solo cerchio")

    # Messaggio finale
    print(f"\n{'━' * 65}")
    print("  NOTE:")
    print("  • Il 'prototipo' usa encoding byte-level (non ottimizzato)")
    print("  • Il 'TEORICO' stima il risultato con bit-packing ottimale")
    print("  • La Fase 2 implementerà il bit-packing → ratio reale ≈ teorico")
    print("  • Il vero test sarà su dati IQ reali (Fase 2 con RTL-SDR)")
    print(f"{'━' * 65}\n")
