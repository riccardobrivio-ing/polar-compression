"""
demo.py — Dimostrazione rapida dell'algoritmo
===============================================

Esegui con:  python examples/demo.py

Mostra i 3 step fondamentali:
  1. Comprime un testo → visualizza il risultato
  2. Decomprime → verifica che sia identico
  3. Mostra le statistiche
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoder import encode, compression_report
from src.decoder import decode, inspect_header
from src.utils import print_report


def main():
    print("=" * 55)
    print("  DEMO — Compressione Polare")
    print("=" * 55)

    # ── STEP 1: Prepara i dati ──
    original = b"Hello, Polar Compression! This is a test of the algorithm."
    print(f"\n  Dati originali: {original}")
    print(f"  Lunghezza: {len(original)} byte")

    # ── STEP 2: Comprimi (lossless) ──
    print("\n  --- Compressione LOSSLESS ---")
    compressed = encode(original, delta_r=1.0, bits_theta=12, lossless=True)
    print(f"  Compresso: {len(compressed)} byte")
    print(f"  Header: {compressed[:4]}")  # mostra "POLR"

    # ── STEP 3: Decompri ──
    decompressed = decode(compressed)
    print(f"  Decompresso: {decompressed}")
    print(f"  Match: {'✓ PERFETTO' if decompressed == original else '✗ ERRORE!'}")

    # ── STEP 4: Ispeziona l'header ──
    header = inspect_header(compressed)
    print(f"\n  Header del file .polr:")
    for key, value in header.items():
        print(f"    {key}: {value}")

    # ── STEP 5: Report di compressione ──
    report = compression_report(original, delta_r=1.0, bits_theta=12, lossless=True)
    print_report(report)

    # ── STEP 6: Confronto lossy ──
    print("\n  --- Compressione LOSSY (Δr=2.0) ---")
    compressed_lossy = encode(original, delta_r=2.0, bits_theta=12, lossless=False)
    decompressed_lossy = decode(compressed_lossy)

    print(f"  Originale:    {original}")
    print(f"  Ricostruito:  {decompressed_lossy}")

    # Calcola errore
    min_len = min(len(original), len(decompressed_lossy))
    errors = [abs(a - b) for a, b in
              zip(original[:min_len], decompressed_lossy[:min_len])]
    avg_error = sum(errors) / len(errors) if errors else 0
    print(f"  Errore medio per byte: {avg_error:.2f}")
    print(f"  Byte diversi: {sum(1 for e in errors if e > 0)} / {min_len}")

    # ── STEP 7: Mostra ratio teorico per vari Δr ──
    print("\n  --- Sweep Δr (ratio teorico) ---")
    print(f"  {'Δr':>6} {'Cerchi':>8} {'Ratio teorico':>15}")
    print(f"  {'─' * 35}")
    for dr in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        r = compression_report(original, delta_r=dr, bits_theta=12, lossless=False)
        print(f"  {dr:>6.1f} {r['n_circles']:>8} {r['theoretical_ratio']:>14.1%}")

    print(f"\n{'=' * 55}")
    print("  Demo completata!")
    print("  Prossimi passi:")
    print("    python -m pytest tests/ -v          # esegui tutti i test")
    print("    python benchmarks/compare.py        # benchmark vs gzip/bz2/lzma")
    print("    python visualizations/visualize.py  # genera grafici")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
