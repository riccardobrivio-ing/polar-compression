"""
compare_v1_v2.py — Confronto diretto v1 vs v2 vs standard
============================================================

Esegui con:  python benchmarks/compare_v1_v2.py

Mostra per ogni dataset:
  - Dimensione compressa v1, v2 (con e senza zlib), gzip, bz2, lzma
  - Miglioramento % di v2 rispetto a v1
  - Dettaglio di dove vanno i byte in v1 e v2
"""

import sys
import os
import gzip
import bz2
import lzma
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.encoder import encode as encode_v1, compression_report as report_v1
from src.encoder_v2 import encode_v2, compression_report_v2
from src.decoder_v2 import decode_v2


def separator(char='━', width=72):
    print(f"  {char * width}")


def run_comparison(data: bytes, label: str, delta_r: float = 2.0):
    """Confronto completo su un dataset."""

    n = len(data)
    print(f"\n{'━' * 76}")
    print(f"  {label}")
    print(f"  Originale: {n:,} byte")
    print(f"{'━' * 76}")

    # ── Comprimi con tutto ──
    gz = gzip.compress(data, compresslevel=9)
    bz = bz2.compress(data)
    lz = lzma.compress(data)

    v1_ll = encode_v1(data, delta_r=delta_r, lossless=True)
    v1_ly = encode_v1(data, delta_r=delta_r, lossless=False)

    v2_ll = encode_v2(data, delta_r=delta_r, lossless=True, use_zlib=True)
    v2_ly = encode_v2(data, delta_r=delta_r, lossless=False, use_zlib=True)
    v2_ly_nz = encode_v2(data, delta_r=delta_r, lossless=False, use_zlib=False)

    # ── Tabella risultati ──
    print(f"\n  {'Algoritmo':<30} {'Byte':>10} {'Ratio':>8} {'vs orig':>10}")
    separator('─')

    results = [
        ("gzip (livello 9)", len(gz)),
        ("bz2", len(bz)),
        ("lzma", len(lz)),
        ("", None),  # separatore
        ("v1 lossless", len(v1_ll)),
        ("v1 lossy", len(v1_ly)),
        ("", None),
        ("v2 lossy (no zlib)", len(v2_ly_nz)),
        ("v2 lossy (+ zlib)", len(v2_ly)),
        ("v2 lossless (+ zlib)", len(v2_ll)),
    ]

    for name, size in results:
        if size is None:
            separator('·')
            continue
        ratio = size / n
        delta = size - n
        sign = "+" if delta >= 0 else ""
        compresses = "✓" if ratio < 1.0 else " "
        print(f"  {compresses} {name:<28} {size:>10,} {ratio:>7.1%} "
              f"{sign}{delta:>+9,}")

    # ── Dettaglio: dove vanno i byte ──
    print(f"\n  DETTAGLIO STRUTTURA (lossy, Δr={delta_r})")
    separator('─')

    r1 = report_v1(data, delta_r=delta_r, lossless=False)
    r2 = compression_report_v2(data, delta_r=delta_r, lossless=False,
                                use_zlib=True)

    print(f"  {'Componente':<28} {'v1':>10} {'v2 (no zlib)':>14} {'Diff':>10}")
    separator('·')

    components = [
        ("Header", r1['header_bytes'], r2['v2_header']),
        ("Tabella cerchi", r1['circle_table_bytes'], r2['v2_circle_table']),
        ("Angoli θ", r1['point_data_bytes'], r2['v2_raw_bitstream']),
        ("Mappa ordine", r1['order_map_bytes'], 0),
        ("Dim. sezione dati", 0, r2['v2_data_size_field']),
    ]

    total_v1 = 0
    total_v2 = 0
    for name, s1, s2 in components:
        diff = s2 - s1
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<28} {s1:>10,} {s2:>14,} {sign}{diff:>+9,}")
        total_v1 += s1
        total_v2 += s2

    separator('·')
    print(f"  {'TOTALE (no zlib)':<28} {total_v1:>10,} {total_v2:>14,} "
          f"{total_v2 - total_v1:>+9,}")
    print(f"  {'TOTALE (v2 + zlib)':<28} {'':>10} {r2['v2_zlib_total']:>14,}")
    print(f"  {'zlib saving':<28} {'':>10} "
          f"{r2['zlib_saving']:>13.1%}")

    # ── Miglioramento complessivo ──
    separator('─')
    impr_no_zlib = 1 - len(v2_ly_nz) / len(v1_ly) if len(v1_ly) > 0 else 0
    impr_zlib = 1 - len(v2_ly) / len(v1_ly) if len(v1_ly) > 0 else 0

    print(f"  Miglioramento v2/v1 (no zlib): {impr_no_zlib:>8.1%}")
    print(f"  Miglioramento v2/v1 (+ zlib):  {impr_zlib:>8.1%}")
    print(f"  Cerchi: {r2['n_circles']}, "
          f"Media punti/cerchio: {r2['avg_points_per_circle']:.1f}, "
          f"bit/punto: {r2['bits_per_point']}")

    # ── Confronto con standard (il dato che conta) ──
    if len(v2_ly) < n:
        best_std = min(len(gz), len(bz), len(lz))
        best_name = "gzip" if best_std == len(gz) else (
            "bz2" if best_std == len(bz) else "lzma")
        v2_vs_best = len(v2_ly) / best_std
        verdict = "BATTE" if v2_vs_best < 1.0 else "PERDE contro"
        print(f"\n  ★ v2 lossy+zlib ({len(v2_ly):,}) {verdict} "
              f"{best_name} ({best_std:,}) → ratio {v2_vs_best:.2f}x")

    return {
        'label': label,
        'original': n,
        'gzip': len(gz), 'bz2': len(bz), 'lzma': len(lz),
        'v1_lossy': len(v1_ly), 'v2_lossy': len(v2_ly),
        'v2_lossy_no_zlib': len(v2_ly_nz),
        'v2_ratio': len(v2_ly) / n,
    }


def generate_iq(n_bytes, amplitudes):
    """Genera segnale IQ simulato."""
    n_pairs = n_bytes // 2
    data = bytearray()
    for _ in range(n_pairs):
        amp = np.random.choice(amplitudes)
        phase = np.random.uniform(0, np.pi / 2)
        data.extend([
            int(np.clip(round(amp * np.cos(phase)), 0, 255)),
            int(np.clip(round(amp * np.sin(phase)), 0, 255)),
        ])
    return bytes(data)


if __name__ == "__main__":

    print("\n" + "═" * 76)
    print("  CONFRONTO v1 vs v2 — IMPATTO DELLE 3 OTTIMIZZAZIONI")
    print("  1. Eliminazione mappa ordine")
    print("  2. Bit-packing (θ e circle_id)")
    print("  3. Entropy coding (zlib/DEFLATE)")
    print("═" * 76)

    all_results = []

    # Dataset 1: Random
    all_results.append(
        run_comparison(os.urandom(10000),
                       "Random puro (10 KB) — worst case", delta_r=5.0)
    )

    # Dataset 2: Testo ripetuto
    all_results.append(
        run_comparison(b"the quick brown fox jumps " * 400,
                       "Testo ripetuto (10 KB) — best case classico")
    )

    # Dataset 3: IQ 2 ampiezze
    all_results.append(
        run_comparison(generate_iq(10000, [80, 120]),
                       "IQ simulato, 2 ampiezze (10 KB) — best case polare")
    )

    # Dataset 4: IQ 36 ampiezze
    all_results.append(
        run_comparison(generate_iq(10000, list(range(20, 200, 5))),
                       "IQ simulato, 36 ampiezze (10 KB)")
    )

    # Dataset 5: Coppia ripetuta
    all_results.append(
        run_comparison(bytes([100, 100] * 5000),
                       "Coppia (100,100) ripetuta (10 KB) — 1 cerchio")
    )

    # Dataset 6: 100KB random (scalabilità)
    all_results.append(
        run_comparison(os.urandom(100000),
                       "Random (100 KB) — scalabilità", delta_r=5.0)
    )

    # Dataset 7: IQ 2 ampiezze 100KB
    all_results.append(
        run_comparison(generate_iq(100000, [80, 120]),
                       "IQ 2 ampiezze (100 KB) — scalabilità polare")
    )

    # ── RIEPILOGO FINALE ──
    print(f"\n\n{'═' * 76}")
    print(f"  RIEPILOGO — v2 lossy+zlib vs standard")
    print(f"{'═' * 76}")
    print(f"  {'Dataset':<45} {'v2':>7} {'gzip':>7} {'bz2':>7} {'Vince':>8}")
    separator('─')

    for r in all_results:
        v2r = f"{r['v2_ratio']:.0%}"
        gzr = f"{r['gzip']/r['original']:.0%}"
        bzr = f"{r['bz2']/r['original']:.0%}"
        best = min(r['v2_lossy'], r['gzip'], r['bz2'])
        winner = ("v2" if best == r['v2_lossy'] else
                  "gzip" if best == r['gzip'] else "bz2")
        print(f"  {r['label'][:44]:<45} {v2r:>7} {gzr:>7} {bzr:>7} {winner:>8}")

    print(f"{'═' * 76}")
    print(f"  NOTA: v2 lossy = compressione con perdita, gzip/bz2 = lossless")
    print(f"  Il confronto equo è lossy vs lossy, ma mostra la competitività")
    print(f"{'═' * 76}\n")
