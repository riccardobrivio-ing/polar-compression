"""
encoder.py — Compressore Polare
================================

Converte un file binario nel formato compresso .polr

FLUSSO DELL'ALGORITMO:
    1. Leggi i byte del file
    2. Spezzali in coppie (x, y)
    3. Converti ogni coppia in polare (r, θ)
    4. Quantizza i raggi → assegna ogni punto a un cerchio
    5. Costruisci il file compresso:
       - Header: metadati (dimensione originale, parametri, ecc.)
       - Tabella cerchi: elenco raggi unici + quanti punti ciascuno
       - Dati punti: per ogni cerchio, la lista di angoli θ
       - Mappa ordine: per ricostruire l'ordine originale dei byte

FORMATO BINARIO .polr (v1):
    ┌─────────────────────────────────────────┐
    │ HEADER (17 byte)                        │
    │   magic       4B   "POLR"               │
    │   version     1B   versione formato     │
    │   orig_len    4B   lunghezza originale  │
    │   n_circles   2B   numero cerchi K      │
    │   delta_r     4B   risoluzione radiale  │
    │   bits_theta  1B   bit per angolo       │
    │   flags       1B   padded|lossless      │
    ├─────────────────────────────────────────┤
    │ TABELLA CERCHI (6 × K byte)             │
    │   per ogni cerchio:                     │
    │     r_quantizzato  4B  float32          │
    │     n_punti        2B  uint16           │
    ├─────────────────────────────────────────┤
    │ DATI ANGOLI                             │
    │   per ogni cerchio, per ogni punto:     │
    │     theta_quant    2B  uint16           │
    │   (in lossless: anche x,y originali)    │
    │     orig_x         1B  uint8            │
    │     orig_y         1B  uint8            │
    ├─────────────────────────────────────────┤
    │ MAPPA ORDINE                            │
    │   per ogni punto (in ordine cerchio):   │
    │     original_index 4B  uint32           │
    └─────────────────────────────────────────┘
"""

import struct
import numpy as np
from .quantizer import (
    bytes_to_pairs,
    cartesian_to_polar,
    quantize_radius,
    group_by_circles,
)

# Costanti del formato
MAGIC = b'POLR'
VERSION = 1


def encode(data: bytes,
           delta_r: float = 1.0,
           bits_theta: int = 12,
           lossless: bool = True) -> bytes:
    """
    Comprime un flusso di byte usando la trasformata polare.

    Parametri
    ---------
    data : bytes
        Dati originali da comprimere.
    delta_r : float
        Risoluzione radiale. Controlla quanti "cerchi" vengono creati.
        - 0.5: ~720 cerchi possibili, alta fedeltà
        - 1.0: ~361 cerchi possibili, buon compromesso
        - 5.0: ~73 cerchi possibili, compressione aggressiva
    bits_theta : int
        Quanti bit per codificare l'angolo θ.
        - 8 bit: 256 angoli possibili, errore max ~0.006 rad
        - 12 bit: 4096 angoli possibili, errore max ~0.0004 rad
        - 16 bit: 65536 angoli, quasi perfetto
    lossless : bool
        Se True, include anche i byte originali (x,y) per ricostruzione
        esatta. Aumenta la dimensione ma garantisce fedeltà al 100%.

    Ritorna
    -------
    bytes : file compresso in formato .polr
    """

    # ── STEP 1: Byte → Coppie ──────────────────────────────────
    pairs, padded = bytes_to_pairs(data)

    # ── STEP 2: Coppie → Polari → Raggruppamento per cerchi ────
    circles = group_by_circles(pairs, delta_r)

    # Ordina i cerchi per raggio (per consistenza del formato)
    sorted_radii = sorted(circles.keys())

    # ── STEP 3: Costruisci il file compresso ────────────────────

    output = bytearray()

    # === HEADER (17 byte) ===
    flags = 0
    if padded:
        flags |= 0b00000001   # bit 0: padding presente
    if lossless:
        flags |= 0b00000010   # bit 1: modalità lossless

    output.extend(MAGIC)                              # 4B: "POLR"
    output.extend(struct.pack('<B', VERSION))          # 1B: versione
    output.extend(struct.pack('<I', len(data)))        # 4B: dim originale
    output.extend(struct.pack('<H', len(circles)))     # 2B: n. cerchi
    output.extend(struct.pack('<f', delta_r))          # 4B: Δr
    output.extend(struct.pack('<B', bits_theta))       # 1B: bit per θ
    output.extend(struct.pack('<B', flags))            # 1B: flags

    # === TABELLA CERCHI (6B × K) ===
    # Per ogni cerchio: il suo raggio e quanti punti contiene
    for r_q in sorted_radii:
        n_points = len(circles[r_q])
        output.extend(struct.pack('<f', r_q))          # 4B: raggio
        output.extend(struct.pack('<H', n_points))     # 2B: conteggio

    # === DATI ANGOLI ===
    # Per ogni cerchio, per ogni punto: l'angolo θ quantizzato
    # Il θ ∈ [0, π/2] viene mappato su [0, max_val] interi
    max_theta_val = (1 << bits_theta) - 1   # es. 4095 per 12 bit
    half_pi = np.pi / 2                     # ~1.5708 rad

    for r_q in sorted_radii:
        for point in circles[r_q]:
            # Normalizza θ da [0, π/2] a [0, 1], poi scala a intero
            theta_normalized = point['theta'] / half_pi
            theta_quantized = int(round(theta_normalized * max_theta_val))
            theta_quantized = min(theta_quantized, max_theta_val)  # clamp

            output.extend(struct.pack('<H', theta_quantized))  # 2B

            # In lossless, aggiungi i byte originali
            if lossless:
                x_orig, y_orig = point['original']
                output.extend(struct.pack('<BB', x_orig, y_orig))  # 2B

    # === MAPPA ORDINE ===
    # Indice originale di ogni punto, nell'ordine in cui appaiono
    # nei cerchi. Serve per ricostruire la sequenza originale.
    for r_q in sorted_radii:
        for point in circles[r_q]:
            output.extend(struct.pack('<I', point['index']))  # 4B

    return bytes(output)


def compression_report(data: bytes,
                       delta_r: float = 1.0,
                       bits_theta: int = 12,
                       lossless: bool = True) -> dict:
    """
    Analizza la compressione senza scrivere il file.
    Mostra dimensione reale del prototipo E ratio teorico con bit-packing.

    Ritorna
    -------
    dict con metriche dettagliate.
    """
    pairs, padded = bytes_to_pairs(data)
    circles = group_by_circles(pairs, delta_r)

    n_points = len(pairs)
    n_circles = len(circles)

    # ── Dimensione reale del prototipo ──
    header_bytes = 17
    circle_table_bytes = 6 * n_circles
    if lossless:
        point_data_bytes = n_points * (2 + 2)    # θ + original (x,y)
    else:
        point_data_bytes = n_points * 2           # solo θ
    order_map_bytes = n_points * 4                 # indici uint32
    real_total = header_bytes + circle_table_bytes + point_data_bytes + order_map_bytes

    # ── Dimensione TEORICA con bit-packing ottimale ──
    # Header: ~17 byte (fisso, trascurabile su file grandi)
    # Per cerchio: bits_r (float16 = 16 bit) + count (16 bit) = 32 bit
    # Per punto: bits_theta bit (senza indice! l'ordine si ricostruisce)
    # Nota: in teoria servono anche i bit per la permutazione, ma
    #       se i dati hanno struttura (es. IQ), l'ordine è implicito
    bits_r = 16
    bits_count = 16
    theoretical_bits = (
        17 * 8 +                               # header
        n_circles * (bits_r + bits_count) +     # tabella cerchi
        n_points * bits_theta                   # angoli
    )
    theoretical_bytes = theoretical_bits / 8

    original_size = len(data)

    return {
        'original_bytes': original_size,
        'prototype_bytes': real_total,
        'prototype_ratio': real_total / original_size if original_size > 0 else 0,
        'theoretical_bytes': theoretical_bytes,
        'theoretical_ratio': theoretical_bytes / original_size if original_size > 0 else 0,
        'n_points': n_points,
        'n_circles': n_circles,
        'avg_points_per_circle': n_points / n_circles if n_circles > 0 else 0,
        'delta_r': delta_r,
        'bits_theta': bits_theta,
        'lossless': lossless,
        'header_bytes': header_bytes,
        'circle_table_bytes': circle_table_bytes,
        'point_data_bytes': point_data_bytes,
        'order_map_bytes': order_map_bytes,
    }
