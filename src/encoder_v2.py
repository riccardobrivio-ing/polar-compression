"""
encoder_v2.py — Compressore Polare Ottimizzato (v2)
=====================================================

3 OTTIMIZZAZIONI rispetto al v1:

1. ELIMINAZIONE MAPPA ORDINE (-20,000 byte su 10KB)
   ─────────────────────────────────────────────────
   v1: raggruppa i punti per cerchio, poi salva un indice uint32
       per ognuno per ricordare la posizione originale.
       Costo: 4 byte × N_punti = 20,000 byte su 5000 punti.

   v2: scorre i punti NELL'ORDINE ORIGINALE del file.
       Per ogni punto salva: (circle_id, θ).
       L'ordine è implicito nella sequenza → 0 byte extra.

2. BIT-PACKING (da 10,000 a 7,500 byte per θ)
   ────────────────────────────────────────────
   v1: ogni θ usa uint16 (16 bit fissi), anche se servono solo 12.
       Spreco: 4 bit × 5000 punti = 2,500 byte.

   v2: usa esattamente bits_theta bit per θ, impaccati in un
       bitstream continuo. Anche il circle_id usa solo
       ceil(log2(K)) bit, dove K = numero di cerchi.

3. ENTROPY CODING FINALE (ulteriore 10-30%)
   ─────────────────────────────────────────
   Dopo il bit-packing, applica zlib (DEFLATE) sull'intera
   sezione dati. Sfrutta la ridondanza residua negli angoli:
   su segnali reali, certi angoli sono più frequenti di altri.

FORMATO .polr v2:
┌──────────────────────────────────────────┐
│ HEADER (18 byte)                         │
│   magic         4B  "PLR2"               │
│   version       1B  2                    │
│   orig_len      4B  lunghezza originale  │
│   n_circles     2B  K                    │
│   delta_r       4B  float32              │
│   bits_theta    1B                       │
│   bits_circle   1B  ceil(log2(K))        │
│   flags         1B  padded|lossless|zlib │
├──────────────────────────────────────────┤
│ TABELLA CERCHI (4B × K)                  │
│   r_quantizzato float32 per ogni cerchio │
├──────────────────────────────────────────┤
│ DIMENSIONE DATI (4B)                     │
│   n_byte della sezione dati              │
├──────────────────────────────────────────┤
│ DATI PUNTI (bitstream, opz. zlib)        │
│   per ogni punto in ordine originale:    │
│     circle_id    bits_circle bit         │
│     theta_quant  bits_theta bit          │
│   se lossless:                           │
│     orig_x       8 bit                   │
│     orig_y       8 bit                   │
└──────────────────────────────────────────┘
"""

import struct
import math
import zlib
import numpy as np

from .quantizer import (
    bytes_to_pairs,
    cartesian_to_polar,
    quantize_radius,
)
from .bitstream import BitWriter

MAGIC_V2 = b'PLR2'
VERSION = 2


def encode_v2(data: bytes,
              delta_r: float = 1.0,
              bits_theta: int = 12,
              lossless: bool = True,
              use_zlib: bool = True) -> bytes:
    """
    Comprime con il formato ottimizzato v2.

    Parametri
    ---------
    data : bytes
        Dati originali.
    delta_r : float
        Risoluzione radiale.
    bits_theta : int
        Bit per angolo (8-16).
    lossless : bool
        Se True, include byte originali per ricostruzione esatta.
    use_zlib : bool
        Se True, applica DEFLATE sulla sezione dati (risparmio 10-30%).

    Ritorna
    -------
    bytes : file compresso formato .polr v2.
    """
    if len(data) == 0:
        # Gestione file vuoto
        return _build_header(0, 0, delta_r, bits_theta, 0,
                             False, lossless, use_zlib) + struct.pack('<I', 0)

    # ── STEP 1: Byte → Coppie → Polari ────────────────────────
    pairs, padded = bytes_to_pairs(data)

    # ── STEP 2: Identifica i cerchi (set unico di raggi) ──────
    # Per ogni punto, calcola il raggio quantizzato
    point_data = []  # lista di (r_q, theta) per ogni punto
    radius_set = set()

    for x, y in pairs:
        r, theta = cartesian_to_polar(x, y)
        r_q = quantize_radius(r, delta_r)
        radius_set.add(r_q)
        point_data.append((r_q, theta, x, y))

    # Ordina i raggi e crea la mappa raggio → ID
    sorted_radii = sorted(radius_set)
    n_circles = len(sorted_radii)
    radius_to_id = {r: i for i, r in enumerate(sorted_radii)}

    # ── STEP 3: Calcola bits necessari per circle_id ──────────
    # Se ci sono K cerchi, servono ceil(log2(K)) bit per l'ID.
    # Caso speciale: K=1 → 1 bit (non 0, altrimenti non scrivi nulla)
    if n_circles <= 1:
        bits_circle = 1
    else:
        bits_circle = math.ceil(math.log2(n_circles))
        # Assicurati che 2^bits_circle >= n_circles
        if (1 << bits_circle) < n_circles:
            bits_circle += 1

    # ── STEP 4: Scrivi il bitstream dei punti ─────────────────
    max_theta_val = (1 << bits_theta) - 1
    half_pi = np.pi / 2

    writer = BitWriter()

    for r_q, theta, x, y in point_data:
        circle_id = radius_to_id[r_q]

        # Quantizza θ: [0, π/2] → [0, max_theta_val]
        theta_norm = theta / half_pi
        theta_quant = int(round(theta_norm * max_theta_val))
        theta_quant = min(max(theta_quant, 0), max_theta_val)

        # Scrivi circle_id e θ nel bitstream
        writer.write_bits(circle_id, bits_circle)
        writer.write_bits(theta_quant, bits_theta)

        # In lossless: aggiungi i byte originali (8+8 bit)
        if lossless:
            writer.write_bits(x, 8)
            writer.write_bits(y, 8)

    raw_bitstream = writer.flush()

    # ── STEP 5: Entropy coding opzionale ──────────────────────
    if use_zlib:
        point_section = zlib.compress(raw_bitstream, level=9)
    else:
        point_section = raw_bitstream

    # ── STEP 6: Assembla il file ──────────────────────────────
    output = bytearray()

    # HEADER (18 byte)
    output.extend(
        _build_header(len(data), n_circles, delta_r, bits_theta,
                      bits_circle, padded, lossless, use_zlib)
    )

    # TABELLA CERCHI (4B × K)
    for r_q in sorted_radii:
        output.extend(struct.pack('<f', r_q))

    # DIMENSIONE SEZIONE DATI (4B)
    output.extend(struct.pack('<I', len(point_section)))

    # DATI PUNTI
    output.extend(point_section)

    return bytes(output)


def _build_header(orig_len: int, n_circles: int, delta_r: float,
                  bits_theta: int, bits_circle: int,
                  padded: bool, lossless: bool, use_zlib: bool) -> bytes:
    """Costruisce l'header di 18 byte."""
    flags = 0
    if padded:
        flags |= 0b00000001    # bit 0
    if lossless:
        flags |= 0b00000010    # bit 1
    if use_zlib:
        flags |= 0b00000100    # bit 2

    header = bytearray()
    header.extend(MAGIC_V2)                              # 4B
    header.extend(struct.pack('<B', VERSION))             # 1B
    header.extend(struct.pack('<I', orig_len))            # 4B
    header.extend(struct.pack('<H', n_circles))           # 2B
    header.extend(struct.pack('<f', delta_r))             # 4B
    header.extend(struct.pack('<B', bits_theta))          # 1B
    header.extend(struct.pack('<B', bits_circle))         # 1B
    header.extend(struct.pack('<B', flags))               # 1B
    return bytes(header)                                  # tot: 18B


def compression_report_v2(data: bytes,
                          delta_r: float = 1.0,
                          bits_theta: int = 12,
                          lossless: bool = True,
                          use_zlib: bool = True) -> dict:
    """
    Report dettagliato della compressione v2.
    Mostra il peso di ogni componente e il confronto con v1.
    """
    if len(data) == 0:
        return {'original_bytes': 0, 'v2_total_bytes': 0,
                'v2_ratio': 0, 'n_circles': 0, 'n_points': 0}

    # Comprimi con entrambe le versioni per confronto
    from .encoder import encode as encode_v1, compression_report as report_v1

    compressed_v2 = encode_v2(data, delta_r, bits_theta, lossless, use_zlib)
    compressed_v2_no_zlib = encode_v2(data, delta_r, bits_theta, lossless,
                                      use_zlib=False)
    compressed_v1 = encode_v1(data, delta_r, bits_theta, lossless)
    r_v1 = report_v1(data, delta_r, bits_theta, lossless)

    # Calcola componenti v2
    pairs, padded = bytes_to_pairs(data)
    n_points = len(pairs)

    radius_set = set()
    for x, y in pairs:
        r, theta = cartesian_to_polar(x, y)
        r_q = quantize_radius(r, delta_r)
        radius_set.add(r_q)
    n_circles = len(radius_set)
    bits_circle = max(1, math.ceil(math.log2(max(n_circles, 2))))

    header_bytes = 18
    circle_table_bytes = 4 * n_circles
    data_size_field = 4  # uint32 per la dimensione della sezione dati

    if lossless:
        bits_per_point = bits_circle + bits_theta + 16  # +16 per orig x,y
    else:
        bits_per_point = bits_circle + bits_theta

    raw_bitstream_bytes = math.ceil(n_points * bits_per_point / 8)

    # Dimensione reale della sezione dati (con o senza zlib)
    actual_data_bytes = (len(compressed_v2) - header_bytes
                         - circle_table_bytes - data_size_field)
    actual_data_no_zlib = (len(compressed_v2_no_zlib) - header_bytes
                           - circle_table_bytes - data_size_field)

    return {
        'original_bytes': len(data),
        'n_points': n_points,
        'n_circles': n_circles,
        'bits_circle': bits_circle,
        'bits_theta': bits_theta,
        'bits_per_point': bits_per_point,
        'delta_r': delta_r,
        'lossless': lossless,
        # ── v2 senza zlib ──
        'v2_header': header_bytes,
        'v2_circle_table': circle_table_bytes,
        'v2_data_size_field': data_size_field,
        'v2_raw_bitstream': raw_bitstream_bytes,
        'v2_no_zlib_total': len(compressed_v2_no_zlib),
        'v2_no_zlib_ratio': len(compressed_v2_no_zlib) / len(data),
        # ── v2 con zlib ──
        'v2_zlib_data': actual_data_bytes,
        'v2_zlib_total': len(compressed_v2),
        'v2_zlib_ratio': len(compressed_v2) / len(data),
        'zlib_saving': 1 - actual_data_bytes / max(actual_data_no_zlib, 1),
        # ── confronto con v1 ──
        'v1_total': len(compressed_v1),
        'v1_ratio': len(compressed_v1) / len(data),
        'v1_theoretical_ratio': r_v1['theoretical_ratio'],
        # ── miglioramento ──
        'v2_vs_v1_saving': 1 - len(compressed_v2) / max(len(compressed_v1), 1),
        'avg_points_per_circle': n_points / max(n_circles, 1),
    }
