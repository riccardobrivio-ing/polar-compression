"""
decoder_v2.py — Decompressore Polare Ottimizzato (v2)
======================================================

Legge il formato .polr v2 e ricostruisce i dati originali.

DIFFERENZE con v1:
  - Nessuna mappa ordine da leggere (i punti sono già in ordine)
  - Bitstream compatto: legge esattamente N bit per valore
  - Supporto decompressione zlib sulla sezione dati
"""

import struct
import math
import zlib
import numpy as np

from .quantizer import polar_to_cartesian, pairs_to_bytes
from .bitstream import BitReader

MAGIC_V2 = b'PLR2'


def decode_v2(compressed: bytes) -> bytes:
    """
    Decomprime un file in formato .polr v2.

    Flusso:
    1. Leggi header → parametri
    2. Leggi tabella cerchi → raggi
    3. Decomprimi sezione dati (zlib se flaggato)
    4. Leggi bitstream: per ogni punto, (circle_id, θ)
    5. Ricostruisci coppie (x, y)
    6. Converti in byte

    Parametri
    ---------
    compressed : bytes
        File compresso formato .polr v2.

    Ritorna
    -------
    bytes : dati originali ricostruiti.
    """
    pos = 0

    # ── HEADER (18 byte) ──────────────────────────────────────

    magic = compressed[pos:pos + 4]; pos += 4
    if magic != MAGIC_V2:
        raise ValueError(
            f"Formato non riconosciuto. Atteso 'PLR2', trovato {magic!r}"
        )

    version = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    if version != 2:
        raise ValueError(f"Versione {version} non supportata (attesa: 2)")

    orig_len = struct.unpack_from('<I', compressed, pos)[0]; pos += 4
    n_circles = struct.unpack_from('<H', compressed, pos)[0]; pos += 2
    delta_r = struct.unpack_from('<f', compressed, pos)[0]; pos += 4
    bits_theta = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    bits_circle = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    flags = struct.unpack_from('<B', compressed, pos)[0]; pos += 1

    padded = bool(flags & 0b00000001)
    lossless = bool(flags & 0b00000010)
    use_zlib = bool(flags & 0b00000100)

    # File vuoto
    if orig_len == 0:
        return b''

    # ── TABELLA CERCHI (4B × K) ───────────────────────────────

    radii = []
    for _ in range(n_circles):
        r_q = struct.unpack_from('<f', compressed, pos)[0]; pos += 4
        radii.append(r_q)

    # ── DIMENSIONE SEZIONE DATI ───────────────────────────────

    data_section_len = struct.unpack_from('<I', compressed, pos)[0]; pos += 4

    # ── SEZIONE DATI ──────────────────────────────────────────

    data_section = compressed[pos:pos + data_section_len]

    # Decomprimi zlib se necessario
    if use_zlib:
        raw_bitstream = zlib.decompress(data_section)
    else:
        raw_bitstream = data_section

    # ── LEGGI IL BITSTREAM ────────────────────────────────────

    reader = BitReader(raw_bitstream)
    max_theta_val = (1 << bits_theta) - 1
    half_pi = np.pi / 2

    # Calcola quanti punti ci sono
    # orig_len byte → ceil(orig_len/2) coppie
    n_points = (orig_len + 1) // 2

    pairs = []
    for _ in range(n_points):
        circle_id = reader.read_bits(bits_circle)
        theta_quant = reader.read_bits(bits_theta)

        if lossless:
            # Leggi byte originali
            x = reader.read_bits(8)
            y = reader.read_bits(8)
            pairs.append((x, y))
        else:
            # Ricostruisci da (r, θ)
            r_q = radii[circle_id]
            theta = (theta_quant / max_theta_val) * half_pi
            x, y = polar_to_cartesian(r_q, theta)
            pairs.append((x, y))

    # ── RICOSTRUISCI I BYTE ───────────────────────────────────

    return pairs_to_bytes(pairs, padded)


def inspect_header_v2(compressed: bytes) -> dict:
    """Legge solo l'header senza decomprimere."""
    if len(compressed) < 18:
        raise ValueError("File troppo corto per .polr v2")

    pos = 0
    magic = compressed[pos:pos + 4]; pos += 4
    version = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    orig_len = struct.unpack_from('<I', compressed, pos)[0]; pos += 4
    n_circles = struct.unpack_from('<H', compressed, pos)[0]; pos += 2
    delta_r = struct.unpack_from('<f', compressed, pos)[0]; pos += 4
    bits_theta = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    bits_circle = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    flags = struct.unpack_from('<B', compressed, pos)[0]; pos += 1

    return {
        'magic': magic.decode('ascii', errors='replace'),
        'version': version,
        'original_length': orig_len,
        'n_circles': n_circles,
        'delta_r': delta_r,
        'bits_theta': bits_theta,
        'bits_circle': bits_circle,
        'padded': bool(flags & 1),
        'lossless': bool(flags & 2),
        'zlib': bool(flags & 4),
    }
