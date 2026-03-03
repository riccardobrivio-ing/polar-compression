"""
decoder.py — Decompressore Polare
==================================

Legge un file .polr e ricostruisce i dati originali.

FLUSSO:
    1. Leggi l'header → estrai parametri (Δr, bits_θ, flags)
    2. Leggi la tabella cerchi → quanti cerchi, quanti punti ciascuno
    3. Leggi gli angoli θ (e i byte originali se lossless)
    4. Leggi la mappa ordine → indici per ricostruire la sequenza
    5. Ricomponi le coppie (x, y) nell'ordine originale
    6. Converti le coppie in flusso di byte
"""

import struct
import numpy as np
from .quantizer import polar_to_cartesian, pairs_to_bytes

MAGIC = b'POLR'


def decode(compressed: bytes) -> bytes:
    """
    Decomprime un file in formato .polr.

    In modalità lossless: usa i byte originali salvati nel file.
    In modalità lossy: ricostruisce (x, y) da (r_quantizzato, θ).

    Parametri
    ---------
    compressed : bytes
        File compresso in formato .polr

    Ritorna
    -------
    bytes : dati originali ricostruiti.

    Errori
    ------
    ValueError : se il file non è in formato .polr valido.
    """

    pos = 0  # cursore di lettura nel flusso di byte

    # ── STEP 1: Leggi l'HEADER (17 byte) ───────────────────────

    # Magic number: verifica che sia un file POLR
    magic = compressed[pos:pos + 4]
    pos += 4
    if magic != MAGIC:
        raise ValueError(
            f"Formato non riconosciuto. Atteso 'POLR', trovato '{magic}'"
        )

    # Versione del formato
    version = struct.unpack_from('<B', compressed, pos)[0]
    pos += 1
    if version != 1:
        raise ValueError(f"Versione {version} non supportata (attesa: 1)")

    # Lunghezza file originale
    orig_len = struct.unpack_from('<I', compressed, pos)[0]
    pos += 4

    # Numero di cerchi
    n_circles = struct.unpack_from('<H', compressed, pos)[0]
    pos += 2

    # Delta_r (non usato in decodifica, ma utile per debug)
    delta_r = struct.unpack_from('<f', compressed, pos)[0]
    pos += 4

    # Bit per theta
    bits_theta = struct.unpack_from('<B', compressed, pos)[0]
    pos += 1

    # Flags
    flags = struct.unpack_from('<B', compressed, pos)[0]
    pos += 1

    padded = bool(flags & 0b00000001)
    lossless = bool(flags & 0b00000010)

    # ── STEP 2: Leggi la TABELLA CERCHI (6B × K) ───────────────

    circle_table = []  # lista di (raggio, n_punti)
    total_points = 0

    for _ in range(n_circles):
        r_q = struct.unpack_from('<f', compressed, pos)[0]
        pos += 4
        n_points = struct.unpack_from('<H', compressed, pos)[0]
        pos += 2
        circle_table.append((r_q, n_points))
        total_points += n_points

    # ── STEP 3: Leggi i DATI ANGOLI ────────────────────────────

    max_theta_val = (1 << bits_theta) - 1
    half_pi = np.pi / 2

    # Lista di coppie (x, y) ricostruite, nell'ordine dei cerchi
    # (NON ancora nell'ordine originale del file)
    circle_order_pairs = []

    for r_q, n_points in circle_table:
        for _ in range(n_points):
            # Leggi θ quantizzato
            theta_quant = struct.unpack_from('<H', compressed, pos)[0]
            pos += 2

            if lossless:
                # Leggi i byte originali (ricostruzione esatta)
                x_orig = struct.unpack_from('<B', compressed, pos)[0]
                pos += 1
                y_orig = struct.unpack_from('<B', compressed, pos)[0]
                pos += 1
                circle_order_pairs.append((x_orig, y_orig))
            else:
                # Ricostruisci da (r_q, θ) → (x, y)
                theta = (theta_quant / max_theta_val) * half_pi
                x, y = polar_to_cartesian(r_q, theta)
                circle_order_pairs.append((x, y))

    # ── STEP 4: Leggi la MAPPA ORDINE ──────────────────────────

    # Ogni entry dice: "il punto i-esimo nell'ordine dei cerchi
    # va nella posizione original_index nel file originale"
    original_indices = []
    for _ in range(total_points):
        idx = struct.unpack_from('<I', compressed, pos)[0]
        pos += 4
        original_indices.append(idx)

    # ── STEP 5: Riordina nell'ordine originale ─────────────────

    # Crea un array vuoto della dimensione giusta
    pairs = [None] * total_points

    for i, idx in enumerate(original_indices):
        pairs[idx] = circle_order_pairs[i]

    # Verifica: nessun buco
    if any(p is None for p in pairs):
        raise ValueError("Errore nella ricostruzione: punti mancanti")

    # ── STEP 6: Coppie → Byte ─────────────────────────────────

    return pairs_to_bytes(pairs, padded)


def inspect_header(compressed: bytes) -> dict:
    """
    Legge solo l'header di un file .polr senza decomprimerlo.
    Utile per debug e analisi rapida.

    Ritorna
    -------
    dict con tutti i campi dell'header.
    """
    if len(compressed) < 17:
        raise ValueError("File troppo corto per essere un .polr valido")

    pos = 0
    magic = compressed[pos:pos + 4]; pos += 4
    version = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    orig_len = struct.unpack_from('<I', compressed, pos)[0]; pos += 4
    n_circles = struct.unpack_from('<H', compressed, pos)[0]; pos += 2
    delta_r = struct.unpack_from('<f', compressed, pos)[0]; pos += 4
    bits_theta = struct.unpack_from('<B', compressed, pos)[0]; pos += 1
    flags = struct.unpack_from('<B', compressed, pos)[0]; pos += 1

    return {
        'magic': magic.decode('ascii', errors='replace'),
        'version': version,
        'original_length': orig_len,
        'n_circles': n_circles,
        'delta_r': delta_r,
        'bits_theta': bits_theta,
        'padded': bool(flags & 1),
        'lossless': bool(flags & 2),
    }
