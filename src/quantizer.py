"""
quantizer.py — Cuore matematico della compressione polare
==========================================================

Questo modulo gestisce:
1. Conversione byte ↔ coppie (x, y)
2. Conversione cartesiano ↔ polare
3. Quantizzazione radiale (raggruppamento per cerchi)

TEORIA
------
Ogni coppia di byte consecutivi (x, y) diventa un punto nel piano [0,255]×[0,255].
Convertendo in coordinate polari (r, θ), i punti con raggio simile vengono
raggruppati sullo stesso "cerchio". Se un cerchio ha N punti, il raggio r si
memorizza UNA volta → risparmio proporzionale a N.

    r = √(x² + y²)          raggio: distanza dall'origine
    θ = atan2(y, x)          angolo: direzione del punto
    r_q = round(r/Δr) × Δr  raggio quantizzato (assegnazione al cerchio)
"""

import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# PASSO 1: Byte ↔ Coppie
# ═══════════════════════════════════════════════════════════════

def bytes_to_pairs(data: bytes) -> tuple[list[tuple[int, int]], bool]:
    """
    Spezza il flusso di byte in coppie consecutive (x, y).

    Esempio:
        [0x41, 0x6C, 0x67, 0x6F]  →  [(65, 108), (103, 111)]

    Se il numero di byte è dispari, l'ultimo byte viene accoppiato
    con 0 (padding). Il flag 'padded' serve per rimuoverlo in decodifica.

    Parametri
    ---------
    data : bytes
        Sequenza di byte da convertire.

    Ritorna
    -------
    pairs : lista di tuple (int, int)
        Ogni tupla è un punto (x, y) con x,y ∈ [0, 255].
    padded : bool
        True se è stato aggiunto un byte di padding.
    """
    padded = (len(data) % 2 != 0)

    pairs = []
    for i in range(0, len(data) - 1, 2):
        pairs.append((data[i], data[i + 1]))

    # Se dispari, l'ultimo byte va accoppiato con 0
    if padded:
        pairs.append((data[-1], 0))

    return pairs, padded


def pairs_to_bytes(pairs: list[tuple[int, int]], padded: bool) -> bytes:
    """
    Operazione inversa: ricostruisce il flusso di byte dalle coppie.

    Parametri
    ---------
    pairs : lista di tuple (int, int)
    padded : bool
        Se True, rimuove l'ultimo byte (era padding).

    Ritorna
    -------
    bytes : sequenza originale ricostruita.
    """
    result = bytearray()
    for x, y in pairs:
        result.append(x)
        result.append(y)

    if padded:
        result = result[:-1]  # via il byte di padding

    return bytes(result)


# ═══════════════════════════════════════════════════════════════
# PASSO 2: Cartesiano ↔ Polare
# ═══════════════════════════════════════════════════════════════

def cartesian_to_polar(x: int, y: int) -> tuple[float, float]:
    """
    Converte un punto (x, y) in coordinate polari (r, θ).

    Formule:
        r = √(x² + y²)
        θ = atan2(y, x)

    Poiché x, y ∈ [0, 255] (primo quadrante):
        r ∈ [0, 360.62]    (max = √(255² + 255²))
        θ ∈ [0, π/2]       (≈ 0 a 1.5708 radianti)

    Caso speciale: (0, 0) → r=0, θ=0
    """
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return float(r), float(theta)


def polar_to_cartesian(r: float, theta: float) -> tuple[int, int]:
    """
    Riconverte da polare a cartesiano con arrotondamento a interi.

    Formule:
        x = round(r × cos(θ))
        y = round(r × sin(θ))

    Il clamp a [0, 255] garantisce che i valori siano byte validi.
    Questa è la fonte di errore nella compressione lossy:
    l'arrotondamento può spostare il punto di ±1 in x e/o y.
    """
    x = int(np.clip(round(r * np.cos(theta)), 0, 255))
    y = int(np.clip(round(r * np.sin(theta)), 0, 255))
    return x, y


# ═══════════════════════════════════════════════════════════════
# PASSO 3: Quantizzazione Radiale
# ═══════════════════════════════════════════════════════════════

def quantize_radius(r: float, delta_r: float) -> float:
    """
    Assegna il raggio alla fascia (cerchio) più vicina.

    Formula:
        r_quantizzato = round(r / Δr) × Δr

    Esempio con Δr = 5.0:
        r = 102.3  →  round(102.3/5) × 5 = round(20.46) × 5 = 20 × 5 = 100.0
        r = 103.8  →  round(103.8/5) × 5 = round(20.76) × 5 = 21 × 5 = 105.0

    TRADE-OFF (Δr):
        Δr piccolo (0.5) → molti cerchi, pochi punti ciascuno → poca compressione
        Δr grande (5.0)  → pochi cerchi, molti punti ciascuno → molta compressione
                           ma più errore nella ricostruzione lossy
    """
    if delta_r <= 0:
        raise ValueError("delta_r deve essere > 0")
    return round(r / delta_r) * delta_r


# ═══════════════════════════════════════════════════════════════
# PASSO 4: Raggruppamento per Cerchi
# ═══════════════════════════════════════════════════════════════

def group_by_circles(pairs: list[tuple[int, int]], delta_r: float) -> dict:
    """
    Raggruppa tutti i punti per cerchio (raggio quantizzato).

    Questo è il cuore della compressione: punti con raggio simile
    finiscono sullo stesso cerchio. Il raggio si memorizza UNA volta
    per tutti i punti del cerchio.

    Parametri
    ---------
    pairs : lista di tuple (x, y)
    delta_r : float
        Risoluzione radiale.

    Ritorna
    -------
    dict : {r_quantizzato: [lista di dict con info per ogni punto]}
        Ogni dict contiene:
        - 'theta': angolo in radianti
        - 'r_exact': raggio esatto (prima della quantizzazione)
        - 'original': tupla (x, y) originale
        - 'index': posizione nel flusso originale (per riordinamento)
    """
    circles = defaultdict(list)

    for idx, (x, y) in enumerate(pairs):
        r, theta = cartesian_to_polar(x, y)
        r_q = quantize_radius(r, delta_r)

        circles[r_q].append({
            'theta': theta,
            'r_exact': r,
            'original': (x, y),
            'index': idx
        })

    return dict(circles)


# ═══════════════════════════════════════════════════════════════
# UTILITÀ: Statistiche sulla distribuzione
# ═══════════════════════════════════════════════════════════════

def circle_stats(pairs: list[tuple[int, int]], delta_r: float) -> dict:
    """
    Calcola statistiche sulla distribuzione dei punti nei cerchi.
    Utile per capire se il dataset è adatto alla compressione polare.

    Ritorna
    -------
    dict con:
        'n_points': numero totale di punti
        'n_circles': numero di cerchi distinti
        'avg_per_circle': media punti per cerchio
        'max_per_circle': massimo punti su un cerchio
        'single_point_circles': cerchi con 1 solo punto (spreco)
        'top_5_circles': i 5 cerchi più popolati (r, count)
        'theoretical_ratio': ratio di compressione teorico ideale
    """
    circles = group_by_circles(pairs, delta_r)

    counts = {r: len(pts) for r, pts in circles.items()}
    sorted_circles = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    n_points = len(pairs)
    n_circles = len(circles)

    # Ratio teorico: con bit-packing ideale
    # Originale: 16 bit per coppia
    # Polare: per ogni cerchio, r costa ~12 bit, ogni θ costa ~10 bit
    bits_r = 12  # sufficiente per coprire [0, 361] con precisione 0.1
    bits_theta = 10  # risoluzione angolare ~0.001 rad (sufficiente per primo quadrante)
    original_bits = n_points * 16
    polar_bits = sum(bits_r + count * bits_theta for count in counts.values())
    theoretical_ratio = polar_bits / original_bits if original_bits > 0 else 1.0

    return {
        'n_points': n_points,
        'n_circles': n_circles,
        'avg_per_circle': n_points / n_circles if n_circles > 0 else 0,
        'max_per_circle': max(counts.values()) if counts else 0,
        'single_point_circles': sum(1 for c in counts.values() if c == 1),
        'top_5_circles': sorted_circles[:5],
        'theoretical_ratio': theoretical_ratio,
    }
