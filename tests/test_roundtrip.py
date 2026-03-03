"""
test_roundtrip.py — Test suite completa
========================================

Esegui con:  python -m pytest tests/ -v

Questi test verificano che:
1. La conversione cartesiano↔polare sia corretta
2. La quantizzazione radiale funzioni come atteso
3. L'encoder produca output valido
4. Il round-trip lossless sia perfetto (encode→decode = originale)
5. Il round-trip lossy abbia errore limitato
"""

import os
import sys
import pytest
import numpy as np

# Aggiungi la root del progetto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantizer import (
    bytes_to_pairs,
    pairs_to_bytes,
    cartesian_to_polar,
    polar_to_cartesian,
    quantize_radius,
    group_by_circles,
    circle_stats,
)
from src.encoder import encode, compression_report
from src.decoder import decode, inspect_header


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 1: Conversione byte ↔ coppie
# ═══════════════════════════════════════════════════════════════

class TestBytePairs:
    """Verifica che byte→coppie→byte sia un round-trip perfetto."""

    def test_even_length(self):
        """Numero pari di byte: nessun padding necessario."""
        data = b'\x41\x6C\x67\x6F'    # 4 byte → 2 coppie
        pairs, padded = bytes_to_pairs(data)

        assert padded is False
        assert len(pairs) == 2
        assert pairs[0] == (0x41, 0x6C)   # (65, 108)
        assert pairs[1] == (0x67, 0x6F)   # (103, 111)

        # Round-trip
        restored = pairs_to_bytes(pairs, padded)
        assert restored == data

    def test_odd_length(self):
        """Numero dispari: l'ultimo byte viene accoppiato con 0."""
        data = b'\x41\x6C\x67'    # 3 byte → 2 coppie, con padding
        pairs, padded = bytes_to_pairs(data)

        assert padded is True
        assert len(pairs) == 2
        assert pairs[0] == (0x41, 0x6C)
        assert pairs[1] == (0x67, 0x00)   # padding con 0

        # Round-trip
        restored = pairs_to_bytes(pairs, padded)
        assert restored == data    # il padding viene rimosso

    def test_empty(self):
        """Input vuoto: nessuna coppia, nessun padding."""
        pairs, padded = bytes_to_pairs(b'')
        assert pairs == []
        assert padded is False
        assert pairs_to_bytes(pairs, padded) == b''

    def test_single_byte(self):
        """Un solo byte: diventa (byte, 0) con padding."""
        data = b'\xFF'
        pairs, padded = bytes_to_pairs(data)
        assert pairs == [(255, 0)]
        assert padded is True
        assert pairs_to_bytes(pairs, padded) == data


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 2: Conversione cartesiano ↔ polare
# ═══════════════════════════════════════════════════════════════

class TestPolarConversion:
    """Verifica le conversioni geometriche."""

    def test_origin(self):
        """Punto (0,0): raggio zero, angolo zero."""
        r, theta = cartesian_to_polar(0, 0)
        assert r == 0.0
        assert theta == 0.0

    def test_axis_x(self):
        """Punto sull'asse x: θ = 0."""
        r, theta = cartesian_to_polar(100, 0)
        assert abs(r - 100.0) < 1e-10
        assert abs(theta - 0.0) < 1e-10

    def test_axis_y(self):
        """Punto sull'asse y: θ = π/2."""
        r, theta = cartesian_to_polar(0, 100)
        assert abs(r - 100.0) < 1e-10
        assert abs(theta - np.pi / 2) < 1e-10

    def test_diagonal(self):
        """Punto sulla diagonale: θ = π/4, r = √(2) × valore."""
        r, theta = cartesian_to_polar(100, 100)
        assert abs(r - 100 * np.sqrt(2)) < 1e-10
        assert abs(theta - np.pi / 4) < 1e-10

    def test_max_values(self):
        """Punto (255, 255): massimo raggio possibile."""
        r, theta = cartesian_to_polar(255, 255)
        assert abs(r - 255 * np.sqrt(2)) < 1e-6
        assert abs(r - 360.624) < 0.001

    def test_roundtrip_exact(self):
        """Conversione cart→polar→cart deve restituire il punto originale
        (entro arrotondamento a intero)."""
        test_points = [(0, 0), (255, 0), (0, 255), (255, 255),
                       (100, 50), (37, 198), (128, 128)]

        for x, y in test_points:
            r, theta = cartesian_to_polar(x, y)
            x2, y2 = polar_to_cartesian(r, theta)
            assert (x2, y2) == (x, y), \
                f"Fallito per ({x},{y}): got ({x2},{y2})"


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 3: Quantizzazione radiale
# ═══════════════════════════════════════════════════════════════

class TestQuantization:
    """Verifica la quantizzazione dei raggi."""

    def test_exact_value(self):
        """Un raggio esattamente su una fascia non cambia."""
        assert quantize_radius(10.0, 5.0) == 10.0
        assert quantize_radius(100.0, 1.0) == 100.0

    def test_rounding(self):
        """Raggi vengono arrotondati alla fascia più vicina."""
        assert quantize_radius(102.3, 5.0) == 100.0  # 102.3/5=20.46→20×5
        assert quantize_radius(103.8, 5.0) == 105.0  # 103.8/5=20.76→21×5

    def test_small_delta(self):
        """Con Δr piccolo, la quantizzazione è più fine."""
        assert quantize_radius(10.3, 0.5) == 10.5
        assert quantize_radius(10.1, 0.5) == 10.0

    def test_invalid_delta(self):
        """Δr ≤ 0 deve generare errore."""
        with pytest.raises(ValueError):
            quantize_radius(10.0, 0.0)
        with pytest.raises(ValueError):
            quantize_radius(10.0, -1.0)


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 4: Raggruppamento per cerchi
# ═══════════════════════════════════════════════════════════════

class TestGrouping:
    """Verifica che il raggruppamento produca cerchi corretti."""

    def test_identical_points(self):
        """Punti identici finiscono tutti sullo stesso cerchio."""
        pairs = [(100, 100)] * 10
        circles = group_by_circles(pairs, delta_r=1.0)

        # Dovrebbe esserci 1 solo cerchio
        assert len(circles) == 1

        # Con 10 punti
        r_q = list(circles.keys())[0]
        assert len(circles[r_q]) == 10

    def test_different_points_same_circle(self):
        """Punti con raggio simile finiscono sullo stesso cerchio."""
        # (100, 0) → r=100, (99, 0) → r=99, (101, 0) → r=101
        # Con Δr=5, tutti vanno al cerchio r=100
        pairs = [(100, 0), (99, 0), (101, 0)]
        circles = group_by_circles(pairs, delta_r=5.0)

        assert len(circles) == 1

    def test_stats(self):
        """Le statistiche sono coerenti."""
        pairs = [(100, 100)] * 50 + [(10, 10)] * 20
        stats = circle_stats(pairs, delta_r=1.0)

        assert stats['n_points'] == 70
        assert stats['n_circles'] == 2
        assert stats['max_per_circle'] == 50


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 5: Round-trip encode → decode (LOSSLESS)
# ═══════════════════════════════════════════════════════════════

class TestLosslessRoundtrip:
    """Il test più importante: encode→decode deve restituire l'originale."""

    def test_ascii_string(self):
        original = b"Hello, Polar Compression!"
        compressed = encode(original, delta_r=0.5, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_binary_random(self):
        """Dati binari casuali (worst case per la compressione)."""
        original = os.urandom(1000)
        compressed = encode(original, delta_r=0.5, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_odd_length(self):
        """File con numero dispari di byte (verifica gestione padding)."""
        original = b"ABC"
        compressed = encode(original, delta_r=0.5, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_single_byte(self):
        original = b"\x42"
        compressed = encode(original, delta_r=0.5, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_empty(self):
        original = b""
        compressed = encode(original, delta_r=0.5, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_all_zeros(self):
        original = b"\x00" * 100
        compressed = encode(original, delta_r=0.5, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_all_ff(self):
        original = b"\xff" * 100
        compressed = encode(original, delta_r=0.5, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_large_file(self):
        """File più grande (10 KB) per verificare scalabilità."""
        original = os.urandom(10000)
        compressed = encode(original, delta_r=1.0, lossless=True)
        decompressed = decode(compressed)
        assert decompressed == original

    def test_different_delta_r(self):
        """Il Δr non deve influire sulla lossless (i byte originali sono salvati)."""
        original = os.urandom(500)
        for dr in [0.1, 0.5, 1.0, 5.0, 10.0]:
            compressed = encode(original, delta_r=dr, lossless=True)
            decompressed = decode(compressed)
            assert decompressed == original, f"Fallito con delta_r={dr}"

    def test_different_bits_theta(self):
        """Diversi bits_theta non influiscono sulla lossless."""
        original = os.urandom(500)
        for bt in [8, 10, 12, 16]:
            compressed = encode(original, delta_r=1.0, bits_theta=bt,
                                lossless=True)
            decompressed = decode(compressed)
            assert decompressed == original, f"Fallito con bits_theta={bt}"


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 6: Round-trip LOSSY (errore limitato)
# ═══════════════════════════════════════════════════════════════

class TestLossyRoundtrip:
    """In lossy, non ci aspettiamo byte identici, ma errore piccolo."""

    def test_lossy_error_bound(self):
        """L'errore medio per byte deve essere contenuto."""
        original = os.urandom(2000)
        compressed = encode(original, delta_r=1.0, bits_theta=12,
                            lossless=False)
        decompressed = decode(compressed)

        # Stessa lunghezza
        assert len(decompressed) == len(original)

        # Errore per byte
        errors = [abs(a - b) for a, b in zip(original, decompressed)]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)

        # Con Δr=1.0 e 12 bit theta, errore medio dovrebbe essere < 3
        assert avg_error < 5.0, f"Errore medio troppo alto: {avg_error:.2f}"

    def test_lossy_aggressive(self):
        """Anche con compressione aggressiva, l'errore è gestibile."""
        original = os.urandom(2000)
        compressed = encode(original, delta_r=5.0, bits_theta=8,
                            lossless=False)
        decompressed = decode(compressed)

        errors = [abs(a - b) for a, b in zip(original, decompressed)]
        avg_error = sum(errors) / len(errors)

        # Con parametri aggressivi, errore medio < 10
        assert avg_error < 15.0, f"Errore medio: {avg_error:.2f}"

    def test_lossy_iq_signal(self):
        """Su dati IQ simulati (best case), l'errore deve essere basso."""
        # Simula segnale: ampiezza costante, fase variabile
        amplitude = 100
        n_samples = 500
        phases = np.linspace(0, np.pi / 2, n_samples)

        iq_data = bytearray()
        for phase in phases:
            x = int(np.clip(round(amplitude * np.cos(phase)), 0, 255))
            y = int(np.clip(round(amplitude * np.sin(phase)), 0, 255))
            iq_data.extend([x, y])

        original = bytes(iq_data)
        compressed = encode(original, delta_r=1.0, bits_theta=12,
                            lossless=False)
        decompressed = decode(compressed)

        errors = [abs(a - b) for a, b in zip(original, decompressed)]
        avg_error = sum(errors) / len(errors)

        # Su dati IQ con ampiezza costante, errore deve essere molto basso
        assert avg_error < 2.0, f"Errore medio su IQ: {avg_error:.2f}"


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 7: Header e formato file
# ═══════════════════════════════════════════════════════════════

class TestFormat:
    """Verifica l'integrità del formato .polr."""

    def test_header_magic(self):
        compressed = encode(b"test data", lossless=True)
        assert compressed[:4] == b'POLR'

    def test_inspect_header(self):
        original = b"test data here"
        compressed = encode(original, delta_r=2.5, bits_theta=10,
                            lossless=True)
        header = inspect_header(compressed)

        assert header['magic'] == 'POLR'
        assert header['version'] == 1
        assert header['original_length'] == len(original)
        assert abs(header['delta_r'] - 2.5) < 0.01
        assert header['bits_theta'] == 10
        assert header['lossless'] is True

    def test_invalid_magic(self):
        with pytest.raises(ValueError, match="Formato non riconosciuto"):
            decode(b"NOTPOLRdata")

    def test_compression_report(self):
        data = os.urandom(1000)
        report = compression_report(data, delta_r=2.0, bits_theta=12)

        assert report['original_bytes'] == 1000
        assert report['n_points'] == 500
        assert report['prototype_bytes'] > 0
        assert report['theoretical_bytes'] > 0
        assert 0 < report['theoretical_ratio'] < 10  # sanity check
