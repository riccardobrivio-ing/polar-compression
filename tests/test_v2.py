"""
test_v2.py — Test suite per il formato ottimizzato v2
======================================================

Verifica:
  1. BitWriter/BitReader: round-trip su valori arbitrari
  2. Lossless v2: encode→decode = originale (identico a v1)
  3. Lossy v2: errore contenuto
  4. Compatibilità: v2 produce output più piccolo di v1
  5. Zlib: con/senza produce lo stesso risultato in decodifica
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bitstream import BitWriter, BitReader
from src.encoder_v2 import encode_v2, compression_report_v2
from src.decoder_v2 import decode_v2, inspect_header_v2
from src.encoder import encode as encode_v1


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 1: BitWriter / BitReader
# ═══════════════════════════════════════════════════════════════

class TestBitstream:
    """Verifica che il bitstream sia corretto bit per bit."""

    def test_single_byte(self):
        """8 bit → 1 byte esatto."""
        w = BitWriter()
        w.write_bits(0b10110011, 8)
        data = w.flush()
        assert data == bytes([0b10110011])

        r = BitReader(data)
        assert r.read_bits(8) == 0b10110011

    def test_12_bits(self):
        """12 bit → 2 byte (con 4 bit di padding)."""
        w = BitWriter()
        w.write_bits(0b101010101010, 12)  # = 2730
        data = w.flush()
        assert len(data) == 2

        r = BitReader(data)
        assert r.read_bits(12) == 2730

    def test_multiple_values(self):
        """Scrivi più valori con larghezze diverse, rileggili."""
        values = [(7, 3), (255, 8), (4095, 12), (1, 1), (0, 5)]
        # (valore, n_bits)

        w = BitWriter()
        for val, bits in values:
            w.write_bits(val, bits)
        data = w.flush()

        r = BitReader(data)
        for val, bits in values:
            assert r.read_bits(bits) == val

    def test_many_12bit_values(self):
        """Simula 1000 angoli θ a 12 bit."""
        rng = np.random.RandomState(42)
        values = rng.randint(0, 4096, size=1000).tolist()

        w = BitWriter()
        for v in values:
            w.write_bits(v, 12)
        data = w.flush()

        # Dimensione attesa: 1000 × 12 / 8 = 1500 byte
        assert len(data) == 1500

        r = BitReader(data)
        for v in values:
            assert r.read_bits(12) == v

    def test_mixed_widths(self):
        """Simula il formato v2: circle_id (3 bit) + θ (12 bit)."""
        rng = np.random.RandomState(42)
        n_points = 500
        circle_ids = rng.randint(0, 8, size=n_points).tolist()  # 3 bit
        thetas = rng.randint(0, 4096, size=n_points).tolist()   # 12 bit

        w = BitWriter()
        for cid, th in zip(circle_ids, thetas):
            w.write_bits(cid, 3)
            w.write_bits(th, 12)
        data = w.flush()

        # 500 × (3 + 12) = 7500 bit = 937.5 → 938 byte
        assert len(data) == 938

        r = BitReader(data)
        for cid, th in zip(circle_ids, thetas):
            assert r.read_bits(3) == cid
            assert r.read_bits(12) == th


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 2: Lossless v2 round-trip
# ═══════════════════════════════════════════════════════════════

class TestV2Lossless:
    """encode_v2 → decode_v2 deve restituire l'originale."""

    def test_ascii(self):
        original = b"Hello, Polar Compression v2!"
        compressed = encode_v2(original, delta_r=1.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_random_1kb(self):
        original = os.urandom(1000)
        compressed = encode_v2(original, delta_r=1.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_random_10kb(self):
        original = os.urandom(10000)
        compressed = encode_v2(original, delta_r=2.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_odd_length(self):
        original = b"ABC"
        compressed = encode_v2(original, delta_r=1.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_single_byte(self):
        original = b"\x42"
        compressed = encode_v2(original, delta_r=1.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_empty(self):
        original = b""
        compressed = encode_v2(original, delta_r=1.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_all_zeros(self):
        original = b"\x00" * 200
        compressed = encode_v2(original, delta_r=1.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_all_ff(self):
        original = b"\xff" * 200
        compressed = encode_v2(original, delta_r=1.0, lossless=True)
        assert decode_v2(compressed) == original

    def test_no_zlib(self):
        """Lossless senza zlib: deve funzionare ugualmente."""
        original = os.urandom(1000)
        compressed = encode_v2(original, delta_r=1.0, lossless=True,
                               use_zlib=False)
        assert decode_v2(compressed) == original

    def test_various_delta_r(self):
        original = os.urandom(500)
        for dr in [0.5, 1.0, 2.0, 5.0, 10.0]:
            compressed = encode_v2(original, delta_r=dr, lossless=True)
            assert decode_v2(compressed) == original, f"Fallito Δr={dr}"

    def test_various_bits_theta(self):
        original = os.urandom(500)
        for bt in [8, 10, 12, 14, 16]:
            compressed = encode_v2(original, delta_r=1.0, bits_theta=bt,
                                   lossless=True)
            assert decode_v2(compressed) == original, f"Fallito bits_θ={bt}"


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 3: Lossy v2
# ═══════════════════════════════════════════════════════════════

class TestV2Lossy:
    """Errore contenuto nella ricostruzione lossy."""

    def test_error_bound(self):
        original = os.urandom(2000)
        compressed = encode_v2(original, delta_r=1.0, bits_theta=12,
                               lossless=False)
        decompressed = decode_v2(compressed)

        assert len(decompressed) == len(original)
        errors = [abs(a - b) for a, b in zip(original, decompressed)]
        avg = sum(errors) / len(errors)
        assert avg < 5.0, f"Errore medio: {avg:.2f}"

    def test_iq_signal(self):
        """Su dati IQ simulati: errore molto basso."""
        amp = 100
        phases = np.linspace(0, np.pi / 2, 500)
        iq = bytearray()
        for ph in phases:
            iq.extend([
                int(np.clip(round(amp * np.cos(ph)), 0, 255)),
                int(np.clip(round(amp * np.sin(ph)), 0, 255)),
            ])

        original = bytes(iq)
        compressed = encode_v2(original, delta_r=1.0, bits_theta=12,
                               lossless=False)
        decompressed = decode_v2(compressed)

        errors = [abs(a - b) for a, b in zip(original, decompressed)]
        avg = sum(errors) / len(errors)
        assert avg < 2.0, f"Errore medio IQ: {avg:.2f}"


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 4: v2 più piccolo di v1
# ═══════════════════════════════════════════════════════════════

class TestV2VsV1:
    """Verifica che v2 produca file più piccoli di v1."""

    def test_smaller_lossy(self):
        """Su 10KB random, v2 lossy deve essere più piccolo di v1 lossy."""
        data = os.urandom(10000)
        v1 = encode_v1(data, delta_r=2.0, bits_theta=12, lossless=False)
        v2 = encode_v2(data, delta_r=2.0, bits_theta=12, lossless=False)

        assert len(v2) < len(v1), (
            f"v2 ({len(v2):,}) non è più piccolo di v1 ({len(v1):,})"
        )

    def test_smaller_lossless(self):
        data = os.urandom(10000)
        v1 = encode_v1(data, delta_r=2.0, bits_theta=12, lossless=True)
        v2 = encode_v2(data, delta_r=2.0, bits_theta=12, lossless=True)

        assert len(v2) < len(v1)

    def test_dramatic_improvement_iq(self):
        """Su IQ simulato a 2 ampiezze, v2 deve essere MOLTO più piccolo."""
        amps = [80, 120]
        phases = np.random.uniform(0, np.pi / 2, 5000)
        iq = bytearray()
        for ph in phases:
            a = np.random.choice(amps)
            iq.extend([
                int(np.clip(round(a * np.cos(ph)), 0, 255)),
                int(np.clip(round(a * np.sin(ph)), 0, 255)),
            ])

        data = bytes(iq)
        v1 = encode_v1(data, delta_r=2.0, bits_theta=12, lossless=False)
        v2 = encode_v2(data, delta_r=2.0, bits_theta=12, lossless=False)

        improvement = 1 - len(v2) / len(v1)
        print(f"\n  IQ 2 ampiezze: v1={len(v1):,}  v2={len(v2):,}  "
              f"miglioramento={improvement:.1%}")

        assert improvement > 0.5, f"Miglioramento solo {improvement:.1%}"


# ═══════════════════════════════════════════════════════════════
# TEST GRUPPO 5: Header e formato
# ═══════════════════════════════════════════════════════════════

class TestV2Format:

    def test_magic(self):
        compressed = encode_v2(b"test", lossless=True)
        assert compressed[:4] == b'PLR2'

    def test_header(self):
        original = b"test data here!!"
        compressed = encode_v2(original, delta_r=3.0, bits_theta=10,
                               lossless=True, use_zlib=False)
        h = inspect_header_v2(compressed)

        assert h['magic'] == 'PLR2'
        assert h['version'] == 2
        assert h['original_length'] == len(original)
        assert abs(h['delta_r'] - 3.0) < 0.01
        assert h['bits_theta'] == 10
        assert h['lossless'] is True
        assert h['zlib'] is False

    def test_invalid_magic(self):
        with pytest.raises(ValueError):
            decode_v2(b"NOTPLRdata_here!")
