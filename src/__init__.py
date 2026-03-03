"""
Polar Compression Algorithm
============================
Compressione di file basata sulla trasformata in coordinate polari.

Moduli:
    quantizer  — conversione cartesiano ↔ polare, raggruppamento per cerchi
    encoder    — compressione (file → formato .polr)
    decoder    — decompressione (formato .polr → file originale)
    utils      — funzioni di supporto (I/O, statistiche)
"""

from .encoder import encode
from .decoder import decode

__version__ = "0.1.0"
