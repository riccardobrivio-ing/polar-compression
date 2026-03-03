"""
bitstream.py — Lettura/scrittura di valori a bit arbitrari
============================================================

Il prototipo v1 spreca byte perché usa uint16 (16 bit) per valori
che ne richiedono solo 12, e uint32 (32 bit) per indici.

Questa classe permette di scrivere/leggere valori con esattamente
N bit, impaccati in un flusso continuo senza sprechi.

ESEMPIO: scrivere 3 valori da 12 bit
─────────────────────────────────────
  v1 (byte-level):  3 × 2 byte = 6 byte = 48 bit (spreco: 12 bit)
  v2 (bit-packing): 3 × 12 bit = 36 bit = 5 byte (risparmio: 1 byte)

Su 5000 punti con 12 bit per θ:
  v1: 10,000 byte
  v2:  7,500 byte  → risparmio 2,500 byte

COME FUNZIONA
─────────────
I bit vengono accumulati in un buffer interno. Quando il buffer
raggiunge 8 bit, un byte completo viene emesso nell'output.

  write_bits(valore=0b101010101010, n_bits=12):
    buffer: [1 0 1 0 1 0 1 0 | 1 0 1 0 ...]
             ^^^^^^^^^^^^^^^^   ^^^^^^^^^^
             primo byte (0xAA)  restano nel buffer
"""


class BitWriter:
    """Scrive valori con larghezza in bit arbitraria in un bytearray."""

    def __init__(self):
        self._buffer = 0       # bit in attesa di essere scritti
        self._bits_in_buf = 0  # quanti bit validi ci sono nel buffer
        self._output = bytearray()

    def write_bits(self, value: int, n_bits: int):
        """
        Scrive esattamente n_bits del valore nel flusso.

        I bit vengono scritti dal più significativo al meno significativo
        (big-endian bit order), che è lo standard per bitstream di compressione.

        Parametri
        ---------
        value : int
            Il valore da scrivere (deve essere < 2^n_bits).
        n_bits : int
            Quanti bit scrivere (1-32).
        """
        if n_bits <= 0:
            return
        # Maschera per sicurezza
        value &= (1 << n_bits) - 1

        # Aggiungi i bit al buffer
        self._buffer = (self._buffer << n_bits) | value
        self._bits_in_buf += n_bits

        # Emetti tutti i byte completi
        while self._bits_in_buf >= 8:
            self._bits_in_buf -= 8
            byte = (self._buffer >> self._bits_in_buf) & 0xFF
            self._output.append(byte)

    def flush(self) -> bytes:
        """
        Chiudi il flusso: se ci sono bit residui (< 8), vengono
        paddati con zeri a destra per completare l'ultimo byte.

        Ritorna
        -------
        bytes : il flusso di bit compattato.
        """
        if self._bits_in_buf > 0:
            # Pad con zeri a destra
            byte = (self._buffer << (8 - self._bits_in_buf)) & 0xFF
            self._output.append(byte)

        result = bytes(self._output)
        # Reset
        self._buffer = 0
        self._bits_in_buf = 0
        self._output = bytearray()
        return result

    @property
    def bits_written(self) -> int:
        """Numero totale di bit scritti (inclusi quelli nel buffer)."""
        return len(self._output) * 8 + self._bits_in_buf


class BitReader:
    """Legge valori con larghezza in bit arbitraria da un flusso di byte."""

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0     # posizione nel flusso di byte
        self._buffer = 0       # bit già letti ma non ancora consumati
        self._bits_in_buf = 0

    def read_bits(self, n_bits: int) -> int:
        """
        Legge esattamente n_bits dal flusso e li restituisce come intero.

        Parametri
        ---------
        n_bits : int
            Quanti bit leggere (1-32).

        Ritorna
        -------
        int : il valore letto.
        """
        # Riempi il buffer finché non ha abbastanza bit
        while self._bits_in_buf < n_bits:
            if self._byte_pos >= len(self._data):
                # Fine del flusso: pad con zeri
                self._buffer <<= 8
                self._bits_in_buf += 8
            else:
                self._buffer = (self._buffer << 8) | self._data[self._byte_pos]
                self._byte_pos += 1
                self._bits_in_buf += 8

        # Estrai i bit richiesti dalla cima del buffer
        self._bits_in_buf -= n_bits
        value = (self._buffer >> self._bits_in_buf) & ((1 << n_bits) - 1)
        return value

    @property
    def bits_remaining(self) -> int:
        """Bit ancora disponibili nel flusso (stima)."""
        return (len(self._data) - self._byte_pos) * 8 + self._bits_in_buf
