"""
utils.py — Funzioni di supporto
================================

I/O per file .polr e reportistica.
"""

import os


def compress_file(input_path: str,
                  output_path: str = None,
                  delta_r: float = 1.0,
                  bits_theta: int = 12,
                  lossless: bool = True) -> dict:
    """
    Comprime un file su disco.

    Parametri
    ---------
    input_path : str
        Percorso del file da comprimere.
    output_path : str, optional
        Percorso del file compresso. Default: input_path + '.polr'
    delta_r, bits_theta, lossless : parametri di compressione.

    Ritorna
    -------
    dict con statistiche sulla compressione.
    """
    from .encoder import encode, compression_report

    if output_path is None:
        output_path = input_path + '.polr'

    with open(input_path, 'rb') as f:
        data = f.read()

    compressed = encode(data, delta_r=delta_r, bits_theta=bits_theta,
                        lossless=lossless)

    with open(output_path, 'wb') as f:
        f.write(compressed)

    report = compression_report(data, delta_r=delta_r, bits_theta=bits_theta,
                                lossless=lossless)
    report['input_path'] = input_path
    report['output_path'] = output_path
    report['compressed_file_bytes'] = len(compressed)

    return report


def decompress_file(input_path: str, output_path: str = None) -> dict:
    """
    Decomprime un file .polr su disco.

    Parametri
    ---------
    input_path : str
        Percorso del file .polr
    output_path : str, optional
        Percorso del file decompresso. Default: rimuove .polr

    Ritorna
    -------
    dict con info sulla decompressione.
    """
    from .decoder import decode, inspect_header

    if output_path is None:
        if input_path.endswith('.polr'):
            output_path = input_path[:-5]
        else:
            output_path = input_path + '.decoded'

    with open(input_path, 'rb') as f:
        compressed = f.read()

    header = inspect_header(compressed)
    decompressed = decode(compressed)

    with open(output_path, 'wb') as f:
        f.write(decompressed)

    return {
        'input_path': input_path,
        'output_path': output_path,
        'compressed_bytes': len(compressed),
        'decompressed_bytes': len(decompressed),
        'header': header,
    }


def print_report(report: dict):
    """Stampa un report leggibile della compressione."""
    print("\n" + "=" * 55)
    print("  REPORT COMPRESSIONE POLARE")
    print("=" * 55)
    print(f"  Originale:       {report['original_bytes']:>10,} byte")
    print(f"  Prototipo:       {report['prototype_bytes']:>10,} byte "
          f"({report['prototype_ratio']:.1%})")
    print(f"  Teorico ideale:  {report['theoretical_bytes']:>10,.0f} byte "
          f"({report['theoretical_ratio']:.1%})")
    print(f"  ─────────────────────────────────────")
    print(f"  Cerchi unici:    {report['n_circles']:>10,}")
    print(f"  Punti totali:    {report['n_points']:>10,}")
    print(f"  Media punti/cerchio: {report['avg_points_per_circle']:>7.1f}")
    print(f"  Δr = {report['delta_r']},  bits_θ = {report['bits_theta']},  "
          f"lossless = {report['lossless']}")
    print(f"  ─────────────────────────────────────")
    print(f"  Dettaglio dimensione prototipo:")
    print(f"    Header:         {report['header_bytes']:>8,} byte")
    print(f"    Tabella cerchi: {report['circle_table_bytes']:>8,} byte")
    print(f"    Dati angoli:    {report['point_data_bytes']:>8,} byte")
    print(f"    Mappa ordine:   {report['order_map_bytes']:>8,} byte")
    print("=" * 55)
