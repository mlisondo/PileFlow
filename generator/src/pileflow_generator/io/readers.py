"""
Input-file readers and preparation helpers.
"""

import os
import gzip
import shutil


def decompress_lhe_if_needed(lhe_file: str) -> str:
    """
    Decompress a .lhe.gz file if needed.

    Parameters
    ----------
    lhe_file:
        Path to an LHE file. May be either `.lhe` or `.lhe.gz`.

    Returns
    -------
    str
        Path to the uncompressed `.lhe` file.
    """
    if not lhe_file.endswith(".gz"):
        return lhe_file

    out_path = lhe_file[:-3]

    if os.path.exists(out_path):
        return out_path

    with gzip.open(lhe_file, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)

    return out_path