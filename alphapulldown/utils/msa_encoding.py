from __future__ import annotations

from typing import Dict, List

import numpy as np
from alphafold.common import residue_constants


def get_id_to_char_map() -> Dict[int, str]:
    # Protein and gap mapping; unknown -> X
    mapping = {i: ch for i, ch in residue_constants.ID_TO_HHBLITS_AA.items()}
    mapping[21] = '-'
    return mapping


def get_char_to_id_map() -> Dict[str, int]:
    mapping = {ch: i for i, ch in residue_constants.ID_TO_HHBLITS_AA.items()}
    mapping['-'] = 21
    return mapping


def ids_to_a3m(rows: np.ndarray) -> str:
    id_to_char = get_id_to_char_map()
    lines: List[str] = []
    for i, row in enumerate(rows):
        lines.append(f">sequence_{i}")
        lines.append(''.join(id_to_char.get(int(x), 'X') for x in row))
    return '\n'.join(lines) + '\n'


def a3m_to_ids(a3m_text: str) -> np.ndarray:
    char_to_id = get_char_to_id_map()
    seqs: List[str] = []
    curr: str | None = None
    for ln in a3m_text.splitlines():
        if not ln.strip():
            continue
        if ln.startswith('>'):
            if curr is not None:
                # Remove lowercase insertions per A3M convention
                seqs.append(''.join(ch for ch in curr if not ch.islower()))
            curr = ''
        else:
            curr = (curr or '') + ln.strip()
    if curr is not None:
        seqs.append(''.join(ch for ch in curr if not ch.islower()))
    if not seqs:
        return np.empty((0, 0), dtype=np.int32)
    rows = [np.array([char_to_id.get(ch, 20) for ch in s], dtype=np.int32) for s in seqs]
    return np.stack(rows, axis=0)


# -----------------------------
# AlphaFold 3 MSA encodings
# -----------------------------

# AF3 uses a different integer alphabet for MSA arrays than AF2 (HHblits order).
# See alphafold3/data/msa_features.py for details.

AF3_ID_TO_CHAR: Dict[int, str] = {
    # Protein
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
    10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V',
    20: 'X', 21: '-',
    # RNA
    22: 'A', 23: 'G', 24: 'C', 25: 'U',
    # DNA
    26: 'A', 27: 'G', 28: 'C', 29: 'T',
    # Unknown nucleic
    30: 'N',
}


def ids_to_a3m_af3(rows: np.ndarray) -> str:
    """Convert AF3-encoded MSA integer rows to A3M text.

    This uses the AF3 integer alphabet (protein 0-21, RNA/DNA 22-29, 30 unknown).
    """
    lines: List[str] = []
    for i, row in enumerate(rows):
        lines.append(f">sequence_{i}")
        lines.append(''.join(AF3_ID_TO_CHAR.get(int(x), 'X') for x in row))
    return '\n'.join(lines) + '\n'


