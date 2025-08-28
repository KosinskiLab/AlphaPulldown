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


