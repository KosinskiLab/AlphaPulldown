import numpy as np

from alphapulldown.utils.af2_to_af3_msa import (
    msa_rows_and_deletions_to_a3m,
    translate_af2_complex_msa_to_af3_chain_msas,
)


AF2_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX-"
AF2_TOKEN_BY_RESIDUE = {residue: index for index, residue in enumerate(AF2_ALPHABET)}


def _encode(sequence: str) -> np.ndarray:
    return np.array([AF2_TOKEN_BY_RESIDUE[residue] for residue in sequence], dtype=np.int32)


def _a3m_sequences(a3m: str) -> list[str]:
    if not a3m:
        return []
    lines = [line.strip() for line in a3m.splitlines() if line.strip()]
    return [lines[index] for index in range(1, len(lines), 2)]


def _a3m_payload_sequences(a3m: str) -> list[str]:
    sequences = _a3m_sequences(a3m)
    return sequences[1:]


def _aligned_and_deletions(sequence: str) -> tuple[str, list[int]]:
    aligned_chars = []
    deletion_counts = []
    pending_deletions = 0
    for residue in sequence:
        if residue.islower():
            pending_deletions += 1
            continue
        aligned_chars.append(residue)
        deletion_counts.append(pending_deletions)
        pending_deletions = 0
    return "".join(aligned_chars), deletion_counts


def test_msa_rows_and_deletions_to_a3m_preserves_lowercase_compression():
    a3m = msa_rows_and_deletions_to_a3m(
        msa_rows=np.stack([_encode("A-C")]),
        deletion_rows=np.array([[0, 2, 1]], dtype=np.int32),
        query_sequence="AQC",
    )

    assert _a3m_sequences(a3m) == ["AQC", "Aaa-aC"]
    assert _a3m_payload_sequences(a3m) == ["Aaa-aC"]
    assert _aligned_and_deletions(_a3m_payload_sequences(a3m)[0]) == ("A-C", [0, 2, 1])


def test_translate_af2_complex_msa_splits_paired_and_unpaired_rows():
    merged_msa = np.stack(
        [
            _encode("ACGT"),
            _encode("A-G-"),
            _encode("AA--"),
            _encode("--GG"),
            _encode("----"),
        ]
    )
    deletion_matrix = np.array(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    translated = translate_af2_complex_msa_to_af3_chain_msas(
        merged_msa=merged_msa,
        chain_sequences=["AC", "GT"],
        num_alignments=4,
        deletion_matrix=deletion_matrix,
        asym_id=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    assert _a3m_payload_sequences(translated[0].paired_msa) == ["Aaa-"]
    assert _a3m_payload_sequences(translated[1].paired_msa) == ["Ga-"]
    assert _a3m_payload_sequences(translated[0].unpaired_msa) == ["aAA"]
    assert _a3m_payload_sequences(translated[1].unpaired_msa) == ["aaGG"]
    assert _aligned_and_deletions(_a3m_payload_sequences(translated[0].paired_msa)[0]) == (
        "A-",
        [0, 2],
    )
    assert _aligned_and_deletions(_a3m_payload_sequences(translated[1].paired_msa)[0]) == (
        "G-",
        [0, 1],
    )
    assert _aligned_and_deletions(_a3m_payload_sequences(translated[0].unpaired_msa)[0]) == (
        "AA",
        [1, 0],
    )
    assert _aligned_and_deletions(_a3m_payload_sequences(translated[1].unpaired_msa)[0]) == (
        "GG",
        [2, 0],
    )


def test_translate_af2_complex_msa_keeps_block_diagonal_rows_unpaired():
    merged_msa = np.stack(
        [
            _encode("AC--"),
            _encode("AA--"),
            _encode("--GT"),
            _encode("--GG"),
            _encode("----"),
        ]
    )
    deletion_matrix = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 3, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    translated = translate_af2_complex_msa_to_af3_chain_msas(
        merged_msa=merged_msa,
        chain_sequences=["AC", "GT"],
        num_alignments=4,
        deletion_matrix=deletion_matrix,
        asym_id=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    assert translated[0].paired_msa == ""
    assert translated[1].paired_msa == ""
    assert _a3m_payload_sequences(translated[0].unpaired_msa) == ["aAA"]
    assert _a3m_payload_sequences(translated[1].unpaired_msa) == ["aaaGG"]
    assert _aligned_and_deletions(_a3m_payload_sequences(translated[0].unpaired_msa)[0]) == (
        "AA",
        [1, 0],
    )
    assert _aligned_and_deletions(_a3m_payload_sequences(translated[1].unpaired_msa)[0]) == (
        "GG",
        [3, 0],
    )
