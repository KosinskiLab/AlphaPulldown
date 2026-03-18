import numpy as np

from alphapulldown.utils.af2_to_af3_msa import (
    msa_rows_and_deletions_to_a3m,
    translate_af2_complex_msa_to_af3_chain_msas,
    translate_af2_complex_msa_to_af3_chain_msas_with_stats,
    translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats,
    translate_af2_individual_chain_features_to_af3_msas_with_stats,
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


def _a3m_descriptions(a3m: str) -> list[str]:
    if not a3m:
        return []
    lines = [line.strip() for line in a3m.splitlines() if line.strip()]
    return [lines[index][1:] for index in range(0, len(lines), 2)]


def _a3m_payload_sequences(a3m: str) -> list[str]:
    sequences = _a3m_sequences(a3m)
    return sequences[1:]


def _a3m_payload_descriptions(a3m: str) -> list[str]:
    descriptions = _a3m_descriptions(a3m)
    return descriptions[1:]


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


def test_translation_stats_preserve_deletion_aware_rows():
    result = translate_af2_complex_msa_to_af3_chain_msas_with_stats(
        merged_msa=np.stack(
            [
                _encode("ACGT"),
                _encode("A-G-"),
                _encode("AA--"),
            ]
        ),
        chain_sequences=["AC", "GT"],
        num_alignments=3,
        deletion_matrix=np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        asym_id=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    assert result.total_rows_considered == 3
    assert result.occupancy_histogram == {"0": 0, "1": 1, "ge_2": 2}
    assert result.paired_row_count == 2
    assert result.invalid_paired_rows == 0
    assert result.invalid_unpaired_rows == 0
    assert result.chain_stats[0].paired_msa_row_count == 1
    assert result.chain_stats[0].unpaired_msa_row_count == 1
    assert _a3m_payload_sequences(result.chain_msas[0].paired_msa) == ["Aaa-"]
    assert _a3m_payload_sequences(result.chain_msas[1].paired_msa) == ["Ga-"]


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


def test_translation_stats_report_mixed_paired_and_unpaired_rows():
    result = translate_af2_complex_msa_to_af3_chain_msas_with_stats(
        merged_msa=np.stack(
            [
                _encode("ACGT"),
                _encode("A-G-"),
                _encode("AA--"),
                _encode("--GG"),
                _encode("----"),
            ]
        ),
        chain_sequences=["AC", "GT"],
        num_alignments=4,
        deletion_matrix=np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        asym_id=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    assert result.total_rows_considered == 4
    assert result.occupancy_histogram == {"0": 0, "1": 2, "ge_2": 2}
    assert result.paired_row_count == 2
    assert result.per_chain_unpaired_row_counts == (1, 1)
    assert result.invalid_paired_rows == 0
    assert result.invalid_unpaired_rows == 0
    assert result.chain_stats[0].paired_msa_row_count == 1
    assert result.chain_stats[1].paired_msa_row_count == 1
    assert result.chain_stats[0].unpaired_msa_row_count == 1
    assert result.chain_stats[1].unpaired_msa_row_count == 1


def test_translation_stats_report_block_diagonal_rows_as_unpaired():
    result = translate_af2_complex_msa_to_af3_chain_msas_with_stats(
        merged_msa=np.stack(
            [
                _encode("AC--"),
                _encode("AA--"),
                _encode("--GT"),
                _encode("--GG"),
                _encode("----"),
            ]
        ),
        chain_sequences=["AC", "GT"],
        num_alignments=4,
        deletion_matrix=np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 2],
                [0, 0, 3, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        asym_id=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    assert result.total_rows_considered == 4
    assert result.occupancy_histogram == {"0": 0, "1": 4, "ge_2": 0}
    assert result.paired_row_count == 0
    assert result.per_chain_unpaired_row_counts == (1, 1)
    assert result.invalid_paired_rows == 0
    assert result.invalid_unpaired_rows == 0
    assert result.chain_stats[0].paired_msa_row_count == 0
    assert result.chain_stats[1].paired_msa_row_count == 0
    assert result.chain_stats[0].unpaired_msa_row_count == 1
    assert result.chain_stats[1].unpaired_msa_row_count == 1


def test_manual_unpaired_translation_preserves_full_af2_row_order():
    result = translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats(
        merged_msa=np.stack(
            [
                _encode("ACGT"),
                _encode("A-G-"),
                _encode("AA--"),
                _encode("--GG"),
            ]
        ),
        chain_sequences=["AC", "GT"],
        num_alignments=4,
        deletion_matrix=np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 2, 0],
            ],
            dtype=np.int32,
        ),
        asym_id=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    assert result.translation_mode == "manual_unpaired_from_af2_multimer"
    assert result.total_rows_considered == 4
    assert result.occupancy_histogram == {"0": 0, "1": 2, "ge_2": 2}
    assert result.paired_row_count == 2
    assert result.invalid_paired_rows == 0
    assert result.invalid_unpaired_rows == 0
    assert result.chain_stats[0].paired_msa_row_count == 0
    assert result.chain_stats[1].paired_msa_row_count == 0
    assert result.chain_stats[0].unpaired_msa_row_count == 3
    assert result.chain_stats[1].unpaired_msa_row_count == 3
    assert result.chain_msas[0].paired_msa == ""
    assert result.chain_msas[1].paired_msa == ""
    assert _a3m_payload_sequences(result.chain_msas[0].unpaired_msa) == [
        "Aaa-",
        "aAA",
        "--",
    ]
    assert _a3m_payload_sequences(result.chain_msas[1].unpaired_msa) == [
        "Ga-",
        "--",
        "aaGG",
    ]


def test_manual_unpaired_translation_keeps_block_diagonal_rows_for_alignment():
    result = translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats(
        merged_msa=np.stack(
            [
                _encode("ACGT"),
                _encode("AA--"),
                _encode("--GT"),
                _encode("--GG"),
            ]
        ),
        chain_sequences=["AC", "GT"],
        num_alignments=4,
        deletion_matrix=np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 2],
                [0, 0, 3, 0],
            ],
            dtype=np.int32,
        ),
        asym_id=np.array([1, 1, 2, 2], dtype=np.int32),
    )

    assert result.translation_mode == "manual_unpaired_from_af2_multimer"
    assert result.occupancy_histogram == {"0": 0, "1": 3, "ge_2": 1}
    assert result.paired_row_count == 1
    assert result.chain_stats[0].unpaired_msa_row_count == 3
    assert result.chain_stats[1].unpaired_msa_row_count == 3
    assert _a3m_payload_sequences(result.chain_msas[0].unpaired_msa) == [
        "aAA",
        "--",
        "--",
    ]
    assert _a3m_payload_sequences(result.chain_msas[1].unpaired_msa) == [
        "--",
        "GaaT",
        "aaaGG",
    ]


def test_translate_af2_individual_chain_features_builds_species_aware_paired_msas():
    result = translate_af2_individual_chain_features_to_af3_msas_with_stats(
        chain_feature_dicts=[
            {
                "msa_all_seq": np.stack(
                    [
                        _encode("AC"),
                        _encode("A-"),
                        _encode("AA"),
                    ]
                ),
                "deletion_matrix_int_all_seq": np.array(
                    [
                        [0, 0],
                        [0, 2],
                        [1, 0],
                    ],
                    dtype=np.int32,
                ),
                "msa_species_identifiers_all_seq": np.array(
                    [b"", b"ECOLX", b"SHIDY"],
                    dtype=object,
                ),
                "msa": np.stack(
                    [
                        _encode("AC"),
                        _encode("AA"),
                    ]
                ),
                "deletion_matrix_int": np.array(
                    [
                        [0, 0],
                        [1, 0],
                    ],
                    dtype=np.int32,
                ),
            },
            {
                "msa_all_seq": np.stack(
                    [
                        _encode("GT"),
                        _encode("G-"),
                        _encode("GG"),
                    ]
                ),
                "deletion_matrix_int_all_seq": np.array(
                    [
                        [0, 0],
                        [0, 1],
                        [2, 0],
                    ],
                    dtype=np.int32,
                ),
                "msa_species_identifiers_all_seq": np.array(
                    [b"", b"ECOLX", b"SHIDY"],
                    dtype=object,
                ),
                "msa": np.stack(
                    [
                        _encode("GT"),
                        _encode("GG"),
                    ]
                ),
                "deletion_matrix_int": np.array(
                    [
                        [0, 0],
                        [2, 0],
                    ],
                    dtype=np.int32,
                ),
            },
        ],
        chain_sequences=["AC", "GT"],
    )

    assert result.translation_mode == "af3_species_pairing_from_af2_individual_msas"
    assert result.paired_row_count == 2
    assert result.per_chain_unpaired_row_counts == (1, 1)
    assert result.chain_stats[0].paired_species_identifier_count == 2
    assert result.chain_stats[1].paired_species_identifier_count == 2
    assert result.chain_stats[0].paired_rows_without_species_identifier_count == 0
    assert result.chain_stats[1].paired_rows_without_species_identifier_count == 0
    assert result.chain_stats[0].paired_rows_with_generated_accession_count == 2
    assert result.chain_stats[1].paired_rows_with_generated_accession_count == 2

    assert _a3m_payload_sequences(result.chain_msas[0].paired_msa) == ["Aaa-", "aAA"]
    assert _a3m_payload_sequences(result.chain_msas[1].paired_msa) == ["Ga-", "aaGG"]
    assert _a3m_payload_descriptions(result.chain_msas[0].paired_msa) == [
        "tr|APA0000001|APA0000001_ECOLX",
        "tr|APA0000002|APA0000002_SHIDY",
    ]
    assert _a3m_payload_descriptions(result.chain_msas[1].paired_msa) == [
        "tr|APB0000001|APB0000001_ECOLX",
        "tr|APB0000002|APB0000002_SHIDY",
    ]
    assert _a3m_payload_sequences(result.chain_msas[0].unpaired_msa) == ["aAA"]
    assert _a3m_payload_sequences(result.chain_msas[1].unpaired_msa) == ["aaGG"]


def test_translate_af2_individual_chain_features_tracks_missing_species_ids():
    result = translate_af2_individual_chain_features_to_af3_msas_with_stats(
        chain_feature_dicts=[
            {
                "msa_all_seq": np.stack(
                    [
                        _encode("AC"),
                        _encode("A-"),
                        _encode("AA"),
                    ]
                ),
                "deletion_matrix_int_all_seq": np.array(
                    [
                        [0, 0],
                        [0, 1],
                        [1, 0],
                    ],
                    dtype=np.int32,
                ),
                "msa_species_identifiers_all_seq": np.array(
                    [b"", b"", b"ECOLX"],
                    dtype=object,
                ),
                "msa": np.stack([_encode("AC")]),
                "deletion_matrix_int": np.array([[0, 0]], dtype=np.int32),
            },
            {
                "msa_all_seq": np.stack(
                    [
                        _encode("GT"),
                        _encode("G-"),
                    ]
                ),
                "deletion_matrix_int_all_seq": np.array(
                    [
                        [0, 0],
                        [0, 1],
                    ],
                    dtype=np.int32,
                ),
                "msa_species_identifiers_all_seq": np.array(
                    [b"", b""],
                    dtype=object,
                ),
                "msa": np.stack([_encode("GT")]),
                "deletion_matrix_int": np.array([[0, 0]], dtype=np.int32),
            },
        ],
        chain_sequences=["AC", "GT"],
    )

    assert result.chain_stats[0].paired_species_identifier_count == 1
    assert result.chain_stats[0].paired_rows_without_species_identifier_count == 1
    assert result.chain_stats[1].paired_species_identifier_count == 0
    assert result.chain_stats[1].paired_rows_without_species_identifier_count == 1
    assert _a3m_payload_descriptions(result.chain_msas[0].paired_msa) == [
        "sequence_1",
        "tr|APA0000002|APA0000002_ECOLX",
    ]
    assert _a3m_payload_descriptions(result.chain_msas[1].paired_msa) == ["sequence_1"]
