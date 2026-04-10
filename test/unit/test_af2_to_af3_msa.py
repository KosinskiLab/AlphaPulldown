import itertools
from types import SimpleNamespace

import numpy as np
from alphafold3.model import data_constants as af3_data_constants
from alphafold3.model import msa_pairing as af3_msa_pairing

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


def _make_af2_chain_feature_dict(
    sequence: str,
    *,
    paired_rows: list[tuple[str, str]],
    unpaired_rows: list[str] | None = None,
) -> dict[str, np.ndarray]:
    if unpaired_rows is None:
        unpaired_rows = []

    return {
        "msa_all_seq": np.stack(
            [_encode(sequence)] + [_encode(row) for _, row in paired_rows]
        ),
        "deletion_matrix_int_all_seq": np.zeros(
            (len(paired_rows) + 1, len(sequence)), dtype=np.int32
        ),
        "msa_species_identifiers_all_seq": np.asarray(
            [b""] + [species_id.encode("utf-8") for species_id, _ in paired_rows],
            dtype=object,
        ),
        "msa": np.stack([_encode(sequence)] + [_encode(row) for row in unpaired_rows]),
        "deletion_matrix_int": np.zeros(
            (len(unpaired_rows) + 1, len(sequence)), dtype=np.int32
        ),
    }


def _pair_translated_msas_with_af3(
    *,
    chain_ids: list[str],
    chain_sequences: list[str],
    chain_msas,
):
    paired_chains = []
    for chain_id, sequence, chain_msa in zip(
        chain_ids, chain_sequences, chain_msas, strict=True
    ):
        paired_sequences = [sequence]
        paired_descriptions = ["query"]
        if chain_msa.paired_msa:
            paired_sequences = [
                _aligned_and_deletions(a3m_sequence)[0]
                for a3m_sequence in _a3m_sequences(chain_msa.paired_msa)
            ]
            paired_descriptions = _a3m_descriptions(chain_msa.paired_msa)

        species_identifiers = []
        for description in paired_descriptions:
            if description.startswith("tr|") and "_" in description:
                species_identifiers.append(description.rsplit("_", maxsplit=1)[-1].encode())
            else:
                species_identifiers.append(b"")

        paired_chains.append(
            {
                "chain_id": chain_id,
                "msa_all_seq": np.stack(
                    [_encode(aligned_sequence) for aligned_sequence in paired_sequences]
                ),
                "deletion_matrix_all_seq": np.zeros(
                    (len(paired_sequences), len(sequence)), dtype=np.int32
                ),
                "msa_species_identifiers_all_seq": np.asarray(
                    species_identifiers, dtype=object
                ),
            }
        )

    return af3_msa_pairing.create_paired_features(
        chains=paired_chains,
        max_paired_sequences=512,
        nonempty_chain_ids=set(chain_ids),
        max_hits_per_species=600,
    )


def _non_gap_payload_rows(msa_rows: np.ndarray) -> int:
    return int(
        np.sum(
            np.any(msa_rows[1:] != af3_data_constants.MSA_GAP_IDX, axis=1),
        )
    )


def _all_gap_payload_rows(msa_rows: np.ndarray) -> int:
    return int(
        np.sum(
            np.all(msa_rows[1:] == af3_data_constants.MSA_GAP_IDX, axis=1),
        )
    )


def _canonical_af3_paired_rows(chains) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
    canonical_rows = []
    for chain in sorted(chains, key=lambda chain: chain["chain_id"]):
        canonical_rows.append(
            (
                chain["chain_id"],
                tuple(tuple(int(token) for token in row) for row in chain["msa_all_seq"]),
            )
        )
    return tuple(canonical_rows)


def _build_native_af3_chain_msas(
    *,
    chain_ids: list[str],
    chain_sequences: list[str],
    paired_rows_by_chain: dict[str, list[tuple[str, str]]],
    unpaired_rows_by_chain: dict[str, list[str]] | None = None,
):
    if unpaired_rows_by_chain is None:
        unpaired_rows_by_chain = {}

    chain_msas = []
    for chain_id, sequence in zip(chain_ids, chain_sequences, strict=True):
        paired_rows = paired_rows_by_chain.get(chain_id, [])
        paired_msa = ""
        if paired_rows:
            paired_lines = [">query", sequence]
            for index, (species_id, row) in enumerate(paired_rows, start=1):
                if species_id:
                    accession = f"AP{chain_id}{index:07d}"
                    description = f"tr|{accession}|{accession}_{species_id}"
                else:
                    description = f"sequence_{index}"
                paired_lines.extend([f">{description}", row])
            paired_msa = "\n".join(paired_lines) + "\n"

        unpaired_rows = unpaired_rows_by_chain.get(chain_id, [])
        unpaired_msa = ""
        if unpaired_rows:
            unpaired_lines = [">query", sequence]
            for index, row in enumerate(unpaired_rows, start=1):
                unpaired_lines.extend([f">sequence_{index}", row])
            unpaired_msa = "\n".join(unpaired_lines) + "\n"

        chain_msas.append(
            SimpleNamespace(
                paired_msa=paired_msa,
                unpaired_msa=unpaired_msa,
            )
        )

    return chain_msas


def _assert_translated_pairing_matches_native_af3_across_permutations(
    *,
    base_chain_ids: list[str],
    base_chain_sequences: dict[str, str],
    base_chain_features: dict[str, dict[str, np.ndarray]],
    native_paired_rows: dict[str, list[tuple[str, str]]],
    native_unpaired_rows: dict[str, list[str]] | None = None,
) -> None:
    canonical_rows = None
    for permutation in itertools.permutations(base_chain_ids):
        chain_ids = list(permutation)
        chain_sequences = [base_chain_sequences[chain_id] for chain_id in chain_ids]
        translated = translate_af2_individual_chain_features_to_af3_msas_with_stats(
            chain_feature_dicts=[base_chain_features[chain_id] for chain_id in chain_ids],
            chain_sequences=chain_sequences,
        )
        translated_rows = _canonical_af3_paired_rows(
            _pair_translated_msas_with_af3(
                chain_ids=chain_ids,
                chain_sequences=chain_sequences,
                chain_msas=translated.chain_msas,
            )
        )
        native_rows = _canonical_af3_paired_rows(
            _pair_translated_msas_with_af3(
                chain_ids=chain_ids,
                chain_sequences=chain_sequences,
                chain_msas=_build_native_af3_chain_msas(
                    chain_ids=chain_ids,
                    chain_sequences=chain_sequences,
                    paired_rows_by_chain=native_paired_rows,
                    unpaired_rows_by_chain=native_unpaired_rows,
                ),
            )
        )

        assert translated_rows == native_rows
        if canonical_rows is None:
            canonical_rows = translated_rows
        assert translated_rows == canonical_rows


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


def test_translate_af2_individual_chain_features_supports_three_chain_sparse_middle_pairing():
    chain_ids = ["A", "B", "C"]
    chain_sequences = ["AC", "GT", "MK"]
    result = translate_af2_individual_chain_features_to_af3_msas_with_stats(
        chain_feature_dicts=[
            _make_af2_chain_feature_dict(
                "AC",
                paired_rows=[("ECOLX", "A-"), ("ECOLX", "AA")],
                unpaired_rows=["AA"],
            ),
            _make_af2_chain_feature_dict(
                "GT",
                paired_rows=[],
                unpaired_rows=["G-"],
            ),
            _make_af2_chain_feature_dict(
                "MK",
                paired_rows=[("ECOLX", "M-"), ("ECOLX", "MM")],
                unpaired_rows=["MM"],
            ),
        ],
        chain_sequences=chain_sequences,
    )

    assert [stats.paired_msa_row_count for stats in result.chain_stats] == [2, 0, 2]
    assert result.chain_msas[1].paired_msa == ""

    paired_chains = _pair_translated_msas_with_af3(
        chain_ids=chain_ids,
        chain_sequences=chain_sequences,
        chain_msas=result.chain_msas,
    )

    assert [chain["msa_all_seq"].shape[0] for chain in paired_chains] == [3, 3, 3]
    assert _non_gap_payload_rows(paired_chains[0]["msa_all_seq"]) == 2
    assert _non_gap_payload_rows(paired_chains[1]["msa_all_seq"]) == 0
    assert _non_gap_payload_rows(paired_chains[2]["msa_all_seq"]) == 2
    assert _all_gap_payload_rows(paired_chains[1]["msa_all_seq"]) == 2


def test_translate_af2_individual_chain_features_supports_three_chain_min_count_crop():
    chain_ids = ["A", "B", "C"]
    chain_sequences = ["AC", "GT", "MK"]
    result = translate_af2_individual_chain_features_to_af3_msas_with_stats(
        chain_feature_dicts=[
            _make_af2_chain_feature_dict(
                "AC",
                paired_rows=[("ECOLX", "A-"), ("ECOLX", "AA"), ("ECOLX", "AC")],
            ),
            _make_af2_chain_feature_dict(
                "GT",
                paired_rows=[("ECOLX", "G-")],
            ),
            _make_af2_chain_feature_dict(
                "MK",
                paired_rows=[("ECOLX", "M-"), ("ECOLX", "MM"), ("ECOLX", "MK")],
            ),
        ],
        chain_sequences=chain_sequences,
    )

    assert [stats.paired_msa_row_count for stats in result.chain_stats] == [3, 1, 3]

    paired_chains = _pair_translated_msas_with_af3(
        chain_ids=chain_ids,
        chain_sequences=chain_sequences,
        chain_msas=result.chain_msas,
    )

    assert [chain["msa_all_seq"].shape[0] for chain in paired_chains] == [2, 2, 2]
    assert [_non_gap_payload_rows(chain["msa_all_seq"]) for chain in paired_chains] == [
        1,
        1,
        1,
    ]
    assert [_all_gap_payload_rows(chain["msa_all_seq"]) for chain in paired_chains] == [
        0,
        0,
        0,
    ]


def test_translate_af2_individual_chain_features_is_permutation_invariant_for_three_chains():
    base_chain_ids = ["A", "B", "C"]
    base_chain_sequences = {
        "A": "AC",
        "B": "GT",
        "C": "MK",
    }
    base_chain_features = {
        "A": _make_af2_chain_feature_dict(
            "AC",
            paired_rows=[("S1", "A-"), ("S1", "AA"), ("S2", "AC")],
        ),
        "B": _make_af2_chain_feature_dict(
            "GT",
            paired_rows=[("S1", "G-")],
        ),
        "C": _make_af2_chain_feature_dict(
            "MK",
            paired_rows=[("S1", "M-"), ("S1", "MM"), ("S2", "MK")],
        ),
    }

    canonical_rows = None
    for permutation in itertools.permutations(base_chain_ids):
        chain_sequences = [base_chain_sequences[chain_id] for chain_id in permutation]
        result = translate_af2_individual_chain_features_to_af3_msas_with_stats(
            chain_feature_dicts=[
                base_chain_features[chain_id] for chain_id in permutation
            ],
            chain_sequences=chain_sequences,
        )
        paired_chains = _pair_translated_msas_with_af3(
            chain_ids=list(permutation),
            chain_sequences=chain_sequences,
            chain_msas=result.chain_msas,
        )

        permutation_rows = _canonical_af3_paired_rows(paired_chains)
        if canonical_rows is None:
            canonical_rows = permutation_rows
        assert permutation_rows == canonical_rows


def test_translate_af2_individual_chain_features_matches_native_af3_pairing_for_sparse_trimer_permutations():
    base_chain_ids = ["A", "B", "C"]
    base_chain_sequences = {
        "A": "AC",
        "B": "GT",
        "C": "MK",
    }
    base_chain_features = {
        "A": _make_af2_chain_feature_dict(
            "AC",
            paired_rows=[("ECOLX", "A-"), ("ECOLX", "AA")],
            unpaired_rows=["AA"],
        ),
        "B": _make_af2_chain_feature_dict(
            "GT",
            paired_rows=[],
            unpaired_rows=["G-"],
        ),
        "C": _make_af2_chain_feature_dict(
            "MK",
            paired_rows=[("ECOLX", "M-"), ("ECOLX", "MM")],
            unpaired_rows=["MM"],
        ),
    }
    native_paired_rows = {
        "A": [("ECOLX", "A-"), ("ECOLX", "AA")],
        "B": [],
        "C": [("ECOLX", "M-"), ("ECOLX", "MM")],
    }
    native_unpaired_rows = {
        "A": ["AA"],
        "B": ["G-"],
        "C": ["MM"],
    }

    _assert_translated_pairing_matches_native_af3_across_permutations(
        base_chain_ids=base_chain_ids,
        base_chain_sequences=base_chain_sequences,
        base_chain_features=base_chain_features,
        native_paired_rows=native_paired_rows,
        native_unpaired_rows=native_unpaired_rows,
    )


def test_translate_af2_individual_chain_features_matches_native_af3_pairing_for_min_count_trimer_permutations():
    base_chain_ids = ["A", "B", "C"]
    base_chain_sequences = {
        "A": "AC",
        "B": "GT",
        "C": "MK",
    }
    base_chain_features = {
        "A": _make_af2_chain_feature_dict(
            "AC",
            paired_rows=[("S1", "A-"), ("S1", "AA"), ("S2", "AC")],
        ),
        "B": _make_af2_chain_feature_dict(
            "GT",
            paired_rows=[("S1", "G-")],
        ),
        "C": _make_af2_chain_feature_dict(
            "MK",
            paired_rows=[("S1", "M-"), ("S1", "MM"), ("S2", "MK")],
        ),
    }
    native_paired_rows = {
        "A": [("S1", "A-"), ("S1", "AA"), ("S2", "AC")],
        "B": [("S1", "G-")],
        "C": [("S1", "M-"), ("S1", "MM"), ("S2", "MK")],
    }

    _assert_translated_pairing_matches_native_af3_across_permutations(
        base_chain_ids=base_chain_ids,
        base_chain_sequences=base_chain_sequences,
        base_chain_features=base_chain_features,
        native_paired_rows=native_paired_rows,
    )
