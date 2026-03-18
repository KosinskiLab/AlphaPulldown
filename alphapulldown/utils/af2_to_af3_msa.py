"""Utilities for translating AF2 multimer MSAs into AF3 chain inputs."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np


AF2_ID_TO_A3M = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
    "-",
)
AF2_GAP_ID = len(AF2_ID_TO_A3M) - 1


@dataclasses.dataclass(frozen=True, slots=True)
class Af3ChainMsas:
    """AF3-ready custom MSA strings for one chain."""

    paired_msa: str
    unpaired_msa: str


@dataclasses.dataclass(frozen=True, slots=True)
class Af2ToAf3ChainTranslationStats:
    """Per-chain stats for AF2->AF3 MSA translation."""

    paired_msa_row_count: int
    unpaired_msa_row_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class Af2ToAf3TranslationResult:
    """Translated AF3 chain MSAs plus row-classification diagnostics."""

    chain_msas: tuple[Af3ChainMsas, ...]
    chain_stats: tuple[Af2ToAf3ChainTranslationStats, ...]
    translation_mode: str
    total_rows_considered: int
    occupancy_histogram: dict[str, int]
    paired_row_count: int
    per_chain_unpaired_row_counts: tuple[int, ...]
    invalid_paired_rows: int
    invalid_unpaired_rows: int


@dataclasses.dataclass(frozen=True, slots=True)
class Af2MsaRows:
    """AF2 aligned rows plus deletion counts for A3M reconstruction."""

    msa: np.ndarray
    deletion_matrix: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedAf2ComplexMsa:
    """Normalised AF2 complex MSA inputs split into per-chain column blocks."""

    merged_msa: np.ndarray
    deletion_matrix: np.ndarray
    chain_blocks: tuple[np.ndarray, ...]
    deletion_blocks: tuple[np.ndarray, ...]


def _normalise_num_alignments(
    num_alignments: int | np.ndarray | None, max_rows: int
) -> int:
    if num_alignments is None:
        return max_rows
    value = int(np.asarray(num_alignments).reshape(-1)[0])
    return max(0, min(value, max_rows))


def _chain_column_slices_from_lengths(
    chain_lengths: Sequence[int],
) -> list[slice]:
    column_slices: list[slice] = []
    start = 0
    for chain_length in chain_lengths:
        stop = start + int(chain_length)
        column_slices.append(slice(start, stop))
        start = stop
    return column_slices


def _chain_column_slices_from_asym_id(asym_id: np.ndarray) -> list[slice]:
    asym_id = np.asarray(asym_id).reshape(-1)
    if asym_id.size == 0:
        return []

    column_slices: list[slice] = []
    start = 0
    for index in range(1, asym_id.size + 1):
        if index == asym_id.size or asym_id[index] != asym_id[start]:
            column_slices.append(slice(start, index))
            start = index
    return column_slices


def resolve_chain_column_slices(
    total_columns: int,
    chain_lengths: Sequence[int],
    asym_id: np.ndarray | None = None,
) -> list[slice]:
    """Resolves per-chain column boundaries for a merged AF2 multimer MSA."""
    expected_columns = sum(int(length) for length in chain_lengths)
    if expected_columns != total_columns:
        raise ValueError(
            f"Chain lengths sum to {expected_columns}, but MSA has {total_columns} columns."
        )

    fallback_slices = _chain_column_slices_from_lengths(chain_lengths)
    if asym_id is None:
        return fallback_slices

    asym_slices = _chain_column_slices_from_asym_id(asym_id)
    asym_lengths = [chain_slice.stop - chain_slice.start for chain_slice in asym_slices]
    if len(asym_slices) == len(chain_lengths) and asym_lengths == list(chain_lengths):
        return asym_slices
    return fallback_slices


def msa_rows_to_a3m(msa_rows: np.ndarray, query_sequence: str) -> str:
    """Converts AF2 integer MSA rows to an AF3-compatible A3M string."""
    deletion_rows = np.zeros_like(msa_rows, dtype=np.int32)
    return msa_rows_and_deletions_to_a3m(
        msa_rows=msa_rows,
        deletion_rows=deletion_rows,
        query_sequence=query_sequence,
    )


def normalise_deletion_rows(
    deletion_rows: np.ndarray | None, msa_shape: tuple[int, int]
) -> np.ndarray:
    """Converts AF2 deletion rows to an integer matrix aligned with the MSA rows."""
    if deletion_rows is None:
        return np.zeros(msa_shape, dtype=np.int32)

    deletion_rows = np.asarray(deletion_rows)
    if deletion_rows.shape != msa_shape:
        raise ValueError(
            f"Deletion matrix shape {deletion_rows.shape} does not match MSA shape {msa_shape}."
        )

    deletion_rows = np.rint(deletion_rows).astype(np.int32, copy=False)
    if np.any(deletion_rows < 0):
        raise ValueError("Deletion matrix must not contain negative values.")
    return deletion_rows


def aligned_row_and_deletions_to_a3m(
    msa_row: np.ndarray, deletion_row: np.ndarray
) -> str:
    """Builds one A3M row from AF2 aligned tokens plus deletion counts."""
    if msa_row.shape != deletion_row.shape:
        raise ValueError(
            f"MSA row shape {msa_row.shape} does not match deletion row shape {deletion_row.shape}."
        )

    row_chunks: list[str] = []
    for token, deletion_count in zip(msa_row, deletion_row, strict=True):
        row_chunks.append("a" * int(deletion_count))
        row_chunks.append(AF2_ID_TO_A3M[int(token)])
    return "".join(row_chunks)


def msa_rows_and_deletions_to_a3m(
    msa_rows: np.ndarray,
    deletion_rows: np.ndarray | None,
    query_sequence: str,
) -> str:
    """Converts AF2 aligned rows plus deletion counts to an AF3-compatible A3M string."""
    if msa_rows.shape[0] == 0:
        return ""

    deletion_rows = normalise_deletion_rows(deletion_rows, msa_rows.shape)
    msa_lines = [">query", query_sequence]
    for index, (row, deletions) in enumerate(
        zip(msa_rows, deletion_rows, strict=True)
    ):
        sequence = aligned_row_and_deletions_to_a3m(row, deletions)
        msa_lines.extend((f">sequence_{index}", sequence))
    return "\n".join(msa_lines)


def _drop_leading_query_row(rows: Af2MsaRows, query_sequence: str) -> Af2MsaRows:
    if rows.msa.shape[0] == 0:
        return rows

    first_row = "".join(AF2_ID_TO_A3M[int(token)] for token in rows.msa[0])
    if first_row == query_sequence:
        return Af2MsaRows(
            msa=rows.msa[1:],
            deletion_matrix=rows.deletion_matrix[1:],
        )
    return rows


def _stack_rows(
    rows: list[np.ndarray], num_columns: int, dtype: np.dtype
) -> np.ndarray:
    if not rows:
        return np.empty((0, num_columns), dtype=dtype)
    return np.stack(rows).astype(dtype, copy=False)


def _row_occupancy_histogram(
    chain_blocks: Sequence[np.ndarray],
) -> tuple[dict[str, int], int]:
    occupancy_histogram = {"0": 0, "1": 0, "ge_2": 0}
    paired_row_count = 0

    if not chain_blocks:
        return occupancy_histogram, paired_row_count

    num_rows = chain_blocks[0].shape[0]
    for row_index in range(num_rows):
        occupancy = sum(
            not np.all(chain_block[row_index] == AF2_GAP_ID)
            for chain_block in chain_blocks
        )
        if occupancy >= 2:
            occupancy_histogram["ge_2"] += 1
            paired_row_count += 1
        else:
            occupancy_histogram[str(occupancy)] += 1

    return occupancy_histogram, paired_row_count


def _prepare_complex_msa(
    merged_msa: np.ndarray,
    chain_sequences: Sequence[str],
    *,
    num_alignments: int | np.ndarray | None = None,
    deletion_matrix: np.ndarray | None = None,
    asym_id: np.ndarray | None = None,
) -> PreparedAf2ComplexMsa:
    merged_msa = np.asarray(merged_msa)
    if merged_msa.ndim != 2:
        raise ValueError(f"Merged MSA must be rank-2, got shape {merged_msa.shape}.")

    chain_lengths = [len(sequence) for sequence in chain_sequences]
    real_num_alignments = _normalise_num_alignments(num_alignments, merged_msa.shape[0])
    trimmed_msa = merged_msa[:real_num_alignments]
    trimmed_deletion_matrix = normalise_deletion_rows(
        None if deletion_matrix is None else np.asarray(deletion_matrix)[:real_num_alignments],
        trimmed_msa.shape,
    )
    chain_slices = resolve_chain_column_slices(
        total_columns=trimmed_msa.shape[1],
        chain_lengths=chain_lengths,
        asym_id=asym_id,
    )

    return PreparedAf2ComplexMsa(
        merged_msa=trimmed_msa,
        deletion_matrix=trimmed_deletion_matrix,
        chain_blocks=tuple(trimmed_msa[:, chain_slice] for chain_slice in chain_slices),
        deletion_blocks=tuple(
            trimmed_deletion_matrix[:, chain_slice] for chain_slice in chain_slices
        ),
    )


def _trim_chain_rows(
    *,
    sequence: str,
    msa_rows: np.ndarray,
    deletion_rows: np.ndarray,
) -> Af2MsaRows:
    return _drop_leading_query_row(
        Af2MsaRows(msa=msa_rows, deletion_matrix=deletion_rows),
        sequence,
    )


def _chain_rows_to_a3m(
    *,
    sequence: str,
    msa_rows: np.ndarray,
    deletion_rows: np.ndarray,
) -> str:
    trimmed_rows = _trim_chain_rows(
        sequence=sequence,
        msa_rows=msa_rows,
        deletion_rows=deletion_rows,
    )
    return msa_rows_and_deletions_to_a3m(
        msa_rows=trimmed_rows.msa,
        deletion_rows=trimmed_rows.deletion_matrix,
        query_sequence=sequence,
    )


def translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats(
    merged_msa: np.ndarray,
    chain_sequences: Sequence[str],
    *,
    num_alignments: int | np.ndarray | None = None,
    deletion_matrix: np.ndarray | None = None,
    asym_id: np.ndarray | None = None,
) -> Af2ToAf3TranslationResult:
    """Converts an AF2 multimer complex MSA into manual AF3 per-chain MSAs.

    The AF2 multimer feature dict already contains the post-pairing, post-dedup,
    post-crop merged complex MSA in final row order. AF3's `pairedMsa` path relies
    on header metadata to do its own pairing, but that metadata is not retained in
    AlphaFold2 pickles. The faithful transport format is therefore the exact AF2
    merged complex MSA carried through AF3's manual `unpairedMsa` input path.
    """
    prepared_msa = _prepare_complex_msa(
        merged_msa=merged_msa,
        chain_sequences=chain_sequences,
        num_alignments=num_alignments,
        deletion_matrix=deletion_matrix,
        asym_id=asym_id,
    )
    occupancy_histogram, paired_row_count = _row_occupancy_histogram(
        prepared_msa.chain_blocks
    )

    translated_msas: list[Af3ChainMsas] = []
    translated_chain_stats: list[Af2ToAf3ChainTranslationStats] = []
    for sequence, chain_block, deletion_block in zip(
        chain_sequences,
        prepared_msa.chain_blocks,
        prepared_msa.deletion_blocks,
        strict=True,
    ):
        trimmed_rows = _trim_chain_rows(
            sequence=sequence,
            msa_rows=chain_block,
            deletion_rows=deletion_block,
        )
        unpaired_msa = _chain_rows_to_a3m(
            sequence=sequence,
            msa_rows=chain_block,
            deletion_rows=deletion_block,
        )
        translated_msas.append(
            Af3ChainMsas(
                paired_msa="",
                unpaired_msa=unpaired_msa,
            )
        )
        translated_chain_stats.append(
            Af2ToAf3ChainTranslationStats(
                paired_msa_row_count=0,
                unpaired_msa_row_count=int(trimmed_rows.msa.shape[0]),
            )
        )

    return Af2ToAf3TranslationResult(
        chain_msas=tuple(translated_msas),
        chain_stats=tuple(translated_chain_stats),
        translation_mode="manual_unpaired_from_af2_multimer",
        total_rows_considered=int(prepared_msa.merged_msa.shape[0]),
        occupancy_histogram=occupancy_histogram,
        paired_row_count=paired_row_count,
        per_chain_unpaired_row_counts=tuple(
            chain_stats.unpaired_msa_row_count for chain_stats in translated_chain_stats
        ),
        invalid_paired_rows=0,
        invalid_unpaired_rows=0,
    )


def translate_af2_complex_msa_to_af3_chain_msas(
    merged_msa: np.ndarray,
    chain_sequences: Sequence[str],
    *,
    num_alignments: int | np.ndarray | None = None,
    deletion_matrix: np.ndarray | None = None,
    asym_id: np.ndarray | None = None,
) -> list[Af3ChainMsas]:
    """Splits an AF2 multimer complex MSA into AF3 paired and unpaired chain MSAs."""
    return list(
        translate_af2_complex_msa_to_af3_chain_msas_with_stats(
            merged_msa=merged_msa,
            chain_sequences=chain_sequences,
            num_alignments=num_alignments,
            deletion_matrix=deletion_matrix,
            asym_id=asym_id,
        ).chain_msas
    )


def translate_af2_complex_msa_to_af3_chain_msas_with_stats(
    merged_msa: np.ndarray,
    chain_sequences: Sequence[str],
    *,
    num_alignments: int | np.ndarray | None = None,
    deletion_matrix: np.ndarray | None = None,
    asym_id: np.ndarray | None = None,
) -> Af2ToAf3TranslationResult:
    """Splits an AF2 multimer complex MSA and returns translation diagnostics."""
    prepared_msa = _prepare_complex_msa(
        merged_msa=merged_msa,
        chain_sequences=chain_sequences,
        num_alignments=num_alignments,
        deletion_matrix=deletion_matrix,
        asym_id=asym_id,
    )

    paired_rows_by_chain = [[] for _ in chain_sequences]
    paired_deletions_by_chain = [[] for _ in chain_sequences]
    unpaired_rows_by_chain = [[] for _ in chain_sequences]
    unpaired_deletions_by_chain = [[] for _ in chain_sequences]
    occupancy_histogram = {"0": 0, "1": 0, "ge_2": 0}
    paired_row_count = 0

    for row_index in range(prepared_msa.merged_msa.shape[0]):
        occupied_chain_indices = [
            chain_index
            for chain_index, chain_block in enumerate(prepared_msa.chain_blocks)
            if not np.all(chain_block[row_index] == AF2_GAP_ID)
        ]
        occupancy = len(occupied_chain_indices)
        if occupancy >= 2:
            occupancy_histogram["ge_2"] += 1
        else:
            occupancy_histogram[str(occupancy)] += 1

        if occupancy >= 2:
            paired_row_count += 1
            for chain_index, (chain_block, deletion_block) in enumerate(
                zip(prepared_msa.chain_blocks, prepared_msa.deletion_blocks, strict=True)
            ):
                paired_rows_by_chain[chain_index].append(chain_block[row_index])
                paired_deletions_by_chain[chain_index].append(deletion_block[row_index])
        elif occupancy == 1:
            chain_index = occupied_chain_indices[0]
            unpaired_rows_by_chain[chain_index].append(
                prepared_msa.chain_blocks[chain_index][row_index]
            )
            unpaired_deletions_by_chain[chain_index].append(
                prepared_msa.deletion_blocks[chain_index][row_index]
            )

    translated_msas: list[Af3ChainMsas] = []
    translated_chain_stats: list[Af2ToAf3ChainTranslationStats] = []
    for (
        sequence,
        paired_rows,
        paired_deletions,
        unpaired_rows,
        unpaired_deletions,
    ) in zip(
        chain_sequences,
        paired_rows_by_chain,
        paired_deletions_by_chain,
        unpaired_rows_by_chain,
        unpaired_deletions_by_chain,
        strict=True,
    ):
        paired_rows_array = _stack_rows(
            paired_rows, len(sequence), prepared_msa.merged_msa.dtype
        )
        paired_deletions_array = _stack_rows(
            paired_deletions, len(sequence), prepared_msa.deletion_matrix.dtype
        )
        unpaired_rows_array = _stack_rows(
            unpaired_rows, len(sequence), prepared_msa.merged_msa.dtype
        )
        unpaired_deletions_array = _stack_rows(
            unpaired_deletions, len(sequence), prepared_msa.deletion_matrix.dtype
        )
        paired_rows_data = _trim_chain_rows(
            sequence=sequence,
            msa_rows=paired_rows_array,
            deletion_rows=paired_deletions_array,
        )
        unpaired_rows_data = _trim_chain_rows(
            sequence=sequence,
            msa_rows=unpaired_rows_array,
            deletion_rows=unpaired_deletions_array,
        )
        translated_msas.append(
            Af3ChainMsas(
                paired_msa=_chain_rows_to_a3m(
                    sequence=sequence,
                    msa_rows=paired_rows_array,
                    deletion_rows=paired_deletions_array,
                ),
                unpaired_msa=_chain_rows_to_a3m(
                    sequence=sequence,
                    msa_rows=unpaired_rows_array,
                    deletion_rows=unpaired_deletions_array,
                ),
            )
        )
        translated_chain_stats.append(
            Af2ToAf3ChainTranslationStats(
                paired_msa_row_count=int(paired_rows_data.msa.shape[0]),
                unpaired_msa_row_count=int(unpaired_rows_data.msa.shape[0]),
            )
        )

    return Af2ToAf3TranslationResult(
        chain_msas=tuple(translated_msas),
        chain_stats=tuple(translated_chain_stats),
        translation_mode="split_paired_and_unpaired",
        total_rows_considered=int(prepared_msa.merged_msa.shape[0]),
        occupancy_histogram=occupancy_histogram,
        paired_row_count=paired_row_count,
        per_chain_unpaired_row_counts=tuple(
            chain_stats.unpaired_msa_row_count for chain_stats in translated_chain_stats
        ),
        invalid_paired_rows=0,
        invalid_unpaired_rows=0,
    )
