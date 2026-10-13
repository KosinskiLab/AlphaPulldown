"""
Debug artifacts of AlphaFold 3 input preparation and inference: the MSAs translated from
AF2 features with their summary, and each seed's final complex MSA.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

import dataclasses
import json
import logging
import os
from collections import Counter

import numpy as np
from alphafold3.data import parsers as af3_parsers

from alphapulldown.utils.af2_to_af3_msa import Af2ToAf3TranslationResult
from alphapulldown.utils.msa_encoding import ids_to_a3m_af3


@dataclasses.dataclass(frozen=True, slots=True)
class _TranslatedMsaDebugRecord:
    chain_id: str
    chain_description: str
    chain_length: int
    paired_msa: str
    unpaired_msa: str
    paired_msa_row_count: int
    unpaired_msa_row_count: int
    paired_species_identifier_count: int
    paired_rows_without_species_identifier_count: int
    paired_rows_with_generated_accession_count: int


def _write_translated_msa_debug_artifacts(
    *,
    job_name: str,
    output_dir: str,
    chain_records: list[_TranslatedMsaDebugRecord],
    translation_results: list[Af2ToAf3TranslationResult],
) -> None:
    if not chain_records or not translation_results:
        return

    def _extract_paired_species_identifier(description: str) -> str:
        if not description.startswith(("tr|", "sp|")):
            return ""
        description_tail = description.split("|")[-1]
        if "_" not in description_tail:
            return ""
        return description_tail.rsplit("_", maxsplit=1)[-1].strip()

    def _summarise_effective_af3_pairing(
        records: list[_TranslatedMsaDebugRecord],
    ) -> dict[str, object]:
        per_chain_species_counts: dict[str, Counter[str]] = {}
        effective_non_gap_rows_by_chain: dict[str, int] = {}
        effective_gap_rows_by_chain: dict[str, int] = {}

        for record in records:
            _, descriptions = af3_parsers.parse_fasta(record.paired_msa)
            species_counts: Counter[str] = Counter()
            for description in descriptions[1:]:
                species_identifier = _extract_paired_species_identifier(description)
                if species_identifier:
                    species_counts[species_identifier] += 1
            per_chain_species_counts[record.chain_id] = species_counts
            effective_non_gap_rows_by_chain[record.chain_id] = 0
            effective_gap_rows_by_chain[record.chain_id] = 0

        effective_paired_row_count = 0
        effective_paired_row_histogram_by_num_chains: Counter[str] = Counter()
        all_species = sorted(
            {
                species_identifier
                for species_counts in per_chain_species_counts.values()
                for species_identifier in species_counts
            }
        )
        for species_identifier in all_species:
            present_chain_ids = [
                chain_id
                for chain_id, species_counts in per_chain_species_counts.items()
                if species_counts.get(species_identifier, 0) > 0
            ]
            if len(present_chain_ids) <= 1:
                continue

            kept_rows = min(
                per_chain_species_counts[chain_id][species_identifier]
                for chain_id in present_chain_ids
            )
            effective_paired_row_count += kept_rows
            effective_paired_row_histogram_by_num_chains[
                str(len(present_chain_ids))
            ] += kept_rows
            for chain_id in per_chain_species_counts:
                if chain_id in present_chain_ids:
                    effective_non_gap_rows_by_chain[chain_id] += kept_rows
                else:
                    effective_gap_rows_by_chain[chain_id] += kept_rows

        return {
            "effective_paired_row_count": int(effective_paired_row_count),
            "effective_paired_row_histogram_by_num_chains": {
                key: int(value)
                for key, value in sorted(
                    effective_paired_row_histogram_by_num_chains.items(),
                    key=lambda item: int(item[0]),
                )
            },
            "effective_non_gap_rows_by_chain": {
                key: int(value)
                for key, value in sorted(effective_non_gap_rows_by_chain.items())
            },
            "effective_gap_rows_by_chain": {
                key: int(value)
                for key, value in sorted(effective_gap_rows_by_chain.items())
            },
        }

    os.makedirs(output_dir, exist_ok=True)
    effective_pairing_summary = _summarise_effective_af3_pairing(chain_records)
    summary = {
        "job_name": job_name,
        "translation_modes": sorted(
            {result.translation_mode for result in translation_results}
        ),
        "total_rows_considered": int(
            sum(result.total_rows_considered for result in translation_results)
        ),
        "occupancy_histogram": {
            "0": int(
                sum(result.occupancy_histogram.get("0", 0) for result in translation_results)
            ),
            "1": int(
                sum(result.occupancy_histogram.get("1", 0) for result in translation_results)
            ),
            "ge_2": int(
                sum(result.occupancy_histogram.get("ge_2", 0) for result in translation_results)
            ),
        },
        "paired_row_count": int(
            effective_pairing_summary["effective_paired_row_count"]
        ),
        "translated_paired_input_row_count": int(
            sum(result.paired_row_count for result in translation_results)
        ),
        "effective_paired_row_count": int(
            effective_pairing_summary["effective_paired_row_count"]
        ),
        "effective_paired_row_histogram_by_num_chains": dict(
            effective_pairing_summary["effective_paired_row_histogram_by_num_chains"]
        ),
        "invalid_paired_rows": int(
            sum(result.invalid_paired_rows for result in translation_results)
        ),
        "invalid_unpaired_rows": int(
            sum(result.invalid_unpaired_rows for result in translation_results)
        ),
        "paired_rows_valid": all(
            result.invalid_paired_rows == 0 for result in translation_results
        ),
        "unpaired_rows_valid": all(
            result.invalid_unpaired_rows == 0 for result in translation_results
        ),
        "chains": [],
    }

    for record in chain_records:
        chain_id = record.chain_id
        paired_path = os.path.join(
            output_dir, f"{job_name}_chain-{chain_id}_paired_input.a3m"
        )
        unpaired_path = os.path.join(
            output_dir, f"{job_name}_chain-{chain_id}_unpaired_input.a3m"
        )
        with open(paired_path, "wt") as handle:
            handle.write(record.paired_msa)
        with open(unpaired_path, "wt") as handle:
            handle.write(record.unpaired_msa)

        summary["chains"].append(
            {
                "chain_id": chain_id,
                "chain_description": record.chain_description,
                "chain_length": int(record.chain_length),
                "paired_msa_row_count": int(record.paired_msa_row_count),
                "unpaired_msa_row_count": int(record.unpaired_msa_row_count),
                "paired_species_identifier_count": int(
                    record.paired_species_identifier_count
                ),
                "paired_rows_without_species_identifier_count": int(
                    record.paired_rows_without_species_identifier_count
                ),
                "paired_rows_with_generated_accession_count": int(
                    record.paired_rows_with_generated_accession_count
                ),
                "effective_paired_msa_row_count": int(
                    effective_pairing_summary["effective_non_gap_rows_by_chain"].get(
                        chain_id, 0
                    )
                ),
                "effective_paired_gap_row_count": int(
                    effective_pairing_summary["effective_gap_rows_by_chain"].get(
                        chain_id, 0
                    )
                ),
            }
        )

    summary_path = os.path.join(
        output_dir, f"{job_name}_af2_to_af3_translation_summary.json"
    )
    with open(summary_path, "wt") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


# Utility: write final complex MSA (after pairing/dedup/merge) to A3M
def _write_final_msa_a3m(
    example_batch,
    seed_value: int,
    *,
    output_dir: os.PathLike[str] | str | None,
    job_name: str,
) -> None:
    try:
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)

        def write_from_array(rows: np.ndarray, suffix: str):
            # AF3 uses a different integer alphabet for MSA arrays.
            a3m_text = ids_to_a3m_af3(rows)
            a3m_path = os.path.join(output_dir, f"{job_name}_seed-{seed_value}_{suffix}.a3m")
            with open(a3m_path, 'wt') as f:
                f.write(a3m_text)
            logging.info(f"Wrote {suffix} A3M to {a3m_path}")

        # Write merged msa
        msa_rows = example_batch.get('msa', None)
        if msa_rows is not None:
            num_alignments = int(example_batch.get('num_alignments', 0)) if 'num_alignments' in example_batch else msa_rows.shape[0]
            if num_alignments > 0:
                write_from_array(np.asarray(msa_rows)[:num_alignments], 'final_complex_msa')

    except Exception as e:
        logging.error(f"Failed to write final complex MSA A3M: {e}")
