"""
Translates AlphaPulldown objects and AlphaFold 3 JSON inputs into one AlphaFold 3 fold
input: chain construction, monomer normalisation, region slicing of JSON chains, template
mmCIF fixes, chain IDs and the job name.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""

import functools
import inspect
import logging
import os
import typing
from collections.abc import Sequence
from typing import Any, Dict, Union

import numpy as np
from alphafold3.common import folding_input
from alphafold3.data import parsers as af3_parsers
from alphafold3.structure import mmcif as af3_mmcif

from alphafold.common import residue_constants
from alphafold.common.protein import Protein, to_mmcif
from alphapulldown.folding_backend.alphafold3_debug import _TranslatedMsaDebugRecord
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject
from alphapulldown.utils.af2_to_af3_msa import (
    Af2ToAf3TranslationResult,
    msa_rows_and_deletions_to_a3m,
    translate_af2_individual_chain_features_to_af3_msas_with_stats,
    translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats,
)
from alphapulldown.utils.file_handling import read_maybe_xz, resolve_af3_input
from alphapulldown.utils.output_paths import (
    _collapse_repeated_name_fragments,
    _compact_existing_compound_name,
    _compact_output_job_name,
    _json_input_basename,
    _normalise_output_name_fragment,
    _regions_to_name_fragment,
)
from alphapulldown.utils.feature_metadata import (
    encode_metadata_in_description,
    find_feature_metadata,
)


def _object_name_fragment(obj: typing.Any) -> str:
    """Builds a deterministic output-name fragment for one modelling object."""
    if isinstance(obj, dict) and "json_input" in obj:
        fragment = _json_input_basename(obj["json_input"])
        regions = obj.get("regions")
        if isinstance(regions, Sequence) and regions:
            fragment = f"{fragment}__{_regions_to_name_fragment(regions)}"
        return _normalise_output_name_fragment(fragment)

    if isinstance(obj, MultimericObject):
        return _compact_existing_compound_name(obj.description or "multimer")

    if isinstance(obj, (MonomericObject, ChoppedObject)):
        return _normalise_output_name_fragment(obj.description or "monomer")

    if isinstance(obj, folding_input.Input):
        return _compact_existing_compound_name(obj.name)

    return _normalise_output_name_fragment(type(obj).__name__)


def _build_output_job_name(objects_to_model: Sequence[dict]) -> str:
    """Builds a readable AF3 job name from the requested modelling objects."""
    fragments: list[str] = []
    for entry in objects_to_model:
        object_to_model = entry["object"]
        if isinstance(object_to_model, list):
            fragments.extend(_object_name_fragment(obj) for obj in object_to_model)
        else:
            fragments.append(_object_name_fragment(object_to_model))
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return "ranked_0"
    readable_name = "_and_".join(_collapse_repeated_name_fragments(fragments))
    return _compact_output_job_name(readable_name)


def _supported_chain_kwargs(chain_cls: type) -> frozenset[str]:
    return frozenset(inspect.signature(chain_cls).parameters)


def _construct_chain(chain_cls: type, **kwargs):
    supported_kwargs = _supported_chain_kwargs(chain_cls)
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_kwargs
    }
    return chain_cls(**filtered_kwargs)


def _chain_input_sequence(
    chain: (
        folding_input.ProteinChain
        | folding_input.RnaChain
        | folding_input.DnaChain
    ),
) -> str:
    canonical_sequence = getattr(chain, "_sequence", None)
    if isinstance(canonical_sequence, str):
        return canonical_sequence
    return chain.sequence


def _clone_chain_with_id(chain, new_id: str):
    description = getattr(chain, "description", None)
    if isinstance(chain, folding_input.ProteinChain):
        return _construct_chain(
            folding_input.ProteinChain,
            id=new_id,
            sequence=_chain_input_sequence(chain),
            description=description,
            residue_ids=getattr(chain, "residue_ids", None),
            ptms=chain.ptms,
            paired_msa=chain.paired_msa,
            unpaired_msa=chain.unpaired_msa,
            templates=chain.templates,
        )
    if isinstance(chain, folding_input.RnaChain):
        return _construct_chain(
            folding_input.RnaChain,
            id=new_id,
            sequence=_chain_input_sequence(chain),
            description=description,
            residue_ids=getattr(chain, "residue_ids", None),
            modifications=chain.modifications,
            unpaired_msa=chain.unpaired_msa,
        )
    if isinstance(chain, folding_input.DnaChain):
        return _construct_chain(
            folding_input.DnaChain,
            id=new_id,
            sequence=_chain_input_sequence(chain),
            description=description,
            residue_ids=getattr(chain, "residue_ids", None),
            modifications=chain.modifications(),
        )
    if isinstance(chain, folding_input.Ligand):
        return _construct_chain(
            folding_input.Ligand,
            id=new_id,
            ccd_ids=chain.ccd_ids,
            smiles=chain.smiles,
            description=description,
        )
    raise TypeError(f"Unsupported chain type: {type(chain)}")


def _adjacent_duplicate_keep_indices(
    residue_ids: Sequence[int] | None,
) -> np.ndarray | None:
    if residue_ids is None:
        return None
    residue_ids_array = np.asarray(residue_ids, dtype=np.int32).reshape(-1)
    if residue_ids_array.size <= 1:
        return None
    keep_mask = np.ones(residue_ids_array.shape[0], dtype=bool)
    keep_mask[1:] = residue_ids_array[1:] != residue_ids_array[:-1]
    if np.all(keep_mask):
        return None
    return np.flatnonzero(keep_mask)


def _slice_sequence_like(
    value: str | bytes | bytearray,
    keep_indices: np.ndarray,
    *,
    tolerate_shorter_input: bool = False,
    context: str = "sequence",
) -> str | bytes:
    as_text = (
        value.decode("utf-8")
        if isinstance(value, (bytes, bytearray))
        else str(value)
    )
    if keep_indices.size:
        max_index = int(np.max(keep_indices))
        if max_index >= len(as_text):
            if tolerate_shorter_input:
                logging.warning(
                    "Leaving %s unsliced because length %d is shorter "
                    "than requested keep index %d",
                    context,
                    len(as_text),
                    max_index,
                )
                return value
            raise IndexError(
                f"{context} length {len(as_text)} is shorter than keep index {max_index}"
            )
    sliced = "".join(as_text[int(index)] for index in keep_indices.tolist())
    if isinstance(value, (bytes, bytearray)):
        return sliced.encode("utf-8")
    return sliced


def _normalise_monomeric_inputs_for_af3(
    mono_obj: Union[MonomericObject, ChoppedObject],
) -> tuple[str, Dict[str, Any], list[int] | None, bool]:
    sequence = mono_obj.sequence
    feature_dict = mono_obj.feature_dict
    residue_index = feature_dict.get("residue_index")
    if residue_index is None:
        return sequence, feature_dict, None, False

    residue_ids = (
        np.asarray(residue_index, dtype=np.int32).reshape(-1) + 1
    ).astype(int).tolist()
    keep_indices = _adjacent_duplicate_keep_indices(residue_ids)
    if keep_indices is None:
        return sequence, feature_dict, residue_ids, False

    if len(sequence) != len(residue_ids):
        logging.warning(
            "Skipping adjacent-duplicate AF3 normalization for %s because "
            "sequence length %d != residue ID count %d",
            mono_obj.description,
            len(sequence),
            len(residue_ids),
        )
        return sequence, feature_dict, residue_ids, False

    normalized_feature_dict = dict(feature_dict)
    normalized_sequence = _slice_sequence_like(sequence, keep_indices)
    if not isinstance(normalized_sequence, str):
        normalized_sequence = normalized_sequence.decode("utf-8")
    normalized_residue_ids = [residue_ids[int(index)] for index in keep_indices.tolist()]

    for key in ("aatype", "between_segment_residues", "residue_index"):
        if key in normalized_feature_dict:
            normalized_feature_dict[key] = np.asarray(
                normalized_feature_dict[key]
            )[keep_indices]

    for key in (
        "msa",
        "deletion_matrix_int",
        "deletion_matrix",
        "msa_all_seq",
        "deletion_matrix_int_all_seq",
        "deletion_matrix_all_seq",
    ):
        if key in normalized_feature_dict:
            normalized_feature_dict[key] = np.asarray(
                normalized_feature_dict[key]
            )[:, keep_indices]

    for key in (
        "template_aatype",
        "template_all_atom_masks",
        "template_confidence_scores",
    ):
        if key in normalized_feature_dict:
            normalized_feature_dict[key] = np.asarray(
                normalized_feature_dict[key]
            )[:, keep_indices]

    if "template_all_atom_positions" in normalized_feature_dict:
        normalized_feature_dict["template_all_atom_positions"] = np.asarray(
            normalized_feature_dict["template_all_atom_positions"]
        )[:, keep_indices, :, :]

    if "template_sequence" in normalized_feature_dict:
        normalized_feature_dict["template_sequence"] = np.array(
            [
                _slice_sequence_like(
                    template_sequence,
                    keep_indices,
                    tolerate_shorter_input=True,
                    context="template_sequence",
                )
                for template_sequence in normalized_feature_dict["template_sequence"]
            ],
            dtype=object,
        )

    if "sequence" in normalized_feature_dict:
        normalized_feature_dict["sequence"] = np.array(
            [normalized_sequence.encode("utf-8")]
        )
    if "seq_length" in normalized_feature_dict:
        normalized_feature_dict["seq_length"] = np.full(
            len(normalized_sequence),
            len(normalized_sequence),
            dtype=np.int32,
        )
    if "num_alignments" in normalized_feature_dict:
        num_alignments = int(
            np.asarray(normalized_feature_dict["num_alignments"]).reshape(-1)[0]
        )
        normalized_feature_dict["num_alignments"] = np.full(
            len(normalized_sequence),
            num_alignments,
            dtype=np.int32,
        )

    logging.info(
        "Collapsed %d adjacent duplicate residue position(s) for %s to "
        "satisfy AF3 input constraints",
        len(residue_ids) - len(normalized_residue_ids),
        mono_obj.description,
    )
    return (
        normalized_sequence,
        normalized_feature_dict,
        normalized_residue_ids,
        True,
    )


def _validate_json_template_mmcif(
    mmcif_string: str,
) -> tuple[bool, str | None]:
    try:
        af3_mmcif.from_string(mmcif_string)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _slice_a3m_row_to_region(a3m_row: str, start: int, end: int) -> str:
    query_position = 0
    sliced_chars: list[str] = []
    for char in a3m_row:
        if char.islower():
            # Lowercase A3M letters are insertions after the current
            # query position, so keep only those between retained
            # residues and drop insertions before `start` or after `end`.
            if start <= query_position < end:
                sliced_chars.append(char)
            continue

        query_position += 1
        if query_position < start:
            continue
        if query_position > end:
            break
        sliced_chars.append(char)
    return "".join(sliced_chars)


def _slice_a3m_to_regions(
    a3m_text: str | None,
    regions: Sequence[tuple[int, int]],
) -> str | None:
    if a3m_text in (None, "") or not regions:
        return a3m_text

    sequences, descriptions = af3_parsers.parse_fasta(a3m_text)
    sliced_sequences = [
        "".join(
            _slice_a3m_row_to_region(sequence, start, end)
            for start, end in regions
        )
        for sequence in sequences
    ]
    return (
        "\n".join(
            f">{description}\n{sequence}"
            for description, sequence in zip(
                descriptions,
                sliced_sequences,
                strict=True,
            )
        )
        + "\n"
    )


def _slice_positioned_modifications_to_regions(
    modifications: Sequence[tuple[str, int]],
    regions: Sequence[tuple[int, int]],
) -> list[tuple[str, int]]:
    sliced_modifications: list[tuple[str, int]] = []
    offset = 0
    for start, end in regions:
        sliced_modifications.extend(
            (
                modification_name,
                offset + modification_position - start + 1,
            )
            for modification_name, modification_position in modifications
            if start <= modification_position <= end
        )
        offset += end - start + 1
    return sliced_modifications


def _slice_templates_to_regions(
    templates: Sequence[folding_input.Template] | None,
    regions: Sequence[tuple[int, int]],
) -> Sequence[folding_input.Template] | None:
    if templates is None:
        return None

    sliced_templates = []
    for template in templates:
        remapped_indices = {}
        offset = 0
        for start, end in regions:
            start_index = start - 1
            remapped_indices.update({
                offset + (query_index - start_index): template_index
                for query_index, template_index in template.query_to_template_map.items()
                if start_index <= query_index < end
            })
            offset += end - start + 1
        if remapped_indices:
            sliced_templates.append(
                folding_input.Template(
                    mmcif=template.mmcif,
                    query_to_template_map=remapped_indices,
                )
            )
    return sliced_templates


def _chain_residue_ids(
    chain: (
        folding_input.ProteinChain
        | folding_input.RnaChain
        | folding_input.DnaChain
        | folding_input.Ligand
    ),
) -> list[int] | None:
    residue_ids = getattr(chain, "residue_ids", None)
    if residue_ids is not None:
        return [int(residue_id) for residue_id in residue_ids]
    if isinstance(
        chain,
        (
            folding_input.ProteinChain,
            folding_input.RnaChain,
            folding_input.DnaChain,
        ),
    ):
        return list(range(1, len(_chain_input_sequence(chain)) + 1))
    return None


def _slice_af3_chain_to_regions(
    chain: (
        folding_input.ProteinChain
        | folding_input.RnaChain
        | folding_input.DnaChain
        | folding_input.Ligand
    ),
    *,
    regions: Sequence[tuple[int, int]],
    json_path: str,
):
    if isinstance(chain, folding_input.Ligand):
        raise ValueError(
            f"Region ranges are not supported for ligand AF3 JSON inputs: {json_path}"
        )

    chain_sequence = _chain_input_sequence(chain)
    sequence_length = len(chain_sequence)
    for start, end in regions:
        if not 1 <= start <= end <= sequence_length:
            raise ValueError(
                f"Requested region {start}-{end} is outside the sequence "
                f"length {sequence_length} for AF3 JSON input {json_path}."
            )

    sliced_sequence = "".join(
        chain_sequence[start - 1:end]
        for start, end in regions
    )
    sliced_residue_ids = [
        residue_id
        for start, end in regions
        for residue_id in (_chain_residue_ids(chain) or [])[start - 1:end]
    ]
    if isinstance(chain, folding_input.ProteinChain):
        return _construct_chain(
            folding_input.ProteinChain,
            id=chain.id,
            sequence=sliced_sequence,
            description=getattr(chain, "description", None),
            ptms=_slice_positioned_modifications_to_regions(
                chain.ptms,
                regions,
            ),
            residue_ids=sliced_residue_ids,
            unpaired_msa=_slice_a3m_to_regions(
                chain.unpaired_msa,
                regions,
            ),
            paired_msa=_slice_a3m_to_regions(
                chain.paired_msa,
                regions,
            ),
            templates=_slice_templates_to_regions(
                chain.templates,
                regions,
            ),
        )

    if isinstance(chain, folding_input.RnaChain):
        return _construct_chain(
            folding_input.RnaChain,
            id=chain.id,
            sequence=sliced_sequence,
            description=getattr(chain, "description", None),
            modifications=_slice_positioned_modifications_to_regions(
                chain.modifications,
                regions,
            ),
            residue_ids=sliced_residue_ids,
            unpaired_msa=_slice_a3m_to_regions(
                chain.unpaired_msa,
                regions,
            ),
        )

    if isinstance(chain, folding_input.DnaChain):
        return _construct_chain(
            folding_input.DnaChain,
            id=chain.id,
            sequence=sliced_sequence,
            description=getattr(chain, "description", None),
            modifications=_slice_positioned_modifications_to_regions(
                chain.modifications(),
                regions,
            ),
            residue_ids=sliced_residue_ids,
        )

    raise TypeError(f"Unsupported chain type for AF3 JSON slicing: {type(chain)}")


def get_chain_id(index: int) -> str:
    if index < 26:
        return chr(ord('A') + index)
    else:
        first_char = chr(ord('A') + (index // 26) - 1)
        second_char = chr(ord('A') + (index % 26))
        return first_char + second_char


def insert_release_date_into_mmcif(
    mmcif_string: str, revision_date: str = '2100-01-01'
) -> str:
    lines = mmcif_string.splitlines()
    insert_index = None
    for i, line in enumerate(lines):
        if line.startswith('data_'):
            insert_index = i + 1
            break
    if insert_index is None:
        return mmcif_string
    revision_lines = [
        "",
        "loop_",
        "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.data_content_type", 
        "_pdbx_audit_revision_history.major_revision",
        "_pdbx_audit_revision_history.minor_revision",
        "_pdbx_audit_revision_history.revision_date",
        "1 'Structure model' 1 0 " + revision_date,
        ""
    ]
    lines[insert_index:insert_index] = revision_lines
    return '\n'.join(lines)


def strip_entity_poly_seq_block(mmcif_string: str) -> str:
    """Remove _entity_poly_seq loop to let AF3 reconstruct scheme from _atom_site.

    AF3's parser can infer _pdbx_poly_seq_scheme directly from _atom_site and
    avoids mismatches when our synthetic _entity_poly_seq has UNK entries.
    """
    lines = mmcif_string.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == 'loop_':
            # Peek ahead to see if this loop is for _entity_poly_seq
            j = i + 1
            is_entity_poly_seq = False
            while j < len(lines) and lines[j].strip().startswith('_'):
                if lines[j].strip().startswith('_entity_poly_seq.'):
                    is_entity_poly_seq = True
                # Stop header scan when we reach first data row (no leading underscore)
                j += 1
                if j < len(lines) and not lines[j].strip().startswith('_'):
                    break
            if is_entity_poly_seq:
                # Skip until next blank line or next loop_/category header
                k = j
                while k < len(lines):
                    if lines[k].strip() == '' or lines[k].strip() == '#':
                        k += 1
                        break
                    if lines[k].strip() == 'loop_':
                        break
                    k += 1
                i = k
                continue
        out_lines.append(line)
        i += 1
    return '\n'.join(out_lines)


def _expand_monomeric_object(
    mono_obj: Union[MonomericObject, ChoppedObject],
) -> list[Union[MonomericObject, ChoppedObject]]:
    return [mono_obj]


class FoldInputBuilder:
    """Translates one preparation's AlphaPulldown objects and AF3 JSON inputs into AF3 chains.

    It holds what translating one object needs from the others: the chains so far, the
    chain IDs in use, which chains carry MSAs translated from AF2 features, the debug
    records, and caches. ``prepare_input`` makes a fresh builder per call, so none of
    them outlives the preparation.
    """

    def __init__(
        self,
        *,
        features_directory: str | None = None,
        debug_templates: bool | None = None,
        debug_msas: bool = False,
    ) -> None:
        self.features_directory = features_directory
        self.debug_templates = debug_templates
        self.debug_msas = debug_msas
        self.chains: list[
            folding_input.ProteinChain
            | folding_input.RnaChain
            | folding_input.DnaChain
            | folding_input.Ligand
        ] = []
        self.used_chain_ids: set[str] = set()
        self._chain_id_counter = 0
        # Track chains whose MSAs were translated from AF2 features; they must
        # not be rewritten by the promotion heuristic in build().
        self.af2_translated_msa_chain_ids: set[str] = set()
        self.translation_debug_chain_records: list[_TranslatedMsaDebugRecord] = []
        self.translation_debug_results: list[Af2ToAf3TranslationResult] = []
        # Per-preparation caches, as lru_cache wrappers on this instance.
        self._description_with_af2_feature_metadata = functools.lru_cache(maxsize=None)(
            self._description_with_af2_feature_metadata
        )
        self._validate_json_template_mmcif = functools.lru_cache(maxsize=None)(
            _validate_json_template_mmcif
        )

    def _next_chain_id(self) -> str:
        """Get the next available chain ID that's not already used."""
        while True:
            chain_id = get_chain_id(self._chain_id_counter)
            self._chain_id_counter += 1
            if chain_id not in self.used_chain_ids:
                return chain_id

    def _description_with_af2_feature_metadata(
        self,
        output_description: str,
        source_description: str,
    ) -> str:
        try:
            metadata = find_feature_metadata(
                source_description, self.features_directory
            )
        except (OSError, ValueError) as exc:
            logging.warning(
                "Could not read feature metadata for %s: %s",
                source_description,
                exc,
            )
            return output_description
        if metadata is None:
            return output_description
        return encode_metadata_in_description(output_description, metadata)

    def _sanitize_json_input_chain_templates(
        self,
        chain: (
            folding_input.ProteinChain
            | folding_input.RnaChain
            | folding_input.DnaChain
            | folding_input.Ligand
        ),
        *,
        json_path: str,
    ):
        if not isinstance(chain, folding_input.ProteinChain):
            return chain
        if chain.templates is None:
            return chain

        valid_templates = []
        dropped_template_count = 0
        for template_index, template in enumerate(chain.templates):
            is_valid, error_message = self._validate_json_template_mmcif(
                template.mmcif
            )
            if is_valid:
                valid_templates.append(template)
                continue

            dropped_template_count += 1
            logging.warning(
                "Skipping invalid template %d for JSON input %s chain %s: %s",
                template_index,
                json_path,
                chain.id,
                error_message,
            )

        if dropped_template_count == 0:
            return chain

        logging.info(
            "Kept %d/%d template(s) for JSON input %s chain %s after validation",
            len(valid_templates),
            len(chain.templates),
            json_path,
            chain.id,
        )
        return _construct_chain(
            folding_input.ProteinChain,
            id=chain.id,
            sequence=_chain_input_sequence(chain),
            description=getattr(chain, "description", None),
            residue_ids=getattr(chain, "residue_ids", None),
            ptms=chain.ptms,
            paired_msa=chain.paired_msa,
            unpaired_msa=chain.unpaired_msa,
            templates=valid_templates,
        )

    def _expand_json_input_chains(
        self,
        *,
        chains: Sequence[
            folding_input.ProteinChain
            | folding_input.RnaChain
            | folding_input.DnaChain
            | folding_input.Ligand
        ],
        json_path: str,
        regions: list[tuple[int, int]] | None,
    ) -> list[
        folding_input.ProteinChain
        | folding_input.RnaChain
        | folding_input.DnaChain
        | folding_input.Ligand
        ]:
        sanitized_chains = [
            self._sanitize_json_input_chain_templates(chain, json_path=json_path)
            for chain in chains
        ]
        if not regions:
            return list(sanitized_chains)

        if len(sanitized_chains) != 1:
            raise ValueError(
                "Region ranges for AF3 JSON feature inputs require exactly "
                f"one chain per file, but {json_path} contains {len(sanitized_chains)} chains."
            )

        merged_chain = _slice_af3_chain_to_regions(
            sanitized_chains[0],
            regions=regions,
            json_path=json_path,
        )
        logging.info(
            "Collapsed AF3 JSON input %s into one gapped chain for regions %s",
            json_path,
            regions,
        )
        return [merged_chain]

    def _monomeric_to_chain(
        self,
        mono_obj: Union[MonomericObject, ChoppedObject],
        chain_id: str
    ) -> folding_input.ProteinChain:
        sequence, feature_dict, residue_ids, _ = _normalise_monomeric_inputs_for_af3(
            mono_obj
        )
        msa_array = feature_dict.get('msa')
        deletion_matrix = feature_dict.get('deletion_matrix_int')
        if deletion_matrix is None:
            deletion_matrix = feature_dict.get('deletion_matrix')
        # Standalone AlphaPulldown monomer objects carry only a single custom MSA.
        # MultimericObject instances override this with an AF2->AF3 translation.
        unpaired_msa = (
            msa_rows_and_deletions_to_a3m(
                msa_rows=np.asarray(msa_array),
                deletion_rows=None if deletion_matrix is None else np.asarray(deletion_matrix),
                query_sequence=sequence,
            )
            if msa_array is not None
            else ""
        )
        paired_msa = ""
        templates = []
        if 'template_aatype' in feature_dict:
            num_templates = feature_dict['template_aatype'].shape[0]
            for i in range(num_templates):
                try:
                    pdb_code_chain = feature_dict["template_domain_names"][i].decode('utf-8')
                    if '_' in pdb_code_chain:
                        pdb_code, chain_id_template = pdb_code_chain.split('_')
                    else:
                        pdb_code = pdb_code_chain
                        chain_id_template = 'A'
                    template_sequence = feature_dict["template_sequence"][i]
                    if isinstance(template_sequence, bytes):
                        template_sequence = template_sequence.decode('utf-8')
                    template_mask = feature_dict["template_all_atom_masks"][i]
                    hh_ids = np.argmax(feature_dict["template_aatype"][i], axis=-1)
                    if np.sum(template_mask) == 0:
                        logging.info(f"Skipping template {i} ({pdb_code_chain}) - no atoms in region")
                        continue
                    tmpl_aatype = np.array([
                        residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE[j]
                        for j in hh_ids
                    ], dtype=np.int32)
                    tmpl_aatype[tmpl_aatype == 21] = 20
                    if np.any(tmpl_aatype < 0) or np.any(tmpl_aatype >= len(residue_constants.resnames)):
                        raise ValueError(f"Invalid aatype: {tmpl_aatype}")
                    protein = Protein(
                        atom_positions=feature_dict["template_all_atom_positions"][i],
                        atom_mask=feature_dict["template_all_atom_masks"][i],
                        aatype=tmpl_aatype,
                        residue_index=feature_dict["residue_index"],
                        chain_index=np.zeros_like(feature_dict["residue_index"], dtype=int),
                        b_factors=np.zeros(template_mask.shape, dtype=float),
                    )
                    mmcif_string = to_mmcif(
                        prot=protein, file_id=pdb_code, model_type='Monomer'
                    )
                    new_mmcif_string = insert_release_date_into_mmcif(mmcif_string)
                    # Remove _entity_poly_seq to avoid mismatches during AF3 parsing
                    new_mmcif_string = strip_entity_poly_seq_block(new_mmcif_string)
                    # Optional debug dump of generated template mmCIF
                    if self.debug_templates:
                        try:
                            os.makedirs("templates_debug", exist_ok=True)
                            debug_fname = os.path.join(
                                "templates_debug",
                                f"generated_template_{pdb_code}_{chain_id_template}_chain{chain_id}_idx{i}.cif",
                            )
                            with open(debug_fname, "w") as f_dbg:
                                f_dbg.write(new_mmcif_string)
                        except Exception:
                            # Best-effort debug; ignore failures
                            pass
                    # Build query->template mapping only for residues that have any atoms
                    # present in the template (avoid gaps causing index errors downstream).
                    try:
                        per_res_atom_mask = feature_dict["template_all_atom_masks"][i]
                        # Boolean mask: True if any atom present for this residue
                        present_res_mask = np.any(per_res_atom_mask > 0, axis=-1)
                        # Map compact template indices (0..N_present-1) to original residue indices
                        residue_indices_present = [idx for idx, present in enumerate(present_res_mask) if present]
                        compact_index_by_residue = {res_idx: compact_idx for compact_idx, res_idx in enumerate(residue_indices_present)}
                        # Query indices are positions in the template sequence aligned to the query
                        query_to_template_map = {res_idx: compact_index_by_residue[res_idx] for res_idx in residue_indices_present}
                    except Exception:
                        # Fallback to identity if masks missing, though this may error later
                        query_to_template_map = {j: j for j in range(len(template_sequence))}
                    templates.append(
                        folding_input.Template(
                            mmcif=new_mmcif_string,
                            query_to_template_map=query_to_template_map,
                        )
                    )
                except Exception as e:
                    logging.error(f"Error processing template {i} for chain {chain_id}: {e}")
                    try:
                        os.makedirs("templates_debug", exist_ok=True)
                        error_filename = os.path.join("templates_debug", f"ERROR_template_{i}_chain_{chain_id}.txt")
                        with open(error_filename, "w") as error_file:
                            error_file.write(f"Error: {e}\n")
                            error_file.write(f"Template index: {i}\n")
                            error_file.write(f"Chain ID: {chain_id}\n")
                            if 'template_domain_names' in feature_dict:
                                error_file.write(f"Domain name: {feature_dict['template_domain_names'][i]}\n")
                            if 'template_sequence' in feature_dict:
                                error_file.write(f"Template sequence: {feature_dict['template_sequence'][i]}\n")
                    except Exception as save_error:
                        logging.error(f"Failed to save error info: {save_error}")
                    raise
        source_description = getattr(
            mono_obj, "monomeric_description", mono_obj.description
        )
        chain_description = self._description_with_af2_feature_metadata(
            mono_obj.description,
            source_description,
        )
        chain = _construct_chain(
            folding_input.ProteinChain,
            id=chain_id,
            sequence=sequence,
            ptms=[],
            residue_ids=residue_ids,
            description=chain_description,
            unpaired_msa=unpaired_msa,
            paired_msa=paired_msa,
            templates=templates,
        )
        return chain

    def add(self, obj) -> None:
        """Translate one AlphaPulldown object or AF3 JSON input into chains, with chain IDs
        unique across this preparation."""
        if isinstance(obj, dict) and 'json_input' in obj:
            json_path = obj['json_input']
            json_regions = obj.get('regions')
            logging.info(f"Processing JSON file: {json_path}")
            try:
                # Accept a compressed feature file: the published database
                # ships AF3 inputs as .json.xz, mirroring AF2's .pkl.xz.
                resolved = resolve_af3_input(json_path)
                if resolved is None:
                    raise FileNotFoundError(json_path)
                if resolved != json_path:
                    logging.info(f"Using compressed feature file: {resolved}")
                json_str = read_maybe_xz(resolved)
                input_obj = folding_input.Input.from_json(json_str)
                chains_to_add = self._expand_json_input_chains(
                    chains=input_obj.chains,
                    json_path=json_path,
                    regions=(
                        json_regions if isinstance(json_regions, list) and json_regions else None
                    ),
                )
                # Track the chain IDs from the JSON file
                logging.info(
                    "JSON file %s contributes chains with IDs: %s",
                    json_path,
                    [chain.id for chain in chains_to_add],
                )

                # Check for duplicate chain IDs and modify them if necessary
                modified_chains = []
                for chain in chains_to_add:
                    original_id = chain.id
                    new_id = original_id

                    # If this chain ID is already used, generate a new one
                    if original_id in self.used_chain_ids:
                        new_id = self._next_chain_id()
                        logging.info(f"Chain ID '{original_id}' already used, changing to '{new_id}'")

                    self.used_chain_ids.add(new_id)
                    logging.info(f"Added chain ID '{new_id}' from JSON file {json_path}")

                    modified_chains.append(_clone_chain_with_id(chain, new_id))

                self.chains.extend(modified_chains)
            except Exception as e:
                logging.error(f"Failed to parse JSON file {json_path}: {e}")
                raise
        elif isinstance(obj, (MonomericObject, ChoppedObject)):
            for expanded_obj in _expand_monomeric_object(obj):
                chain_id = self._next_chain_id()
                self.used_chain_ids.add(chain_id)
                logging.info(
                    f"Added chain ID '{chain_id}' for AlphaPulldown object "
                    f"{expanded_obj.description}"
                )
                self.chains.append(self._monomeric_to_chain(expanded_obj, chain_id))
        elif isinstance(obj, MultimericObject):
            chains = []
            translated_result = None
            combined_msa = obj.feature_dict.get('msa')
            expanded_interactors = list(obj.interactors)
            normalized_interactor_inputs = [
                _normalise_monomeric_inputs_for_af3(interactor)
                for interactor in expanded_interactors
            ]
            has_adjacent_duplicate_residue_ids = any(
                normalized_input[3] for normalized_input in normalized_interactor_inputs
            )

            translated_result = (
                translate_af2_individual_chain_features_to_af3_msas_with_stats(
                    chain_feature_dicts=[
                        normalized_input[1]
                        for normalized_input in normalized_interactor_inputs
                    ],
                    chain_sequences=[
                        normalized_input[0]
                        for normalized_input in normalized_interactor_inputs
                    ],
                )
            )
            num_pairable_chains = sum(
                chain_stats.paired_species_identifier_count > 0
                for chain_stats in translated_result.chain_stats
            )
            if (
                has_adjacent_duplicate_residue_ids
                and num_pairable_chains < 2
                and combined_msa is not None
            ):
                logging.info(
                    "Skipping AF2 merged-MSA fallback for multimeric object %s "
                    "because adjacent duplicate residue IDs were collapsed for AF3 input compatibility.",
                    obj.description,
                )
            elif num_pairable_chains < 2 and combined_msa is not None:
                # Fall back to the merged AF2 multimer MSA transport path when the
                # individual `_all_seq` features do not carry usable species IDs.
                translated_result = (
                    translate_af2_complex_msa_to_af3_unpaired_chain_msas_with_stats(
                        merged_msa=np.asarray(combined_msa),
                        chain_sequences=[
                            interactor.sequence for interactor in obj.interactors
                        ],
                        num_alignments=obj.feature_dict.get('num_alignments'),
                        deletion_matrix=(
                            obj.feature_dict.get('deletion_matrix_int')
                            if obj.feature_dict.get('deletion_matrix_int') is not None
                            else obj.feature_dict.get('deletion_matrix')
                        ),
                        asym_id=obj.feature_dict.get('asym_id'),
                    )
                )
                expanded_interactors = list(obj.interactors)

            for chain_index, interactor in enumerate(expanded_interactors):
                chain_id = self._next_chain_id()
                self.used_chain_ids.add(chain_id)
                logging.info(f"Added chain ID '{chain_id}' for multimeric interactor")
                base_chain = self._monomeric_to_chain(interactor, chain_id)
                if translated_result is not None:
                    chain_msas = translated_result.chain_msas[chain_index]
                    chain_stats = translated_result.chain_stats[chain_index]
                    base_chain = _construct_chain(
                        folding_input.ProteinChain,
                        id=base_chain.id,
                        sequence=_chain_input_sequence(base_chain),
                        ptms=base_chain.ptms,
                        residue_ids=getattr(base_chain, "residue_ids", None),
                        description=getattr(base_chain, "description", None),
                        unpaired_msa=chain_msas.unpaired_msa,
                        paired_msa=chain_msas.paired_msa,
                        templates=base_chain.templates,
                    )
                    self.af2_translated_msa_chain_ids.add(base_chain.id)
                    if self.debug_msas:
                        self.translation_debug_chain_records.append(
                            _TranslatedMsaDebugRecord(
                                chain_id=base_chain.id,
                                chain_description=interactor.description,
                                chain_length=len(interactor.sequence),
                                paired_msa=chain_msas.paired_msa,
                                unpaired_msa=chain_msas.unpaired_msa,
                                paired_msa_row_count=chain_stats.paired_msa_row_count,
                                unpaired_msa_row_count=chain_stats.unpaired_msa_row_count,
                                paired_species_identifier_count=(
                                    chain_stats.paired_species_identifier_count
                                ),
                                paired_rows_without_species_identifier_count=(
                                    chain_stats.paired_rows_without_species_identifier_count
                                ),
                                paired_rows_with_generated_accession_count=(
                                    chain_stats.paired_rows_with_generated_accession_count
                                ),
                            )
                        )
                chains.append(base_chain)
            if self.debug_msas and translated_result is not None:
                self.translation_debug_results.append(translated_result)
            self.chains.extend(chains)
        else:
            raise TypeError(f"Unsupported object type for folding input conversion: {type(obj)}")

    def build(self, *, job_name: str, random_seed: int) -> folding_input.Input:
        """The fold input of all chains added, with unpaired MSAs promoted to paired."""
        # Promote unpaired->paired only for chains not constructed from AF2 multimer features.
        promoted_chains: list[folding_input.ProteinChain | folding_input.RnaChain | folding_input.DnaChain | folding_input.Ligand] = []
        for ch in self.chains:
            if (
                isinstance(ch, folding_input.ProteinChain)
                and ch.id not in self.af2_translated_msa_chain_ids
            ):
                try:
                    has_empty_paired = (getattr(ch, 'paired_msa', None) in (None, ''))
                    has_unpaired = bool(getattr(ch, 'unpaired_msa', ''))
                    if has_empty_paired and has_unpaired:
                        new_chain = _construct_chain(
                            folding_input.ProteinChain,
                            id=ch.id,
                            sequence=_chain_input_sequence(ch),
                            ptms=ch.ptms,
                            residue_ids=getattr(ch, "residue_ids", None),
                            description=getattr(ch, "description", None),
                            paired_msa=ch.unpaired_msa,
                            unpaired_msa='',
                            templates=ch.templates,
                        )
                        promoted_chains.append(new_chain)
                        continue
                except Exception:
                    pass
            promoted_chains.append(ch)

        return folding_input.Input(
            name=job_name,
            rng_seeds=[random_seed],
            chains=promoted_chains,
        )
