"""AlphaFold 2 features from a local MMseqs2 MSA bundle.

The search stage (``feature_batch.MsaBatch``) is backend-neutral: it writes one MSA
bundle per chain. This module turns a bundle into what the AlphaFold 2 backend
reads -- a pickled ``alphapulldown.objects.MonomericObject`` -- building it the way
``alphafold.data.pipeline.DataPipeline.process`` builds features from jackhmmer,
so the result drops into existing AlphaFold 2 inference unchanged.

What is reproduced from native AlphaFold 2:

- per-database caps, counting the query row as AlphaFold 2 does: uniref90 at
  10 000 and mgnify at 501, small BFD uncapped (``pipeline.py``: jackhmmer's
  ``max_sto_sequences``, which stops at that many sequence names, query first)
- merge order uniref90, BFD, MGnify (``pipeline.py:265``). Row order matters:
  AlphaFold 2 deduplicates in that order and samples its MSA from the top
- templates searched from uniref90 ALONE, never from the merged alignment
- the paired UniProt alignment truncated to 50 000 rows, as
  ``MonomericObject.all_seq_msa_features`` does, and exposed as ``*_all_seq``

Where it cannot be exact, and why that is acceptable:

- The database recipe is uniref90, MGnify, small BFD and UniProt -- AlphaFold 2's
  ``reduced_dbs`` set. There is no BFD/UniRef30 HHblits arm, so these features are
  comparable to ``--db_preset=reduced_dbs``, not ``full_dbs``.
- The bundle deduplicated rows across databases before this module sees them, so
  the caps apply to each database's rows after that deduplication. Native
  AlphaFold 2 caps raw hits first. The mgnify cap can therefore admit a few more
  unique rows than native, and a sequence both MGnify and small BFD found counts
  against MGnify's cap. Both effects touch only rows duplicated across databases.
- The template profile is built from the uniref90 A3M with insertions removed.
  jackhmmer's Stockholm carries insert columns; A3M insertions are per row and not
  aligned to each other, so they cannot be turned back into columns faithfully.
  Match columns -- which decide the profile's states -- are identical.

And one deliberate improvement: UniProt accession identifiers are filled from the
alignment headers, where native features leave them empty. Nothing here ever
queries UniProt over the network; see :func:`_no_network`.
"""

from __future__ import annotations

import copy
import dataclasses
from datetime import date
import json
import lzma
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping, Sequence

from alphapulldown.feature_batch import (
    PROTEIN,
    UNPAIRED_DATABASE_NAMES,
    FeatureArtifact,
    FeatureBatchResult,
    FeatureFailure,
    FeatureRequest,
    SearchedMsas,
    _fasta_records,
    _validate_feature_requests,
    read_msa_bundle,
    searched_msas_from_payload,
)


# AlphaFold 2's own caps, counting the query row the way it does.
AF2_MAX_SEQUENCES = {"uniref90": 10_000, "mgnify": 501}
# pipeline.py:265 -- make_msa_features((uniref90_msa, bfd_msa, mgnify_msa)).
AF2_MERGE_ORDER = ("uniref90", "small_bfd", "mgnify")
# MonomericObject.all_seq_msa_features truncates the UniProt alignment here.
PAIRED_MAX_SEQUENCES = 50_000
TEMPLATE_SEARCHERS = ("hmmsearch", "hhsearch")
# Bump when the construction below changes, so features an older construction
# built are regenerated rather than reused.
_AF2_FEATURE_SCHEMA = 1


@dataclasses.dataclass(frozen=True, slots=True)
class Af2MsaInputs:
    """The three alignments AlphaFold 2 features are built from."""

    # Becomes msa / deletion_matrix_int: all three databases, capped, in AF2 order.
    main_a3m: str
    # uniref90 alone: what the template profile is built from.
    template_a3m: str
    # UniProt: becomes *_all_seq, which pairs chains by species.
    paired_a3m: str


def _a3m(records: Sequence[tuple[str, str]]) -> str:
    return "".join(f">{description}\n{sequence}\n" for description, sequence in records)


def af2_msa_inputs(msas: SearchedMsas) -> Af2MsaInputs:
    """Cut a bundle's alignments into the shapes native AlphaFold 2 would have."""
    query, by_database = msas.rows_by_database()
    missing = [name for name in UNPAIRED_DATABASE_NAMES if name not in by_database]
    if missing:
        raise ValueError(
            "MSA bundle does not say which rows came from "
            + ", ".join(missing)
            + "; AlphaFold 2 needs each database's rows separately"
        )

    def capped(name: str) -> list[tuple[str, str]]:
        rows = by_database[name]
        cap = AF2_MAX_SEQUENCES.get(name)
        # AlphaFold 2 counts the query toward its cap, so a cap of N keeps N - 1 hits.
        return rows if cap is None else rows[: cap - 1]

    blocks = {name: capped(name) for name in AF2_MERGE_ORDER}
    paired_records = _fasta_records(msas.paired)
    if not paired_records:
        raise ValueError(
            "MSA bundle has no paired UniProt alignment; AlphaFold 2 multimer "
            "pairing needs one"
        )
    return Af2MsaInputs(
        main_a3m=_a3m(
            [query, *(row for name in AF2_MERGE_ORDER for row in blocks[name])]
        ),
        template_a3m=_a3m([query, *blocks["uniref90"]]),
        paired_a3m=_a3m(paired_records[:PAIRED_MAX_SEQUENCES]),
    )


def _no_network(accessions: Sequence[str]) -> dict[str, str]:
    """Resolve nothing, rather than ask UniProt.

    The shared identifier helper falls back to UniProt REST lookups for
    accessions whose header carries no species. A local batch of thousands of
    chains must not start doing that unannounced -- and it has no need to: the
    UniProt database's canonical ``sp|ACC|NAME_SPECIES`` headers carry the species
    already, and AlphaFold 2 reads it from them directly. Anything unparseable
    stays unresolved, exactly as it would in native features.
    """
    del accessions
    return {}


def _offline_accessions(a3m: str, *, rows: int):
    """Accession per alignment row, deduplicated exactly as make_msa_features is."""
    from alphapulldown.utils.mmseqs_species_identifiers import (
        build_mmseq_identifier_features,
    )

    identifiers = build_mmseq_identifier_features(
        a3m, species_resolver=_no_network, expected_rows=rows
    )
    accessions = identifiers["msa_uniprot_accession_identifiers"]
    if len(accessions) != rows:
        raise ValueError(
            f"accession identifiers cover {len(accessions)} rows but the MSA has "
            f"{rows}; they would index the wrong sequences"
        )
    return accessions


@dataclasses.dataclass(frozen=True, slots=True)
class Af2TemplateStackSettings:
    """Everything the AlphaFold 2 template searcher and featurizer are built from.

    Explicit values rather than global flags, so a caller states what it uses
    instead of depending on some earlier call having rewritten FLAGS.
    """

    template_mmcif_dir: str
    max_template_date: str
    kalign_binary_path: str
    obsolete_pdbs_path: str | None = None
    use_hhsearch: bool = False
    # hmmsearch against PDB seqres, the default.
    hmmsearch_binary_path: str | None = None
    hmmbuild_binary_path: str | None = None
    pdb_seqres_database_path: str | None = None
    # hhsearch against PDB70, under --use_hhsearch.
    hhsearch_binary_path: str | None = None
    pdb70_database_path: str | None = None

    @property
    def searcher_name(self) -> str:
        return "hhsearch" if self.use_hhsearch else "hmmsearch"


def build_af2_template_stack(settings: Af2TemplateStackSettings):
    """The AlphaFold 2 template searcher and featurizer, as native features use."""
    from alphafold.data import templates
    from alphafold.data.tools import hhsearch, hmmsearch

    if settings.use_hhsearch:
        searcher = hhsearch.HHSearch(
            binary_path=settings.hhsearch_binary_path,
            databases=[settings.pdb70_database_path],
        )
        featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=settings.template_mmcif_dir,
            max_template_date=settings.max_template_date,
            max_hits=20,
            kalign_binary_path=settings.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=settings.obsolete_pdbs_path,
        )
        return searcher, featurizer
    featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=settings.template_mmcif_dir,
        max_template_date=settings.max_template_date,
        max_hits=20,
        kalign_binary_path=settings.kalign_binary_path,
        obsolete_pdbs_path=settings.obsolete_pdbs_path,
        release_dates_path=None,
    )
    searcher = hmmsearch.Hmmsearch(
        binary_path=settings.hmmsearch_binary_path,
        hmmbuild_binary_path=settings.hmmbuild_binary_path,
        database_path=settings.pdb_seqres_database_path,
    )
    return searcher, featurizer


@dataclasses.dataclass(frozen=True, slots=True)
class Af2FeatureFinalizationSettings:
    """CPU AlphaFold 2 finalization settings and complete template provenance."""

    output_dir: Path
    msa_input_dir: Path
    max_template_date: str
    template_seqres_database_id: str
    template_mmcif_database_id: str
    # hmmsearch against PDB seqres, or hhsearch against PDB70 (--use_hhsearch).
    # The two find different templates, so which one ran is part of the identity.
    template_searcher: str = "hmmsearch"
    compress: bool = False
    base_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class Af2FeatureFinalizer:
    """Turn persisted MSA bundles into AlphaFold 2 feature pickles on CPU."""

    def __init__(
        self,
        *,
        settings: Af2FeatureFinalizationSettings,
        template_searcher: Any,
        template_featurizer: Any,
    ):
        if not isinstance(settings, Af2FeatureFinalizationSettings):
            raise TypeError(
                "Af2FeatureFinalizer settings must be Af2FeatureFinalizationSettings"
            )
        self._settings = settings
        self._template_searcher = template_searcher
        self._template_featurizer = template_featurizer

    def generate(self, requests: Sequence[FeatureRequest]) -> FeatureBatchResult:
        requests = tuple(requests)
        self._validate(requests)
        self._settings.output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        reused = []
        failures = []
        for request in requests:
            try:
                if request.molecule_type != PROTEIN:
                    raise ValueError(
                        f"{request.name!r} is {request.molecule_type}: AlphaFold 2 "
                        "has no MSA features for anything but protein chains"
                    )
                payload = read_msa_bundle(self._settings.msa_input_dir, request)
                msas = searched_msas_from_payload(payload)
                provenance = self._provenance(payload)
                cached = self._read_matching_artifact(request, provenance)
                if cached is not None:
                    reused.append(FeatureArtifact(name=request.name, path=cached))
                    continue
                feature_dict = self._feature_dict(request, af2_msa_inputs(msas))
                path = self._publish(request, feature_dict, provenance)
                written.append(FeatureArtifact(name=request.name, path=path))
            except Exception as exc:
                failures.append(FeatureFailure(name=request.name, error=str(exc)))
        return FeatureBatchResult(
            written=tuple(written), reused=tuple(reused), failures=tuple(failures)
        )

    def _validate(self, requests: Sequence[FeatureRequest]) -> None:
        _validate_feature_requests(requests)
        for field_name in (
            "max_template_date",
            "template_seqres_database_id",
            "template_mmcif_database_id",
        ):
            if not str(getattr(self._settings, field_name)).strip():
                raise ValueError(f"{field_name} requires a non-empty value")
        if self._settings.template_searcher not in TEMPLATE_SEARCHERS:
            raise ValueError(
                "template_searcher must be one of "
                f"{', '.join(TEMPLATE_SEARCHERS)}, not "
                f"{self._settings.template_searcher!r}"
            )

    def _feature_dict(
        self, request: FeatureRequest, inputs: Af2MsaInputs
    ) -> dict[str, Any]:
        from alphafold.data import msa_pairing, parsers, pipeline

        from alphapulldown.objects import add_template_feature_defaults
        from alphapulldown.utils.template_reuse import (
            search_templates,
            stockholm_from_a3m,
        )

        sequence = request.sequence
        features = dict(
            pipeline.make_sequence_features(
                sequence=sequence, description=request.name, num_res=len(sequence)
            )
        )

        main = pipeline.make_msa_features([parsers.parse_a3m(inputs.main_a3m)])
        main["msa_uniprot_accession_identifiers"] = _offline_accessions(
            inputs.main_a3m, rows=main["msa"].shape[0]
        )
        features.update(main)

        features.update(
            search_templates(
                self._template_searcher,
                self._template_featurizer,
                query_sequence=sequence,
                stockholm_msa=stockholm_from_a3m(inputs.template_a3m),
            )
        )

        # The paired alignment is searched separately against UniProt, whose
        # headers carry the species AlphaFold 2 pairs chains by. It is not a copy
        # of the unpaired features, which is what the remote path makes do with.
        paired = pipeline.make_msa_features([parsers.parse_a3m(inputs.paired_a3m)])
        paired["msa_uniprot_accession_identifiers"] = _offline_accessions(
            inputs.paired_a3m, rows=paired["msa"].shape[0]
        )
        pairable = msa_pairing.MSA_FEATURES + (
            "msa_species_identifiers",
            "msa_uniprot_accession_identifiers",
        )
        features.update(
            {f"{key}_all_seq": value for key, value in paired.items() if key in pairable}
        )

        add_template_feature_defaults(features, sequence)
        return features

    def _provenance(self, msa_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Everything that decides the pickle's content, for cache reuse."""
        return {
            "af2_feature_schema": _AF2_FEATURE_SCHEMA,
            "mmseqs2": msa_payload["provenance"],
            "af2_templates": self._template_signature(),
        }

    def _template_signature(self) -> dict[str, Any]:
        # Software versions, not base_metadata's wall-clock date, which would miss
        # on every run -- the same choice the AlphaFold 3 finalizer makes.
        return {
            "max_template_date": self._settings.max_template_date,
            "pdb_seqres_database_id": self._settings.template_seqres_database_id,
            "mmcif_database_id": self._settings.template_mmcif_database_id,
            "template_searcher": self._settings.template_searcher,
            "software": dict(self._settings.base_metadata.get("software", {})),
        }

    def _read_matching_artifact(
        self, request: FeatureRequest, provenance: Mapping[str, Any]
    ) -> Path | None:
        from alphapulldown.utils.lightweight_pickles import load_lightweight_pickle

        path = self._artifact_path(request.name)
        if not path.exists():
            return None
        try:
            monomer = load_lightweight_pickle(path)
        except Exception:
            return None
        if getattr(monomer, "sequence", None) != request.sequence:
            return None
        if getattr(monomer, "local_msa_provenance", None) != provenance:
            return None
        return path

    def _publish(
        self,
        request: FeatureRequest,
        feature_dict: dict[str, Any],
        provenance: Mapping[str, Any],
    ) -> Path:
        from alphapulldown.objects import MonomericObject

        monomer = MonomericObject(request.name, request.sequence)
        monomer.feature_dict = feature_dict
        monomer.skip_msa = False
        # Carried on the object so a later run can tell, from the pickle alone,
        # whether it was built from the same search and template settings.
        monomer.local_msa_provenance = dict(provenance)

        metadata = copy.deepcopy(dict(self._settings.base_metadata))
        other = metadata.setdefault("other", {})
        # The same keys the AlphaFold 3 finalizer writes, so one reader serves both.
        # The search mode, cpu or gpu, is inside the provenance.
        other["msa_backend"] = "mmseqs2-gpu"
        other["mmseqs2_gpu"] = provenance["mmseqs2"]
        other["af2_templates"] = provenance["af2_templates"]
        metadata_suffix = ".json.xz" if self._settings.compress else ".json"
        _write_atomic(
            self._settings.output_dir
            / f"{request.name}_feature_metadata_{date.today()}{metadata_suffix}",
            json.dumps(metadata).encode("utf-8"),
        )
        # The pickle is the output a workflow waits for, so it is published last:
        # its existence implies its metadata exists too.
        path = self._artifact_path(request.name)
        _write_atomic(path, pickle.dumps(monomer))
        return path

    def _artifact_path(self, name: str) -> Path:
        suffix = ".pkl.xz" if self._settings.compress else ".pkl"
        return self._settings.output_dir / f"{name}{suffix}"


def _write_atomic(path: Path, content: bytes) -> None:
    """Publish a file whole or not at all, compressed when its name says .xz."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw_handle:
            if path.suffix == ".xz":
                with lzma.open(raw_handle, "wb") as handle:
                    handle.write(content)
            else:
                raw_handle.write(content)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
