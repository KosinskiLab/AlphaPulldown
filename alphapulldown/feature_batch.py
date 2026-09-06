"""Batched local MMseqs2 feature generation for AlphaFold 3 proteins and RNA."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from absl import logging

from alphapulldown.utils.feature_metadata import (
    embed_metadata_in_af3_json,
    extract_metadata_from_af3_json,
)


PROTEIN = "protein"
RNA = "rna"
# Molecule types in the order they are searched. Protein comes first so that a
# protein-only batch issues exactly the same MMseqs2 calls, in the same order, as
# it did before RNA existed.
MOLECULE_TYPES = (PROTEIN, RNA)

_PROTEIN_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")
# AlphaFold 3 tokenises an RNA MSA over A/C/G/U and maps every other letter to the
# unknown nucleotide, so a T here would be silently discarded rather than read as U.
_RNA_RESIDUES = frozenset("ACGUN")
_RESIDUES = {PROTEIN: _PROTEIN_RESIDUES, RNA: _RNA_RESIDUES}
_MOLECULE_NOUNS = {PROTEIN: "proteins", RNA: "RNA chains"}

# The two protein database roles. Unpaired hits are merged into one MSA; paired hits
# keep their UniProt taxon headers so AlphaFold 3 can pair chains by species. Getting
# these the wrong way round produces a plausible-looking MSA and silently wrong
# pairing, so the roles are named here rather than recovered from a position in a tuple.
UNPAIRED_DATABASE_NAMES = ("uniref90", "mgnify", "small_bfd")
PAIRED_DATABASE_NAME = "uniprot"
DATABASE_NAMES = (*UNPAIRED_DATABASE_NAMES, PAIRED_DATABASE_NAME)

# AlphaFold 3 searches three RNA databases and merges them into one unpaired MSA, in
# this order (alphafold3.data.pipeline._get_rna_msa). RNA chains have no paired MSA and
# no templates at all, so there is no RNA counterpart to PAIRED_DATABASE_NAME.
RNA_DATABASE_NAMES = ("rfam", "rnacentral", "nt_rna")

DEFAULT_MAX_SEQUENCES = {
    "uniref90": 10_000,
    "mgnify": 5_000,
    "small_bfd": 5_000,
    "uniprot": 50_000,
}
# AlphaFold 3 caps every RNA database at 10,000 hits and searches them at 1e-3.
DEFAULT_RNA_MAX_SEQUENCES = {name: 10_000 for name in RNA_DATABASE_NAMES}
DEFAULT_RNA_E_VALUE = 1e-3
_FALLBACK_MAX_SEQUENCES = 5_000

# MMseqs2's GPU prefilter needs a padded protein database, so a nucleotide search runs
# on CPU however the process was configured. Recorded in RNA provenance so that a
# future GPU-capable nucleotide search does not silently reuse these bundles.
NUCLEOTIDE_SEARCH_MODE = "cpu"


def _validate_feature_requests(requests: Sequence[FeatureRequest]) -> None:
    names = [request.name for request in requests]
    if len(set(names)) != len(names):
        raise ValueError("Feature request names must be unique")
    for request in requests:
        if not request.name or Path(request.name).name != request.name:
            raise ValueError(f"Invalid feature request name: {request.name!r}")
        residues = _RESIDUES.get(request.molecule_type)
        if residues is None:
            raise ValueError(
                f"Feature request {request.name!r} has an unsupported molecule "
                f"type: {request.molecule_type!r}"
            )
        sequence = request.sequence.upper()
        if not sequence or set(sequence) - residues:
            raise ValueError(
                f"Feature request {request.name!r} is not a "
                f"{request.molecule_type} sequence"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureRequest:
    """One named sequence requiring an AF3 feature artifact."""

    name: str
    sequence: str
    molecule_type: str = PROTEIN


def feature_requests_from_fastas(
    fasta_paths: Sequence[str | Path],
    *,
    molecule_types: Sequence[str] = (PROTEIN,),
) -> tuple[FeatureRequest, ...]:
    """Read requests of the accepted molecule types without importing AlphaFold."""
    from alphapulldown.utils.file_handling import iter_seqs
    from alphapulldown.utils.sequence_types import get_af3_chain_kind

    accepted = frozenset(molecule_types)
    unsupported = accepted - frozenset(MOLECULE_TYPES)
    if unsupported:
        raise ValueError(
            "Unsupported molecule types: " + ", ".join(sorted(unsupported))
        )
    ordered = tuple(kind for kind in MOLECULE_TYPES if kind in accepted)
    nouns = " and ".join(_MOLECULE_NOUNS[kind] for kind in ordered)
    rejection = (
        f"is not a {ordered[0]}" if len(ordered) == 1 else "is neither of those"
    )

    requests = []
    for sequence, description in iter_seqs([str(path) for path in fasta_paths]):
        chain_kind = get_af3_chain_kind(description, sequence)
        if chain_kind not in accepted:
            message = (
                f"Batched local MMseqs2-GPU features accept {nouns} only; "
                f"{description!r} {rejection}"
            )
            if chain_kind == RNA and RNA not in accepted:
                message += (
                    ". Configure the RNA databases (--mmseqs_rfam_database_path, "
                    "--mmseqs_rnacentral_database_path, "
                    "--mmseqs_nt_rna_database_path) to search RNA chains too"
                )
            raise ValueError(message)
        requests.append(
            FeatureRequest(
                name=description, sequence=sequence, molecule_type=chain_kind
            )
        )
    return tuple(requests)


def protein_requests_from_fastas(
    fasta_paths: Sequence[str | Path],
) -> tuple[FeatureRequest, ...]:
    """Read protein requests without importing AlphaFold or JAX."""
    return feature_requests_from_fastas(fasta_paths)


@dataclasses.dataclass(frozen=True, slots=True)
class DatabaseSpec:
    """An explicit MMseqs2 database and its immutable cache identity."""

    name: str
    path: Path
    identifier: str
    max_sequences: int | None = None
    molecule_type: str = PROTEIN


@dataclasses.dataclass(frozen=True, slots=True)
class DatabaseSelection:
    """The configured databases with their roles named."""

    unpaired: tuple[DatabaseSpec, ...]
    paired: DatabaseSpec | None
    # All three RNA databases are unpaired; empty when RNA is not configured.
    rna: tuple[DatabaseSpec, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class MsaBatchSettings:
    """MSA-stage settings; no AlphaFold/JAX configuration belongs here."""

    output_dir: Path
    temp_dir: Path
    # Both may be empty/None when the batch contains no protein requests.
    unpaired_databases: tuple[DatabaseSpec, ...]
    paired_database: DatabaseSpec | None
    max_sequences_per_batch: int
    max_residues_per_batch: int
    threads: int
    e_value: float = 1e-4
    # MMseqs2 sizes its database splits from 90% of the PHYSICAL node memory
    # (sysconf(_SC_PHYS_PAGES)), which ignores the cgroup a batch scheduler puts it in.
    # On a large node with a small allocation it therefore declines to split and is
    # OOM-killed instead. Pass the allocation explicitly, e.g. "150G".
    split_memory_limit: str | None = None
    # Empty unless RNA is configured; a protein-only batch never reads these.
    rna_databases: tuple[DatabaseSpec, ...] = ()
    rna_e_value: float = DEFAULT_RNA_E_VALUE


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureFinalizationSettings:
    """CPU AF3 finalization settings and complete template provenance."""

    output_dir: Path
    msa_input_dir: Path
    max_template_date: str
    template_seqres_database_id: str
    template_mmcif_database_id: str
    compress: bool = False
    base_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureBatchSettings:
    """Compatibility settings for the composed all-stage interface."""

    output_dir: Path
    temp_dir: Path
    unpaired_databases: tuple[DatabaseSpec, ...]
    paired_database: DatabaseSpec | None
    max_sequences_per_batch: int
    max_residues_per_batch: int
    threads: int
    msa_output_dir: Path | None = None
    e_value: float = 1e-4
    split_memory_limit: str | None = None
    rna_databases: tuple[DatabaseSpec, ...] = ()
    rna_e_value: float = DEFAULT_RNA_E_VALUE
    compress: bool = False
    base_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    max_template_date: str = ""
    template_seqres_database_id: str = ""
    template_mmcif_database_id: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class MsaArtifact:
    """One durable per-protein MSA bundle."""

    name: str
    path: Path


@dataclasses.dataclass(frozen=True, slots=True)
class MsaFailure:
    name: str
    error: str


@dataclasses.dataclass(frozen=True, slots=True)
class MsaBatchResult:
    written: tuple[MsaArtifact, ...]
    reused: tuple[MsaArtifact, ...]
    failures: tuple[MsaFailure, ...]
    query_only: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureArtifact:
    name: str
    path: Path


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureFailure:
    name: str
    error: str


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureBatchResult:
    written: tuple[FeatureArtifact, ...]
    reused: tuple[FeatureArtifact, ...]
    failures: tuple[FeatureFailure, ...]
    query_only: tuple[str, ...] = ()


class _InvalidMsaBundle(ValueError):
    """An on-disk MSA bundle whose contents are unsafe to reuse."""


def write_batch_summary(path: Path, result: MsaBatchResult) -> None:
    """Atomically publish a completion record for an entirely successful stage."""
    if result.failures:
        raise ValueError("A completion summary cannot represent a failed batch")
    artifacts = (*result.written, *result.reused)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        path,
        {
            "schemaVersion": 2,
            "written": [artifact.name for artifact in result.written],
            "reused": [artifact.name for artifact in result.reused],
            "artifacts": [_manifest_record(artifact) for artifact in artifacts],
        },
    )


def _manifest_record(artifact: MsaArtifact) -> dict[str, Any]:
    digest = hashlib.sha256()
    with artifact.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = artifact.path.stat()
    return {
        "name": artifact.name,
        "file": artifact.path.name,
        "sizeBytes": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


class MmseqsProcess(Protocol):
    """Process operations performed by the external MMseqs2 executable."""

    def identity(self) -> str: ...

    def search_mode(self) -> str: ...

    def create_query_database(self, query_fasta: Path, query_db: Path) -> None: ...

    def search(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        work_dir: Path,
        settings: MsaBatchSettings,
    ) -> None: ...

    def result_to_msa(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        msa_db: Path,
    ) -> None: ...

    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None: ...


class SubprocessMmseqsProcess:
    """Production adapter for one local MMseqs2 executable."""

    def __init__(self, binary_path: str | Path, *, gpu: bool = True):
        self._binary_path = str(binary_path)
        self._gpu = gpu

    def _run(self, command: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                [self._binary_path, *command],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            raise RuntimeError(
                f"MMseqs2 command failed: {' '.join(command)}\n{stderr.strip()}"
            ) from exc
        return completed.stdout.strip()

    def identity(self) -> str:
        """Return the executable version used to validate persisted artifacts."""
        version = " ".join(self._run(("version",)).split())
        if not version:
            raise RuntimeError("MMseqs2 version command returned no identity")
        return version

    def search_mode(self) -> str:
        # Recorded in each bundle's provenance: CPU and GPU search are both supported and
        # a cached result should not be reused across the two.
        return "gpu" if self._gpu else "cpu"

    def create_query_database(self, query_fasta: Path, query_db: Path) -> None:
        self._run(("createdb", str(query_fasta), str(query_db)))

    def search(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        work_dir: Path,
        settings: MsaBatchSettings,
    ) -> None:
        max_sequences = database.max_sequences or _default_max_sequences(database.name)
        nucleotide = database.molecule_type == RNA
        # A nucleotide search cannot use the GPU prefilter (it needs a padded protein
        # database), and it needs the nucleotide scoring model stated explicitly rather
        # than left to auto-detection.
        gpu = self._gpu and not nucleotide
        e_value = settings.rna_e_value if nucleotide else settings.e_value
        self._run(
            (
                "search",
                str(query_db),
                str(database.path),
                str(result_db),
                str(work_dir),
                "-a",
                "-e",
                str(e_value),
                "--threads",
                str(settings.threads),
                "--max-seqs",
                str(max_sequences),
                "--gpu",
                "1" if gpu else "0",
            )
            + (("--search-type", "3") if nucleotide else ())
            + (
                ("--split-memory-limit", settings.split_memory_limit)
                if settings.split_memory_limit
                else ()
            )
        )

    def result_to_msa(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        msa_db: Path,
    ) -> None:
        self._run(
            (
                "result2msa",
                str(query_db),
                str(database.path),
                str(result_db),
                str(msa_db),
                "--msa-format-mode",
                "2",
            )
        )

    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None:
        del query_db
        self._run(
            (
                "unpackdb",
                str(msa_db),
                str(output_dir),
                "--unpack-name-mode",
                "0",
                "--unpack-suffix",
                ".fasta",
            )
        )


def _default_max_sequences(database_name: str) -> int:
    if database_name in DEFAULT_RNA_MAX_SEQUENCES:
        return DEFAULT_RNA_MAX_SEQUENCES[database_name]
    return DEFAULT_MAX_SEQUENCES.get(database_name, _FALLBACK_MAX_SEQUENCES)


def _request_key(request: FeatureRequest) -> tuple[str, str]:
    """Identity of the search one request needs: its molecule type and sequence."""
    return (request.molecule_type, request.sequence)


class MsaBatch:
    """Search and persist reusable per-chain MMseqs2 MSA bundles."""

    def __init__(
        self,
        *,
        settings: MsaBatchSettings,
        mmseqs_process: MmseqsProcess,
    ):
        if not isinstance(settings, MsaBatchSettings):
            raise TypeError("MsaBatch settings must be MsaBatchSettings")
        self._settings = settings
        self._mmseqs = mmseqs_process
        self._mmseqs_identity: str | None = None

    def generate(self, requests: Sequence[FeatureRequest]) -> MsaBatchResult:
        requests = tuple(requests)
        self._validate(requests)
        self._settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._settings.temp_dir.mkdir(parents=True, exist_ok=True)

        reused = []
        missing_requests = []
        # Keyed by molecule type as well as sequence: the same letters searched as a
        # protein and as RNA are two different searches with two different answers.
        msa_by_sequence: dict[tuple[str, str], tuple[str, str]] = {}
        for request in requests:
            cached = self._read_matching_msa(request)
            if cached is None:
                missing_requests.append(request)
                continue
            path, payload = cached
            reused.append(MsaArtifact(name=request.name, path=path))
            msa_by_sequence.setdefault(
                _request_key(request),
                (payload["unpairedMsa"], payload["pairedMsa"]),
            )

        sequence_to_requests: dict[tuple[str, str], list[FeatureRequest]] = {}
        for request in missing_requests:
            sequence_to_requests.setdefault(_request_key(request), []).append(request)

        written = []
        failures = []
        # A cached sequence can satisfy another name without a new search.
        for key, matching_requests in sequence_to_requests.items():
            cached_msas = msa_by_sequence.get(key)
            if cached_msas is None:
                continue
            for request in matching_requests:
                try:
                    payload = self._msa_payload(request, *cached_msas)
                    path = self._msa_path(request.name)
                    _write_atomic(path, payload)
                    written.append(MsaArtifact(name=request.name, path=path))
                except Exception as exc:
                    failures.append(MsaFailure(name=request.name, error=str(exc)))

        keys_to_search = tuple(
            key for key in sequence_to_requests if key not in msa_by_sequence
        )
        search_errors: dict[tuple[str, str], str] = {}
        if keys_to_search:
            try:
                self._process_identity()
            except Exception as exc:
                search_errors.update((key, str(exc)) for key in keys_to_search)
        for molecule_type in MOLECULE_TYPES:
            sequences_to_search = tuple(
                sequence
                for kind, sequence in keys_to_search
                if kind == molecule_type
            )
            for chunk in self._pack(sequences_to_search):
                chunk_keys = tuple((molecule_type, sequence) for sequence in chunk)
                if any(key in search_errors for key in chunk_keys):
                    continue
                try:
                    chunk_msas = self._search_chunk(chunk, molecule_type)
                except Exception as exc:
                    for key in chunk_keys:
                        search_errors[key] = str(exc)
                    continue
                # Publish each completed chunk before the next expensive search.
                for sequence, msas in chunk_msas.items():
                    for request in sequence_to_requests[(molecule_type, sequence)]:
                        try:
                            payload = self._msa_payload(request, *msas)
                            path = self._msa_path(request.name)
                            _write_atomic(path, payload)
                            written.append(MsaArtifact(name=request.name, path=path))
                        except Exception as exc:
                            failures.append(
                                MsaFailure(name=request.name, error=str(exc))
                            )

        for key, error in search_errors.items():
            failures.extend(
                MsaFailure(name=request.name, error=error)
                for request in sequence_to_requests[key]
            )

        # A search that returned nothing yields a query-only MSA. That is legitimate for
        # an orphan sequence but is also exactly what a misconfigured or half-built
        # database produces, and the two are indistinguishable from provenance alone -
        # so make it visible instead of recording it as an ordinary success.
        query_only = []
        for artifact in (*written, *reused):
            depths = self._artifact_depths(artifact.path)
            if depths is not None and max(depths) <= 1:
                query_only.append(artifact.name)
                logging.warning(
                    "MMseqs2 returned no hits for %s: the MSA contains only the query. "
                    "Check the configured databases before using this artifact.",
                    artifact.name,
                )
        return MsaBatchResult(
            written=tuple(written),
            reused=tuple(reused),
            failures=tuple(failures),
            query_only=tuple(query_only),
        )

    @staticmethod
    def _artifact_depths(path: Path) -> tuple[int, int] | None:
        try:
            with open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        unpaired = payload.get("unpairedDepth")
        paired = payload.get("pairedDepth")
        if isinstance(unpaired, int) and isinstance(paired, int):
            return unpaired, paired
        return (
            _msa_depth(payload.get("unpairedMsa", "")),
            _msa_depth(payload.get("pairedMsa", "")),
        )

    def _read_matching_msa(
        self, request: FeatureRequest
    ) -> tuple[Path, dict[str, Any]] | None:
        path = self._msa_path(request.name)
        if not path.exists():
            return None
        try:
            with open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("sequence") != request.sequence:
                return None
            # Bundles written before RNA existed carry no molecule type and are protein.
            if payload.get("moleculeType", PROTEIN) != request.molecule_type:
                return None
            if payload.get("provenance") != self._cache_signature(
                request.molecule_type
            ):
                return None
            if not isinstance(payload.get("unpairedMsa"), str) or not isinstance(
                payload.get("pairedMsa"), str
            ):
                return None
            return path, payload
        except (
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    def _validate(self, requests: Sequence[FeatureRequest]) -> None:
        _validate_feature_requests(requests)
        if self._settings.max_sequences_per_batch < 1:
            raise ValueError("max_sequences_per_batch must be at least 1")
        if self._settings.max_residues_per_batch < 1:
            raise ValueError("max_residues_per_batch must be at least 1")
        if self._settings.threads < 1:
            raise ValueError("threads must be at least 1")
        if self._settings.e_value <= 0:
            raise ValueError("e_value must be greater than 0")

        # Only the database sets this batch will actually search have to be configured,
        # so an RNA-only batch does not need the protein databases and vice versa. An
        # empty batch is validated as protein, as it always has been.
        molecule_types = {request.molecule_type for request in requests} or {PROTEIN}
        databases: list[DatabaseSpec] = []
        if PROTEIN in molecule_types:
            unpaired_names = tuple(
                database.name for database in self._settings.unpaired_databases
            )
            if unpaired_names != UNPAIRED_DATABASE_NAMES:
                raise ValueError(
                    "unpaired_databases must explicitly provide "
                    f"{', '.join(UNPAIRED_DATABASE_NAMES)} in that order"
                )
            paired = self._settings.paired_database
            if paired is None or paired.name != PAIRED_DATABASE_NAME:
                raise ValueError(
                    f"paired_database must explicitly provide {PAIRED_DATABASE_NAME}"
                )
            databases.extend((*self._settings.unpaired_databases, paired))
        if RNA in molecule_types:
            if self._settings.rna_e_value <= 0:
                raise ValueError("rna_e_value must be greater than 0")
            rna_names = tuple(
                database.name for database in self._settings.rna_databases
            )
            if rna_names != RNA_DATABASE_NAMES:
                raise ValueError(
                    "rna_databases must explicitly provide "
                    f"{', '.join(RNA_DATABASE_NAMES)} in that order"
                )
            databases.extend(self._settings.rna_databases)

        for database in databases:
            if database.path == Path("."):
                raise ValueError(
                    f"Database {database.name!r} requires an explicit path"
                )
            if not database.identifier.strip():
                raise ValueError(
                    f"Database {database.name!r} requires a non-empty identifier"
                )
            if database.max_sequences is not None and database.max_sequences < 1:
                raise ValueError(
                    f"Database {database.name!r} max_sequences must be at least 1"
                )

    def _databases(
        self, molecule_type: str
    ) -> tuple[tuple[DatabaseSpec, ...], DatabaseSpec | None]:
        """The unpaired databases and the paired one, if the molecule type has one."""
        if molecule_type == RNA:
            # AlphaFold 3 does not pair RNA chains, so all three are unpaired.
            return self._settings.rna_databases, None
        return self._settings.unpaired_databases, self._settings.paired_database

    def _pack(self, sequences: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        chunks: list[tuple[str, ...]] = []
        current: list[str] = []
        residues = 0
        for sequence in sequences:
            would_exceed = current and (
                len(current) >= self._settings.max_sequences_per_batch
                or residues + len(sequence) > self._settings.max_residues_per_batch
            )
            if would_exceed:
                chunks.append(tuple(current))
                current = []
                residues = 0
            current.append(sequence)
            residues += len(sequence)
        if current:
            chunks.append(tuple(current))
        return tuple(chunks)

    def _search_chunk(
        self, sequences: Sequence[str], molecule_type: str = PROTEIN
    ) -> dict[str, tuple[str, str]]:
        unpaired_databases, paired_database = self._databases(molecule_type)
        with tempfile.TemporaryDirectory(
            prefix="alphapulldown_mmseqs_", dir=self._settings.temp_dir
        ) as temporary_directory:
            root = Path(temporary_directory)
            query_fasta = root / "queries.fasta"
            query_ids = {
                f"query_{index}": sequence for index, sequence in enumerate(sequences)
            }
            # query_ids keeps the sequence as the caller spelled it; only what
            # MMseqs2 reads is respelled.
            as_query = (
                _reverse_transcribe if molecule_type == RNA else (lambda s: s)
            )
            query_fasta.write_text(
                "".join(
                    f">{query_id}\n{as_query(sequence)}\n"
                    for query_id, sequence in query_ids.items()
                ),
                encoding="utf-8",
            )
            query_db = root / "query_db"
            self._mmseqs.create_query_database(query_fasta, query_db)

            by_database: dict[str, dict[str, str]] = {}
            for database in (
                *unpaired_databases,
                *((paired_database,) if paired_database is not None else ()),
            ):
                database_root = root / database.name
                database_root.mkdir()
                result_db = database_root / "result_db"
                work_dir = database_root / "work"
                work_dir.mkdir()
                msa_db = database_root / "msa_db"
                output_dir = database_root / "a3m"
                output_dir.mkdir()
                self._mmseqs.search(
                    query_db, database, result_db, work_dir, self._settings
                )
                self._mmseqs.result_to_msa(query_db, database, result_db, msa_db)
                self._mmseqs.unpack_msa(query_db, msa_db, output_dir)
                by_database[database.name] = self._read_results(
                    query_db, output_dir, query_ids, molecule_type
                )

            results = {}
            for query_id, sequence in query_ids.items():
                unpaired = _merge_a3ms(
                    sequence,
                    [
                        by_database[database.name][query_id]
                        for database in unpaired_databases
                    ],
                )
                paired = (
                    _normalise_query(
                        by_database[paired_database.name][query_id], sequence
                    )
                    if paired_database is not None
                    else ""
                )
                results[sequence] = (unpaired, paired)
            return results

    @staticmethod
    def _read_results(
        query_db: Path,
        output_dir: Path,
        query_ids: Mapping[str, str],
        molecule_type: str = PROTEIN,
    ) -> dict[str, str]:
        index_to_query = {}
        lookup_path = Path(f"{query_db}.lookup")
        if lookup_path.exists():
            for line in lookup_path.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) >= 2:
                    index_to_query[fields[0]] = fields[1]
        if not index_to_query:
            index_to_query = {
                str(index): query_id for index, query_id in enumerate(query_ids)
            }

        results = {}
        for index, query_id in index_to_query.items():
            if query_id not in query_ids:
                continue
            candidates = (
                output_dir / f"{index}.fasta",
                output_dir / f"{index}.a3m",
                output_dir / index,
                output_dir / f"{query_id}.fasta",
                output_dir / f"{query_id}.a3m",
            )
            result_path = next((path for path in candidates if path.exists()), None)
            if result_path is None:
                raise RuntimeError(
                    "MMseqs2 unpackdb did not produce an alignment for "
                    f"{query_id!r} in {output_dir.parent.name!r}"
                )
            aligned_fasta = result_path.read_text(encoding="utf-8")
            if not _fasta_records(aligned_fasta):
                raise RuntimeError(
                    "MMseqs2 unpackdb produced no FASTA records for "
                    f"{query_id!r} in {output_dir.parent.name!r}"
                )
            results[query_id] = _aligned_fasta_to_a3m(
                aligned_fasta, query_ids[query_id], molecule_type
            )
        missing_queries = set(query_ids) - set(results)
        if missing_queries:
            raise RuntimeError(
                "MMseqs2 lookup/unpack output omitted queries: "
                + ", ".join(sorted(missing_queries))
            )
        return results

    def _msa_payload(
        self, request: FeatureRequest, unpaired_msa: str, paired_msa: str
    ) -> dict[str, Any]:
        payload = {
            "schemaVersion": 2,
            "name": request.name,
            "sequence": request.sequence,
            "unpairedMsa": unpaired_msa,
            "pairedMsa": paired_msa,
            "unpairedDepth": _msa_depth(unpaired_msa),
            "pairedDepth": _msa_depth(paired_msa),
            "provenance": self._cache_signature(request.molecule_type),
        }
        # Only non-protein bundles record their molecule type, so a protein bundle is
        # byte-for-byte what this stage wrote before RNA existed and every already
        # cached protein MSA stays valid.
        if request.molecule_type != PROTEIN:
            payload["moleculeType"] = request.molecule_type
        return payload

    def _cache_signature(self, molecule_type: str = PROTEIN) -> dict[str, Any]:
        def database_value(database: DatabaseSpec) -> dict[str, Any]:
            return {
                "name": database.name,
                "identifier": database.identifier,
                "max_sequences": database.max_sequences
                or _default_max_sequences(database.name),
                # `identifier` is operator-supplied, so it cannot detect a database that
                # was rebuilt, truncated or half-copied under the same name. The index
                # size is a cheap content-derived witness that changes when it does.
                "index_size": _database_index_size(database.path),
            }

        if molecule_type == RNA:
            return {
                "schema_version": 1,
                "molecule_type": RNA,
                "mmseqs_identity": self._process_identity(),
                "search_mode": NUCLEOTIDE_SEARCH_MODE,
                "e_value": self._settings.rna_e_value,
                "unpaired_databases": [
                    database_value(database)
                    for database in self._settings.rna_databases
                ],
            }

        return {
            "schema_version": 4,
            "mmseqs_identity": self._process_identity(),
            "search_mode": self._search_mode(),
            "e_value": self._settings.e_value,
            "unpaired_databases": [
                database_value(database)
                for database in self._settings.unpaired_databases
            ],
            "paired_database": database_value(self._settings.paired_database),
        }

    def _search_mode(self) -> str:
        operation = getattr(self._mmseqs, "search_mode", None)
        return operation() if operation is not None else "gpu"

    def _process_identity(self) -> str:
        if self._mmseqs_identity is None:
            try:
                identity = self._mmseqs.identity().strip()
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "Cannot identify the configured MMseqs2 executable; cached MSAs "
                    f"cannot be trusted and search cannot proceed: {exc}"
                ) from exc
            if not identity:
                raise ValueError("MMseqs2 process identity must not be empty")
            self._mmseqs_identity = identity
        return self._mmseqs_identity

    def _msa_path(self, name: str) -> Path:
        return self._settings.output_dir / f"{name}_mmseqs_msa.json"


class FeatureFinalizer:
    """Turn persisted MSA bundles into standard AF3 feature artifacts on CPU."""

    def __init__(
        self,
        *,
        settings: FeatureFinalizationSettings,
        af3_pipeline: Any,
    ):
        if not isinstance(settings, FeatureFinalizationSettings):
            raise TypeError(
                "FeatureFinalizer settings must be FeatureFinalizationSettings"
            )
        self._settings = settings
        self._af3_pipeline = af3_pipeline

    def generate(self, requests: Sequence[FeatureRequest]) -> FeatureBatchResult:
        requests = tuple(requests)
        self._validate(requests)
        self._settings.output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        reused = []
        failures = []
        for request in requests:
            try:
                msa_payload = self._read_msa(request)
                cached = self._read_matching_artifact(request, msa_payload)
                if cached is not None:
                    reused.append(FeatureArtifact(name=request.name, path=cached))
                    continue
                payload = self._process_with_af3(request, msa_payload)
                path = self._artifact_path(request.name)
                _write_atomic(path, payload)
                written.append(FeatureArtifact(name=request.name, path=path))
            except Exception as exc:
                failures.append(FeatureFailure(name=request.name, error=str(exc)))
        return FeatureBatchResult(
            written=tuple(written), reused=tuple(reused), failures=tuple(failures)
        )

    def _validate(self, requests: Sequence[FeatureRequest]) -> None:
        _validate_feature_requests(requests)
        # AlphaFold 3 searches templates for protein chains only, so template
        # provenance is required exactly when this batch contains one. An empty batch
        # is treated as protein, as it always has been.
        if requests and not any(
            request.molecule_type == PROTEIN for request in requests
        ):
            return
        for field_name in (
            "max_template_date",
            "template_seqres_database_id",
            "template_mmcif_database_id",
        ):
            if not str(getattr(self._settings, field_name)).strip():
                raise ValueError(f"{field_name} requires a non-empty value")

    def _read_msa(self, request: FeatureRequest) -> dict[str, Any]:
        path = self._settings.msa_input_dir / f"{request.name}_mmseqs_msa.json"
        try:
            encoded = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Cannot read MMseqs2 MSA bundle {path}: {exc}") from exc
        try:
            try:
                payload = json.loads(encoded)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise _InvalidMsaBundle(
                    f"Cannot parse MMseqs2 MSA bundle {path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise _InvalidMsaBundle(
                    f"MMseqs2 MSA bundle {path} is not a JSON object"
                )
            if payload.get("sequence") != request.sequence:
                raise _InvalidMsaBundle(
                    f"MMseqs2 MSA bundle sequence does not match {request.name!r}"
                )
            if payload.get("moleculeType", PROTEIN) != request.molecule_type:
                raise _InvalidMsaBundle(
                    "MMseqs2 MSA bundle molecule type does not match "
                    f"{request.name!r}"
                )
            if not isinstance(payload.get("provenance"), dict):
                raise _InvalidMsaBundle(
                    f"MMseqs2 MSA bundle lacks provenance for {request.name!r}"
                )
            for key in ("unpairedMsa", "pairedMsa"):
                if not isinstance(payload.get(key), str):
                    raise _InvalidMsaBundle(
                        f"MMseqs2 MSA bundle lacks {key} for {request.name!r}"
                    )
            return payload
        except _InvalidMsaBundle:
            path.unlink(missing_ok=True)
            raise

    def _read_matching_artifact(
        self, request: FeatureRequest, msa_payload: Mapping[str, Any]
    ) -> Path | None:
        path = self._artifact_path(request.name)
        if not path.exists():
            return None
        try:
            opener = lzma.open if path.suffix == ".xz" else open
            with opener(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            # An artifact written for a different molecule type has a different chain
            # key, so this raises KeyError and is treated as a miss.
            chain = payload["sequences"][0][request.molecule_type]
            metadata = extract_metadata_from_af3_json(payload)
            if chain.get("sequence") != request.sequence or not metadata:
                return None
            other = metadata[0].get("other", {})
            if other.get("mmseqs2_gpu") != msa_payload["provenance"]:
                return None
            if other.get("af3_templates") != self._template_signature():
                return None
            return path
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _process_with_af3(
        self, request: FeatureRequest, msa_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        from alphafold3.common import folding_input

        if request.molecule_type == RNA:
            # An AF3 RNA chain carries one unpaired MSA and nothing else: RNA chains
            # are never paired by species and never get templates.
            chain = {
                "rna": {
                    "id": "A",
                    "sequence": request.sequence,
                    "description": request.name,
                    "unpairedMsa": msa_payload["unpairedMsa"],
                }
            }
        else:
            chain = {
                "protein": {
                    "id": "A",
                    "sequence": request.sequence,
                    "description": request.name,
                    "unpairedMsa": msa_payload["unpairedMsa"],
                    "pairedMsa": msa_payload["pairedMsa"],
                    # Native AF3 searches templates from the merged unpaired MSA.
                    "templates": None,
                }
            }
        source_payload = {
            "name": request.name,
            "modelSeeds": [42],
            "sequences": [chain],
            "dialect": "alphafold3",
            "version": 1,
        }
        fold_input = folding_input.Input.from_json(json.dumps(source_payload))
        processed = self._af3_pipeline.process(fold_input)
        payload = json.loads(processed.to_json())
        payload["name"] = request.name
        metadata = copy.deepcopy(dict(self._settings.base_metadata))
        other = metadata.setdefault("other", {})
        other["msa_backend"] = "mmseqs2-gpu"
        other["mmseqs2_gpu"] = msa_payload["provenance"]
        other["af3_templates"] = self._template_signature()
        return embed_metadata_in_af3_json(payload, metadata)

    def _template_signature(self) -> dict[str, str | int]:
        return {
            "schema_version": 1,
            "max_template_date": self._settings.max_template_date,
            "pdb_seqres_database_id": self._settings.template_seqres_database_id,
            "mmcif_database_id": self._settings.template_mmcif_database_id,
        }

    def _artifact_path(self, name: str) -> Path:
        suffix = "_af3_input.json.xz" if self._settings.compress else "_af3_input.json"
        return self._settings.output_dir / f"{name}{suffix}"


class FeatureBatch:
    """Compatibility facade that composes GPU MSA and CPU AF3 stages."""

    def __init__(
        self,
        *,
        settings: FeatureBatchSettings,
        mmseqs_process: MmseqsProcess,
        af3_pipeline: Any,
    ):
        msa_output_dir = settings.msa_output_dir or settings.output_dir / ".mmseqs_msas"
        self._msa_batch = MsaBatch(
            settings=MsaBatchSettings(
                output_dir=msa_output_dir,
                temp_dir=settings.temp_dir,
                unpaired_databases=settings.unpaired_databases,
                paired_database=settings.paired_database,
                max_sequences_per_batch=settings.max_sequences_per_batch,
                max_residues_per_batch=settings.max_residues_per_batch,
                threads=settings.threads,
                e_value=settings.e_value,
                split_memory_limit=settings.split_memory_limit,
                rna_databases=settings.rna_databases,
                rna_e_value=settings.rna_e_value,
            ),
            mmseqs_process=mmseqs_process,
        )
        self._finalizer = FeatureFinalizer(
            settings=FeatureFinalizationSettings(
                output_dir=settings.output_dir,
                msa_input_dir=msa_output_dir,
                max_template_date=settings.max_template_date,
                template_seqres_database_id=settings.template_seqres_database_id,
                template_mmcif_database_id=settings.template_mmcif_database_id,
                compress=settings.compress,
                base_metadata=settings.base_metadata,
            ),
            af3_pipeline=af3_pipeline,
        )

    def generate(self, requests: Sequence[FeatureRequest]) -> FeatureBatchResult:
        msa_result = self._msa_batch.generate(requests)
        failed_names = {failure.name for failure in msa_result.failures}
        final_result = self._finalizer.generate(
            [request for request in requests if request.name not in failed_names]
        )
        return FeatureBatchResult(
            written=final_result.written,
            reused=final_result.reused,
            query_only=msa_result.query_only,
            failures=(
                *(
                    FeatureFailure(name=failure.name, error=failure.error)
                    for failure in msa_result.failures
                ),
                *final_result.failures,
            ),
        )


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish JSON so completed cache entries survive worker failure."""
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw_handle:
            if path.suffix == ".xz":
                with lzma.open(raw_handle, "wt", encoding="utf-8") as handle:
                    handle.write(text)
            else:
                raw_handle.write(text.encode("utf-8"))
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


def _database_index_size(path: Path) -> int:
    """Size of an MMseqs2 database index, or 0 when it cannot be read."""
    try:
        return Path(f"{path}.index").stat().st_size
    except OSError:
        return 0


def _msa_depth(a3m: str) -> int:
    """Number of alignment rows, including the query."""
    return sum(1 for line in a3m.splitlines() if line.startswith(">"))


def _fasta_records(fasta: str) -> list[tuple[str, str]]:
    records = []
    description = None
    sequence_parts = []
    for line in fasta.splitlines():
        if line.startswith(">"):
            if description is not None:
                records.append((description, "".join(sequence_parts)))
            description = line[1:].strip()
            sequence_parts = []
        elif description is not None:
            sequence_parts.append(line.strip())
    if description is not None:
        records.append((description, "".join(sequence_parts)))
    return records


def _reverse_transcribe(sequence: str) -> str:
    """Spell an RNA query as DNA before it reaches ``mmseqs createdb``.

    A, C, G and U are all valid amino-acid codes -- U is selenocysteine -- so
    MMseqs2 reads a U-spelled RNA as a protein and maps every U to X, the unknown
    residue. The query then returns from ``result2msa`` with its uracils
    destroyed, and no normalisation can recover them. Spelling uracil as T keeps
    the query in the nucleotide alphabet; :func:`_transcribe` puts the U back on
    the way out.
    """
    return sequence.replace("U", "T").replace("u", "t")


def _transcribe(sequence: str) -> str:
    """Read a nucleotide sequence as RNA, whichever alphabet it was written in.

    MMseqs2 nucleotide databases are built from FASTAs that may spell uracil as T, but
    AlphaFold 3 tokenises an RNA MSA over A/C/G/U and turns every other letter into the
    unknown nucleotide. Left alone, a hit from a DNA-alphabet database would therefore
    reach the model as a row of unknowns rather than as a homologue.
    """
    return sequence.replace("T", "U").replace("t", "u")


def _aligned_fasta_to_a3m(
    aligned_fasta: str, query_sequence: str, molecule_type: str = PROTEIN
) -> str:
    """Remove query-gap columns while retaining insertions and full headers."""
    from alphafold3.cpp import msa_conversion

    records = _fasta_records(aligned_fasta)
    if not records:
        raise ValueError("MMseqs2 aligned FASTA contains no records")
    query_alignment = records[0][1]
    normalise = _transcribe if molecule_type == RNA else (lambda sequence: sequence)
    if normalise(
        query_alignment.replace("-", "").replace(".", "").upper()
    ) != normalise(query_sequence.upper()):
        raise ValueError(
            "MMseqs2 aligned FASTA query does not match its input sequence"
        )

    converted = []
    for description, sequence in records:
        a3m_sequence = msa_conversion.align_sequence_to_gapless_query(
            sequence=sequence,
            query_sequence=query_alignment,
        ).replace(".", "")
        if molecule_type == RNA:
            a3m_sequence = _transcribe(a3m_sequence)
        converted.append(f">{description}\n{a3m_sequence}\n")
    return "".join(converted)


def _normalise_query(a3m: str, query_sequence: str) -> str:
    records = _fasta_records(a3m)
    if not records:
        raise ValueError("MMseqs2 A3M contains no records")
    records[0] = ("query", query_sequence)
    return "".join(f">{description}\n{sequence}\n" for description, sequence in records)


def _merge_a3ms(query_sequence: str, a3ms: Sequence[str]) -> str:
    rows = [("query", query_sequence)]
    seen = {query_sequence}
    for a3m in a3ms:
        for _, (description, sequence) in enumerate(_fasta_records(a3m)):
            if sequence in seen:
                continue
            seen.add(sequence)
            rows.append((description, sequence))
    return "".join(f">{description}\n{sequence}\n" for description, sequence in rows)
