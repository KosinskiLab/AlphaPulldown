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
from alphapulldown.utils.msa_formats import (
    StitchMismatch,
    stitch_headers_and_insertions,
    strip_insertions,
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

# How the alignment text was produced, recorded in every bundle's provenance.
# Bundles written before insertions were recovered came from mode 2 alone and carry
# none -- 81.7% of small_bfd hits and 86.4% of uniprot hits lost theirs, measured on
# a real 586-residue query -- so they must never satisfy a request made now.
_MSA_FORMAT = {"headers": "result2msa mode 2", "sequences": "result2msa mode 5"}


def _describe_exit(returncode) -> str:
    """Say what an MMseqs2 exit code means, especially when it died on a signal.

    A crash leaves stderr EMPTY, so the bare message "MMseqs2 command failed" told the
    user nothing and looked like a configuration error. Measured: mmseqs result2msa
    segfaults on a specific RNA query against the 37M-entry nt_rna database, while the
    same command succeeds for other queries on that same database and for this query on
    the smaller ones. That is an upstream crash, not something to fix by reconfiguring.
    """
    if returncode is None:
        return "MMseqs2 could not be executed."
    # Python reports a signal death as a negative code; a shell in between turns the
    # same event into 128+N. Recognise both.
    signal_number = None
    if returncode < 0:
        signal_number = -returncode
    elif returncode > 128:
        signal_number = returncode - 128
    if signal_number is None:
        return f"MMseqs2 exited with status {returncode}."
    try:
        import signal as _signal

        name = _signal.Signals(signal_number).name
    except (ImportError, ValueError):
        name = f"signal {signal_number}"
    detail = (
        "MMseqs2 crashed rather than reporting an error, so it wrote nothing to stderr. "
        "This is a fault inside MMseqs2, not a configuration problem: the same command "
        "succeeds for other queries against the same database. Retrying will not help; "
        "exclude the sequence, or search it against the smaller databases."
    )
    return f"MMseqs2 was killed by {name} ({signal_number}). {detail}"


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
    # Iterative profile search, as jackhmmer does. AlphaFold 3 runs jackhmmer with
    # n_iter=3: it builds a profile from the hits and searches again, which is what
    # reaches remote homologues. MMseqs2 defaults to 1 -- a plain sequence-sequence
    # search -- so leaving this alone compares a single pass against three profile
    # iterations. Raising it costs time and recovers sensitivity on shallow families;
    # ColabFold uses 3 for UniRef for the same reason.
    num_iterations: int = 1
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
    num_iterations: int = 1
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
class SearchedMsas:
    """What one search produced for one sequence, before it becomes a bundle."""

    unpaired: str
    paired: str
    # How many rows of ``unpaired`` each database contributed, in merge order and
    # after deduplication. The query row belongs to none of them. AlphaFold 2 builds
    # its template profile from uniref90 ALONE and caps each database separately, so
    # the merged alignment is unusable to it unless these boundaries are kept.
    unpaired_rows: tuple[tuple[str, int], ...] = ()

    def rows_by_database(
        self,
    ) -> tuple[tuple[str, str], dict[str, list[tuple[str, str]]]]:
        """The query record, and each database's own rows, recovered from the spans."""
        records = _fasta_records(self.unpaired)
        if not records:
            raise ValueError("MSA bundle has an empty unpaired alignment")
        query, hits = records[0], records[1:]
        by_database: dict[str, list[tuple[str, str]]] = {}
        start = 0
        for name, count in self.unpaired_rows:
            by_database[name] = hits[start : start + count]
            start += count
        if start != len(hits):
            raise ValueError(
                f"MSA bundle row spans cover {start} rows but the alignment has "
                f"{len(hits)} besides the query"
            )
        return query, by_database


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
    ) -> None:
        """Format the hits with their full database headers and no insertions."""
        ...

    def result_to_a3m(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        msa_db: Path,
    ) -> None:
        """Format the same hits as A3M: insertions kept, headers cut to one token."""
        ...

    def unpack_msa(self, query_db: Path, msa_db: Path, output_dir: Path) -> None: ...


class SubprocessMmseqsProcess:
    """Production adapter for one local MMseqs2 executable."""

    def __init__(
        self,
        binary_path: str | Path,
        *,
        gpu: bool = True,
        db_load_mode: int | None = None,
    ):
        self._binary_path = str(binary_path)
        self._gpu = gpu
        # How MMseqs2 reads the target database (0 auto, 1 fread, 2 mmap,
        # 3 mmap+touch). Deliberately a process-level setting rather than part of
        # MsaBatchSettings: it changes memory behaviour and nothing else, so it
        # must never reach the cache signature. Two runs that differ only here
        # produce the same alignment and must keep reusing each other's bundles.
        self._db_load_mode = db_load_mode

    def _db_load_mode_option(self) -> tuple[str, ...]:
        if self._db_load_mode is None:
            return ()
        return ("--db-load-mode", str(self._db_load_mode))

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
            stderr = (getattr(exc, "stderr", "") or "").strip()
            raise RuntimeError(
                f"MMseqs2 command failed: {' '.join(command)}\n"
                f"{_describe_exit(getattr(exc, 'returncode', None))}"
                + (f"\n{stderr}" if stderr else "")
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
            + (
                # Iterative profile search is protein-only: measured on the pinned
                # build, --num-iterations 3 on a nucleotide search exits 1 ("Alignment
                # died", or "no diagonal information" on a one-sequence database).
                # RNA provenance correctly never records it.
                ("--num-iterations", str(settings.num_iterations))
                if settings.num_iterations
                and settings.num_iterations > 1
                and not nucleotide
                else ()
            )
            + (("--search-type", "3") if nucleotide else ())
            + (
                ("--split-memory-limit", settings.split_memory_limit)
                if settings.split_memory_limit
                else ()
            )
            + self._db_load_mode_option()
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
            + self._db_load_mode_option()
        )

    def result_to_a3m(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        msa_db: Path,
    ) -> None:
        # Mode 5 is the only format that keeps insertions, and it keeps nothing of
        # the header but the database key -- so it is run alongside mode 2, not
        # instead of it. Measured at 2 s against a 947 s search on small_bfd and
        # 13 s against 3739 s on uniprot, roughly 16 ms per hit.
        self._run(
            (
                "result2msa",
                str(query_db),
                str(database.path),
                str(result_db),
                str(msa_db),
                "--msa-format-mode",
                "5",
            )
            + self._db_load_mode_option()
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
        msa_by_sequence: dict[tuple[str, str], SearchedMsas] = {}
        for request in requests:
            cached = self._read_matching_msa(request)
            if cached is None:
                missing_requests.append(request)
                continue
            path, payload = cached
            reused.append(MsaArtifact(name=request.name, path=path))
            msa_by_sequence.setdefault(
                _request_key(request), searched_msas_from_payload(payload)
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
                    payload = self._msa_payload(request, cached_msas)
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
                            payload = self._msa_payload(request, msas)
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
            # Raises on a bundle whose row spans are missing or do not add up, so a
            # damaged one is re-searched instead of handed on to a consumer that
            # would slice it at the wrong rows.
            searched_msas_from_payload(payload)
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
    ) -> dict[str, SearchedMsas]:
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
                self._mmseqs.search(
                    query_db, database, result_db, work_dir, self._settings
                )
                # One search, formatted twice: headers from one pass, insertions
                # from the other. See msa_formats for why neither alone will do.
                headers_db = database_root / "headers_db"
                headers_dir = database_root / "headers"
                headers_dir.mkdir()
                self._mmseqs.result_to_msa(query_db, database, result_db, headers_db)
                self._mmseqs.unpack_msa(query_db, headers_db, headers_dir)
                insertions_db = database_root / "insertions_db"
                insertions_dir = database_root / "insertions"
                insertions_dir.mkdir()
                self._mmseqs.result_to_a3m(
                    query_db, database, result_db, insertions_db
                )
                self._mmseqs.unpack_msa(query_db, insertions_db, insertions_dir)
                by_database[database.name] = self._read_results(
                    query_db, headers_dir, insertions_dir, query_ids, molecule_type
                )

            results = {}
            for query_id, sequence in query_ids.items():
                unpaired, unpaired_rows = _merge_a3ms(
                    sequence,
                    [
                        (database.name, by_database[database.name][query_id])
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
                results[sequence] = SearchedMsas(
                    unpaired=unpaired, paired=paired, unpaired_rows=unpaired_rows
                )
            return results

    @staticmethod
    def _read_results(
        query_db: Path,
        headers_dir: Path,
        insertions_dir: Path,
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

        def unpacked_records(output_dir: Path, index: str, query_id: str):
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
            records = _fasta_records(result_path.read_text(encoding="utf-8"))
            if not records:
                raise RuntimeError(
                    "MMseqs2 unpackdb produced no FASTA records for "
                    f"{query_id!r} in {output_dir.parent.name!r}"
                )
            return records

        results = {}
        for index, query_id in index_to_query.items():
            if query_id not in query_ids:
                continue
            try:
                stitched = stitch_headers_and_insertions(
                    unpacked_records(headers_dir, index, query_id),
                    unpacked_records(insertions_dir, index, query_id),
                )
            except StitchMismatch as exc:
                raise RuntimeError(
                    f"{query_id!r} in {headers_dir.parent.name!r}: {exc}"
                ) from exc
            results[query_id] = _stitched_to_a3m(
                stitched, query_ids[query_id], molecule_type
            )
        missing_queries = set(query_ids) - set(results)
        if missing_queries:
            raise RuntimeError(
                "MMseqs2 lookup/unpack output omitted queries: "
                + ", ".join(sorted(missing_queries))
            )
        return results

    def _msa_payload(
        self, request: FeatureRequest, msas: SearchedMsas
    ) -> dict[str, Any]:
        payload = {
            "schemaVersion": 3,
            "name": request.name,
            "sequence": request.sequence,
            "unpairedMsa": msas.unpaired,
            "pairedMsa": msas.paired,
            "unpairedDepth": _msa_depth(msas.unpaired),
            "pairedDepth": _msa_depth(msas.paired),
            "unpairedDatabaseRows": [
                {"name": name, "rows": rows} for name, rows in msas.unpaired_rows
            ],
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
                "schema_version": 2,
                "molecule_type": RNA,
                "mmseqs_identity": self._process_identity(),
                "search_mode": NUCLEOTIDE_SEARCH_MODE,
                "msa_format": _MSA_FORMAT,
                "e_value": self._settings.rna_e_value,
                "unpaired_databases": [
                    database_value(database)
                    for database in self._settings.rna_databases
                ],
            }

        signature = {
            "schema_version": 5,
            "mmseqs_identity": self._process_identity(),
            "search_mode": self._search_mode(),
            "msa_format": _MSA_FORMAT,
            "e_value": self._settings.e_value,
            "unpaired_databases": [
                database_value(database)
                for database in self._settings.unpaired_databases
            ],
            "paired_database": database_value(self._settings.paired_database),
        }
        # Only recorded when raised, so every MSA cached at the default stays valid.
        # It must be recorded though: three profile iterations find sequences one pass
        # does not, so the same databases produce a different alignment.
        if self._settings.num_iterations > 1:
            signature["num_iterations"] = self._settings.num_iterations
        return signature

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
        return read_msa_bundle(self._settings.msa_input_dir, request)

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

    def _template_signature(self) -> dict[str, Any]:
        """Cache identity for a finalized artifact.

        The MSA stage keys its cache on the mmseqs2 binary's own identity, so an
        upgraded binary invalidates the MSAs it produced. This stage is the same
        shape of problem -- AlphaFold 3, hmmsearch and hmmbuild all shape the
        templates and features written here -- so it is keyed the same way. Without
        the software block, upgrading any of them silently reuses artifacts built
        by the older implementation.

        Only the software versions are included. ``base_metadata`` also carries a
        wall-clock ``date``, which would miss on every run.
        """
        return {
            "schema_version": 2,
            "max_template_date": self._settings.max_template_date,
            "pdb_seqres_database_id": self._settings.template_seqres_database_id,
            "mmcif_database_id": self._settings.template_mmcif_database_id,
            "software": dict(self._settings.base_metadata.get("software", {})),
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
            num_iterations=settings.num_iterations,
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


def _stitched_to_a3m(
    records: Sequence[tuple[str, str]],
    query_sequence: str,
    molecule_type: str = PROTEIN,
) -> str:
    """Validate stitched A3M records against the query and write them out.

    The mode-5 sequences are already A3M against the gapless query, so nothing is
    rewritten. What is checked is that every row spans exactly the query: a row
    whose match columns number anything else would silently shift every residue
    after the discrepancy once AlphaFold reads it as a table. The stitch already
    implies this -- mode 2 rows equal mode 5 rows minus insertions -- but a
    malformed row should fail here with the query named, not deep in a parser.
    """
    if not records:
        raise ValueError("MMseqs2 alignment contains no records")
    normalise = _transcribe if molecule_type == RNA else (lambda sequence: sequence)
    query_row = strip_insertions(records[0][1])
    if normalise(query_row.replace("-", "").upper()) != normalise(
        query_sequence.upper()
    ):
        raise ValueError("MMseqs2 alignment query does not match its input sequence")

    converted = []
    for description, sequence in records:
        if len(strip_insertions(sequence)) != len(query_sequence):
            raise ValueError(
                f"MMseqs2 row {description.split()[0]!r} spans "
                f"{len(strip_insertions(sequence))} query positions, not "
                f"{len(query_sequence)}"
            )
        if molecule_type == RNA:
            sequence = _transcribe(sequence)
        converted.append(f">{description}\n{sequence}\n")
    return "".join(converted)


def _normalise_query(a3m: str, query_sequence: str) -> str:
    records = _fasta_records(a3m)
    if not records:
        raise ValueError("MMseqs2 A3M contains no records")
    records[0] = ("query", query_sequence)
    return "".join(f">{description}\n{sequence}\n" for description, sequence in records)


def _merge_a3ms(
    query_sequence: str, a3ms: Sequence[tuple[str, str]]
) -> tuple[str, tuple[tuple[str, int], ...]]:
    """Merge per-database A3Ms in order, and say how many rows each contributed.

    Rows are deduplicated on their aligned residues with the insertions removed.
    That is the key this stage used before insertions were recovered, when the two
    were the same string, so a merged alignment keeps exactly the rows it kept
    then and only their content grows. Keying on the full row instead would keep a
    second copy of every hit that two databases aligned with different insertions.

    Databases are appended whole, one after another, so each one's rows are
    contiguous and a count is enough to recover them.
    """
    rows = [("query", query_sequence)]
    seen = {query_sequence}
    contributed = []
    for database_name, a3m in a3ms:
        added = 0
        for description, sequence in _fasta_records(a3m):
            key = strip_insertions(sequence)
            if key in seen:
                continue
            seen.add(key)
            rows.append((description, sequence))
            added += 1
        contributed.append((database_name, added))
    text = "".join(f">{description}\n{sequence}\n" for description, sequence in rows)
    return text, tuple(contributed)


def read_msa_bundle(
    msa_input_dir: Path, request: FeatureRequest, *, require_row_spans: bool = False
) -> dict[str, Any]:
    """Read one request's MSA bundle, deleting it if it is unsafe to use.

    Shared by every finalizer, so each backend refuses the same damaged bundles.
    Deleting is what gets a bundle rebuilt: the workflow's Shard completion then no
    longer validates, and a repair shard is scheduled. A bundle that is merely
    rejected, and left in place, fails the same finalization on every retry.

    ``require_row_spans`` is for consumers that slice the unpaired alignment by
    database -- AlphaFold 2 does. A bundle without usable spans is unusable to them,
    so it is treated as damaged, not as a failure to report and keep.
    """
    path = msa_input_dir / f"{request.name}_mmseqs_msa.json"
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
        if require_row_spans:
            try:
                searched_msas_from_payload(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise _InvalidMsaBundle(
                    f"MMseqs2 MSA bundle {path} has no usable per-database row "
                    f"spans: {exc}"
                ) from exc
        return payload
    except _InvalidMsaBundle:
        path.unlink(missing_ok=True)
        raise


def searched_msas_from_payload(payload: Mapping[str, Any]) -> SearchedMsas:
    """Read a bundle back, refusing row spans that do not describe its alignment.

    The spans are what a consumer slices by, so a wrong count does not fail -- it
    hands AlphaFold 2 the wrong database's rows as uniref90. Check that they add up.
    """
    unpaired = payload["unpairedMsa"]
    raw_rows = payload.get("unpairedDatabaseRows")
    if not isinstance(raw_rows, list):
        raise ValueError("MSA bundle lacks unpairedDatabaseRows")
    rows = []
    for entry in raw_rows:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("rows"), int)
            or isinstance(entry.get("rows"), bool)
            or entry["rows"] < 0
        ):
            raise ValueError(f"MSA bundle has a malformed row span: {entry!r}")
        rows.append((entry["name"], entry["rows"]))
    # The query row belongs to no database.
    if unpaired and sum(count for _, count in rows) != _msa_depth(unpaired) - 1:
        raise ValueError(
            "MSA bundle row spans account for "
            f"{sum(count for _, count in rows)} rows but the alignment has "
            f"{_msa_depth(unpaired) - 1} besides the query"
        )
    return SearchedMsas(
        unpaired=unpaired, paired=payload["pairedMsa"], unpaired_rows=tuple(rows)
    )
