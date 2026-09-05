"""Batched local MMseqs2-GPU feature generation for AlphaFold 3 proteins."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from absl import logging

from alphapulldown.utils.feature_metadata import (
    embed_metadata_in_af3_json,
    extract_metadata_from_af3_json,
)
from alphapulldown.utils.file_handling import write_atomic_json as _write_atomic


_PROTEIN_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")

# The two database roles. Unpaired hits are merged into one MSA; paired hits keep their
# UniProt taxon headers so AlphaFold 3 can pair chains by species. Getting these the
# wrong way round produces a plausible-looking MSA and silently wrong pairing, so the
# roles are named here rather than recovered from a position in a tuple.
UNPAIRED_DATABASE_NAMES = ("uniref90", "mgnify", "small_bfd")
PAIRED_DATABASE_NAME = "uniprot"
DATABASE_NAMES = (*UNPAIRED_DATABASE_NAMES, PAIRED_DATABASE_NAME)

DEFAULT_MAX_SEQUENCES = {
    "uniref90": 10_000,
    "mgnify": 5_000,
    "small_bfd": 5_000,
    "uniprot": 50_000,
}
_FALLBACK_MAX_SEQUENCES = 5_000


def _validate_feature_requests(requests: Sequence[FeatureRequest]) -> None:
    names = [request.name for request in requests]
    if len(set(names)) != len(names):
        raise ValueError("Feature request names must be unique")
    for request in requests:
        if not request.name or Path(request.name).name != request.name:
            raise ValueError(f"Invalid feature request name: {request.name!r}")
        sequence = request.sequence.upper()
        if not sequence or set(sequence) - _PROTEIN_RESIDUES:
            raise ValueError(
                f"Feature request {request.name!r} is not a protein sequence"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureRequest:
    """One named protein sequence requiring an AF3 feature artifact."""

    name: str
    sequence: str


def protein_requests_from_fastas(
    fasta_paths: Sequence[str | Path],
) -> tuple[FeatureRequest, ...]:
    """Read protein requests without importing AlphaFold or JAX."""
    from alphapulldown.utils.file_handling import iter_seqs
    from alphapulldown.utils.sequence_types import get_af3_chain_kind

    requests = []
    for sequence, description in iter_seqs([str(path) for path in fasta_paths]):
        if get_af3_chain_kind(description, sequence) != "protein":
            raise ValueError(
                "Batched local MMseqs2-GPU features accept proteins only; "
                f"{description!r} is not a protein"
            )
        requests.append(FeatureRequest(name=description, sequence=sequence))
    return tuple(requests)


@dataclasses.dataclass(frozen=True, slots=True)
class DatabaseSpec:
    """An explicit MMseqs2 database and its immutable cache identity."""

    name: str
    path: Path
    identifier: str
    max_sequences: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class DatabaseSelection:
    """The configured databases with their roles named."""

    unpaired: tuple[DatabaseSpec, ...]
    paired: DatabaseSpec


@dataclasses.dataclass(frozen=True, slots=True)
class MsaBatchSettings:
    """GPU MSA-stage settings; no AlphaFold/JAX configuration belongs here."""

    output_dir: Path
    temp_dir: Path
    unpaired_databases: tuple[DatabaseSpec, ...]
    paired_database: DatabaseSpec
    max_sequences_per_batch: int
    max_residues_per_batch: int
    threads: int
    e_value: float = 1e-4
    # MMseqs2 sizes its database splits from 90% of the PHYSICAL node memory
    # (sysconf(_SC_PHYS_PAGES)), which ignores the cgroup a batch scheduler puts it in.
    # On a large node with a small allocation it therefore declines to split and is
    # OOM-killed instead. Pass the allocation explicitly, e.g. "150G".
    split_memory_limit: str | None = None


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
    paired_database: DatabaseSpec
    max_sequences_per_batch: int
    max_residues_per_batch: int
    threads: int
    msa_output_dir: Path | None = None
    e_value: float = 1e-4
    split_memory_limit: str | None = None
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
        self._run(
            (
                "search",
                str(query_db),
                str(database.path),
                str(result_db),
                str(work_dir),
                "-a",
                "-e",
                str(settings.e_value),
                "--threads",
                str(settings.threads),
                "--max-seqs",
                str(max_sequences),
                "--gpu",
                "1" if self._gpu else "0",
            )
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
    return DEFAULT_MAX_SEQUENCES.get(database_name, _FALLBACK_MAX_SEQUENCES)


class MsaBatch:
    """Search and persist reusable per-protein MMseqs2 MSA bundles."""

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
        msa_by_sequence: dict[str, tuple[str, str]] = {}
        for request in requests:
            cached = self._read_matching_msa(request)
            if cached is None:
                missing_requests.append(request)
                continue
            path, payload = cached
            reused.append(MsaArtifact(name=request.name, path=path))
            msa_by_sequence.setdefault(
                request.sequence,
                (payload["unpairedMsa"], payload["pairedMsa"]),
            )

        sequence_to_requests: dict[str, list[FeatureRequest]] = {}
        for request in missing_requests:
            sequence_to_requests.setdefault(request.sequence, []).append(request)

        written = []
        failures = []
        # A cached sequence can satisfy another name without a new search.
        for sequence, matching_requests in sequence_to_requests.items():
            cached_msas = msa_by_sequence.get(sequence)
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

        sequences_to_search = tuple(
            sequence
            for sequence in sequence_to_requests
            if sequence not in msa_by_sequence
        )
        search_errors: dict[str, str] = {}
        if sequences_to_search:
            try:
                self._process_identity()
            except Exception as exc:
                search_errors.update(
                    (sequence, str(exc)) for sequence in sequences_to_search
                )
        for chunk in self._pack(sequences_to_search):
            if any(sequence in search_errors for sequence in chunk):
                continue
            try:
                chunk_msas = self._search_chunk(chunk)
            except Exception as exc:
                for sequence in chunk:
                    search_errors[sequence] = str(exc)
                continue
            # Publish each completed chunk before starting the next expensive search.
            for sequence, msas in chunk_msas.items():
                for request in sequence_to_requests[sequence]:
                    try:
                        payload = self._msa_payload(request, *msas)
                        path = self._msa_path(request.name)
                        _write_atomic(path, payload)
                        written.append(MsaArtifact(name=request.name, path=path))
                    except Exception as exc:
                        failures.append(
                            MsaFailure(name=request.name, error=str(exc))
                        )

        for sequence, error in search_errors.items():
            failures.extend(
                MsaFailure(name=request.name, error=error)
                for request in sequence_to_requests[sequence]
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
            if payload.get("provenance") != self._cache_signature():
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
        unpaired_names = tuple(
            database.name for database in self._settings.unpaired_databases
        )
        if unpaired_names != UNPAIRED_DATABASE_NAMES:
            raise ValueError(
                "unpaired_databases must explicitly provide "
                f"{', '.join(UNPAIRED_DATABASE_NAMES)} in that order"
            )
        if self._settings.paired_database.name != PAIRED_DATABASE_NAME:
            raise ValueError(
                f"paired_database must explicitly provide {PAIRED_DATABASE_NAME}"
            )
        for database in (
            *self._settings.unpaired_databases,
            self._settings.paired_database,
        ):
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

    def _search_chunk(self, sequences: Sequence[str]) -> dict[str, tuple[str, str]]:
        with tempfile.TemporaryDirectory(
            prefix="alphapulldown_mmseqs_", dir=self._settings.temp_dir
        ) as temporary_directory:
            root = Path(temporary_directory)
            query_fasta = root / "queries.fasta"
            query_ids = {
                f"query_{index}": sequence for index, sequence in enumerate(sequences)
            }
            query_fasta.write_text(
                "".join(
                    f">{query_id}\n{sequence}\n"
                    for query_id, sequence in query_ids.items()
                ),
                encoding="utf-8",
            )
            query_db = root / "query_db"
            self._mmseqs.create_query_database(query_fasta, query_db)

            by_database: dict[str, dict[str, str]] = {}
            for database in (
                *self._settings.unpaired_databases,
                self._settings.paired_database,
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
                    query_db, output_dir, query_ids
                )

            results = {}
            for query_id, sequence in query_ids.items():
                unpaired = _merge_a3ms(
                    sequence,
                    [
                        by_database[database.name][query_id]
                        for database in self._settings.unpaired_databases
                    ],
                )
                paired = _normalise_query(
                    by_database[self._settings.paired_database.name][query_id], sequence
                )
                results[sequence] = (unpaired, paired)
            return results

    @staticmethod
    def _read_results(
        query_db: Path, output_dir: Path, query_ids: Mapping[str, str]
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
                aligned_fasta, query_ids[query_id]
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
        return {
            "schemaVersion": 2,
            "name": request.name,
            "sequence": request.sequence,
            "unpairedMsa": unpaired_msa,
            "pairedMsa": paired_msa,
            "unpairedDepth": _msa_depth(unpaired_msa),
            "pairedDepth": _msa_depth(paired_msa),
            "provenance": self._cache_signature(),
        }

    def _cache_signature(self) -> dict[str, Any]:
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
            protein = payload["sequences"][0]["protein"]
            metadata = extract_metadata_from_af3_json(payload)
            if protein.get("sequence") != request.sequence or not metadata:
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

        source_payload = {
            "name": request.name,
            "modelSeeds": [42],
            "sequences": [
                {
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
            ],
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


def _aligned_fasta_to_a3m(aligned_fasta: str, query_sequence: str) -> str:
    """Remove query-gap columns while retaining insertions and full headers."""
    from alphafold3.cpp import msa_conversion

    records = _fasta_records(aligned_fasta)
    if not records:
        raise ValueError("MMseqs2 aligned FASTA contains no records")
    query_alignment = records[0][1]
    if (
        query_alignment.replace("-", "").replace(".", "").upper()
        != query_sequence.upper()
    ):
        raise ValueError(
            "MMseqs2 aligned FASTA query does not match its input sequence"
        )

    converted = []
    for description, sequence in records:
        a3m_sequence = msa_conversion.align_sequence_to_gapless_query(
            sequence=sequence,
            query_sequence=query_alignment,
        ).replace(".", "")
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
