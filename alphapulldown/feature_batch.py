"""Batched local MMseqs2-GPU feature generation for AlphaFold 3 proteins."""

from __future__ import annotations

import copy
import dataclasses
import json
import lzma
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from alphapulldown.utils.feature_metadata import (
    embed_metadata_in_af3_json,
    extract_metadata_from_af3_json,
)


_PROTEIN_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureRequest:
    """One named protein sequence requiring an AF3 feature artifact."""

    name: str
    sequence: str


@dataclasses.dataclass(frozen=True, slots=True)
class DatabaseSpec:
    """An explicit MMseqs2 database and its immutable cache identity."""

    name: str
    path: Path
    identifier: str
    max_sequences: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureBatchSettings:
    """Settings that determine batching, search results, and persistence."""

    output_dir: Path
    temp_dir: Path
    unpaired_databases: tuple[DatabaseSpec, ...]
    paired_database: DatabaseSpec
    max_sequences_per_batch: int
    max_residues_per_batch: int
    threads: int
    sensitivity: float = 7.5
    e_value: float = 1e-4
    compress: bool = False
    base_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


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


class MmseqsProcess(Protocol):
    """Process operations performed by the external MMseqs2 executable."""

    def create_query_database(self, query_fasta: Path, query_db: Path) -> None: ...

    def search(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        work_dir: Path,
        settings: FeatureBatchSettings,
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

    def __init__(self, binary_path: str | Path):
        self._binary_path = str(binary_path)

    def _run(self, command: Sequence[str]) -> None:
        try:
            subprocess.run(
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

    def create_query_database(self, query_fasta: Path, query_db: Path) -> None:
        self._run(("createdb", str(query_fasta), str(query_db)))

    def search(
        self,
        query_db: Path,
        database: DatabaseSpec,
        result_db: Path,
        work_dir: Path,
        settings: FeatureBatchSettings,
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
                "-s",
                str(settings.sensitivity),
                "-e",
                str(settings.e_value),
                "--threads",
                str(settings.threads),
                "--max-seqs",
                str(max_sequences),
                "--gpu",
                "1",
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
                "6",
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
                ".a3m",
            )
        )


def _default_max_sequences(database_name: str) -> int:
    return {
        "uniref90": 10_000,
        "mgnify": 5_000,
        "small_bfd": 5_000,
        "uniprot": 50_000,
    }.get(database_name, 5_000)


class FeatureBatch:
    """Generate many independent AF3 protein artifacts behind one interface."""

    def __init__(
        self,
        *,
        settings: FeatureBatchSettings,
        mmseqs_process: MmseqsProcess,
        af3_pipeline: Any,
    ):
        self._settings = settings
        self._mmseqs = mmseqs_process
        self._af3_pipeline = af3_pipeline

    def generate(self, requests: Sequence[FeatureRequest]) -> FeatureBatchResult:
        requests = tuple(requests)
        self._validate(requests)
        self._settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._settings.temp_dir.mkdir(parents=True, exist_ok=True)

        reused = []
        missing_requests = []
        msa_by_sequence: dict[str, tuple[str, str]] = {}
        for request in requests:
            cached = self._read_matching_artifact(request)
            if cached is None:
                missing_requests.append(request)
                continue
            path, payload = cached
            reused.append(FeatureArtifact(name=request.name, path=path))
            protein = payload["sequences"][0]["protein"]
            msa_by_sequence.setdefault(
                request.sequence,
                (protein["unpairedMsa"], protein["pairedMsa"]),
            )

        sequence_to_requests: dict[str, list[FeatureRequest]] = {}
        for request in missing_requests:
            sequence_to_requests.setdefault(request.sequence, []).append(request)

        sequences_to_search = tuple(
            sequence
            for sequence in sequence_to_requests
            if sequence not in msa_by_sequence
        )
        search_errors: dict[str, str] = {}
        for chunk in self._pack(sequences_to_search):
            try:
                msa_by_sequence.update(self._search_chunk(chunk))
            except Exception as exc:
                for sequence in chunk:
                    search_errors[sequence] = str(exc)

        written = []
        failures = []
        for request in missing_requests:
            if request.sequence in search_errors:
                failures.append(
                    FeatureFailure(
                        name=request.name, error=search_errors[request.sequence]
                    )
                )
                continue
            try:
                unpaired_msa, paired_msa = msa_by_sequence[request.sequence]
                payload = self._process_with_af3(request, unpaired_msa, paired_msa)
                path = self._artifact_path(request.name)
                self._write_atomic(path, payload)
                written.append(FeatureArtifact(name=request.name, path=path))
            except Exception as exc:  # one bad sequence must not hide other outputs
                failures.append(FeatureFailure(name=request.name, error=str(exc)))

        return FeatureBatchResult(
            written=tuple(written), reused=tuple(reused), failures=tuple(failures)
        )

    def _read_matching_artifact(
        self, request: FeatureRequest
    ) -> tuple[Path, dict[str, Any]] | None:
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
            if (
                metadata[0].get("other", {}).get("mmseqs2_gpu")
                != self._cache_signature()
            ):
                return None
            if not isinstance(protein.get("unpairedMsa"), str) or not isinstance(
                protein.get("pairedMsa"), str
            ):
                return None
            return path, payload
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _validate(self, requests: Sequence[FeatureRequest]) -> None:
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
        if self._settings.max_sequences_per_batch < 1:
            raise ValueError("max_sequences_per_batch must be at least 1")
        if self._settings.max_residues_per_batch < 1:
            raise ValueError("max_residues_per_batch must be at least 1")
        if self._settings.threads < 1:
            raise ValueError("threads must be at least 1")
        if self._settings.sensitivity <= 0:
            raise ValueError("sensitivity must be greater than 0")
        if self._settings.e_value <= 0:
            raise ValueError("e_value must be greater than 0")
        unpaired_names = tuple(
            database.name for database in self._settings.unpaired_databases
        )
        if unpaired_names != ("uniref90", "mgnify", "small_bfd"):
            raise ValueError(
                "unpaired_databases must explicitly provide uniref90, mgnify, "
                "and small_bfd in that order"
            )
        if self._settings.paired_database.name != "uniprot":
            raise ValueError("paired_database must explicitly provide uniprot")
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
                f"query_{index}": sequence
                for index, sequence in enumerate(sequences)
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
                output_dir / f"{index}.a3m",
                output_dir / index,
                output_dir / f"{query_id}.a3m",
            )
            result_path = next((path for path in candidates if path.exists()), None)
            if result_path is None:
                results[query_id] = f">query\n{query_ids[query_id]}\n"
            else:
                results[query_id] = result_path.read_text(encoding="utf-8")
        for query_id, sequence in query_ids.items():
            results.setdefault(query_id, f">query\n{sequence}\n")
        return results

    def _process_with_af3(
        self, request: FeatureRequest, unpaired_msa: str, paired_msa: str
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
                        "unpairedMsa": unpaired_msa,
                        "pairedMsa": paired_msa,
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
        return embed_metadata_in_af3_json(payload, self._metadata())

    def _metadata(self) -> dict[str, Any]:
        metadata = copy.deepcopy(dict(self._settings.base_metadata))
        other = metadata.setdefault("other", {})
        other["msa_backend"] = "mmseqs2-gpu"
        other["mmseqs2_gpu"] = self._cache_signature()
        return metadata

    def _cache_signature(self) -> dict[str, Any]:
        def database_value(database: DatabaseSpec) -> dict[str, Any]:
            return {
                "name": database.name,
                "identifier": database.identifier,
                "max_sequences": database.max_sequences
                or _default_max_sequences(database.name),
            }

        return {
            "schema_version": 1,
            "sensitivity": self._settings.sensitivity,
            "e_value": self._settings.e_value,
            "unpaired_databases": [
                database_value(database)
                for database in self._settings.unpaired_databases
            ],
            "paired_database": database_value(self._settings.paired_database),
        }

    def _artifact_path(self, name: str) -> Path:
        suffix = "_af3_input.json.xz" if self._settings.compress else "_af3_input.json"
        return self._settings.output_dir / f"{name}{suffix}"

    @staticmethod
    def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            if path.suffix == ".xz":
                with lzma.open(temporary_path, "wt", encoding="utf-8") as handle:
                    handle.write(text)
            else:
                temporary_path.write_text(text, encoding="utf-8")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _a3m_records(a3m: str) -> list[tuple[str, str]]:
    records = []
    description = None
    sequence_parts = []
    for line in a3m.splitlines():
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


def _normalise_query(a3m: str, query_sequence: str) -> str:
    records = _a3m_records(a3m)
    if not records:
        return f">query\n{query_sequence}\n"
    records[0] = ("query", query_sequence)
    return "".join(f">{description}\n{sequence}\n" for description, sequence in records)


def _merge_a3ms(query_sequence: str, a3ms: Sequence[str]) -> str:
    rows = [("query", query_sequence)]
    seen = {query_sequence}
    for a3m in a3ms:
        for _, (description, sequence) in enumerate(_a3m_records(a3m)):
            if sequence in seen:
                continue
            seen.add(sequence)
            rows.append((description, sequence))
    return "".join(f">{description}\n{sequence}\n" for description, sequence in rows)
