"""Structural template search: ESMFold folds the query, Foldseek finds neighbours.

The default AlphaFold 2 path finds templates by *sequence*: an HMM profile built
from the uniref90 alignment is searched against ``pdb_seqres`` (or a PDB70 HHM
database) and the resulting alignments become template features. That misses
templates whose sequence has drifted beyond profile detection but whose fold has
not, which is precisely the case where a template would help most.

This module offers a second source of the same kind of hit. A structure is
predicted for the query with ESMFold, Foldseek searches it against a local
structure database, and each alignment it returns is converted into the
``parsers.TemplateHit`` that AlphaFold 2 already knows how to featurise.

It is deliberately only a *source of hits*. Featurisation stays with AlphaFold
2's ``HhsearchHitFeaturizer``, so a Foldseek hit is still resolved to a chain in
the local mmCIF directory and still passes the same release-date, coverage and
duplicate prefilters as a sequence hit. The consequence worth stating plainly:
the Foldseek database must describe chains that exist in ``--template_mmcif_dir``
as ``<pdbid>.cif``, because that is the file the featuriser opens. Building the
Foldseek database from that same directory is the way to guarantee it.

Both tools run locally. Nothing here contacts a web service.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from absl import logging

from alphapulldown.utils.file_handling import write_atomic_json


# Columns requested from Foldseek, in the order Foldseek writes them. The parser
# is driven by this tuple and every cache entry records it, so reordering or
# extending the set invalidates cached alignments rather than silently reading
# one column as another.
FOLDSEEK_OUTPUT_COLUMNS = (
    "query",
    "target",
    "fident",
    "alnlen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
    "qaln",
    "taln",
    "alntmscore",
)

# Foldseek's 3Di+AA Gotoh-Smith-Waterman alignment. TMalign (1) is the other
# option and is considerably more expensive per hit.
ALIGNMENT_TYPE_3DI_AA = 2
ALIGNMENT_TYPE_TMALIGN = 1

_PROTEIN_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")

# Suffixes a Foldseek target name carries when its database was built straight
# from a directory of structure files.
_STRUCTURE_FILE_SUFFIXES = (".gz", ".zst", ".cif", ".mmcif", ".bcif", ".pdb", ".ent")

_ASSEMBLY_SUFFIX = re.compile(r"-assembly\d+$", re.IGNORECASE)
_CHAIN_ID = re.compile(r"[A-Za-z0-9.]+")


class StructuralTemplateToolMissing(RuntimeError):
    """A tool this path needs is not installed, or its weights are not present.

    Raised instead of letting ``FileNotFoundError`` or a Hugging Face download
    attempt escape, because the fix is always an installation step and the user
    should be told which one.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class StructureDatabaseSpec:
    """A local Foldseek database and its immutable cache identity."""

    name: str
    path: Path
    identifier: str


@dataclasses.dataclass(frozen=True, slots=True)
class EsmfoldSettings:
    """Where the ESMFold weights are and how to run them."""

    model_dir: Path
    device: str = "cuda"
    # ESMFold's memory use grows with the square of the sequence length. The
    # trunk can be told to process the pair representation in chunks, trading
    # time for memory; None leaves the model's own default in place.
    chunk_size: int | None = None
    # A refusal with a clear message beats an out-of-memory kill on a GPU node.
    max_sequence_length: int = 1500


@dataclasses.dataclass(frozen=True, slots=True)
class FoldseekSearchSettings:
    """Everything the Foldseek stage needs, including its cache location."""

    database: StructureDatabaseSpec
    temp_dir: Path
    cache_dir: Path
    e_value: float = 1e-3
    # Hits Foldseek may return. The featuriser keeps far fewer, but it discards
    # hits that fail its prefilters, so the search has to over-supply.
    max_hits: int = 100
    min_alignment_tm_score: float = 0.0
    alignment_type: int = ALIGNMENT_TYPE_3DI_AA
    threads: int = 8


class StructurePredictor(Protocol):
    """Predicting one structure for one sequence."""

    def identity(self) -> str:
        """Stable identity of the predictor; cached structures depend on it."""

    def predict(self, sequence: str) -> str:
        """Return the predicted structure as PDB-format text."""


class FoldseekProcess(Protocol):
    """The operations performed by the external Foldseek executable."""

    def identity(self) -> str: ...

    def search(
        self, query_structure: Path, settings: FoldseekSearchSettings
    ) -> str: ...


class SubprocessFoldseekProcess:
    """Production adapter for one local Foldseek executable."""

    def __init__(self, binary_path: str | Path | None):
        self._binary_path = str(binary_path) if binary_path else ""

    def _run(self, command: Sequence[str]) -> str:
        if not self._binary_path:
            raise StructuralTemplateToolMissing(
                "No Foldseek executable is configured. Install Foldseek "
                "(https://github.com/steineggerlab/foldseek) and pass "
                "--foldseek_binary_path, or put 'foldseek' on PATH."
            )
        try:
            completed = subprocess.run(
                [self._binary_path, *command],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise StructuralTemplateToolMissing(
                f"Foldseek executable not found at {self._binary_path!r}. Install "
                "Foldseek (https://github.com/steineggerlab/foldseek) and pass "
                "--foldseek_binary_path."
            ) from exc
        except OSError as exc:
            raise StructuralTemplateToolMissing(
                f"Foldseek executable at {self._binary_path!r} cannot be run: {exc}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                f"Foldseek command failed: {' '.join(command)}\n{stderr}"
            ) from exc
        return completed.stdout.strip()

    def identity(self) -> str:
        """The executable's own version string, recorded in every cache entry."""
        version = " ".join(self._run(("version",)).split())
        if not version:
            raise RuntimeError("Foldseek version command returned no identity")
        return f"foldseek {version}"

    def search(self, query_structure: Path, settings: FoldseekSearchSettings) -> str:
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="alphapulldown_foldseek_", dir=settings.temp_dir
        ) as temporary_directory:
            root = Path(temporary_directory)
            alignments = root / "alignments.m8"
            work_dir = root / "work"
            work_dir.mkdir()
            self._run(
                (
                    "easy-search",
                    str(query_structure),
                    str(settings.database.path),
                    str(alignments),
                    str(work_dir),
                    "--format-output",
                    ",".join(FOLDSEEK_OUTPUT_COLUMNS),
                    "-e",
                    str(settings.e_value),
                    "--max-seqs",
                    str(settings.max_hits),
                    "--alignment-type",
                    str(settings.alignment_type),
                    "--threads",
                    str(settings.threads),
                )
            )
            try:
                return alignments.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(
                    f"Foldseek produced no alignment file at {alignments}: {exc}"
                ) from exc


class EsmfoldStructurePredictor:
    """ESMFold loaded from a local weights directory, held for the process life.

    The weights are several gigabytes and loading them dominates a single
    prediction, so the loaded model is kept on the instance and reused for every
    sequence in a run.
    """

    def __init__(self, settings: EsmfoldSettings):
        self._settings = settings
        self._torch = None
        self._model = None
        self._tokenizer = None

    def identity(self) -> str:
        """Identify the weights without paying to load them.

        The directory name alone would not notice a re-downloaded or partially
        copied checkpoint, so a cheap content-derived witness goes in alongside
        it, the same way the MMseqs2 path witnesses a database by its index size.
        """
        model_dir = Path(self._settings.model_dir)
        weights = sorted(
            path
            for pattern in ("*.safetensors", "*.bin", "*.pt")
            for path in model_dir.glob(pattern)
        )
        fingerprint = ":".join(
            f"{path.name}={path.stat().st_size}" for path in weights if path.is_file()
        )
        if not fingerprint:
            raise StructuralTemplateToolMissing(
                f"No ESMFold weight files found in {model_dir}. Download the "
                "ESMFold checkpoint (for example facebook/esmfold_v1) into that "
                "directory and pass it with --esmfold_model_dir."
            )
        return f"esmfold:{model_dir.name}:{_short_digest(fingerprint)}"

    def predict(self, sequence: str) -> str:
        sequence = sequence.strip().upper()
        _validate_protein_sequence(sequence)
        if len(sequence) > self._settings.max_sequence_length:
            raise ValueError(
                f"Sequence of {len(sequence)} residues exceeds the configured "
                f"ESMFold limit of {self._settings.max_sequence_length}. Raise "
                "--esmfold_max_sequence_length if the GPU can take it."
            )
        torch, model, tokenizer = self._loaded()
        encoded = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )
        encoded = {
            key: value.to(self._settings.device) for key, value in encoded.items()
        }
        with torch.no_grad():
            output = model(**encoded)
        structures = model.output_to_pdb(output)
        if not structures:
            raise RuntimeError("ESMFold returned no structure for the query sequence")
        return structures[0]

    def _loaded(self):
        if self._model is not None:
            return self._torch, self._model, self._tokenizer
        try:
            import torch
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError as exc:
            raise StructuralTemplateToolMissing(
                "ESMFold needs PyTorch and transformers, which AlphaPulldown does "
                "not install by default. Install them into the environment "
                "(pip install 'torch' 'transformers') before using "
                "--use_foldseek_templates."
            ) from exc

        model_dir = str(self._settings.model_dir)
        try:
            # local_files_only keeps a missing checkpoint a local error instead
            # of a silent multi-gigabyte download on a compute node.
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True
            )
            model = EsmForProteinFolding.from_pretrained(
                model_dir, local_files_only=True
            )
        except (OSError, ValueError) as exc:
            raise StructuralTemplateToolMissing(
                f"Cannot load ESMFold weights from {model_dir}: {exc}. Download "
                "the checkpoint (for example facebook/esmfold_v1) into that "
                "directory first."
            ) from exc

        model = model.to(self._settings.device)
        model.eval()
        if self._settings.chunk_size is not None:
            model.trunk.set_chunk_size(self._settings.chunk_size)
        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        return torch, model, tokenizer


@dataclasses.dataclass(frozen=True, slots=True)
class StructuralTemplateHit:
    """One Foldseek alignment, in the terms AlphaFold 2's featuriser needs.

    Kept separate from ``parsers.TemplateHit`` so that parsing, filtering and
    naming can be tested without importing AlphaFold.
    """

    index: int
    name: str
    query_alignment: str
    hit_alignment: str
    query_start: int
    hit_start: int
    score: float
    e_value: float
    alignment_tm_score: float | None

    @property
    def aligned_cols(self) -> int:
        """Columns where query and template both have a residue."""
        return sum(
            1
            for query_residue, hit_residue in zip(
                self.query_alignment, self.hit_alignment
            )
            if query_residue != "-" and hit_residue != "-"
        )

    def to_template_hit(self):
        """Convert to the AlphaFold 2 hit the template featuriser consumes."""
        from alphafold.data import parsers

        indices_query, indices_hit = alignment_indices(
            self.query_alignment,
            self.hit_alignment,
            query_start=self.query_start,
            hit_start=self.hit_start,
        )
        return parsers.TemplateHit(
            index=self.index,
            name=self.name,
            aligned_cols=self.aligned_cols,
            # Only ever used to rank hits and to fill template_sum_probs. The
            # Foldseek bit score orders hits the same way its own output does.
            sum_probs=self.score,
            query=self.query_alignment,
            hit_sequence=self.hit_alignment,
            indices_query=indices_query,
            indices_hit=indices_hit,
        )


def alignment_indices(
    query_alignment: str,
    hit_alignment: str,
    *,
    query_start: int,
    hit_start: int,
) -> tuple[list[int], list[int]]:
    """Per-column residue indices, with -1 for a gap, as AlphaFold 2 expects.

    Foldseek reports 1-based inclusive alignment starts; AlphaFold 2 works in
    0-based indices, so the starts are shifted by one here and nowhere else.
    """
    if len(query_alignment) != len(hit_alignment):
        raise ValueError(
            "Query and template alignment rows differ in length "
            f"({len(query_alignment)} vs {len(hit_alignment)})"
        )
    if query_start < 1 or hit_start < 1:
        raise ValueError("Alignment starts are 1-based and must be positive")

    indices_query: list[int] = []
    indices_hit: list[int] = []
    query_index = query_start - 1
    hit_index = hit_start - 1
    for query_residue, hit_residue in zip(query_alignment, hit_alignment):
        if query_residue == "-":
            indices_query.append(-1)
        else:
            indices_query.append(query_index)
            query_index += 1
        if hit_residue == "-":
            indices_hit.append(-1)
        else:
            indices_hit.append(hit_index)
            hit_index += 1
    return indices_query, indices_hit


def pdb_chain_name(target: str) -> str | None:
    """Normalise a Foldseek target name to ``<pdbid>_<chain>``, or None.

    AlphaFold 2 resolves a hit by reading ``<pdbid>.cif`` out of the mmCIF
    directory, and it parses that identifier straight off the front of the hit
    name. Foldseek, meanwhile, names targets after the file the database was
    built from -- ``1abc.cif.gz_A``, ``pdb1abc.ent.gz_A``,
    ``1abc-assembly1.cif.gz_A`` -- so the name has to be reduced before the hit
    can be handed over. Anything that does not reduce to a PDB chain (an AFDB
    model, say) has no mmCIF file to read and returns None so the caller can
    skip it with a word rather than fail deep inside the featuriser.
    """
    token = target.split()[0] if target.split() else ""
    token = token.rsplit("/", 1)[-1]
    stem, separator, chain = token.rpartition("_")
    if not separator or not chain or not _CHAIN_ID.fullmatch(chain):
        return None

    trimmed = True
    while trimmed:
        trimmed = False
        for suffix in _STRUCTURE_FILE_SUFFIXES:
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                trimmed = True
    stem = _ASSEMBLY_SUFFIX.sub("", stem)
    # RCSB's own PDB-format distribution names files pdb<id>.ent.
    if len(stem) == 7 and stem.lower().startswith("pdb"):
        stem = stem[3:]
    if len(stem) != 4 or not stem.isalnum():
        return None
    return f"{stem.lower()}_{chain}"


def query_sequence_from_a3m(a3m: str) -> str:
    """The query sequence of an A3M alignment: its first record, ungapped."""
    started = False
    parts: list[str] = []
    for line in a3m.splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
        elif started and line.strip():
            parts.append(line.strip())
    if not started or not parts:
        raise ValueError("A3M alignment contains no query sequence")
    query = "".join(parts).replace("-", "").replace(".", "").upper()
    _validate_protein_sequence(query)
    return query


def parse_foldseek_alignments(
    output: str,
    query_sequence: str,
    *,
    min_alignment_tm_score: float = 0.0,
    columns: Sequence[str] = FOLDSEEK_OUTPUT_COLUMNS,
) -> tuple[StructuralTemplateHit, ...]:
    """Turn Foldseek tabular output into hits AlphaFold 2 can featurise.

    A malformed or unusable row is dropped with a warning rather than failing the
    whole search: one unreadable target should not cost a query its remaining
    templates.
    """
    field_index = {name: position for position, name in enumerate(columns)}
    query_sequence = query_sequence.upper()
    hits: list[StructuralTemplateHit] = []
    for line in output.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        # Foldseek can be asked for a header row; the query column of a real row
        # holds the query structure file name, which is never literally "query".
        if fields[0] == "query" and len(fields) > 1 and fields[1] == "target":
            continue
        if len(fields) != len(columns):
            logging.warning(
                "Ignoring a Foldseek row with %d fields, expected %d: %.120s",
                len(fields),
                len(columns),
                line,
            )
            continue

        target = fields[field_index["target"]]
        name = pdb_chain_name(target)
        if name is None:
            logging.warning(
                "Ignoring Foldseek hit %r: it does not name a PDB chain, so no "
                "mmCIF file can be read for it.",
                target,
            )
            continue

        try:
            query_start = int(fields[field_index["qstart"]])
            hit_start = int(fields[field_index["tstart"]])
            query_end = int(fields[field_index["qend"]])
            score = float(fields[field_index["bits"]])
            e_value = float(fields[field_index["evalue"]])
        except ValueError:
            logging.warning(
                "Ignoring a Foldseek row with unreadable numbers: %.120s", line
            )
            continue

        query_alignment = fields[field_index["qaln"]].upper()
        hit_alignment = fields[field_index["taln"]].upper()
        if len(query_alignment) != len(hit_alignment) or not query_alignment:
            logging.warning(
                "Ignoring Foldseek hit %r: its alignment rows are unusable.", target
            )
            continue

        # The featuriser locates the aligned region by searching for it in the
        # query sequence, so a row that does not correspond to this query would
        # be mapped to the wrong residues instead of rejected.
        aligned_query = query_alignment.replace("-", "")
        if query_sequence[query_start - 1 : query_end] != aligned_query:
            logging.warning(
                "Ignoring Foldseek hit %r: its query alignment does not match the "
                "query sequence at residues %d-%d.",
                target,
                query_start,
                query_end,
            )
            continue

        alignment_tm_score = _optional_float(fields[field_index["alntmscore"]])
        if (
            min_alignment_tm_score > 0.0
            and alignment_tm_score is not None
            and alignment_tm_score < min_alignment_tm_score
        ):
            continue

        hits.append(
            StructuralTemplateHit(
                index=len(hits),
                name=name,
                query_alignment=query_alignment,
                hit_alignment=hit_alignment,
                query_start=query_start,
                hit_start=hit_start,
                score=score,
                e_value=e_value,
                alignment_tm_score=alignment_tm_score,
            )
        )
    return tuple(hits)


class PredictedStructureCache:
    """Predicted query structures, published durably and keyed by sequence.

    Folding is the only step here that wants a GPU, and it depends on nothing but
    the sequence and the weights. Keeping it behind its own cache means it can be
    run once, ahead of time, on a GPU node -- and that re-searching a refreshed
    Foldseek database later costs a Foldseek run rather than a GPU run.
    """

    def __init__(self, *, cache_dir: Path, predictor: StructurePredictor):
        self._cache_dir = Path(cache_dir)
        self._predictor = predictor
        self._identity: str | None = None

    def signature(self) -> dict[str, Any]:
        """What a cached structure has to have been produced by to be reused."""
        return {"schema_version": 1, "predictor": self.predictor_identity()}

    def predictor_identity(self) -> str:
        if self._identity is None:
            identity = self._predictor.identity().strip()
            if not identity:
                raise ValueError("Structure predictor identity must not be empty")
            self._identity = identity
        return self._identity

    def structure(self, sequence: str) -> str:
        """The predicted structure for one sequence, folding it if necessary."""
        sequence = sequence.strip().upper()
        _validate_protein_sequence(sequence)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path(sequence)
        cached = _read_cached(path, sequence, self.signature())
        if cached is not None:
            logging.info("Reusing a cached predicted structure for the query sequence")
            return cached["structure"]
        structure = self._predictor.predict(sequence)
        if not structure.strip():
            raise RuntimeError("The structure predictor returned an empty structure")
        write_atomic_json(
            path,
            {
                "schemaVersion": 1,
                "sequence": sequence,
                "structure": structure,
                "provenance": self.signature(),
            },
        )
        return structure

    def cached(self, sequence: str) -> bool:
        """Whether a reusable structure for this sequence is already published."""
        sequence = sequence.strip().upper()
        return _read_cached(self.path(sequence), sequence, self.signature()) is not None

    def path(self, sequence: str) -> Path:
        """Where the structure for one sequence is cached, folded or not yet."""
        digest = _short_digest(sequence.strip().upper())
        return self._cache_dir / f"{digest}_esmfold.json"


class FoldseekTemplateSearcher:
    """AlphaFold 2's template-searcher interface, backed by ESMFold and Foldseek.

    ``input_format``, ``output_format``, ``query`` and ``get_template_hits`` are
    the four things AlphaFold 2's data pipeline asks of a template searcher, so
    implementing them is all it takes to substitute this for hmmsearch or
    HHsearch. The pipeline hands ``query`` the uniref90 alignment; only its first
    row is used, because a fold is predicted from the query sequence alone.
    """

    def __init__(
        self,
        *,
        settings: FoldseekSearchSettings,
        structures: PredictedStructureCache,
        foldseek_process: FoldseekProcess,
    ):
        if not isinstance(settings, FoldseekSearchSettings):
            raise TypeError(
                "FoldseekTemplateSearcher settings must be FoldseekSearchSettings"
            )
        _validate_search_settings(settings)
        self._settings = settings
        self._structures = structures
        self._foldseek = foldseek_process
        self._foldseek_identity: str | None = None

    @property
    def input_format(self) -> str:
        return "a3m"

    @property
    def output_format(self) -> str:
        return "m8"

    def query(self, a3m: str) -> str:
        """Return Foldseek tabular output for the alignment's query sequence."""
        sequence = query_sequence_from_a3m(a3m)
        self._settings.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = _read_cached(
            self._alignment_path(sequence), sequence, self._alignment_signature()
        )
        if cached is not None:
            logging.info("Reusing cached Foldseek alignments for the query sequence")
            return cached["alignments"]

        structure = self._structures.structure(sequence)
        self._settings.temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="alphapulldown_foldseek_query_", dir=self._settings.temp_dir
        ) as temporary_directory:
            # Named after the sequence digest so a stray file in a shared scratch
            # directory can be traced back, and so no target can be mistaken for
            # a header row.
            query_structure = (
                Path(temporary_directory) / f"{_short_digest(sequence)}.pdb"
            )
            query_structure.write_text(structure, encoding="utf-8")
            alignments = self._foldseek.search(query_structure, self._settings)

        write_atomic_json(
            self._alignment_path(sequence),
            {
                "schemaVersion": 1,
                "sequence": sequence,
                "alignments": alignments,
                "provenance": self._alignment_signature(),
            },
        )
        return alignments

    def get_template_hits(self, output_string: str, input_sequence: str):
        """Parsed AlphaFold 2 template hits for one Foldseek output."""
        hits = parse_foldseek_alignments(
            output_string,
            input_sequence,
            min_alignment_tm_score=self._settings.min_alignment_tm_score,
        )
        if not hits:
            logging.warning(
                "Foldseek returned no usable structural template hits for this query."
            )
        return [hit.to_template_hit() for hit in hits]

    def provenance(self) -> dict[str, Any]:
        """Complete identity of this search, for feature metadata."""
        return self._alignment_signature()

    def _alignment_signature(self) -> dict[str, Any]:
        database = self._settings.database
        return {
            "schema_version": 1,
            "predictor": self._structures.predictor_identity(),
            "foldseek": self._foldseek_id(),
            "database": {
                "name": database.name,
                "identifier": database.identifier,
                # `identifier` is operator-supplied and cannot notice a database
                # rebuilt or half-copied under the same name; the index size is a
                # cheap content-derived witness that changes when it does.
                "index_size": _database_index_size(database.path),
            },
            "e_value": self._settings.e_value,
            "max_hits": self._settings.max_hits,
            "alignment_type": self._settings.alignment_type,
            "min_alignment_tm_score": self._settings.min_alignment_tm_score,
            "columns": list(FOLDSEEK_OUTPUT_COLUMNS),
        }

    def _foldseek_id(self) -> str:
        if self._foldseek_identity is None:
            identity = self._foldseek.identity().strip()
            if not identity:
                raise ValueError("Foldseek identity must not be empty")
            self._foldseek_identity = identity
        return self._foldseek_identity

    def _alignment_path(self, sequence: str) -> Path:
        return self._settings.cache_dir / f"{_short_digest(sequence)}_foldseek.json"


def _validate_search_settings(settings: FoldseekSearchSettings) -> None:
    if not str(settings.database.path):
        raise ValueError("The Foldseek database requires an explicit path")
    if not settings.database.identifier.strip():
        raise ValueError("The Foldseek database requires a non-empty identifier")
    if settings.e_value <= 0:
        raise ValueError("e_value must be greater than 0")
    if settings.max_hits < 1:
        raise ValueError("max_hits must be at least 1")
    if settings.threads < 1:
        raise ValueError("threads must be at least 1")
    if not 0.0 <= settings.min_alignment_tm_score <= 1.0:
        raise ValueError("min_alignment_tm_score must lie between 0 and 1")
    if settings.alignment_type not in (ALIGNMENT_TYPE_TMALIGN, ALIGNMENT_TYPE_3DI_AA):
        raise ValueError(
            "alignment_type must be "
            f"{ALIGNMENT_TYPE_TMALIGN} (TMalign) or {ALIGNMENT_TYPE_3DI_AA} (3Di+AA)"
        )


def _read_cached(
    path: Path, sequence: str, signature: Mapping[str, Any]
) -> dict[str, Any] | None:
    """A cache entry for this sequence and these settings, or None."""
    if not path.exists():
        return None
    try:
        with open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        if payload.get("sequence") != sequence:
            return None
        if payload.get("provenance") != dict(signature):
            return None
        return payload
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _validate_protein_sequence(sequence: str) -> None:
    if not sequence or set(sequence.upper()) - _PROTEIN_RESIDUES:
        raise ValueError(
            "Structural template search needs a protein sequence; got "
            f"{sequence[:40]!r}"
        )


def _optional_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed


def _short_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _database_index_size(path: Path) -> int:
    """Size of a Foldseek database index, or 0 when it cannot be read."""
    try:
        return Path(f"{path}.index").stat().st_size
    except OSError:
        return 0
