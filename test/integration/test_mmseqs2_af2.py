"""Local MMseqs2 to AlphaFold 2 features, end to end, with nothing faked.

A real MMseqs2 search over a small UniProt-headed database, through the real
``create_batch_msas.py``; then the real ``finalize_batch_features.py
--data_pipeline=alphafold2``, running real hmmsearch against a PDB seqres whose
entry is backed by a real mmCIF (3L4Q, influenza NS1 bound to p85beta). The
pickle it publishes is then loaded and checked for what the two stages exist to
deliver: a real template, recovered insertions, and parseable species.

Opt-in: needs MMSEQS_INTEGRATION_BINARY and the AlphaFold 2 template tools.
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.external_tools]

pytest.importorskip("alphafold.data.pipeline", reason="needs AlphaFold 2")

TEMPLATES = Path(__file__).resolve().parents[1] / "test_data" / "templates"

# 3L4Q chain A without its His tag: the NS1 effector domain.
NS1 = (
    "SDEALKMTMASVPASRYLTDMTLEEMSRDWSMLIPKQKVAGPLCIRMDQAIMDKNIILKANFSVIFDRLETLIL"
    "LRAFTEEGAIVGEISPLPSLPGHTAEDVKNAVGVLIGGLEWNDNTVRVSETLQRFAWRSSNENGRPPLTPKQ"
    "KREMAGTIRSEV"
)
NS1_WITH_TAG = "HHHHHH" + NS1
P85B = (
    "YQQDQIVKEDSVEAVGAQLKVYHQQYQDKSREYDQLYEEYTRTSQELQMKRTAIEAFNETIKIFEEQGQTQEKSS"
    "KEYLERFRREGNEKEMQRILLNSERLKSRIAEIHESRTKLEQELRAQASDNREIDKRMNSLKPDLMQLRKIRDQ"
    "YLVWLTQKGARQKKINEWLGI"
)
SPECIES = ("HUMAN", "MOUSE", "CHICK", "PIG", "BOVIN", "RAT")


def _mutated(sequence: str, count: int, seed: int) -> str:
    rng = random.Random(seed)
    residues = list(sequence)
    for position in rng.sample(range(len(residues)), count):
        residues[position] = "W" if residues[position] != "W" else "F"
    return "".join(residues)


# The query is an NS1 HOMOLOG, about 90% identical, not NS1 itself. AlphaFold 2
# rightly discards a template identical to its query -- a DuplicateError it drops
# without so much as a warning -- so a query copied from 3L4Q would find no
# template at all, and that is not the case features exist for.
QUERY = _mutated(NS1, 16, seed=11)


def _tool(name: str) -> str:
    path = shutil.which(name) or str(Path(sys.executable).parent / name)
    if not Path(path).exists():
        pytest.skip(f"{name} is not installed")
    return path


def _run(*command, env=None):
    completed = subprocess.run(
        [str(part) for part in command],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if completed.returncode:
        raise AssertionError(
            f"{command[0]} failed ({completed.returncode}):\n"
            f"{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
        )
    return completed


def _homologs() -> str:
    """Query homologs under UniProt headers: substitutions, and one insertion."""
    rng = random.Random(3)
    residues = "ACDEFGHIKLMNPQRSTVWY"
    records = []
    for index, species in enumerate(SPECIES):
        sequence = list(QUERY)
        for _ in range(8):
            position = rng.randrange(len(sequence))
            sequence[position] = rng.choice(residues)
        if index == 0:
            # Well inside the domain, so the alignment has to open an insertion.
            sequence[60:60] = list("WWWWW")
        accession = f"P{index:05d}"
        records.append(
            f">sp|{accession}|NS1_{species} Non-structural protein 1 "
            f"OS=Influenza OX={1000 + index}\n{''.join(sequence)}\n"
        )
    return "".join(records)


@pytest.fixture
def mmseqs_binary() -> Path:
    configured = os.environ.get("MMSEQS_INTEGRATION_BINARY")
    if not configured:
        pytest.skip("set MMSEQS_INTEGRATION_BINARY to run against real MMseqs2")
    binary = Path(configured)
    if not binary.is_file():
        pytest.fail(f"MMSEQS_INTEGRATION_BINARY does not exist: {binary}")
    return binary


def test_local_mmseqs2_to_alphafold2_features_end_to_end(tmp_path, mmseqs_binary):
    hmmsearch, hmmbuild, kalign = (_tool(n) for n in ("hmmsearch", "hmmbuild", "kalign"))
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}

    # A padded MMseqs2 database standing in for all four protein databases.
    target_fasta = tmp_path / "homologs.fasta"
    target_fasta.write_text(_homologs(), encoding="utf-8")
    target_db, padded_db = tmp_path / "homologs", tmp_path / "homologs_gpu"
    _run(mmseqs_binary, "createdb", target_fasta, target_db, "--threads", "1")
    _run(mmseqs_binary, "makepaddedseqdb", target_db, padded_db, "--threads", "1")

    # A PDB seqres whose NS1 entry is backed by the real 3L4Q mmCIF.
    template_root = tmp_path / "pdb"
    mmcif_dir = template_root / "mmcif_files"
    mmcif_dir.mkdir(parents=True)
    shutil.copy(TEMPLATES / "3L4Q.cif", mmcif_dir / "3l4q.cif")
    seqres = template_root / "pdb_seqres.txt"
    seqres.write_text(
        f">3l4q_A mol:protein length:{len(NS1_WITH_TAG)}  NS1\n{NS1_WITH_TAG}\n"
        f">3l4q_C mol:protein length:{len(P85B)}  P85B\n{P85B}\n",
        encoding="utf-8",
    )
    obsolete = template_root / "obsolete.dat"
    obsolete.write_text("", encoding="utf-8")

    query_fasta = tmp_path / "ns1.fasta"
    query_fasta.write_text(f">ns1\n{QUERY}\n", encoding="utf-8")
    msa_dir = tmp_path / "msas"
    database_flags = [
        flag
        for name in ("uniref90", "mgnify", "small_bfd", "uniprot")
        for flag in (
            f"--mmseqs_{name}_database_path={padded_db}",
            f"--mmseqs_{name}_database_id=homologs-v1",
        )
    ]

    _run(
        sys.executable, "-m", "alphapulldown.scripts.create_batch_msas",
        f"--fasta_paths={query_fasta}",
        f"--msa_output_dir={msa_dir}",
        f"--summary_path={tmp_path / 'summary.json'}",
        f"--mmseqs_binary_path={mmseqs_binary}",
        f"--mmseqs_temp_dir={tmp_path / 'work'}",
        "--mmseqs_batch_max_sequences=4",
        "--mmseqs_batch_max_residues=10000",
        "--mmseqs_threads=2",
        f"--mmseqs_use_gpu={'true' if os.environ.get('MMSEQS_INTEGRATION_GPU') == '1' else 'false'}",
        *database_flags,
        env=env,
    )
    assert (msa_dir / "ns1_mmseqs_msa.json").exists()

    features_dir = tmp_path / "features"
    _run(
        sys.executable, "-m", "alphapulldown.scripts.finalize_batch_features",
        "--data_pipeline=alphafold2",
        f"--fasta_paths={query_fasta}",
        f"--msa_input_dir={msa_dir}",
        f"--output_dir={features_dir}",
        f"--data_dir={tmp_path}",
        "--max_template_date=2050-01-01",
        "--template_seqres_database_id=seqres-3l4q",
        "--template_mmcif_database_id=mmcif-3l4q",
        f"--pdb_seqres_database_path={seqres}",
        f"--template_mmcif_dir={mmcif_dir}",
        f"--obsolete_pdbs_path={obsolete}",
        f"--hmmsearch_binary_path={hmmsearch}",
        f"--hmmbuild_binary_path={hmmbuild}",
        f"--kalign_binary_path={kalign}",
        env=env,
    )

    with open(features_dir / "ns1.pkl", "rb") as handle:
        monomer = pickle.load(handle)
    features = monomer.feature_dict

    assert type(monomer).__module__ == "alphapulldown.objects"
    assert monomer.sequence == QUERY

    # A real template, found by real hmmsearch from the uniref90 profile.
    assert b"3l4q_A" in list(features["template_domain_names"])
    assert features["template_aatype"].shape[1] == len(QUERY)

    # The inserted homolog's five residues reach the deletion matrix.
    assert features["deletion_matrix_int"].sum() >= 5

    # Species parsed from the UniProt headers, for pairing.
    species = {value.decode() for value in features["msa_species_identifiers_all_seq"]}
    assert set(SPECIES) <= species

    # Provenance records what built it, and not the tools that did not run.
    metadata = next(features_dir.glob("ns1_feature_metadata_*.json")).read_text()
    assert "jackhmmer" not in metadata and "hhblits" not in metadata
    assert '"msa_backend": "mmseqs2-gpu"' in metadata
