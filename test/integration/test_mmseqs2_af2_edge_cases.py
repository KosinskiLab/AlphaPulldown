"""Edge cases for local MMseqs2 AlphaFold 2 features, end to end, nothing faked.

Four separate MMseqs2 databases stand in for UniRef90, MGnify, small BFD and
UniProt, so every stage and every cap is exercised on its own database. One real
pass through ``create_batch_msas.py`` and ``finalize_batch_features.py
--data_pipeline=alphafold2`` covers all the cases at once, as a workflow shard
would; each test then checks one case's pickle.

The cases are the ones that fail silently when wrong: a cap that never bites, an
orphan chain whose empty alignment breaks a later stage, a template search that
finds nothing, residues the pipeline does not know, headers with no species.

``build_edge_case_features`` is importable, so the same features can be written
somewhere durable and folded on a GPU.

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

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.external_tools]

TEMPLATES = Path(__file__).resolve().parents[1] / "test_data" / "templates"
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

# 3L4Q chain A, NS1 effector domain, His tag removed.
NS1 = (
    "SDEALKMTMASVPASRYLTDMTLEEMSRDWSMLIPKQKVAGPLCIRMDQAIMDKNIILKANFSVIFDRLETLIL"
    "LRAFTEEGAIVGEISPLPSLPGHTAEDVKNAVGVLIGGLEWNDNTVRVSETLQRFAWRSSNENGRPPLTPKQ"
    "KREMAGTIRSEV"
)
MGNIFY_HITS = 600          # past AlphaFold 2's 501 cap, which counts the query
AF2_MGNIFY_CAP = 501
SPECIES = ("HUMAN", "MOUSE", "RAT", "BOVIN", "PIG", "CHICK", "DANRE", "XENLA")


def _random_protein(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(RESIDUES) for _ in range(length))


def _homolog(rng: random.Random, sequence: str, fraction: float = 0.15) -> str:
    """About 85% identical: found easily, never mistaken for the query itself."""
    residues = list(sequence)
    for position in rng.sample(range(len(residues)), max(1, int(len(residues) * fraction))):
        residues[position] = rng.choice(RESIDUES.replace(residues[position], ""))
    return "".join(residues)


def _mutated(sequence: str, count: int, seed: int) -> str:
    rng = random.Random(seed)
    residues = list(sequence)
    for position in rng.sample(range(len(residues)), count):
        residues[position] = "W" if residues[position] != "W" else "F"
    return "".join(residues)


def _cases():
    """Query sequences, each unrelated to the others, and where their hits live."""
    rng = random.Random(20260911)
    deep = _random_protein(rng, 120)
    x_query = list(_random_protein(rng, 110))
    for position in (10, 55, 90):
        x_query[position] = "X"
    return {
        "deep": deep,
        "deep_copy": deep,                     # same sequence, second name
        "shallow": _random_protein(rng, 100),
        "orphan": _random_protein(rng, 90),
        "notemplate": _random_protein(rng, 130),
        "templated": _mutated(NS1, 16, seed=11),   # NS1 homolog: 3L4Q is a template
        "xresidues": "".join(x_query),
        "nospecies": _random_protein(rng, 115),
        "peptide": _random_protein(rng, 20),
    }


def _databases(cases):
    """FASTA text per database: which hits each case gets, and where."""
    rng = random.Random(7)
    uniref90, mgnify, small_bfd, uniprot = [], [], [], []

    def uniref(name, sequence, count):
        for index in range(count):
            uniref90.append((f"UniRef90_U{name}{index:04d} {name} homolog",
                             _homolog(rng, sequence)))

    def uniprot_hits(name, sequence, count, *, with_species=True):
        for index in range(count):
            accession = f"Q{abs(hash((name, index))) % 10**5:05d}"
            header = (
                f"sp|{accession}|{name.upper()[:5]}_{SPECIES[index % len(SPECIES)]} "
                f"{name} OS=Organism OX={9000 + index}"
                if with_species
                # A UniParc-style header: no species anywhere in it.
                else f"UPI{index:010d} {name} unassigned"
            )
            uniprot.append((header, _homolog(rng, sequence)))

    uniref("deep", cases["deep"], 30)
    for index in range(MGNIFY_HITS):
        mgnify.append((f"MGYP{index:012d}", _homolog(rng, cases["deep"])))
    for index in range(20):
        small_bfd.append((f"BFD_deep_{index:04d}", _homolog(rng, cases["deep"])))
    uniprot_hits("deep", cases["deep"], 8)

    uniref("shallow", cases["shallow"], 1)
    uniref("notemplate", cases["notemplate"], 12)
    uniprot_hits("notemplate", cases["notemplate"], 4)
    uniref("templated", cases["templated"], 12)
    uniprot_hits("templated", cases["templated"], 6)
    uniref("xresidues", cases["xresidues"], 10)
    uniprot_hits("xresidues", cases["xresidues"], 3)
    uniref("nospecies", cases["nospecies"], 6)
    uniprot_hits("nospecies", cases["nospecies"], 5, with_species=False)
    uniref("peptide", cases["peptide"], 3)
    # "orphan" gets nothing, anywhere.
    return {"uniref90": uniref90, "mgnify": mgnify, "small_bfd": small_bfd,
            "uniprot": uniprot}


def _run(*command, env=None):
    completed = subprocess.run(
        [str(part) for part in command], capture_output=True, text=True, env=env,
        timeout=1200,
    )
    if completed.returncode:
        raise AssertionError(
            f"{command[:4]} failed ({completed.returncode}):\n"
            f"{completed.stdout[-3000:]}\n{completed.stderr[-4000:]}"
        )
    return completed


def build_edge_case_features(workdir: Path, mmseqs: Path, *, tools: dict, python=None,
                             use_gpu: bool = False) -> dict[str, Path]:
    """Build every case's AlphaFold 2 pickle through the real CLIs; return the paths."""
    python = python or sys.executable
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    cases = _cases()
    workdir.mkdir(parents=True, exist_ok=True)

    database_flags = []
    for name, records in _databases(cases).items():
        fasta = workdir / f"{name}.fasta"
        fasta.write_text("".join(f">{h}\n{s}\n" for h, s in records), encoding="utf-8")
        plain, padded = workdir / f"{name}_db", workdir / f"{name}_gpu"
        _run(mmseqs, "createdb", fasta, plain, "--threads", "2")
        _run(mmseqs, "makepaddedseqdb", plain, padded, "--threads", "2")
        database_flags += [f"--mmseqs_{name}_database_path={padded}",
                           f"--mmseqs_{name}_database_id={name}-edge-v1",
                           # High enough that AlphaFold 2's own cap is what bites.
                           f"--mmseqs_{name}_max_sequences=2000"]

    template_root = workdir / "pdb"
    mmcif_dir = template_root / "mmcif_files"
    mmcif_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATES / "3L4Q.cif", mmcif_dir / "3l4q.cif")
    seqres = template_root / "pdb_seqres.txt"
    seqres.write_text(f">3l4q_A mol:protein length:{len(NS1) + 6}  NS1\nHHHHHH{NS1}\n",
                      encoding="utf-8")
    (template_root / "obsolete.dat").write_text("", encoding="utf-8")

    queries = workdir / "queries.fasta"
    queries.write_text("".join(f">{n}\n{s}\n" for n, s in cases.items()), encoding="utf-8")
    msa_dir, features_dir = workdir / "msas", workdir / "features"

    _run(python, "-m", "alphapulldown.scripts.create_batch_msas",
         f"--fasta_paths={queries}", f"--msa_output_dir={msa_dir}",
         f"--summary_path={workdir / 'summary.json'}",
         f"--mmseqs_binary_path={mmseqs}", f"--mmseqs_temp_dir={workdir / 'work'}",
         "--mmseqs_batch_max_sequences=32", "--mmseqs_batch_max_residues=100000",
         "--mmseqs_threads=2", f"--mmseqs_use_gpu={'true' if use_gpu else 'false'}",
         *database_flags, env=env)
    _run(python, "-m", "alphapulldown.scripts.finalize_batch_features",
         "--data_pipeline=alphafold2", f"--fasta_paths={queries}",
         f"--msa_input_dir={msa_dir}", f"--output_dir={features_dir}",
         f"--data_dir={workdir}", "--max_template_date=2050-01-01",
         "--template_seqres_database_id=seqres-edge", "--template_mmcif_database_id=mmcif-edge",
         f"--pdb_seqres_database_path={seqres}", f"--template_mmcif_dir={mmcif_dir}",
         f"--obsolete_pdbs_path={template_root / 'obsolete.dat'}",
         f"--hmmsearch_binary_path={tools['hmmsearch']}",
         f"--hmmbuild_binary_path={tools['hmmbuild']}",
         f"--kalign_binary_path={tools['kalign']}", env=env)
    return {name: features_dir / f"{name}.pkl" for name in cases}


def _tool(name: str) -> str:
    path = shutil.which(name) or str(Path(sys.executable).parent / name)
    if not Path(path).exists():
        pytest.skip(f"{name} is not installed")
    return path


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    pytest.importorskip("alphafold.data.pipeline", reason="needs AlphaFold 2")
    configured = os.environ.get("MMSEQS_INTEGRATION_BINARY")
    if not configured:
        pytest.skip("set MMSEQS_INTEGRATION_BINARY to run against real MMseqs2")
    tools = {name: _tool(name) for name in ("hmmsearch", "hmmbuild", "kalign")}
    paths = build_edge_case_features(
        tmp_path_factory.mktemp("edge"), Path(configured), tools=tools,
        use_gpu=os.environ.get("MMSEQS_INTEGRATION_GPU") == "1",
    )
    loaded = {}
    for name, path in paths.items():
        with open(path, "rb") as handle:
            loaded[name] = pickle.load(handle)
    return loaded


def _msa_rows(monomer) -> int:
    return int(monomer.feature_dict["msa"].shape[0])


def test_deep_alignment_hits_alphafold2s_mgnify_cap(features):
    """600 MGnify hits were found; AlphaFold 2 keeps 501 counting the query."""
    deep = features["deep"].feature_dict
    rows = _msa_rows(features["deep"])
    # query + 30 uniref90 + 20 BFD + 500 MGnify, all distinct by construction.
    assert rows == 1 + 30 + 20 + (AF2_MGNIFY_CAP - 1), rows
    assert deep["num_alignments"][0] == rows


def test_the_same_sequence_under_two_names_gets_two_identical_pickles(features):
    np.testing.assert_array_equal(features["deep"].feature_dict["msa"],
                                  features["deep_copy"].feature_dict["msa"])
    assert features["deep_copy"].description == "deep_copy"


def test_a_shallow_alignment_is_kept_as_it_is(features):
    assert _msa_rows(features["shallow"]) == 2


def test_an_orphan_chain_gets_query_only_features(features):
    orphan = features["orphan"].feature_dict
    assert orphan["msa"].shape[0] == 1
    assert orphan["msa_all_seq"].shape[0] == 1
    assert orphan["deletion_matrix_int"].sum() == 0


def test_no_template_gives_alphafold2s_empty_template(features):
    templates = features["notemplate"].feature_dict
    assert list(templates["template_domain_names"]) == [b""]
    assert templates["template_aatype"].shape == (1, len(features["notemplate"].sequence), 22)
    assert not templates["template_all_atom_masks"].any()


def test_a_homolog_of_a_known_structure_finds_its_template(features):
    assert b"3l4q_A" in list(features["templated"].feature_dict["template_domain_names"])


def test_unknown_residues_survive_as_alphafold2s_unknown(features):
    monomer = features["xresidues"]
    aatype = monomer.feature_dict["aatype"].argmax(axis=1)
    unknown = 20  # residue_constants.restype_order_with_x["X"]
    assert [int(aatype[i]) for i in (10, 55, 90)] == [unknown] * 3
    assert _msa_rows(monomer) > 1


def test_pairing_rows_without_species_are_kept_but_unlabelled(features):
    """No species in a UniProt header means no pairing -- not a failure, and not
    a UniProt lookup to recover one."""
    all_seq = features["nospecies"].feature_dict
    assert all_seq["msa_all_seq"].shape[0] == 1 + 5
    assert set(all_seq["msa_species_identifiers_all_seq"]) == {b""}


def test_a_short_peptide_is_featurised(features):
    peptide = features["peptide"]
    assert peptide.feature_dict["aatype"].shape[0] == 20
    assert _msa_rows(peptide) >= 1


def test_every_case_carries_the_same_feature_keys(features):
    keys = {name: frozenset(m.feature_dict) for name, m in features.items()}
    assert len(set(keys.values())) == 1, keys
