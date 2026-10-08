"""The utility scripts' command lines, run as their users run them, on test templates."""

import subprocess
import sys
from pathlib import Path

import pytest
from Bio.PDB import MMCIFParser, PDBParser

REPOSITORY = Path(__file__).resolve().parents[2]
UTILS = REPOSITORY / "alphapulldown" / "utils"
TEMPLATES = REPOSITORY / "test" / "test_data" / "templates"


def _run(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(UTILS / script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result


def test_calculate_rmsd_superposes_a_structure_onto_itself(tmp_path):
    reference = tmp_path / "reference.pdb"
    target = tmp_path / "target.pdb"
    reference.write_bytes((TEMPLATES / "ranked_0.pdb").read_bytes())
    target.write_bytes((TEMPLATES / "ranked_0.pdb").read_bytes())

    result = _run(
        "calculate_rmsd.py", f"--reference_pdb={reference}", f"--target_pdb={target}",
        cwd=tmp_path,
    )

    assert "RMSD between" in result.stderr and ": 0.0000" in result.stderr
    assert (tmp_path / "superposed_reference.pdb").is_file()
    assert (tmp_path / "superposed_target.pdb").is_file()


def test_create_custom_template_db_builds_a_one_template_database(tmp_path):
    out_path = tmp_path / "template_db"

    _run(
        "create_custom_template_db.py",
        f"--out_path={out_path}",
        f"--template={TEMPLATES / '3L4Q.cif'}",
        "--multimeric_chain=A",
        cwd=tmp_path,
    )

    seqres = (out_path / "pdb_seqres.txt").read_text()
    # Entries are named after the template's code; chain A is 3L4Q's NS1 domain.
    assert seqres.startswith(">3l4") and "_A mol:protein" in seqres
    assert "SDEALKMTMASVPASRYLTDMTLEEMSRDWSMLIPKQKVAGPLCIRMDQ" in seqres
    assert list((out_path / "templates").iterdir())


@pytest.mark.parametrize("suffix, parser", [(".cif", MMCIFParser), (".pdb", PDBParser)])
def test_remove_clashes_low_plddt_writes_the_filtered_chain(tmp_path, suffix, parser):
    output = tmp_path / f"filtered{suffix}"

    _run(
        "remove_clashes_low_plddt.py",
        f"--input_file_path={TEMPLATES / 'ranked_0.pdb'}",
        f"--output_file_path={output}",
        "--chain=B",
        "--plddt_threshold=0",
        cwd=tmp_path,
    )

    structure = parser(QUIET=True).get_structure("filtered", str(output))
    assert len(list(structure.get_residues())) > 0
