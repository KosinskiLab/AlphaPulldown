import logging
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

from alphapulldown.utils import calculate_rmsd


def _atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    *,
    bfactor: float = 20.0,
) -> str:
    element = atom_name.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom_name:<4}{' ':1}{res_name:>3} {chain_id:1}"
        f"{residue_number:4d}{' ':1}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{bfactor:6.2f}          {element:>2}\n"
    )


def _write_pdb(path: Path, atoms: list[str]) -> Path:
    path.write_text("".join(atoms) + "TER\nEND\n", encoding="utf-8")
    return path


def _parse_structure(path: Path):
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def test_setup_logging_configures_basic_config(monkeypatch):
    calls = {}

    def fake_basic_config(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    calculate_rmsd.setup_logging()

    assert calls == {
        "format": "%(asctime)s - %(levelname)s: %(message)s",
        "level": logging.INFO,
    }


def test_extract_ca_sequence_skips_missing_ca_and_marks_unknown_residues(tmp_path):
    pdb_path = _write_pdb(
        tmp_path / "sequence.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0),
            _atom_line(3, "N", "UNK", "A", 2, 2.0, 0.0, 0.0),
            _atom_line(4, "CA", "UNK", "A", 2, 3.0, 0.0, 0.0),
            _atom_line(5, "N", "GLY", "A", 3, 4.0, 0.0, 0.0),
        ],
    )

    sequence = calculate_rmsd.extract_ca_sequence(_parse_structure(pdb_path))

    assert sequence == "A-"


def test_align_sequences_returns_global_alignment_for_identical_sequences():
    alignment = calculate_rmsd.align_sequences("ACD", "ACD")

    assert alignment.score == 6
    assert alignment.target == "ACD"
    assert alignment.query == "ACD"


def test_get_common_atoms_returns_only_shared_atom_ids(tmp_path):
    ref_path = _write_pdb(
        tmp_path / "ref_atoms.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0),
            _atom_line(3, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0),
        ],
    )
    target_path = _write_pdb(
        tmp_path / "target_atoms.pdb",
        [
            _atom_line(1, "CA", "ALA", "A", 1, 10.0, 0.0, 0.0),
            _atom_line(2, "C", "ALA", "A", 1, 11.0, 0.0, 0.0),
            _atom_line(3, "O", "ALA", "A", 1, 12.0, 0.0, 0.0),
        ],
    )

    ref_res = next(_parse_structure(ref_path).get_residues())
    target_res = next(_parse_structure(target_path).get_residues())

    common_atoms = calculate_rmsd.get_common_atoms(ref_res, target_res)

    assert [ref_atom.get_id() for ref_atom, _ in common_atoms] == ["CA"]


def test_process_chain_collects_common_atoms_from_matching_residues(tmp_path):
    ref_path = _write_pdb(
        tmp_path / "ref_chain.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0),
            _atom_line(3, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0),
            _atom_line(4, "N", "GLY", "A", 2, 2.0, 0.0, 0.0),
            _atom_line(5, "CA", "GLY", "A", 2, 3.0, 0.0, 0.0),
        ],
    )
    target_path = _write_pdb(
        tmp_path / "target_chain.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 10.0, 0.0, 0.0),
            _atom_line(2, "CA", "ALA", "A", 1, 11.0, 0.0, 0.0),
            _atom_line(3, "CA", "GLY", "A", 2, 13.0, 0.0, 0.0),
        ],
    )

    ref_structure = _parse_structure(ref_path)
    target_structure = _parse_structure(target_path)
    alignment = calculate_rmsd.align_sequences("AG", "AG")

    ref_atoms, target_atoms = calculate_rmsd.process_chain(
        "A",
        ref_structure,
        target_structure,
        alignment,
    )

    assert [atom.get_id() for atom in ref_atoms] == ["N", "CA", "CA"]
    assert [atom.get_id() for atom in target_atoms] == ["N", "CA", "CA"]


def test_calculate_rmsd_and_superpose_returns_rmsd_and_writes_outputs(tmp_path):
    ref_path = _write_pdb(
        tmp_path / "ref.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0),
            _atom_line(3, "C", "ALA", "A", 1, 2.0, 0.0, 0.0),
            _atom_line(4, "N", "GLY", "A", 2, 3.0, 0.0, 0.0),
            _atom_line(5, "CA", "GLY", "A", 2, 4.0, 0.0, 0.0),
            _atom_line(6, "C", "GLY", "A", 2, 5.0, 0.0, 0.0),
        ],
    )
    target_path = _write_pdb(
        tmp_path / "target.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 10.0, 0.0, 0.0),
            _atom_line(2, "CA", "ALA", "A", 1, 11.0, 0.0, 0.0),
            _atom_line(3, "C", "ALA", "A", 1, 12.0, 0.0, 0.0),
            _atom_line(4, "N", "GLY", "A", 2, 13.0, 0.0, 0.0),
            _atom_line(5, "CA", "GLY", "A", 2, 14.0, 0.0, 0.0),
            _atom_line(6, "C", "GLY", "A", 2, 15.0, 0.0, 0.0),
        ],
    )

    rmsd = calculate_rmsd.calculate_rmsd_and_superpose(
        str(ref_path),
        str(target_path),
        temp_dir=str(tmp_path),
    )

    assert rmsd == pytest.approx(0.0)
    assert (tmp_path / "superposed_ref.pdb").is_file()
    assert (tmp_path / "superposed_target.pdb").is_file()


def test_calculate_rmsd_and_superpose_returns_none_without_matching_chains(
    tmp_path,
    caplog,
):
    ref_path = _write_pdb(
        tmp_path / "ref_nomatch.pdb",
        [_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)],
    )
    target_path = _write_pdb(
        tmp_path / "target_nomatch.pdb",
        [_atom_line(1, "CA", "ALA", "B", 1, 10.0, 0.0, 0.0)],
    )

    with caplog.at_level(logging.ERROR):
        rmsd = calculate_rmsd.calculate_rmsd_and_superpose(
            str(ref_path),
            str(target_path),
            temp_dir=str(tmp_path),
        )

    assert rmsd is None
    assert "No suitable atoms found for RMSD calculation." in caplog.text


def test_calculate_rmsd_and_superpose_writes_to_cwd_when_temp_dir_is_none(
    tmp_path,
    monkeypatch,
):
    ref_path = _write_pdb(
        tmp_path / "cwd_ref.pdb",
        [
            _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CB", "ALA", "A", 1, 1.0, 0.0, 0.0),
        ],
    )
    target_path = _write_pdb(
        tmp_path / "cwd_target.pdb",
        [
            _atom_line(1, "CA", "ALA", "A", 1, 5.0, 0.0, 0.0),
            _atom_line(2, "CB", "ALA", "A", 1, 6.0, 0.0, 0.0),
        ],
    )

    monkeypatch.chdir(tmp_path)

    rmsd = calculate_rmsd.calculate_rmsd_and_superpose(
        str(ref_path),
        str(target_path),
    )

    assert rmsd == pytest.approx(0.0)
    assert (tmp_path / "superposed_cwd_ref.pdb").is_file()
    assert (tmp_path / "superposed_cwd_target.pdb").is_file()
