import gzip
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from alphapulldown.analysis_pipeline import calculate_mpdockq as mpdockq


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
    occupancy: float = 1.0,
    bfactor: float = 50.0,
) -> str:
    element = atom_name.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom_name:<4}{' ':1}{res_name:>3} {chain_id:1}"
        f"{residue_number:4d}{' ':1}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element:>2}\n"
    )


def _write_pdb(path: Path, atoms: list[str]) -> Path:
    path.write_text("".join(atoms) + "TER\nEND\n", encoding="utf-8")
    return path


def test_parse_atm_record_extracts_fixed_width_fields():
    line = _atom_line(12, "CB", "SER", "B", 7, 1.5, 2.5, 3.5, occupancy=0.5, bfactor=42.0)

    record = mpdockq.parse_atm_record(line)

    assert record["name"] == "ATOM"
    assert record["atm_no"] == 12
    assert record["atm_name"] == "CB"
    assert record["res_name"] == "SER"
    assert record["chain"] == "B"
    assert record["res_no"] == 7
    assert record["x"] == pytest.approx(1.5)
    assert record["y"] == pytest.approx(2.5)
    assert record["z"] == pytest.approx(3.5)
    assert record["occ"] == pytest.approx(0.5)
    assert record["B"] == pytest.approx(42.0)


def test_read_pdb_extracts_chain_coordinates_and_ca_cb_indices(tmp_path):
    pdb_path = _write_pdb(
        tmp_path / "chains.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0),
            _atom_line(3, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0),
            _atom_line(4, "N", "GLY", "B", 1, 5.0, 0.0, 0.0),
            _atom_line(5, "CA", "GLY", "B", 1, 6.0, 0.0, 0.0),
        ],
    )

    pdb_chains, chain_coords, chain_ca_inds, chain_cb_inds = mpdockq.read_pdb(
        str(pdb_path)
    )

    assert sorted(pdb_chains) == ["A", "B"]
    assert len(chain_coords["A"]) == 3
    assert len(chain_coords["B"]) == 2
    assert chain_ca_inds == {"A": [1], "B": [1]}
    assert chain_cb_inds == {"A": [2], "B": [1]}


def test_parse_bfactor_averages_residue_bfactors(tmp_path):
    pdb_path = _write_pdb(
        tmp_path / "bfactor.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, bfactor=10.0),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, bfactor=20.0),
            _atom_line(3, "N", "GLY", "A", 2, 2.0, 0.0, 0.0, bfactor=40.0),
            _atom_line(4, "CA", "GLY", "A", 2, 3.0, 0.0, 0.0, bfactor=60.0),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        residue_bfactors = mpdockq.parse_bfactor(str(pdb_path))

    np.testing.assert_allclose(residue_bfactors, np.asarray([15.0, 50.0]))


def test_get_best_plddt_prefers_uncompressed_pickle(tmp_path):
    (tmp_path / "ranking_debug.json").write_text(
        json.dumps({"order": ["model_1"]}),
        encoding="utf-8",
    )
    expected = np.asarray([91.0, 92.0])
    with open(tmp_path / "result_model_1.pkl", "wb") as handle:
        pickle.dump({"plddt": expected}, handle)

    plddt = mpdockq.get_best_plddt(str(tmp_path))

    np.testing.assert_array_equal(plddt, expected)


def test_get_best_plddt_falls_back_to_gzipped_pickle(tmp_path):
    (tmp_path / "ranking_debug.json").write_text(
        json.dumps({"order": ["model_2"]}),
        encoding="utf-8",
    )
    expected = np.asarray([81.0, 82.0])
    with gzip.open(tmp_path / "result_model_2.pkl.gz", "wb") as handle:
        pickle.dump({"plddt": expected}, handle)

    plddt = mpdockq.get_best_plddt(str(tmp_path))

    np.testing.assert_array_equal(plddt, expected)


def test_get_best_plddt_falls_back_to_ranked_pdb_bfactors(monkeypatch, tmp_path):
    (tmp_path / "ranking_debug.json").write_text(
        json.dumps({"order": ["model_3"]}),
        encoding="utf-8",
    )
    (tmp_path / "ranked_0.pdb").write_text("HEADER\n", encoding="utf-8")
    expected = np.asarray([71.0, 72.0])
    monkeypatch.setattr(mpdockq, "parse_bfactor", lambda pdb_path: expected)

    plddt = mpdockq.get_best_plddt(str(tmp_path))

    np.testing.assert_array_equal(plddt, expected)


def test_get_best_plddt_returns_none_when_no_sources_exist(tmp_path, capsys):
    (tmp_path / "ranking_debug.json").write_text(
        json.dumps({"order": ["model_4"]}),
        encoding="utf-8",
    )

    plddt = mpdockq.get_best_plddt(str(tmp_path))

    captured = capsys.readouterr()
    assert plddt is None
    assert "ranked_0.pdb not found" in captured.out


def test_read_plddt_slices_values_per_chain():
    best_plddt = np.asarray([90.0, 91.0, 50.0])
    chain_ca_inds = {"A": [0, 1], "B": [0]}

    per_chain = mpdockq.read_plddt(best_plddt, chain_ca_inds)

    np.testing.assert_array_equal(per_chain["A"], np.asarray([90.0, 91.0]))
    np.testing.assert_array_equal(per_chain["B"], np.asarray([50.0]))


def test_score_complex_returns_zero_when_no_interface_contacts():
    complex_score, chain_count = mpdockq.score_complex(
        {
            "A": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "B": [[30.0, 0.0, 0.0], [31.0, 0.0, 0.0]],
        },
        {"A": [1], "B": [1]},
        {"A": np.asarray([80.0]), "B": np.asarray([85.0])},
    )

    assert complex_score == 0
    assert chain_count == 2


def test_score_complex_and_mpdockq_are_positive_for_contacting_chains():
    complex_score, chain_count = mpdockq.score_complex(
        {
            "A": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "B": [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]],
        },
        {"A": [1], "B": [1]},
        {"A": np.asarray([90.0]), "B": np.asarray([90.0])},
    )

    mpdockq_score = mpdockq.calculate_mpDockQ(complex_score)
    expected = 0.827 / (1 + math.exp(-0.036 * (complex_score - 261.398))) + 0.221

    assert chain_count == 2
    assert complex_score > 0
    assert mpdockq_score == pytest.approx(expected)


def test_read_pdb_pdockq_uses_cb_and_gly_ca_coordinates(tmp_path):
    pdb_path = _write_pdb(
        tmp_path / "pdockq.pdb",
        [
            _atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, bfactor=20.0),
            _atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, bfactor=30.0),
            _atom_line(3, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0, bfactor=40.0),
            _atom_line(4, "N", "GLY", "B", 1, 5.0, 0.0, 0.0, bfactor=50.0),
            _atom_line(5, "CA", "GLY", "B", 1, 6.0, 0.0, 0.0, bfactor=60.0),
        ],
    )

    chain_coords, chain_plddt = mpdockq.read_pdb_pdockq(str(pdb_path))

    np.testing.assert_array_equal(chain_coords["A"], np.asarray([[1.0, 1.0, 0.0]]))
    np.testing.assert_array_equal(chain_coords["B"], np.asarray([[6.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(chain_plddt["A"], np.asarray([40.0]))
    np.testing.assert_array_equal(chain_plddt["B"], np.asarray([60.0]))


def test_read_pdb_pdockq_appends_multiple_qualifying_atoms_per_chain(tmp_path):
    pdb_path = _write_pdb(
        tmp_path / "pdockq_multi.pdb",
        [
            _atom_line(1, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0, bfactor=40.0),
            _atom_line(2, "CB", "SER", "A", 2, 2.0, 1.0, 0.0, bfactor=45.0),
            _atom_line(3, "CA", "GLY", "B", 1, 6.0, 0.0, 0.0, bfactor=60.0),
        ],
    )

    chain_coords, chain_plddt = mpdockq.read_pdb_pdockq(str(pdb_path))

    np.testing.assert_array_equal(
        chain_coords["A"],
        np.asarray([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
    )
    np.testing.assert_array_equal(chain_plddt["A"], np.asarray([40.0, 45.0]))
    np.testing.assert_array_equal(chain_coords["B"], np.asarray([[6.0, 0.0, 0.0]]))


def test_calc_pdockq_returns_zero_without_contacts():
    score = mpdockq.calc_pdockq(
        {
            "A": np.asarray([[0.0, 0.0, 0.0]]),
            "B": np.asarray([[20.0, 0.0, 0.0]]),
        },
        {
            "A": np.asarray([90.0]),
            "B": np.asarray([80.0]),
        },
        8,
    )

    assert score == 0


def test_calc_pdockq_returns_positive_score_for_contacting_chains():
    score = mpdockq.calc_pdockq(
        {
            "A": np.asarray([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
            "B": np.asarray([[0.0, 0.0, 5.0], [0.0, 3.0, 5.0]]),
        },
        {
            "A": np.asarray([90.0, 92.0]),
            "B": np.asarray([88.0, 86.0]),
        },
        8,
    )

    assert 0 < score < 1
