from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alphafold.data.templates import SingleHitResult
from alphapulldown.utils import multimeric_template_utils as mtu


def test_prepare_multimeric_template_meta_info_parses_valid_csv(tmp_path):
    csv_path = tmp_path / "templates.csv"
    template_path = tmp_path / "template1.cif"
    template_path.write_text("data", encoding="utf-8")
    csv_path.write_text("proteinA,template1.cif,A\n", encoding="utf-8")

    result = mtu.prepare_multimeric_template_meta_info(str(csv_path), str(tmp_path))

    assert result == {"proteinA": [("template1.cif", "A")]}


def test_prepare_multimeric_template_meta_info_keeps_duplicate_rows_for_homo_oligomers(tmp_path):
    csv_path = tmp_path / "templates.csv"
    template_path = tmp_path / "template1.cif"
    template_path.write_text("data", encoding="utf-8")
    csv_path.write_text(
        "proteinA,template1.cif,A\nproteinA,template1.cif,B\n",
        encoding="utf-8",
    )

    result = mtu.prepare_multimeric_template_meta_info(str(csv_path), str(tmp_path))

    assert result == {"proteinA": [("template1.cif", "A"), ("template1.cif", "B")]}


def test_prepare_multimeric_template_meta_info_exits_on_invalid_row(tmp_path):
    csv_path = tmp_path / "templates.csv"
    csv_path.write_text("proteinA,template1.cif\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        mtu.prepare_multimeric_template_meta_info(str(csv_path), str(tmp_path))


def test_obtain_kalign_binary_path_returns_discovered_binary(monkeypatch):
    monkeypatch.setattr(mtu.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert mtu.obtain_kalign_binary_path() == "/usr/bin/kalign"


def test_obtain_kalign_binary_path_asserts_when_binary_missing(monkeypatch):
    monkeypatch.setattr(mtu.shutil, "which", lambda name: None)

    with pytest.raises(AssertionError, match="Could not find kalign"):
        mtu.obtain_kalign_binary_path()


def test_parse_mmcif_file_returns_parsing_result(monkeypatch, tmp_path):
    expected = SimpleNamespace(name="parsed")
    calls = []

    class FakeFiltered:
        def __init__(self, path, file_id, chain_id):
            assert path == tmp_path / "template.cif"
            assert file_id == "1abc"
            assert chain_id == "A"
            self.parsing_result = expected

        def remove_clashes(self, threshold, hb_allowance):
            calls.append(("remove_clashes", threshold, hb_allowance))

        def remove_low_plddt(self, threshold):
            calls.append(("remove_low_plddt", threshold))

    monkeypatch.setattr(mtu, "MmcifChainFiltered", FakeFiltered)

    result = mtu.parse_mmcif_file(
        "1abc",
        str(tmp_path / "template.cif"),
        "A",
        threshold_clashes=12.5,
        hb_allowance=0.7,
        plddt_threshold=42.0,
    )

    assert result is expected
    assert calls == [
        ("remove_clashes", 12.5, 0.7),
        ("remove_low_plddt", 42.0),
    ]


def test_parse_mmcif_file_returns_none_when_file_missing(monkeypatch, tmp_path):
    class FakeFiltered:
        def __init__(self, path, file_id, chain_id):
            raise FileNotFoundError("missing")

    monkeypatch.setattr(mtu, "MmcifChainFiltered", FakeFiltered)

    result = mtu.parse_mmcif_file("1abc", str(tmp_path / "missing.cif"), "A")

    assert result is None


def test_obtain_mapping_uses_kalign_and_template_index_builder(monkeypatch):
    mmcif_result = SimpleNamespace(
        mmcif_object=SimpleNamespace(chain_to_seqres={"A": "ACXE"})
    )
    captured = {}

    class FakeKalign:
        def __init__(self, binary_path):
            captured["binary_path"] = binary_path

        def align(self, seqs):
            captured["align_input"] = seqs
            return "fake-a3m"

    monkeypatch.setattr(mtu, "obtain_kalign_binary_path", lambda: "/usr/bin/kalign")
    monkeypatch.setattr(mtu.kalign, "Kalign", FakeKalign)
    monkeypatch.setattr(
        mtu.parsers,
        "parse_a3m",
        lambda _: SimpleNamespace(sequences=("ACDE", "AC-E")),
    )
    monkeypatch.setattr(
        mtu.parsers,
        "_get_indices",
        lambda seq, start=0: list(range(start, start + len(seq.replace("-", "")))),
    )
    monkeypatch.setattr(
        mtu,
        "_build_query_to_hit_index_mapping",
        lambda *args: {0: 0, 1: 1, 3: 2},
    )

    mapping, parsed_sequence = mtu._obtain_mapping(mmcif_result, "A", "ACDE")

    assert captured["binary_path"] == "/usr/bin/kalign"
    assert captured["align_input"] == ["ACDE", "ACXE"]
    assert mapping == {0: 0, 1: 1, 3: 2}
    assert parsed_sequence == "ACXE"


def test_extract_multimeric_template_features_for_single_chain_replicates_features(monkeypatch):
    parse_result = SimpleNamespace(mmcif_object=SimpleNamespace())
    base_positions = np.zeros((1, 37, 3), dtype=np.float32)
    base_masks = np.ones((1, 37), dtype=np.float32)
    base_aatype = np.zeros((1, 22), dtype=np.float32)

    monkeypatch.setattr(mtu, "parse_mmcif_file", lambda *args, **kwargs: parse_result)
    monkeypatch.setattr(mtu, "_obtain_mapping", lambda **kwargs: ({0: 0}, "A"))
    monkeypatch.setattr(mtu, "obtain_kalign_binary_path", lambda: "/usr/bin/kalign")
    monkeypatch.setattr(
        mtu,
        "_extract_template_features",
        lambda **kwargs: (
            {
                "template_all_atom_positions": base_positions,
                "template_all_atom_masks": base_masks,
                "template_sequence": b"A",
                "template_domain_names": b"1abc_A",
                "template_aatype": base_aatype,
            },
            "realign warning",
        ),
    )

    result = mtu.extract_multimeric_template_features_for_single_chain(
        query_seq="A",
        pdb_id="1abc",
        chain_id="A",
        mmcif_file="template.cif",
    )

    assert isinstance(result, SingleHitResult)
    assert result.warning == "realign warning"
    assert result.features["template_sum_probs"] == [0, 0, 0, 0]
    assert result.features["template_all_atom_positions"].shape == (4, 1, 37, 3)
    assert result.features["template_all_atom_position"].shape == (4, 1, 37, 3)
    assert result.features["template_all_atom_mask"].shape == (1, 1, 37)
    assert result.features["template_sequence"] == [b"A"] * 4
    assert result.features["template_domain_names"] == [b"1abc_A"] * 4


def test_extract_multimeric_template_features_for_single_chain_returns_empty_result_on_extraction_failure(monkeypatch):
    parse_result = SimpleNamespace(mmcif_object=SimpleNamespace())

    monkeypatch.setattr(mtu, "parse_mmcif_file", lambda *args, **kwargs: parse_result)
    monkeypatch.setattr(mtu, "_obtain_mapping", lambda **kwargs: ({0: 0}, "A"))
    monkeypatch.setattr(mtu, "_extract_template_features", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    result = mtu.extract_multimeric_template_features_for_single_chain(
        query_seq="A",
        pdb_id="1abc",
        chain_id="A",
        mmcif_file="template.cif",
    )

    assert isinstance(result, SingleHitResult)
    assert result.features is None
    assert result.error is None
    assert result.warning is None


def test_extract_multimeric_template_features_for_single_chain_returns_none_when_parse_result_missing(monkeypatch):
    monkeypatch.setattr(mtu, "parse_mmcif_file", lambda *args, **kwargs: None)

    result = mtu.extract_multimeric_template_features_for_single_chain(
        query_seq="A",
        pdb_id="1abc",
        chain_id="A",
        mmcif_file="template.cif",
    )

    assert result is None
