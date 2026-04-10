from pathlib import Path
from types import SimpleNamespace

import lzma
import numpy as np
import pickle
import pytest

from alphapulldown.objects import MonomericObject
from alphapulldown.utils import modelling_setup
from alphapulldown_input_parser import RegionSelection
from alphapulldown_input_parser.parser import Region


def test_normalise_fold_entry_handles_json_input_and_region_selection():
    entry = {
        "json_input": "input.json",
        "regions": RegionSelection((Region(2, 5), Region(9, 11))),
    }

    normalised = modelling_setup._normalise_fold_entry(entry)

    assert normalised == {
        "json_input": "input.json",
        "regions": [(2, 5), (9, 11)],
    }


def test_normalise_fold_entry_handles_all_and_explicit_regions():
    all_entry = {"proteinA": RegionSelection.all()}
    region_entry = {"proteinB": RegionSelection((Region(1, 3),))}

    assert modelling_setup._normalise_fold_entry(all_entry) == {"proteinA": "all"}
    assert modelling_setup._normalise_fold_entry(region_entry) == {"proteinB": [(1, 3)]}


def test_parse_fold_normalises_external_parser_output(monkeypatch):
    parsed_jobs = [[
        {"proteinA": RegionSelection.all()},
        {"proteinB": RegionSelection((Region(4, 8),))},
        {"json_input": "job.json", "regions": RegionSelection((Region(2, 6),))},
    ]]

    monkeypatch.setattr(modelling_setup, "_external_parse_fold", lambda **_: parsed_jobs)

    result = modelling_setup.parse_fold("input.txt", "/tmp/features", "+")

    assert result == [[
        {"proteinA": "all"},
        {"proteinB": [(4, 8)]},
        {"json_input": "job.json", "regions": [(2, 6)]},
    ]]


def test_pad_input_features_pads_residue_and_msa_axes_and_restores_metadata():
    feature_dict = {
        "msa": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "deletion_matrix_int": np.zeros((2, 3), dtype=np.int32),
        "aatype": np.ones((3, 21), dtype=np.float32),
        "pair_repr": np.ones((3, 3), dtype=np.float32),
        "assembly_num_chains": np.array([2]),
        "num_templates": np.array([1]),
        "seq_length": np.array([3]),
        "num_alignments": np.array([2]),
    }

    modelling_setup.pad_input_features(feature_dict, desired_num_res=5, desired_num_msa=4)

    assert feature_dict["msa"].shape == (4, 5)
    assert feature_dict["deletion_matrix_int"].shape == (4, 5)
    assert feature_dict["aatype"].shape == (5, 21)
    # The helper pads only the first matching residue-sized axis when the
    # same dimension appears more than once.
    assert feature_dict["pair_repr"].shape == (5, 3)
    assert feature_dict["seq_length"].tolist() == [5]
    assert feature_dict["num_alignments"].tolist() == [4]
    assert feature_dict["assembly_num_chains"].tolist() == [2]
    assert feature_dict["num_templates"].tolist() == [1]


def test_create_custom_info_wraps_columns_by_position():
    result = modelling_setup.create_custom_info([
        [{"proteinA": "all"}, {"proteinB": [(1, 3)]}],
        [{"json_input": "job.json"}],
    ])

    assert result == [
        {"col_1": [{"proteinA": "all"}], "col_2": [{"proteinB": [(1, 3)]}]},
        {"col_1": [{"json_input": "job.json"}]},
    ]


def test_create_uniprot_runner_passes_expected_kwargs(monkeypatch):
    captured = {}

    class FakeJackhmmer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(modelling_setup.jackhmmer, "Jackhmmer", FakeJackhmmer)

    runner = modelling_setup.create_uniprot_runner("jackhmmer", "uniprot.fasta")

    assert isinstance(runner, FakeJackhmmer)
    assert captured == {
        "binary_path": "jackhmmer",
        "database_path": "uniprot.fasta",
    }


@pytest.mark.parametrize("mask_key", ["template_all_atom_masks", "template_all_atom_mask"])
def test_check_empty_templates_supports_both_mask_keys(mask_key):
    empty_features = {
        mask_key: np.zeros((0,), dtype=np.float32),
        "template_aatype": np.zeros((1,), dtype=np.float32),
    }
    populated_features = {
        mask_key: np.ones((1,), dtype=np.float32),
        "template_aatype": np.ones((1,), dtype=np.float32),
    }

    assert modelling_setup.check_empty_templates(empty_features) is True
    assert modelling_setup.check_empty_templates(populated_features) is False


def test_mk_mock_template_adds_expected_template_keys():
    feature_dict = {"aatype": np.zeros((4, 21), dtype=np.float32)}

    updated = modelling_setup.mk_mock_template(feature_dict)

    assert updated["template_all_atom_positions"].shape[0] == 1
    assert updated["template_all_atom_positions"].shape[1] == 4
    assert updated["template_all_atom_masks"].shape[:2] == (1, 4)
    assert updated["template_aatype"].shape[:2] == (1, 4)
    assert updated["template_sequence"] == [b"none"]
    assert updated["template_domain_names"] == [b"none"]
    assert updated["template_sum_probs"].tolist() == [0.0]


@pytest.mark.parametrize("suffix", [".pkl", ".pkl.xz"])
def test_load_monomer_objects_loads_pickles_and_injects_mock_templates(tmp_path, monkeypatch, suffix):
    monomer = MonomericObject("proteinA", "ACDE")
    monomer.feature_dict = {"template_aatype": np.zeros((0,), dtype=np.float32)}
    target = tmp_path / f"proteinA{suffix}"
    if suffix == ".pkl":
        with target.open("wb") as handle:
            pickle.dump(monomer, handle)
    else:
        with lzma.open(target, "wb") as handle:
            pickle.dump(monomer, handle)

    monkeypatch.setattr(modelling_setup, "check_empty_templates", lambda _: True)
    monkeypatch.setattr(
        modelling_setup,
        "mk_mock_template",
        lambda features: {**features, "mock_template": True},
    )

    loaded = modelling_setup.load_monomer_objects({target.name: str(tmp_path)}, "proteinA")

    assert loaded.description == "proteinA"
    assert loaded.feature_dict["mock_template"] is True


def test_load_monomer_objects_raises_for_missing_file():
    with pytest.raises(FileNotFoundError, match="No file found for missing"):
        modelling_setup.load_monomer_objects({}, "missing")


def test_create_interactors_handles_json_and_all_region(monkeypatch):
    monomer = MonomericObject("proteinA", "ACDE")
    monomer.feature_dict = {"template_aatype": np.ones((1,), dtype=np.float32)}

    monkeypatch.setattr(
        modelling_setup,
        "make_dir_monomer_dictionary",
        lambda _: {"proteinA.pkl": "/unused"},
    )
    monkeypatch.setattr(modelling_setup, "load_monomer_objects", lambda *_: monomer)
    monkeypatch.setattr(modelling_setup, "check_empty_templates", lambda _: False)

    result = modelling_setup.create_interactors(
        [{"col_1": [{"json_input": "job.json"}], "col_2": [{"proteinA": "all"}]}],
        ["/unused"],
    )

    assert result == [[{"json_input": "job.json"}, monomer]]


def test_create_interactors_builds_chopped_object_for_region_lists(monkeypatch):
    monomer = MonomericObject("proteinA", "ACDEFG")
    monomer.feature_dict = {"template_aatype": np.ones((1,), dtype=np.float32)}
    calls = {}

    class FakeChoppedObject:
        def __init__(self, description, sequence, feature_dict, regions):
            calls["args"] = (description, sequence, feature_dict, regions)
            self.prepared = False

        def prepare_final_sliced_feature_dict(self):
            self.prepared = True

    monkeypatch.setattr(
        modelling_setup,
        "make_dir_monomer_dictionary",
        lambda _: {"proteinA.pkl": "/unused"},
    )
    monkeypatch.setattr(modelling_setup, "load_monomer_objects", lambda *_: monomer)
    monkeypatch.setattr(modelling_setup, "check_empty_templates", lambda _: False)
    monkeypatch.setattr(modelling_setup, "ChoppedObject", FakeChoppedObject)

    result = modelling_setup.create_interactors(
        [{"col_1": [{"proteinA": [(2, 4)]}]}],
        ["/unused"],
    )

    chopped = result[0][0]
    assert isinstance(chopped, FakeChoppedObject)
    assert chopped.prepared is True
    assert calls["args"] == ("proteinA", "ACDEFG", monomer.feature_dict, [(2, 4)])


def test_create_interactors_propagates_skip_msa_marker_to_chopped_objects(monkeypatch):
    monomer = MonomericObject("proteinA", "ACDEFG")
    monomer.feature_dict = {"template_aatype": np.ones((1,), dtype=np.float32)}
    monomer.skip_msa = True

    class FakeChoppedObject:
        def __init__(self, description, sequence, feature_dict, regions):
            self.description = description
            self.sequence = sequence
            self.feature_dict = feature_dict
            self.regions = regions
            self.prepared = False

        def prepare_final_sliced_feature_dict(self):
            self.prepared = True

    monkeypatch.setattr(
        modelling_setup,
        "make_dir_monomer_dictionary",
        lambda _: {"proteinA.pkl": "/unused"},
    )
    monkeypatch.setattr(modelling_setup, "load_monomer_objects", lambda *_: monomer)
    monkeypatch.setattr(modelling_setup, "check_empty_templates", lambda _: False)
    monkeypatch.setattr(modelling_setup, "ChoppedObject", FakeChoppedObject)

    result = modelling_setup.create_interactors(
        [{"col_1": [{"proteinA": [(2, 4)]}]}],
        ["/unused"],
    )

    chopped = result[0][0]
    assert isinstance(chopped, FakeChoppedObject)
    assert chopped.prepared is True
    assert chopped.skip_msa is True


def test_create_interactors_currently_skips_append_when_templates_are_empty(monkeypatch):
    monomer = MonomericObject("proteinA", "ACDE")
    monomer.feature_dict = {}

    monkeypatch.setattr(
        modelling_setup,
        "make_dir_monomer_dictionary",
        lambda _: {"proteinA.pkl": "/unused"},
    )
    monkeypatch.setattr(modelling_setup, "load_monomer_objects", lambda *_: monomer)
    monkeypatch.setattr(modelling_setup, "check_empty_templates", lambda _: True)
    monkeypatch.setattr(
        modelling_setup,
        "mk_mock_template",
        lambda features: {**features, "mock_template": True},
    )

    result = modelling_setup.create_interactors(
        [{"col_1": [{"proteinA": "all"}]}],
        ["/unused"],
    )

    assert result == [[]]
    assert monomer.feature_dict["mock_template"] is True
