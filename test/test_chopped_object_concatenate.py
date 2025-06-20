# test/test_chopped_object_concatenate.py
import pytest
import numpy as np
import pickle
from pathlib import Path

from alphapulldown.objects import ChoppedObject

# -- test regions --
REGIONS_SINGLE = [(0, 10)]
REGIONS_DIMER = [(0, 10), (1, 5), (2, 12)]   # total residues = 10 + 4 + 10 = 24
REGIONS_LONG = [(0, 3), (3, 5), (5, 7), (6, 8)]  # total = 3 + 2 + 2 + 2 = 9

# -- fixtures --
@pytest.fixture(scope="module")
def monomer_obj():
    p = Path(__file__).parent / "test_data" / "features" / "A0A075B6L2.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)

@pytest.fixture
def chopped_factory(monomer_obj):
    def _make(regions):
        return ChoppedObject(
            description=monomer_obj.description,
            sequence=monomer_obj.sequence,
            feature_dict=monomer_obj.feature_dict,
            regions=regions
        )
    return _make

# -- helpers --
def slice_dicts(co, regions):
    return [
        co.prepare_individual_sliced_feature_dict(co.feature_dict, s+1, e)
        for s, e in regions
    ]

# -- sequence + length tests --
@pytest.mark.parametrize("regions", [
    REGIONS_SINGLE,
    REGIONS_DIMER,
    REGIONS_LONG,
])
def test_sequence_and_length(chopped_factory, monomer_obj, regions):
    co = chopped_factory(regions)
    out = co.concatenate_sliced_feature_dict(slice_dicts(co, regions))
    # Compute expected sequence
    expected = ""
    for s, e in regions:
        expected += monomer_obj.sequence[s:e]
    seq = out["sequence"][0].decode()
    assert seq == expected
    assert out["seq_length"][0] == len(expected)

# -- high-D template shape tests --
@pytest.mark.parametrize("key, expected_dims", [
    ("template_aatype",    (35,  None, 22)),
    ("template_all_atom_masks",    (35,  None, 37)),
    ("template_all_atom_positions", (35, None, 37, 3)),
])
def test_template_highdim_shapes(chopped_factory, key, expected_dims):
    co = chopped_factory(REGIONS_DIMER)
    out = co.concatenate_sliced_feature_dict(slice_dicts(co, REGIONS_DIMER))
    shape = out[key].shape
    total_len = sum(e - s for s, e in REGIONS_DIMER)
    assert shape[0] == expected_dims[0]
    assert shape[-1] == expected_dims[-1]
    assert shape[1] == total_len

# -- MSA concatenation tests --
@pytest.mark.parametrize("key, expected_dims", [
    ("msa", (2832, None)),
    ("msa_all_seq", (1736, None)),
    ("deletion_matrix_int", (2832, None)),
    ("deletion_matrix_int_all_seq", (1736, None)),
])
def test_msa_concatenation_shapes(chopped_factory, key, expected_dims):
    co = chopped_factory(REGIONS_DIMER)
    out = co.concatenate_sliced_feature_dict(slice_dicts(co, REGIONS_DIMER))
    shape = out[key].shape
    total_len = sum(e - s for s, e in REGIONS_DIMER)
    assert shape[0] == expected_dims[0]
    assert shape[1] == total_len

def test_msa_species_identifiers_preserved(chopped_factory, monomer_obj):
    co = chopped_factory(REGIONS_DIMER)
    out = co.concatenate_sliced_feature_dict(slice_dicts(co, REGIONS_DIMER))
    # MSA species identifiers should be preserved from first slice
    assert len(out["msa_species_identifiers"]) == len(monomer_obj.feature_dict["msa_species_identifiers"])
    assert len(out["msa_species_identifiers_all_seq"]) == len(monomer_obj.feature_dict["msa_species_identifiers_all_seq"])

# -- empty input raises --
def test_empty_raises(chopped_factory):
    co = chopped_factory(REGIONS_SINGLE)
    with pytest.raises(IndexError):
        co.concatenate_sliced_feature_dict([])

# -- single-slice identity --
def test_single_dict_identity(chopped_factory, monomer_obj):
    co = chopped_factory(REGIONS_SINGLE)
    single = co.prepare_individual_sliced_feature_dict(
        monomer_obj.feature_dict, 1, 10
    )
    out = co.concatenate_sliced_feature_dict([single])
    for k, v in single.items():
        if k in ("sequence", "seq_length", "num_alignments"):
            continue
        np.testing.assert_array_equal(out[k], v)

# -- 1‑D template fields: now asserting correct shapes so failures show up --
def test_template_confidence_scores_shape(chopped_factory):
    co = chopped_factory(REGIONS_DIMER)
    out = co.concatenate_sliced_feature_dict(slice_dicts(co, REGIONS_DIMER))
    expected = sum(e - s for s, e in REGIONS_DIMER)
    assert out["template_confidence_scores"].shape == (1, expected)

def test_template_sum_probs_shape(chopped_factory, monomer_obj):
    co = chopped_factory(REGIONS_DIMER)
    out = co.concatenate_sliced_feature_dict(slice_dicts(co, REGIONS_DIMER))
    n_templates = monomer_obj.feature_dict["template_aatype"].shape[0]
    assert out["template_sum_probs"].shape == (n_templates, 1)

def test_template_domain_names_length(chopped_factory, monomer_obj):
    co = chopped_factory(REGIONS_DIMER)
    out = co.concatenate_sliced_feature_dict(slice_dicts(co, REGIONS_DIMER))
    n_templates = monomer_obj.feature_dict["template_aatype"].shape[0]
    assert len(out["template_domain_names"]) == n_templates
