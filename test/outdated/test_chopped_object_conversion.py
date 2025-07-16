import pytest
import pickle
from pathlib import Path

from alphapulldown.objects import ChoppedObject, MultimericObject
from alphapulldown.folding_backend.alphafold3_backend import _convert_to_fold_input

# Path to your pickled MonomericObject
DATA_PKL = Path(__file__).parent / "test_data" / "features" / "A0A075B6L2.pkl"

@pytest.fixture(scope="module")
def monomer_obj():
    with open(DATA_PKL, "rb") as f:
        return pickle.load(f)

@pytest.fixture
def make_chopped(monomer_obj):
    def _make(regions):
        co = ChoppedObject(
            description=monomer_obj.description,
            sequence=monomer_obj.sequence,
            feature_dict=monomer_obj.feature_dict,
            regions=regions
        )
        co.prepare_final_sliced_feature_dict()
        return co
    return _make

@pytest.mark.parametrize("regions, expected_seq", [
    # Single region: first 10 residues
    ([(0, 10)], "MPLVVAVIFF"),
    # Three regions: 1-10 + 2-5 + 3-12
    ([(0, 10), (1, 5), (2, 12)],
     "MPLVVAVIFF" + "PLVV" + "LVVAVIFFSL"),
    # Four regions: 1-3, 4-5, 6-7, 7-8
    ([(0, 3), (3, 5), (5, 7), (6, 8)],
     "MPL" + "VV" + "AV" + "VI"),
])
def test_chopped_to_input_sequence_and_mapping(make_chopped, regions, expected_seq):
    co = make_chopped(regions)
    inp = _convert_to_fold_input(co, random_seed=0)
    
    # One chain, correct sequence
    assert len(inp.chains) == 1
    chain = inp.chains[0]
    assert chain.id == "A"
    assert chain.sequence == expected_seq
    
    # Every mapping key/value stays within [0, len(seq))
    for tpl in chain.templates or []:
        for q_idx, t_idx in tpl.query_to_template_map.items():
            assert 0 <= q_idx < len(expected_seq)
            assert 0 <= t_idx < len(expected_seq)

def test_overlapping_regions_mapping(make_chopped):
    # Overlapping: 1-5, 3-8, 5-10 → total length = 5 + 5 + 5 = 15
    regions = [(0, 5), (2, 8), (4, 10)]
    co = make_chopped(regions)
    inp = _convert_to_fold_input(co, random_seed=0)
    
    chain = inp.chains[0]
    total_len = sum(e - s for s, e in regions)
    assert len(chain.sequence) == total_len
    
    for tpl in chain.templates or []:
        for q_idx, t_idx in tpl.query_to_template_map.items():
            assert 0 <= q_idx < total_len
            assert 0 <= t_idx < total_len

def test_multimeric_conversion(make_chopped):
    # Two chopped chains: first 10, then 4 residues
    c1 = make_chopped([(0, 10)])
    c2 = make_chopped([(1, 5)])
    multi = MultimericObject(interactors=[c1, c2], pair_msa=True)
    
    inp = _convert_to_fold_input(multi, random_seed=0)
    assert len(inp.chains) == 2
    
    seqs = [ch.sequence for ch in inp.chains]
    assert seqs == ["MPLVVAVIFF", "PLVV"]

def test_no_templates(monomer_obj):
    # Strip out all template keys from the feature dict
    no_tpl = {k: v for k, v in monomer_obj.feature_dict.items()
              if not k.startswith("template_")}
    
    co = ChoppedObject(
        description=monomer_obj.description,
        sequence=monomer_obj.sequence,
        feature_dict=no_tpl,
        regions=[(0, 5)]
    )
    co.prepare_final_sliced_feature_dict()
    inp = _convert_to_fold_input(co, random_seed=0)
    
    chain = inp.chains[0]
    # Expect no templates at all
    assert not chain.templates
