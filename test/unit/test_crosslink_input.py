from pathlib import Path
import pickle

import numpy as np
import pytest
from alphafold.data.pipeline_multimer import _FastaChain


torch = pytest.importorskip("torch", reason="AlphaLink crosslink helpers require torch")
from unifold.dataset import bin_xl, calculate_offsets, create_xl_features


TEST_ROOT = Path(__file__).resolve().parents[1]
CROSSLINK_FIXTURE = TEST_ROOT / "alphalink" / "test_xl_input.pkl"


@pytest.fixture
def crosslink_context():
    return {
        "crosslinks": pickle.load(CROSSLINK_FIXTURE.open("rb")),
        "asym_id": torch.tensor([1] * 10 + [2] * 25 + [3] * 40),
        "chain_id_map": {
            "A": _FastaChain(sequence="", description="chain1"),
            "B": _FastaChain(sequence="", description="chain2"),
            "C": _FastaChain(sequence="", description="chain3"),
        },
        "bins": torch.arange(0, 1.05, 0.05),
    }


def test_calculate_offsets(crosslink_context):
    offsets = calculate_offsets(crosslink_context["asym_id"]).tolist()
    assert offsets == [0, 10, 35, 75]


def test_create_xl_inputs(crosslink_context):
    offsets = calculate_offsets(crosslink_context["asym_id"])
    xl = create_xl_features(
        crosslink_context["crosslinks"],
        offsets,
        chain_id_map=crosslink_context["chain_id_map"],
    )
    expected_xl = torch.tensor(
        [
            [10, 35, 0.01],
            [3, 27, 0.01],
            [5, 56, 0.01],
            [20, 65, 0.01],
        ]
    )
    assert torch.equal(xl, expected_xl)


def test_bin_xl(crosslink_context):
    offsets = calculate_offsets(crosslink_context["asym_id"])
    xl = create_xl_features(
        crosslink_context["crosslinks"],
        offsets,
        chain_id_map=crosslink_context["chain_id_map"],
    )
    num_res = len(crosslink_context["asym_id"])
    xl = bin_xl(xl, num_res)
    expected_xl = np.zeros((num_res, num_res, 1))
    expected_xl[3, 27, 0] = expected_xl[27, 3, 0] = torch.bucketize(
        0.99, crosslink_context["bins"]
    )
    expected_xl[10, 35, 0] = expected_xl[35, 10, 0] = torch.bucketize(
        0.99, crosslink_context["bins"]
    )
    expected_xl[5, 56, 0] = expected_xl[56, 5, 0] = torch.bucketize(
        0.99, crosslink_context["bins"]
    )
    expected_xl[20, 65, 0] = expected_xl[65, 20, 0] = torch.bucketize(
        0.99, crosslink_context["bins"]
    )
    assert np.array_equal(xl, expected_xl)
