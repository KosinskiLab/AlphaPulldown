"""The AF3 GPU tests' own checks, which judge a GPU and its outputs without JAX."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DISPATCH_PATH = ROOT / "alphafold3/src/alphafold3/jax/fused_triangle/dispatch.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the class is created
    spec.loader.exec_module(module)
    return module


checks = _load("af3_gpu_checks", ROOT / "test/cluster/af3_gpu_checks.py")
dispatch = _load("af3_dispatch_gpu_checks", DISPATCH_PATH)


# --------------------------------------------------------------------------------------
# Is the GPU one the fork's fused triangle kernels are enabled on?


def test_nvidia_smi_output_gives_the_first_gpu():
    gpu = checks.parse_nvidia_smi(
        "9.0, 81559 MiB, Disabled\n9.0, 81559 MiB, Disabled\n", "GPU 0: NVIDIA H100"
    )

    assert gpu.compute_capability == "9.0"
    assert gpu.memory_gib == pytest.approx(79.65, abs=0.01)


def test_a_mig_slice_has_the_memory_of_its_profile():
    listing = (
        "GPU 0: NVIDIA RTX PRO 4500 Blackwell (UUID: GPU-1)\n"
        "  MIG 1g.8gb       Device  0: (UUID: MIG-2)\n"
    )

    gpu = checks.parse_nvidia_smi("12.0, 32623 MiB, Enabled", listing)

    assert gpu.compute_capability == "12.0"
    assert gpu.memory_gib == pytest.approx(8e9 / 2**30)


@pytest.mark.parametrize(
    "query, listing",
    [
        ("", ""),
        ("9.0", ""),
        ("9.0, [N/A], Disabled", ""),
        ("12.0, 32623 MiB, Enabled", "GPU 0: NVIDIA RTX PRO 4500 Blackwell"),
    ],
)
def test_unreadable_nvidia_smi_output_is_an_error(query, listing):
    with pytest.raises(ValueError):
        checks.parse_nvidia_smi(query, listing)


def test_the_jax_memory_fraction_comes_from_the_fold_environment():
    assert checks.jax_memory_fraction({"XLA_CLIENT_MEM_FRACTION": "0.95"}) == 0.95
    assert checks.jax_memory_fraction({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5"}) == 0.5
    assert checks.jax_memory_fraction({}) == 0.75


@pytest.mark.parametrize(
    "compute_capability, memory_gib",
    [
        ("8.0", 39.5),  # A100 40 GB
        ("8.6", 23.6),  # RTX 3090, A40 below 48
        ("8.9", 44.4),  # L40S
        ("9.0", 79.6),  # H100
        ("12.0", 95.0),  # RTX Pro 6000 Blackwell
    ],
)
def test_the_measured_gpus_are_supported(compute_capability, memory_gib):
    gpu = checks.Gpu(compute_capability, memory_gib)

    support = checks.fused_triangle_support(gpu, 0.95, dispatch.device_policy)

    assert support.supported is True, support.detail


@pytest.mark.parametrize(
    "gpu, fraction, reason",
    [
        (("10.0", 178.0), 0.95, "unvalidated_compute_capability"),  # B200
        (("7.0", 31.7), 0.95, "unvalidated_compute_capability"),  # V100
        (("12.0", 8e9 / 2**30), 0.95, "memory_budget_below_12_gib"),  # MIG 1g.8gb
        (("9.0", 15.0), 0.75, "memory_budget_below_12_gib"),  # 11.25 GiB for JAX
    ],
)
def test_other_gpus_are_not_supported_and_the_detail_says_why(gpu, fraction, reason):
    support = checks.fused_triangle_support(checks.Gpu(*gpu), fraction, dispatch.device_policy)

    assert support.supported is False
    assert support.detail.endswith(f": {reason}")
    assert f"compute capability {gpu[0]}" in support.detail


def test_an_alphafold3_without_the_fork_supports_no_gpu():
    support = checks.fused_triangle_support(checks.Gpu("9.0", 79.6), 0.95, None)

    assert support.supported is False
    assert "alphafold3.jax.fused_triangle" in support.detail


def test_without_a_described_gpu_support_cannot_be_told():
    support = checks.fused_triangle_support(None, 0.95, dispatch.device_policy)

    assert support.supported is None


def test_the_check_reads_the_forks_policy_without_importing_jax():
    # The test process must leave the GPU to the fold it starts.
    code = (
        "import sys\n"
        f"sys.path[:0] = [{str(ROOT / 'test/cluster')!r}, {str(ROOT / 'alphafold3/src')!r}]\n"
        "import af3_gpu_checks\n"
        "policy = af3_gpu_checks.installed_device_policy()\n"
        "assert policy is not None\n"
        "assert policy('9.0', 75.7).enabled\n"
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------------------
# Are the scores AF3 wrote real numbers? AF3 writes NaN as null.

MONOMER_SUMMARY = """{
 "chain_iptm": [null], "chain_pair_iptm": [[0.84]], "chain_pair_pae_min": [[0.76]],
 "chain_ptm": [0.84], "fraction_disordered": 0.0, "has_clash": 0.0, "iptm": null,
 "ptm": 0.84, "ranking_score": 0.84, "chain_ids": ["A"]
}"""
DIMER_SUMMARY = """{
 "chain_iptm": [0.61, 0.61], "chain_pair_iptm": [[0.8, 0.61], [0.61, 0.79]],
 "chain_pair_pae_min": [[0.8, 4.2], [4.3, 0.8]], "chain_ptm": [0.8, 0.79],
 "fraction_disordered": 0.05, "has_clash": 0.0, "iptm": 0.61, "ptm": 0.7,
 "ranking_score": 0.66
}"""
FULL = """{
 "atom_chain_ids": ["A", "A"], "atom_plddts": [91.2, 88.4],
 "contact_probs": [[1.0, 0.9], [0.9, 1.0]], "pae": [[0.8, 1.4], [1.3, 0.8]],
 "token_chain_ids": ["A", "A"], "token_res_ids": [1, 2]
}"""


def _summary(text: str, **changes) -> dict:
    payload = json.loads(text)
    payload.update(changes)
    return payload


def test_a_monomer_may_have_no_iptm():
    assert checks.confidence_problems("summary_confidences.json", _summary(MONOMER_SUMMARY)) == []


def test_complete_dimer_and_full_confidences_pass():
    assert checks.confidence_problems("x_summary_confidences.json", _summary(DIMER_SUMMARY)) == []
    assert checks.confidence_problems("confidences.json", json.loads(FULL)) == []


@pytest.mark.parametrize(
    "changes, problem",
    [
        ({"ptm": None}, "ptm is null (NaN)"),
        ({"ranking_score": None}, "ranking_score is null (NaN)"),
        ({"chain_ptm": [None]}, "chain_ptm[0] is null (NaN)"),
        ({"chain_pair_iptm": [[None]]}, "chain_pair_iptm[0][0] is null (NaN)"),
        ({"ptm": float("nan")}, "ptm is nan"),
        ({"fraction_disordered": float("inf")}, "fraction_disordered is inf"),
        ({"ptm": True}, "ptm is a boolean, not a number"),
        ({"ptm": "0.84"}, "ptm is not a number: '0.84'"),
        ({"ptm": [0.84]}, "ptm is an array, not a number"),
        ({"chain_ptm": []}, "chain_ptm is not a non-empty array: []"),
    ],
)
def test_missing_or_non_finite_monomer_scores_fail(changes, problem):
    problems = checks.confidence_problems(
        "summary_confidences.json", _summary(MONOMER_SUMMARY, **changes)
    )

    assert problem in problems, problems


def test_nan_written_by_python_json_fails():
    payload = json.loads(MONOMER_SUMMARY.replace('"ptm": 0.84', '"ptm": NaN'))

    assert checks.confidence_problems("summary_confidences.json", payload) == ["ptm is nan"]


def test_a_missing_score_fails():
    payload = _summary(MONOMER_SUMMARY)
    del payload["ranking_score"]

    assert checks.confidence_problems("summary_confidences.json", payload) == [
        "ranking_score is missing"
    ]


def test_has_clash_may_be_a_boolean():
    payload = _summary(MONOMER_SUMMARY, has_clash=False)

    assert checks.confidence_problems("summary_confidences.json", payload) == []


@pytest.mark.parametrize(
    "changes, problem",
    [
        ({"iptm": None}, "iptm is null (NaN)"),
        ({"chain_iptm": [0.61, None]}, "chain_iptm[1] is null (NaN)"),
    ],
)
def test_a_complex_must_have_an_iptm(changes, problem):
    problems = checks.confidence_problems(
        "summary_confidences.json", _summary(DIMER_SUMMARY, **changes)
    )

    assert problems == [problem]


@pytest.mark.parametrize(
    "field, value, problem",
    [
        ("atom_plddts", [91.2, None], "atom_plddts[1] is null (NaN)"),
        ("pae", [[0.8, float("nan")], [1.3, 0.8]], "pae[0][1] is nan"),
        ("contact_probs", None, "contact_probs is not a non-empty array: None"),
    ],
)
def test_full_confidences_need_finite_plddt_pae_and_contacts(field, value, problem):
    payload = json.loads(FULL)
    payload[field] = value

    assert checks.confidence_problems("seed-1_sample-0/confidences.json", payload) == [problem]


def test_an_all_nan_array_is_reported_briefly():
    payload = json.loads(FULL)
    payload["pae"] = [[None] * 64 for _ in range(64)]

    problems = checks.confidence_problems("confidences.json", payload)

    assert problems[:5] == [f"pae[0][{index}] is null (NaN)" for index in range(5)]
    assert problems[5:] == [f"pae: {64 * 64 - 5} more"]
