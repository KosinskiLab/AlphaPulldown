"""--fast_kernels: checking requirements before a model is built."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from alphapulldown.prediction import fast_kernels


def _choice_for(monkeypatch, problem, capability=90):
    monkeypatch.setattr(fast_kernels, "_requirement_problem", lambda: (problem, capability))
    monkeypatch.setattr(fast_kernels, "_package_version", lambda: "0.4.0")


def test_off_checks_nothing(monkeypatch):
    def fail():
        raise AssertionError("off must not look at the GPU or the packages")

    monkeypatch.setattr(fast_kernels, "_requirement_problem", fail)

    choice = fast_kernels.resolve("off")

    assert not choice.enabled
    assert choice.global_config_update() == {}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="one of off, on, auto"):
        fast_kernels.resolve("fast")


@pytest.mark.parametrize("mode", ["on", "auto", "AUTO"])
def test_enabled_when_every_requirement_is_met(monkeypatch, mode):
    _choice_for(monkeypatch, None, capability=90)

    choice = fast_kernels.resolve(mode)

    assert choice.enabled
    assert choice.package_version == "0.4.0"
    assert choice.global_config_update() == {"use_pallas": True, "compute_capability": 90}


def test_on_fails_loudly_and_auto_falls_back(monkeypatch):
    _choice_for(monkeypatch, "Tesla V100 has compute capability 7.0", capability=70)

    with pytest.raises(ValueError, match="--fast_kernels=on, but Tesla V100"):
        fast_kernels.resolve("on")

    choice = fast_kernels.resolve("auto")
    assert not choice.enabled
    assert "Tesla V100" in choice.reason
    assert choice.global_config_update() == {}


def _fake_jax(monkeypatch, devices):
    def gpu_devices(backend=None):
        if devices is None:
            raise RuntimeError("Unknown backend: 'gpu'")
        return devices

    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(devices=gpu_devices))


@pytest.fixture
def packages_present(monkeypatch):
    real_find_spec = importlib.util.find_spec
    real_import = importlib.import_module
    monkeypatch.setattr(
        fast_kernels.importlib.util, "find_spec",
        lambda name, *a: object() if name == "colabfold_kernels" else real_find_spec(name, *a),
    )
    monkeypatch.setattr(
        fast_kernels.importlib, "import_module",
        lambda name, *a: types.SimpleNamespace() if name == "alphafold.model.fused_kernels"
        else real_import(name, *a),
    )


def test_missing_package_is_reported_with_the_install_hint(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        fast_kernels.importlib.util, "find_spec",
        lambda name, *a: None if name == "colabfold_kernels" else real_find_spec(name, *a),
    )

    problem, _ = fast_kernels._requirement_problem()

    assert "alphapulldown[fast-kernels]" in problem


def test_alphafold_without_hooks_is_reported(monkeypatch):
    real_import = importlib.import_module

    def import_module(name, *a):
        if name == "alphafold.model.fused_kernels":
            raise ImportError(name)
        return real_import(name, *a)

    monkeypatch.setattr(fast_kernels.importlib.util, "find_spec", lambda name, *a: object())
    monkeypatch.setattr(fast_kernels.importlib, "import_module", import_module)

    problem, _ = fast_kernels._requirement_problem()

    assert "fused-kernel hooks" in problem


@pytest.mark.parametrize(
    "devices, expected",
    [
        (None, "no GPU"),
        ([], "no GPU"),
        ([types.SimpleNamespace(device_kind="AMD Instinct MI210")], "not an NVIDIA CUDA GPU"),
        ([types.SimpleNamespace(device_kind="Tesla T4", compute_capability="7.5")], "7.5"),
    ],
)
def test_unsupported_gpus_are_reported(monkeypatch, packages_present, devices, expected):
    _fake_jax(monkeypatch, devices)

    problem, _ = fast_kernels._requirement_problem()

    assert expected in problem


def test_supported_gpu_runs_the_smoke_test(monkeypatch, packages_present):
    _fake_jax(monkeypatch, [types.SimpleNamespace(device_kind="NVIDIA H100", compute_capability="9.0")])
    ran = []
    monkeypatch.setattr(fast_kernels, "_smoke_test", lambda: ran.append(True))

    assert fast_kernels._requirement_problem() == (None, 90)
    assert ran == [True]


def test_a_failing_smoke_test_is_reported(monkeypatch, packages_present):
    _fake_jax(monkeypatch, [types.SimpleNamespace(device_kind="NVIDIA RTX", compute_capability="12.0")])

    def broken():
        raise RuntimeError("ptxas does not know sm_120")

    monkeypatch.setattr(fast_kernels, "_smoke_test", broken)

    problem, capability = fast_kernels._requirement_problem()

    assert "test kernel failed" in problem and "sm_120" in problem
    assert capability == 120
