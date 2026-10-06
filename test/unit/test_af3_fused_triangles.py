"""--fast_kernels for AlphaFold 3: checking the device before a model is built."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from alphapulldown.prediction import af3_fused_triangles

DISPATCH_PATH = (
    Path(__file__).resolve().parents[2]
    / "alphafold3/src/alphafold3/jax/fused_triangle/dispatch.py"
)
SETTINGS = {
    "triangle_multiplication_implementation": "pallas",
    "triangle_attention_implementation": "auto",
}


def _fail(*args):
    raise AssertionError("off must not inspect the device or import the fork")


def _unavailable(*args):
    raise RuntimeError("unavailable test device")


def _global_config(**overrides) -> types.SimpleNamespace:
    fields = dict(
        triangle_multiplication_implementation="pallas",
        triangle_attention_implementation="auto",
        fused_triangle_compute_capability="8.6",
        fused_triangle_memory_gib=45,
        bfloat16="all",
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


@pytest.fixture
def dispatch(monkeypatch):
    """The fork's dispatch module from the submodule, importable without the fork."""
    spec = importlib.util.spec_from_file_location("af3_dispatch_test", DISPATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    package = types.ModuleType("alphafold3.jax.fused_triangle")
    package.dispatch = module
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return module


@pytest.fixture
def model_config(monkeypatch):
    """A stub ``alphafold3.model.model_config`` whose GlobalConfig has the hooks."""
    module = types.ModuleType("alphafold3.model.model_config")
    module.GlobalConfig = lambda: types.SimpleNamespace(
        triangle_multiplication_implementation="default"
    )
    parent = types.ModuleType("alphafold3.model")
    parent.model_config = module
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_off_checks_nothing(monkeypatch):
    monkeypatch.setattr(af3_fused_triangles, "device_settings", _fail)
    monkeypatch.setattr(af3_fused_triangles, "smoke_test", _fail)

    choice = af3_fused_triangles.resolve("off", object())

    assert not choice.enabled
    assert choice.mode == "off"
    assert choice.global_config_update() == {}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="one of off, on, auto"):
        af3_fused_triangles.resolve("fast", object())


@pytest.mark.parametrize("stage", ["device_settings", "smoke_test"])
def test_on_fails_loudly_and_auto_falls_back(monkeypatch, stage):
    monkeypatch.setattr(af3_fused_triangles, "device_settings", lambda device: SETTINGS)
    monkeypatch.setattr(af3_fused_triangles, "smoke_test", lambda *args: None)
    monkeypatch.setattr(af3_fused_triangles, stage, _unavailable)

    with pytest.raises(ValueError, match="--fast_kernels=on, but .*unavailable test device"):
        af3_fused_triangles.resolve("on", object())

    choice = af3_fused_triangles.resolve("auto", object())
    assert not choice.enabled
    assert choice.mode == "auto"
    assert choice.reason.startswith("--fast_kernels=auto: ")
    assert "unavailable test device" in choice.reason
    assert choice.global_config_update() == {}


def test_checks_the_given_device_before_enabling(monkeypatch):
    device = object()
    seen = []
    monkeypatch.setattr(
        af3_fused_triangles, "device_settings", lambda d: seen.append(d) or SETTINGS
    )
    monkeypatch.setattr(
        af3_fused_triangles, "smoke_test", lambda *args: seen.append(args)
    )

    choice = af3_fused_triangles.resolve("on", device)

    assert seen[0] is device
    assert seen[1] == (SETTINGS, device)
    assert choice.enabled and choice.mode == "on"
    assert choice.global_config_update() == SETTINGS
    # The update is a copy: changing it does not change the choice.
    choice.global_config_update().clear()
    assert choice.global_config_update() == SETTINGS


def test_smoke_test_runs_on_the_device_it_checks(monkeypatch):
    # Stub JAX, Haiku and the AF3 modules, recording the default device of each step.
    device = object()
    current = []
    steps = []

    @contextlib.contextmanager
    def default_device(chosen):
        current.append(chosen)
        try:
            yield
        finally:
            current.pop()

    def step(name):
        def run(*args, **kwargs):
            steps.append((name, current[-1] if current else None))
            return np.ones(1)

        return run

    jnp = types.SimpleNamespace(
        ones=step("ones"), bfloat16="bfloat16", all=np.all, isfinite=np.isfinite
    )
    jax = types.SimpleNamespace(
        numpy=jnp,
        default_device=default_device,
        jit=lambda function: step("jit"),
        random=types.SimpleNamespace(PRNGKey=step("PRNGKey")),
    )
    transformed = types.SimpleNamespace(init=step("init"), apply=None)
    haiku = types.SimpleNamespace(
        transform=lambda function: transformed, without_apply_rng=lambda t: t
    )
    model = types.ModuleType("alphafold3.model")
    model.model_config = types.SimpleNamespace(GlobalConfig=lambda **fields: fields)
    model.components = types.SimpleNamespace(utils=None)
    model.network = types.SimpleNamespace(modules=None)
    for name, module in {
        "jax": jax,
        "jax.numpy": jnp,
        "haiku": haiku,
        "alphafold3.model": model,
        "alphafold3.model.components": model.components,
        "alphafold3.model.network": model.network,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    af3_fused_triangles.smoke_test(SETTINGS, device)

    assert [name for name, _ in steps] == ["ones", "ones", "PRNGKey", "init", "jit"]
    assert all(used is device for _, used in steps)


def test_device_settings_use_the_allocator_budget(dispatch, model_config):
    device = types.SimpleNamespace(
        compute_capability="12.0",
        device_kind="Blackwell slice",
        memory_stats=lambda: {"bytes_limit": 15 * 2**30},
    )

    settings = af3_fused_triangles.device_settings(device)

    assert settings == {
        "triangle_multiplication_implementation": "pallas",
        "triangle_attention_implementation": "auto",
        "fused_triangle_compute_capability": "12.0",
        "fused_triangle_memory_gib": 15,
    }


@pytest.mark.parametrize(
    "compute_capability, memory_stats, reason",
    [
        ("12.0", lambda: None, "unknown_memory_budget"),
        ("10.0", lambda: {"bytes_limit": 80 * 2**30}, "unvalidated_compute_capability"),
        ("8.0", lambda: {"bytes_limit": 8 * 2**30}, "memory_budget_below_12_gib"),
    ],
)
def test_device_settings_reject_devices_the_policy_does_not_enable(
    dispatch, model_config, compute_capability, memory_stats, reason
):
    device = types.SimpleNamespace(
        compute_capability=compute_capability, device_kind="GPU", memory_stats=memory_stats
    )

    with pytest.raises(RuntimeError, match=reason):
        af3_fused_triangles.device_settings(device)


def test_device_settings_reject_an_alphafold3_without_hooks(model_config):
    model_config.GlobalConfig = lambda: types.SimpleNamespace()

    with pytest.raises(RuntimeError, match="no fused-triangle hooks"):
        af3_fused_triangles.device_settings(types.SimpleNamespace())


def test_metadata_of_the_original_layers_does_not_import_the_fork(monkeypatch):
    monkeypatch.setitem(sys.modules, "alphafold3.jax.fused_triangle", None)

    record = af3_fused_triangles.metadata(
        types.SimpleNamespace(), af3_fused_triangles.OFF, num_tokens=256
    )

    assert record == {
        "backend": "alphafold3",
        "requested_mode": "off",
        "padded_tokens": 256,
        "fused_kernels": False,
        "reason": "--fast_kernels=off",
    }


def test_metadata_records_per_operation_fallback(dispatch):
    choice = af3_fused_triangles.FusedTriangleChoice(True, "--fast_kernels=auto", "auto")

    record = af3_fused_triangles.metadata(_global_config(), choice, num_tokens=2560)

    assert record["fused_kernels"] is True
    assert record["requested_mode"] == "auto"
    assert record["policy_version"] == dispatch.POLICY_VERSION
    assert record["device_policy"]["attention_implementation"] == "pallas_tokamax_core"
    operations = record["operations"]
    assert set(operations) == {
        "triangle_multiplication_c128",
        "triangle_attention_c128",
        "triangle_multiplication_c64",
        "triangle_attention_c64",
    }
    assert operations["triangle_multiplication_c128"] == {
        "implementation": "pallas",
        "reason": "",
    }
    # Attention on a 45 GiB budget stops at 2048 tokens.
    assert operations["triangle_attention_c128"] == {
        "implementation": "default",
        "reason": "size_limit",
    }


def test_metadata_is_the_same_on_a_compile_cache_hit(dispatch):
    choice = af3_fused_triangles.FusedTriangleChoice(True, "--fast_kernels=on", "on")
    global_config = _global_config()

    first = af3_fused_triangles.metadata(global_config, choice, num_tokens=1024)

    assert first == af3_fused_triangles.metadata(global_config, choice, num_tokens=1024)


def test_metadata_float32_layers_run_the_original_body(dispatch):
    choice = af3_fused_triangles.FusedTriangleChoice(True, "--fast_kernels=on", "on")

    record = af3_fused_triangles.metadata(
        _global_config(bfloat16="none"), choice, num_tokens=256
    )

    assert record["fused_kernels"] is False
    assert {operation["reason"] for operation in record["operations"].values()} == {
        "dtype_not_bfloat16"
    }
