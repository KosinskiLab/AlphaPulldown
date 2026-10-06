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


@pytest.mark.parametrize(
    "value, mode",
    [("off", "off"), ("ON", "on"), (" auto ", "auto"), (True, "on"), (False, "off"),
     ("True", "on"), ("false", "off"), ("yes", "on"), ("0", "off"), (None, "off")],
)
def test_yaml_boolean_spellings_mean_on_and_off(value, mode):
    # PyYAML reads an unquoted on/off in the workflow config as a boolean.
    assert fast_kernels.normalise_mode(value) == mode
    assert fast_kernels.is_valid_mode(value)


def test_invalid_values_are_not_valid_modes():
    assert not fast_kernels.is_valid_mode("fast")
    assert not fast_kernels.is_valid_mode("2")


def test_package_version_falls_back_to_the_module(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(fast_kernels.importlib.metadata, "version", missing)
    monkeypatch.setitem(sys.modules, "colabfold_kernels", types.SimpleNamespace(__version__="0.4.0"))

    assert fast_kernels._package_version() == "0.4.0"


def test_af3_off_does_not_inspect_or_import_the_fork(monkeypatch):
    def fail(*args):
        raise AssertionError('off must not inspect devices or import fused kernels')
    monkeypatch.setattr(fast_kernels, '_af3_settings', fail)
    monkeypatch.setattr(fast_kernels, '_af3_smoke_test', fail)
    choice = fast_kernels.resolve('off', backend='alphafold3')
    assert not choice.enabled
    assert choice.global_config_update() == {}
    assert fast_kernels.af3_metadata(types.SimpleNamespace(), requested_mode='off', tokens=256)['fused_kernels'] is False


@pytest.mark.parametrize('stage', ['settings', 'smoke'])
def test_af3_on_fails_and_auto_falls_back(monkeypatch, stage):
    def fail(*args):
        raise RuntimeError('unavailable test device')
    monkeypatch.setattr(fast_kernels, '_af3_settings', lambda device: {'sentinel': True})
    monkeypatch.setattr(fast_kernels, '_af3_smoke_test', lambda settings: None)
    monkeypatch.setattr(fast_kernels, '_af3_' + ('settings' if stage == 'settings' else 'smoke_test'), fail)
    with pytest.raises(ValueError, match='unavailable test device'):
        fast_kernels.resolve('on', backend='alphafold3')
    choice = fast_kernels.resolve('auto', backend='alphafold3')
    assert not choice.enabled and choice.global_config_update() == {}


def test_af3_uses_requested_device_and_smokes_before_enabling(monkeypatch):
    device = object()
    seen = []
    settings = {'triangle_multiplication_implementation': 'pallas',
                'triangle_attention_implementation': 'auto'}
    monkeypatch.setattr(fast_kernels, '_af3_settings', lambda d: seen.append(d) or settings)
    monkeypatch.setattr(fast_kernels, '_af3_smoke_test', lambda s: seen.append(dict(s)))
    choice = fast_kernels.resolve('on', backend='alphafold3', device=device)
    assert seen == [device, settings]
    assert choice.enabled and choice.global_config_update() == settings
    choice.global_config_update().clear()
    assert choice.global_config_update() == settings


def test_unknown_backend_cannot_enable_kernels():
    with pytest.raises(ValueError, match='not supported'):
        fast_kernels.resolve('auto', backend='alphalink')


@pytest.fixture
def af3_dispatch(monkeypatch):
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / 'alphafold3/src/alphafold3/jax/fused_triangle/dispatch.py'
    spec = importlib.util.spec_from_file_location('af3_dispatch_test', source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    package = types.ModuleType('alphafold3.jax.fused_triangle')
    package.dispatch = module
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return module


def test_af3_metadata_records_partial_fallback_on_cache_hits(af3_dispatch):
    config = types.SimpleNamespace(
        triangle_multiplication_implementation='pallas',
        triangle_attention_implementation='auto',
        fused_triangle_compute_capability='8.6', fused_triangle_memory_gib=45,
        bfloat16='all')
    first = fast_kernels.af3_metadata(config, requested_mode='auto', tokens=2560)
    assert first == fast_kernels.af3_metadata(config, requested_mode='auto', tokens=2560)
    assert first['fused_kernels']
    assert first['operations']['triangle_multiplication_c128']['implementation'] == 'pallas'
    assert first['operations']['triangle_attention_c128'] == dict(
        implementation='default', reason='size_limit')
    config.bfloat16 = 'none'
    assert not fast_kernels.af3_metadata(config, requested_mode='auto', tokens=256)['fused_kernels']


def test_af3_device_policy_uses_allocator_budget(af3_dispatch, monkeypatch):
    config_module = types.ModuleType('alphafold3.model.model_config')
    config_module.GlobalConfig = lambda: types.SimpleNamespace(
        triangle_multiplication_implementation='default')
    parent = types.ModuleType('alphafold3.model')
    parent.model_config = config_module
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    device = types.SimpleNamespace(compute_capability='12.0', device_kind='Blackwell slice',
                                   memory_stats=lambda: {'bytes_limit': 15 * 2**30})
    result = fast_kernels._af3_settings(device)
    assert result['fused_triangle_memory_gib'] == 15
    assert result['fused_triangle_compute_capability'] == '12.0'
    device.memory_stats = lambda: None
    with pytest.raises(RuntimeError, match='unknown_memory_budget'):
        fast_kernels._af3_settings(device)
