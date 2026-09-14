"""Preserve imports and serialized globals across the package reorganization."""

import subprocess
import sys

import pytest


MODULES = (
    ("feature_batch", "features", "FeatureRequest"),
    ("af2_feature_finalizer", "features", "Af2FeatureFinalizationSettings"),
    ("af3_pipeline", "features", "AF3PipelineSettings"),
    ("prediction_batch", "prediction", "PredictionJob"),
    ("fold_preparation", "prediction", "prepare_fold"),
    ("inference_flags", "prediction", "model_flags"),
)


@pytest.mark.parametrize("name,package,symbol", MODULES)
@pytest.mark.parametrize("legacy_first", (True, False))
def test_legacy_imports_and_pickle_globals(name, package, symbol, legacy_first):
    # Separate interpreters exercise both import orders without pytest's imports
    # or mocked dependency modules masking a broken compatibility path.
    probe = f"""
import importlib
import pickle

legacy_name = 'alphapulldown.{name}'
canonical_name = 'alphapulldown.{package}.{name}'
names = [legacy_name, canonical_name]
if not {legacy_first!r}:
    names.reverse()
for module_name in names:
    importlib.import_module(module_name)
legacy = importlib.import_module(legacy_name)
canonical = importlib.import_module(canonical_name)
assert legacy is canonical

# Protocol 0 GLOBAL is the import lookup encoded by old pickles. Resolve it
# without generating the fixture using the renamed class's new __module__.
payload = b'calphapulldown.{name}\\n{symbol}\\n.'
assert pickle.loads(payload) is getattr(canonical, '{symbol}')
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_patching_the_legacy_module_reaches_canonical_callers(monkeypatch):
    import alphapulldown.inference_flags as legacy
    from alphapulldown.prediction import inference_flags

    monkeypatch.setattr(legacy, "FLAGS_BY_BACKEND", {"custom": {"supported"}})
    assert inference_flags.unsupported_flags("custom", ["supported", "unknown"]) == ["unknown"]


def test_feature_pickle_class_path_is_preserved():
    import pickle

    from alphapulldown.objects import MonomericObject

    assert MonomericObject.__module__ == "alphapulldown.objects"
    assert pickle.loads(b"calphapulldown.objects\nMonomericObject\n.") is MonomericObject
