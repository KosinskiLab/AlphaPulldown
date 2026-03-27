import types

import pytest

import alphapulldown.folding_backend as folding_backend


def test_try_import_returns_attribute_on_success():
    imported = folding_backend._try_import("pathlib", "Path")

    from pathlib import Path

    assert imported is Path


def test_try_import_logs_warning_and_returns_none_on_failure(monkeypatch):
    warnings = []

    def fake_import_module(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(folding_backend, "import_module", fake_import_module)
    monkeypatch.setattr(folding_backend.logging, "warning", warnings.append)

    imported = folding_backend._try_import("missing.mod", "Thing")

    assert imported is None
    assert len(warnings) == 1
    assert "missing.mod:Thing" in warnings[0]


def test_backend_manager_repr_and_getattr_without_backend():
    manager = folding_backend.FoldingBackendManager()

    assert repr(manager) == "<BackendManager: no backend selected>"
    with pytest.raises(AttributeError):
        _ = manager.predict


def test_backend_manager_dir_includes_backend_attributes():
    manager = folding_backend.FoldingBackendManager()
    manager._backend = types.SimpleNamespace(custom_attr=1)

    names = manager.__dir__()

    assert "custom_attr" in names
    assert "available_backends" in names


def test_available_backends_filters_by_successful_import(monkeypatch):
    manager = folding_backend.FoldingBackendManager()

    def fake_try_import(module_path, class_name):
        return object if class_name in {"AlphaFold2Backend", "AlphaLinkBackend"} else None

    monkeypatch.setattr(folding_backend, "_try_import", fake_try_import)

    assert manager.available_backends() == ["alphafold2", "alphalink"]


def test_load_backend_class_validates_names_and_imports(monkeypatch):
    manager = folding_backend.FoldingBackendManager()

    with pytest.raises(NotImplementedError):
        manager._load_backend_class("missing")

    monkeypatch.setattr(folding_backend, "_try_import", lambda *_args: None)
    with pytest.raises(ImportError):
        manager._load_backend_class("alphafold2")

    class FakeBackend:
        pass

    monkeypatch.setattr(folding_backend, "_try_import", lambda *_args: FakeBackend)
    assert manager._load_backend_class("alphafold2") is FakeBackend


def test_change_backend_uses_default_and_stores_kwargs(monkeypatch):
    manager = folding_backend.FoldingBackendManager()

    class FakeBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(manager, "_load_backend_class", lambda name: FakeBackend)

    manager.change_backend(param=3)

    assert manager._backend_name == "alphafold2"
    assert isinstance(manager._backend, FakeBackend)
    assert manager._backend.kwargs == {"param": 3}
    assert manager._backend_args == {"param": 3}
    assert repr(manager) == "<BackendManager: using alphafold2>"


def test_module_change_backend_delegates_to_manager(monkeypatch):
    calls = []
    fake_manager = types.SimpleNamespace(change_backend=lambda name, **kwargs: calls.append((name, kwargs)))

    monkeypatch.setattr(folding_backend, "_get_manager", lambda: fake_manager)

    folding_backend.change_backend("alphafold3", debug=True)

    assert calls == [("alphafold3", {"debug": True})]
