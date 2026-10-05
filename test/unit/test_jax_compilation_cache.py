"""The persistent compile cache both folding backends share."""

from __future__ import annotations

import os
import sys
import types

import pytest

from alphapulldown.prediction import jax_compilation_cache


@pytest.fixture
def fake_jax(monkeypatch):
    """A stand-in `jax` recording config updates; options in `missing` are unknown."""
    updates = []
    missing = set()

    def update(name, value):
        if name in missing:
            raise AttributeError(f"Unrecognized config option: {name}")
        updates.append((name, value))

    module = types.SimpleNamespace(config=types.SimpleNamespace(update=update))
    monkeypatch.setitem(sys.modules, "jax", module)
    monkeypatch.delenv(jax_compilation_cache.XLA_CACHES_ENV, raising=False)
    return types.SimpleNamespace(updates=updates, missing=missing)


def test_no_directory_leaves_caching_off(fake_jax):
    assert jax_compilation_cache.enable_persistent_compilation_cache(None) is None
    assert jax_compilation_cache.enable_persistent_compilation_cache("") is None
    assert fake_jax.updates == []


def test_creates_directory_and_keeps_only_jax_entries(fake_jax, tmp_path):
    cache = tmp_path / "nested" / "jax-cache"

    assert jax_compilation_cache.enable_persistent_compilation_cache(cache) == str(cache)
    assert cache.is_dir()
    assert fake_jax.updates == [
        ("jax_compilation_cache_dir", str(cache)),
        ("jax_persistent_cache_min_compile_time_secs", 0),
        ("jax_persistent_cache_min_entry_size_bytes", 0),
        ("jax_persistent_cache_enable_xla_caches", "none"),
    ]


def test_explicit_xla_caches_setting_is_respected(fake_jax, tmp_path, monkeypatch):
    monkeypatch.setenv(jax_compilation_cache.XLA_CACHES_ENV, "all")

    jax_compilation_cache.enable_persistent_compilation_cache(tmp_path)

    assert ("jax_persistent_cache_enable_xla_caches", "none") not in fake_jax.updates
    assert ("jax_compilation_cache_dir", str(tmp_path)) in fake_jax.updates


def test_older_jax_without_xla_caches_option(fake_jax, tmp_path):
    fake_jax.missing.add("jax_persistent_cache_enable_xla_caches")

    assert jax_compilation_cache.enable_persistent_compilation_cache(tmp_path) == str(tmp_path)
    assert ("jax_compilation_cache_dir", str(tmp_path)) in fake_jax.updates


def test_unusable_directory_is_skipped_not_fatal(fake_jax, tmp_path, monkeypatch):
    def refuse(path, exist_ok=False):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(os, "makedirs", refuse)

    assert jax_compilation_cache.enable_persistent_compilation_cache(tmp_path / "x") is None
    assert fake_jax.updates == []


def test_read_only_directory_is_skipped(fake_jax, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "access", lambda path, mode: False)

    assert jax_compilation_cache.enable_persistent_compilation_cache(tmp_path) is None
    assert fake_jax.updates == []


def test_user_directory_is_expanded(fake_jax, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    used = jax_compilation_cache.enable_persistent_compilation_cache("~/jax-cache")

    assert used == str(tmp_path / "jax-cache")
    assert (tmp_path / "jax-cache").is_dir()
