"""AlphaPulldown's persistent JAX compile cache with real JAX on CPU.

Each case runs a fresh interpreter that turns the cache on through
``enable_persistent_compilation_cache`` and calls one tiny jitted function, so the
second process of a pair must load it from disk instead of compiling it. The unit tests
in test/unit/test_jax_compilation_cache.py only check the configuration calls.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("jaxlib") is None:  # conftest may stub jax itself
    pytest.skip("needs a real JAX installation", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
XLA_CACHES_ENV = "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"

_PROBE = r"""
import json, sys
import jax
import jax.numpy as jnp
from jax._src import compiler

events = []
jax.monitoring.register_event_listener(lambda event, **kwargs: events.append(event))

from alphapulldown.prediction.jax_compilation_cache import (
    enable_persistent_compilation_cache,
)

used = enable_persistent_compilation_cache(sys.argv[1])
forward = jax.jit(lambda x: jnp.tanh(x @ x.T).sum())
forward(jnp.ones((8, 8))).block_until_ready()

# The XLA cache paths JAX hands the compiler when the persistent cache is on. Keep
# every object alive: the protobuf accessors return views into their parent.
options = compiler.get_compile_options(num_replicas=1, num_partitions=1)
build = options.executable_build_options
debug = build.debug_options
print("PROBE " + json.dumps({
    "cache_dir": used,
    "hits": events.count("/jax/compilation_cache/cache_hits"),
    "misses": events.count("/jax/compilation_cache/cache_misses"),
    "xla_caches": jax.config.jax_persistent_cache_enable_xla_caches,
    "autotune_cache_dir": debug.xla_gpu_per_fusion_autotune_cache_dir,
    "kernel_cache_file": debug.xla_gpu_kernel_cache_file,
}))
"""


def _run_probe(cache_dir: Path, xla_caches: str | None = None) -> dict:
    env = os.environ.copy()
    for name in (XLA_CACHES_ENV, "JAX_COMPILATION_CACHE_DIR", "PYTEST_ADDOPTS"):
        env.pop(name, None)
    if xla_caches is not None:
        env[XLA_CACHES_ENV] = xla_caches
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(cache_dir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("PROBE "))
    return json.loads(line[len("PROBE "):])


def _entries(cache_dir: Path) -> dict[str, bool]:
    """Name -> is a regular file, for everything in the cache directory."""
    return {path.name: path.is_file() for path in cache_dir.iterdir()}


def test_a_second_process_loads_the_compiled_function_from_the_cache(tmp_path):
    cache_dir = tmp_path / "jax-cache"

    first = _run_probe(cache_dir)
    written = _entries(cache_dir)
    second = _run_probe(cache_dir)

    assert first["cache_dir"] == str(cache_dir)
    assert first["hits"] == 0 and first["misses"] >= 1
    assert written and all(written.values())
    assert second["hits"] >= 1 and second["misses"] == 0
    assert _entries(cache_dir) == written  # nothing compiled, nothing new written


@pytest.mark.parametrize("xla_caches", [None, "none"], ids=["default", "env-none"])
def test_only_jax_entries_are_written_without_xla_caches(tmp_path, xla_caches):
    # XLA's autotune cache segfaults the compiler on shared filesystems, so AlphaPulldown
    # turns XLA's caches off unless the environment chooses otherwise.
    cache_dir = tmp_path / "jax-cache"

    probe = _run_probe(cache_dir, xla_caches)

    assert probe["xla_caches"] == "none"
    assert probe["autotune_cache_dir"] == ""
    assert probe["kernel_cache_file"] == ""
    entries = _entries(cache_dir)
    assert entries and all(entries.values())
    assert not [name for name in entries if name.startswith("xla_gpu_")]


def test_the_environment_can_still_turn_xla_caches_on(tmp_path):
    # The probe above would see them: with "all", JAX points XLA into the cache dir.
    cache_dir = tmp_path / "jax-cache"

    probe = _run_probe(cache_dir, "all")

    assert probe["xla_caches"] == "all"
    assert probe["autotune_cache_dir"] == str(cache_dir / "xla_gpu_per_fusion_autotune_cache_dir")
    assert probe["kernel_cache_file"] == str(cache_dir / "xla_gpu_kernel_cache_file")
