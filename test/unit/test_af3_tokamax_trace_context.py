"""AF3 must compile its model once per process, not again on the second prediction.

tokamax creates a JAX user context the first time an op looks up its autotuning cache,
which is while the model is first traced. That changes JAX's trace context, part of
every jit cache key, so the next call re-traced and recompiled the whole model
(~50 s per prediction). AlphaPulldown calls tokamax's private hook before the first
trace (``alphafold3_backend._initialise_tokamax_trace_context``).

These tests use the real tokamax and JAX, on CPU, so they run in the AF3 image and
skip elsewhere. On CPU tokamax's XLA implementations need no config and never look up
the cache, so the jitted function does the lookup a GPU kernel does while it is traced.
Each case runs in a fresh process: the context is created once per process (thread).
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tokamax")

REPO_ROOT = Path(__file__).resolve().parents[2]

_PROBE = r"""
import json, sys
import jax
import jax.numpy as jnp
import tokamax
from tokamax._src.ops.gated_linear_unit import base

if sys.argv[1] == "alphapulldown":
    from alphapulldown.folding_backend.alphafold3_backend import (
        _initialise_tokamax_trace_context,
    )
    _initialise_tokamax_trace_context()

op = base.GatedLinearUnit()

@jax.jit
def forward(x, weights):
    # What a tokamax GPU kernel does while it is traced: resolve its config.
    op.bind(x, weights, activation=jax.nn.swish).cached_autotuning_data
    return tokamax.gated_linear_unit(
        x=x, weights=weights, activation=jax.nn.swish, implementation="xla"
    )

args = (jnp.ones((8, 16)), jnp.ones((16, 2, 32)))
for _ in range(3):
    jax.block_until_ready(forward(*args))
print("PROBE", json.dumps({"compiled": forward._cache_size()}))
"""


def _compiled_versions(variant: str) -> int:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, variant],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("PROBE "))
    return json.loads(line[len("PROBE "):])["compiled"]


def test_tokamax_still_has_the_hook_alphapulldown_calls():
    op = importlib.import_module("tokamax._src.ops.op")

    assert callable(getattr(op, "get_autotuning_cache_overlay_state", None))
    # The lookup a kernel makes while traced goes through the same hook.
    lookup = inspect.getsource(op.BoundArguments.cached_autotuning_data.fget)
    assert "get_autotuning_cache_overlay_state" in lookup


def test_a_jitted_tokamax_op_recompiles_on_its_second_call_without_the_hook():
    # The regression itself: if this stops recompiling, tokamax no longer creates its
    # context lazily and the AlphaPulldown hook call may be unnecessary.
    assert _compiled_versions("stock") == 2


def test_alphapulldown_hook_keeps_a_jitted_tokamax_op_to_one_compile():
    assert _compiled_versions("alphapulldown") == 1
