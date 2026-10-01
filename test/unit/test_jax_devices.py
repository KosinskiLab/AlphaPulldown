"""The import-time GPU probe must report a missing GPU, not raise.

``jax.local_devices(backend='gpu')`` raises on any machine without a GPU backend,
so the prediction commands could not even print ``--help`` on a login node.
"""

import sys
import types

from alphapulldown.prediction import jax_devices


def _with_jax(monkeypatch, local_devices):
    stub = types.ModuleType("jax")
    stub.local_devices = local_devices
    monkeypatch.setitem(sys.modules, "jax", stub)


def test_reports_the_gpu_devices_jax_sees(monkeypatch):
    _with_jax(monkeypatch, lambda backend: [f"{backend}:0", f"{backend}:1"])

    assert jax_devices.initialise_jax_gpu_backend() == ["gpu:0", "gpu:1"]


def test_a_missing_gpu_backend_is_reported_not_fatal(monkeypatch):
    def no_gpu_backend(backend):
        raise RuntimeError(
            "Unknown backend: 'gpu' requested, but no platforms that are instances "
            "of gpu are present. Platforms present are: cpu"
        )

    _with_jax(monkeypatch, no_gpu_backend)

    assert jax_devices.initialise_jax_gpu_backend() == []
