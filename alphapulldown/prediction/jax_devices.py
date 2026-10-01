"""Initialise JAX's GPU backend before anything else can claim the device.

The prediction commands call :func:`initialise_jax_gpu_backend` at import time,
before the folding backends import TensorFlow and OpenMM, so that JAX initialises
CUDA first and its device memory is not pre-empted by a library that merely got
imported earlier. That used to be a bare ``jax.local_devices(backend='gpu')``,
which raises when JAX has no GPU backend, so ``--help`` and the head-node flag
validation the workflow documentation recommends failed on any machine without a
GPU. A missing GPU is reported here, not fatal: whether inference can run
without one is each backend's decision (AlphaFold 3 refuses, AlphaFold 2 falls
back to CPU).
"""

from __future__ import annotations

from typing import Any, List

from absl import logging


def initialise_jax_gpu_backend() -> List[Any]:
    """Return the GPU devices JAX can see, or an empty list when it sees none."""
    import jax

    try:
        devices = list(jax.local_devices(backend="gpu"))
    except RuntimeError as exc:
        logging.warning(
            "JAX found no usable GPU backend (%s); only CPU inference is possible.",
            exc,
        )
        return []
    logging.info("JAX GPU devices: %s", devices)
    return devices
