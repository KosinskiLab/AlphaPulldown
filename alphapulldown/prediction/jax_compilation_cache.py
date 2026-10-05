"""Point JAX at a persistent on-disk compile cache that is safe on shared storage.

Both folding backends are JAX-compiled, and a fresh process recompiles every model:
about 50 s per AlphaFold 3 prediction call, and minutes per AlphaFold 2 fold. JAX's
persistent compilation cache keeps the executables on disk, keyed by the GPU model and
the JAX/XLA version, so later processes, and later folds of the same shape, load them
instead of compiling.

By default JAX also writes XLA's own caches into that directory, among them a
per-fusion autotune cache. Inserting into it fails an atomic rename on shared
filesystems ("Failed to insert autotune cache: FAILED_PRECONDITION"), and XLA then
segfaults while compiling. That was seen with jaxlib 0.9.1 (the AlphaFold 3 image) on
BeeGFS scratch and on a job's TMPDIR alike. Keeping only JAX's own entries avoids the
crash and still carries the compiled model; XLA re-runs its autotuning, which takes
seconds.
"""

from __future__ import annotations

import os
from typing import Optional

from absl import logging

XLA_CACHES_ENV = "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"


def enable_persistent_compilation_cache(cache_dir) -> Optional[str]:
    """Use ``cache_dir`` as JAX's compile cache; return the directory used, or None.

    An empty value leaves caching off. A directory that cannot be created or written
    is logged and skipped instead of failing the prediction. A value of
    ``JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES`` set in the environment is respected.
    """
    if not cache_dir:
        return None
    cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir)))
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        logging.warning("Not using the JAX compilation cache at %s: %s", cache_dir, exc)
        return None
    if not os.access(cache_dir, os.W_OK | os.X_OK):
        logging.warning("Not using the JAX compilation cache at %s: not writable", cache_dir)
        return None

    import jax

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Keep every executable, however quick its compile: a model is many jits.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    if XLA_CACHES_ENV not in os.environ:
        try:
            jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
        except AttributeError:
            # Older JAX has no such option and writes no XLA caches alongside.
            pass
    logging.info("JAX compilation cache: %s", cache_dir)
    return cache_dir
