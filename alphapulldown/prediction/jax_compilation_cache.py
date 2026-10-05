"""Point JAX at a persistent on-disk compile cache that is safe on shared storage.

Both folding backends are JAX-compiled, and every new process compiles its models from
scratch: about 50 s per AlphaFold 3 token bucket, and minutes per AlphaFold 2 fold. JAX's
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

The command-line tools use a per-user cache unless told otherwise
(:func:`resolve_cache_dir`). That default directory is kept under
:data:`DEFAULT_MAX_BYTES`: AlphaFold 3 adds about 3 MB per token bucket, but AlphaFold 2
adds about 5 MB per new complex size, without bound.
"""

from __future__ import annotations

import os
from typing import Optional

from absl import logging

XLA_CACHES_ENV = "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"
CACHE_DIR_ENV = "JAX_COMPILATION_CACHE_DIR"
DEFAULT_MAX_BYTES = 10 * 1024**3
# Flag values that turn the cache off instead of naming a directory.
OFF_VALUES = frozenset({"", "none", "null", "false", "no", "off", "0"})


def user_cache_dir() -> str:
    """The per-user default: ``$XDG_CACHE_HOME`` (else ``~/.cache``)/alphapulldown/..."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "alphapulldown", "jax_compilation_cache")


def resolve_cache_dir(value) -> Optional[str]:
    """The cache directory for a ``--jax_compilation_cache_dir`` value.

    Not given (None): ``$JAX_COMPILATION_CACHE_DIR`` when set, else :func:`user_cache_dir`.
    An off value (``none``, ``false``, ``""``...): None, no cache. Anything else: as given.
    """
    if value is None:
        return os.environ.get(CACHE_DIR_ENV) or user_cache_dir()
    if str(value).strip().lower() in OFF_VALUES:
        return None
    return str(value)


def _prune_oldest(cache_dir: str, max_bytes: int) -> None:
    """Delete the oldest entries until the directory holds at most ``max_bytes``.

    Best effort and lock-free: another process may be reading or writing meanwhile, and
    an entry that disappears only costs JAX a recompile.
    """
    entries = []
    try:
        with os.scandir(cache_dir) as scan:
            for entry in scan:
                try:
                    if entry.is_file(follow_symlinks=False):
                        stat = entry.stat(follow_symlinks=False)
                        entries.append((stat.st_mtime, stat.st_size, entry.path))
                except OSError:
                    continue
    except OSError:
        return
    total = sum(size for _, size, _ in entries)
    if total <= max_bytes:
        return
    removed = 0
    for _, size, path in sorted(entries):
        if total <= max_bytes:
            break
        try:
            os.remove(path)
        except OSError:
            continue
        total -= size
        removed += 1
    logging.info("Pruned %d old entries from the JAX compilation cache at %s", removed, cache_dir)


def enable_persistent_compilation_cache(cache_dir) -> Optional[str]:
    """Use ``cache_dir`` as JAX's compile cache; return the directory used, or None.

    None or an off value leaves caching off. A directory that cannot be created or
    written is logged and skipped instead of failing the prediction. The per-user default
    directory is pruned to :data:`DEFAULT_MAX_BYTES`; a directory chosen explicitly never
    is. A value of ``JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES`` set in the environment is
    respected.
    """
    if cache_dir is None or str(cache_dir).strip().lower() in OFF_VALUES:
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
    if cache_dir == os.path.abspath(user_cache_dir()):
        _prune_oldest(cache_dir, DEFAULT_MAX_BYTES)

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
