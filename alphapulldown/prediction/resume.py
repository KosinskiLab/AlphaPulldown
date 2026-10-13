"""Writing prediction outputs so that an interrupted run can be resumed.

A resumed run decides what is already done from which files exist, so every file it
relies on has to appear whole or not at all: a job killed mid-write must not leave a
truncated file that looks finished.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import IO, Iterator, Union

PathLike = Union[str, "os.PathLike[str]"]


@contextlib.contextmanager
def open_atomic(path: PathLike, mode: str = "w", **open_kwargs) -> Iterator[IO]:
    """Write ``path`` through a hidden sibling file that replaces it only on success.

    The sibling (``.<name>.<random>.tmp``) never matches an output glob. If the body
    raises, it is removed and ``path`` keeps its previous content. Files are created
    with the usual umask permissions, not ``tempfile``'s owner-only ones, because
    prediction directories are often shared.
    """
    if mode not in ("w", "wb"):
        raise ValueError(f"open_atomic writes whole files, not mode {mode!r}")
    directory, name = os.path.split(os.path.abspath(os.fspath(path)))
    temporary_path = os.path.join(directory, f".{name}.{uuid.uuid4().hex[:12]}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, mode, **open_kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, os.path.join(directory, name))
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary_path)
        raise
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
