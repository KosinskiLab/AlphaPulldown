"""
Tools for creating ``.conda``-format archives.

Uses ``tempfile.SpooledTemporaryFile`` to buffer ``pkg-*.tar`` and
``info-*.tar``, then compress directly into an open `ZipFile` at the end.
`SpooledTemporaryFile` buffers the first 10MB of the package and its metadata in
memory, but writes out to disk for larger packages.

Uses more disk space than ``conda-package-handling`` (temporary uncompressed
tarballs of the package contents) but accepts streams instead of just
files-on-the-filesystem.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import zstandard

# increase to reduce speed and increase compression (levels above 19 use much
# more memory)
ZSTD_COMPRESS_LEVEL = 19
# increase to reduce compression and increase speed
ZSTD_COMPRESS_THREADS = 1

CONDA_PACKAGE_FORMAT_VERSION = 2

# Account for growth from "2 GB of /dev/urandom" to not exceed ZIP64_LIMIT after
# compression
CONDA_ZIP64_LIMIT = zipfile.ZIP64_LIMIT - (1 << 18) - 1


def anonymize(tarinfo: tarfile.TarInfo):
    """
    Pass to ``tarfile.add(..., filter=anonymize)`` to anonymize uid/gid.

    Does not anonymize mtime or any other field.
    """
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = ""
    return tarinfo


class CondaTarFile(tarfile.TarFile):
    """
    Subclass of :external+python:py:class:`tarfile.TarFile` that adds members to
    a second ``info`` tar if they match ``is_info(name)``.

    Create this with ``conda_builder(...)`` which sets up the component
    archives, then wraps them into a ``.conda`` on exit.

    Only useful for creating, not extracting ``.conda``.
    """

    info_tar: tarfile.TarFile
    is_info: Callable

    def __init__(
        self,
        *args,
        info_tar: tarfile.TarFile,
        is_info=lambda name: name.startswith("info/"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.info_tar = info_tar
        self.is_info = is_info

    def addfile(self, tarinfo, fileobj=None):
        """
        Add the TarInfo object ``tarinfo`` to the archive. If ``fileobj`` is
        given, it should be a binary file, and tarinfo.size bytes are read from
        it and added to the archive. You can create TarInfo objects directly, or
        by using ``gettarinfo()``.

        If ``self.is_info(tarinfo.name)`` returns ``True``, add ``tarinfo`` to
        ``self.info_tar`` instead.
        """
        if self.is_info(tarinfo.name):
            return self.info_tar.addfile(tarinfo, fileobj=fileobj)
        else:
            return super().addfile(tarinfo, fileobj)


@contextmanager
def conda_builder(
    stem,
    path,
    *,
    compressor: Callable[
        [], zstandard.ZstdCompressor
    ] = lambda: zstandard.ZstdCompressor(
        level=ZSTD_COMPRESS_LEVEL, threads=ZSTD_COMPRESS_THREADS
    ),
    is_info: Callable[[str], bool] = lambda filename: filename.startswith("info/"),
    encoding="utf-8",
) -> Iterator[CondaTarFile]:
    """
    Produce a ``TarFile`` subclass used to build a ``.conda`` package. The
    subclass delegates ``addfile()`` to the ``info-`` component when ``is_info``
    returns True.

    When the context manager exits, ``{path}/{stem}.conda`` is written with
    the component tar archives.

    Args:
        stem: output filename without extension

        path: destination path for transmuted .conda package compressor: A
            function that creates instances of ``zstandard.ZstdCompressor()``.

        encoding: passed to TarFile constructor. Keep default "utf-8" for valid
            .conda.

    Yields:
        ``CondaTarFile``
    """
    output_path = Path(path, f"{stem}.conda")
    with tempfile.SpooledTemporaryFile() as info_file, tempfile.SpooledTemporaryFile() as pkg_file:
        with tarfile.TarFile(
            fileobj=info_file, mode="w", encoding=encoding
        ) as info_tar, CondaTarFile(
            fileobj=pkg_file,
            mode="w",
            info_tar=info_tar,
            is_info=is_info,
            encoding=encoding,
        ) as pkg_tar:
            # If we wanted to compress these at a low setting to save temporary
            # space, we could insert a file object that counts bytes written in
            # front of a zstd (level between 1..3) compressor.
            yield pkg_tar

            info_tar.close()
            pkg_tar.close()

            info_size = info_file.tell()
            pkg_size = pkg_file.tell()

            info_file.seek(0)
            pkg_file.seek(0)

        with zipfile.ZipFile(
            output_path,
            "x",  # x to not append to existing
            compresslevel=zipfile.ZIP_STORED,
        ) as conda_file:
            # Use a maximum of one Zstd compressor, stream_writer at a time to save memory.
            data_compress = compressor()

            pkg_metadata = {"conda_pkg_format_version": CONDA_PACKAGE_FORMAT_VERSION}
            conda_file.writestr("metadata.json", json.dumps(pkg_metadata))

            with conda_file.open(
                f"pkg-{stem}.tar.zst",
                "w",
                force_zip64=(pkg_size > CONDA_ZIP64_LIMIT),
            ) as pkg_file_zip, data_compress.stream_writer(
                pkg_file_zip, size=pkg_size, closefd=False
            ) as pkg_stream:
                shutil.copyfileobj(pkg_file._file, pkg_stream)

            with conda_file.open(
                f"info-{stem}.tar.zst",
                "w",
                force_zip64=(info_size > CONDA_ZIP64_LIMIT),
            ) as info_file_zip, data_compress.stream_writer(
                info_file_zip,
                size=info_size,
                closefd=False,
            ) as info_stream:
                shutil.copyfileobj(info_file._file, info_stream)
