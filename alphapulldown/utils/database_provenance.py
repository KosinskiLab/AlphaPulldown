"""Explicit release assertions bound to a particular local FASTA snapshot."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def read_database_manifest(path, database):
    """Read a manifest next to the resolved file, rejecting stale assertions.

    Size and nanosecond mtime are cheap freshness checks, not a cryptographic
    verification. SHA-256 is calculated once by the manifest writer, never once
    per protein. The dataset must remain immutable after registration.
    """
    target = Path(path).resolve()
    manifest = Path(str(target) + ".metadata.json")
    if not manifest.exists():
        return None
    try:
        with manifest.open() as handle:
            content = handle.read(65537)
        if len(content) > 65536:
            raise ValueError("manifest exceeds 64 KiB")
        data = json.loads(content)
        if data.get("schema_version") != 1 or data.get("database") != database:
            raise ValueError("wrong schema or database")
        if not isinstance(data.get("version"), str) or not data["version"].strip():
            raise ValueError("an explicit release identifier is required")
        if data.get("release_date") is not None:
            datetime.date.fromisoformat(data["release_date"])
        if not re.fullmatch(r"[0-9a-f]{64}", data.get("sha256", "")):
            raise ValueError("missing SHA-256")
        st = target.stat()
        if data.get("size_bytes") != st.st_size or data.get("mtime_ns") != st.st_mtime_ns:
            raise ValueError("database changed since the manifest was written")
        urls = data.get("location_url", [])
        if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
            raise ValueError("location_url must be a list of source URLs")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid database manifest {manifest}: {exc}") from exc
    return {
        "version": data["version"],
        "release_date": data.get("release_date"),
        "location_url": urls,
        "sha256": data["sha256"],
        "version_source": "database_manifest",
        "manifest_path": str(manifest),
    }


def write_database_manifest(path, database, version, *, release_date=None,
                            location_url=(), overwrite=False):
    """Hash an immutable file once and attach the caller's known release ID."""
    if database not in {"UniProt", "UniRef90"} or not isinstance(version, str) or not version.strip():
        raise ValueError("database must be UniProt/UniRef90 and version must be known")
    if isinstance(location_url, str) or any(not isinstance(url, str) for url in location_url):
        raise ValueError("location_url must be a sequence of source URLs")
    if release_date is not None:
        datetime.date.fromisoformat(release_date)
    target = Path(path).resolve(strict=True)
    before = target.stat()
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = target.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("database changed while hashing")
    data = dict(schema_version=1, database=database, version=version,
                release_date=release_date, location_url=list(location_url),
                size_bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                sha256=digest.hexdigest())
    destination = Path(str(target) + ".metadata.json")
    fd, temporary = tempfile.mkstemp(prefix=".database-manifest-", dir=target.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)  # Atomic create; never clobber a manifest.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
