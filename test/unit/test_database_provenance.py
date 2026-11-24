import hashlib
import json
import os
import subprocess
import sys

import pytest

from alphapulldown.utils.database_provenance import read_database_manifest, write_database_manifest
from alphapulldown.utils.save_meta_data import get_metadata_for_database


def test_manifest_roundtrip_and_staleness(tmp_path):
    fasta = tmp_path / "uniprot.fasta"
    fasta.write_bytes(b">a\nACDE\n")
    manifest = write_database_manifest(fasta, "UniProt", "2024_04", release_date="2024-07-24",
                                       location_url=["https://example.org/archive.fa"])
    meta = get_metadata_for_database("uniprot_database_path", str(fasta))["UniProt"]
    assert meta["version"] == "2024_04"
    assert meta["release_date"] == "2024-07-24"
    assert meta["sha256"] == hashlib.sha256(fasta.read_bytes()).hexdigest()
    assert meta["version_source"] == "database_manifest"
    with pytest.raises(FileExistsError):
        write_database_manifest(fasta, "UniProt", "2025_01")
    assert json.loads(manifest.read_text())["version"] == "2024_04"
    st = fasta.stat()
    os.utime(fasta, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match="changed"):
        read_database_manifest(fasta, "UniProt")


def test_resolved_manifest_and_conflicting_filename(tmp_path):
    fasta = tmp_path / "uniref90_2024_04.fa"
    fasta.write_text(">a\nACDE\n")
    link = tmp_path / "uniref90.fasta"
    link.symlink_to(fasta)
    (tmp_path / "uniref90.fasta.metadata.json").write_text("invalid obsolete manifest")
    write_database_manifest(fasta, "UniRef90", "2024_04")
    meta = get_metadata_for_database("uniref90_database_path", str(link))["UniRef90"]
    assert meta["version"] == "2024_04"
    assert meta["resolved_path"] == str(fasta)
    write_database_manifest(fasta, "UniRef90", "2025_01", overwrite=True)
    with pytest.raises(ValueError, match="disagree"):
        get_metadata_for_database("uniref90_database_path", str(link))


@pytest.mark.parametrize("change", [dict(schema_version=2), dict(database="BFD"),
    dict(version=""), dict(sha256="invalid"), dict(location_url="url"), dict(release_date="yesterday")])
def test_invalid_manifests_fail_explicitly(tmp_path, change):
    fasta = tmp_path / "db.fa"
    fasta.write_text(">a\nA\n")
    manifest = write_database_manifest(fasta, "UniProt", "2024_04")
    data = json.loads(manifest.read_text())
    data.update(change)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid database manifest"):
        read_database_manifest(fasta, "UniProt")


def test_cli_registers_explicit_release(tmp_path):
    fasta = tmp_path / "uniref90.fasta"
    fasta.write_text(">a\nACDE\n")
    command = [sys.executable, "-m", "alphapulldown.scripts.create_database_metadata", str(fasta),
               "--database", "UniRef90", "--version", "2024_04"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert read_database_manifest(fasta, "UniRef90")["version"] == "2024_04"
    assert subprocess.run(command, capture_output=True).returncode != 0
