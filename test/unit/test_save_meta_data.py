import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphapulldown.utils import save_meta_data


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("tool version 1.2.3", "1.2.3"),
        ("release 2.4", "2.4"),
        ("Kalign version 3.2", "3.2"),
        ("no version here", None),
    ],
)
def test_parse_version_matches_expected_patterns(output, expected):
    assert save_meta_data.parse_version(output) == expected


def test_get_program_version_tries_help_then_h(monkeypatch):
    calls = []
    responses = iter([
        SimpleNamespace(stdout="usage only", stderr=""),
        SimpleNamespace(stdout="", stderr="version 2.3.4"),
    ])

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return next(responses)

    monkeypatch.setattr(save_meta_data.subprocess, "run", fake_run)

    version = save_meta_data.get_program_version("/usr/bin/tool")

    assert version == "2.3.4"
    assert calls == [
        ["/usr/bin/tool", "--help"],
        ["/usr/bin/tool", "-h"],
    ]


def test_get_program_version_returns_none_when_subprocess_fails(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError("tool missing")

    monkeypatch.setattr(save_meta_data.subprocess, "run", fake_run)

    assert save_meta_data.get_program_version("/usr/bin/missing") is None


def test_get_metadata_for_binary_uses_program_version(monkeypatch):
    monkeypatch.setattr(save_meta_data, "get_program_version", lambda _: "9.9.9")

    metadata = save_meta_data.get_metadata_for_binary("jackhmmer_binary_path", "/usr/bin/jackhmmer")

    assert metadata == {"jackhmmer": {"version": "9.9.9"}}


def test_get_metadata_for_database_handles_pdb70_and_bfd(monkeypatch):
    monkeypatch.setattr(save_meta_data, "get_hash", lambda path: "799f308b20627088129847709f1abed6" if "bfd" in path else "pdb70hash")
    monkeypatch.setattr(save_meta_data, "get_last_modified_date", lambda path: "2024-05-01 00:00:00")

    pdb70 = save_meta_data.get_metadata_for_database("pdb70_database_path", "/db/pdb70")
    bfd = save_meta_data.get_metadata_for_database("bfd_database_path", "/db/bfd")

    assert pdb70["PDB70"]["version"] == "pdb70hash"
    assert pdb70["PDB70"]["release_date"] == "2024-05-01 00:00:00"
    assert bfd["BFD"]["version"] == save_meta_data.BFD_HASH_HHM_FFINDEX
    assert bfd["BFD"]["release_date"] == "AF2"


@pytest.mark.parametrize(
    ("key", "path", "expected_name", "expected_version"),
    [
        ("small_bfd_database_path", "/db/small_bfd", "Reduced BFD", None),
        ("uniprot_database_path", "/db/uniprot", "UniProt", None),
        ("uniref90_database_path", "/db/uniref90", "UniRef90", None),
        ("pdb_seqres_database_path", "/db/pdb_seqres", "PDB seqres", "seqreshash"),
    ],
)
def test_get_metadata_for_database_handles_other_named_databases(
    monkeypatch, key, path, expected_name, expected_version
):
    monkeypatch.setattr(save_meta_data, "get_last_modified_date", lambda _: "2024-04-02 00:00:00")
    monkeypatch.setattr(save_meta_data, "get_hash", lambda _: "seqreshash")

    metadata = save_meta_data.get_metadata_for_database(key, path)

    assert list(metadata) == [expected_name]
    assert metadata[expected_name]["release_date"] == "2024-04-02 00:00:00"
    assert metadata[expected_name]["version"] == expected_version


def test_get_metadata_for_database_handles_release_dated_databases(monkeypatch):
    monkeypatch.setattr(
        save_meta_data,
        "get_hash",
        lambda path: "unirefhash" if path.endswith("_hhm.ffindex") else None,
    )

    uniref30 = save_meta_data.get_metadata_for_database(
        "uniref30_database_path", "/db/UniRef30_2024_02"
    )
    mgnify = save_meta_data.get_metadata_for_database(
        "mgnify_database_path", "/db/mgy_clusters_2022_05"
    )

    assert uniref30["UniRef30"]["version"] == "unirefhash"
    assert "2024_02" in uniref30["UniRef30"]["location_url"][0]
    assert mgnify["MGnify"]["version"] == "2022_05"
    assert "2022_05" in mgnify["MGnify"]["location_url"][0]


def test_get_metadata_for_database_returns_empty_for_unknown_key():
    assert save_meta_data.get_metadata_for_database("custom_path", "/db/custom") == {}


def test_get_meta_dict_collects_other_software_databases_and_mmseqs(monkeypatch):
    class FakeDateTime(datetime.datetime):
        @classmethod
        def now(cls):
            return cls(2026, 3, 27, 10, 11, 12)

    monkeypatch.setattr(save_meta_data.datetime, "datetime", FakeDateTime)
    monkeypatch.setattr(
        save_meta_data,
        "get_metadata_for_binary",
        lambda k, v: {"jackhmmer": {"version": "1.0"}},
    )
    monkeypatch.setattr(
        save_meta_data,
        "get_metadata_for_database",
        lambda k, v: {"UniProt": {"version": "dbv"}},
    )

    metadata = save_meta_data.get_meta_dict(
        {
            "jackhmmer_binary_path": "/usr/bin/jackhmmer",
            "uniprot_database_path": "/db/uniprot",
            "template_mmcif_dir": "/db/mmcif",
            "use_mmseqs2": True,
            "test_flag": "ignored",
            "helpfull": "ignored",
            "use_cprofile_for_profiling": True,
            "none_value": None,
        }
    )

    assert metadata["software"]["jackhmmer"]["version"] == "1.0"
    assert metadata["software"]["AlphaPulldown"]["version"]
    assert metadata["software"]["AlphaFold"]["version"]
    assert metadata["databases"]["UniProt"]["version"] == "dbv"
    assert metadata["databases"]["ColabFold"]["version"] == "2026-03-27"
    assert metadata["other"]["jackhmmer_binary_path"] == "/usr/bin/jackhmmer"
    assert metadata["other"]["use_mmseqs2"] == "True"
    assert "test_flag" not in metadata["other"]
    assert "helpfull" not in metadata["other"]
    assert "use_cprofile_for_profiling" not in metadata["other"]
    assert "none_value" not in metadata["other"]
    assert metadata["date"] == "2026-03-27 10:11:12"


def test_get_last_modified_date_returns_none_for_missing_path(tmp_path):
    missing = tmp_path / "missing.txt"

    assert save_meta_data.get_last_modified_date(str(missing)) is None


def test_get_last_modified_date_returns_timestamp_for_regular_file(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("content", encoding="utf-8")

    assert save_meta_data.get_last_modified_date(str(path)) is not None


def test_get_last_modified_date_uses_globbed_directory_entries(monkeypatch):
    class FakeStat:
        def __init__(self, ts):
            self.st_mtime = ts

    class FakeEntry:
        def __init__(self, ts, is_file=True):
            self._ts = ts
            self._is_file = is_file

        def is_file(self):
            return self._is_file

        def stat(self):
            return FakeStat(self._ts)

    monkeypatch.setattr(save_meta_data.os.path, "exists", lambda _: True)
    monkeypatch.setattr(save_meta_data.os.path, "isfile", lambda _: False)
    monkeypatch.setattr(
        save_meta_data.glob,
        "glob",
        lambda pattern: [FakeEntry(10), FakeEntry(20), FakeEntry(5, is_file=False)],
    )

    result = save_meta_data.get_last_modified_date("/db/dir")

    assert result == datetime.datetime.fromtimestamp(20).strftime("%Y-%m-%d %H:%M:%S")


def test_get_hash_matches_md5_digest(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"AlphaPulldown")

    digest = save_meta_data.get_hash(str(path))

    assert digest == hashlib.md5(b"AlphaPulldown").hexdigest()
