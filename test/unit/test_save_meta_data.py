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


@pytest.mark.parametrize(
    ("key", "path", "expected_name", "expected_version"),
    [
        (
            "ntrna_database_path",
            "/db/nt_rna_2023_02_23_clust.fasta",
            "NT-RNA",
            "2023_02_23",
        ),
        ("rfam_database_path", "/db/rfam_14_9.fasta", "Rfam", "14_9"),
    ],
)
def test_get_metadata_for_database_handles_af3_databases(
    monkeypatch, key, path, expected_name, expected_version
):
    monkeypatch.setattr(
        save_meta_data,
        "get_last_modified_date",
        lambda _: "2026-08-19 12:00:00",
    )

    metadata = save_meta_data.get_metadata_for_database(key, path)

    assert list(metadata) == [expected_name]
    assert metadata[expected_name]["version"] == expected_version
    assert metadata[expected_name]["release_date"] == "2026-08-19 12:00:00"
    assert metadata[expected_name]["location_url"]


def test_get_metadata_for_database_uses_official_af3_bundle_facts(tmp_path):
    bundle_root = tmp_path / "3.0.0"
    bundle_root.mkdir()
    for filename in save_meta_data.AF3_BUNDLE_SIGNATURE_FILES:
        (bundle_root / filename).write_text("fixture", encoding="utf-8")
    mmcif_dir = bundle_root / "mmcif_files"
    mmcif_dir.mkdir()

    rna_metadata = save_meta_data.get_metadata_for_database(
        "rna_central_database_path",
        str(bundle_root / "rnacentral_active_seq_id_90_cov_80_linclust.fasta"),
    )["RNAcentral"]
    mmcif_metadata = save_meta_data.get_metadata_for_database(
        "template_mmcif_dir", str(mmcif_dir)
    )["PDB mmCIF"]

    assert rna_metadata["version"] == "21_0"
    assert rna_metadata["location_url"] == save_meta_data.DB_NAME_TO_URL["RNAcentral"]
    assert mmcif_metadata == {
        "release_date": "2022-09-28",
        "version": "2022-09-28",
        "location_url": save_meta_data.DB_NAME_TO_URL["PDB mmCIF"],
    }

    custom_mmcif_dir = bundle_root / "updated_mmcif_mirror"
    custom_mmcif_dir.mkdir()
    assert save_meta_data.get_metadata_for_database(
        "template_mmcif_dir", str(custom_mmcif_dir)
    )["PDB mmCIF"] == {
        "release_date": None,
        "version": None,
        "location_url": [],
    }


def test_get_metadata_for_database_does_not_label_custom_af3_mirrors(tmp_path):
    custom_root = tmp_path / "custom"
    custom_root.mkdir()
    rna_path = custom_root / "rnacentral.fasta"
    rna_path.write_text("fixture", encoding="utf-8")
    mmcif_dir = custom_root / "mmcif_files"
    mmcif_dir.mkdir()

    rna_metadata = save_meta_data.get_metadata_for_database(
        "rna_central_database_path", str(rna_path)
    )["RNAcentral"]
    mmcif_metadata = save_meta_data.get_metadata_for_database(
        "template_mmcif_dir", str(mmcif_dir)
    )["PDB mmCIF"]

    assert rna_metadata["version"] is None
    assert rna_metadata["location_url"] == []
    assert rna_metadata["release_date"] is not None
    assert mmcif_metadata == {
        "release_date": None,
        "version": None,
        "location_url": [],
    }


def test_get_metadata_for_database_parses_custom_af3_versions_from_paths(tmp_path):
    rna_metadata = save_meta_data.get_metadata_for_database(
        "rna_central_database_path",
        str(tmp_path / "rnacentral_25_0.fasta"),
    )["RNAcentral"]
    mmcif_metadata = save_meta_data.get_metadata_for_database(
        "template_mmcif_dir",
        str(tmp_path / "pdb_2026_07_15_mmcif_files"),
    )["PDB mmCIF"]

    assert rna_metadata["version"] == "25_0"
    assert mmcif_metadata["version"] == "2026-07-15"
    assert mmcif_metadata["release_date"] == "2026-07-15"


def test_get_metadata_for_database_uses_af3_pdb_seqres_release_without_hashing(
    monkeypatch,
):
    def fail_if_called(_):
        raise AssertionError("AF3 PDB seqres should not be hashed")

    monkeypatch.setattr(save_meta_data, "get_hash", fail_if_called)

    metadata = save_meta_data.get_metadata_for_database(
        "pdb_seqres_database_path",
        "/db/pdb_seqres_2022_09_28.fasta",
    )["PDB seqres"]

    assert metadata["version"] == "2022_09_28"
    assert metadata["release_date"] == "2022-09-28"
    assert "pdb_seqres_2022_09_28" in metadata["location_url"][0]


def test_af3_pdb_seqres_release_follows_a_symlink(tmp_path, monkeypatch):
    """A refreshed database behind AF3's pinned name must report its real date.

    AF3's fetch_databases.sh pins pdb_seqres_2022_09_28.fasta, so sites that
    update the database in place keep that name as a symlink to the current
    file. Reporting the pinned name's date would put a wrong template cutoff
    into the metadata and from there into a paper's methods.
    """
    monkeypatch.setattr(
        save_meta_data, "get_hash", lambda _: pytest.fail("should not hash")
    )

    real = tmp_path / "pdb_seqres_2026_08_19.fasta"
    real.write_text(">1abc_A\nACDE\n")
    pinned = tmp_path / "pdb_seqres_2022_09_28.fasta"
    pinned.symlink_to(real)

    metadata = save_meta_data.get_metadata_for_database(
        "pdb_seqres_database_path", str(pinned)
    )["PDB seqres"]

    assert metadata["version"] == "2026_08_19"
    assert metadata["release_date"] == "2026-08-19"
    # Both ends of the symlink are recorded so the substitution stays auditable.
    assert metadata["configured_path"] == str(pinned)
    assert metadata["resolved_path"] == str(real)


def test_af3_pdb_seqres_plain_path_records_no_symlink_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        save_meta_data, "get_hash", lambda _: pytest.fail("should not hash")
    )
    real = tmp_path / "pdb_seqres_2022_09_28.fasta"
    real.write_text(">1abc_A\nACDE\n")

    metadata = save_meta_data.get_metadata_for_database(
        "pdb_seqres_database_path", str(real)
    )["PDB seqres"]

    assert metadata["version"] == "2022_09_28"
    assert "configured_path" not in metadata
    assert "resolved_path" not in metadata


def test_resolve_database_path_handles_missing_path():
    """A configured path that does not exist must not raise during metadata."""
    resolved, via_symlink = save_meta_data.resolve_database_path("/no/such/db.fasta")
    assert resolved.endswith("db.fasta")
    assert via_symlink is False


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


def test_get_meta_dict_does_not_attribute_colabfold_when_mmseqs_is_disabled():
    metadata = save_meta_data.get_meta_dict({"use_mmseqs2": False})

    assert "ColabFold" not in metadata["databases"]
    assert metadata["other"]["use_mmseqs2"] == "False"


def test_get_last_modified_date_returns_none_for_missing_path(tmp_path):
    missing = tmp_path / "missing.txt"

    assert save_meta_data.get_last_modified_date(str(missing)) is None


def test_get_last_modified_date_returns_timestamp_for_regular_file(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("content", encoding="utf-8")

    assert save_meta_data.get_last_modified_date(str(path)) is not None


def test_get_last_modified_date_uses_globbed_directory_entries(monkeypatch):
    monkeypatch.setattr(save_meta_data.os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        save_meta_data.os.path,
        "isfile",
        lambda path: path in {"/db/dir/a", "/db/dir/b"},
    )
    monkeypatch.setattr(
        save_meta_data.os.path,
        "getmtime",
        lambda path: {"/db/dir/a": 10, "/db/dir/b": 20}[path],
    )
    monkeypatch.setattr(
        save_meta_data.glob,
        "glob",
        lambda pattern: ["/db/dir/a", "/db/dir/b", "/db/dir/subdir"],
    )

    result = save_meta_data.get_last_modified_date("/db/dir")

    assert result == datetime.datetime.fromtimestamp(20).strftime("%Y-%m-%d %H:%M:%S")


def test_get_hash_matches_md5_digest(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"AlphaPulldown")

    digest = save_meta_data.get_hash(str(path))

    assert digest == hashlib.md5(b"AlphaPulldown").hexdigest()


def test_get_hash_returns_none_for_missing_database_file(tmp_path):
    assert save_meta_data.get_hash(str(tmp_path / "missing")) is None
