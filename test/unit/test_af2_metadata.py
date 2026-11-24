import copy

import pytest

from alphapulldown.utils import af2_metadata as metadata
from alphapulldown.utils.feature_metadata import retain_msa_provenance


def flags(**overrides):
    result = {key: "/db/" + name for key, name in metadata.MSA_DATABASES.items()}
    result.update(pdb70_database_path="/db/pdb70", pdb_seqres_database_path="/db/seqres",
                  template_mmcif_dir="/db/mmcif", use_mmseqs2=False, skip_msa=False,
                  use_hhsearch=False, db_preset="full_dbs")
    result.update(overrides)
    return result


@pytest.mark.parametrize("overrides,expected", [
    ({}, {"uniref90", "mgnify", "uniprot", "bfd", "uniref30", "pdb_seqres"}),
    ({"db_preset": "reduced_dbs"}, {"uniref90", "mgnify", "uniprot", "small_bfd", "pdb_seqres"}),
    ({"skip_msa": True}, {"pdb_seqres"}),
    ({"skip_msa": True, "use_hhsearch": True}, {"pdb70"}),
    ({"use_mmseqs2": True}, set()),
])
def test_only_selected_database_paths(overrides, expected):
    selected = metadata.filter_af2_metadata_flags(flags(**overrides))
    assert {key.removesuffix("_database_path") for key, value in selected.items()
            if key.endswith("_database_path") and value} == expected


def test_cached_alignments_do_not_acquire_current_releases(tmp_path, monkeypatch):
    (tmp_path / "uniref90_hits.sto").write_text("validated cache")
    (tmp_path / "uniprot_hits.sto").write_text("validated cache")
    cached = metadata.cached_msa_databases(tmp_path)
    assert cached == ["UniProt", "UniRef90"]
    selected = metadata.filter_af2_metadata_flags(flags(), cached_databases=cached)
    assert selected["uniref90_database_path"] is None
    assert selected["uniprot_database_path"] is None
    assert selected["mgnify_database_path"]
    monkeypatch.setattr(metadata.save_meta_data, "get_meta_dict", lambda _: {"databases": {}})
    result = metadata.get_af2_feature_metadata(flags(), cached_databases=cached)
    assert result["msa_provenance"] == dict(mode="cached_alignments", unknown_source=True, cached_databases=cached)


def test_reuse_keeps_original_source_and_new_templates_without_nesting():
    old = dict(date="2020-01-01", software={"AlphaPulldown": "old"},
               other=dict(use_mmseqs2="False", db_preset="full_dbs"),
               databases={"UniProt": {"version": "2020_01"}, "ColabFold": {},
                          "Reduced BFD": {}, "PDB seqres": {"version": "old"}})
    original = copy.deepcopy(old)
    current = dict(date="2026-01-01", software={"AlphaPulldown": "new"},
                   databases={"PDB seqres": {"version": "new"}})
    once = retain_msa_provenance(current, old)
    twice = retain_msa_provenance(current, once)
    assert once == twice
    assert once["databases"] == {"UniProt": {"version": "2020_01"}, "PDB seqres": {"version": "new"}}
    assert once["msa_provenance"]["source_date"] == "2020-01-01"
    assert once["msa_provenance"]["source_software"] == original["software"]
    assert old == original


@pytest.mark.parametrize("release_date", [None, "2024-09-01"])
def test_colabfold_survives_repeated_template_updates(release_date):
    old = dict(other={"use_mmseqs2": "True"}, databases={"ColabFold": {
        "version": "2024-09-01", "release_date": release_date,
        "location_url": ["https://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz"],
    }})
    original = copy.deepcopy(old)
    current = dict(other={"use_mmseqs2": "False"}, databases={})
    once = retain_msa_provenance(current, old)
    twice = retain_msa_provenance(current, once)
    assert once == twice
    assert twice["databases"] == {"ColabFold": {
        "version": None, "release_date": None, "location_url": [],
        "version_source": "not_reported_by_service",
    }}
    assert old == original
    assert not twice["msa_provenance"]["unknown_source"]


@pytest.mark.parametrize("name", ["UniProt", "UniRef90"])
@pytest.mark.parametrize("pipeline", ["alphafold2", "alphafold3"])
def test_reuse_normalizes_legacy_file_dates(name, pipeline):
    old = {"date": "2024-09-01", "other": {"data_pipeline": pipeline},
           "databases": {name: {"version": None, "release_date": "2022-12-12 04:05:06",
                                "location_url": ["https://example.org/db.fa"]}}}
    original = copy.deepcopy(old)
    once = retain_msa_provenance({"databases": {}}, old)
    twice = retain_msa_provenance({"databases": {}}, once)
    assert once == twice
    assert once["databases"][name] == {
        "version": None, "release_date": None, "file_modified_date": "2022-12-12 04:05:06",
        "location_url": ["https://example.org/db.fa"],
    }
    assert once["msa_provenance"]["source_date"] == "2024-09-01"
    assert old == original


@pytest.mark.parametrize("name,entry", [
    ("UniProt", {"version": "2024_04", "release_date": "2024-07-24", "version_source": "database_manifest"}),
    ("UniRef90", {"version": "2022_05", "release_date": None, "version_source": "resolved_filename"}),
    ("UniRef90", {"version": "2022_05", "release_date": None}),
    ("ColabFold", {"version": "verified-release", "release_date": "2024-07-24", "version_source": "service_response"}),
])
def test_reuse_preserves_explicitly_sourced_or_known_releases(name, entry):
    old = {"other": {"use_mmseqs2": "True"}, "databases": {name: entry}}
    assert retain_msa_provenance({"databases": {}}, old)["databases"][name] == entry


def test_missing_reused_metadata_stays_unknown(monkeypatch):
    captured = []
    def collect(selected):
        captured.append(selected)
        return {"databases": {}}
    monkeypatch.setattr(metadata.save_meta_data, "get_meta_dict", collect)
    result = metadata.get_af2_feature_metadata(flags(), reuse_features=True, direct_template="custom.cif")
    assert result["msa_provenance"]["unknown_source"]
    assert result["template_provenance"] == dict(mode="direct_mmcif", path="custom.cif")
    assert not any(value for key, value in captured[0].items() if key.endswith("_database_path"))


@pytest.mark.parametrize("value,expected", [(False, False), ("False", False), (True, True), ("true", True)])
def test_colabfold_flag_is_boolean(value, expected):
    selected = metadata.filter_af2_metadata_flags(flags(use_mmseqs2=value))
    assert selected["use_mmseqs2"] is expected
    cached = metadata.filter_af2_metadata_flags(flags(use_mmseqs2=value), cached_databases=["ColabFold"])
    assert cached["use_mmseqs2"] is False


def test_local_mmseqs_snapshot_survives_template_refresh():
    backend = {"msa_backend": "mmseqs2-gpu", "mmseqs2_gpu": {"databases": {"uniref": "digest"}}}
    old = {"other": backend, "databases": {"PDB seqres": {"version": "old"}}}
    once = retain_msa_provenance({"databases": {}}, old)
    twice = retain_msa_provenance({"databases": {}}, once)
    assert twice["msa_provenance"]["source_backend"] == backend
    assert not twice["msa_provenance"]["unknown_source"]
    assert not twice["databases"]


def test_cache_from_unused_preset_is_not_reported(monkeypatch):
    monkeypatch.setattr(metadata.save_meta_data, "get_meta_dict", lambda _: {"databases": {}})
    result = metadata.get_af2_feature_metadata(flags(db_preset="reduced_dbs"), cached_databases=["BFD", "UniRef30"])
    assert result["msa_provenance"] == {"mode": "generated"}
