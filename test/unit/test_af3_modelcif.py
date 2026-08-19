from pathlib import Path

import pytest

from alphapulldown.utils.af3_modelcif import (
    augment_af3_modelcif_file,
    build_alphapulldown_mmcif_updates,
    find_af3_modelcif_files,
)
from alphapulldown.utils.feature_metadata import encode_metadata_in_description


METADATA_AF3 = {
    "software": {
        "AlphaPulldown": {"version": "2.5.0"},
        "AlphaFold": {"version": "3.0.2"},
        "jackhmmer": {"version": "3.4"},
    },
    "databases": {
        "UniRef90": {
            "location_url": ["https://example.test/uniref90"],
            "version": "2022_05",
            "release_date": "2022-05-01 00:00:00",
        }
    },
    "other": {
        "data_pipeline": "alphafold3",
        "use_precomputed_msas": "False",
        "max_template_date": "2021-09-30",
    },
}


METADATA_AF2 = {
    "software": {
        "AlphaPulldown": {"version": "2.5.0"},
        "AlphaFold": {"version": "2.3.2"},
        "hhblits": {"version": "3.3.0"},
    },
    "databases": {},
    "other": {"data_pipeline": "alphafold2", "db_preset": "full_dbs"},
}


def _baseline_cif():
    mmcif = pytest.importorskip("alphafold3.structure.mmcif")
    return mmcif.from_string(
        """data_test
_entry.id test
_ma_data.id 1
_ma_data.name Model
_ma_data.content_type 'model coordinates'
_software.pdbx_ordinal 1
_software.name AlphaFold
_software.classification other
_software.description 'Structure prediction'
_software.version 3.0.2
_software.type package
_software.date ?
_ma_software_group.ordinal_id 1
_ma_software_group.group_id 1
_ma_software_group.software_id 1
loop_
_ma_protocol_step.ordinal_id
_ma_protocol_step.protocol_id
_ma_protocol_step.step_id
_ma_protocol_step.method_type
1 1 1 'coevolution MSA'
2 1 2 'template search'
3 1 3 modeling
#
"""
    )


def test_build_updates_covers_af3_and_af2_feature_provenance():
    updates = build_alphapulldown_mmcif_updates(
        _baseline_cif(), [METADATA_AF3, METADATA_AF2]
    )

    assert updates["_software.name"] == [
        "AlphaFold",
        "AlphaPulldown",
        "jackhmmer",
        "AlphaFold 2",
        "hhblits",
    ]
    assert updates["_ma_data_ref_db.name"] == ["UniRef90"]
    assert updates["_ma_data_ref_db.release_date"] == ["2022-05-01"]
    assert updates["_ma_protocol_step.software_group_id"] == ["2", "2", "3"]
    assert "--data_pipeline" in updates["_ma_software_parameter.name"]
    assert set(updates["_ma_software_group.group_id"]) == {"1", "2", "3"}


def test_build_updates_recognises_legacy_af2_metadata_without_pipeline_flag():
    legacy_af2 = {
        "software": {
            "AlphaPulldown": {"version": "2.5.0"},
            "AlphaFold": {"version": "2.3.2"},
        },
        "databases": {},
        "other": {},
    }

    updates = build_alphapulldown_mmcif_updates(_baseline_cif(), [legacy_af2])

    assert updates["_software.name"] == [
        "AlphaFold",
        "AlphaPulldown",
        "AlphaFold 2",
    ]


def test_build_updates_uses_omitted_value_for_unknown_database_release_date():
    metadata = {
        "software": {},
        "databases": {
            "Legacy DB": {
                "location_url": ["https://example.test/legacy"],
                "version": "unknown",
                "release_date": "NA",
            }
        },
        "other": {},
    }

    updates = build_alphapulldown_mmcif_updates(_baseline_cif(), [metadata])

    # A CIF unknown value (`?`) is an object in python-modelcif and is not a
    # parseable ISO date. The omitted value (`.`) correctly maps to None.
    assert updates["_ma_data_ref_db.release_date"] == ["."]


def test_build_updates_removes_transport_envelope_from_entity_description():
    cif = _baseline_cif().copy_and_update(
        {
            "_entity.id": ["1", "2"],
            "_entity.type": ["polymer", "polymer"],
            "_entity.pdbx_description": [
                encode_metadata_in_description("Human-readable name", METADATA_AF3),
                "Unaffected entity",
            ],
        }
    )

    updates = build_alphapulldown_mmcif_updates(cif, [METADATA_AF3])

    assert updates["_entity.pdbx_description"] == [
        "Human-readable name",
        "Unaffected entity",
    ]


def test_augment_real_af3_modelcif_preserves_comments_and_is_modelcif_readable(
    tmp_path,
):
    pytest.importorskip("alphafold3.structure.mmcif")
    modelcif_reader = pytest.importorskip("modelcif.reader")
    fixture = (
        Path(__file__).resolve().parents[1]
        / "test_data/templates/ranbp5_pb1_181_216_noxl_af3_dertemp_model.cif"
    )
    cif_path = tmp_path / "job_model.cif"
    cif_path.write_bytes(fixture.read_bytes())
    original_prefix = cif_path.read_text(encoding="utf-8").split("data_", 1)[0]

    assert augment_af3_modelcif_file(cif_path, [METADATA_AF3]) is True
    rendered = cif_path.read_text(encoding="utf-8")
    assert rendered.startswith(original_prefix)

    with cif_path.open(encoding="utf-8") as handle:
        systems = modelcif_reader.read(handle)
    assert len(systems) == 1
    assert [software.name for software in systems[0].software] == [
        "AlphaFold",
        "AlphaPulldown",
        "jackhmmer",
    ]


def test_augment_is_noop_without_metadata(tmp_path):
    path = tmp_path / "model.cif"
    path.write_text("unchanged", encoding="utf-8")
    assert augment_af3_modelcif_file(path, []) is False
    assert path.read_text(encoding="utf-8") == "unchanged"


def test_find_af3_modelcif_files_covers_best_and_samples(tmp_path):
    best = tmp_path / "job_model.cif"
    sample = tmp_path / "seed-1_sample-0/model.cif"
    sample.parent.mkdir()
    best.write_text("best", encoding="utf-8")
    sample.write_text("sample", encoding="utf-8")
    (tmp_path / "other.cif").write_text("other", encoding="utf-8")

    assert find_af3_modelcif_files(tmp_path, "job") == [best, sample]
