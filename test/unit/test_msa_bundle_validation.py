"""Validate persisted alignment data before finalization can trust it."""

import json

import pytest

from alphapulldown.features.feature_batch import (
    FeatureRequest,
    RNA,
    RNA_DATABASE_NAMES,
    read_msa_bundle,
)


@pytest.fixture
def bundle(tmp_path):
    payload = {
        "sequence": "ACDE",
        "provenance": {"fixture": "bundle-validation"},
        "unpairedMsa": ">query\nACDE\n>hit\nACDF\n",
        "pairedMsa": ">query\nACDE\n",
        "unpairedDatabaseRows": [
            {"name": "uniref90", "rows": 1},
            {"name": "mgnify", "rows": 0},
            {"name": "small_bfd", "rows": 0},
        ],
    }
    return tmp_path / "alpha_mmseqs_msa.json", payload


@pytest.mark.parametrize(
    "damage", ("missing_role", "duplicate_role", "unknown_role", "empty_paired", "empty_unpaired")
)
def test_unusable_alignment_data_is_deleted_before_finalization(bundle, damage):
    path, payload = bundle
    if damage == "missing_role":
        payload["unpairedDatabaseRows"].pop()
    elif damage == "duplicate_role":
        # Keep every required role and the correct total, but duplicate UniRef90.
        # A decoder keyed only by name would silently replace its nonempty span.
        payload["unpairedDatabaseRows"].append({"name": "uniref90", "rows": 0})
    elif damage == "unknown_role":
        payload["unpairedDatabaseRows"][-1]["name"] = "unknown"
    elif damage == "empty_paired":
        payload["pairedMsa"] = ""
    else:
        payload["unpairedMsa"] = ""
        payload["unpairedDatabaseRows"][0]["rows"] = 0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        read_msa_bundle(
            path.parent, FeatureRequest("alpha", "ACDE"), require_row_spans=True
        )

    assert not path.exists(), "An unusable bundle must be removed so the shard repairs it"


def test_valid_protein_bundle_keeps_its_query_only_pairing_alignment(bundle):
    path, payload = bundle
    path.write_text(json.dumps(payload))

    result = read_msa_bundle(
        path.parent, FeatureRequest("alpha", "ACDE"), require_row_spans=True
    )

    assert result == payload
    assert path.exists()


def test_rna_bundle_has_no_paired_alignment_and_uses_its_own_database_roles(bundle):
    path, payload = bundle
    payload.update(
        sequence="ACGU", moleculeType=RNA,
        unpairedMsa=">query\nACGU\n", pairedMsa="",
        unpairedDatabaseRows=[{"name": name, "rows": 0} for name in RNA_DATABASE_NAMES],
    )
    path.write_text(json.dumps(payload))

    result = read_msa_bundle(
        path.parent, FeatureRequest("alpha", "ACGU", RNA), require_row_spans=True
    )

    assert result == payload
    assert path.exists()


def test_template_search_failure_preserves_a_valid_af2_msa_bundle(bundle):
    pytest.importorskip("alphafold.data.pipeline", reason="needs AlphaFold 2")
    from alphapulldown.features.af2_feature_finalizer import (
        Af2FeatureFinalizationSettings,
        Af2FeatureFinalizer,
    )

    class FailingTemplateSearcher:
        input_format = "sto"
        output_format = "sto"

        def query(self, text):
            raise RuntimeError("template database is temporarily unavailable")

    path, payload = bundle
    encoded = json.dumps(payload)
    path.write_text(encoded)
    finalizer = Af2FeatureFinalizer(
        settings=Af2FeatureFinalizationSettings(
            output_dir=path.parent / "features", msa_input_dir=path.parent,
            max_template_date="2050-01-01", template_seqres_database_id="seqres-v1",
            template_mmcif_database_id="mmcif-v1",
        ),
        template_searcher=FailingTemplateSearcher(),
        template_featurizer=None,
    )

    result = finalizer.generate([FeatureRequest("alpha", "ACDE")])

    assert not result.written
    assert [(failure.name, failure.error) for failure in result.failures] == [
        ("alpha", "template database is temporarily unavailable")
    ]
    assert path.read_text() == encoded
