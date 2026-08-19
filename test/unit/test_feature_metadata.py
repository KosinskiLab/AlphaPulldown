import json
import lzma
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from alphapulldown.utils.feature_metadata import (
    AF3_METADATA_MARKER,
    decode_metadata_from_description,
    embed_metadata_in_af3_json,
    encode_metadata_in_description,
    extract_metadata_from_af3_json,
    extract_metadata_from_fold_input,
    find_feature_metadata,
    load_feature_metadata_sidecars,
)


METADATA = {
    "databases": {"UniRef90": {"version": "2022_05"}},
    "software": {
        "AlphaPulldown": {"version": "2.5.0"},
        "AlphaFold": {"version": "3.0.2"},
    },
    "other": {"data_pipeline": "alphafold3", "max_template_date": "2021-09-30"},
}


def _af3_payload(polymer_type="protein"):
    polymer = {
        "id": "A",
        "sequence": "ACDE",
        "description": "original description",
    }
    if polymer_type == "protein":
        polymer.update(
            {
                "modifications": [],
                "unpairedMsa": "",
                "pairedMsa": "",
                "templates": [],
            }
        )
    else:
        polymer["modifications"] = []
        if polymer_type == "rna":
            polymer["unpairedMsa"] = ""
    return {
        "dialect": "alphafold3",
        "version": 1,
        "name": "compatibility_test",
        "modelSeeds": [1],
        "sequences": [{polymer_type: polymer}],
        "bondedAtomPairs": [],
        "userCCD": None,
    }


def test_description_envelope_round_trips_and_is_idempotent():
    encoded = encode_metadata_in_description("protein A", METADATA)
    assert AF3_METADATA_MARKER in encoded
    assert decode_metadata_from_description(encoded) == ("protein A", METADATA)

    encoded_again = encode_metadata_in_description(encoded, METADATA)
    assert encoded_again == encoded
    assert encoded_again.count(AF3_METADATA_MARKER) == 1


def test_malformed_or_foreign_description_is_left_untouched():
    malformed = f"protein A{AF3_METADATA_MARKER}not-json"
    assert decode_metadata_from_description(malformed) == (malformed, None)
    assert decode_metadata_from_description("ordinary") == ("ordinary", None)
    assert decode_metadata_from_description(None) == (None, None)


@pytest.mark.parametrize("polymer_type", ["protein", "rna", "dna"])
def test_embed_metadata_uses_only_standard_af3_polymer_description(polymer_type):
    payload = _af3_payload(polymer_type)
    embedded = embed_metadata_in_af3_json(payload, METADATA)

    assert set(embedded) == set(payload)
    assert "__meta__" not in embedded
    assert extract_metadata_from_af3_json(embedded) == [METADATA]
    original_description, recovered = decode_metadata_from_description(
        embedded["sequences"][0][polymer_type]["description"]
    )
    assert original_description == "original description"
    assert recovered == METADATA
    assert payload["sequences"][0][polymer_type]["description"] == "original description"


def test_embedded_json_is_accepted_and_round_tripped_by_vanilla_alphafold3():
    embedded = embed_metadata_in_af3_json(_af3_payload(), METADATA)
    script = """
import json
import sys
from alphafold3.common import folding_input

payload = json.load(sys.stdin)
parsed = folding_input.Input.from_json(json.dumps(payload))
assert parsed.chains[0].sequence == "ACDE"
try:
    folding_input.Input.from_json(json.dumps(dict(payload, __meta__={})))
except ValueError as exc:
    assert "Unexpected JSON keys" in str(exc)
else:
    raise AssertionError("vanilla AF3 unexpectedly accepted a custom top-level key")
print(parsed.to_json())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(embedded),
        text=True,
        capture_output=True,
        check=False,
    )
    unavailable_errors = (
        "No module named 'alphafold3",
        "cannot import name 'Self' from 'typing'",
    )
    if result.returncode != 0 and any(
        error in result.stderr for error in unavailable_errors
    ):
        pytest.skip("vanilla AF3 parser is unavailable in this test environment")
    assert result.returncode == 0, result.stderr
    round_tripped = json.loads(result.stdout)
    assert extract_metadata_from_af3_json(round_tripped) == [METADATA]


def test_extract_metadata_from_fold_input_handles_mixed_and_legacy_chains():
    fold_input = SimpleNamespace(
        chains=[
            SimpleNamespace(description=encode_metadata_in_description("A", METADATA)),
            SimpleNamespace(description="legacy input"),
        ]
    )
    assert extract_metadata_from_fold_input(fold_input) == [METADATA]


def test_af2_sidecar_readers_support_xz_and_select_newest(tmp_path):
    old_path = tmp_path / "protA_feature_metadata_2026-01-01.json"
    old_path.write_text(json.dumps({"source": "old"}), encoding="utf-8")
    new_path = tmp_path / "protA_feature_metadata_2026-01-02.json.xz"
    with lzma.open(new_path, "wt", encoding="utf-8") as handle:
        json.dump({"source": "new"}, handle)
    os.utime(old_path, (1, 1))
    os.utime(new_path, (2, 2))

    assert find_feature_metadata("protA", [tmp_path]) == {"source": "new"}
    assert load_feature_metadata_sidecars(tmp_path) == [
        {"source": "old"},
        {"source": "new"},
    ]


def test_embed_rejects_non_af3_or_ligand_only_payloads():
    with pytest.raises(ValueError, match="sequences list"):
        embed_metadata_in_af3_json({"plain": "mapping"}, METADATA)
    with pytest.raises(ValueError, match="polymer entity"):
        embed_metadata_in_af3_json(
            {"sequences": [{"ligand": {"id": "L", "ccdCodes": ["ATP"]}}]},
            METADATA,
        )
