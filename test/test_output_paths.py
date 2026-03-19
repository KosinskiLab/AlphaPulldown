import json

from alphapulldown.utils.output_paths import (
    derive_af3_job_name_from_json,
    resolve_af3_json_output_dir,
    sanitise_af3_job_name,
)


def test_derive_af3_job_name_from_json_uses_sanitised_json_name(tmp_path):
    json_path = tmp_path / "input.json"
    json_path.write_text(
        json.dumps(
            {
                "name": "My Job (Chain A+B)",
                "dialect": "alphafold3",
                "version": 1,
            }
        ),
        encoding="utf-8",
    )

    assert derive_af3_job_name_from_json(str(json_path)) == "my_job_chain_ab"


def test_derive_af3_job_name_from_json_falls_back_to_file_stem(tmp_path):
    json_path = tmp_path / "Complex Name.json"
    json_path.write_text("{", encoding="utf-8")

    assert derive_af3_job_name_from_json(str(json_path)) == "complex_name"


def test_sanitise_af3_job_name_rewrites_dot_only_relative_names():
    assert sanitise_af3_job_name(".") == "ranked_0"
    assert sanitise_af3_job_name("..") == "ranked_0"


def test_derive_af3_job_name_from_json_rejects_dot_only_json_name(tmp_path):
    json_path = tmp_path / "Safe Name.json"
    json_path.write_text(
        json.dumps(
            {
                "name": "..",
                "dialect": "alphafold3",
                "version": 1,
            }
        ),
        encoding="utf-8",
    )

    assert derive_af3_job_name_from_json(str(json_path)) == "safe_name"


def test_resolve_af3_json_output_dir_nests_only_for_shared_ap_style_root(tmp_path):
    json_path = tmp_path / "input.json"
    json_path.write_text(
        json.dumps(
            {
                "name": "Pair One",
                "dialect": "alphafold3",
                "version": 1,
            }
        ),
        encoding="utf-8",
    )

    shared_root = str(tmp_path / "predictions")
    per_job_root = str(tmp_path / "predictions" / "pair_one")

    assert resolve_af3_json_output_dir(
        str(json_path),
        shared_root,
        use_ap_style=True,
        shared_output_root=True,
    ) == str(tmp_path / "predictions" / "pair_one")
    assert resolve_af3_json_output_dir(
        str(json_path),
        per_job_root,
        use_ap_style=True,
        shared_output_root=False,
    ) == per_job_root
    assert resolve_af3_json_output_dir(
        str(json_path),
        shared_root,
        use_ap_style=False,
        shared_output_root=True,
    ) == shared_root


def test_resolve_af3_json_output_dir_keeps_unsafe_json_name_within_root(tmp_path):
    json_path = tmp_path / "Input Name.json"
    json_path.write_text(
        json.dumps(
            {
                "name": "..",
                "dialect": "alphafold3",
                "version": 1,
            }
        ),
        encoding="utf-8",
    )

    assert resolve_af3_json_output_dir(
        str(json_path),
        str(tmp_path / "predictions"),
        use_ap_style=True,
        shared_output_root=True,
    ) == str(tmp_path / "predictions" / "input_name")
