from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


def test_af2_image_installs_the_checked_out_source_and_smokes_batch_command():
    dockerfile = (REPOSITORY / "docker/alphafold2.dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY . /AlphaPulldown" in dockerfile
    assert "git clone --recurse-submodules https://github.com/KosinskiLab/AlphaPulldown" not in dockerfile
    assert "run_structure_prediction_batch.py --helpshort" in dockerfile


def test_pull_requests_build_af2_without_registry_or_ssh_credentials():
    workflow = (REPOSITORY / ".github/workflows/github_actions.yml").read_text(
        encoding="utf-8"
    )
    af2_job = workflow.split("  build-alphafold2-container:", 1)[1].split(
        "  # build-alphalink-container:", 1
    )[0]

    assert "submodules: recursive" in af2_job
    assert "if: github.event_name != 'pull_request'" in af2_job
    assert "if: github.event_name == 'pull_request'" in af2_job
    assert "push: false" in af2_job
