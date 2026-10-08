import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from alphapulldown.scripts import split_jobs_into_clusters as split_jobs  # noqa: E402

FEATURES = Path(__file__).resolve().parents[1] / "test_data" / "features"


def test_jobs_are_profiled_from_their_features_and_binned_by_length(tmp_path):
    args = SimpleNamespace(
        features_directory=[str(FEATURES)], protein_delimiter="+", output_dir=str(tmp_path)
    )

    profile = split_jobs.profile_all_jobs_and_cluster(["TEST", "TEST+TEST"], args)

    monomer, dimer = profile.to_dict("records")
    assert monomer["name"] == "TEST" and dimer["name"] == "TEST+TEST"
    assert dimer["seq_length"] == 2 * monomer["seq_length"]
    assert monomer["msa_depth"] > 0 and dimer["msa_depth"] > 0

    split_jobs.cluster_jobs(["TEST", "TEST+TEST"], args)

    # Both lengths fall in one 150-residue bin, so one cluster lists both jobs.
    (cluster_file,) = tmp_path.glob("job_cluster*.txt")
    assert cluster_file.read_text().split() == ["TEST", "TEST+TEST"]
    assert (tmp_path / "clustered_prediction_jobs.png").stat().st_size > 0


def test_command_line_reads_protein_lists(tmp_path, monkeypatch):
    protein_list = tmp_path / "proteins.txt"
    protein_list.write_text("TEST\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_jobs_into_clusters.py",
            "--protein_lists", str(protein_list),
            "--mode", "custom",
            "--features_directory", str(FEATURES),
            "--output_dir", str(output_dir),
        ],
    )

    split_jobs.main()

    (cluster_file,) = output_dir.glob("job_cluster*.txt")
    assert cluster_file.read_text().split() == ["TEST"]
