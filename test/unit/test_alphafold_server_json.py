import json
from pathlib import Path

import pytest

from alphapulldown.utils.alphafold_server_json import (
    build_alphafold_server_jobs,
    build_alphafold_server_job,
    write_jobs_to_json_files,
)


TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_build_server_job_collapses_homodimer_features():
    jobs = build_alphafold_server_jobs(
        protein_lists=[str(TEST_DATA / "protein_lists" / "test_dimer.txt")],
        monomer_directories=[str(TEST_DATA / "features")],
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job["dialect"] == "alphafoldserver"
    assert job["version"] == 1
    assert job["modelSeeds"] == []
    assert job["name"] == "TEST_and_TEST"
    assert job["sequences"] == [
        {
            "proteinChain": {
                "sequence": "MESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA",
                "count": 2,
            }
        }
    ]


def test_build_server_job_slices_regions_for_fragments():
    job = build_alphafold_server_jobs(
        protein_lists=[str(TEST_DATA / "protein_lists" / "test_dimer_chopped.txt")],
        monomer_directories=[str(TEST_DATA / "features")],
    )[0]

    assert job["name"] == "TEST_and_A0A075B6L2"
    assert (
        job["sequences"][0]["proteinChain"]["sequence"]
        == "MESAIAEGGASRFSASSGGGGSRGAPQHYPKTAGNSEFLGKTPGQNAQKWIPARSTRRDDNSAA"
    )
    assert job["sequences"][0]["proteinChain"]["count"] == 1
    assert job["sequences"][1]["proteinChain"]["sequence"] == "MPLVVAVIFFPLVVLWVF"
    assert job["sequences"][1]["proteinChain"]["count"] == 1


def test_build_server_job_converts_af3_json_inputs_for_dna():
    jobs = build_alphafold_server_jobs(
        protein_lists=[str(TEST_DATA / "protein_lists" / "test_monomer_with_dna.txt")],
        monomer_directories=[str(TEST_DATA / "features")],
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job["name"] == "A0A024R1R8_and_dna"
    assert job["sequences"] == [
        {
            "proteinChain": {
                "sequence": "MSSHEGGKKKALKQPKKQAKEMDEEEKAFKQKQKEEQKKLEVLKAKVVGKGPLATGGIKKSGKK",
                "count": 1,
            }
        },
        {"dnaSequence": {"sequence": "GATTACA", "count": 1}},
        {"dnaSequence": {"sequence": "TGTAATC", "count": 1}},
    ]


def test_build_server_job_converts_local_af3_json_homodimer():
    jobs = build_alphafold_server_jobs(
        protein_lists=[
            str(TEST_DATA / "protein_lists" / "test_homodimer_from_json_features.txt")
        ],
        monomer_directories=[str(TEST_DATA / "features" / "af3_features" / "protein")],
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job["name"] == "P61626_and_P61626"
    assert len(job["sequences"]) == 1
    assert job["sequences"][0]["proteinChain"]["count"] == 2


def test_write_jobs_to_json_files_splits_large_batches(tmp_path):
    job = build_alphafold_server_job(
        [{"TEST": "all"}],
        [str(TEST_DATA / "features")],
    )
    jobs = [job, job, job]

    written_paths = write_jobs_to_json_files(
        jobs,
        tmp_path / "server_jobs.json",
        jobs_per_file=2,
    )

    assert [path.name for path in written_paths] == [
        "server_jobs_001.json",
        "server_jobs_002.json",
    ]
    first_payload = json.loads(written_paths[0].read_text(encoding="utf-8"))
    second_payload = json.loads(written_paths[1].read_text(encoding="utf-8"))
    assert len(first_payload) == 2
    assert len(second_payload) == 1


def test_job_index_is_one_based():
    with pytest.raises(IndexError):
        build_alphafold_server_jobs(
            protein_lists=[str(TEST_DATA / "protein_lists" / "test_dimer.txt")],
            monomer_directories=[str(TEST_DATA / "features")],
            job_index=2,
        )
