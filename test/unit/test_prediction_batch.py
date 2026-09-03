import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest


def _prediction_flags(tmp_path, backend):
    return SimpleNamespace(
        fold_backend=backend,
        features_directory=[str(tmp_path)],
        protein_delimiter="+",
        data_directory="/models",
        random_seed=7,
        num_cycle=3,
        num_predictions_per_model=1,
        crosslinks=None,
        desired_num_res=None,
        desired_num_msa=None,
        skip_templates=False,
        allow_resume=True,
        num_diffusion_samples=5,
        num_recycles=10,
        save_embeddings=False,
        save_distogram=False,
        flash_attention_implementation="triton",
        buckets=["64", "128"],
        jax_compilation_cache_dir=None,
        num_seeds=None,
        debug_templates=False,
        debug_msas=False,
        dropout=False,
        compress_result_pickles=False,
        remove_result_pickles=False,
        remove_keys_from_pickles=True,
        storage_mode="vanilla",
        use_gpu_relax=True,
        models_to_relax="none",
        relax_best_score_threshold=None,
        convert_to_modelcif=False,
        use_ap_style=False,
        pair_msa=True,
        multimeric_template=False,
        description_file=None,
        path_to_mmt=None,
        threshold_clashes=1000,
        hb_allowance=0.4,
        plddt_threshold=0,
        save_features_for_multimeric_object=False,
        msa_depth_scan=False,
        model_names=None,
        msa_depth=None,
    )


def test_manifest_loads_jobs_in_order_and_resolves_output_paths(tmp_path):
    from alphapulldown.prediction_batch import PredictionBatch

    manifest = tmp_path / "batches" / "small.jsonl"
    manifest.parent.mkdir()
    records = [
        {
            "job_id": "A_and_B",
            "input": "A+B",
            "output_directory": "../predictions/A_and_B",
        },
        {
            "job_id": "C_and_D",
            "input": "C+D",
            "output_directory": "../predictions/C_and_D",
        },
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    batch = PredictionBatch.from_jsonl(manifest)

    assert [job.job_id for job in batch.jobs] == ["A_and_B", "C_and_D"]
    assert [job.input for job in batch.jobs] == ["A+B", "C+D"]
    assert [job.output_directory for job in batch.jobs] == [
        (manifest.parent / "../predictions/A_and_B").resolve(),
        (manifest.parent / "../predictions/C_and_D").resolve(),
    ]


def test_manifest_resolves_relative_file_input_from_its_parent(tmp_path):
    from alphapulldown.prediction_batch import PredictionBatch

    manifest = tmp_path / "batch" / "jobs.jsonl"
    input_path = manifest.parent / "inputs" / "fold.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("{}", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "job_id": "json-fold",
                "input": "inputs/fold.json",
                "output_directory": "outputs/json-fold",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    batch = PredictionBatch.from_jsonl(manifest)

    assert batch.jobs[0].input == str(input_path.resolve())


def test_manifest_relative_json_is_available_to_the_parser_adapter(tmp_path):
    from alphapulldown.prediction_batch import (
        AlphaPulldownPredictionAdapter,
        PredictionBatch,
    )

    manifest = tmp_path / "batch" / "jobs.jsonl"
    manifest.parent.mkdir()
    (manifest.parent / "fold.json").write_text(
        json.dumps({"name": "fold", "sequences": [], "modelSeeds": [1]}),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "job_id": "fold",
                "input": "fold.json",
                "output_directory": "output",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    unrelated_features = tmp_path / "features"
    unrelated_features.mkdir()
    flags = _prediction_flags(unrelated_features, "alphafold3")

    class Backend:
        def change_backend(self, backend_name):
            return None

        def setup(self, **kwargs):
            return {"model_runner": object()}

        def predict(self, **kwargs):
            entry = kwargs["objects_to_model"][0]
            yield {
                "object": entry["object"],
                "output_dir": entry["output_dir"],
                "prediction_results": {},
            }

        def postprocess(self, **kwargs):
            return None

    summary = PredictionBatch.from_jsonl(manifest).run(
        AlphaPulldownPredictionAdapter(flags, backend=Backend())
    )

    assert summary.exit_code == 0


@pytest.mark.parametrize("duplicate", ["job_id", "output_directory"])
def test_invalid_batch_is_rejected_before_backend_setup(tmp_path, duplicate):
    from alphapulldown.prediction_batch import (
        PredictionBatch,
        PredictionBatchError,
        PredictionJob,
    )

    first = PredictionJob("job-a", "A+B", (tmp_path / "a").resolve())
    second = PredictionJob(
        "job-a" if duplicate == "job_id" else "job-b",
        "C+D",
        (tmp_path / ("a" if duplicate == "output_directory" else "b")).resolve(),
    )

    class RecordingAdapter:
        def __init__(self):
            self.setup_calls = 0

        def configuration_for(self, job):
            return "alphafold3"

        def setup(self, configuration):
            self.setup_calls += 1
            return object()

        def predict(self, session, job):
            return None

    adapter = RecordingAdapter()

    with pytest.raises(PredictionBatchError, match="Duplicate"):
        PredictionBatch((first, second)).run(adapter)

    assert adapter.setup_calls == 0


def test_batch_sets_up_once_and_reports_failures_after_running_remaining_jobs(tmp_path):
    from alphapulldown.prediction_batch import PredictionBatch, PredictionJob

    jobs = tuple(
        PredictionJob(job_id, fold, (tmp_path / job_id).resolve())
        for job_id, fold in (
            ("first", "A+B"),
            ("broken", "C+D"),
            ("last", "E+F"),
        )
    )

    class RecordingAdapter:
        def __init__(self):
            self.setup_calls = []
            self.predicted = []

        def configuration_for(self, job):
            return ("alphafold3", "shared-model-config")

        def setup(self, configuration):
            self.setup_calls.append(configuration)
            return "resident-session"

        def predict(self, session, job):
            assert session == "resident-session"
            self.predicted.append(job.job_id)
            if job.job_id == "broken":
                raise RuntimeError("bad fold")

    adapter = RecordingAdapter()

    summary = PredictionBatch(jobs).run(adapter)

    assert adapter.setup_calls == [("alphafold3", "shared-model-config")]
    assert adapter.predicted == ["first", "broken", "last"]
    assert summary.completed_job_ids == ("first", "last")
    assert [(failure.job_id, failure.message) for failure in summary.failures] == [
        ("broken", "bad fold")
    ]
    assert summary.exit_code == 1


def test_batch_continues_after_a_recoverable_job_preparation_failure(tmp_path):
    from alphapulldown.prediction_batch import PredictionBatch, PredictionJob

    jobs = (
        PredictionJob("invalid", "missing.json", tmp_path / "invalid"),
        PredictionJob("valid", "A+B", tmp_path / "valid"),
    )

    class RecordingAdapter:
        def __init__(self):
            self.setup_calls = []
            self.predicted = []

        def configuration_for(self, job):
            if job.job_id == "invalid":
                raise FileNotFoundError(job.input)
            return "shared-configuration"

        def setup(self, configuration):
            self.setup_calls.append(configuration)
            return "resident-session"

        def predict(self, session, job):
            self.predicted.append(job.job_id)

    adapter = RecordingAdapter()

    summary = PredictionBatch(jobs).run(adapter)

    assert adapter.setup_calls == ["shared-configuration"]
    assert adapter.predicted == ["valid"]
    assert summary.completed_job_ids == ("valid",)
    assert [(failure.job_id, failure.message) for failure in summary.failures] == [
        ("invalid", "missing.json")
    ]
    assert summary.exit_code == 1


def test_heterogeneous_batch_is_rejected_before_backend_setup(tmp_path):
    from alphapulldown.prediction_batch import (
        PredictionBatch,
        PredictionBatchError,
        PredictionJob,
    )

    jobs = (
        PredictionJob("monomer", "A", tmp_path / "monomer"),
        PredictionJob("multimer", "A+B", tmp_path / "multimer"),
    )

    class MixedConfigurationAdapter:
        setup_calls = 0

        def configuration_for(self, job):
            return "multimer" if "+" in job.input else "monomer"

        def setup(self, configuration):
            self.setup_calls += 1

        def predict(self, session, job):
            raise AssertionError("prediction must not start")

    adapter = MixedConfigurationAdapter()

    with pytest.raises(PredictionBatchError, match="heterogeneous"):
        PredictionBatch(jobs).run(adapter)

    assert adapter.setup_calls == 0


def test_manifest_command_returns_a_concise_rejection_for_malformed_json(tmp_path):
    from alphapulldown.prediction_batch import execute_prediction_manifest

    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text("{not-json}\n", encoding="utf-8")

    outcome = execute_prediction_manifest(manifest, adapter=object())

    assert outcome.exit_code == 2
    assert outcome.summary_message == (
        "Prediction batch summary: 0 completed, batch rejected"
    )
    assert outcome.rejection == (
        "Invalid JSON on manifest line 1: "
        "Expecting property name enclosed in double quotes"
    )


def test_manifest_command_returns_a_concise_rejection_when_file_is_missing(tmp_path):
    from alphapulldown.prediction_batch import execute_prediction_manifest

    missing_manifest = tmp_path / "missing.jsonl"

    outcome = execute_prediction_manifest(missing_manifest, adapter=object())

    assert outcome.exit_code == 2
    assert outcome.rejection.startswith("Cannot read prediction batch manifest ")
    assert str(missing_manifest) in outcome.rejection


def test_manifest_command_returns_a_concise_rejection_for_mixed_configurations(
    tmp_path,
):
    from alphapulldown.prediction_batch import execute_prediction_manifest

    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "job_id": job_id,
                    "input": input_value,
                    "output_directory": output_directory,
                }
            )
            for job_id, input_value, output_directory in (
                ("monomer", "A", "monomer-output"),
                ("multimer", "A+B", "multimer-output"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    class MixedConfigurationAdapter:
        def configuration_for(self, job):
            return "multimer" if "+" in job.input else "monomer"

        def setup(self, configuration):
            raise AssertionError("backend setup must not run")

        def predict(self, session, job):
            raise AssertionError("prediction must not run")

    outcome = execute_prediction_manifest(
        manifest, adapter=MixedConfigurationAdapter()
    )

    assert outcome.exit_code == 2
    assert outcome.summary_message == (
        "Prediction batch summary: 0 completed, batch rejected"
    )
    assert outcome.rejection == (
        "Prediction batch contains heterogeneous backend/model configurations"
    )


def test_manifest_command_does_not_misreport_backend_setup_errors_as_read_errors(
    tmp_path,
):
    from alphapulldown.prediction_batch import execute_prediction_manifest

    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "job_id": "fold",
                "input": "A",
                "output_directory": "output",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class MissingModelAdapter:
        def configuration_for(self, job):
            return "shared-configuration"

        def setup(self, configuration):
            raise FileNotFoundError("/models/params")

        def predict(self, session, job):
            raise AssertionError("prediction must not run")

    with pytest.raises(FileNotFoundError, match="/models/params"):
        execute_prediction_manifest(manifest, adapter=MissingModelAdapter())


@pytest.mark.parametrize(
    "outcome_kwargs",
    [
        {},
        {
            "summary": SimpleNamespace(exit_code=0),
            "rejection": "invalid manifest",
        },
    ],
)
def test_batch_outcome_requires_exactly_one_result(outcome_kwargs):
    from alphapulldown.prediction_batch import PredictionBatchOutcome

    with pytest.raises(ValueError, match="exactly one"):
        PredictionBatchOutcome(**outcome_kwargs)


def test_alphapulldown_adapter_keeps_af3_jobs_independent_with_one_setup(tmp_path):
    from alphapulldown.prediction_batch import (
        AlphaPulldownPredictionAdapter,
        PredictionBatch,
        PredictionJob,
    )

    for name in ("first.json", "second.json", "third.json"):
        (tmp_path / name).write_text(
            json.dumps({"name": name, "sequences": [], "modelSeeds": [1]}),
            encoding="utf-8",
        )

    class RecordingBackend:
        def __init__(self):
            self.setup_calls = []
            self.prediction_jobs = []

        def change_backend(self, backend_name):
            assert backend_name == "alphafold3"

        def setup(self, **model_flags):
            self.setup_calls.append(model_flags)
            return {"model_runner": "resident-runner"}

        def predict(self, **kwargs):
            objects = kwargs["objects_to_model"]
            self.prediction_jobs.append(objects)
            entry = objects[0]
            yield {
                "object": entry["object"],
                "output_dir": entry["output_dir"],
                "prediction_results": {"ok": True},
            }

        def postprocess(self, **kwargs):
            return None

    flags = _prediction_flags(tmp_path, "alphafold3")
    jobs = (
        PredictionJob("first", "first.json+second.json", tmp_path / "out-first"),
        PredictionJob("second", "third.json", tmp_path / "out-second"),
    )
    backend = RecordingBackend()

    summary = PredictionBatch(jobs).run(
        AlphaPulldownPredictionAdapter(flags, backend=backend)
    )

    assert summary.exit_code == 0
    assert len(backend.setup_calls) == 1
    assert [len(job) for job in backend.prediction_jobs] == [2, 1]
    assert [job[0]["output_dir"] for job in backend.prediction_jobs] == [
        str(tmp_path / "out-first"),
        str(tmp_path / "out-second"),
    ]


def test_alphapulldown_adapter_reuses_af2_runners_across_jobs(tmp_path):
    from alphapulldown.objects import MonomericObject
    from alphapulldown.prediction_batch import (
        AlphaPulldownPredictionAdapter,
        PredictionBatch,
        PredictionJob,
    )

    for name in ("A", "B"):
        with (tmp_path / f"{name}.pkl").open("wb") as handle:
            pickle.dump(MonomericObject(name, "ACDE"), handle)

    class RecordingBackend:
        def __init__(self):
            self.setup_calls = []
            self.predicted = []

        def change_backend(self, backend_name):
            assert backend_name == "alphafold2"

        def setup(self, **model_flags):
            self.setup_calls.append(model_flags)
            return {"model_runners": {"model": "resident-runner"}}

        def predict(self, **kwargs):
            entry = kwargs["objects_to_model"][0]
            self.predicted.append(entry["object"].description)
            yield {
                "object": entry["object"],
                "output_dir": entry["output_dir"],
                "prediction_results": {"model": {}},
            }

        def postprocess(self, **kwargs):
            return None

    backend = RecordingBackend()
    batch = PredictionBatch(
        (
            PredictionJob("A", "A", tmp_path / "out-A"),
            PredictionJob("B", "B", tmp_path / "out-B"),
        )
    )

    summary = batch.run(
        AlphaPulldownPredictionAdapter(
            _prediction_flags(tmp_path, "alphafold2"), backend=backend
        )
    )

    assert summary.exit_code == 0
    assert len(backend.setup_calls) == 1
    assert backend.setup_calls[0]["model_name"] == "monomer_ptm"
    assert backend.predicted == ["A", "B"]


@pytest.mark.parametrize(
    ("backend_name", "fold_specification", "expected_model_name"),
    [
        ("alphafold2", "A", "monomer_ptm"),
        ("alphafold2", "A+B", "multimer"),
        ("alphafold2", "A:2", "multimer"),
        ("alphafold2", "A:1-3", "monomer_ptm"),
        ("alphafold3", "fold.json", "monomer_ptm"),
    ],
)
def test_adapter_configuration_matches_the_shared_parser_contract(
    tmp_path, backend_name, fold_specification, expected_model_name
):
    from alphapulldown.prediction_batch import (
        AlphaPulldownPredictionAdapter,
        PredictionJob,
    )

    for name in ("A", "B"):
        (tmp_path / f"{name}.pkl").touch()
    (tmp_path / "fold.json").write_text(
        json.dumps({"name": "fold", "sequences": [], "modelSeeds": [1]}),
        encoding="utf-8",
    )
    adapter = AlphaPulldownPredictionAdapter(
        _prediction_flags(tmp_path, backend_name), backend=object()
    )
    job = PredictionJob(
        "contract-case", fold_specification, tmp_path / "prediction"
    )

    configuration = adapter.configuration_for(job)

    assert configuration["model_name"] == expected_model_name


def test_batch_command_is_in_the_installed_script_interface():
    repository = Path(__file__).resolve().parents[2]
    command = repository / "alphapulldown/scripts/run_structure_prediction_batch.py"

    assert command.is_file()
    assert "./alphapulldown/scripts/run_structure_prediction_batch.py" in (
        repository / "pyproject.toml"
    ).read_text(encoding="utf-8")
