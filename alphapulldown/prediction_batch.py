"""Resident execution of independent structure-prediction jobs."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import lzma
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
from typing import Any, Dict, Protocol, Tuple, Union


PathLike = Union[str, Path]


class PredictionBatchError(ValueError):
    """A batch manifest does not satisfy the public contract."""


@dataclass(frozen=True)
class PredictionJob:
    """One independent fold in a prediction batch."""

    job_id: str
    input: str
    output_directory: Path


@dataclass(frozen=True)
class PredictionFailure:
    """One recoverable job failure."""

    job_id: str
    message: str
    exception: Exception


@dataclass(frozen=True)
class PredictionBatchSummary:
    """Observable outcome of running a prediction batch."""

    completed_job_ids: Tuple[str, ...]
    failures: Tuple[PredictionFailure, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0


class PredictionAdapter(Protocol):
    """Backend-specific work hidden behind the prediction-batch seam."""

    def configuration_for(self, job: PredictionJob) -> Any:
        ...

    def setup(self, configuration: Any) -> Any:
        ...

    def predict(self, session: Any, job: PredictionJob) -> None:
        ...


@dataclass(frozen=True)
class PredictionBatch:
    """An ordered collection of independent prediction jobs."""

    jobs: Tuple[PredictionJob, ...]

    def _validate(self) -> None:
        seen_job_ids = set()
        seen_output_directories = set()
        for job in self.jobs:
            if job.job_id in seen_job_ids:
                raise PredictionBatchError(f"Duplicate job_id: {job.job_id}")
            seen_job_ids.add(job.job_id)

            output_directory = job.output_directory.resolve()
            if output_directory in seen_output_directories:
                raise PredictionBatchError(
                    f"Duplicate output_directory: {output_directory}"
                )
            seen_output_directories.add(output_directory)

    def run(self, adapter: PredictionAdapter) -> PredictionBatchSummary:
        """Run the batch through one prediction adapter session."""
        self._validate()
        if not self.jobs:
            raise PredictionBatchError("A prediction batch must contain at least one job")

        runnable_jobs = []
        indexed_failures = []
        for position, job in enumerate(self.jobs):
            try:
                configuration = adapter.configuration_for(job)
            except Exception as exc:
                indexed_failures.append(
                    (position, PredictionFailure(job.job_id, str(exc), exc))
                )
            else:
                runnable_jobs.append((position, job, configuration))

        if not runnable_jobs:
            return PredictionBatchSummary(
                (), tuple(failure for _, failure in indexed_failures)
            )

        configuration = runnable_jobs[0][2]
        if any(candidate != configuration for _, _, candidate in runnable_jobs[1:]):
            raise PredictionBatchError(
                "Prediction batch contains heterogeneous backend/model configurations"
            )

        session = adapter.setup(configuration)
        completed = []
        for position, job, _ in runnable_jobs:
            try:
                adapter.predict(session, job)
            except Exception as exc:
                indexed_failures.append(
                    (position, PredictionFailure(job.job_id, str(exc), exc))
                )
            else:
                completed.append(job.job_id)
        indexed_failures.sort(key=lambda item: item[0])
        return PredictionBatchSummary(
            tuple(completed), tuple(failure for _, failure in indexed_failures)
        )

    @classmethod
    def from_jsonl(cls, manifest_path: PathLike) -> "PredictionBatch":
        manifest = Path(manifest_path).expanduser().resolve()
        jobs = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise PredictionBatchError(
                        f"Invalid JSON on manifest line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise PredictionBatchError(
                        f"Manifest line {line_number} must be a JSON object"
                    )
                expected_fields = {"job_id", "input", "output_directory"}
                if set(record) != expected_fields:
                    raise PredictionBatchError(
                        f"Manifest line {line_number} must contain exactly "
                        f"{sorted(expected_fields)}"
                    )
                for field in expected_fields:
                    value = record[field]
                    if not isinstance(value, str) or not value.strip():
                        raise PredictionBatchError(
                            f"Manifest line {line_number} field {field!r} must be a "
                            "non-empty string"
                        )
                output_directory = Path(record["output_directory"]).expanduser()
                if not output_directory.is_absolute():
                    output_directory = manifest.parent / output_directory
                input_value = record["input"]
                input_path = Path(input_value).expanduser()
                if input_path.is_absolute():
                    input_value = os.fspath(input_path.resolve())
                else:
                    manifest_relative_input = manifest.parent / input_path
                    if manifest_relative_input.exists():
                        input_value = os.fspath(manifest_relative_input.resolve())
                jobs.append(
                    PredictionJob(
                        job_id=record["job_id"],
                        input=input_value,
                        output_directory=output_directory.resolve(),
                    )
                )
        return cls(tuple(jobs))


@dataclass(frozen=True)
class _AlphaPulldownSession:
    configuration: Dict[str, Any]
    backend_values: Dict[str, Any]


class PreparedPredictionAdapter:
    """Adapter for the already-parsed invocation used by the legacy command."""

    def __init__(
        self,
        *,
        backend: Any,
        fold_backend: str,
        objects_to_model: list,
        model_flags: Dict[str, Any],
        postprocess_flags: Dict[str, Any],
        random_seed: Any,
    ) -> None:
        self._backend = backend
        self._fold_backend = fold_backend
        self._objects_to_model = objects_to_model
        self._model_flags = model_flags
        self._postprocess_flags = postprocess_flags
        self._requested_random_seed = random_seed

    def configuration_for(self, job: PredictionJob) -> Dict[str, Any]:
        del job
        return self._model_flags

    def setup(self, configuration: Dict[str, Any]) -> _AlphaPulldownSession:
        self._backend.change_backend(backend_name=self._fold_backend)
        return _AlphaPulldownSession(
            dict(configuration), self._backend.setup(**configuration)
        )

    def _random_seed(self, session: _AlphaPulldownSession) -> int:
        if self._requested_random_seed is not None:
            return self._requested_random_seed
        if self._fold_backend == "alphafold2":
            count = len(session.backend_values["model_runners"])
            return random.randrange(sys.maxsize // count)
        if self._fold_backend == "alphafold3":
            return random.randrange(2**32 - 1)
        return random.randrange(sys.maxsize)

    def predict(self, session: _AlphaPulldownSession, job: PredictionJob) -> None:
        del job
        predicted_jobs = self._backend.predict(
            **session.backend_values,
            objects_to_model=self._objects_to_model,
            random_seed=self._random_seed(session),
            **session.configuration,
        )
        for predicted_job in predicted_jobs:
            self._backend.postprocess(
                **self._postprocess_flags,
                multimeric_object=predicted_job["object"],
                prediction_results=predicted_job["prediction_results"],
                output_dir=predicted_job["output_dir"],
            )


class AlphaPulldownPredictionAdapter:
    """Adapter from prediction jobs to AlphaPulldown's folding backends."""

    def __init__(self, prediction_flags: Any, *, backend: Any) -> None:
        self._flags = prediction_flags
        self._backend = backend
        self._parsed_jobs: Dict[str, Tuple[list, Tuple[str, ...]]] = {}

    def _base_model_flags(self) -> Dict[str, Any]:
        flags = self._flags
        return {
            "model_name": "monomer_ptm",
            "num_cycle": flags.num_cycle,
            "model_dir": flags.data_directory,
            "num_predictions_per_model": flags.num_predictions_per_model,
            "crosslinks": flags.crosslinks,
            "desired_num_res": flags.desired_num_res,
            "desired_num_msa": flags.desired_num_msa,
            "skip_templates": flags.skip_templates,
            "allow_resume": flags.allow_resume,
            "num_diffusion_samples": flags.num_diffusion_samples,
            "num_recycles": flags.num_recycles,
            "return_embeddings": flags.save_embeddings,
            "return_distogram": flags.save_distogram,
            "flash_attention_implementation": flags.flash_attention_implementation,
            "buckets": flags.buckets,
            "jax_compilation_cache_dir": flags.jax_compilation_cache_dir,
            "features_directory": flags.features_directory,
            "num_seeds": flags.num_seeds,
            "debug_templates": flags.debug_templates,
            "debug_msas": flags.debug_msas,
            "dropout": flags.dropout,
        }

    def _postprocess_flags(self) -> Dict[str, Any]:
        flags = self._flags
        return {
            "compress_pickles": flags.compress_result_pickles,
            "remove_pickles": flags.remove_result_pickles,
            "remove_keys_from_pickles": flags.remove_keys_from_pickles,
            "storage_mode": flags.storage_mode,
            "use_gpu_relax": flags.use_gpu_relax,
            "models_to_relax": flags.models_to_relax,
            "relax_best_score_threshold": flags.relax_best_score_threshold,
            "features_directory": flags.features_directory,
            "convert_to_modelcif": flags.convert_to_modelcif,
        }

    def configuration_for(self, job: PredictionJob) -> Dict[str, Any]:
        from alphapulldown.utils.modelling_setup import parse_fold

        feature_directories = [str(path) for path in self._flags.features_directory]
        input_path = Path(job.input)
        if input_path.is_absolute() and input_path.is_file():
            input_parent = os.fspath(input_path.parent)
            feature_directories = [
                input_parent,
                *(path for path in feature_directories if path != input_parent),
            ]
        parsed = parse_fold(
            [job.input],
            feature_directories,
            self._flags.protein_delimiter,
        )
        self._parsed_jobs[job.job_id] = (parsed, tuple(feature_directories))

        model_flags = self._base_model_flags()
        if self._flags.fold_backend == "alphalink":
            model_flags["model_name"] = "multimer_af2_crop"
        elif self._flags.fold_backend == "alphafold2":
            protein_counts = [
                sum(1 for entry in fold if "json_input" not in entry)
                for fold in parsed
            ]
            if any(count > 1 for count in protein_counts):
                model_flags.update(
                    {
                        "model_name": "multimer",
                        "msa_depth_scan": self._flags.msa_depth_scan,
                        "model_names_custom": self._flags.model_names,
                        "msa_depth": self._flags.msa_depth,
                    }
                )
        return model_flags

    def setup(self, configuration: Dict[str, Any]) -> _AlphaPulldownSession:
        self._backend.change_backend(backend_name=self._flags.fold_backend)
        backend_values = self._backend.setup(**configuration)
        return _AlphaPulldownSession(dict(configuration), backend_values)

    def _random_seed(self, session: _AlphaPulldownSession) -> int:
        if self._flags.random_seed is not None:
            return self._flags.random_seed
        if self._flags.fold_backend == "alphafold2":
            model_count = len(session.backend_values["model_runners"])
            return random.randrange(sys.maxsize // model_count)
        if self._flags.fold_backend == "alphafold3":
            return random.randrange(2**32 - 1)
        return random.randrange(sys.maxsize)

    def _prepare_protein_object(self, interactors: list, output_dir: str):
        from alphapulldown.objects import ChoppedObject, MonomericObject, MultimericObject

        flags = self._flags
        if (
            len(interactors) > 1
            and flags.pair_msa
            and any(getattr(interactor, "skip_msa", False) for interactor in interactors)
        ):
            raise ValueError(
                "--skip_msa generates query-only MSAs and cannot be combined with "
                "--pair_msa=True. Re-run structure prediction with --pair_msa=False."
            )

        if len(interactors) > 1:
            object_to_model = MultimericObject(
                interactors=interactors,
                pair_msa=flags.pair_msa,
                multimeric_template=flags.multimeric_template,
                multimeric_template_meta_data=flags.description_file,
                multimeric_template_dir=flags.path_to_mmt,
                threshold_clashes=flags.threshold_clashes,
                hb_allowance=flags.hb_allowance,
                plddt_threshold=flags.plddt_threshold,
            )
            if flags.save_features_for_multimeric_object:
                with open(
                    os.path.join(output_dir, "multimeric_object_features.pkl"), "wb"
                ) as handle:
                    pickle.dump(MultimericObject.feature_dict, handle)
        else:
            object_to_model = interactors[0]
            object_to_model.input_seqs = [object_to_model.sequence]

        if flags.use_ap_style:
            oligos = object_to_model.description.split("_and_")
            if len(oligos) == len(set(oligos)):
                output_dir = os.path.join(output_dir, object_to_model.description)
            else:
                fragments = []
                for oligo in dict.fromkeys(oligos):
                    count = oligos.count(oligo)
                    fragments.append(oligo if count == 1 else f"{oligo}_homo_{count}er")
                output_dir = os.path.join(output_dir, "_and_".join(fragments))

        if len(output_dir) > 4096:
            from absl import logging

            logging.warning(
                f"Output directory path is too long: {output_dir}."
                "Please use a shorter path with --output_directory."
            )
        os.makedirs(output_dir, exist_ok=True)

        for interactor in interactors:
            metadata_files = []
            if isinstance(interactor, ChoppedObject):
                description = interactor.monomeric_description
            elif isinstance(interactor, MonomericObject):
                description = interactor.description
            else:
                continue
            for feature_dir in flags.features_directory:
                metadata_files.extend(
                    glob.glob(
                        os.path.join(
                            feature_dir, f"{description}_feature_metadata_*.json*"
                        )
                    )
                )
            if not metadata_files:
                continue
            latest = max(metadata_files, key=os.path.getmtime)
            destination = os.path.join(output_dir, os.path.basename(latest))
            if latest.endswith(".json.xz"):
                with lzma.open(latest, "rb") as source, open(
                    destination[:-3], "wb"
                ) as target:
                    target.write(source.read())
            else:
                shutil.copyfile(latest, destination)
        return object_to_model, output_dir

    def _objects_to_model(self, job: PredictionJob) -> list:
        from alphapulldown.utils.modelling_setup import (
            create_custom_info,
            create_interactors,
        )
        from alphapulldown.utils.output_paths import (
            resolve_af3_combined_json_output_dir,
            resolve_af3_json_output_dir,
        )

        parsed, feature_directories = self._parsed_jobs.pop(job.job_id)
        all_interactors = create_interactors(
            create_custom_info(parsed), feature_directories
        )
        objects_to_model = []
        for interactors in all_interactors:
            if not interactors:
                continue
            json_dicts = [
                value
                for value in interactors
                if isinstance(value, dict) and "json_input" in value
            ]
            protein_objects = [
                value
                for value in interactors
                if not (isinstance(value, dict) and "json_input" in value)
            ]
            output_dir = os.fspath(job.output_directory)
            if protein_objects:
                protein_object, actual_output_dir = self._prepare_protein_object(
                    protein_objects, output_dir
                )
                objects_to_model.append(
                    {"object": protein_object, "output_dir": actual_output_dir}
                )
                json_output_dir = actual_output_dir
            elif len(json_dicts) > 1:
                json_output_dir = resolve_af3_combined_json_output_dir(
                    json_dicts,
                    output_dir,
                    use_ap_style=self._flags.use_ap_style,
                )
            else:
                json_output_dir = None

            for json_dict in json_dicts:
                current_output_dir = json_output_dir
                if current_output_dir is None:
                    current_output_dir = resolve_af3_json_output_dir(
                        json_dict["json_input"],
                        output_dir,
                        use_ap_style=self._flags.use_ap_style,
                        shared_output_root=False,
                    )
                objects_to_model.append(
                    {"object": json_dict, "output_dir": current_output_dir}
                )

        return objects_to_model

    def predict(self, session: _AlphaPulldownSession, job: PredictionJob) -> None:
        objects_to_model = self._objects_to_model(job)
        predicted_jobs = self._backend.predict(
            **session.backend_values,
            objects_to_model=objects_to_model,
            random_seed=self._random_seed(session),
            **session.configuration,
        )
        postprocess_flags = self._postprocess_flags()
        for predicted_job in predicted_jobs:
            self._backend.postprocess(
                **postprocess_flags,
                multimeric_object=predicted_job["object"],
                prediction_results=predicted_job["prediction_results"],
                output_dir=predicted_job["output_dir"],
            )
