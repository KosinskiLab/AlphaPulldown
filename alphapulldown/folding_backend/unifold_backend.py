""" Implements structure prediction backend using UniFold.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List

from absl import logging

from .folding_backend import FoldingBackend


# The model configurations UniFold ships; the default is what run_multimer_jobs
# always offered through --unifold_model_name.
UNIFOLD_MODEL_NAMES = (
    "multimer_af2",
    "multimer_ft",
    "multimer",
    "multimer_af2_v3",
    "multimer_af2_model45_v3",
)
DEFAULT_UNIFOLD_MODEL_NAME = "multimer_af2"


class UnifoldBackend(FoldingBackend):
    """
    A backend class for running protein structure predictions using the UniFold model.

    It follows the same contract as the other backends, which is what the
    prediction adapters drive: ``setup(**model_flags)`` builds a session,
    ``predict(**session, objects_to_model=..., random_seed=..., **model_flags)``
    yields one record per fold, and ``postprocess`` receives the record. The
    previous version required ``output_dir`` and ``multimeric_object`` in
    ``setup`` and took a single object in an instance-method ``predict``, so no
    caller in the package could invoke it.
    """

    @staticmethod
    def setup(
        model_dir: str,
        model_name: str = DEFAULT_UNIFOLD_MODEL_NAME,
        **kwargs,
    ) -> Dict:
        """
        Resolve the UniFold model configuration for this session.

        Parameters
        ----------
        model_dir : str
            The directory where the UniFold parameters are located.
        model_name : str
            The UniFold model configuration (``--unifold_model_name``).
        **kwargs : dict
            Additional keyword arguments for model configuration. Ignored.

        Returns
        -------
        Dict
            The model configuration under ``model_config``. The weights are
            loaded per fold in :py:meth:`UnifoldBackend.predict`, because
            UniFold ties its runner to the target name and output directory.
        """
        from unifold.config import model_config

        if model_name not in UNIFOLD_MODEL_NAMES:
            raise ValueError(
                f"Unknown UniFold model {model_name!r}; choose one of "
                f"{', '.join(UNIFOLD_MODEL_NAMES)}"
            )
        return {"model_config": model_config(model_name)}

    @staticmethod
    def predict(
        objects_to_model: List[Dict[str, Any]],
        model_config: Dict,
        model_dir: str,
        random_seed: int = 42,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        Predicts the structure of each object with the configured UniFold model.

        Parameters
        ----------
        objects_to_model : List[Dict[str, Any]]
            One ``{"object": ..., "output_dir": ...}`` record per fold; the object
            carries ``description`` and ``feature_dict``.
        model_config : Dict
            Configuration obtained from :py:meth:`UnifoldBackend.setup`.
        model_dir : str
            The directory where the UniFold parameters are located.
        random_seed : int, optional
            The random seed for prediction reproducibility, default is 42.
        **kwargs : dict
            Additional keyword arguments for prediction. Ignored.

        Yields
        ------
        Dict
            ``object``, ``prediction_results`` and ``output_dir`` for each fold,
            in order. UniFold writes its structures itself, so the results are
            an empty mapping.
        """
        from unifold.dataset import process_ap
        from unifold.inference import config_args, unifold_config_model, unifold_predict

        for entry in objects_to_model:
            object_to_model = entry["object"]
            output_dir = entry["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            logging.info(
                "Now running UniFold prediction on %s", object_to_model.description
            )
            general_args = config_args(
                model_dir,
                target_name=object_to_model.description,
                output_dir=output_dir,
            )
            model_runner = unifold_config_model(general_args)
            processed_features, _ = process_ap(
                config=model_config,
                features=object_to_model.feature_dict,
                mode="predict",
                labels=None,
                seed=random_seed,
                batch_idx=None,
                data_idx=None,
                is_distillation=False,
            )
            unifold_predict(model_runner, general_args, processed_features)
            yield {
                "object": object_to_model,
                "prediction_results": {},
                "output_dir": output_dir,
            }

    @staticmethod
    def postprocess(**kwargs) -> None:
        """UniFold writes its own outputs; nothing to post-process."""
        return None
