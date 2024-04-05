""" Implements structure prediction backend using UniFold.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from typing import Dict

from alphapulldown.objects import MultimericObject

from .folding_backend import FoldingBackend


class UnifoldBackend(FoldingBackend):
    """
    A backend class for running protein structure predictions using the UniFold model.
    """
    @staticmethod
    def setup(
        model_name: str,
        model_dir: str,
        output_dir: str,
        multimeric_object: MultimericObject,
        **kwargs,
    ) -> Dict:
        """
        Initializes and configures a UniFold model runner.

        Parameters
        ----------
        model_name : str
            The name of the model to use for prediction.
        model_dir : str
            The directory where the model files are located.
        output_dir : str
            The directory where the prediction outputs will be saved.
        multimeric_object : MultimericObject
            An object containing the description and features of the
            multimeric protein to predict.
        **kwargs : dict
            Additional keyword arguments for model configuration.

        Returns
        -------
        Dict
            A dictionary containing the model runner, arguments, and configuration.
        """
        from unifold.config import model_config
        from unifold.inference import config_args, unifold_config_model

        configs = model_config(model_name)
        general_args = config_args(
            model_dir, target_name=multimeric_object.description, output_dir=output_dir
        )
        model_runner = unifold_config_model(general_args)

        return {
            "model_runner": model_runner,
            "model_args": general_args,
            "model_config": configs,
        }

    def predict(
        self,
        model_runner,
        model_args,
        model_config: Dict,
        multimeric_object: MultimericObject,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Predicts the structure of proteins using configured UniFold models.

        Parameters
        ----------
        model_runner
            The configured model runner for predictions obtained
            from :py:meth:`UnifoldBackend.setup`.
        model_args
            Arguments used for running the UniFold prediction obtained from
            from :py:meth:`UnifoldBackend.setup`.
        model_config : Dict
            Configuration dictionary for the UniFold model obtained from
            from :py:meth:`UnifoldBackend.setup`.
        multimeric_object : MultimericObject
            An object containing the features of the multimeric protein to predict.
        random_seed : int, optional
            The random seed for prediction reproducibility, default is 42.
        **kwargs : dict
            Additional keyword arguments for prediction.
        """
        from unifold.dataset import process_ap
        from unifold.inference import unifold_predict

        processed_features, _ = process_ap(
            config=model_config,
            features=multimeric_object.feature_dict,
            mode="predict",
            labels=None,
            seed=random_seed,
            batch_idx=None,
            data_idx=None,
            is_distillation=False,
        )
        unifold_predict(model_runner, model_args, processed_features)

        return None

    def postprocess(**kwargs) -> None:
        return None
