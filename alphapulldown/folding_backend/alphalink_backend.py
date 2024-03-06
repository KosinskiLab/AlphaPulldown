""" Implements structure prediction backend using AlphaLink.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from typing import Dict
from os.path import join

from alphapulldown.objects import MultimericObject

from .folding_backend import FoldingBackend


class AlphaLinkBackend(FoldingBackend):
    """
    A backend class for running protein structure predictions using the AlphaLink model.
    """

    def create_model_runner(
        model_name: str,
        model_dir: str,
        output_dir: str,
        crosslinks_path: str,
        multimeric_object: MultimericObject,
        **kwargs,
    ) -> Dict:
        """
        Initializes and configures an AlphaLink model runner including crosslinking data.

        Parameters
        ----------
        model_name : str
            The name of the model to use for prediction.
        model_dir : str
            The directory where the model files are located.
        output_dir : str
            The directory where the prediction outputs will be saved.
        crosslinks_path : str
            The path to the file containing crosslinking data.
        multimeric_object : MultimericObject
            An object containing the description and features of the
            multimeric protein to predict.
        **kwargs : dict
            Additional keyword arguments for model configuration.

        Returns
        -------
        Dict
            A dictionary containing the paths and configuration for the
            AlphaLink model runner.
        """
        from unifold.config import model_config

        configs = model_config(model_name)

        return {
            "crosslinks": crosslinks_path,
            "param_path": model_dir,
            "model_config": configs,
        }

    @staticmethod
    def predict(
        model_params: Dict,
        model_config: Dict,
        multimeric_object: MultimericObject,
        output_dir: str,
        **kwargs,
    ):
        """
        Predicts the structure of proteins using configured AlphaLink models.

        Parameters
        ----------
        model_params : Dict
            Parameters specific to the AlphaLink model, including paths and settings.
        model_config : Dict
            Configuration dictionary for the AlphaLink model obtained from
            py:meth:`AlphaLinkBackend.create_model_runner`.
        multimeric_object : MultimericObject
            An object containing the features of the multimeric protein to predict.
        output_dir : str
            The directory where the prediction outputs will be saved.
        **kwargs : dict
            Additional keyword arguments for prediction.
        """

        from unifold.alphalink_inference import alphalink_prediction

        alphalink_prediction(
            multimeric_object.feature_dict,
            join(output_dir, multimeric_object.description),
            input_seqs=multimeric_object.input_seqs,
            configs=model_config,
            chain_id_map=multimeric_object.chain_id_map,
            **model_params,
        )

    def postprocess(**kwargs) -> None:
        return None
