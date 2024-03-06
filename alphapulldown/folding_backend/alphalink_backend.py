""" Implements structure prediction backend using AlphaLink2.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from typing import Dict
from os.path import join, exists
from alphapulldown.objects import MultimericObject

from .folding_backend import FoldingBackend


class AlphaLinkBackend(FoldingBackend):
    """
    A backend class for running protein structure predictions using the AlphaLink model.
    """

    def setup(
        alphalink_weight: str,
        crosslinks_path: str,
        model_name: str = "multimer_af2_crop",
        **kwargs,
    ) -> Dict:
        """
        Initializes and configures an AlphaLink model runner including crosslinking data.

        Parameters
        ----------
        model_name : str
            The name of the model to use for prediction. Set to be multimer_af2_crop as used in AlphaLink2
        alphalink_weight : str
            Path to the pytorch checkpoint that corresponds to the neural network weights from AlphaLink2.
        crosslinks_path : str
            The path to the file containing crosslinking data.
        **kwargs : dict
            Additional keyword arguments for model configuration.

        Returns
        -------
        Dict
            A dictionary records the path to the AlphaLink2 neural network weights
            i.e. a pytorch checkpoint file, crosslink information,
            and Pytorch model configs
        """
        from unifold.config import model_config

        if not exists(alphalink_weight):
            raise FileNotFoundError(
                f"AlphaLink2 network weight does not exist at: {alphalink_weight}"
            )
        if not alphalink_weight.endswith(".pt"):
            f"{alphalink_weight} does not seem to be a pytorch checkpoint."

        configs = model_config(model_name)

        return {
            "crosslinks": crosslinks_path,
            "param_path": alphalink_weight,
            "configs": configs,
        }

    @staticmethod
    def predict(
        configs: Dict,
        param_path: str,
        crosslinks: str,
        multimeric_object: MultimericObject,
        output_dir: str,
        **kwargs,
    ):
        """
        Predicts the structure of proteins using configured AlphaLink models.

        Parameters
        ----------
        model_config : Dict
            Configuration dictionary for the AlphaLink model obtained from
            py:meth:`AlphaLinkBackend.setup`.
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
            chain_id_map=multimeric_object.chain_id_map,
            configs=configs,
            param_path=param_path,
            crosslinks=crosslinks,
        )

    def postprocess(**kwargs) -> None:
        return None
