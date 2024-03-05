""" Implements structure prediction strategy class.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from abc import ABC, abstractmethod

class FoldingBackend(ABC):
    """
    A strategy class for structure prediction using various folding backends.
    """
    @abstractmethod
    def predict(self, **kwargs) -> None:
        """
        Abstract method for predicting protein structures.

        This method should be implemented by subclasses to perform protein structure
        prediction given a set of input features and parameters specific to the
        implementation. Implementations may vary in terms of accepted parameters and
        the method of prediction.

        Parameters
        ----------
        **kwargs : dict
            A flexible set of keyword arguments that can include input features,
            model configuration, and other prediction-related parameters.
        """


    @abstractmethod
    def postprocess(self, **kwargs):
        """
        Abstract method for post-processing predicted protein structures.

        This method should be implemented by subclasses to perform any necessary
        post-processing on the predicted protein structures, such as generating plots,
        modifying the structure data, or cleaning up temporary files. The specifics
        of the post-processing steps can vary between implementations.

        Parameters
        ----------
        **kwargs : dict
            A flexible set of keyword arguments that can include paths to prediction results, \
            options for file handling, and other post-processing related parameters.
        """

