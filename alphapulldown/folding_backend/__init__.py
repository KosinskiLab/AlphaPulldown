""" Implements class to represent electron density maps.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
            Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""

from typing import Dict, List
from absl import logging
from .alphafold_backend import AlphaFoldBackend
logging.set_verbosity(logging.INFO)

class FoldingBackendManager:
    """
    Manager for structure prediction backends.

    Attributes
    ----------
    _BACKEND_REGISTRY : dict
        A dictionary mapping backend names to their respective classes or instances.
    _backend : instance of MatchingBackend
        An instance of the currently active backend. Defaults to AlphaFold.
    _backend_name : str
        Name of the current backend.
    _backend_args : Dict
        Arguments passed to create current backend.

    """

    def __init__(self):
        self._BACKEND_REGISTRY = {
            "alphafold": AlphaFoldBackend
        }
        self.import_backends()
        self._backend_name = "alphafold"
        self._backend = self._BACKEND_REGISTRY[self._backend_name]()
        self._backend_args = {}

    def import_backends(self) -> None:
        """Import all available backends"""
        try:
            from .alphalink_backend import AlphaLinkBackend
            self._BACKEND_REGISTRY.update({"alphalink": AlphaLinkBackend})
        except Exception as e:
            logging.warning(
                f"Failed to import AlphaLinkBackend: {e}. Perhaps you haven't installed all the required dependencies.")

        try:
            from .unifold_backend import UnifoldBackend
            self._BACKEND_REGISTRY.update({"unifold": UnifoldBackend})
        except Exception as e:
            logging.warning(
                f"Failed to import UnifoldBackend: {e}. Perhaps you haven't installed all the required dependencies.")

        try:
            from .alphafold3_backend import AlphaFold3Backend
            self._BACKEND_REGISTRY.update({"alphafold3": AlphaFold3Backend})
        except Exception as e:
            logging.warning(
                f"Failed to import AlphaFold3Backend: {e}. Perhaps you haven't installed all the required dependencies.")

    def __repr__(self):
        return f"<BackendManager: using {self._backend_name}>"

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def __dir__(self) -> List:
        """
        Return a list of attributes available in this object,
        including those from the backend.

        Returns
        -------
        list
            Sorted list of attributes.
        """
        base_attributes = []
        base_attributes.extend(dir(self.__class__))
        base_attributes.extend(self.__dict__.keys())
        base_attributes.extend(dir(self._backend))
        return sorted(base_attributes)

    def change_backend(self, backend_name: str, **backend_kwargs: Dict) -> None:
        """
        Change the backend.

        Parameters
        ----------
        backend_name : str
            Name of the new backend that should be used.
        **backend_kwargs : Dict, optional
            Parameters passed to __init__ method of backend.

        Raises
        ------
        NotImplementedError
            If no backend is found with the provided name.
        """
        if backend_name not in self._BACKEND_REGISTRY:
            available_backends = ", ".join(
                [str(x) for x in self._BACKEND_REGISTRY.keys()]
            )
            raise NotImplementedError(
                f"Available backends are {available_backends} - not {backend_name}."
            )
        self._backend = self._BACKEND_REGISTRY[backend_name](**backend_kwargs)
        self._backend_name = backend_name
        self._backend_args = backend_kwargs


backend = FoldingBackendManager()

def change_backend(backend_name: str) -> None:
    """Change the backend for structure prediction.

    Args:
        backend_name: Name of the backend to use.
    """
    if backend_name not in ["alphafold", "unifold", "alphafold3"]:
        raise NotImplementedError(
            f"Available backends are alphafold, unifold, alphafold3 - not {backend_name}."
        )
    global backend
    backend.change_backend(backend_name)
