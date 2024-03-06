""" Implements class to represent electron density maps.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Dict, List

from .alphafold_backend import AlphaFold


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
            "alphafold": AlphaFold,
        }
        self._backend = AlphaFold()
        self._backend_name = "alphafold"
        self._backend_args = {}

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
