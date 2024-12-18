"""
    Copyright (c) 2024 European Molecular Biology Laboratory

    Email: alphapulldown@embl-hamburg.de
"""

from typing import Dict, List
from absl import logging
logging.set_verbosity(logging.INFO)

class FoldingBackendManager:
    """
    Manager for structure prediction backends.
    """

    def __init__(self):
        self._BACKEND_REGISTRY = {
            "alphafold": self._lazy_import("alphafold_backend", "AlphaFoldBackend")
        }
        self.import_backends()
        self._backend_name = "alphafold"
        self._backend = self._BACKEND_REGISTRY[self._backend_name]()
        self._backend_args = {}

    def import_backends(self) -> None:
        """Import all available backends"""
        self._try_import("alphalink_backend", "AlphaLinkBackend", "alphalink")
        self._try_import("unifold_backend", "UnifoldBackend", "unifold")
        self._try_import("alphafold3_backend", "AlphaFold3Backend", "alphafold3")

    def _lazy_import(self, module_name: str, class_name: str):
        def _imported_class(*args, **kwargs):
            mod = __import__(f"alphapulldown.folding_backend.{module_name}", fromlist=[class_name])
            cls = getattr(mod, class_name)
            return cls(*args, **kwargs)
        return _imported_class

    def _try_import(self, module_name: str, class_name: str, backend_key: str):
        try:
            mod = __import__(f"alphapulldown.folding_backend.{module_name}", fromlist=[class_name])
            cls = getattr(mod, class_name)
            self._BACKEND_REGISTRY[backend_key] = cls
        except Exception as e:
            logging.warning(
                f"Failed to import {class_name}: {e}. Perhaps dependencies are not installed?"
            )

    def __repr__(self):
        return f"<BackendManager: using {self._backend_name}>"

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def __dir__(self) -> List:
        base_attributes = []
        base_attributes.extend(dir(self.__class__))
        base_attributes.extend(self.__dict__.keys())
        base_attributes.extend(dir(self._backend))
        return sorted(base_attributes)

    def change_backend(self, backend_name: str, **backend_kwargs: Dict) -> None:
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
