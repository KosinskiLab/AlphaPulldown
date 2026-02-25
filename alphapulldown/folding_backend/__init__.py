from __future__ import annotations

from typing import Dict, List, Optional, Type, Any
from importlib import import_module

from absl import logging

logging.set_verbosity(logging.INFO)


def _try_import(path: str, attr: str) -> Optional[type]:
    """Best-effort import helper. Returns None if import fails."""
    try:
        mod = import_module(path)
        return getattr(mod, attr)
    except Exception as e:
        logging.warning(f"Failed to import {path}:{attr}: {e}. Perhaps dependencies are missing.")
        return None


class FoldingBackendManager:
    """
    Manager for structure prediction backends.
    """

    def __init__(self):
        # Registry is lazy: map name -> "module.path:ClassName"
        self._BACKEND_REGISTRY: Dict[str, str] = {}

        self._BACKEND_REGISTRY.update(
            {
                "alphafold2": "alphapulldown.folding_backend.alphafold2_backend:AlphaFold2Backend",
                "alphafold3": "alphapulldown.folding_backend.alphafold3_backend:AlphaFold3Backend",
                "unifold": "alphapulldown.folding_backend.unifold_backend:UnifoldBackend",
                "alphalink": "alphapulldown.folding_backend.alphalink_backend:AlphaLinkBackend",
            }
        )

        self._backend_name: Optional[str] = None
        self._backend: Any = None
        self._backend_args: Dict[str, Any] = {}

        # Back-compat: previous code implicitly selected alphafold2 on init.
        self._default_backend_name: str = "alphafold2"

    def __repr__(self):
        if self._backend_name is None:
            return "<BackendManager: no backend selected>"
        return f"<BackendManager: using {self._backend_name}>"

    def __getattr__(self, name):
        if self._backend is None:
            raise AttributeError(
                f"No backend selected yet. Call change_backend(...) first. Missing attribute: {name}"
            )
        return getattr(self._backend, name)

    def __dir__(self) -> List[str]:
        base_attributes: List[str] = []
        base_attributes.extend(dir(self.__class__))
        base_attributes.extend(self.__dict__.keys())
        if self._backend is not None:
            base_attributes.extend(dir(self._backend))
        return sorted(set(base_attributes))

    def available_backends(self) -> List[str]:
        """Return backend names that can actually be imported in this environment."""
        ok: List[str] = []
        for name, spec in self._BACKEND_REGISTRY.items():
            mod, cls = spec.split(":")
            if _try_import(mod, cls) is not None:
                ok.append(name)
        return sorted(ok)

    def _load_backend_class(self, backend_name: str) -> Type:
        if backend_name not in self._BACKEND_REGISTRY:
            available = ", ".join(sorted(self._BACKEND_REGISTRY.keys()))
            raise NotImplementedError(
                f"Available backends are {available} - not {backend_name}."
            )
        spec = self._BACKEND_REGISTRY[backend_name]
        mod, cls = spec.split(":")
        backend_cls = _try_import(mod, cls)
        if backend_cls is None:
            raise ImportError(
                f"Backend '{backend_name}' is registered but could not be imported. "
                f"Missing dependencies? ({spec})"
            )
        return backend_cls

    def change_backend(self, backend_name: Optional[str] = None, **backend_kwargs: Dict) -> None:
        """
        Change the backend.

        Parameters
        ----------
        backend_name : str
            Name of the new backend that should be used.
            If None, uses the default backend preference (alphafold2).
        **backend_kwargs : Dict, optional
            Parameters passed to __init__ method of backend.

        Raises
        ------
        NotImplementedError
            If no backend is found with the provided name.
        ImportError
            If the backend exists but can't be imported due to missing deps.
        """
        if backend_name is None:
            backend_name = self._default_backend_name

        backend_cls = self._load_backend_class(backend_name)
        self._backend = backend_cls(**backend_kwargs)
        self._backend_name = backend_name
        self._backend_args = backend_kwargs


# Keep a module-level backend object, but make it lazy (no import-time side effects)
backend: Optional[FoldingBackendManager] = None


def _get_manager() -> FoldingBackendManager:
    global backend
    if backend is None:
        backend = FoldingBackendManager()
    return backend
    
backend = _get_manager()

def change_backend(backend_name: str, **backend_kwargs) -> None:
    """Change the backend for structure prediction."""
    mgr = _get_manager()
    mgr.change_backend(backend_name, **backend_kwargs)

