"""Decide whether AlphaFold 2 can use ColabFold's fused Pallas kernels here.

``--fast_kernels`` turns on the optional fused kernels of the ``colabfold-kernels``
package (attention, LayerNorm, triangle multiplication) in AlphaFold-Multimer, through
hooks in KosinskiLab's AlphaFold fork (``alphafold.model.fused_kernels``). They roughly
double AF2-Multimer inference speed on NVIDIA GPUs of compute capability 8.0 or newer.

AlphaPulldown runs on many clusters, so this checks every requirement before a model is
built, and actually runs one small kernel, rather than letting an unsupported GPU, driver
or JAX fail in the middle of a prediction:

* ``off``: the stock XLA code, exactly as before.
* ``on``: the kernels, or a ``ValueError`` saying which requirement is missing.
* ``auto``: the kernels where they work, the stock code (with a log line) elsewhere.

Monomer models run in fp32 and the kernels need bf16, so only multimer models use them.
JAX is imported only when a mode other than ``off`` is resolved.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import importlib.util
from typing import Optional

from absl import logging

MODES = ("off", "on", "auto")
# YAML 1.1 (PyYAML, so the Snakemake config) reads unquoted on/off as booleans, which
# then arrive here as "True"/"False". Accept the boolean spellings as on/off.
_ALIASES = {"true": "on", "yes": "on", "1": "on",
            "false": "off", "no": "off", "0": "off", "none": "off", "": "off"}
MIN_COMPUTE_CAPABILITY = 80


def normalise_mode(mode) -> str:
    """A --fast_kernels value as one of MODES; ValueError for anything else."""
    text = "off" if mode is None else str(mode).strip().lower()
    text = _ALIASES.get(text, text)
    if text not in MODES:
        raise ValueError(f"--fast_kernels must be one of {', '.join(MODES)}, not {mode!r}")
    return text


def is_valid_mode(mode) -> bool:
    try:
        normalise_mode(mode)
    except ValueError:
        return False
    return True


@dataclasses.dataclass(frozen=True)
class KernelChoice:
    """What ``resolve`` decided, and why."""

    enabled: bool
    reason: str
    compute_capability: Optional[int] = None
    package_version: Optional[str] = None

    def global_config_update(self) -> dict:
        """Keys for an AlphaFold-Multimer ``global_config``."""
        if not self.enabled:
            return {}
        return {"use_pallas": True, "compute_capability": self.compute_capability}


def _compute_capability(device) -> Optional[int]:
    value = getattr(device, "compute_capability", None)
    if value is None:
        return None
    text = str(value)
    return int(round(float(text) * 10)) if "." in text else int(text)


def _package_version() -> Optional[str]:
    try:
        return importlib.metadata.version("colabfold-kernels")
    except importlib.metadata.PackageNotFoundError:
        # Importable without install metadata (e.g. on PYTHONPATH): ask the module.
        try:
            return getattr(importlib.import_module("colabfold_kernels"), "__version__", None)
        except ImportError:
            return None


def _smoke_test() -> None:
    """Compile and run one small fused LayerNorm, so a broken toolchain shows up now."""
    import jax.numpy as jnp

    fused_kernels = importlib.import_module("alphafold.model.fused_kernels")
    kernel = fused_kernels.layer_norm({"use_pallas": True}, jnp.bfloat16)
    if kernel is None:
        raise RuntimeError("colabfold-kernels offered no bf16 LayerNorm kernel")
    x = jnp.ones((64, 128), jnp.bfloat16)
    scale = jnp.ones((128,), jnp.float32)
    offset = jnp.zeros((128,), jnp.float32)
    kernel(x, scale, offset, eps=1e-5).block_until_ready()


def _requirement_problem() -> tuple[Optional[str], Optional[int]]:
    """The first unmet requirement (None if all are met) and the GPU's capability."""
    if importlib.util.find_spec("colabfold_kernels") is None:
        return (
            "the colabfold-kernels package is not installed "
            "(pip install 'alphapulldown[fast-kernels]')",
            None,
        )
    try:
        importlib.import_module("alphafold.model.fused_kernels")
    except ImportError:
        return (
            "this AlphaFold installation has no fused-kernel hooks "
            "(alphafold.model.fused_kernels); update the KosinskiLab alphafold fork",
            None,
        )
    import jax

    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        return "JAX sees no GPU", None
    capability = _compute_capability(devices[0])
    if capability is None:
        return (
            f"{devices[0].device_kind} is not an NVIDIA CUDA GPU; the kernels need one",
            None,
        )
    if capability < MIN_COMPUTE_CAPABILITY:
        return (
            f"{devices[0].device_kind} has compute capability {capability / 10:.1f}; "
            f"the kernels need {MIN_COMPUTE_CAPABILITY / 10:.1f} or newer",
            capability,
        )
    try:
        _smoke_test()
    except Exception as exc:  # any failure here would recur inside the model
        return f"a test kernel failed on {devices[0].device_kind}: {exc}", capability
    return None, capability


def resolve(mode: str) -> KernelChoice:
    """Turn a ``--fast_kernels`` value into a decision, checking requirements first."""
    mode = normalise_mode(mode)
    if mode == "off":
        return KernelChoice(False, "--fast_kernels=off")
    problem, capability = _requirement_problem()
    if problem is None:
        choice = KernelChoice(
            True,
            f"--fast_kernels={mode}",
            compute_capability=capability,
            package_version=_package_version(),
        )
        logging.info(
            "Fused kernels on for AlphaFold-Multimer (colabfold-kernels %s, compute "
            "capability %.1f).", choice.package_version, capability / 10,
        )
        return choice
    if mode == "on":
        raise ValueError(f"--fast_kernels=on, but {problem}.")
    logging.warning("Fused kernels off (--fast_kernels=auto): %s.", problem)
    return KernelChoice(False, f"--fast_kernels=auto: {problem}", compute_capability=capability)
