"""Checks for the AlphaFold 3 GPU tests that need neither JAX nor a GPU.

check_alphafold3_predictions.py uses them, and test/unit/test_af3_gpu_checks.py tests
them on a CPU. Nothing here imports JAX: the test process must leave the GPU's memory to
the fold it starts.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, NamedTuple

# JAX's allocator takes this share of GPU memory unless the environment says otherwise.
_JAX_DEFAULT_MEMORY_FRACTION = 0.75


@dataclasses.dataclass(frozen=True)
class Gpu:
    """What nvidia-smi says about the visible GPU."""

    compute_capability: str
    memory_gib: float  # the card's memory, or its MIG slice's


class Support(NamedTuple):
    """Whether the fork's fused triangle kernels are enabled on a GPU.

    ``supported`` is None when that cannot be told; ``detail`` says what was compared.
    """

    supported: bool | None
    detail: str


def parse_nvidia_smi(query: str, listing: str = "") -> Gpu:
    """The first GPU in nvidia-smi's output.

    Args:
        query: the output of ``nvidia-smi --query-gpu=compute_cap,memory.total,
            mig.mode.current --format=csv,noheader``, e.g. ``9.0, 81559 MiB, Disabled``.
        listing: the output of ``nvidia-smi -L``. With MIG enabled, memory.total is the
            whole card's, so the memory is read from the first MIG profile listed there
            (``MIG 2g.20gb``).

    Raises:
        ValueError: if the output does not have that form.
    """
    lines = query.strip().splitlines()
    fields = [field.strip() for field in lines[0].split(",")] if lines else []
    if len(fields) < 2:
        raise ValueError(f"unexpected nvidia-smi output: {query!r}")
    memory = fields[1].split()
    if len(memory) != 2 or memory[1] != "MiB":
        raise ValueError(f"unexpected nvidia-smi memory.total: {fields[1]!r}")
    memory_gib = float(memory[0]) / 1024
    if len(fields) > 2 and fields[2] == "Enabled":
        profile = re.search(r"\bMIG\s+\S*?(\d+)gb\b", listing)
        if profile is None:
            raise ValueError(f"MIG is enabled, but nvidia-smi -L names no MIG device: {listing!r}")
        memory_gib = int(profile.group(1)) * 1e9 / 2**30
    return Gpu(fields[0], memory_gib)


def visible_gpu() -> Gpu | None:
    """The first GPU that nvidia-smi reports, or None if it reports none we can read."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        query, listing = (
            subprocess.run(
                [nvidia_smi, *args], capture_output=True, text=True, check=True
            ).stdout
            for args in (
                [
                    "--query-gpu=compute_cap,memory.total,mig.mode.current",
                    "--format=csv,noheader",
                ],
                ["-L"],
            )
        )
        return parse_nvidia_smi(query, listing)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def jax_memory_fraction(env: Mapping[str, str]) -> float:
    """The share of GPU memory JAX's allocator may take in a process with ``env``."""
    for name in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        if env.get(name):
            return float(env[name])
    return _JAX_DEFAULT_MEMORY_FRACTION


def installed_device_policy() -> Callable[[str, float], Any] | None:
    """The installed fork's ``device_policy``, or None for an AF3 without fused kernels.

    The fork's dispatch module decides without importing JAX, so this import is cheap.
    """
    try:
        from alphafold3.jax.fused_triangle import dispatch
    except ImportError:
        return None
    return dispatch.device_policy


def fused_triangle_support(
    gpu: Gpu | None,
    memory_fraction: float,
    device_policy: Callable[[str, float], Any] | None,
) -> Support:
    """Whether ``device_policy`` enables the fused kernels on ``gpu``, judged without JAX.

    The backend asks the same policy with the compute capability and allocator budget
    that JAX reports. Here the budget is estimated as ``memory_fraction`` of the memory
    nvidia-smi reports, so a test can check the backend's verdict against the hardware.
    """
    if device_policy is None:
        return Support(False, "the installed alphafold3 has no alphafold3.jax.fused_triangle")
    if gpu is None:
        return Support(None, "nvidia-smi reports no GPU it can describe")
    budget_gib = gpu.memory_gib * memory_fraction
    policy = device_policy(gpu.compute_capability, budget_gib)
    detail = (
        f"compute capability {gpu.compute_capability}, about {budget_gib:.1f} GiB for JAX "
        f"({memory_fraction:g} of {gpu.memory_gib:.1f} GiB)"
    )
    if policy.enabled:
        return Support(True, detail)
    return Support(False, f"{detail}: {policy.reason}")
