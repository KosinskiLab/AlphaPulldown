"""Decide whether AlphaFold 3 can use the fork's fused triangle kernels here.

``--fast_kernels`` with ``--fold_backend=alphafold3`` turns on fused Pallas kernels for
the triangle multiplication and triangle attention layers of AF3's pair stack. They ship
with KosinskiLab's AlphaFold 3 fork (``alphafold3.jax.fused_triangle``), so unlike
AlphaFold 2's ``colabfold-kernels`` they need no extra package.

The fork decides per layer: a layer outside the measured GPUs, dtype, shapes or size
limits runs the original module body. This module checks the device once, before a model
is built, and runs both kernels on it, so that a GPU, driver or JAX that cannot compile
them fails now rather than in the middle of a prediction:

* ``off``: the original AF3 layers, exactly as before.
* ``on``: the kernels wherever a layer qualifies, or a ``ValueError`` saying why this
  device cannot run them.
* ``auto``: as ``on`` where the device can run them, the original layers (with a log
  line) elsewhere.

JAX and the fork are imported only when a mode other than ``off`` is resolved, or when
the record of a model that requested the kernels is written.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping

from absl import logging

from alphapulldown.prediction.fast_kernels import normalise_mode

if TYPE_CHECKING:
    import jax

_OPERATIONS = ("triangle_multiplication", "triangle_attention")
# Pair channels of AF3's triangle layers: 128 in the pair stack, 64 in the template stack.
_RECORDED_NUM_CHANNELS = (128, 64)


@dataclasses.dataclass(frozen=True)
class FusedTriangleChoice:
    """What ``resolve`` decided for AF3's triangle layers, and why."""

    enabled: bool
    reason: str
    mode: str = "off"
    settings: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def global_config_update(self) -> dict[str, Any]:
        """Fields for AF3's ``GlobalConfig``; empty when the kernels are off."""
        return dict(self.settings) if self.enabled else {}


OFF = FusedTriangleChoice(False, "--fast_kernels=off")


def device_settings(device: jax.Device) -> dict[str, Any]:
    """The ``GlobalConfig`` fields that turn the kernels on for ``device``.

    The fork's policy reads the device's compute capability and its JAX allocator budget,
    which set the per-operation size limits. No model parameters are loaded.

    Raises:
        RuntimeError: if the installed AF3 has no fused-triangle hooks, or the fork's
            policy does not enable the kernels on ``device``.
    """
    from alphafold3.model import model_config

    if not hasattr(model_config.GlobalConfig(), "triangle_multiplication_implementation"):
        raise RuntimeError(
            "this AlphaFold 3 installation has no fused-triangle hooks "
            "(alphafold3.jax.fused_triangle); install the KosinskiLab alphafold3 fork"
        )
    from alphafold3.jax.fused_triangle import dispatch

    memory_gib = (device.memory_stats() or {}).get("bytes_limit", 0) / 2**30
    policy = dispatch.device_policy(getattr(device, "compute_capability", ""), memory_gib)
    if not policy.enabled:
        raise RuntimeError(f"{getattr(device, 'device_kind', 'GPU')}: {policy.reason}")
    return {
        "triangle_multiplication_implementation": "pallas",
        "triangle_attention_implementation": "auto",
        "fused_triangle_compute_capability": policy.compute_capability,
        "fused_triangle_memory_gib": memory_gib,
    }


def smoke_test(settings: Mapping[str, Any], device: jax.Device) -> None:
    """Compile and run both fused triangle layers on ``device``, without model weights.

    One 64-token, 128-channel pair goes through TriangleMultiplication and then
    GridSelfAttention, configured with ``settings``.

    Raises:
        RuntimeError: if the output is not finite. Whatever compiling or running the
            kernels raises is passed on.
    """
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphafold3.model import model_config
    from alphafold3.model.components import utils
    from alphafold3.model.network import modules

    global_config = model_config.GlobalConfig(final_init="linear", **settings)

    def forward(act, mask):
        with utils.bfloat16_context():
            act = modules.TriangleMultiplication(
                modules.TriangleMultiplication.Config(equation="ikc,jkc->ijc"),
                global_config,
                name="triangle_multiplication",
            )(act, mask)
            return modules.GridSelfAttention(
                modules.GridSelfAttention.Config(),
                global_config,
                transpose=True,
                name="triangle_attention",
            )(act, mask)

    transformed = hk.without_apply_rng(hk.transform(forward))
    with jax.default_device(device):
        act = jnp.ones((64, 64, 128), jnp.bfloat16)
        mask = jnp.ones((64, 64), jnp.bfloat16)
        params = transformed.init(jax.random.PRNGKey(0), act, mask)
        output = jax.jit(transformed.apply)(params, act, mask)
        finite = bool(jnp.all(jnp.isfinite(output)))
    if not finite:
        raise RuntimeError("the fused triangle layers returned non-finite values")


def resolve(mode: Any, device: jax.Device) -> FusedTriangleChoice:
    """Turn a ``--fast_kernels`` value into a decision for ``device``, checking it first.

    Raises:
        ValueError: for an unknown mode, or for ``on`` when ``device`` cannot run the
            kernels.
    """
    mode = normalise_mode(mode)
    if mode == "off":
        return OFF
    try:
        settings = device_settings(device)
        smoke_test(settings, device)
    except Exception as exc:  # any failure here would recur inside the model
        problem = f"AF3 fused triangle kernels are unavailable: {exc}"
        if mode == "on":
            raise ValueError(f"--fast_kernels=on, but {problem}.") from exc
        logging.warning("Fused kernels off (--fast_kernels=auto): %s.", problem)
        return FusedTriangleChoice(False, f"--fast_kernels=auto: {problem}", mode)
    logging.info(
        "AF3 fused triangle kernels on; per-operation size limits apply: %s", settings
    )
    return FusedTriangleChoice(True, f"--fast_kernels={mode}", mode, settings)


def metadata(
    global_config: Any, choice: FusedTriangleChoice, *, num_tokens: int
) -> dict[str, Any]:
    """Which implementation AF3's triangle layers dispatch to for one padded bucket.

    Lists the 128-channel pair stack and the 64-channel template stack. Dispatch depends
    only on ``global_config`` and the shapes, so the record also holds when a compiled
    model comes from the compile cache. It does not claim that templates were supplied.
    """
    record = {
        "backend": "alphafold3",
        "requested_mode": choice.mode,
        "padded_tokens": num_tokens,
        "fused_kernels": False,
        "reason": choice.reason,
    }
    requested = {
        operation: getattr(global_config, f"{operation}_implementation", "default")
        for operation in _OPERATIONS
    }
    if all(implementation == "default" for implementation in requested.values()):
        return record
    from alphafold3.jax.fused_triangle import dispatch

    policy = dispatch.device_policy(
        global_config.fused_triangle_compute_capability,
        global_config.fused_triangle_memory_gib,
    )
    dtype = "float32" if global_config.bfloat16 == "none" else "bfloat16"
    operations = {}
    for num_channels in _RECORDED_NUM_CHANNELS:
        for operation, implementation in requested.items():
            selected, reason = dispatch.select_implementation(
                operation,
                implementation,
                policy,
                (num_tokens, num_tokens, num_channels),
                dtype,
                (num_tokens, num_tokens),
            )
            operations[f"{operation}_c{num_channels}"] = {
                "implementation": selected,
                "reason": reason,
            }
    record.update(
        fused_kernels=any(
            operation["implementation"] != "default" for operation in operations.values()
        ),
        operations=operations,
        policy_version=dispatch.POLICY_VERSION,
        device_policy=dataclasses.asdict(policy),
    )
    return record
