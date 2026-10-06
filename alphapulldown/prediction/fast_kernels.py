"""Resolve optional fused kernels before constructing AF2 or AF3 models.

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

AF2 monomer models run in fp32 and the kernels need bf16, so only multimer models use them.
AF3 uses the fork's vendored triangle kernels with independent per-operation limits.
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


def resolve(mode: str, *, backend="alphafold2", device=None) -> KernelChoice | AF3KernelChoice:
    """Turn a ``--fast_kernels`` value into a decision, checking requirements first."""
    if backend == "alphafold3":
        return _resolve_af3(mode, device=device)
    if backend != "alphafold2":
        raise ValueError(f"Fused kernels are not supported by {backend}")
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


@dataclasses.dataclass(frozen=True)
class AF3KernelChoice:
    """Validated AF3 device policy, separate from AF2's optional dependency."""

    enabled: bool
    reason: str
    settings: dict = dataclasses.field(default_factory=dict)

    def global_config_update(self) -> dict:
        return dict(self.settings) if self.enabled else {}


def _af3_smoke_test(settings):
    """Compile and execute both complete fused blocks before loading weights."""
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphafold3.model import model_config
    from alphafold3.model.components import utils
    from alphafold3.model.network import modules

    config = model_config.GlobalConfig(final_init='linear', **settings)
    def fn(x, mask):
        with utils.bfloat16_context():
            x = modules.TriangleMultiplication(
                modules.TriangleMultiplication.Config(equation='ikc,jkc->ijc'),
                config, name='trimul')(x, mask)
            return modules.GridSelfAttention(
                modules.GridSelfAttention.Config(), config, transpose=True,
                name='attention')(x, mask)
    transformed = hk.without_apply_rng(hk.transform(fn))
    x = jnp.ones((64, 64, 128), jnp.bfloat16)
    mask = jnp.ones((64, 64), jnp.bfloat16)
    params = transformed.init(jax.random.PRNGKey(0), x, mask)
    output = jax.jit(transformed.apply)(params, x, mask)
    output.block_until_ready()
    if not bool(jnp.all(jnp.isfinite(output))):
        raise RuntimeError('AF3 fused smoke returned non-finite values')


def _af3_settings(device):
    """Inspect the fork and allocator budget; do not load model parameters."""
    from alphafold3.model import model_config
    from alphafold3.model.network.fused_triangle import fpf_pallas_serve as dispatch

    if not hasattr(model_config.GlobalConfig(), 'fused_triangle_multiplication'):
        raise RuntimeError('the AlphaFold 3 fork has no fused-triangle hooks')
    if device is None:
        import jax
        device = jax.local_devices(backend='gpu')[0]
    memory = (device.memory_stats() or {}).get('bytes_limit', 0) / 2**30
    policy = dispatch.card_policy(getattr(device, 'compute_capability', ''), memory)
    if not policy.enabled:
        raise RuntimeError(f'{getattr(device, "device_kind", "GPU")}: {policy.reason}')
    return dict(fused_triangle_multiplication=True, fused_triangle_attention='auto',
                fused_triangle_compute_capability=policy.compute_capability,
                fused_triangle_memory_gib=memory)


def _resolve_af3(mode, *, device=None):
    mode = normalise_mode(mode)
    if mode == 'off':
        return AF3KernelChoice(False, '--fast_kernels=off')
    try:
        settings = _af3_settings(device)
        _af3_smoke_test(settings)
    except Exception as exc:
        problem = f'AF3 fused triangle kernels are unavailable: {exc}'
        if mode == 'on':
            raise ValueError(f'--fast_kernels=on, but {problem}') from exc
        logging.warning('Fused kernels off (--fast_kernels=auto): %s', problem)
        return AF3KernelChoice(False, problem)
    logging.info('AF3 fused triangle kernels enabled; per-operation size limits apply: %s', settings)
    return AF3KernelChoice(True, f'--fast_kernels={mode}', settings)


def af3_metadata(config, *, requested_mode, tokens, reason=''):
    """Resolve provenance for this padded bucket, including cache-hit runs.

    Lists dispatch for AF3's standard C=128 pair and C=64 template modules.
    It is not a trace counter or a claim that populated templates exist.
    """
    enabled = (getattr(config, 'fused_triangle_multiplication', False)
               or getattr(config, 'fused_triangle_attention', 'off') != 'off')
    result = dict(backend='alphafold3', requested_mode=requested_mode,
                  padded_tokens=tokens, fused_kernels=False, reason=reason)
    if not enabled:
        return result
    from alphafold3.model.network.fused_triangle import fpf_pallas_serve as dispatch
    dtype = 'float32' if config.bfloat16 == 'none' else 'bfloat16'
    selections = {}
    for channels in (128, 64):
        for kind in ('trimul', 'attention'):
            backend, fallback = dispatch.select(
                config, kind, (tokens, tokens, channels), dtype, (tokens, tokens))
            selections[f'{kind}_c{channels}'] = dict(backend=backend, reason=fallback)
    result.update(fused_kernels=any(v['backend'] != 'stock' for v in selections.values()),
                  operations=selections, source_commit=dispatch.SOURCE_COMMIT,
                  policy_version=dispatch.POLICY_VERSION,
                  device_policy=dataclasses.asdict(dispatch.policy_from_config(config)))
    return result
