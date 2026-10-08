# Fused triangle kernels for AlphaFold 3

AlphaFold 3 spends much of its time in the triangle layers of its pair stack:
triangle multiplication and triangle attention. `--fast_kernels` can run these layers
as fused GPU kernels written in Pallas, JAX's language for custom kernels, which
combine several steps into one pass over memory. The kernels are Anthropic's
FlashPairformer triangle kernels, vendored into KosinskiLab's AlphaFold 3 fork. Off by
default; predictions without the flag run exactly as before.

The same flag turns on ColabFold's fused kernels for AlphaFold-Multimer, a separate
implementation; AlphaFold 3 does not need the `colabfold-kernels` package.

## Turning them on

```bash
run_structure_prediction.py --fold_backend=alphafold3 --fast_kernels=auto ...
```

| value | effect |
| --- | --- |
| `off` (default) | the original AlphaFold 3 layers |
| `on` | the fused kernels, or an error before any weights load if this GPU cannot run them |
| `auto` | the fused kernels where this GPU can run them, the original layers (with a warning) elsewhere |

`true`/`false` are read as `on`/`off`, so an unquoted YAML `on` works too.

Before building the model, AlphaPulldown checks the GPU against the measured
architectures and memory limits below, then compiles and runs both fused layers once
on a small input on that GPU. A GPU, driver or JAX that cannot compile them therefore
fails, or falls back, at that point rather than in the middle of a prediction.

The kernels need the `alphafold3` package at the submodule revision of this
AlphaPulldown checkout: build the AF3 environment or image from it
([backend installation](backend_installation.md#alphafold3-backend)). AlphaPulldown
2.9.1 and older do not accept `--fast_kernels` for AlphaFold 3. With an older
`alphafold3` package, `on` fails with "no fused-triangle hooks" and `auto` keeps the
original layers.

## What runs on which GPU

Each layer is decided on its own, and a layer that does not qualify runs the original
code; the rest of the model is unchanged. A layer qualifies only with bf16
activations, a square pair representation of 64 or 128 channels whose token count is
a multiple of 64, and (for attention) four heads. These are the shapes of AlphaFold 3's
pair stack (128 channels) and template stack (64 channels).

- **Triangle multiplication:** fused Pallas steps around XLA's matrix multiplication, on
  every measured architecture.
- **Triangle attention** comes in two variants. They differ in the attention core, the
  step that computes softmax(QKᵀ + bias)·V without storing the full score matrix
  (flash attention):
  - `pallas`: the whole layer in Pallas, including Anthropic's own Pallas
    flash-attention core.
  - `pallas_tokamax_core`: the same Pallas steps before and after the core, around the
    flash-attention core of tokamax, the DeepMind kernel library that AlphaFold 3
    already uses for attention (`--flash_attention_implementation` picks its kernel).
    Used where Anthropic's core was slower than the original layer.

| compute capability | GPUs measured | multiplication | attention |
| --- | --- | --- | --- |
| 8.0 | A100 | fused | `pallas` |
| 8.6 | A40, RTX 3090 | fused | `pallas_tokamax_core` |
| 8.9 | L40S | fused | `pallas_tokamax_core` |
| 9.0 | H100 | fused | `pallas` |
| 12.0 | RTX Pro 6000 Blackwell | fused | `pallas` (with the 9.0 tile sizes) |

Any other architecture keeps the original layers, including compute capability 10.0
(B200) and anything newer: a higher compute capability is not evidence that the
kernels compile, or are faster, there.

## Memory limits

Fused attention processes all rows of the pair representation at once, where the
original layer works through them in chunks, so it needs more memory. Each operation
therefore has a token limit set by the GPU memory JAX may allocate (the allocator's
`bytes_limit`, which `XLA_PYTHON_CLIENT_MEM_FRACTION` and similar settings change),
not by the card's advertised memory. Above its limit a layer runs the original code.

| JAX memory budget | multiplication up to | attention up to |
| --- | ---: | ---: |
| below 12 GiB | not used | not used |
| 12 to <20 GiB | 1,536 tokens | 768 tokens |
| 20 to <32 GiB | 2,560 tokens | 1,536 tokens |
| 32 to <64 GiB | 3,584 tokens (5,120 on 9.0 and 12.0) | 2,048 tokens |
| 64 GiB and more | 3,584 tokens (5,120 on 9.0 and 12.0) | 3,072 tokens |

Token counts are AlphaFold 3's padded bucket sizes (`--buckets`). The limits are
conservative settings, not measured capacities of the full model.

## How it was measured

The kernels were checked layer by layer ("Gate 0"): each fused layer against the
original layer and a float64 reference, with random and DeepMind weights and padded
masks, on A100, A40, RTX 3090, L40S, H100 and RTX Pro 6000 Blackwell. The full-model
comparison of `off` and `on` predictions ran on the same cards: whole predictions were
1.08-2.03x faster depending on card and size, the largest complex each card folds was
unchanged, and on 12 heterodimers released after AlphaFold 3's training cutoff the
ranking scores and DockQ of paired seeds agreed. Both used JAX 0.9.1 and tokamax
0.0.11. On the AlphaFold 3 v3.0.4 image (JAX 0.10.2, tokamax 0.0.12), the GPU test suite
checks on each card it runs on that `on` and `auto` use the kernels and that their
scores stay within 0.02 of the original layers.

The fused kernels are skipped while parameters are initialised, so parameter names,
shapes, initialisers and random-number order are those of the original model, and
DeepMind's weights load unchanged.

## Output record

Every AlphaFold 3 output directory gets an `inference_kernels.json` with one entry per
seed, keyed like AlphaFold 3's other outputs:

```json
{
  "seed-1": {
    "backend": "alphafold3",
    "requested_mode": "auto",
    "reason": "--fast_kernels=auto",
    "padded_tokens": 2560,
    "fused_kernels": true,
    "operations": {
      "triangle_attention_c128": {"implementation": "default", "reason": "size_limit"},
      "triangle_multiplication_c128": {"implementation": "pallas", "reason": ""},
      "...": "..."
    },
    "policy_version": 1,
    "device_policy": {"compute_capability": "8.6", "memory_gib": 44.5, "...": "..."}
  }
}
```

`operations` states, for the 128-channel pair stack and the 64-channel template stack,
which implementation a layer of that bucket runs (`default` is the original layer) and
why it falls back. It is derived from the configuration and the bucket, so it also
holds for a model loaded from the compile cache. It does not mean that templates were
supplied. When the kernels are off, the entry has no `operations`. Writing the record
never fails a prediction: an unreadable file is replaced, and other problems are
logged.

## Troubleshooting

`--fast_kernels=on` stops before loading weights and gives the reason; `auto` logs the
same reason as `Fused kernels off (--fast_kernels=auto): ...` and keeps the original
layers. The reason starts with one of two phrases:

- `AF3 fused triangle kernels are not supported here: ...` (a warning under `auto`):
  this installation or GPU is not one the kernels are enabled on.

  | reason contains | meaning |
  | --- | --- |
  | `no fused-triangle hooks` | the installed `alphafold3` predates the kernels; rebuild it from the submodule |
  | `unvalidated_compute_capability` | the GPU's architecture was not measured (table above) |
  | `unknown_memory_budget` | JAX reports no allocator limit for the GPU |
  | `memory_budget_below_12_gib` | JAX may allocate less than 12 GiB |

- `AF3 fused triangle kernels failed their self-check: ...` (an error, with the
  traceback, under `auto`): the GPU is supported, but compiling or running the small
  check on it failed, or the kernels returned non-finite values. That points to a
  kernel, driver or JAX problem; please report it with the log.

During a prediction, a warning such as `triangle_attention_implementation='auto' was
requested, but a layer ... runs the default implementation (size_limit)` is expected
for buckets above the limits. It is logged once per reason.

Scripts that import `alphapulldown.folding_backend.alphafold3_backend` themselves
should call `jax.devices()` before that import. In the AlphaFold 3 image, importing the
backend first can leave JAX with only its CPU and TPU platforms
(`Backend 'cuda' is not in the list of known backends`), so the GPU check cannot see
the GPU.

## Source and licence

The kernel files are vendored unchanged from Anthropic's
[uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling)
repository (Apache License 2.0). The fork's `src/alphafold3/jax/fused_triangle/SOURCE.md`
names the commit and the files, and `NOTICE` carries the attribution.
