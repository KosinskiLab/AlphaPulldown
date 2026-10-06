# Experimental AF3 fused triangles

Use `--fold_backend=alphafold3 --fast_kernels=on` to enable the bundled triangle
kernels. `off` is the default. `auto` enables them after a device check and a
small compile/run check, and otherwise logs the reason and uses stock AF3.
`on` reports an error if that check fails. Both modes fall back independently
for layers outside the supported dtype, shape or size limits.

Install the matching `alphafold3` submodule revision in the AF3 environment, or
build an AF3 image from this branch. Existing released images do not contain
these hooks. AF3 does not need the AF2 `colabfold-kernels` optional package.

The measured architectures are sm_80, sm_86, sm_89, sm_90 and sm_120. Multiplication
uses fused Pallas steps around XLA matrix multiplication. Attention uses the
kit's core on sm_80/90/120 and tokamax's core on sm_86/89. Only bf16 square pair
layers with 64 or 128 channels and four attention heads are enabled. Unmeasured
architectures fall back; compute capability alone is not evidence for new cards.

Attention removes row chunking and needs more memory. Initial token limits use
the JAX allocator budget, rather than the card's advertised memory:

| Allocator budget | Multiplication limit | Attention limit |
| --- | ---: | ---: |
| 12–<20 GiB | 1,536 | 768 |
| 20–<32 GiB | 2,560 | 1,536 |
| 32–<64 GiB | 3,584 (5,120 on sm_90/120) | 2,048 |
| ≥64 GiB | 3,584 (5,120 on sm_90/120) | 3,072 |

These are conservative experimental limits, not guarantees of full-model
capacity. The kernels are off during parameter initialisation, preserving names,
shapes, initialisers and random-number ordering for existing DeepMind weights.

Each output directory contains `inference_kernels.json`, keyed by seed, recording
the padded bucket, device policy, kernel source revision and selections for the
128-channel pair and 64-channel template layers. It records dispatch eligibility,
including cache hits; it does not claim that populated templates were supplied.

Layer measurements justify an end-to-end pilot, not a production speed claim.
Keep this feature opt-in until the paired speed, memory and DeepMind-weight
accuracy validation is complete. Kernel source and attribution are recorded in
the fork's `fused_triangle/SOURCE.md` and `NOTICE`.
