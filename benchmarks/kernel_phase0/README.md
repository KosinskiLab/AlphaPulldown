# Phase 0: do fused AF2/AF3 kernels pay off for AlphaPulldown?

Experimental benchmark, branch `exp/kernel-bench-phase0`. It changes no AlphaPulldown code. It
decides which fused-kernel source, if any, Phase 1 should port. The two candidates:

- **ColabFold's own kernels.** ColabFold 1.6.3 / alphafold-colabfold 2.3.20 (`--use-fast-kernels`):
  Pallas/Triton, maintained upstream, the same AF2 model code lineage as our fork.
- **Anthropic's ColabFold kit.** From [uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling),
  pinned at `f4f62fa6` and built on ColabFold 1.6.1: about 4.2× on H100 against ColabFold
  without fused kernels. It is unmaintained upstream.

## Questions and the comparisons that answer them

| question | comparison |
|---|---|
| How much do ColabFold's own kernels give on our cards? | `cf163_fast` vs `cf163_stock` |
| How much do Anthropic's kernels give? | `kit_fast` (and `kit_exact`) vs `kit_off` |
| Do Anthropic's kernels beat ColabFold's? | `kit_fast` vs `cf163_fast` |
| Does a ColabFold speedup transfer to AlphaPulldown? | `ap_stock` vs `cf163_stock` / `kit_off` on the same input |
| Do the fast kernels change predictions more than a seed does? | accuracy suite: DockQ and ranking confidence vs the stock arms' seed-to-seed spread |
| What will an AF3 port (Phase 2) be measured against? | `ap_af3`: AlphaPulldown AF3 with DeepMind weights, plus the largest fold each card holds |

The AF3 side has no optimized arm. Anthropic's AF3 kit runs only sokrypton's fork with
OpenFold3 weights, so it cannot be measured on our model.

## What makes the comparison fair

- **Identical inputs.**
  - `make_inputs.py` builds each fold with AlphaPulldown's own code (species pairing, de-duplication, cropping).
  - It writes the per-chain MSAs as a ColabFold complex a3m.
  - `verify_inputs.py` parses that a3m with ColabFold's own functions, in both ColabFold versions,
    and requires the final feature arrays to equal AlphaPulldown's, bit for bit (`inputs/verify_*.tsv`).
- **The same work in every arm.**
  - model_1_multimer_v3, `NUM_RECYCLE` recycles with early stopping off, no templates, no relaxation, one seed.
  - ColabFold's 10-residue recompile padding is off (`--recompile-padding 0`).
  - AlphaPulldown's early stop is disabled by `ap_fixed_recycles.py`, a wrapper outside its code.
    The wrapper also logs each `predict` call's time and the number of recycles it ran.
- **Compile time excluded.**
  - Each fold runs in its own process with `REPS` copies of the input. Rep 1 compiles; the median of reps 2.. is the forward time.
  - ColabFold reports times to 0.1 s, so the smallest folds on fast cards carry about ±3 %.
- **One memory environment.**
  - No unified memory and a 0.95 cap.
  - The allocator grows instead of preallocating, so the 500 ms `nvidia-smi` samples track real use.
- **Kits that don't engage are visible.**
  - Kit runs keep their `ACTIVE` / `NOT ACTIVE` and per-lever counters (`report/kit_levers.json`).
  - A lever that falls back to stock shows up instead of hiding in an average.

## Inputs

All inputs are frozen into `$BENCH/inputs` with `SHA256SUMS`.

- **Speed ladder** (`folds.tsv`): 164 to 2,546 tokens.
  - Real heterodimers from the 2026-09 MMseqs2 benchmark.
  - The large rungs stack copies of them so every rung keeps a deep, paired MSA.
- **AF3 reach probes:** 3,584 and 5,376 tokens, only on cards with ≥ 40 GB.
- **Accuracy set:** the 12 heterodimers of `af2_mmseqs_bench`.
  - All were released after AF2-multimer's training cutoff.
  - Features were built with a 2021-09-30 template cutoff, and each has an experimental structure for DockQ.

## Running it

Everything is a SLURM job chained by dependencies, so the submitting shell can be closed.
`$BENCH` (default `/scratch/dima/kernel_bench_phase0`) holds images, inputs, runs and the report.

```bash
cd benchmarks/kernel_phase0
./submit.sh smoke                    # every arm + AF3 on the 164-token fold, one 3090 (SMOKE_GPU)
./submit.sh full                     # A40, 3090, H100: speed ladder + AF3 baseline per card, accuracy on the A40
./submit.sh full l40s a100 rtx6000   # second wave on gpu-el8 (queues for weeks as of 2026-10-01)
./submit.sh collect                  # rebuild the report from whatever has finished
```

Card routing and its reasons are in `bench.env`. As of 2026-10-01:
- gpu-el10's H100 and L40S nodes have a driver mismatch and are excluded (`EXCLUDE_NODES`).
- H100 runs on gpu-training only, about 11 days' queue.
- A job waiting in the queue runs the scripts as they are in the worktree when it starts, and
  records that commit in its `meta.txt`.
- One collector per card refreshes `REPORT.md` as each card finishes.

`submit.sh` first builds or pulls the four images (`images/`) and the inputs when they are
missing. Every job ID lands in `$BENCH/jobs.tsv`.

Results:
- `$BENCH/RESULT.txt`: status and paths.
- `$BENCH/REPORT.md`: the tables above.
- `$BENCH/report/runs.tsv`: one row per fold and arm, with forward, compile, per-pass and wall seconds, peak MiB, confidences and DockQ.
- `$BENCH/report/speedups.tsv`: per-fold ratios.

A run directory, `runs/<suite>/<gpu>/<arm>/`, holds `RESULT.txt`, `runs.tsv`, `gpu.txt`,
`gpu_mem.csv`, a log per fold and the raw outputs.

## Images

| image | what | how |
|---|---|---|
| `alphapulldown_af2_2.9.0.sif`, `alphapulldown_af3_2.9.0.sif` | our release images | `apptainer pull docker://kosinskilab/…:2.9.0` |
| `colabfold_1.6.3.sif` | stock ColabFold 1.6.3 + jax 0.11.2 from PyPI | `images/colabfold-1.6.3.def`; no published image exists for 1.6.3 |
| `colabfold_kit_f4f62fa6.sif` | Anthropic's kit on ColabFold 1.6.1 / jax 0.5.3 | `images/colabfold-kit.def` |

`images/colabfold-kit.def` mirrors the kit's own Dockerfile step for step. The kit's
`apptainer.def` needs a local Docker daemon, which this cluster lacks.

## Caveats

- **Template rows.** AlphaPulldown's `--skip_templates` embeds one empty template row and ColabFold
  embeds four. This makes AlphaPulldown's trunk a few percent cheaper in `ap_stock` vs `cf163_stock`;
  speedups within one tool are unaffected.
- **Different jax versions.** ColabFold 1.6.3 runs jax 0.11.2, the kit jax 0.5.3, and our AF2 image
  jax 0.5.3. Each arm runs as its maintainers ship it, which is the comparison that matters for
  choosing a source. Porting ColabFold's kernels would still need jax ≥ 0.6 in our AF2 stack, or a shim.
- **Limited accuracy check.** The accuracy suite is a sanity check: 12 complexes, one model, one seed.
  It is there to catch kernels that are wrong, as ColabFold 1.6.2's were. It cannot measure a small accuracy change.
- **Some cards are not covered.** B200 (node `bgx1`) is reserved and is not in the matrix.
