# AF3 integration validation

`bash submit.sh full [a40 h100 a100 l40s 3090 rtx6000]` freezes committed
AlphaPulldown, AF3 fork and harness sources, then submits QOS `high` SLURM jobs.
`pilot` limits the campaign to layers, bit identity and the public `auto` CLI.
Every job gets a unique output directory; no prior results are deleted.

The fork is compared against `86b9ea3`, fork main updated to AlphaFold 3 v3.0.3
before the fused-triangle work, which is also what the AlphaPulldown 2.9.0
image ships. Each arm binds the fork's `alphafold3/model` and `alphafold3/jax`
(adapters, kernels, dispatch) over the image's copies; `baseline` binds both
directories at `86b9ea3`.

For each GPU, the integrated layer sweep must pass before the bit-identity job
runs. That job shares a persistent cache between the original fork, the new
fork with the flag off, and the fused path; it checks original/off identity,
fused repeatability and that the fused path dispatches every triangle
operation. The subsequent speed, accuracy and input jobs depend on it. Invalid
dependencies are cancelled, so the `afterany` collector can finish. Workers,
collector and supervisor are batch jobs and survive logout.

The layer sweep uses Gate 0's independent float64 reference and random plus
DeepMind parameters from trunk layers 0/24/47 and template layers 0/1. It runs
the fork's own modules with its fused-triangle `GlobalConfig` fields. It
requires all 66 fused records, each dispatched by the fork to the requested
kernel (a stock fallback would pass trivially), padding/leak, finiteness and
repeat checks, and a failing wrong-bias-orientation control.

Full campaigns run the existing size ladder, ending with the 3,584- and
5,376-token capacity probes, 12 heterodimers (stock seeds 0–2, fused 0–1 on
H100/A100/A40), separate multiplication/attention ablations, protein/RNA,
protein/ligand and a populated template input. A failed fold, OOM included,
stops the remaining folds of that arm and seed; the next seed or arm still
runs. The probes come last, so an OOM there costs no smaller fold.

## Arms

| arm | what runs |
| --- | --- |
| `baseline` | fork at `86b9ea3`, `--fast_kernels=off` (identity suite only) |
| `off` | fork under test, `--fast_kernels=off`: the original AF3 layers |
| `on` | fork under test, `--fast_kernels=on`: fused triangle layers wherever the fork's dispatch allows |
| `auto` | `--fast_kernels=auto` through the public CLI (`cli` suite) |
| `trimul`, `attention` | `on` with only triangle multiplication or only triangle attention fused (`launch.py` sets the other operation's `*_implementation` to `default`) |

## What the launcher records

The launcher instruments the public batch CLI; it does not replace the model
classes. It records full forward-result hashes, finiteness, package versions,
dispatch metadata (AlphaPulldown's `af3_fused_triangles.metadata`) and
allocator peak bytes. A disposable `nvidia-smi` sampler records total GPU
memory every 500 ms. Compilation is logged separately.

## Report

`collect.py` writes `REPORT.md` and `REPORT.json`:

- **Gates**: the latest layer-gate and bit-identity job per card.
- **Accuracy**, the pre-registered paired rule: stock and fused are paired by
  seed. A fold is confident if every stock seed ranks ≥ 0.7; it passes if mean
  |Δranking| ≤ 0.02 and mean |ΔDockQ| ≤ 0.05 over the paired seeds. Other folds
  report whether every fused value lies within the stock seed range. A fold is
  flagged `follow_up` when a paired |ΔDockQ| > 0.2, a paired ranking crosses 0.7
  in one arm only, or a confident fold misses the thresholds. Each trigger names
  its outlier: `stock` (that stock seed lies outside the other stock seeds'
  range, i.e. seed-to-seed variation) or `fused` (the fused value lies outside
  the stock range). A low-confidence flip needs a reviewed 10-seed follow-up;
  other triggers need investigation, which the outlier side directs.
- **Capacity**: per card and speed arm, the largest completed fold and the first
  fold that ran out of memory, reported as a capacity limit, not a failure, and
  "capacity unchanged" when stock and fused stop at the same fold. Speed jobs
  therefore end `FAILED` in sacct whenever their last probe runs out of memory.
- **Speed**: ratios only for completed finite runs with exactly two predictions,
  zero model compilation on the second prediction, and matching GPU models.

No production speed or accuracy gate is declared passed by submission.

## Supervisor

`submit.sh` ends by calling `start_supervisor.sh CAMPAIGN`, which freezes the
committed harness into `CAMPAIGN/supervisor/harness` and queues three
singleton runs of `supervise.sbatch` on `htc-el8,htc-el10` (3 days each; a
successor resumes from `supervisor/tasks.json`). Run it by hand to attach a
supervisor to an older campaign.

- **Retried** (up to 3 attempts per task, failing node excluded): NODE_FAIL,
  BOOT_FAIL, PREEMPTED, DEADLINE; exit 3 (`nvidia-smi` failed); exit 127 (a
  missing interpreter); logs showing that CUDA, the container or storage failed;
  one TIMEOUT (1.5× walltime, at most 12 h). A child cancelled because its
  parent failed is resubmitted after the parent's retry.
- **Never retried**: a failing gate or identity check, an OOM in a capacity
  probe, any other non-zero exit, a user cancel.

When every task is final, the supervisor writes `sacct.txt` for all attempts,
runs the frozen collector (one row per trial, the latest attempt wins) and
then `SUPERVISOR_RESULT.txt`. That file marks the final report. The plain
`afterany` collector (`RESULT.txt`) runs when the first attempts end, so its
`REPORT.md` is provisional while retries are pending. Progress is in
`supervisor/HEARTBEAT.json` and `supervisor/supervisor.log`.

## QOS

Jobs are submitted with `--qos=high`, which caps a user at 8 GPUs (128 CPUs)
at a time. The rest wait with reason `QOSMaxGRESPerUser`. To let them start
under `normal` instead, move pending jobs in place:
`scontrol update jobid=<id> qos=normal`.

## Outputs

The final collector writes `RESULT.txt`, `REPORT.md`, `REPORT.json` and
`sacct.txt` in the printed campaign directory. Per-job `RESULT.txt` includes
the exit code and frozen source path. A successful collector exit means that
collection completed; it does not mean that every GPU or accuracy check passed.
