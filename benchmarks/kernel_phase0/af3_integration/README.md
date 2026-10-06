# AF3 integration validation

`bash submit.sh full [a40 h100 a100 l40s 3090 rtx6000]` freezes committed
AlphaPulldown, AF3 and harness sources, then submits QOS `high` SLURM jobs.
`pilot` limits the campaign to layers, bit identity and the public `auto` CLI.
Every job gets a unique output directory; no prior results are deleted.

For each GPU, the integrated layer sweep must pass before the bit-identity job
runs. That job shares a persistent cache between the original fork, the new
fork with the flag off, and the fused path; it checks original/off identity,
fused repeatability and enabled dispatch. The subsequent speed, accuracy and
input jobs depend on it. Invalid dependencies are cancelled, so the `afterany`
collector can finish. Workers and collector are batch jobs and survive logout.

The layer sweep uses Gate 0's independent float64 reference and random plus
DeepMind parameters from trunk layers 0/24/47 and template layers 0/1. It runs
the actual new module hooks. It requires all 66 fused records, padding/leak,
finiteness and repeat checks, and a failing wrong-bias-orientation control.

Full campaigns run the existing size ladder and 5,376-token capacity probe,
12 heterodimers (stock seeds 0–2, fused 0–1 on H100/A100/A40), separate
multiplication/attention ablations, protein/RNA, protein/ligand and a populated
template input. A failure or OOM stops the remaining folds in that process arm.
Capacity probes can fail without invalidating smaller successful trials.

The launcher instruments the public batch CLI; it does not replace the model
classes. It records full forward-result hashes, finiteness, package versions,
dispatch metadata and allocator peak bytes. A disposable `nvidia-smi` sampler
records total GPU memory every 500 ms. Compilation is logged separately.
`collect.py` forms speed ratios only for completed finite runs with exactly two
predictions, zero model compilation on the second prediction, and matching GPU
models. Raw memory and confidence/DockQ seed comparisons are in `REPORT.json`.

The experimental `on` speed arm corresponds to `ap_af3_fast`; `off` uses current
stock behavior. No production speed or accuracy gate is declared passed by
submission. Low-confidence flips require a reviewed 10-seed follow-up, and
confident changes outside the stock seed range require investigation.

The final collector writes `RESULT.txt`, `REPORT.md`, `REPORT.json` and
`sacct.txt` in the printed campaign directory. Per-job `RESULT.txt` includes
the exit code and frozen source path. A successful collector exit means that
collection completed; it does not mean that every GPU or accuracy check passed.
