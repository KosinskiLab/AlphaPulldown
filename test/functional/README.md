Functional tests live here when they are deterministic, CPU-safe, and heavier than the
unit/integration layers. They may shell into the real feature-generation stack: CI runs
them with the tools from `environment.yml` (kalign, HMMER, HH-suite).

GPU or Slurm smoke wrappers belong under `test/cluster/`.
