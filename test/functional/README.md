Functional tests live here when they are deterministic, CPU-safe, and heavier than the
unit/integration layers.

Some functional suites may still carry the `external_tools` marker when they shell
into the real feature-generation stack or depend on heavyweight local runtimes.
Those stay in this directory because they are package-level workflow tests, but
they are still excluded from the default CPU-only pytest invocation.

GPU or Slurm smoke wrappers belong under `test/cluster/`.
