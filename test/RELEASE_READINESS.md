# Release Readiness

This branch has broad maintained test coverage, but the protection is layered.

## Always-on CI

GitHub Actions runs:

- `test/unit`
- `test/integration`
- coverage collection and reporting
- Python `3.10` and `3.11`

These lanes continuously protect:

- feature creation wiring and CLI dispatch
- fold parsing and object construction
- AF2 backend helper behavior
- AF3 backend helper behavior
- script entrypoint and wrapper argument handling
- post-prediction and ModelCIF integration paths
- CPU-safe AlphaLink helper logic

## Explicitly-run validation

The following are intentionally outside default CI and must be run explicitly when a change touches them:

- `test/cluster/check_alphafold2_predictions.py`
- `test/cluster/check_alphafold3_predictions.py`
- `test/cluster/check_alphalink_predictions.py`
- manual or cluster-backed GPU/Slurm smoke runs

Release-critical examples include:

- AF3 wrapper output isolation for combined JSON folds
- MMseqs-generated `.a3m` and `.pkl` / `.pkl.xz` feature artifacts
- AF3 species pairing regressions for issue `#588`
- AF2 and AF3 dimer inference quality checks such as `ipTM > 0.6`

## Not Continuously Protected

The following areas are only partially protected, optional, or report-only:

- `test/cluster` workflows
- `test/alphalink` workflows beyond CPU-safe helper tests
- legacy scenarios still parked under `test/outdated`
- analysis-pipeline utilities and some deeper ModelCIF internals
- Python `3.8`, which is still advertised in packaging but is not exercised by GitHub Actions

The coverage artifact is useful as an audit input, but it does not prove workflow correctness by itself. In particular, `python test/tools/check_function_coverage.py --report-only` highlights functions that were never executed in CI and should be treated as follow-up audit items, not automatic release blockers.

## Practical Release Gate

Before a final release, the expected evidence is:

1. PR smoke-tests and coverage are green.
2. AF3 JSON wrapper regressions are green.
3. Issue `#588` MMseqs/AF3 pairing regressions are green.
4. Manual or cluster-backed AF2 and AF3 dimer runs from MMseqs features complete successfully with acceptable confidence.

If those gates are green, the branch is in good release shape even though some heavyweight workflows are still validated outside default CI.
