The active pytest layout is:

- `test/unit` for fast helper and mocked component tests
- `test/integration` for CPU-only filesystem, CLI wiring, and module interaction tests
- `test/functional` for heavier workflow tests that still belong to the main package test tree
- `test/cluster` for Slurm, GPU, or workstation smoke wrappers that are run explicitly
- `test/outdated` for legacy tests kept for reference until they are rewritten or deleted

Notes:

- `pytest.ini` only collects `unit`, `integration`, and `functional`.
- `conftest.py` auto-applies markers from the directory layout.
- `test/alphalink` now holds AlphaLink fixture files used by optional tests and cluster smoke runs.
