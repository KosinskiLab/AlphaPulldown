# Backend Installation Guide

This guide is for direct AlphaPulldown use without Snakemake.

Two points matter in practice:

1. Run the install commands from the AlphaPulldown repo root.
2. For AlphaFold3, building the vendored `alphafold3` package is a separate step. A root-level install alone is not enough.

The Docker files remain the long-term reference environments, but the commands below are the simpler cluster-facing paths.
We revalidated them on EMBL on March 30, 2026 by creating fresh environments and running the cluster test suites there.

## Known-good cluster stacks

These are the two EMBL environments that were already working and that we rechecked while validating the wrappers:

- `AlphaPulldown` for AF2:
  - Python `3.10`
  - `jax 0.5.3`
  - `jaxlib 0.5.3`
  - `numpy 1.26.4`
  - `tensorflow-cpu 2.20.0`
  - `openmm 8.1.1`
  - `pdbfixer 1.12`
  - `modelcif 1.6`
- `AlphaPulldown_alphafold3` for AF3:
  - Python `3.11`
  - `jax 0.5.3`
  - `jaxlib 0.5.3`
  - `numpy 1.26.4`
  - `tensorflow-cpu 2.18.0`
  - `openmm 8.3.1`
  - `pdbfixer 1.12`
  - `modelcif 1.6`
  - `jax-triton 0.2.0`
  - `triton 3.1.0`
  - `rdkit 2024.3.5`
  - `typeguard 2.13.3`
  - compiled `alphafold3.cpp`

The installation steps below are written to stay close to those working environments and to keep user-facing environment variables to a minimum.

## AlphaFold2 backend

Create the environment, activate it, move into the repo, and install:

```bash
mamba create -y -n apd-af2 -c conda-forge -c bioconda \
  python=3.10 \
  kalign2 \
  hmmer \
  hhsuite
mamba activate apd-af2
cd /path/to/AlphaPulldown
python -m pip install ".[alphafold2,test]"
```

Check that GPU JAX is visible:

```bash
python - <<'PY'
import jax
print(jax.__version__)
print(jax.local_devices(backend="gpu"))
PY
```

## AlphaFold3 backend

AlphaFold3 needs both the AlphaPulldown install and the vendored `alphafold3` build.

```bash
mamba create -y -n apd-af3 -c conda-forge -c bioconda \
  python=3.11 \
  kalign2 \
  hmmer \
  hhsuite \
  libcifpp \
  sqlite
mamba activate apd-af3
cd /path/to/AlphaPulldown
python -m pip install ".[alphafold3,test]"
python -m pip install --no-deps -e ./alphafold3
build_data
```

Check that the compiled extension and GPU JAX are available:

```bash
python - <<'PY'
import alphafold3.cpp
import jax
print(jax.__version__)
print(jax.local_devices(backend="gpu"))
print(alphafold3.cpp.__file__)
PY
```

Notes:

- The working EMBL AF3 environment is closer to `.[alphafold3,test]` than to the upstream `alphafold3/dev-requirements.txt` stack.
- For cluster installs, prefer `python -m pip install ".[alphafold3,test]"` and then build the vendored `alphafold3` package.
- The compiled `alphafold3.cpp` extension comes from `python -m pip install --no-deps -e ./alphafold3`, not from the root install.
- The vendored AF3 package provides the `build_data` entry point. Use that directly.
- If you are actively developing inside the checkout and want the root package editable as well, add:

```bash
python -m pip install -e . --no-deps
```

## Cluster validation

On EMBL, the AF2 and AF3 functional tests already have default database roots baked into the test files, so the only environment variable you need for the standard cluster runs is:

```bash
export RUN_GPU_FUNCTIONAL_TESTS=1
```

Set `ALPHAFOLD_DATA_DIR` only if your databases are not in the EMBL default locations.

Two ways to run the cluster tests:

1. Direct `srun` + `pytest`
   - Best for validating a fresh install end-to-end.
   - Keeps everything on one allocated GPU node.
   - More reliable than the wrappers when Slurm priority is poor.
   - Important: when calling `pytest` directly, pass `-o addopts="-ra --strict-markers"`. The repo-level `pytest.ini` excludes cluster tests by default.
2. Wrapper scripts
   - `test/cluster/run_alphafold2_predictions.py`
   - `test/cluster/run_alphafold3_predictions.py`
   - Faster when the queue is healthy, because they fan out one job per pytest node.

### AlphaFold2

Direct full-suite validation:

```bash
srun -p gpu-training --gres=gpu:1 \
  --cpus-per-task=4 --mem=16G --time=12:00:00 \
  bash -lc '
    cd /path/to/AlphaPulldown
    export RUN_GPU_FUNCTIONAL_TESTS=1
    python -m pytest -o addopts="-ra --strict-markers" -vv -s \
      test/cluster/check_alphafold2_predictions.py --use-temp-dir
  '
```

Expected result on the standard suite:

```text
11 passed, 1 skipped
```

The skip is the opt-in MMseqs functional inference check, which only runs when `RUN_MMSEQS_FUNCTIONAL_TESTS=1` is set.

Preview the collected nodes for wrapper mode:

```bash
python test/cluster/run_alphafold2_predictions.py --list
```

Wrapper-based parallel submission:

```bash
python test/cluster/run_alphafold2_predictions.py \
  --max-tests 6 \
  --use-temp-dir \
  --partition gpu-training
```

### AlphaFold3

Direct full-suite validation:

```bash
srun -p gpu-training --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=12:00:00 \
  bash -lc '
    cd /path/to/AlphaPulldown
    export RUN_GPU_FUNCTIONAL_TESTS=1
    python -m pytest -o addopts="-ra --strict-markers" -vv -s \
      test/cluster/check_alphafold3_predictions.py --use-temp-dir
  '
```

On our fresh EMBL validation run, `32G` was not enough for the full AF3 suite on `hgx5`; `64G` was.

Preview the collected nodes for wrapper mode:

```bash
python test/cluster/run_alphafold3_predictions.py --list
```

Wrapper-based parallel submission:

```bash
python test/cluster/run_alphafold3_predictions.py \
  --max-tests 6 \
  --use-temp-dir \
  --partition gpu-training
```

## Troubleshooting

### AlphaFold2: `Unknown backend: 'gpu' requested, ... Platforms present are: cpu`

That means the environment has CPU-only JAX:

```bash
python -m pip install --upgrade --no-cache-dir "jax==0.5.3" "jax[cuda12]==0.5.3"
python - <<'PY'
import jax
print(jax.__version__)
print(jax.local_devices(backend="gpu"))
PY
```

### AlphaFold2: `There is no registered Platform called "CUDA"`

That comes from OpenMM relaxation, not from JAX. Older working EMBL envs exposed a CUDA-enabled OpenMM platform, while a fresh pip-installed OpenMM may expose only `Reference`, `CPU`, and `OpenCL`.

Current AlphaPulldown falls back to CPU relax automatically if CUDA is unavailable, so the AF2 cluster tests still pass. If you want GPU-backed OpenMM relax as well, install the OpenMM stack from conda before the pip install.

### AlphaFold3: `ModuleNotFoundError: No module named 'alphafold3.cpp'`

The root AlphaPulldown install succeeded, but the vendored AF3 package was not built yet:

```bash
cd /path/to/AlphaPulldown
python -m pip install ".[alphafold3,test]"
python -m pip install --no-deps -e ./alphafold3
build_data
```

### AlphaFold3 build error: `Could NOT find SQLite3`

If SQLite is installed in the conda environment but CMake still cannot find it, rerun the AF3 build with the conda prefix exposed:

```bash
mamba activate apd-af3
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export SQLite3_ROOT="$CONDA_PREFIX"
cd /path/to/AlphaPulldown
python -m pip install --no-deps -e ./alphafold3
build_data
```
