# Backend Installation Guide

This guide is for direct AlphaPulldown use without Snakemake.

Two points matter in practice:

1. Run the install commands from the AlphaPulldown repo root.
2. For AlphaFold3, building the vendored `alphafold3` package is a separate step. A root-level install alone is not enough.

The Docker files remain the long-term reference environments, but the commands below are the simpler cluster-facing paths.

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

## Cluster smoke tests

The wrappers submit one Slurm job per pytest node, so they are the fastest way to validate many scenarios in parallel.

On EMBL, the AF2 and AF3 functional tests already have default database roots baked into the test files, so the only environment variable you need for the standard cluster smoke tests is:

```bash
export RUN_GPU_FUNCTIONAL_TESTS=1
```

Set `ALPHAFOLD_DATA_DIR` only if your databases are not in the EMBL default locations.

### AlphaFold2

Preview the collected nodes:

```bash
python test/cluster/run_alphafold2_predictions.py --list
```

Known-good EMBL monomer smoke test:

```bash
python test/cluster/run_alphafold2_predictions.py \
  --partition gpu-training \
  --constraint hgx \
  --extra-sbatch-arg=--nodelist=hgx5 \
  --time 01:00:00 \
  --cpus-per-task 4 \
  --mem 16G \
  --use-temp-dir \
  test/cluster/check_alphafold2_predictions.py::TestRunModes::test__monomer
```

Default queue-based batch example:

```bash
python test/cluster/run_alphafold2_predictions.py \
  --max-tests 6 \
  --use-temp-dir \
  --partition gpu-el8 \
  --qos normal \
  --constraint gaming
```

### AlphaFold3

Preview the collected nodes:

```bash
python test/cluster/run_alphafold3_predictions.py --list
```

Known-good EMBL monomer smoke test:

```bash
python test/cluster/run_alphafold3_predictions.py \
  --partition gpu-training \
  --constraint hgx \
  --extra-sbatch-arg=--nodelist=hgx5 \
  --time 01:00:00 \
  --cpus-per-task 4 \
  --mem 16G \
  --use-temp-dir \
  test/cluster/check_alphafold3_predictions.py::TestAlphaFold3RunModes::test__monomer
```

Default queue-based batch example:

```bash
python test/cluster/run_alphafold3_predictions.py \
  --max-tests 6 \
  --use-temp-dir \
  --partition gpu-el8 \
  --qos normal \
  --constraint gaming
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
