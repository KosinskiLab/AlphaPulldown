# Backend Installation Guide

This guide is for direct AlphaPulldown use without Snakemake.

The Docker recipes in [`docker/alphafold2.dockerfile`](../docker/alphafold2.dockerfile) and [`docker/alphafold3.dockerfile`](../docker/alphafold3.dockerfile), plus the vendored upstream files in [`alphafold/docker/Dockerfile`](../alphafold/docker/Dockerfile) and [`alphafold3/docker/Dockerfile`](../alphafold3/docker/Dockerfile), are the reference environments behind the commands below.

## AlphaFold2 backend

Create a clean environment with the compiled tools first:

```bash
mamba create -y -n apd-af2 -c conda-forge -c bioconda \
  python=3.11 \
  kalign2 \
  hmmer \
  hhsuite \
  "openmm>=8.2" \
  "pdbfixer>=1.10" \
  "modelcif>=1.6" \
  "numpy<2"
mamba activate apd-af2
```

Install AlphaPulldown itself:

```bash
pip install -e ".[alphafold2]"
```

If you created the environment before the AF2 extra started pinning GPU-enabled JAX, refresh it explicitly:

```bash
pip install --upgrade --no-cache-dir "jax==0.5.3" "jax[cuda12]==0.5.3"
```

Verify that JAX can actually see the GPU before running the cluster suite:

```bash
python - <<'PY'
import jax
print(jax.__version__)
print(jax.local_devices(backend="gpu"))
PY
```

If you also want to run the repo test suites in that environment:

```bash
pip install -e ".[alphafold2,test]"
```

## AlphaFold3 backend

AlphaFold3 needs the vendored `alphafold3` package itself to be built, because the backend depends on the compiled `alphafold3.cpp` extension. A root-level `pip install -e .` is not enough on its own.

```bash
mamba create -y -n apd-af3 -c conda-forge -c bioconda \
  python=3.11 \
  hmmer \
  libcifpp \
  c-compiler \
  cxx-compiler \
  sqlite \
  zlib
mamba activate apd-af3
pip install -e ".[alphafold3,test]"
pip install -r alphafold3/dev-requirements.txt
pip install --no-deps -e ./alphafold3
build_data
```

If you also want the repo tests in that environment:

```bash
python - <<'PY'
import numpy
import alphafold3.cpp
from alphafold3.constants import chemical_components
print(numpy.__version__)
print(alphafold3.cpp.__file__)
print(chemical_components.Ccd.component_names()[:3])
PY
```

Notes:

- `alphafold3/dev-requirements.txt` currently pins the supported AF3 stack, including `numpy==2.1.3`.
- The root `.[alphafold3]` extra installs the Python-side AlphaPulldown dependencies, but the compiled `alphafold3.cpp` extension still comes from `pip install --no-deps -e ./alphafold3`.
- The vendored `alphafold3` package installs its own `build_data` entry point; use that rather than a root-level wrapper.
- You still need valid AlphaFold3 model parameters and databases from Google DeepMind for real AF3 runs.

## Troubleshooting

### AlphaFold2: `Unknown backend: 'gpu' requested, ... Platforms present are: cpu`

That means the environment has CPU-only JAX. Reinstall the AF2 JAX stack:

```bash
pip install --upgrade --no-cache-dir "jax==0.5.3" "jax[cuda12]==0.5.3"
python - <<'PY'
import jax
print(jax.__version__)
print(jax.local_devices(backend="gpu"))
PY
```

### AlphaFold3: `ModuleNotFoundError: No module named 'alphafold3.cpp'`

That means AlphaPulldown is installed, but the vendored `alphafold3` package was not built yet:

```bash
pip install -e ".[alphafold3,test]"
pip install -r alphafold3/dev-requirements.txt
pip install --no-deps -e ./alphafold3
build_data
```

### AlphaFold3 build error: `Could NOT find SQLite3`

Install SQLite into the environment, then rerun the editable AF3 build:

```bash
mamba install -y -n apd-af3 -c conda-forge sqlite
mamba activate apd-af3
pip install --no-deps -e ./alphafold3
build_data
```

## Cluster smoke tests

The cluster wrappers submit one Slurm job per pytest node, so they are the fastest way to validate many scenarios in parallel.

### AlphaFold2

Set the AlphaFold2 database root and enable GPU functional tests:

```bash
export ALPHAFOLD_DATA_DIR=/path/to/AlphaFold_DBs/2.3.0
export RUN_GPU_FUNCTIONAL_TESTS=1
```

Preview what would run:

```bash
python test/cluster/run_alphafold2_predictions.py --list
```

Submit a small smoke batch in parallel:

```bash
python test/cluster/run_alphafold2_predictions.py \
  --max-tests 6 \
  --use-temp-dir \
  --partition gpu-el8 \
  --qos normal \
  --constraint gaming
```

Run only a subset:

```bash
python test/cluster/run_alphafold2_predictions.py -k dimer --max-tests 4
```

### AlphaFold3

For the current AF3 cluster tests, `ALPHAFOLD_DATA_DIR` is used as the shared root passed both to feature creation and structure prediction. Point it at the AF3 bundle layout you use on the cluster.

```bash
export ALPHAFOLD_DATA_DIR=/path/to/alphafold3_bundle
export RUN_GPU_FUNCTIONAL_TESTS=1
```

Preview the collected AF3 nodes:

```bash
python test/cluster/run_alphafold3_predictions.py --list
```

Submit a small smoke batch in parallel:

```bash
python test/cluster/run_alphafold3_predictions.py \
  --max-tests 6 \
  --use-temp-dir \
  --partition gpu-el8 \
  --qos normal \
  --constraint gaming
```

Run only selected AF3 scenarios:

```bash
python test/cluster/run_alphafold3_predictions.py -k chopped --max-tests 4
```

Include the AF3 runtime benchmark test as well:

```bash
python test/cluster/run_alphafold3_predictions.py --include-perf --max-tests 2
```

## Optional checks

The end-to-end ModelCIF subprocess tests are still opt-in because they require the optional `ihm` / `modelcif` runtime:

```bash
pytest -o addopts='-ra --strict-markers -m external_tools' test/integration/test_modelcif.py
```
