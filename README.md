# AlphaPulldown v2.x

**[Documentation](https://github.com/KosinskiLab/AlphaPulldown/wiki)** | **[Precalculated Input Database](https://github.com/KosinskiLab/AlphaPulldown/wiki/Features-Database)** | **[Downstream Analysis](https://github.com/KosinskiLab/AlphaPulldown/wiki/Downstream-Analysis)**

[AlphaPulldownSnakemake](https://github.com/KosinskiLab/AlphaPulldownSnakemake) is the recommended way to run AlphaPulldown: it wraps the pipeline in Snakemake so you can focus entirely on **what** you want to compute, rather than **how** to manage dependencies, versioning, and cluster execution. The instructions below cover that route; for running AlphaPulldown without Snakemake, see the [wiki](https://github.com/KosinskiLab/AlphaPulldown/wiki).

## 1. Installation

### Quick install (recommended)

```bash
curl -O https://raw.githubusercontent.com/KosinskiLab/AlphaPulldownSnakemake/2.9.0/install.sh
bash install.sh
conda activate snake
cd AlphaPulldownSnakemake
```

This single command creates the `snake` conda environment, deploys the workflow into
`./AlphaPulldownSnakemake`, and pre-fetches the container images into a **shared** image
directory (`~/.apptainer/snakemake-images` by default). Because the images live outside the
working directory, they are downloaded **once per machine** rather than once per project.

Useful options:

| Option | Meaning |
| --- | --- |
| `-d, --dest DIR` | working directory to deploy into (default `AlphaPulldownSnakemake`) |
| `-v, --version TAG` | workflow version to deploy (default `2.9.0`) |
| `-i, --image-dir DIR` | shared container image directory |
| `-n, --env-name NAME` | conda environment name (default `snake`) |
| `--no-pull` | skip container pre-fetch (Snakemake will fetch on first run) |

The script is idempotent: re-running it leaves an existing conda environment, working
directory and cached images untouched. Deployment copies the workflow's
[config/config.yaml](https://github.com/KosinskiLab/AlphaPulldownSnakemake/blob/main/config/config.yaml)
into your project directory; that copy is the file you edit below.

<details>
<summary>Manual installation</summary>

Create and activate the conda environment:

```bash
conda env create \
  -n snake \
  -f https://raw.githubusercontent.com/KosinskiLab/AlphaPulldownSnakemake/2.9.0/workflow/envs/alphapulldown.yaml
conda activate snake
```

This environment file installs Snakemake and all required plugins via conda and pulls in `alphapulldown-input-parser>=0.5.1` from PyPI in a single step.

Then deploy the workflow into a new processing directory for your project:

```bash
snakedeploy deploy-workflow \
  https://github.com/KosinskiLab/AlphaPulldownSnakemake \
  AlphaPulldownSnakemake \
  --tag 2.9.0
cd AlphaPulldownSnakemake
```

The shipped profiles set `apptainer-prefix: "$HOME/.apptainer/snakemake-images"`, so container
images are still shared across projects. See
[Container image cache](#container-image-cache) to change that location.

</details>

## 2. Configuration

`config/config.yaml` is organised in three sections:

| section | what it holds |
| --- | --- |
| **REQUIRED** | inputs, output directory, databases, weights, matching backend settings, SLURM partitions |
| **COMMON** | precomputed features, feature-only runs, analysis, duplicate complexes |
| **ADVANCED** | batching, resource limits, GPU selection, local MMseqs2, detailed report options |

The comments explain each group and point to the relevant section below. A first run
normally only needs REQUIRED. Desktop runs ignore the SLURM settings. Local MMseqs2
is off by default; its complete configuration is kept together under ADVANCED.

### Setup protein folding jobs

Create or edit the sample sheet `config/sample_sheet.csv` listing the proteins you want to fold. The simplest format uses one folding specification per line, for example UniProt IDs:

```
P01258+P01579
P01258
P01579
```

Each line represents one folding job:
- `P01258+P01579` - fold these two proteins together as a complex
- `P01258` - fold this protein as a monomer
- `P01579` - fold this protein as a monomer

<details>
<summary>Advanced protein specification options</summary>

You can also specify:
- **FASTA file paths** instead of UniProt IDs: `/path/to/protein.fasta`
- **Specific residue regions**: `Q8I2G6:1-100` (residues 1-100 only)
- **Discontinuous regions**: `Q8I2G6:1-100:150-200` (two separate regions from the same protein)
- **Multiple copies**: `Q8I2G6:2` (dimer of the same protein)
- **Combinations**: `Q8I2G6:2:1-100+Q8I5K4` (dimer of residues 1-100 plus another protein)
- **Copies plus discontinuous regions**: `Q8I2G6:2:1-100:150-200+Q8I5K4`

The same copy/range syntax also works when the workflow generates AlphaFold 3
JSON features (`--data_pipeline: alphafold3`). Examples:

- `Q8I2G6_af3_input.json:1-100`
- `Q8I2G6_af3_input.json:1-100:150-200`
- `Q8I2G6_af3_input.json:2:1-100:150-200+Q8I5K4_af3_input.json`

In that mode the Snakefile rewrites logical inputs such as
`Q8I2G6:1-100:150-200` to the corresponding
`Q8I2G6_af3_input.json:1-100:150-200` feature reference automatically.
AlphaPulldown preserves those discontinuous regions as one gapped polymer
chain with preserved residue-number gaps.
This keeps retained fragments intra-chain, so template contacts between those
fragments are not masked as inter-chain interactions.
The original residue IDs are written to the mmCIF author-numbering fields
(`auth_seq_id` and `pdbx_PDB_ins_code`); overlapping IDs are disambiguated with
insertion codes such as `2A`, `2B`, and so on.
Make sure the prediction container or runtime environment includes a matching
AlphaPulldown build together with `alphapulldown-input-parser>=0.5.1`.

</details>

### Configure input files

Edit `config/config.yaml` and set the path to your sample sheet:

```yaml
input_files:
  - "config/sample_sheet.csv"
```

### Setting up databases

If you do not already have the AlphaFold databases, `scripts/setup_databases.sh` from AlphaPulldown
fetches them and builds the MMseqs2 versions:

```bash
curl -O https://raw.githubusercontent.com/KosinskiLab/AlphaPulldown/main/scripts/setup_databases.sh
bash setup_databases.sh --dest /path/to/databases --alphafold3 --mmseqs
```

Pick what you need with `--alphafold2`, `--alphafold3` and `--mmseqs`. Existing
databases are skipped, so it is safe to re-run; `--dry-run` shows what would happen
and `--help` lists the sizes.

<details>
<summary>Options and requirements</summary>

No checkout is needed: the work runs inside the prediction container, which already
carries the AlphaFold downloaders and a pinned MMseqs2. `--alphafold2` is the
exception, because AlphaFold 2's downloader needs `aria2c` and `rsync`, which the
container does not ship; the script extracts it and runs it on the host, and tells you
if those are missing.

Add `--reduced` for AlphaFold 2's `reduced_dbs` mode. `--mmseqs` builds the GPU-padded
databases the local MMseqs2 feature stage needs and prints the config block to paste in.

</details>

### Database configuration

Set the paths to the AlphaFold databases and to the backend weights:

```yaml
databases_directory: "/path/to/alphafold/databases"
backend_weights_directory: "/path/to/backend/weights"
```

### Setup pulldown experiments

If you want to test which proteins from one group interact with proteins from another group, create a second file such as `config/baits.txt`:

```
Q8I2G6
```

And update your config:

```yaml
input_files:
  - "config/sample_sheet.csv"
  - "config/baits.txt"
```

This will test all combinations: every protein in `config/sample_sheet.csv` paired with every protein in `config/baits.txt`.

<details>
<summary>Multi-file pulldown experiments</summary>

You can extend this logic to create complex multi-partner interaction screens by adding more input files. For example, with three files:

```yaml
input_files:
  - "proteins_A.txt"  # 5 proteins
  - "proteins_B.txt"  # 3 proteins
  - "proteins_C.txt"  # 2 proteins
```

This will generate all possible combinations across the three groups, creating 5×3×2 = 30 different folding jobs. Each job will contain one protein from each file, allowing you to systematically explore higher-order protein complex formation.

**Note**: The number of combinations grows multiplicatively, so be mindful of computational costs with many files.

</details>

## 3. Execution

Run the pipeline locally:

```bash
snakemake --profile config/profiles/desktop --cores 8
```

<details>
<summary>Cluster execution</summary>

For running on a SLURM cluster, first create a virtual terminal e.g. using `screen`:

```bash
screen -S snakemake_session
```
Then activate your conda/mamba environment:
```bash
mamba activate snake
```
Finally, use the slurm executor plugin:
```bash
snakemake \
  --executor slurm \
  --profile config/profiles/slurm \
  --jobs 200 \
  --restart-times 5
```

Detach with `Ctrl + A` then `D`. Reattach later with `screen -r snakemake_session`.

Job specific logs are created automatically and stored in your `AlphaPulldownSnakemake/slurm_logs` directory.

</details>

## 4. Results

After completion, you'll find:
- **Predicted structures** in PDB/CIF format in the output directory
- **Per-fold interface scores** in `output/predictions/<fold>/interfaces.csv`
- **Aggregated interface summary** in `output/reports/all_interfaces.csv` when `generate_recursive_report: true`
- **Interactive APLit web viewer (recommended)** for browsing all jobs, PAE plots and AlphaJudge scores
- **Optional Jupyter notebook** with 3D visualizations and quality plots
- **Results table** with confidence scores and interaction metrics

## Recommended: explore results with APLit

[APLit](https://github.com/KosinskiLab/aplit)
 is a Streamlit-based UI for browsing AlphaPulldown runs (AF2 and AF3) and AlphaJudge metrics.

Install APLit (once):
```bash
pip install git+https://github.com/KosinskiLab/aplit.git
```

Then launch it from your project directory, pointing it to the predictions folder:
```bash
aplit --directory output/predictions
```

This starts a local web server (by default at `http://localhost:8501`) where you can:

- Filter and sort jobs by ipTM, PAE or AlphaJudge scores

- Inspect individual models in 3D (3Dmol.js)

- View PAE heatmaps and download structures / JSON files

On a cluster, run aplit on the login node and forward the port via SSH:
```bash
# on cluster
aplit --directory /path/to/project/output/predictions --no-browser
```
```bash
# on your laptop
ssh -N -L 8501:localhost:8501 user@cluster.example.org
```

Then open `http://localhost:8501` in your browser.


---

## Advanced Configuration

### Container image cache

Snakemake stores each container as `<md5-of-container-url>.simg` and skips the download when
that file already exists. By default it keeps them in `<workdir>/.snakemake/singularity`, which
means **every new project re-downloads the same multi-GB images**. Both shipped profiles
therefore set:

```yaml
apptainer-prefix: "$HOME/.apptainer/snakemake-images"
```

<details>
<summary>Moving the cache, and pinning an exact image</summary>

Environment variables and `~` are expanded, so this stays portable across machines. Point it
somewhere else (a group-shared directory, or scratch) by editing the profiles, by passing
`install.sh -i /path/to/images`, or per-run with `snakemake --apptainer-prefix /path/to/images`.

On a cluster the directory must be readable from the compute nodes. If you prefer not to edit
the profiles, exporting `APPTAINER_CACHEDIR` has the same effect, since Snakemake falls back to
it when no prefix is configured. Note that `SINGULARITY_CACHEDIR` does **not** work here: it
only caches the intermediate layers, so the image is still rebuilt for every project.

You can also bypass the registry entirely by building the images once and referencing the
files directly, which additionally pins the exact image you run:

```bash
apptainer build /path/to/images/alphafold3-2.5.0.sif docker://kosinskilab/alphafold3:2.5.0
```

```yaml
prediction_container: "/path/to/images/alphafold3-2.5.0.sif"
```

> **Note**: the cache key is the container URL, not the image digest, so a cached `:latest` is
> never refreshed. `prediction_container` is pinned to a version tag. `kosinskilab/alphajudge`
> publishes only `:latest`; pin it with a digest (`@sha256:<digest>`) if you need it fixed.

</details>

### GPU compatibility

The containers carry their own CUDA runtime (pip `nvidia-*` wheels), so GPU support depends on the
image tag, not on the driver installed on the node. Releases 2.5.0 and newer have been tested on
the following GPUs, with both AlphaFold 2 and AlphaFold 3:

- RTX 3090, 24 GB, sm_86
- A100, 40 GB, sm_80
- A40, 48 GB, sm_86
- L40S, 48 GB, sm_89
- H100, 80 GB, sm_90
- H200, 141 GB, sm_90
- B200, 180 GB, sm_100
- RTX PRO 4500 Blackwell, 16 GB MIG slices, sm_120
- RTX PRO 6000 Blackwell, 96 GB, sm_120

On other clusters the same rule applies by compute capability: sm_80 (Ampere) through sm_120
(Blackwell) all work with a 2.5.0 or newer image.

<details>
<summary>Why Blackwell (sm_120) needs a 2.5.0 or newer image</summary>

Pre-2.5.0 AlphaFold 3 images bundle jaxlib 0.4.34 on CUDA 12.6, whose `ptxas` cannot target sm_120.
They die at the first kernel compilation, before any inference runs:

```
ptxas does not support CC 12.0
XlaRuntimeError: UNIMPLEMENTED: ... ptxas too old
```

This cannot be patched from outside the container. jaxlib calls its own bundled `ptxas`, so
`XLA_FLAGS=--xla_gpu_cuda_data_dir` and `PATH` have no effect, and bind-mounting a newer `ptxas`
still leaves the CUDA runtime and cuDNN too old for the real kernels. From 2.5.0 the images ship a
consistent CUDA >= 12.8 stack (AF3: jax 0.9.1, ptxas 12.9, cuDNN 9.17, Tokamax; AF2: jax 0.5.3,
ptxas 12.9, cuDNN 9.2x) and return the same confidence scores as the older cards. All three AF3
attention implementations (`triton`/Tokamax, `cudnn`, `xla`) work, so no
`--flash_attention_implementation` override is needed.

While you are still on an older image, keep inference off those nodes with `slurm_exclude_nodes`.

</details>

<details>
<summary>MIG slices</summary>

Those nodes are RTX PRO 4500 cards split into 16 GB `1g.16gb` MIG instances. They need no special
`slurm_gres`: a plain `gpu:1` request lands on one slice and SLURM sets
`CUDA_VISIBLE_DEVICES=MIG-<uuid>`. Route work to them by size with a `min_vram_gb: 16` tier in
`structure_inference_gpu_tiers`. They suit monomers and small complexes, while larger jobs belong on
the 96 GB RTX PRO 6000 tier.

One MIG caveat the workflow already handles: `nvidia-smi --query-gpu=memory.total` reports the parent
card (32623 MiB) rather than the slice (~16 GB). Since `structure_inference_xla_mem_fraction: auto`
is `host RAM / GPU VRAM`, taking that number at face value would roughly halve the fraction and
switch off host spill exactly where it is most needed. The workflow therefore reads the slice profile
from `nvidia-smi -L` when `CUDA_VISIBLE_DEVICES` holds a MIG UUID, and falls back to `--query-gpu` on
whole cards.

</details>

### SLURM defaults for structure inference

Override default values to match your cluster:

```yaml
slurm_partition: "gpu"                      # partition(s) to submit inference to; one name,
                                            # "gpu-el8,gpu-training" or a YAML list for several
slurm_qos: "normal"                         # optional QoS if your site uses it
structure_inference_gpus_per_task: 1        # number of GPUs each inference job needs
structure_inference_gpu_model: ""           # "" lets SLURM pick any GPU in the partition; set a model to pin
structure_inference_tasks_per_gpu: 0        # <=0 keeps --ntasks-per-gpu unset in the plugin
slurm_exclude_nodes: ""                     # optional comma-separated nodes to avoid (sbatch --exclude)
structure_inference_max_runtime: 10080      # cap wall time (min) at the partition MaxTime
```

`structure_inference_gpus_per_task` and `structure_inference_gpu_model` are read by the
Snakemake Slurm executor plugin and translated into `--gpus=<model>:<count>` (or `--gpus=<count>` if
no model is specified). We no longer use `slurm_gres`; requesting GPUs exclusively through these
fields keeps the job submission consistent across clusters.

`structure_inference_tasks_per_gpu` toggles whether the plugin also emits `--ntasks-per-gpu`. Leaving
the default `0` prevents that flag, which avoids conflicting with the Tres-per-task request on many
systems. Set it to a positive integer only if your site explicitly requires `--ntasks-per-gpu`.

**Multiple partitions.** `slurm_partition` may name more than one partition — as a comma-separated
string (`"gpu-el8,gpu-training"`) or a YAML list:

```yaml
slurm_partition:          # inference runs on whichever of these frees up first
  - gpu-el8
  - gpu-training
```

The value is passed straight to `sbatch -p`, and SLURM starts each inference job on whichever listed
partition can run it soonest, so jobs aren't stuck behind one busy queue (e.g. they spill onto a
site's larger `gpu-training` cards when the default GPU partition is full). SLURM runs the job on the
first listed partition that fits its GPUs, `--mem` and walltime and skips the ones that don't (e.g. a
partition whose `MaxTime` is below `structure_inference_max_runtime`, or with no matching GPU) — so
make sure **at least one** listed partition can accommodate the job. Only `structure_inference` uses
this; the other (CPU) rules run on the cluster's default partition. A single name (the default) is
unchanged.

The remaining optional fields help with two common cluster issues: keeping inference off GPUs it
can't use, and large complexes running out of GPU memory. Defaults are sensible; expand below only if
you hit these.

<details>
<summary>Avoiding unsuitable GPUs (<code>slurm_exclude_nodes</code>, <code>gpu_model</code>) and the runtime cap</summary>

- **Restrict to one model** with `structure_inference_gpu_model` (e.g. `"A100"`) → the plugin emits
  `--gpus=<model>:<count>`. Accepts a single model name; leave `""` for any.
- **Route by complex size (VRAM)** with `structure_inference_gpu_tiers` → list your GPU pool as
  tiers of `{min_vram_gb, nodes}`. A complex's estimated peak VRAM (≈ `per_token_sq·N²`) selects the
  smallest tier that fits and all *smaller*-GPU nodes are excluded, so the job runs on **any** GPU at
  or above that tier — using the whole pool, not one pinned model. A complex larger than every tier
  uses the biggest tier and spills to host RAM via unified memory.

  ```yaml
  # Example GPU tiers; replace these node names with your cluster's:
  structure_inference_gpu_vram_headroom: 1.0   # <1.0 tolerates that fraction of host spill
  structure_inference_gpu_tiers:
    - {min_vram_gb: 16, nodes: "gpu-16gb-01,gpu-16gb-02"}  # RTX PRO 4500, 16GB MIG
    - {min_vram_gb: 24, nodes: "gpu-24gb-01,gpu-24gb-02"}
    - {min_vram_gb: 40, nodes: "gpu-40gb-01,gpu-40gb-02"}
    - {min_vram_gb: 48, nodes: "gpu-48gb-01,gpu-48gb-02"}
    - {min_vram_gb: 80, nodes: "gpu-80gb-01,gpu-80gb-02"}
    - {min_vram_gb: 96, nodes: "gpu-96gb-01,gpu-96gb-02"}  # RTX PRO 6000 Blackwell
  ```

  When set this drives `--exclude` per job and **overrides** `structure_inference_gpu_model` (the two
  would conflict). It's the practical "fit to GPU" lever: requested host RAM is a separate pool and
  does not size GPU VRAM, but excluding too-small GPUs by length does. Use explicit comma node lists
  (bracket ranges may be glob-expanded by the shell). VRAM-tier routing works *within* the listed
  partition(s); it excludes nodes by name, so if you span **multiple partitions** (see above) make
  sure the tier node lists cover every partition you submit to.
- **Exclude specific nodes** with `slurm_exclude_nodes`, passed verbatim to `sbatch --exclude`
  (e.g. `"gpu-96gb-01,gpu-96gb-02"`). `--exclude` is allowed in `slurm_extra` whereas
  `--constraint`/`--gres`/`--gpus` are not, so it is the supported way to drop a few nodes while
  keeping the rest of the partition. The usual reason to need it is a GPU the container image is too
  old for; see [GPU compatibility](#gpu-compatibility).
- **`structure_inference_max_runtime`** caps per-job wall time (minutes). Wall time scales as
  `1440 * attempt`, so without a cap enough retries exceed the partition `MaxTime` and SLURM rejects
  the job with `Requested time limit is invalid`. Set it to your partition's `MaxTime`
  (`scontrol show partition <name>`); default 7 days (10080).

</details>

<details>
<summary>Unified memory for large complexes (<code>structure_inference_unified_memory</code>)</summary>

Large AlphaFold 3 inputs (or smaller-VRAM GPUs) can fail with `RESOURCE_EXHAUSTED` /
`Allocator (GPU_0_bfc) ran out of memory`. Inference enables JAX/XLA **unified (managed) memory** by
default so the model spills from GPU VRAM into host RAM instead of OOM-ing (slower while spilling, but
it completes) — the
[DeepMind-recommended setting](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)
for large inputs. It is exported inside the prediction container as:

```sh
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab a huge VRAM chunk up front
export XLA_CLIENT_MEM_FRACTION=$FRACTION      # how far past physical VRAM XLA may allocate
export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is required: without it XLA reserves a large
slice of VRAM immediately, which defeats the point of letting the allocator grow into
host RAM on demand.

```yaml
structure_inference_unified_memory: true     # set false to fail fast on OOM instead
structure_inference_xla_mem_fraction: auto   # "auto", or pin a number like 3.2
```

With the default `structure_inference_xla_mem_fraction: auto`, the fraction is computed
**per job at run time** as `(allocated host RAM) / (physical GPU VRAM)`: the GPU VRAM is
read with `nvidia-smi` once the job lands on a node, and the host RAM is the job's SLURM
`--mem` allocation (which scales with retry attempts). This keeps the unified-memory
ceiling within the SLURM allocation so XLA cannot oversubscribe host RAM beyond what the
job requested — which would otherwise get the job OOM-killed. The chosen fraction is
logged as a `[unified-memory]` line at the top of the job log. Pin a number instead if
you want a fixed multiplier regardless of GPU/RAM.

> The fraction is computed in the job shell rather than via the SLURM executor: the
> executor passes the submit environment through with `--export=ALL` but offers no
> per-job env hook, and the value depends on which GPU the job lands on (only known at
> run time). Computing it in the container shell also avoids the apptainer env-crossing
> that submit-side env vars would need.

Because spilling is slower, make sure the job also requests enough host RAM
(`structure_inference_ram_bytes`, in MB) to hold the overflow — under `auto` that RAM is
exactly what the fraction is sized against.

</details>

<details>
<summary>Length-aware memory requests (sized automatically from the input sequences)</summary>

Host RAM for both compute stages is requested **from the input sequence length**, so big
complexes get enough memory on the first attempt instead of failing and climbing the retry
ladder, while small jobs are not over-provisioned. The request is computed at scheduling
time by reading the per-chain FASTA(s) the pipeline already stages under
`<output_directory>/data/`:

```
create_features      mem = safety * (feature_create_ram_bytes + per_residue * seq_len)
structure_inference  mem = safety * (structure_inference_ram_bytes + per_token_sq * N^2)
```

- `seq_len` is the query length; `N` is the **total residues of the complex** (the
  AlphaFold token count, summed over chains and copy numbers). For AlphaFold 3, `N` is
  rounded up to the `--buckets` size the model pads to. AlphaFold's pair
  representation is `O(N^2)`, hence the quadratic inference term.
- **The coefficients default by backend** (selected from `--data_pipeline` / `--fold_backend`).
  AlphaFold-Multimer (AF2) is heavier than AlphaFold 3 — measured AF2 inference host RSS was
  ~4× higher than AF3 at the same complex size, and AF2's feature stage runs HHblits (the
  main OOM source), whereas the AF3 pipeline is lighter. Defaults:

  | backend | feature base | feature /residue | inference base | inference /N² |
  |---|---|---|---|---|
  | `alphafold2` | 64000 MB | 40 MB | 16000 MB | 0.0055 |
  | `alphafold3` | 40000 MB | 25 MB |  8000 MB | 0.0045 |

  The AF3 inference quadratic is sized to the observed GPU-VRAM demand so that, with unified
  memory, the host spill ceiling (`host_mem / gpu_vram`) covers large complexes instead of
  OOM-ing.
- The first attempt already includes `mem_safety_factor` (default `1.25`) of head-room.
  **OOM retries still escalate** on top, multiplying by `..._ram_scaling ** (attempt - 1)`,
  so a bad estimate self-heals.
- Override any backend default by setting the matching key in `config/config.yaml`
  (`feature_create_ram_bytes`, `feature_create_ram_per_residue_mb`,
  `structure_inference_ram_bytes`, `structure_inference_ram_per_token_sq_mb`); an explicit
  value applies to all backends. Also tune `mem_safety_factor`, the `..._ram_scaling`
  factors, `structure_inference_runtime_minutes`, and `max_mem_mb` (set it to your largest
  node's RAM where an over-estimate would otherwise never schedule; `0` = no cap).
- The `..._ram_bytes` keys are the **fixed base** of each model rather than a flat request;
  raising a base only raises the floor. Setting `per_residue`/`per_token_sq` to `0`
  reproduces the old length-blind behaviour (a flat base × retry scaling).
- **Precomputed features:** when a chain is supplied via `feature_directory`, no
  `data/<chain>.fasta` is generated. Length is then recovered from the precomputed
  `<chain>_af3_input.json` (AF3) or from the parse-time length cache written by the length
  filter below (covers AF2 too). If neither is available the job falls back to the base
  allocation plus retry escalation. AF3 ligand atoms are not counted (no sequence), a small
  undercount absorbed by the safety margin.

</details>

<details>
<summary>Skipping over-large complexes (length filtering)</summary>

Folds that are too large to be worth submitting are **skipped before any job is created**,
so a single oversized complex (or one giant chain) doesn't waste a GPU/feature allocation
that will only OOM. Two configurable limits (in `config/config.yaml`):

```yaml
# Max TOTAL complex length (sum of all chains), per backend — selected by --fold_backend.
max_total_length_alphafold2: 5000     # AF2-Multimer
max_total_length_alphafold3: 7000     # AF3 handles larger inputs
# max_total_length: 6000              # optional single override for both backends
# Max length of any SINGLE protein; 0 = off (issue #33). A protein over this drops every
# fold containing it, so it is never even downloaded.
max_protein_length: 0
length_filter_fetch_uniprot: true     # set false for fully offline runs
```

- Lengths are resolved at **parse time** from, in order: a local FASTA, an
  already-downloaded `data/<id>.fasta`, the persistent cache
  `<output_directory>/.sequence_lengths.tsv`, and finally the UniProt REST API (cached for
  next time). Set a limit to `0` to disable it; if both are `0`, no resolution/fetching
  happens at all.
- Skipped folds are listed with reasons in `<output_directory>/skipped_folds.tsv` and logged
  as a `[length-filter]` warning. **Unknown lengths fail open** (the fold is kept), so a
  UniProt outage never silently drops work.
- First parse of a large all-UniProt sheet will fetch each unique length once (cached
  afterwards); already-downloaded inputs and local FASTAs are read without any network call.
- **Applies to every profile, including local/workstation runs** (it runs during workflow
  parsing, not in the executor). It's the only length-aware feature that does — the memory
  and GPU-routing settings are SLURM resources that local runs ignore. To attempt a complex
  larger than the caps on a big workstation, raise or zero the `max_total_length_*` values
  (and set `length_filter_fetch_uniprot: false` for offline use).

</details>

### Batching small jobs into one SLURM job

Many short, inference-only predictions can spend more time waiting in the SLURM queue
than running. To amortise that wait, several folds can share a single
`structure_inference` job: the job runs `run_structure_prediction.py` once per fold in a
loop, so the folds queue **once** between them instead of once each. With a current
AlphaPulldown container, batches of two or more instead use
`run_structure_prediction_batch.py`: one resident process loads the model once and keeps
the folds independent.

```yaml
batch_size: 4          # max folds per inference job (1 = one job per fold, the default)
batch_max_tokens: 0    # optional cap on summed residues per batch (0 = no cap)
```

<details>
<summary>What batching changes, and when not to use it</summary>

- Folds are grouped **by size**. Because folds execute sequentially, the workflow
  requests memory from the largest member's existing per-fold estimate, while
  walltime scales with the number of folds. `batch_max_tokens` keeps a batch's total
  work within the partition's `MaxTime`; a single oversized fold always runs alone.
  AlphaFold2 monomers and multimers are grouped separately because they use different
  model runners; AlphaFold3 retains size-only grouping.
- **AlphaFold2 compiles per input shape**, so a batch whose folds differ in length
  would recompile for each one and save nothing. For AF2 multimer batches the workflow
  therefore adds `--desired_num_res`, sized from the batch's largest fold, so every
  fold shares one shape and the batch compiles once. Padding applies to multimers
  only; a batch of AF2 monomers of differing lengths gains little.
- Works with both AlphaFold2 and AlphaFold3. A JSONL manifest distinguishes independent
  folds from the chains inside each fold, so AF3 does not merge separate folds. The
  backend and model runners are initialized once per batch. Containers predating the
  batch command automatically fall back to the per-fold loop.
- The two backends benefit differently: AlphaFold2 batches gain most when the folds
  share a shape, whereas for AlphaFold3 the point is queueing once instead of once
  per fold.
- For AlphaFold2 batches, `--allow_resume` is enabled automatically, so if a job is
  interrupted a rerun skips folds whose outputs already exist (AlphaFold3 does not accept
  that flag, so its batches recompute the unfinished folds on rerun).
- Analysis and reports are unaffected — `alphajudge` still runs per fold (one
  `interfaces.csv` + `report.pdf` each) and the recursive summary still aggregates them.
- **Trade-off:** a batch is one SLURM job, so a failure reruns the whole batch (minus the
  folds resume can skip), although the resident command attempts the remaining folds
  before returning a failure summary. A native CUDA/XLA abort, process termination,
  or a backend left unusable after an error cannot be isolated and may stop the rest
  of the batch. Keep `batch_size` modest and pair it with `batch_max_tokens` for
  heterogeneous fold sizes.
- Resident batch manifests and completion sentinels include a digest of the complete
  ordered membership. Changing a batch therefore schedules the new composition even
  when Snakemake uses `rerun-triggers: mtime`; single-fold paths remain unchanged.

> [!NOTE]
> **`--jax_compilation_cache_dir` and network filesystems.** XLA's autotune cache write
> can fail with `Device or resource busy` on some network filesystems (BeeGFS in
> particular), which aborts the process during compilation. A resident batch compiles
> once in memory and is not given the flag at all, so batches are unaffected. If you set
> it yourself, point it at node-local storage rather than `output_directory` when that
> lives on such a filesystem.

`batch_size: 1` (the default) is exactly the original one-job-per-fold behaviour.

</details>

### Using precomputed features

If you have precomputed protein features, specify the directory:

```yaml
feature_directory:
  - "/path/to/directory/with/features/"
```

> **Note**: If your features are compressed, set `compress-features: True` in the config.

### Feature generation flags (`create_individual_features.py`)

Tweak the feature-generation step by editing `create_feature_arguments` (or by running the script
manually).

<details>
<summary>Commonly used flags</summary>

- `--data_pipeline {alphafold2,alphafold3}` – choose the feature format to emit.
- `--db_preset {full_dbs,reduced_dbs}` – switch between the full BFD stack or the reduced databases.
- `--use_mmseqs2` – rely on the remote MMseqs2 API; skips local jackhmmer/HHsearch database lookups.
  To reuse a3m files generated locally with `colabfold_search`, also set `--use_precomputed_msas=True`
  (see the [mmseqs2 manual](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/mmseqs2_manual.md));
  otherwise the remote API is contacted again and your a3m files are overwritten.
- `--skip_msa` – generate query-only single-sequence features instead of running bulk MSA searches.
  Use those feature pickles with `run_structure_prediction.py --pair_msa=False`.
- `--use_precomputed_msas` / `--save_msa_files` – reuse stored MSAs (`<output_dir>/<protein>.a3m`) or
  keep new ones for later runs. Required to reuse precomputed MMseqs2/ColabFold a3m files rather
  than regenerating them.
- `--compress_features` – compress the generated features to save space: `*.pkl.xz` for the AlphaFold2 pipeline, `*_af3_input.json.xz` for AlphaFold3. Both are read back transparently, so compressed feature sets can be used directly (this is how the [features database](https://alphapulldown.s3.embl.de) ships them).
- `--skip_existing` – leave existing feature files untouched (safe for reruns).
- `--keep_msas` – refresh **templates only** in features that already exist in `--output_dir`, keeping their MSAs. Use it when the template database or `--max_template_date` has moved but the alignments are still valid: it costs a template search (minutes) instead of a full MSA run (hours). Works for both pipelines — AlphaFold2 features get their `template_*` block replaced, AlphaFold3 features are re-processed through AF3's "search for templates only" path. Proteins with no stored features are generated normally, and it takes precedence over `--skip_existing`. Cannot be combined with `--use_mmseqs2` (which fetches MSAs and templates together) or `--skip_msa` (no MSAs to keep).
- `--seq_index N` – only process the N‑th sequence from the FASTA list.
- `--use_hhsearch`, `--re_search_templates_mmseqs2` – toggle template search implementations.
- `--path_to_mmt`, `--description_file`, `--multiple_mmts` – enable TrueMultimer CSV-driven feature sets.
- `--max_template_date YYYY-MM-DD` – required cutoff for template structures; keeps runs reproducible.

</details>

### Batched local MMseqs2 features (AlphaFold 2 and 3)

<details>
<summary>Faster MSAs using local MMseqs2 instead of jackhmmer/HHblits</summary>

Off by default. Missing proteins are split into bounded shards searched with MMseqs2 —
as GPU jobs, or CPU jobs with `use_gpu: false` — and a separate CPU stage turns each
chain's alignment into standard features, so template work can use CPU and big-memory
partitions in parallel. Which features follows `--data_pipeline` in
`create_feature_arguments`: an AF3 JSON per chain, or an AF2 pickle. The remote
`--use_mmseqs2` path is unchanged. RNA chains are supported for AlphaFold 3 once the RNA
databases are configured.

```yaml
mmseqs2_features:
  enabled: true
  use_gpu: true
  temp_dir: /local-fast-scratch/mmseqs
  template_database_ids:
    pdb_seqres: pdb-seqres-2026-08
    mmcif: pdb-mmcif-2026-08
    # pdb70: pdb70-2026-08  # required for AF2 --use_hhsearch
  databases:
    uniref90:  {path: /db/mmseqs/uniref90_gpu,  identifier: uniref90-2026-08,  max_sequences: 10000}
    mgnify:    {path: /db/mmseqs/mgnify_gpu,    identifier: mgnify-2026-08,    max_sequences: 5000}
    small_bfd: {path: /db/mmseqs/small_bfd_gpu, identifier: small-bfd-2026-08, max_sequences: 5000}
    uniprot:   {path: /db/mmseqs/uniprot_gpu,   identifier: uniprot-2026-08,   max_sequences: 50000}
```

The protein databases must be padded (`makepaddedseqdb`); the RNA ones must not be.
`scripts/setup_databases.sh --mmseqs` builds them. The native AlphaFold 3 database tree
is still required — these are additive, not a replacement. Memory and walltime for both
stages are derived from the configured databases; the remaining knobs live in the
ADVANCED section of `config/config.yaml`.

**The GPU search reads every padded database in full**, about 375 GB for the four
protein ones. On network storage a cold first attempt is bound by that read, not by the
GPU: measured ~150 MB/s from NFS with the GPU idle, about an hour per shard, against
minutes once the node's page cache holds the databases. If first attempts time out,
raise `search_runtime_base_minutes` or stage the databases on local disk; the retry,
with twice the walltime, recovers either way.

**Depth is not identical to the native pipeline.** Measured on eight *B. subtilis*
proteins it was ~90% of jackhmmer's unpaired depth overall, but only 54–68% on the
shallowest families. Whether that costs accuracy is untested, so treat it as opt-in and
spot-check your own targets.

**For AlphaFold 2** set `--data_pipeline: alphafold2` and enable the block above. The
MSA recipe is AlphaFold 2's `reduced_dbs` set (no BFD/UniRef30 HHblits arm); templates
come from the AlphaFold 2 database tree, and `--use_hhsearch` and any explicit template
paths in `create_feature_arguments` reach the finalization stage. The MSA cache
remains reusable when only the template database changes. With
`--use_hhsearch: true`, set `mmseqs2_features.template_database_ids.pdb70` to the
immutable PDB70 build identity; `pdb_seqres` is then unused. Other template searches
require `pdb_seqres`, and every search requires `mmcif`. Update the relevant ID when
rebuilding a database, even at the same path; this invalidates finalized features.
The native MSA arguments do not reach finalization because this stage replaces
that search. Use a matching prediction image, such as
`docker://kosinskilab/alphafold2:2.9.0`, for AlphaFold 2. AlphaFold 2 finalization is
heavier than AlphaFold 3's because template featurization dominates it — median ~1 GB and 2 min, but up to
19 GB and 90 min, set by which structures the templates come from rather than by length
— so its defaults request 16 GB (times the safety factor) and 60 min. Against native
`reduced_dbs` features on 12 heterodimers released after AF2-multimer's training cutoff,
top-ranked DockQ averaged 0.56 against 0.59, with 9 of 12 interfaces acceptable either
way.

Alignments preserve insertions for both backends. Older MSA bundles that lost
insertions are regenerated automatically.

Databases, RNA, AlphaFold 2, tuning, caching and caveats:
[AlphaPulldown docs/mmseqs2_rna.md](https://github.com/KosinskiLab/AlphaPulldown/blob/main/docs/mmseqs2_rna.md).

</details>

### Structure analysis & reporting

Post-inference analysis is enabled by default. You can disable it or add a project-wide summary in `config/config.yaml`:

```yaml
enable_structure_analysis: true             # skip alphaJudge if set to false
generate_recursive_report: true             # disable if you do not need all_interfaces.csv
recursive_report_arguments:                 # optional extra CLI flags for alphajudge
  --models_to_analyse: best
```

### Changing folding backends

To use AlphaFold3 or other backends:

```yaml
structure_inference_arguments:
  --fold_backend: alphafold3
  --<other-flags>
```

> **Note**: AlphaPulldown supports: `alphafold2`, `alphafold3`, and `alphalink` backends.

### Backend-specific flags

You can pass backend CLI switches through `structure_inference_arguments`. Common options are listed below; keep or remove lines based on your needs.

> [!IMPORTANT]
> **These flags are backend-exclusive.** `run_structure_prediction.py` validates every flag
> against the selected `--fold_backend` and aborts the job with
> `ValueError: The following flags are not supported by backend '<name>'` if you pass one the
> backend does not accept. Only use flags from **your** backend's list below — e.g.
> `--allow_resume` is AlphaFold2-only. A single wrong flag fails the job immediately
> (before any prediction runs).
>
> When **batching** (`batch_size > 1`) the workflow adds what each backend needs —
> `--allow_resume` for AlphaFold2, and `--desired_num_res` for AlphaFold2 multimer
> batches — so you don't set them yourself.
>
> `--jax_compilation_cache_dir` is accepted by **both** backends: AlphaFold2 inference is
> JAX-compiled too, and a persistent cache removes most of the per-process compilation
> cost even at `batch_size: 1`. Older prediction images accept it for AlphaFold3 only, so
> the workflow does not add it for AlphaFold2 automatically — set it yourself once your
> image supports it, pointing at node-local storage.
>
> The authoritative, always-current list for your image is the backend validation inside the
> container. Print it with:
> ```bash
> singularity exec <prediction_container> run_structure_prediction.py --help
> ```
> (`alphalink` accepts the AlphaFold2 flags plus `--crosslinks`.)

<details>
<summary>AlphaFold2 flags</summary>

```yaml
structure_inference_arguments:
  --compress_result_pickles: False        # gzip AF2 result pickles
  --remove_result_pickles: False          # delete pickles after summary is created
  --models_to_relax: None                 # all | best | none
  --remove_keys_from_pickles: True        # strip large tensors from pickle outputs
  --convert_to_modelcif: True             # additionally write ModelCIF files
  --allow_resume: True                    # resume from partial runs (auto-added when batching)
  --relax_best_score_threshold: null      # only relax models above this score
  --threshold_clashes: null               # clash threshold for relaxation
  --hb_allowance: null                    # H-bond allowance for relaxation
  --plddt_threshold: null                 # pLDDT cutoff for relaxation
  --num_cycle: 3
  --num_predictions_per_model: 1
  --pair_msa: True
  --save_features_for_multimeric_object: False
  --skip_templates: False
  --msa_depth_scan: False
  --multimeric_template: False
  --model_names: None
  --msa_depth: None
  --description_file: None
  --path_to_mmt: None
  --desired_num_res: None          # pad every fold in a batch to this many residues
  --desired_num_msa: None          # optional; defaults to the fold's own MSA depth
  --jax_compilation_cache_dir: None
  --benchmark: False
  --model_preset: monomer
  --use_ap_style: False
  --use_gpu_relax: True
  --dropout: False
```
</details>

<details>
<summary>AlphaFold3 flags</summary>

```yaml
structure_inference_arguments:
  --jax_compilation_cache_dir: null       # AF3-only; auto-added when batching
  --buckets: ['64','128','256','512','768','1024','1280','1536','2048','2560','3072','3584','4096','4608','5120']
  --flash_attention_implementation: triton
  --num_diffusion_samples: 5
  --num_seeds: null
  --debug_templates: False
  --debug_msas: False
  --num_recycles: 10
  --save_embeddings: False
  --save_distogram: False
  --use_ap_style: False                   # shared with AlphaFold2
```
</details>

---

## How to cite

If AlphaPulldown (or this workflow) contributed to your research, please cite [Molodenskiy et al., 2025](https://doi.org/10.1093/bioinformatics/btaf115):

```bibtex
@article{Molodenskiy2025AlphaPulldown2,
  author    = {Molodenskiy, Dmitry and Maurer, Valentin J. and Yu, Dingquan and
               Chojnowski, Grzegorz and Bienert, Stefan and Tauriello, Gerardo and
               Gilep, Konstantin and Schwede, Torsten and Kosinski, Jan},
  title     = {AlphaPulldown2—a general pipeline for high-throughput structural modeling},
  journal   = {Bioinformatics},
  volume    = {41},
  number    = {3},
  pages     = {btaf115},
  year      = {2025},
  doi       = {10.1093/bioinformatics/btaf115}
}
```
