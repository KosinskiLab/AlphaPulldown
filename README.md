# AlphaPulldown v2.x

**[Documentation](https://github.com/KosinskiLab/AlphaPulldown/wiki)** | **[Precalculated Input Database](https://github.com/KosinskiLab/AlphaPulldown/wiki/Features-Database)** | **[Downstream Analysis](https://github.com/KosinskiLab/AlphaPulldown/wiki/Downstream-Analysis)**

[AlphaPulldownSnakemake](https://github.com/KosinskiLab/AlphaPulldownSnakemake) is the recommended way to run AlphaPulldown: it wraps the pipeline in Snakemake so you can focus entirely on **what** you want to compute, rather than **how** to manage dependencies, versioning, and cluster execution. The instructions below cover that route; for running AlphaPulldown without Snakemake, see the [wiki](https://github.com/KosinskiLab/AlphaPulldown/wiki).

## 1. Installation

### Quick install (recommended)

```bash
curl -O https://raw.githubusercontent.com/KosinskiLab/AlphaPulldownSnakemake/2.7.1/install.sh
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
| `-v, --version TAG` | workflow version to deploy (default `2.7.1`) |
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
  -f https://raw.githubusercontent.com/KosinskiLab/AlphaPulldownSnakemake/2.7.1/workflow/envs/alphapulldown.yaml
conda activate snake
```

This environment file installs Snakemake and all required plugins via conda and pulls in `alphapulldown-input-parser>=0.5.1` from PyPI in a single step.

Then deploy the workflow into a new processing directory for your project:

```bash
snakedeploy deploy-workflow \
  https://github.com/KosinskiLab/AlphaPulldownSnakemake \
  AlphaPulldownSnakemake \
  --tag 2.7.1
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
| **REQUIRED** | inputs, output directory, databases, weights, prediction container |
| **COMMON** | features, backend flags, analysis, batching, SLURM partition |
| **ADVANCED** | memory sizing, length filtering, GPU routing, spilling, CPU partitions |

Each key carries a one-line comment naming the section below that documents it in full.
A first run normally only needs the REQUIRED section.

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

The same copy/range syntax also works with AlphaFold 3 JSON features
(`--data_pipeline: alphafold3`). Examples:

- `Q8I2G6_af3_input.json:1-100`
- `Q8I2G6_af3_input.json:1-100:150-200`
- `Q8I2G6_af3_input.json:2:1-100:150-200+Q8I5K4_af3_input.json`

When a workflow or wrapper maps a logical token such as `Q8I2G6:1-100:150-200`
to `Q8I2G6_af3_input.json:1-100:150-200`, AlphaPulldown preserves the region
selection and keeps the AF3 JSON feature input as one discontinuous polymer
chain with preserved residue-number gaps, so chopped regions stay intra-chain
and template contacts between retained fragments are not masked as inter-chain
interactions. The original residue IDs are written to the mmCIF author-numbering
fields (`auth_seq_id` and `pdbx_PDB_ins_code`); overlapping IDs are disambiguated
with insertion codes such as `2A`, `2B`, and so on.
This syntax is parsed by the shared `alphapulldown-input-parser` package used by
both AlphaPulldown and AlphaPulldownSnakemake; make sure the execution
environment carries `alphapulldown-input-parser>=0.5.1`.

</details>

### Configure input files

Edit `config/config.yaml` and set the path to your sample sheet:

```yaml
input_files:
  - "config/sample_sheet.csv"
```

### Setting up databases

If you do not already have the AlphaFold databases, `scripts/setup_databases.sh`
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
image tag, not on the driver installed on the node. Releases 2.5.0 and newer run on every GPU in the
EMBL cluster, with both AlphaFold 2 and AlphaFold 3:

- RTX 3090, 24 GB, sm_86 (`gpu21-22`, `gpu29-37`)
- A100, 40 GB, sm_80 (`gpu25-28`)
- A40, 48 GB, sm_86 (`sb03-05` to `sb03-20`)
- L40S, 48 GB, sm_89 (`gpu40-48`)
- H100, 80 GB, sm_90 (`gpu38-39`, and `hgx2-3` in `gpu-training`)
- H200, 141 GB, sm_90 (`hgx4-5` in `gpu-training`)
- B200, 180 GB, sm_100 (`bgx1` in `gpu-training`)
- RTX PRO 4500 Blackwell, 16 GB MIG slices, sm_120 (`gpu60-68`)
- RTX PRO 6000 Blackwell, 96 GB, sm_120 (`gpu51-53`)

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
<summary>MIG slices (<code>gpu60-68</code>)</summary>

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
  # Example for the EMBL GPU pool; replace nodes with your cluster's (nothing is hard-coded):
  structure_inference_gpu_vram_headroom: 1.0   # <1.0 tolerates that fraction of host spill
  structure_inference_gpu_tiers:
    - {min_vram_gb: 16, nodes: "gpu60,gpu61,gpu62,gpu63,gpu64,gpu65,gpu66,gpu67,gpu68"}  # RTX PRO 4500, 16GB MIG
    - {min_vram_gb: 24, nodes: "gpu21,gpu22,gpu29,gpu30,gpu31,gpu32,gpu33,gpu34,gpu35,gpu36,gpu37"}
    - {min_vram_gb: 40, nodes: "gpu25,gpu26,gpu27,gpu28"}
    - {min_vram_gb: 48, nodes: "gpu40,gpu41,gpu42,gpu43,gpu44,gpu45,gpu46,gpu47,gpu48"}
    - {min_vram_gb: 80, nodes: "gpu38,gpu39"}
    - {min_vram_gb: 96, nodes: "gpu51,gpu52,gpu53"}  # RTX PRO 6000 Blackwell
  ```

  When set this drives `--exclude` per job and **overrides** `structure_inference_gpu_model` (the two
  would conflict). It's the practical "fit to GPU" lever: requested host RAM is a separate pool and
  does not size GPU VRAM, but excluding too-small GPUs by length does. Use explicit comma node lists
  (bracket ranges may be glob-expanded by the shell). VRAM-tier routing works *within* the listed
  partition(s); it excludes nodes by name, so if you span **multiple partitions** (see above) make
  sure the tier node lists cover every partition you submit to.
- **Exclude specific nodes** with `slurm_exclude_nodes`, passed verbatim to `sbatch --exclude`
  (e.g. `"gpu51,gpu52"`). `--exclude` is allowed in `slurm_extra` whereas
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
you want a fixed multiplier regardless of GPU/RAM (mirrors the EMBL `run_AF_multimer.sh`
convention).

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
  AlphaFold token count, summed over chains and copy numbers). AlphaFold's pair
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

### Batched local MMseqs2-GPU features (AlphaFold 3)

Set `mmseqs2_features.enabled: true` to split missing proteins into bounded
GPU MSA shards. Each shard performs exactly one AlphaPulldown MSA batch and then
releases its GPU. Independent CPU jobs run native AlphaFold 3 template search and
finalize one standard AF3 JSON per protein, so template work can use the CPU and
big-memory partitions in parallel. Existing AF2 feature generation and the remote
`--use_mmseqs2` path are unchanged.

The AlphaFold 3 prediction image bundles the verified MMseqs2-GPU `18-8cc5c`
release at `/opt/mmseqs/bin/mmseqs`, which is the default `binary_path`; it is a
standalone executable rather than a Python/PyPI dependency. The AlphaFold 2
image carries the identical pinned binary so both maintained prediction images
have one reproducible runtime toolchain, as explicitly supported by the project,
although the workflow adapter remains AF3-only until AF2 feature-pickle and
multimer-pairing semantics have a separate interface. The adapter accepts only
the bundled path; arbitrary host executables are not visible inside the image.

```yaml
mmseqs2_features:
  enabled: true
  # binary_path: /opt/mmseqs/bin/mmseqs  # only supported path
  binary_id: 8cc5ce367b5638c4306c2d7cfc652dd099a4643f
  temp_dir: /local-fast-scratch/mmseqs
  batch_max_sequences: 256
  batch_max_residues: 100000
  e_value: 0.0001
  # Size one GPU shard from database footprint plus a modest query term.
  gpu_database_ram_mb: 64000
  gpu_chunk_ram_per_residue_mb: 0.02
  gpu_ram_scaling: 1.1
  gpu_runtime_base_minutes: 15
  gpu_runtime_per_sequence_minutes: 0.5
  gpu_runtime_per_1000_residues: 1.0
  template_database_ids:
    pdb_seqres: pdb-seqres-2026-08
    mmcif: pdb-mmcif-2026-08
  databases:
    uniref90: {path: /db/mmseqs/uniref90, identifier: uniref90-2026-08, max_sequences: 10000}
    mgnify: {path: /db/mmseqs/mgnify, identifier: mgnify-2026-08, max_sequences: 5000}
    small_bfd: {path: /db/mmseqs/small_bfd, identifier: small-bfd-2026-08, max_sequences: 5000}
    uniprot: {path: /db/mmseqs/uniprot, identifier: uniprot-2026-08, max_sequences: 50000}
```

#### How the MSAs compare to the native pipeline

MMseqs2 has been used to build AlphaFold MSAs for years (ColabFold does exactly
this), so this is an established approach rather than a new one. It is not, however,
the *same* search as the jackhmmer pipeline AlphaFold 3 ships with, and the
difference is worth seeing before you switch.

Measured on eight *B. subtilis* proteins, searching the **same four databases** as the
native pipeline, counting unique sequences:

| protein | unpaired recall | paired recall |
| --- | --- | --- |
| P0CI78 | 99.2% | 98.8% |
| O32142 | 98.5% | 99.6% |
| O30472 | 99.5% | 101.3% |
| P80870 | 86.0% | 103.2% |
| O31537 | 83.8% | 81.7% |
| O31843 | 82.8% | 87.1% |
| O07542 | 68.1% | 78.3% |
| O31580 | 53.7% | 61.5% |
| **overall** | **90.2%** | **98.0%** |

Template counts were identical (32 vs 32). Paired MSAs, which drive species pairing
for complexes, are essentially equivalent. Unpaired recall is close to complete on
well-populated families and falls off on shallow ones, which is the expected shape for
a single-pass search against jackhmmer's iterative profile search.

What this does *not* tell you is whether that costs prediction accuracy; that needs
matched inference and DockQ against experimental structures. Treat the table as a
reason to spot-check your own targets, not as a verdict either way.

MMseqs2 GPU search always runs at its maximum sensitivity, so there is no
`sensitivity` setting. Each configured path must name a padded target database;
prepare all four from existing MMseqs2 databases as follows:

```bash
mmseqs makepaddedseqdb /source/uniref90  /db/mmseqs/uniref90
mmseqs makepaddedseqdb /source/mgnify    /db/mmseqs/mgnify
mmseqs makepaddedseqdb /source/small_bfd /db/mmseqs/small_bfd
mmseqs makepaddedseqdb /source/uniprot   /db/mmseqs/uniprot
```

Keep the source and destination prefixes different. A padded database consists
of several files sharing that prefix; allocate storage for all of them and set
`gpu_database_ram_mb` from the largest database footprint plus site-specific
overhead. Ampere or newer GPUs give full performance; Turing is supported at
reduced speed. A database larger than VRAM can stream from host RAM, but requires
enough node RAM and is slower. Put database prefixes and `temp_dir` on fast local
storage where possible.

For repeated searches on a dedicated GPU node, MMseqs2 recommends an index made
with `createindex --index-subset 2` and a same-node `gpuserver`, followed by
searches using `--gpu-server 1 --db-load-mode 2`. The server and client must use
the same GPU visibility, prefilter mode, and `--max-seqs`. This distributed Slurm
adapter does not start a persistent server because separate shards can land on
different nodes; each shard therefore loads its databases once. Pinning shards
to a resident service is an advanced site-specific optimization.

The default `binary_id` is the exact MMseqs commit bundled by the current images;
update it when deliberately changing that binary. Database identifiers, hit
limits, E-value, container identity, template cutoff,
and explicit PDB-seqres/mmCIF identifiers namespace the caches. Changing
scientific provenance schedules fresh outputs even with `rerun-triggers: mtime`.
Partial per-protein MSA bundles are deliberately not Snakemake outputs: a failed
shard loses only its completion summary, and a retry validates and reuses finished
bundles. Completion summaries record each expected bundle's byte size,
nanosecond mtime, and SHA-256. DAG construction uses the stat fields as its fast
path and streams the digest only when metadata changed. If a bundle is missing,
corrupt, or replaced after completion, a fresh repair summary reruns only that
shard; intact bundles are reused. The CPU finalizer also validates bundle
semantics and discards only an invalid bundle so the following Snakemake run can
repair it automatically. AlphaFold 3 finalization retains its native merged
unpaired-MSA template search. The complete native AF3 database tree under `databases_directory`
(including PDB seqres, mmCIF, RNA and other configured databases) is still
required; the four MMseqs2 databases are additive, not a replacement.

Container binds are merged with existing `APPTAINER_BINDPATH` and
`SINGULARITY_BINDPATH` values rather than replacing them. This applies to all
workflow modes, including AF2; MMseqs database and scratch directories add exact
binds while existing user binds are preserved.


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

---

## How to Cite

If AlphaPulldown contributed significantly to your research, please cite [the corresponding publication](https://doi.org/10.1093/bioinformatics/btaf115) in *Bioinformatics*:

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
