# AlphaPulldown v2.x

**[Documentation](https://github.com/KosinskiLab/AlphaPulldown/wiki)** | **[Precalculated Input Database](https://github.com/KosinskiLab/AlphaPulldown/wiki/Features-Database)** | **[Downstream Analysis](https://github.com/KosinskiLab/AlphaPulldown/wiki/Downstream-Analysis)**

AlphaPulldownSnakemake provides a convenient way to run AlphaPulldown using a Snakemake pipeline. This lets you focus entirely on **what** you want to compute, rather than **how** to manage dependencies, versioning, and cluster execution. When you deploy the workflow with `snakedeploy`, the configuration file will be copied after deployment to your local project directory as [config/config.yaml](https://github.com/KosinskiLab/AlphaPulldownSnakemake/blob/main/config/config.yaml).

For running without Snakemake, see this [link](https://github.com/KosinskiLab/AlphaPulldown/wiki).

## 1. Installation

Install required dependencies:

```bash
mamba create -n snake -c conda-forge -c bioconda python=3.12 \
  snakemake snakemake-executor-plugin-slurm snakedeploy pulp click coincbc
mamba activate snake
```

That's it, you're done!

## 2. Configuration

### Create a Working Directory

Create a new processing directory for your project:

```bash
snakedeploy deploy-workflow \
  https://github.com/KosinskiLab/AlphaPulldownSnakemake \
  AlphaPulldownSnakemake \
  --tag 2.1.6
cd AlphaPulldownSnakemake
```

### Setup Protein Folding Jobs

Create a sample sheet `folds.txt` listing the proteins you want to fold. The simplest format uses UniProt IDs:

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
- **Multiple copies**: `Q8I2G6:2` (dimer of the same protein)
- **Combinations**: `Q8I2G6:2:1-100+Q8I5K4` (dimer of residues 1-100 plus another protein)

</details>

### Configure Input Files

Edit `config/config.yaml` and set the path to your sample sheet:

```yaml
input_files:
  - "folds.txt"
```

### Setup Pulldown Experiments

If you want to test which proteins from one group interact with proteins from another group, create a second file `baits.txt`:

```
Q8I2G6
```

And update your config:

```yaml
input_files:
  - "folds.txt"
  - "baits.txt"
```

This will test all combinations: every protein in `folds.txt` paired with every protein in `baits.txt`.

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

For running on a SLURM cluster, use the executor plugin:

```bash
screen -S snakemake_session
snakemake \
  --executor slurm \
  --profile config/profiles/slurm \
  --jobs 200 \
  --restart-times 5
```

Detach with `Ctrl + A` then `D`. Reattach later with `screen -r snakemake_session`.

</details>

## 4. Results

After completion, you'll find:
- **Predicted structures** in PDB/CIF format in the output directory
- **Per-fold interface scores** in `output/predictions/<fold>/interfaces.csv`
- **Aggregated interface summary** in `output/reports/all_interfaces.csv` when `generate_recursive_report: true`
- **Interactive APLit web viewer (recommended)** for browsing all jobs, PAE plots and AlphaJudge scores
- **Optional Jupyter notebook** with 3D visualizations and quality plots
- **Results table** with confidence scores and interaction metrics

# Recommended: explore results with APLit

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

### SLURM defaults for structure inference
Override default values to match your cluster:

```yaml
slurm_partition: "gpu"                      # which partition/queue to submit to
slurm_qos: "normal"                         # optional QoS if your site uses it
structure_inference_gpus_per_task: 1        # number of GPUs each inference job needs
structure_inference_gpu_model: "3090"       # optional GPU model constraint (remove to allow any)
structure_inference_tasks_per_gpu: 0        # <=0 keeps --ntasks-per-gpu unset in the plugin
```

`structure_inference_gpus_per_task` and `structure_inference_gpu_model` are read by the
Snakemake Slurm executor plugin and translated into `--gpus=<model>:<count>` (or `--gpus=<count>` if
no model is specified). We no longer use `slurm_gres`; requesting GPUs exclusively through these
fields keeps the job submission consistent across clusters.

`structure_inference_tasks_per_gpu` toggles whether the plugin also emits `--ntasks-per-gpu`. Leaving
the default `0` prevents that flag, which avoids conflicting with the Tres-per-task request on many
systems. Set it to a positive integer only if your site explicitly requires `--ntasks-per-gpu`.

### Using Precomputed Features

If you have precomputed protein features, specify the directory:

```yaml
feature_directory:
  - "/path/to/directory/with/features/"
```

> **Note**: If your features are compressed, set `compress-features: True` in the config.

### Feature generation flags (`create_individual_features.py`)

You can tweak the feature-generation step by editing `create_feature_arguments` (or by running the
script manually). Commonly used flags:

- `--data_pipeline {alphafold2,alphafold3}` – choose the feature format to emit.
- `--db_preset {full_dbs,reduced_dbs}` – switch between the full BFD stack or the reduced databases.
- `--use_mmseqs2` – rely on the remote MMseqs2 API; skips local jackhmmer/HHsearch database lookups.
- `--use_precomputed_msas` / `--save_msa_files` – reuse stored MSAs or keep new ones for later runs.
- `--compress_features` – zip the generated `*.pkl` files (`.xz` extension) to save space.
- `--skip_existing` – leave existing feature files untouched (safe for reruns).
- `--seq_index N` – only process the N‑th sequence from the FASTA list.
- `--use_hhsearch`, `--re_search_templates_mmseqs2` – toggle template search implementations.
- `--path_to_mmt`, `--description_file`, `--multiple_mmts` – enable TrueMultimer CSV-driven feature sets.
- `--max_template_date YYYY-MM-DD` – required cutoff for template structures; keeps runs reproducible.


### Structure analysis & reporting

Post-inference analysis is enabled by default. You can disable it or add a project-wide summary in `config/config.yaml`:

```yaml
enable_structure_analysis: true             # skip alphaJudge if set to false
generate_recursive_report: true             # disable if you do not need all_interfaces.csv
recursive_report_arguments:                 # optional extra CLI flags for alphajudge
  --models_to_analyse: best
```


### Changing Folding Backends

To use AlphaFold3 or other backends:

```yaml
structure_inference_arguments:
  --fold_backend: alphafold3
  --<other-flags>
```

> **Note**: AlphaPulldown supports: `alphafold`, `alphafold3`, `alphalink`, and `unifold` backends.

### Backend Specific Flags

<details>
<summary>AlphaFold2 Flags</summary>

```yaml
# Whether the result pickles are going to be zipped (.xz).
  --compress_result_pickles: False

# Whether the result pickles are going to be removed.
  --remove_result_pickles: False

# The models to run the final relaxation step on. If `all`, all models are relaxed, which may be time consuming. If `best`, only the most confident model is relaxed. If `none`, relaxation is not run. Turning off relaxation might result in predictions with distracting stereochemical violations but might help in case you are having issues with the relaxation stage.
  --models_to_relax: None

# Whether to remove aligned_confidence_probs, distogram and masked_msa from pickles.
  --remove_keys_from_pickles: True

# Whether to convert predicted pdb files to modelcif format.
  --convert_to_modelcif: True

# Whether to allow resuming predictions from previous runs or start anew.
  --allow_resume: True

# Number of recycles
  --num_cycle: 3

# Number of predictions per model
  --num_predictions_per_model: 1

# Whether to pair the MSAs when constructing multimer objects.
  --pair_msa: True

# Whether to save features for multimeric object.
  --save_features_for_multimeric_object: False

# Do not use template features when modelling.
  --skip_templates: False

# Run predictions for each model with logarithmically distributed MSA depth.
  --msa_depth_scan: False

# Whether to use multimeric templates.
  --multimeric_template: False

# A list of names of models to use, e.g. model_2_multimer_v3 (default: all models).
  --model_names: None

# Number of sequences to use from the MSA (by default is taken from AF model config).
  --msa_depth: None

# Path to the text file with multimeric template instruction.
  --description_file: None

# Path to directory with multimeric template mmCIF files.
  --path_to_mmt: None

# A desired number of residues to pad.
  --desired_num_res: None

# A desired number of msa to pad.
  --desired_num_msa: None

# Run multiple JAX model evaluations to obtain a timing that excludes the compilation time, which should be more indicative of the time required for inferencing many proteins.
  --benchmark: False

# Choose preset model configuration - the monomer model, the monomer model with extra ensembling, monomer model with pTM head, or multimer model.
  --model_preset: monomer

# Change output directory to include a description of the fold as seen in previous alphapulldown versions
  --use_ap_style: False

# Whether to run Amber relaxation on GPU.
  --use_gpu_relax: True

# Whether to use dropout when inferring for more diverse predictions.
  --dropout: False

```
</details>

<details>
<summary>AlphaFold3 Flags</summary>

```yaml

# Path to a directory for the JAX compilation cache.
  --jax_compilation_cache_dir: None

# Strictly increasing order of token sizes for which to cache compilations. For any input with more tokens than the largest bucket size, a new bucket is created for exactly that number of tokens.
  --buckets: ['64', '128', '256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072', '3584', '4096', '4608', '5120']

# Flash attention implementation to use. 'triton' and 'cudnn' uses a Triton and cuDNN flash attention implementation, respectively. The Triton kernel is fastest and has been tested more thoroughly. The Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an XLA attention implementation (no flash attention) and is portable across GPU devices.
  --flash_attention_implementation: triton

# Number of diffusion samples to generate.
  --num_diffusion_samples: 5 

# Number of seeds to use for inference. If set, only a single seed must be provided in the input JSON. AlphaFold 3 will then generate random seeds in sequence, starting from the single seed specified in the input JSON. The full input JSON produced by AlphaFold 3 will include the generated random seeds. If not set, AlphaFold 3 will use the seeds as provided in the input JSON.
  --num_seeds: None

# If set, save generated template mmCIFs to templates_debug/ during AF3 input prep.
  --debug_templates: False

# If set, dump featurised MSA arrays and final complex A3M before inference.
  --debug_msas: False

# Number of recycles to use during AF3 inference.
  --num_recycles: 10

# Whether to save final trunk single/pair embeddings in AF3 output.
  --save_embeddings: False

# Whether to save final distogram in AF3 output.
  --save_distogram: False
```

</details>

### Database configuration

Set the path to your AlphaFold databases:

```yaml
databases_directory: "/path/to/alphafold/databases"
```

***

## How to Cite

If AlphaPulldown contributed significantly to your research, please cite [the corresponding publication](https://doi.org/10.1093/bioinformatics/btaf115) in *Bioinformatics*:

```bibtex
@article{Molodenskiy2025AlphaPulldown2,
  author    = {Molodenskiy, Dmitry and Maurer, Valentin J. and Yu, Dingquan and Chojnowski, Grzegorz and Bienert, Stefan and Tauriello, Gerardo and Gilep, Konstantin and Schwede, Torsten and Kosinski, Jan},
  title     = {AlphaPulldown2—a general pipeline for high-throughput structural modeling},
  journal   = {Bioinformatics},
  volume    = {41},
  number    = {3},
  pages     = {btaf115},
  year      = {2025},
  doi       = {10.1093/bioinformatics/btaf115}
}
```
