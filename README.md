# AlphaPulldown v2.x

**[Documentation](https://github.com/KosinskiLab/AlphaPulldown/wiki)** | **[Features Database](https://github.com/KosinskiLab/AlphaPulldown/wiki/Features-Database)** | **[Downstream Analysis](https://github.com/KosinskiLab/AlphaPulldown/wiki/Downstream-Analysis)**

AlphaPulldownSnakemake provides a convenient way to run AlphaPulldown using a Snakemake pipeline. This lets you focus entirely on **what** you want to compute, rather than **how** to manage dependencies, versioning, and cluster execution.

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
  --tag 2.1.4
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
- **Interactive Jupyter notebook** with 3D visualizations and quality plots
- **Results table** with confidence scores and interaction metrics

Open the Jupyter notebook with:
```bash
jupyter-lab output/reports/output.ipynb
```

---

## Advanced Configuration

### Using Precomputed Features

If you have precomputed protein features, specify the directory:

```yaml
feature_directory:
  - "/path/to/directory/with/features/"
```

> **Note**: If your features are compressed, set `compress-features: True` in the config.

### Using CCP4 for Analysis

Download and install CCP4, then update your config:

```yaml
analysis_container: "/path/to/fold_analysis_2.1.2_withCCP4.sif"
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

`compress_result_pickles = {[False], True}` — Whether the result pickles are going to be gzipped.
`remove_result_pickles = {[False], True}` — Whether the result pickles are going to be removed.
`remove_keys_from_pickles = {[True], False}` — Whether to remove aligned_confidence_probs, distogram and masked_msa from pickles
`num_cycle = [3]` — Number of recycles
`num_predictions_per_model = [1]` — Number of predictions per model
`use_ap_style = {False], True}` — Change output directory to include a description of the fold as seen in previous alphapulldown versions.

</details>

<details>
<summary>AlphaFold3 Flags</summary>

</details>

### Database configuration

Set the path to your AlphaFold databases:

```yaml
databases_directory: "/path/to/alphafold/databases"
```

### Performance tuning

Adjust computational parameters:

```yaml
save_msa: False
use_precomputed_msa: False
predictions_per_model: 1
number_of_recycles: 3
```
