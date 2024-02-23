# Convert models from PDB format to ModelCIF format

With PDB files now being marked as legacy format, here is a way to convert PDB files produced by the [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown) pipeline into [mmCIF](https://mmcif.wwpdb.org) files including the [ModelCIF](https://mmcif.wwpdb.org/dictionaries/mmcif_ma.dic/Index/) extension.

On top of the general mmCIF tables, ModelCIF adds information, relevant for a modeling experiment. There is a bit of target-sequence(s) annotation and a modeling protocol. This describes the process how a model was created including software used with its parameters. To help the user to decide the reliability of a model, various quality metrics can be stored directly in a ModelCIF file, or in associated files registered in the main file. ModelCIF is also the preferred format for [ModelArchive](https://www.modelarchive.org).
As AlphaPulldown is relying on [AlphaFold](https://github.com/google-deepmind/alphafold) to produce model coordinates, there may be multiple models predicted in a single experiment. Respecting that not always all the models need to be converted to ModelCIF, `convert_to_modelcif.py` offers three major modes:

* Convert all models into ModelCIF in separated files

* Only convert a specific single model

* Convert a specific model to ModelCIF but keep additional models in a Zip archive associated with the representative ModelCIF formatted model

## Usage

To run `convert_to_modelcif.py`, the Python module [modelcif](https://pypi.org/project/modelcif/) is needed in addition to the regular AlphaPulldown Python environments. It is recommended to install it with conda command:
`conda install -c conda-forge modelcif`

The script itself lives in the [`alphapulldown/`](https://github.com/KosinskiLab/AlphaPulldown/tree/main/alphapulldown) subdirectory of the AlphaPulldown repository.

### scenario 1: convert all models to separated ModelCIF files

The most general call of the conversion script, without any non-mandatory arguments, will create a ModelCIF file and an associated Zip archive for each model of each complex found in `--ap_output` directory:

```bash
$ source activate AlphaPulldown
$ convert_to_modelcif.py \
  --ap_output <output path of run_multimer_jobs.py> \
  --monomer_objects_dir <output directory of feature creation>
```

Where `--ap_output` needs the output path from the modeling step. It actually takes the same argument as the `--output_path` to the `run_multimer_jobs.py` script. `--monomer_objects_dir` can also be fed from `run_multimer_jobs.py`, it takes the same argument as the `--monomer_objects_dir` parameter there. The creation of the monomer objects directories happens with scripts `create_individual_features.py` and `create_individual_features_with_templates.py`.

The output is stored in the path that `--ap_output` points to. After a run of `convert_to_modelcif.py`, you should find a ModelCIF file and a Zip archive for each model PDB file in the AlphaPulldown output directory:

```
ap_output
    protein1_and_protein2
        |-ranked_0.cif
        |-ranked_0.pdb
        |-ranked_0.zip
        |-ranked_1.cif
        |-ranked_1.pdb
        |-ranked_1.zip
        |-ranked_2.cif
        |-ranked_2.pdb
        |-ranked_2.zip
        |-ranked_3.cif
        |-ranked_3.pdb
        |-ranked_3.zip
        |-ranked_4.cif
        |-ranked_4.pdb
        |-ranked_4.zip
        ...
    ...
```

### scenario 2: only convert a specific single model for each complex

If only a single model should be translated to ModelCIF, the option `--model_selected` is used. As a value, provide the ranking of the model. For example, the following command converts the model from rank 0:

```bash
$ source activate AlphaPulldown
$ convert_to_modelcif.py \
  --ap_output <output path of run_multimer_jobs.py> \
  --monomer_objects_dir <output directory of feature creation> \
  --model_selected 0
```

That will create only one ModelCIF file and Zip archive in the path pointed at by `--ap_output`:

```
ap_output
    protein1_and_protein2
        |-ranked_0.cif
        |-ranked_0.pdb
        |-ranked_0.zip
        |-ranked_1.pdb
        |-ranked_2.pdb
        |-ranked_3.pdb
        |-ranked_4.pdb
        ...
    ...
```

Beside `--model_selected`, the arguments are the same as for [scenario 1](#scenario-1-convert-all-models-to-separated-modelcif-files).


### scenario 3: have a representative model and keep associated models

Sometimes you want to focus on a certain model from the AlphaPulldown pipeline but don't want to completely discard the other models generated. For this, `convert_to_modelcif.py` can translate all models to ModelCIF, but store the excess in the Zip archive of the selected model. This is achieved by adding the option `--add_associated` together with `--model_selected` to the call:

```bash
$ source activate AlphaPulldown
$ convert_to_modelcif.py \
  --ap_output <output path of run_multimer_jobs.py> \
  --monomer_objects_dir <output directory of feature creation> \
  --model_selected 0 \
  --add-associated
```

Arguments are the same as in [scenario 1](#scenario-1-convert-all-models-to-separated-modelcif-files) and [scenario 2](#scenario-2-only-convert-a-specific-single-model) but for `--add-associated`.

The output directory looks like when only converting a single model:

```
ap_output
    protein1_and_protein2
        |-ranked_0.cif
        |-ranked_0.pdb
        |-ranked_0.zip
        |-ranked_1.pdb
        |-ranked_2.pdb
        |-ranked_3.pdb
        |-ranked_4.pdb
        ...
    ...
```

But a peek into `ranked_0.zip` shows that it stored ModelCIF files and Zip archives for all remaining models of this modeling experiment:

```
ranked_0.zip
    |-ranked_0_local_pairwise_qa.cif
    |-ranked_1.cif
    |-ranked_1.zip
    |-ranked_2.cif
    |-ranked_2.zip
    |-ranked_3.cif
    |-ranked_3.zip
    |-ranked_4.cif
    |-ranked_4.zip
```

### associated Zip archives

`convert_to_modelcif.py` produces two kinds of output, ModelCIF files and Zip archives for each model. The latter are called "associated files/ archives" in ModelCIF lingo. Associated files are registered in their corresponding ModelCIF file by categories [`ma_entry_associated_files`](https://mmcif.wwpdb.org/dictionaries/mmcif_ma.dic/Categories/ma_entry_associated_files.html) and [`ma_associated_archive_file_details`](https://mmcif.wwpdb.org/dictionaries/mmcif_ma.dic/Categories/ma_associated_archive_file_details.html). Historically, this scheme was created to offload AlphaFolds' pairwise alignment error lists, which drastically increase file size. Nowadays, the Zip archives are used for all kind of supplementary information on models, not handled by ModelCIF.


### misc. options

At this time, there is only one option left unexplained: `--compress`. It tells the script to compress ModelCIF files using Gzip. In case of `--add-associated`, the ModelCIF files in the associated Zip archive are also compressed.


<!--  LocalWords:  modelcif py
 -->
