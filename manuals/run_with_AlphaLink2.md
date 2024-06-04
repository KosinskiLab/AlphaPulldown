# Instruction of running structural predictions with cross-link data via [AlphaLink2](https://github.com/Rappsilber-Laboratory/AlphaLink2/tree/main)
## Introduction
As [Stahl et al., 2023](https://www.nature.com/articles/s41587-023-01704-z) showed, integrating cross-link data with AlphaFold could improve the modelling quality in 
some challenging cases. Thus AlphaPulldown has integrated [AlphaLink2](https://github.com/Rappsilber-Laboratory/AlphaLink2/tree/main) pipeline 
and allows the user to combine cross-link data with AlphaFold Multimer inference, without the need of calculating MSAs from the scratch again.

In addition, this integration retains all the other benefits from AlphaPulldown, such as the interface for fragmenting protein into regions; automatically 
generating PAE plots after the predictions etc.

## 1st step: configure the Conda environment

After you initialise the same conda environment, where you normally run AlphaPulldown, firstly, you need to compile [UniCore](https://github.com/dptech-corp/Uni-Core).

```bash
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install setuptools==69.5.1 # Downgrade setuptools to avoid crashes when installing unicore
git clone https://github.com/dptech-corp/Uni-Core.git
pip install setuptools==69.5.1 # avoid crashes brought by setuptools v70.x
python setup.py install --disable-cuda-ext

# test whether unicore is successfully installed
python -c "import unicore"
```

You may see the following warning but it's fine: 

```
fused_multi_tensor is not installed corrected
fused_rounding is not installed corrected
fused_layer_norm is not installed corrected
fused_softmax is not installed corrected
```
## 2nd step: download AlphaLink2 checkpoint 

Now please download the PyTorch checkpoints from [Zenodo](https://zenodo.org/records/8007238), unzip it, then you should obtain a file named: ```AlphaLink-Multimer_SDA_v3.pt```

## 3rd step: prepare cross-link input data

As instructed by [AlphaLink2](https://github.com/Rappsilber-Laboratory/AlphaLink2/tree/main), information of cross-linked residues
between 2 proteins, inter-protein crosslinks A->B 1,50 and 30,80 and an FDR=20%, should look like: 

```
{'protein_A': {'protein_B': [(1, 50, 0.2), (30, 80, 0.2)]}}
```

and intra-protein crosslinks follow the same format: 

```
{'protein_A': {'protein_A': [(5, 20, 0.2)]}}
```

The keys in these dictionaries should be the same as your pickle files created by [the first stage of AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown/blob/main/example_1.md). e.g. you should have ```protein_A.pkl``` 
and ```protein_B.pkl``` already calculated. 

Dictionaries like these should be stored in **```.pkl.gz```** files and provided to AlphaPulldown in the next step. You can use the script from [AlphaLink2](https://github.com/Rappsilber-Laboratory/AlphaLink2/tree/main)
to prepare these pickle files. 

### ⚠️ The dictionaries are 0-indexed, i.e., residues start from 0.

## 4th step: run with AlphaLink2 prediction via AlphaPulldown
Within the same conda environment, run in e.g. ```custom``` mode:

```bash
run_multimer_jobs.py --mode=custom \
--num_predictions_per_model=1 \
--output_path=/scratch/scratch/user/output/models \
--data_dir=/g/alphafold/AlphaFold_DBs/2.3.0/ \
--protein_lists=custom.txt \
--monomer_objects_dir=/scratch/user/output/features \
--job_index=$SLURM_ARRAY_TASK_ID --alphalink_weight=/scratch/user/alphalink_weights/AlphaLink-Multimer_SDA_v3.pt \
--use_alphalink=True --crosslinks=/path/to/crosslinks.pkl.gz 
```

The other modes provided by AlphaPulldown also work in the same way.

## 5th Step: citation

If you used this feature, please cite AlphaLink2 as well: 

```
@article {Stahl2023,
	author = {Kolja Stahl and Oliver Brock and Juri Rappsilber},
	title = {Modelling protein complexes with crosslinking mass spectrometry and deep learning},
	elocation-id = {2023.06.07.544059},
	year = {2023},
	doi = {10.1101/2023.06.07.544059},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Scarcity of structural and evolutionary information on protein complexes poses a challenge to deep learning-based structure modelling. We integrated experimental distance restraints obtained by crosslinking mass spectrometry (MS) into AlphaFold-Multimer, by extending AlphaLink to protein complexes. Integrating crosslinking MS data substantially improves modelling performance on challenging targets, by helping to identify interfaces, focusing sampling, and improving model selection. This extends to single crosslinks from whole-cell crosslinking MS, suggesting the possibility of whole-cell structural investigations driven by experimental data.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/06/09/2023.06.07.544059},
	eprint = {https://www.biorxiv.org/content/early/2023/06/09/2023.06.07.544059.full.pdf},
	journal = {bioRxiv}
}
```




