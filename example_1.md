# Example1
# Aim: Find proteins involving human translation pathway that might also interact with eIF4G3 or eIF4G2 
## 1st step: compute multiple sequence alignment (MSA) and template features (run on CPUs)

## Step 0 make sure you have HMMER and HH-suite downloaded and installed. 
As for our **HD cluster**, simply:
```bash
module load HMMER/3.1b2-foss-2016b
module load HH-suite/3.3.0-gompic-2020b
```
Firstly, download all 294 proteins that belong to human tranlsation pathway from Reactome: [link](https://reactome.org/PathwayBrowser/#/R-HSA-72766&DTAB=MT)

Then append the sequence of eIF4G3 (Uniprot:[O43432](https://www.uniprot.org/uniprot/O43432)) and eIF4G2 (Uniprot:[P78344](https://www.uniprot.org/uniprot/P78344)) to the sequence file.

For the purpose of this manual, the expected file is already provided here: [```./example_data/example_1_sequences.fasta```](./example_data/example_1_sequences.fasta). If you want to save time and run fewer jobs, you can use [```./example_data/example_1_sequences_shorter.fasta```](./example_data/example_1_sequences_shorter.fasta) instead.

If you installed via pip, now run:
```bash
create_individual_features.py \
  --fasta_paths=<your path to AlphaPulldown>/example_data/example_1_sequences.fasta \
  --data_dir=<path to alphafold databases> \
  --save_msa_files=False \
  --output_dir=<dir to save the output objects> \ 
  --use_precomputed_msas=False \
  --max_template_date=<any date you want> \
  --skip_existing=False \
  --seq_index=<any number you want or skip the flag to run all one after another>
```
If you use singularity, now run:
```bash
singularity exec --no-home --bind $PWD/example_data/example_1_sequences.fasta:/input_data/example_1_sequences.fasta \
--bind <path to alphafold databases>:/data_dir \
--bind <dir to save output objects>:/output_dir \
<path to your downloaded image>/alphapulldown.sif create_individual_features.py \ 
    --fasta_paths=/input_data/example_1_sequences.fasta \
    --data_dir=/data_dir \
    --output_dir=output_dir \
    --max_template_date=<any date you want> \
    --seq_index=<any number you want>
```

```create_individual_features.py``` will compute necessary features each protein in [```./example_data/example_1_sequences.fasta```](./example_data/example_1_sequences.fasta) and store them in the ```output_dir```. Please be aware that everything after ```>``` will be 
taken as the description of the protein and **please be aware** that any special symbol, such as ```| : ; #```, after ```>``` will be replaced with ```_```. 

The name of the pickles will be the same as the descriptions of the sequences  in fasta files (e.g. ">protein_A" in the fasta file will yield "protein_A.pkl")
 
 ------------------------

## 1.1 Explanation about the parameters
####  **```save_msa_files```** 
By default is **False** to save storage stage but can be changed into **True**. If it is set to be ```True```, the programme will 
create individual folder for each protein. The output directory will look like:
```
 output_dir
      |- protein_A.pkl
      |- protein_A
            |- uniref90_hits.sto
            |- pdb_hits.sto
            |- etc.
      |- protein_B.pkl
      |- protein_B
            |- uniref90_hits.sto
            |- pdb_hits.sto
            |- etc.
 ```
 
 
If ```save_msa_files=False``` then the ```output_dir``` will look like:
 ```
 output_dir
      |- protein_A.pkl
      |- protine_B.pkl
 ```
 
 --------------------
 
 
 ####  **```use_precomputed_msas```**
 Default value is ```False```. However, if you have already had msa files for your proteins, please set the parameter to be True and arrange your msa files in the format as below:
 ```
 example_directory
      |- protein_A 
            |- uniref90_hits.sto
            |- pdb_hits.sto
            |-***.a3m
            |- etc
      |- protein_B
            |- ***.sto
            |- etc
 ```
Then, in the command line, set the ```output_dir=/path/to/example_directory```

####  **```skip_existing```**
Default is ```False``` but if you have run the 1st step already for some proteins and now add new proteins to the list, you can change ```skip_existing``` to ```True``` in the
command line to avoid rerunning the same procedure for the previously calculated proteins.

####  **```seq_index```**
Default is `None` and the programme will run predictions one by one in the given files. However, you can set ```seq_index``` to 
different number if you wish to run an array of jobs in parallel then the programme will only run the corresponding job specified by the ```seq_index```. e.g. the programme only calculate features for the 1st protein in your fasta file if ```seq_index``` is set to be 1.

**NB**: ```seq_index``` starts from 1. 

---------------------

## 2nd step: Predict structures (run on GPU)

#### **Run in pulldown mode**
Inspired by pull-down assays, one can specify one or more proteins as "bait" and another list of proteins as "candidates". Then the programme will use AlphafoldMultimerV2 to predict interactions between baits (as in [example_data/baits.txt](./example_data/baits.txt)) and candidates (as in [example_data/candidates.txt](./example_data/candidates.txt)). 

**Note** If you want to save time and run fewer jobs, you can use [example_data/candidates_shoter.txt](./example_data/candidates_shorter.txt) instead of [example_data/candidates.txt](./example_data/candidates.txt) 

In this example, we selected pulldown mode and make eIF4G3(Uniprot:[O43432](https://www.uniprot.org/uniprot/O43432)) and eIF4G2(Uniprot:[P78344](https://www.uniprot.org/uniprot/P78344)) as baits while the other 294 proteins as candidates. Thus, in total, there will be 2 * 294 = 588 predictions. 

![demo1](./pulldown_mode_demo_1.png)



**NB** The command line interface for using pulldown mode will then become:

If you installed via pip:
```
run_multimer_jobs.py --mode=pulldown \
--num_cycle=3 --num_predictions_per_model=1 \
--output_path=<output directory> \ 
--data_dir=<path to alphafold databases> \ 
--protein_lists=$PWD/example_data/baits.txt,$PWD/example_data/candidates.txt \
--monomer_objects_dir=/path/to/monomer_objects_directory \
--job_index=<any number you want>
```

If you run via singularity:
```bash
singularity exec --no-home \ 
--bind $PWD/example_data/baits.txt:/input_data/baits.txt \
--bind $PWD/example_data/candidates.txt:/input_data/candidates.txt \
--bind <path to alphafold databases>:/data_dir \
--bind <dir to save predicted models>:/output_dir \ 
--bind <path to directory storing monomer objects >:/monomer_object_dir \
<path to your downloaded image>/alphapulldown.sif run_multimer_jobs.py --mode=pulldown \
--num_cycle=3 --num_predictions_per_model=1 \
--output_path=/output_dir \
--data_dir=/data_dir \ 
--protein_lists=/input_data/baits.txt,/input_data/candidates.txt \
--monomer_objects_dir=/monomer_object_dir
--job_index=<any number you want>
```

**Another explanation about the parameters**
####  **```monomer_objects_dir```**
It should be the same directory as ```output_dir``` specified in **Step 1**. It can be one directory or contain multiple directories if you stored pre-calculated objects in different locations. In the case of 
multiple ```monomer_objects_dir```, remember to put a `,` between each e.g. ``` --monomer_objects_dir=<dir_1>,<dir_2>```

####  **```job_index```**
Default is `None` and the programme will run predictions one by one in the given files. However, you can set ```job_index``` to 
different number if you wish to run an array of jobs in parallel then the programme will only run the corresponding job specified by the ```job_index```

**NB** ```job_index``` starts from 1

--------------------



## 3rd step Evalutaion and visualisation
We have also created an analysis pipeline and can be directly run by singularity. 

Firstly, download the singularity image from [here](https://oc.embl.de/index.php/s/cDYsOOdXA1YmInk).

Then execute the singularity image ( i.e. the sif file) by:
```
singularity exec --no-home --bind /path/to/your/output/dir:/mnt \
/path/to/your/sif/file/alpha-analysis.sif run_get_good_pae.sh --output_dir=/mnt --cutoff=5 --create_notebook=True
```

**About the parameters**

```/path/to/your/output/dir``` should be the direct result of the 2nd step as demonstrated above. 

```cutoff``` is to check the value of PAE between chains. In the case of multimers, the analysis programme will check whether any PAE values between two chains are smaller than the cutoff, as illustracted in the figure below:

```create_notebook``` is a boolean variable, for those predictions with good PAE scores between chains, would you like to create a jupyter notebook that shows the PAE, predicted models coloured by plDDT, and predicted models coloured by chains? A screen shot of an example notebook is shown below:

**About the outputs**
By default, you will have a csv file named ```predictions_with_good_interpae.csv``` created in the directory ```/path/to/your/output/dir``` as you have given in the command above. ```predictions_with_good_interpae.csv``` reports:1.iptm, iptm+ptm scores provided by AlphaFold 2. mpDockQ score developed by[ Bryant _et al._, 2022](https://gitlab.com/patrickbryant1/molpc)  3. PI_score developed by [Malhotra _et al._, 2021](https://gitlab.com/sm2185/ppi_scoring/-/wikis/home). The detailed explainations on these scores can be found in out paper.

If ```create_notebook=True```, then there will be a jupyter notebook named ```output.ipynb``` in the  ```/path/to/your/output/dir```. It is recommended uploading this jupyter notebook to google drive and viewing it via google's colabotary APP because the notebook can be very large and some features may not be properly installed in your local IDE.

------------------------------------------------------------
## Appendix: Instructions on running in all_vs_all mode
As the name suggest, all_vs_all means predict all possible combinations within a single input file. The input can be either full-length proteins or regions of a protein, as illustrated in the [example_all_vs_all_list.txt](./example_data/example_all_vs_all_list.txt) and the figure below:
![plot](./all_vs_all_demo.png)

If you installed via pip, run: 
```bash
run_multimer_jobs.py --mode=all_vs_all \
--num_cycle=3 --num_predictions_per_model=1 \
--output_path=<path to output directory> \ 
--data_dir=<path to AlphaFold data directory> \ 
--protein_lists=$PWD/example_data/example_all_vs_all_list.txt \
--monomer_objects_dir=/path/to/monomer_objects_directory \
--job_index=<any number you want>
```
If you run via singularity:
```bash
singulairty exec --no-home \
--bind <output directory>:/output_dir \
--bind <path to monomer_objects_directory>:/monomer_directory \
--bind $PWD/example_data/example_all_vs_all_list.txt:/input_data/example_all_vs_all_list.txt \
--bind <path to alphafold databases>:/data_dir \
<path to your downloaded image>/alphapulldown.sif run_multimer_jobs.py --mode=all_vs_all \
--num_cycle=3 --num_predictions_per_model=1 \
--output_path=/output_dir \ 
--data_dir=/data_dir \ 
--protein_lists=$PWD/example_data/example_all_vs_all_list.txt \
--monomer_objects_dir=/monomer_directory \
--job_index=<any number you want>
```
