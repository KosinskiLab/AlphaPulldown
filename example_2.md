# AlphaPulldown manual:
# Example2
# Aims: Model interactions between Lassa virus L protein and Z matrix protein; predict the structure of Z matrix protein homo 12-mer 
## 1st step: compute multiple sequence alignment (MSA) and template features (run on CPUs)

Firstly, download sequences of L(Uniprot: [O09705](https://www.uniprot.org/uniprotkb/O09705/entry)) and Z(uniprot:[O73557](https://www.uniprot.org/uniprotkb/O73557/entry)) proteins. The result is [```example_data/example_2_sequences.fasta```](./example_data/example_2_sequences.fasta)

Now run:
```bash
  create_individual_features.py \
    --fasta_paths=example_2_sequences.fasta \
    --data_dir=<path to alphafold databases> \
    --save_msa_files=False \
    --output_dir=<dir to save the output objects> \ 
    --use_precomputed_msas=False \
    --max_template_date=<any date you want> \
    --skip_existing=False --seq_index=<any number you want>
```

```create_individual_features.py``` will compute necessary features for O73557 and O09705 then store them as individual pickle files in the ```output_dir```. Please be aware that in the fasta files, everything after ```>``` will be 
taken as the description of the protein and  **please be aware** that any special symbol, such as ```| : ; #```, after ```>``` will be replaced with ```_```. 
 The name of the pickles will be the same as the descriptions of the sequences  in fasta files (e.g. ">protein_A" in the fasta file will yield "protein_A.pkl")
 
 ------------------------

## 1.1 Explanation about the parameters

See [Example 1](https://github.com/KosinskiLab/AlphaPulldown/blob/main/example_1.md#11-explanation-about-the-parameters)

## 2nd step: Predict structures (run on GPU)

#### **Task 1**
We want to predict the structure of full-length L protein together with Z protein. However, as the L protein is very long, many users would not have a GPU card with sufficient memory. Moreover, when attempting modeling the full L-Z, the resulting model does not match the known cryo-EM structure. In [Example 1](https://github.com/KosinskiLab/AlphaPulldown/blob/main/example_1.md), we showed how to use AlphaPulldown to find the interaction site by screening fragments using the ```pullldown``` mode. Here, to demonstrate the ```custom``` mode, we will assume the we know the interaction site and model the fragment using this mode, as demonstrated in the figure below ![custom_demo_2.png](./custom_demo_2.png):


Different proteins are seperated by ```;```. If a particular region is wanted from one protein, simply add ```,``` after that protein and followed by the region. Region comes in the format of ```number1-number2```. An example input file is: [```example_data/cutom_mode.txt```](./example_data/custom_mode.txt)

The command line interface for using custom mode will then become:

```
run_multimer_jobs.py \
  --mode=custom \
  --num_cycle=3 \
  --num_predictions_per_model=1 \
  --output_path=<path to output directory> \ 
  --data_dir=<path to AlphaFold data directory> \ 
  --protein_lists=custom_mode.txt \
  --monomer_objects_dir=/path/to/monomer_objects_directory \
  --job_index=<any number you want>
```

#### **Task 2**
This taks is to model the homo 12-mer of Z protein. Thus, homo-oligomer mode is needed. An oligomer state file will tell the programme the number of units. An example is: [```example_data/example_oligomer_state_file.txt```](./example_data/example_oligomer_state_file.txt)

In the file, oligomeric states of the corresponding proteins should be separated by ```,``` e.g. ```protein_A,3```means a homotrimer for protein_A  
![homo-oligomer_demo](./homooligomer_demo.png)

Instead of homo-oligomers, this mode can also be used to predict monomeric structure by simply adding ```1``` or nothing after the protein. 
The command for homo-oligomer mode is:

```
run_multimer_jobs.py --mode=homo-oligomer --output_path=<path to output directory> \ 
--oligomer_state_file=$PWD/example_data/example_oligomer_state_file.txt \ 
--monomer_objects_dir=<directory that stores monomer pickle files> \ 
--data_dir=/path-to-Alphafold-data-dir \ 
--job_index=<any number you want>
```


----------------------------------

## Explanation about the parameters

See [Example 1](https://github.com/KosinskiLab/AlphaPulldown/blob/main/example_1.md#explanation-about-the-parameters)

--------------------


## 3rd step Evalutaion and visualisation

See [Example 1](https://github.com/KosinskiLab/AlphaPulldown/blob/main/example_1.md#3rd-step-evalutaion-and-visualisation)


------------------------------------------------------------
## Appendix: Instructions on running in all_vs_all mode
As the name suggest, all_vs_all means predict all possible combinations within a single input file. The input can be either full-length proteins or regions of a protein, as illustrated in the [example_all_vs_all_list.txt](./example_data/example_all_vs_all_list.txt) and the figure below:
![plot](./all_vs_all_demo.png)

 The corresponding command is: 
```bash
run_multimer_jobs.py \
  --mode=all_vs_all \
  --num_cycle=3 \
  --num_predictions_per_model=1 \
  --output_path=<path to output directory> \ 
  --data_dir=/path-to-Alphafold-data-dir \ 
  --protein_lists=example_all_vs_all_list.txt \
  --monomer_objects_dir=/path/to/monomer_objects_directory \
  --job_index=<any number you want>
```
