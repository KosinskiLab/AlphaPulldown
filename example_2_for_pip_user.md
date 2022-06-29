# AlphaPulldown manual (installed via pip):

## Step 0:
Remember to activate the conda env created while installing the package:
```bash
source activate <env_name>
```

Also make sure you have [HMMER](http://hmmer.org/documentation.html) and [HH-suite](https://github.com/soedinglab/hh-suite) **installed** and **compiled!**

As for our **HD cluster**, simply:
```bash
module load HMMER/3.3.2-gompic-2020b
module load HH-suite/3.3.0-gompic-2020b
```

# Example2
# Aims: Model interactions between Lassa virus L protein and Z matrix protein; predict the structure of Z matrix protein homo 12-mer 
## 1st step: compute multiple sequence alignment (MSA) and template features (run on CPUs)

Firstly, download sequences of L(Uniprot: [O09705](https://www.uniprot.org/uniprotkb/O09705/entry)) and Z(uniprot:[O73557](https://www.uniprot.org/uniprotkb/O73557/entry)) proteins. Since there are only 2 sequences, you don't have to concatenate them into one single file (but you still can if you want). 

Run the command:
```bash
  create_individual_features.py\
    --fasta_paths=$PWD/example_data/O09705.fasta,$PWD/example_data/O73557.fasta\
    --data_dir=<path to alphafold databases>\
    --save_msa_files=False\
    --output_dir=<dir to save the output objects>\ 
    --use_precomputed_msas=False\
    --max_template_date=<any date you want>\
    --skip_existing=False --seq_index=<any number you want>
```
```create_individual_features.py``` will compute necessary features for O73557 and O09705 then store them in the ```output_dir```. Please be aware that in the fasta files, everything after ```>``` will be 
taken as the description of the protein and make sure do **NOT** include any special symbol, such as ```|```, after ```>```. However, ```-``` or ```_```is allowed. 
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

## 2nd step: Predict structures (run on GPU)

#### **Run in custom mode**
We want to predict the structure of full-length L protein together with Z protein but could not finish the prediction with our computing resources. Thus 
we predicted the interaction between a fragment of L protein and Z protein instead, as domonstrated in the figure below ![custom_demo_2.png](./custom_demo_2.png):


Different proteins are seperated by ```;```. If a particular region is wanted from one protein, simply add ```,``` after that protein and followed by the region. Region comes in the format of ```number1-number2```

The command line interface for using pulldown mode will then become:
```
run_multimer_jobs.py --mode=pulldown\
--num_cycle=3 --num_predictions_per_model=1\
--output_path=/path/to/your/directory\ 
--data_dir=/path-to-Alphafold-data-dir\ 
--protein_lists=$PWD/example_data/baits.txt,$PWD/example_data/candidates.txt\
--monomer_objects_dir=/path/to/monomer_objects_directory
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

