# AlphaPulldown

## Installation: via pip or singularity 

### Option1 (recommended): pip

**Firstly**, create a conda environment and gather necessary dependencies 
```bash
conda create -n <env_name> -c omnia python==3.7 openmm pdbfixer
````
**Secondly**, activate the environment and install AlphaPulldown
```bash
source activate <env_name>
pip install alphapulldown
```
### Option2 : singularity
Simply download AlphaPulldown singularity image from [here](https://oc.embl.de/index.php/s/KR8d4m8ASN9p3gs)

------

## Manual
AlphaPulldown supports 4 different modes of massive predictions: pulldown, all_vs_all, homo-oliogmer, and custom.

We have provided 2 examples:

Example 1 is a case where pulldown and custom modes are used. Manuals: [example_1_for_pip_user](./example_1_for_pip_user.md) or [example_1_for_singularity_user](./singularity_user_example_1.md)

Example 2 is a case where custom and homo-oligomer modes were used. Manuels: [example_2_for_pip_user](./example_2_for_pip_user.md) or [example_2_for_singularity_user](./singularity_user_example_2.md)

all_vs_all mode can be viewed as a special case of pulldown mode thus all manuals comes with an illustration of all_vs_all mode. 

### **all_vs_all mode**
As the name suggest, all_vs_all means predict all possible combinations within a single input file. The input can be either full-length proteins or regions of a protein, as illustrated in the [example_all_vs_all_list.txt](./example_data/example_all_vs_all_list.txt) and the figure below:
![plot](./all_vs_all_demo.png)
 
```bash
run_multimer_jobs.py --mode=all_vs_all\
--num_cycle=3 --num_predictions_per_model=1\
--output_path=/path/to/your/directory\ 
--data_dir=/path-to-Alphafold-data-dir\ 
--protein_lists=$PWD/example_data/example_all_vs_all_list.txt\
--monomer_objects_dir=/path/to/monomer_objects_directory
--job_index=<any number you want>
```


---------------------------------------------

#### **2.3 homo-oligomer mode**
![plot](./homooligomer_demo.png)

The programme also can fold homo-oligomers. Simply create a file indicated the oligomeric state of the corresponding protein, separated by ```,``` e.g. ```protein_A,3```means a homotrimer for protein_A. An example can be found in [example_oligomer_state_file.txt](./example_data/example_oligomer_state_file.txt). 

**NB**: ```homo-oligomer``` mode can also predict monomers. Simply type nothing after the protein, such as : "protiein_A", or specify oligomer state to be 1, such as "protein_A,1", and the programme will predict a monomeric structure of protein_A. **However**, AlphaPulldown used hmmsearch instead of hhsearch when searching for structure templates in 1st step. As a result, predicted monomeric structure could differ from the prediction from default Alphafold monomer.  

Take the sturcture of```test/``` directory as an example, the command line input would be:
```
run_multimer_jobs.py --mode=homo-oligomer --output_path=/path/to/your/directory\ 
--oligomer_state_file=$PWD/example_data/oligomer_state_file.txt\ 
--monomer_objects_dir=/path/to/monomer_objects_directory\ 
--data_dir=/path-to-Alphafold-data-dir\ 
--job_index=<any number you want>
```
---------------------------------------

#### **2.4 custom mode**
The user can also just provide one single file in which each line contains a protein combination of desire. Different proteins are separated by ```;``` and if you'd like specific region of a particular protein, you can denote like ```protein_A,number1-number2```, as illustrated in the ```test/test_data/test_custom.txt``` and also in the picture below:
![plot](./custom_mode_demo.png)

Take the structure of ```test/``` directory as an example, the command line input would be:

```
run_multimer_jobs.py --mode=custom --output_path=/path/to/your/directory\ 
--protein_lists=$PWD/example_data/custom_mode_list.txt\ 
--monomer_objects_dir=/path/to/monomer_objects_directory\ 
--data_dir=/path-to-Alphafold-data-dir\ 
--job_index=<any number you want>
```

----------------------------------
