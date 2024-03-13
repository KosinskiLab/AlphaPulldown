# AlphaPulldown manual:Multimeric template modelling example


# Aims: Model activation of phosphoinositide 3-kinase by the influenza A virus NS1 protein (PDB: 3L4Q)
## 1st step: 3L4Q contains one bovine protein (Uniprot:P23726) and Influenza NS1 protein (Uniprot:P03496). You should create individual pickle using create_individual_features.py as normal. 
## NB: If you have obtained pickles for these 2 proteins, you can skip to the **2nd step**.

In this example we refer to the NS1 protein as chain A and to the bovine protein as chain C in multimeric template 3L4Q.cif.

The rest of commands are the same as in step 1 in [Example 1](https://github.com/KosinskiLab/AlphaPulldown/blob/backend_revision/manuals/example_1.md)

## 2nd step: Predict structures using multimeric templates

### **necessary input files and folders**
Required input files and folders are almost the same as default ```run_multimer_jobs.py``` e.g.  
A directory where feature pickles are stored:
```
monomeric_object_dir/
  |- P23726.pkl
  |- P03496.pkl
```
and a **```custom.txt```** in this case that looks like:
```
P03496;P23726
```
But now we need a directory where your customeric template is :
```
path_to_template/
  |- 3L4Q.cif
```

In addiction, a **```description.csv```** is required to inform which chain in the template is going to be used by which protein:
```
P03496,3L4Q.cif,A
P23726,3L4Q.cif,C
```

☑️ Now you are ready to run multimeric template modelling by :
```bash
python alphapulldown/alphapulldown/scripts/run_multimer_jobs_refactor.py \
    --mode=custom --output_path=/output/path \
    --data_dir=/scratch/AlphaFold_DBs/2.3.0 --multimeric_mode \
    --description_file= description.csv \
    --path_to_mmt= path_to_template\
    --monomer_objects_dir= monomeric_object_dir/ \
    --protein_lists= custom.txt --msa_depth=4
```
The majority of parameters are old familiar fiends except:
<ul>
  <li>
    --multimeric_mode: add this to your command line to turn on multimeric modelling. If you do not want multimeric modelling then remove it from the command line.
  </li>
  <li>
    --msa_depth: The user can also specify the depth of the MSA that is taken for modelling to increase the influence of the template on the predicted model
    Sometimes AlphaFold failed to follow the template if msa_depth is too high. Thus, in the example command shown above, 4 was used.
  </li>
</ul> 

## Extra features:
If you do not know the exact MSA depth, you can add ```--gradient_msa_depth``` to the command for exploring the desired MSA depth. This flag generates a set of logarithmically distributed points (denser at lower end) with the number of points equal to the number of predictions. The MSA depth (```num_msa```) starts from 16 and ends with the maximum value taken from the model config file. The ```extra_num_msa``` is always calculated as ```4*num_msa```.
The command line interface for using custom mode will then become:

```
:exclamation: To speed up computations, by default AlphaPulldown does not run relaxation (energy minimization) of models, which may decrease the quality of local geometry. If you want to enable it either only for the best models or for all predicted models, please add one of these flags to your command:
```
--models_to_relax=best
```
or
```
--models_to_relax=all
```

After the successful run one can evaluate and visualise the results in a usual manner (see e.g. [Example 2](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_2.md#2nd-step-predict-structures-run-on-gpu))
