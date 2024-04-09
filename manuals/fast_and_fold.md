# Fast and fold!
AlphaPulldown V2 (beta) now can speed up your modelling process, save your computing time and resources by making multiple input feature matrices have 
the same shape, as suggested by [AlphaFold](https://github.com/google-deepmind/alphafold#inferencing-many-proteins) themselves. 

For example, using ```pulldown``` mode plus [```baits.txt```](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_data/baits.txt) and [```candidates.txt```](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_data/candidates.txt)
will inform AlphaPulldown to generate 294 different modelling tasks and reserve 294 GPU cards to finish them. However, in AlphaPulldown V2, 
you can model all of them on just **1** GPU and faster by the following command:
```bash
run_multimer_jobs.py --mode=pulldown \
--output_path=/path/to/output_dir \--data_dir=/scratch/AlphaFold_DBs/2.3.0 \
--monomer_objects_dir=/path/to/monomer/features \
--protein_lists=baits.txt,candidates.txt \
--desired_num_res=2600 --desired_num_msa=4090
```
2 extra parameters need to be defined: ```desired_num_res``` and ```desired_num_msa```, which correspond to the longest protein complex
among the 294 structures and the deepest MSA depth among these jobs.

## How could we determin ```desired_num_res``` and ```desired_num_msa```?
