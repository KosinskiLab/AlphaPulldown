# Fast and fold!
AlphaPulldown V2 (beta) now can save your computing time and resources by making multiple input feature matrices have 
the same shape, as suggested by [AlphaFold](https://github.com/google-deepmind/alphafold#inferencing-many-proteins) themselves. 

For example, using ```pulldown``` mode plus [```baits.txt```](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_data/baits.txt) and [```candidates.txt```](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_data/candidates.txt)
will inform AlphaPulldown to generate 294 different modelling tasks.  However, in AlphaPulldown V2, 
you can model all of them on just **1** GPU and faster by the following command:
```bash
run_multimer_jobs.py --mode=pulldown \
--output_path=/path/to/output_dir --data_dir=/scratch/AlphaFold_DBs/2.3.0 \
--monomer_objects_dir=/path/to/monomer/features \
--protein_lists=baits.txt,candidates.txt \
--desired_num_res=2600 --desired_num_msa=4090
```
two extra parameters need to be defined: ```desired_num_res``` and ```desired_num_msa```, which correspond to the longest protein complex
```suggestion
2 extra parameters need to be defined: ```desired_num_res``` and ```desired_num_msa```, which correspond to the longest protein complex
among the 294 structures and the deepest MSA depth among these jobs.

## How could we determine ```desired_num_res``` and ```desired_num_msa```?
AlphaPulldown V2 now provides you with the script [split_jobs_into_clusters.py](https://github.com/KosinskiLab/AlphaPulldown/blob/manuals-2.0-beta/alphapulldown/scripts/split_jobs_into_clusters.py). At the moment, this part of AlphaPulldown is still being benchmarked and tested so this script is not yet executable directly from the command line. You can still use it by:

```
python split_jobs_into_clusters.py  --protein_lists baits.txt,candidates.txt \
--output_dir /path/to/output_dir --mode pulldown \
--features_directory /path/to/monomer/features
```
and your desired number of residues and your desired number of MSAs will be print out in the command line.
