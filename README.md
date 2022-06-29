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

Example 2 is a case where custom and homo-oligomer modes were used. Manuals: [example_2_for_pip_user](./example_2_for_pip_user.md) or [example_2_for_singularity_user](./singularity_user_example_2.md)

all_vs_all mode can be viewed as a special case of pulldown mode thus the instructions of all_vs_all mode are aded as Appendix in all the manuals mentioned above. 
