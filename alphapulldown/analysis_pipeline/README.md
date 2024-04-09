These scripts are used when creating the singularity image as mentioned in the 3rd step in the manuals
When developing, you are free to add you changes to the python scripts in this sub-directory. Then please use ```alpha_analysis_jax0.3.def``` or ```alpha_analysis_jax0.4.def``` to build the singularity image by running 
```
singularity build alpha-analysis_jax_0.4.sif alpha_analysis_jax0.4.def
```
**NOTE** the ```DockerFile``` is retired and no longer used. All necessary dependencies have already been published on Dockerhub, which are used in the ```.def``` files
