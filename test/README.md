# How to run the unittests (for developers @ KosinskiLab)

## tests that need to run with AlphaFold DBs and CPUs:

1st move one level above the ```test/``` folder, e.g. within the ```AlphaPulldown``` folder 

2nd run:

```bash
sbatch test/run_tests_on_cpu.sh <your conda environment e.g. AlphaPulldown>
```

## tests within ```github_CI_tests``` will automatically run during CI/CD

## tests for AlphaLink2 integration:
Make sure you have every dependency installed, including CUDA, cuDNN, pytorch etc. Check [this manual](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/run_with_AlphaLink2.md) for the details.
then run:

```bash
sbatch test/run_alphalink2_tests.sh <your conda environment e.g. AlphaPulldown>
```
