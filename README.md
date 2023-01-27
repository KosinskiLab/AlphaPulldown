# AlphaPulldown

AlphaPulldown is a Python package that streamlines protein-protein interaction screens and high-throughput modelling of higher-order oligomers using AlphaFold-Multimer:
* provides a convenient command line interface to screen a bait protein against many candidates, calculate all-versus-all pairwise comparisons, test alternative homo-oligomeric states, and model various parts of a larger complex
* separates the CPU stages (MSA and template feature generation) from GPU stages (the actual modeling)
* allows modeling fragments of proteins without recalculation of MSAs and keeping the original full-length residue numbering in the models
* summarizes the results in a CSV table with AlphaFold scores, pDockQ and mpDockQ, PI-score, and various physical parameters of the interface
* provides a Jupyter notebook for an interactive analysis of PAE plots and models

## Pre-installation
Check if you have downloaded necessary parameters and databases (e.g. BFD, MGnify etc.) as instructed in [AlphFold's documentation](https://github.com/deepmind/alphafold). You should have a directory like below:
 ```
 alphafold_database/                             # Total: ~ 2.2 TB (download: 438 GB)
    bfd/                                   # ~ 1.7 TB (download: 271.6 GB)
        # 6 files.
    mgnify/                                # ~ 64 GB (download: 32.9 GB)
        mgy_clusters_2018_12.fa
    params/                                # ~ 3.5 GB (download: 3.5 GB)
        # 5 CASP14 models,
        # 5 pTM models,
        # 5 AlphaFold-Multimer models,
        # LICENSE,
        # = 16 files.
    pdb70/                                 # ~ 56 GB (download: 19.5 GB)
        # 9 files.
    pdb_mmcif/                             # ~ 206 GB (download: 46 GB)
        mmcif_files/
            # About 180,000 .cif files.
        obsolete.dat
    pdb_seqres/                            # ~ 0.2 GB (download: 0.2 GB)
        pdb_seqres.txt
    small_bfd/                             # ~ 17 GB (download: 9.6 GB)
        bfd-first_non_consensus_sequences.fasta
    uniclust30/                            # ~ 86 GB (download: 24.9 GB)
        uniclust30_2018_08/
            # 13 files.
    uniprot/                               # ~ 98.3 GB (download: 49 GB)
        uniprot.fasta
    uniref90/                              # ~ 58 GB (download: 29.7 GB)
        uniref90.fasta
 ```

## Installation 

**Firstly**, install [Anaconda](https://www.anaconda.com/) and create AlphaPulldown environment, gathering necessary dependencies 
```bash
conda create -n AlphaPulldown -c omnia -c bioconda -c conda-forge python==3.8 openmm=7.5.1 pdbfixer kalign2=2.04 cctbx-base
```

**Secondly**, activate the AlphaPulldown environment and install AlphaPulldown
```bash
source activate AlphaPulldown
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple alphapulldown==0.30.7
pip install -q "jax[cuda]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Optionally**, if you do not have these software yet on your system, install [HMMER](http://hmmer.org/documentation.html), [HH-suite](https://github.com/soedinglab/hh-suite) from Anaconda
```bash
source activate AlphaPulldown
conda install -c bioconda hmmer hhsuite
```
This usually works, but on some compute systems users may wish to use other versions or optimized builds of already installed HMMER and HH-suite.

------

## Manuals
AlphaPulldown supports four different modes of massive predictions: 

* ```pulldown``` - to screen a list of "bait" proteins against a list or lists of other proteins
* ```all_vs_all``` - to model all pairs of a protein list
* ```homo-oligomer``` - to test alternative oligomeric states
* ```custom``` - to model any combination of proteins and their fragments, such as a pre-defined list of pairs or fragments of a complex

AlphaPulldown will return models of all interactions, summarize results in a score table, and will provide a [Jupyter](https://jupyter.org/) notebook for an interactive analysis, including PAE plots and 3D displays of models colored by chain and pLDDT score.

## Examples

Example 1 is a case where ```pulldown``` mode is used. Manual: [example_1](./example_1.md)

Example 2 is a case where ```custom``` and ```homo-oligomer``` modes are used. Manual: [example_2](./example_2.md) 

```all_vs_all``` mode can be viewed as a special case of the ```pulldown``` mode thus the instructions of this mode are added as Appendix in both manuals mentioned above. 
