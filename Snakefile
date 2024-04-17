""" Snakemake pipeline for automated structure prediction using various backends.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import sys
import tempfile
from sys import exit
from os import makedirs
from os.path import abspath, join, splitext, basename
from source.input_parser import InputParser
from scripts.create_combinations import process_files

configfile: "config/config.yaml"
config["output_directory"] = abspath(config["output_directory"])
makedirs(config["output_directory"], exist_ok = True)

protein_delimiter = config.get("protein_delimiter", ";")

with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_input:
    input_files = config["input_files"]
    if isinstance(input_files, str):
        input_files = [input_files]

    process_files(
        input_files = input_files,
        output_path = tmp_input.name,
        delimiter = protein_delimiter
    )

    dataset = InputParser.from_file(
        filepath = tmp_input.name,
        file_format = "alphaabriss",
        protein_delimiter = protein_delimiter
    )


ruleorder: symlink_local_files > download_uniprot


required_folds = [
    join(
        config["output_directory"],
        "predictions", fold, "completed_fold.txt"
    )
    for fold in dataset.fold_specifications
]
required_reports = [
    join(
        config["output_directory"], "reports", "statistics.csv"
    ),
    join(
        config["output_directory"], "reports", "report.html"
    )
]
total_required_files = [*required_folds, *required_reports]

if config.get("only_generate_features", False):
    total_required_files = [
        join(config["output_directory"], "features", f"{fasta_basename}.pkl")
        for fasta_basename in dataset.unique_sequences
    ]


rule all:
    input:
        total_required_files,


rule symlink_local_files:
    input:
        dataset.sequences_by_origin["local"],
    output:
        [
            join(config["output_directory"], "data", f"{splitext(basename(x))[0]}.fasta")
            for x in dataset.sequences_by_origin["local"]

        ],
    resources:
        avg_mem = lambda wildcards, attempt: 600 * attempt,
        mem_mb = lambda wildcards, attempt: 800 * attempt,
        walltime = lambda wildcards, attempt: 10 * attempt,
        attempt = lambda wildcards, attempt: attempt,
    run:
        dataset.symlink_local_files(output_directory = join(config["output_directory"], "data"))

rule download_uniprot:
    output:
        join(config["output_directory"], "data", "{uniprot_id}.fasta"),
    resources:
        avg_mem = lambda wildcards, attempt: 600 * attempt,
        mem_mb = lambda wildcards, attempt: 800 * attempt,
        walltime = lambda wildcards, attempt: 10 * attempt,
        attempt = lambda wildcards, attempt: attempt,
    shell:"""
        temp_file=$(mktemp)
        curl -o ${{temp_file}} https://rest.uniprot.org/uniprotkb/{wildcards.uniprot_id}.fasta
        echo ">{wildcards.uniprot_id}" > {output}
        tail -n +2 ${{temp_file}} >> {output}
        """


rule create_features:
    input:
        join(config["output_directory"], "data", "{fasta_basename}.fasta"),
    output:
        join(config["output_directory"], "features", "{fasta_basename}.pkl"),
    params:
        data_directory = config["alphafold_data_directory"],
        output_directory = join(config["output_directory"], "features"),
        save_msa = config.get("save_msa", False),
        use_precomputed_msa = config.get("use_precomputed_msa", True),
    resources:
        mem_mb = lambda wildcards, attempt: 64000 * attempt,
        walltime = lambda wildcards, attempt: 1440 * attempt,
        attempt = lambda wildcards, attempt: attempt,
    threads: 8, # everything is harcoded in AF anyways ...
    container:
       "docker://dquz/fold:latest",
    shell:"""
        create_individual_features.py \
            --fasta_paths={input} \
            --data_dir={params.data_directory} \
            --output_dir={params.output_directory} \
            --save_msa_files={params.save_msa} \
            --use_precomputed_msas={params.use_precomputed_msa} \
            --max_template_date=2050-01-01 \
            --skip_existing=False
        """

memscaling_inference = config["alphafold_inference_threads"] / 8
rule alphafold_inference:
    input:
        lambda wildcards : [join(
            config["output_directory"], "features", f"{feature}.pkl")
            for feature in dataset.sequences_by_fold[wildcards.fold]],
    output:
        join(
            config["output_directory"],
            "predictions", "{fold}", "completed_fold.txt"
        ),
    params:
        data_directory = config["alphafold_data_directory"],
        predictions_per_model = config["predictions_per_model"],
        n_recycles = (
            lambda wildcards: config["number_of_recycles"]
            if len(wildcards.fold.split(protein_delimiter)) > 1
            else min(3, config["number_of_recycles"])
        ),
        feature_directory = join(config["output_directory"], "features"),
        output_directory = lambda wildcards: join(
            config["output_directory"], "predictions", wildcards.fold
        ),
        requested_fold = lambda  wildcards : wildcards.fold,
        protein_delimiter = protein_delimiter,
    resources:
        mem_mb = lambda wildcards, attempt: 128000 * attempt * memscaling_inference,
        walltime = lambda wildcards, attempt: 1440 * attempt,
        attempt = lambda wildcards, attempt: attempt,
        slurm = config.get("alphafold_inference", ""),
    threads:
        config["alphafold_inference_threads"],
    container:
       "docker://dquz/fold:latest",
    shell:"""
        #MAXRAM=$(bc <<< "$(ulimit -m) / 1024.0")
        #GPUMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tail -1)
        #export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "scale=3; $MAXRAM / $GPUMEM" | bc)
        #export TF_FORCE_UNIFIED_MEMORY='1'

        python3 workflow/source/run_structure_prediction.py \
            --input {params.requested_fold} \
            --output_directory={params.output_directory} \
            --num_cycle={params.n_recycles} \
            --num_predictions_per_model={params.predictions_per_model} \
            --data_directory={params.data_directory} \
            --features_directory={params.feature_directory} \
            --protein_delimiter {params.protein_delimiter}
        echo "Completed" > {output}
        """


rule compute_stats:
    input:
        required_folds,
    output:
        join(
            config["output_directory"], "reports", "statistics.csv"
        ),
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * attempt,
        walltime = lambda wildcards, attempt: 1440 * attempt,
        attempt = lambda wildcards, attempt: attempt,
    params:
        prediction_dir = join(config["output_directory"], "predictions"),
        report_dir = join(config["output_directory"], "reports"),
        report_cutoff = config["report_cutoff"],
    container:
       "docker://dquz/fold_analysis:latest",
    shell:"""
        cd {params.prediction_dir}

        run_get_good_pae.sh \
            --output_dir={params.prediction_dir} \
            --cutoff={params.report_cutoff}
        mv r4s.res {params.report_dir}
        mv pi_score_outputs {params.report_dir}
        mv predictions_with_good_interpae.csv {output}
        """

rule generate_report:
    input:
        required_folds,
    output:
        join(
            config["output_directory"], "reports", "report.html"
        ),
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * attempt,
        walltime = lambda wildcards, attempt: 1440 * attempt,
        attempt = lambda wildcards, attempt: attempt,
    params:
        prediction_dir = join(config["output_directory"], "predictions"),
        report_dir = join(config["output_directory"], "reports"),
        report_cutoff = config["report_cutoff"],
    container:
       "docker://dquz/fold:latest",
    shell:"""
        cd {params.prediction_dir}
        create_notebook.py \
            --cutoff={params.report_cutoff} \
            --output_dir={params.prediction_dir}
        jupyter nbconvert --to html --execute output.ipynb
        mv output.ipynb {params.report_dir}
        mv output.html {output}
        """