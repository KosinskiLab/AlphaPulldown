#!/bin/bash

snakemake \
  --use-singularity \
  --singularity-args "-B /scratch:/scratch \
    -B /g/kosinski:/g/kosinski \
    --nv \
    -B /home/vmaurer/src:/home/vmaurer/src" \
  --jobs 200 \
  --restart-times 5 \
  --profile slurm_noSidecar \
  --rerun-incomplete \
  --rerun-triggers mtime \
  --latency-wait 30
