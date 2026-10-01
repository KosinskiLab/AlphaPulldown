#!/bin/bash
# Gather every finished run into $BENCH/REPORT.md, $BENCH/report/*.tsv and $BENCH/RESULT.txt.
# Submitted afterany behind every benchmark job, so a failed arm shows up as missing rows
# instead of stalling the report. Safe to resubmit by hand at any time.
#SBATCH -J kb0-collect
#SBATCH -p htc-el10,htc-el8
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00

set -uo pipefail
source "${CODE:?}/bench.env"
"$HOST_PY" "$CODE/collect.py" --bench "$BENCH" --dockq "$DOCKQ" --expected "$BENCH/jobs.tsv"
