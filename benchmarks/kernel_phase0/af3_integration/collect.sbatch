#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 02:00:00
set -uo pipefail
source "$SNAPSHOT/harness/bench.env"
trap 'rc=$?; printf "collector_exit=%s\ncampaign=%s\nreport=%s/REPORT.md\n" "$rc" "$CAMPAIGN" "$CAMPAIGN" > "$CAMPAIGN/RESULT.txt"' EXIT
ids=$(cut -f1 "$CAMPAIGN/jobs.tsv" | paste -sd,)
sacct -j "$ids" -X --format=JobID,State,ExitCode,NodeList,Elapsed -P > "$CAMPAIGN/sacct.txt"
"$HOST_PY" "$SNAPSHOT/harness/af3_integration/collect.py" "$CAMPAIGN" "$BENCH" "$DOCKQ"
