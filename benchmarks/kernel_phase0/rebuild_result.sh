#!/bin/bash
# Rebuild RESULT.txt for a run whose job died after its folds finished (e.g. its script was
# edited under it). Same content as the job writes, from the run's own runs.tsv, meta.txt and
# gpu.txt, with "rebuilt=" in place of the finish time. Usage: rebuild_result.sh <run dir>...
for run in "$@"; do
  [ -f "$run/runs.tsv" ] && [ -f "$run/meta.txt" ] || { echo "skip $run: no runs.tsv/meta.txt"; continue; }
  [ -f "$run/RESULT.txt" ] && { echo "skip $run: RESULT.txt exists"; continue; }
  {
    echo "rebuilt=$(date -Iseconds) $(cat "$run/meta.txt")"
    echo "gpu: $(cat "$run/gpu.txt" 2>/dev/null)"
    cut -f1,2,4,7 "$run/runs.tsv" | tail -n +2 | awk '{print "  " $0}'
  } > "$run/RESULT.txt"
  echo "rebuilt $run ($(($(wc -l < "$run/runs.tsv") - 1)) folds)"
done
