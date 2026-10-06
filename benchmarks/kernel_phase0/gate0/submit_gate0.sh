#!/bin/bash
# Submit Gate 0: one job per (card, size range). Freezes gate0_layers.py first, so every queued job tests the same code.
#   ./submit_gate0.sh [labels...]          default: h100 a100 a40 l40s 3090 and rtx6000 (on the 9.0 table, exploratory)
#   PARTS="small" ./submit_gate0.sh a40    one size range only
# QOS: SBATCH_QOS (default high) is read by sbatch at submission; scontrol update after the fact changes nothing.
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$CODE/bench.env"
G=$BENCH/gate0
mkdir -p "$G/src" "$G/logs"
[ -d "$G/src/kit_f4f62fa6/common/opt_core" ] || { echo "missing frozen kit core in $G/src (git archive f4f62fa6 common/opt_core)"; exit 1; }
SHA=$(sha256sum "$CODE/gate0/gate0_layers.py" | cut -c1-12)
PY=$G/src/gate0_layers.$SHA.py
[ -f "$PY" ] || cp "$CODE/gate0/gate0_layers.py" "$PY"
export SBATCH_QOS=${SBATCH_QOS:-high}
LABELS=${*:-"h100 a100 a40 l40s 3090 rtx6000"}
for gpu in $LABELS; do
  fb=""; [ "$gpu" = rtx6000 ] && fb=1          # sm_120: no tile table and no safe rows; run the 9.0 table, recorded as fallback
  for part in ${PARTS:-small large}; do
    wt=05:00:00; case $gpu in 3090|a40|l40s|rtx6000) wt=06:00:00 ;; esac
    jid=$(sbatch --parsable -p "$(gpu_partition "$gpu")" --gres="$(gpu_gres "$gpu")" -t "$wt" ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} \
          -J "kb0-g0-$gpu-$part" -o "$G/logs/%x_%j.out" \
          --export=ALL,CODE="$CODE",GPU="$gpu",PART="$part",GATE0_PY="$PY",FALLBACK="$fb" "$CODE/gate0/run_gate0.sbatch")
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$jid" "$gpu" "$part" "${fb:-0}" "$SHA" | tee -a "$G/jobs.tsv"
  done
done
