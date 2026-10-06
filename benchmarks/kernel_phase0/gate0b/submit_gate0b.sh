#!/bin/bash
# Submit Gate 0b: one job per (card, arm). Freezes AlphaPulldown main and gate0b/ first, so every queued job runs the same code.
#   ./submit_gate0b.sh [labels...]     default: h100 a40 a100
#   AP_REF=origin/main (default)       the AlphaPulldown commit to test
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$CODE/bench.env"
AP_REPO=${AP_REPO:-/g/kosinski/dima/PycharmProjects/AlphaPulldown}
AP_REF=${AP_REF:-origin/main}
ap=$(git -C "$AP_REPO" rev-parse --short=8 "$AP_REF")
g=$(cat "$CODE/gate0b/fused_af3.py" "$CODE/gate0b/launch.py" | sha256sum | cut -c1-8)
SRC=$BENCH/gate0b/src/ap_${ap}_g0b_$g
if [ ! -d "$SRC" ]; then
  mkdir -p "$SRC/ap" "$SRC/gate0b"
  git -C "$AP_REPO" archive "$AP_REF" alphapulldown | tar -x -C "$SRC/ap"
  cp "$CODE/gate0b/fused_af3.py" "$CODE/gate0b/launch.py" "$SRC/gate0b/"
  echo "$ap" > "$SRC/AP_COMMIT"
fi
mkdir -p "$BENCH/gate0b/logs"
export SBATCH_QOS=${SBATCH_QOS:-high}
for gpu in ${*:-h100 a40 a100}; do
  case $gpu in
    h100|h100pcie|h200|rtx6000) folds="s0599 s0896 s1309 s1792 s2546 r3584 r5376"; wt=05:00:00 ;;
    *) folds="s0599 s0896 s1309 s1792 s2546"; wt=05:00:00 ;;
  esac
  folds=${FOLDS_OVERRIDE:-$folds}
  for arm in ${ARMS:-stock fused}; do
    jid=$(sbatch --parsable -p "$(gpu_partition "$gpu")" --gres="$(gpu_gres "$gpu")" -t "$wt" -J "kb0-g0b-$gpu-$arm" \
          -o "$BENCH/gate0b/logs/%x_%j.out" --export=ALL,RUN_TAG="${RUN_TAG:-}",CODE="$CODE",GPU="$gpu",ARM="$arm",SRC="$SRC",FOLDS="$folds" \
          "$CODE/gate0b/run_gate0b.sbatch")
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$jid" "$gpu" "$arm" "$SRC" | tee -a "$BENCH/gate0b/jobs.tsv"
  done
done
