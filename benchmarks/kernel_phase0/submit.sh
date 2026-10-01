#!/bin/bash
# Submit Phase 0 as SLURM jobs chained by dependencies. Nothing runs on the login node, so the
# shell can be closed as soon as this returns; the last job writes $BENCH/RESULT.txt.
#
#   ./submit.sh smoke                 every arm + AF3 on the smallest fold, one H100 (gpu-el10)
#   ./submit.sh full [gpu ...]        speed ladder per card, accuracy on H100, AF3 baseline per card
#                                     default: h100 l40s a40 (gpu-el10, short queues); the gpu-el8
#                                     cards (3090 a100 rtx6000) queue for weeks: submit them as a
#                                     second wave, `submit.sh full 3090 a100 rtx6000`, then `collect`
#   ./submit.sh collect               re-run the collector only
#
# Images and inputs are built first when missing (images/build_images.sbatch, make_inputs.sbatch);
# every GPU job waits on them with afterok, the collector on everything with afterany.
set -euo pipefail
CODE=$(cd "$(dirname "$0")" && pwd)
export CODE
source "$CODE/bench.env"
MODE=${1:?usage: submit.sh smoke|full [gpu ...]|collect}
shift
GPUS=${*:-$ALL_GPUS}
mkdir -p "$BENCH/logs"
JOBS=$BENCH/jobs.tsv
[ -f "$JOBS" ] || printf "job\tsuite\tgpu\tarm\n" > "$JOBS"
all=""

sb() { sbatch --parsable "$@"; }

prepare() {
  dep=""
  if [ ! -s "$SIF_AP_AF2" ] || [ ! -s "$SIF_AP_AF3" ] || [ ! -s "$SIF_CF163" ] || [ ! -s "$SIF_KIT" ]; then
    dep=$(sb --export=ALL,CODE="$CODE" -o "$BENCH/logs/images_%j.out" "$CODE/images/build_images.sbatch")
    echo "images  $dep"
  fi
  if ! grep -qs '^exit=0' "$BENCH/inputs/RESULT.txt"; then
    dep=$(sb ${dep:+--dependency=afterok:$dep} --export=ALL,CODE="$CODE" -o "$BENCH/logs/inputs_%j.out" "$CODE/make_inputs.sbatch")
    echo "inputs  $dep"
  fi
}

gpu_job() {  # suite gpu arm partition walltime [seeds]
  local suite=$1 gpu=$2 arm=$3 part=$4 wall=$5 seeds=${6:-$SEED} script=$CODE/run_af2_arm.sbatch jid
  [ "$arm" = ap_af3 ] && script=$CODE/run_af3_baseline.sbatch
  jid=$(sb ${dep:+--dependency=afterok:$dep} -p "$part" --gres="$(gpu_gres "$gpu")" -t "$wall" \
        -J "kb0-$suite-$gpu-$arm" -o "$BENCH/logs/${suite}_${gpu}_${arm}_%j.out" \
        --export=ALL,CODE="$CODE",SUITE="$suite",GPU="$gpu",ARM="$arm",SEEDS="$seeds" "$script")
  printf "%s\t%s\t%s\t%s\n" "$jid" "$suite" "$gpu" "$arm" >> "$JOBS"
  echo "$suite/$gpu/$arm  $jid"
  all="$all:$jid"
}

collect() {
  local jid
  jid=$(sb ${all:+--dependency=afterany${all}} --export=ALL,CODE="$CODE" -o "$BENCH/logs/collect_%j.out" "$CODE/collect.sbatch")
  echo "collect $jid  ->  $BENCH/REPORT.md, $BENCH/RESULT.txt"
}

case $MODE in
  smoke)
    prepare
    for arm in $AF2_ARMS ap_af3; do gpu_job smoke h100 "$arm" "$(gpu_partition h100)" 00:45:00; done
    collect ;;
  full)
    prepare
    for gpu in $GPUS; do
      for arm in $AF2_ARMS ap_af3; do gpu_job speed "$gpu" "$arm" "$(gpu_partition "$gpu")" "$(gpu_walltime "$gpu")"; done
    done
    # Accuracy on one card, in the first wave only: stock arms twice (seed 0 and 1) for the
    # seed-to-seed spread every other difference is judged against.
    if [[ " $GPUS " == *" h100 "* ]]; then
      for arm in $AF2_ARMS; do
        seeds=$SEED; case $arm in ap_stock|cf163_stock|kit_off) seeds="0:1" ;; esac
        gpu_job accuracy h100 "$arm" "$(gpu_partition h100)" 03:00:00 "$seeds"
      done
    fi
    collect ;;
  collect)
    collect ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
