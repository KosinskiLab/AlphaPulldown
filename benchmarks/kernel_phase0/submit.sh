#!/bin/bash
# Submit Phase 0 as SLURM jobs chained by dependencies. Nothing runs on the login node, so the
# shell can be closed as soon as this returns; the last job writes $BENCH/RESULT.txt.
#
#   ./submit.sh smoke                 every arm + AF3 on the smallest fold, one H100 (gpu-el10)
#   ./submit.sh full [gpu ...]        speed ladder per card, accuracy on H100, AF3 baseline per card
#                                     (ARMS="cf163_fast kit_fast" limits the speed jobs to those arms,
#                                     WALLTIME=04:00:00 overrides the per-card limit)
#                                     default: h100 l40s a40 (gpu-el10, short queues); the gpu-el8
#                                     cards (3090 a100 rtx6000) queue for weeks: submit them as a
#                                     second wave, `submit.sh full 3090 a100 rtx6000`, then `collect`
#   ./submit.sh accuracy <gpu> [arm ...]   accuracy suite only, on one card
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
  jid=$(sb ${dep:+--dependency=afterok:$dep} ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} -p "$part" --gres="$(gpu_gres "$gpu")" -t "$wall" \
        -J "kb0-$suite-$gpu-$arm" -o "$BENCH/logs/${suite}_${gpu}_${arm}_%j.out" \
        --export=ALL,CODE="$CODE",SUITE="$suite",GPU="$gpu",ARM="$arm",SEEDS="$seeds" "$script")
  printf "%s\t%s\t%s\t%s\n" "$jid" "$suite" "$gpu" "$arm" >> "$JOBS"
  echo "$suite/$gpu/$arm  $jid"
  all="$all:$jid"
}

chunk_job() {  # gpu arm tag folds(colon-sep) reps walltime — one short piece of an arm's speed ladder
  local gpu=$1 arm=$2 tag=$3 folds=$4 reps=$5 wall=$6 jid
  jid=$(sb ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} -p "$(gpu_partition "$gpu")" --gres="$(gpu_gres "$gpu")" -t "$wall" \
        -c 4 --mem=32G -J "kb0-speed-$gpu-$arm-$tag" -o "$BENCH/logs/speed_${gpu}_${arm}_${tag}_%j.out" \
        --export=ALL,CODE="$CODE",SUITE=speed,GPU="$gpu",ARM="$arm",SEEDS="$SEED",RUN_TAG="$tag",FOLD_SUBSET="$folds",REPS="$reps" \
        "$CODE/run_af2_arm.sbatch")
  printf "%s\t%s\t%s\t%s\n" "$jid" speed "$gpu" "$arm#$tag" >> "$JOBS"
  echo "speed/$gpu/$arm#$tag  $jid"
  all="$all:$jid"
}

accuracy_jobs() {  # gpu [arm ...] — stock arms twice (seed 0 and 1) for the seed-to-seed spread
  local gpu=$1 arm seeds; shift
  for arm in ${*:-$AF2_ARMS}; do
    seeds=$SEED; case $arm in ap_stock|cf163_stock|kit_off) seeds="0:1" ;; esac
    gpu_job accuracy "$gpu" "$arm" "$(gpu_partition "$gpu")" "${WALLTIME:-05:00:00}" "$seeds"
  done
}

collect() {
  local jid
  jid=$(sb ${all:+--dependency=afterany${all}} --export=ALL,CODE="$CODE" -o "$BENCH/logs/collect_%j.out" "$CODE/collect.sbatch")
  echo "collect $jid  ->  $BENCH/REPORT.md, $BENCH/RESULT.txt"
}

case $MODE in
  smoke)
    prepare
    for arm in $AF2_ARMS ap_af3; do gpu_job smoke "${SMOKE_GPU:-3090}" "$arm" "$(gpu_partition "${SMOKE_GPU:-3090}")" 00:45:00; done
    collect ;;
  full)
    prepare
    # One collector per card (and one for accuracy), each rebuilding the whole report, so the
    # report fills in as each card finishes instead of waiting for the slowest queue.
    for gpu in $GPUS; do
      all=""
      for arm in ${ARMS:-$AF2_ARMS ap_af3}; do gpu_job speed "$gpu" "$arm" "$(gpu_partition "$gpu")" "${WALLTIME:-$(gpu_walltime "$gpu")}"; done
      collect
    done
    all=""
    # Accuracy on one card (ACCURACY_GPU, default a40), only when that card is in this wave and no ARMS subset is given:
    # the seed-to-seed spread every other difference is judged against. Every accuracy fold is
    # <= 896 tokens, so any card here holds it.
    acc_gpu=${ACCURACY_GPU:-a40}
    if [[ " $GPUS " == *" $acc_gpu "* ]] && [ -z "${ARMS:-}" ]; then accuracy_jobs "$acc_gpu"; fi
    collect ;;
  accuracy)
    # The accuracy suite alone on one card, e.g. where a kernel source only engages there:
    # `submit.sh accuracy h100` (all arms) or `submit.sh accuracy h100 kit_off kit_fast`.
    prepare
    gpu=${1:?usage: submit.sh accuracy <gpu> [arm ...]}; shift
    all=""
    accuracy_jobs "$gpu" "$@"
    collect ;;
  chunked)
    # The AF2 speed ladder per arm in two jobs short enough for a partition that only backfills
    # short work (test's single 3090 node, 2026-10-01): small rungs, then the two largest with
    # one timed rep. Arms named after the card, e.g. `submit.sh chunked 3090 cf163_stock kit_fast`.
    prepare
    gpu=${1:?usage: submit.sh chunked <gpu> [arm ...]}; shift
    arms=${*:-$AF2_ARMS}
    all=""
    for arm in $arms; do
      chunk_job "$gpu" "$arm" c1 s0164:s0357:s0599:s0896:s1309 3 02:00:00
      chunk_job "$gpu" "$arm" c2 s1792:s2546 2 02:00:00
    done
    collect ;;
  collect)
    collect ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
