#!/bin/bash
# Gate 0 on one card: gate0_layers.py (layer-level parity, speed, memory of the kit's fused triangle kernels against our AF3
# fork's modules) in the AlphaPulldown AF3 image, against the kit's frozen core. No model, no AlphaPulldown.
# Env in: CODE, GPU (label), GATE0_PY (a frozen copy, made by submit_gate0.sh), PART (small|large: the size range),
#         FALLBACK=1 runs a card without a tile table on the kit's 9.0 table (OPT_CORE_PALLAS_ALLOW_FALLBACK_CC=1).
# Out: $BENCH/gate0/runs/<GPU>[_fallback]/<PART>_<job>/{cells.jsonl,gate0.log,gpu.txt,meta.txt,RESULT.txt}
# sbatch spools this file at submission and the Python file is a frozen copy, so later edits never reach a queued job.
#SBATCH -J kb0-gate0
#SBATCH -c 16
#SBATCH --mem=128G

set -uo pipefail
source "${CODE:?}/bench.env"
: "${GPU:?}" "${GATE0_PY:?}" "${PART:?}"
TAG=$GPU${FALLBACK:+_fallback}
RUN=$BENCH/gate0/runs/$TAG/${PART}_${SLURM_JOB_ID:-local}
mkdir -p "$RUN"
cd "$RUN" || exit 1
KIT=$BENCH/gate0/src/kit_f4f62fa6/common/opt_core

if ! nvidia-smi -L > /dev/null 2>&1; then   # a broken driver must not produce numbers
  echo "node_unusable node=$(hostname) job=${SLURM_JOB_ID:-none}: $(nvidia-smi 2>&1 | head -1)" | tee RESULT.txt
  exit 3
fi
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null \
           || nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
case $GPU_UUID in GPU-*) ;; *) GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1) ;; esac
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm \
  --format=csv,noheader -i "$GPU_UUID" > gpu.txt

case $PART in
  small) SIZE_ARGS=(--sizes 256,512,768,1024,1280,1536,2048 --asym_sizes 256,512 --dm_weights "$AF3_WEIGHTS") ;;
  large) SIZE_ARGS=(--sizes 2560,3072,3584,4096,4608,5120 --asym_sizes "") ;;
  *) echo "unknown PART $PART"; exit 2 ;;
esac
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} gpu=$GPU part=$PART fallback=${FALLBACK:-0} py=$GATE0_PY sha=$(sha256sum "$GATE0_PY" | cut -c1-12) kit=$KIT image=$SIF_AP_AF3" | tee meta.txt
cat gpu.txt

THREADS=${SLURM_CPUS_PER_TASK:-8}
RUNENV=($MEM_ENV_AF3 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" --env "OMP_NUM_THREADS=$THREADS"
        --env "OPENBLAS_NUM_THREADS=$THREADS" --env "MKL_NUM_THREADS=$THREADS")
[ -n "${FALLBACK:-}" ] && RUNENV+=(--env OPT_CORE_PALLAS_ALLOW_FALLBACK_CC=1)
t0=$(date +%s)
apptainer exec --nv --cleanenv "${RUNENV[@]}" --bind "$BENCH" --bind /g/alphafold "$SIF_AP_AF3" \
  python "$GATE0_PY" --out "$RUN/cells.jsonl" --kit "$KIT" --negative_control "${SIZE_ARGS[@]}" ${GATE0_ARGS:-} > gate0.log 2>&1
rc=$?
{
  echo "finished=$(date -Iseconds) rc=$rc wall_s=$(( $(date +%s) - t0 )) $(cat meta.txt)"
  echo "gpu: $(cat gpu.txt)"
  echo "cells_done=$(grep -c '^DONE' gate0.log) cells_planned=$(grep -m1 '^CELLS' gate0.log | cut -d' ' -f2)"
  "$HOST_PY" -c "
import collections, json, sys
c = collections.Counter()
for l in open('cells.jsonl'):
    r = json.loads(l); c[(r.get('arm'), r.get('status'))] += 1
    if r.get('gate') is False and r.get('arm') != 'fused_biasT': c[('GATE_FAIL', r['arm'])] += 1
for k, v in sorted(c.items(), key=str): print('  %s %s: %d' % (k[0], k[1], v))
" 2>&1
} > RESULT.txt
cat RESULT.txt
exit $rc
