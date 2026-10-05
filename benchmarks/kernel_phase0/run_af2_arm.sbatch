#!/bin/bash
# One AF2 arm on one card. Every fold of the suite runs in its own process with REPS copies
# of the same input: rep 1 pays compilation, reps 2.. are the timed forward calls. A process
# per fold isolates out-of-memory failures and gives each fold its own peak-memory window.
# Folds run smallest first; after the first out-of-memory the larger ones are skipped.
#
# Env in: CODE, ARM (ap_stock|cf163_stock|cf163_fast|kit_off|kit_exact|kit_fast),
#         GPU (label from bench.env), SUITE (speed|accuracy|smoke), optional SEEDS (e.g. 0:1), REPS.
# Out: $BENCH/runs/$SUITE/$GPU/$ARM/{runs.tsv,gpu.txt,gpu_mem.csv,logs/,out/,RESULT.txt}
#SBATCH -J kb0-af2
#SBATCH -c 8
#SBATCH --mem=48G

set -uo pipefail
# Jobs can queue for days: run the script as it is in $CODE when the job starts, not the copy
# sbatch took at submission, so fixes committed meanwhile apply. meta.txt records the commit.
if [ -z "${KB0_INNER:-}" ]; then KB0_INNER=1 exec bash "${CODE:?}/run_af2_arm.sbatch" "$@"; fi
CODE_REV=$(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)$(git -C "$CODE" diff --quiet HEAD -- . 2>/dev/null || echo -dirty)
source "${CODE:?}/bench.env"
: "${ARM:?}" "${GPU:?}" "${SUITE:?}"
SEEDS=${SEEDS:-$SEED}
SEEDS=${SEEDS//:/ }   # colon-separated in --export, which reserves commas
# RUN_TAG splits one arm into chunks that fit short backfill windows: each chunk gets its own
# run directory (<arm>#<tag>) and FOLD_SUBSET (colon-separated fold names) picks its folds.
RUN=$BENCH/runs/$SUITE/$GPU/$ARM${RUN_TAG:+#$RUN_TAG}
mkdir -p "$RUN/logs" "$RUN/out" "$RUN/in" "$RUN/jax"
cd "$RUN" || exit 1

case $SUITE in
  speed) FOLDS=$("$HOST_PY" -c "import json;r=json.load(open('$BENCH/inputs/folds.json'));print(' '.join(x['name'] for x in sorted(r,key=lambda x:x['tokens']) if 'speed' in x['suites']))") ;;
  accuracy) FOLDS=$("$HOST_PY" -c "import json;r=json.load(open('$BENCH/inputs/folds.json'));print(' '.join(x['name'] for x in r if 'accuracy' in x['suites']))"); REPS=1 ;;
  smoke) FOLDS=s0164; REPS=2 ;;
  *) echo "unknown SUITE $SUITE"; exit 2 ;;
esac
[ -n "${FOLD_SUBSET:-}" ] && FOLDS=${FOLD_SUBSET//:/ }

# A node whose driver is broken must not produce numbers: stop before running anything.
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "node_unusable node=$(hostname) job=${SLURM_JOB_ID:-none}: $(nvidia-smi 2>&1 | head -1)" | tee RESULT.txt
  exit 3
fi

# The card, recorded once; memory sampled for this card only, every 500 ms.
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null \
           || nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
# On a MIG slice CUDA_VISIBLE_DEVICES names the slice, which --query-gpu does not accept: label the parent GPU.
case $GPU_UUID in GPU-*) ;; *) GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1) ;; esac
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm \
  --format=csv,noheader -i "$GPU_UUID" > gpu.txt
FORK_INFO=""
case $ARM in ap_fast|ap_fork_off) FORK_INFO=" fork=$(basename "$AF_FORK") colabfold_kernels=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$CFK_PKGS/colabfold_kernels/__init__.py")" ;; esac
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} arm=$ARM gpu=$GPU suite=$SUITE folds=[$FOLDS] reps=$REPS seeds=[$SEEDS] code=$CODE_REV$FORK_INFO" | tee meta.txt
cat gpu.txt
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -i "$GPU_UUID" -lms 500 > gpu_mem.csv 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
sleep 2

BINDS=(--bind "$BENCH" --bind "$CODE" --bind /g/alphafold)
RUNENV=($MEM_ENV --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" --env "JAX_COMPILATION_CACHE_DIR=$RUN/jax")
CF_ARGS=(--model-type alphafold2_multimer_v3 --num-models 1 --model-order 1 --num-recycle "$NUM_RECYCLE"
         --recycle-early-stop-tolerance 0 --num-seeds 1 --recompile-padding 0 --data "$BENCH/params_cf")

predict() {  # $1 fold  $2 seed  $3 tag ; writes logs/$tag.log, out/$tag/
  local fold=$1 seed=$2 tag=$3 in=$RUN/in/$3 out=$RUN/out/$3 r
  rm -rf "$in" "$out"; mkdir -p "$in" "$out"
  case $ARM in
    ap_stock|ap_fast|ap_fork_off)
      local spec; spec=$("$HOST_PY" -c "import json;print(next(x['fold'] for x in json.load(open('$BENCH/inputs/folds.json')) if x['name']=='$fold'))")
      for r in $(seq 1 "$REPS"); do
        printf '{"job_id": "%s_r%d", "input": "%s", "output_directory": "%s/%s_r%d"}\n' "$fold" "$r" "$spec" "$out" "$fold" "$r"
      done > "$in/manifest.jsonl"
      # ap_fast / ap_fork_off: the fork branch and colabfold_kernels ahead of the image's alphafold,
      # with the fused kernels on / off (ap_fork_off must reproduce ap_stock).
      local overlay=()
      case $ARM in ap_fast|ap_fork_off) overlay=(--env "PYTHONPATH=$AF_FORK:$CFK_PKGS" --bind "$AF_FORK" --bind "$CFK_PKGS") ;; esac
      [ "$ARM" = ap_fast ] && overlay+=(--env AP_FUSED_KERNELS=1)
      apptainer exec --nv --cleanenv "${RUNENV[@]}" "${overlay[@]}" "${BINDS[@]}" "$SIF_AP_AF2" \
        python "$CODE/ap_fixed_recycles.py" --manifest "$in/manifest.jsonl" \
        --data_directory "$AF2_DATA" --features_directory "$BENCH/inputs/features" --fold_backend alphafold2 \
        --model_names model_1_multimer_v3 --num_cycle "$NUM_RECYCLE" --num_predictions_per_model 1 \
        --skip_templates --random_seed "$seed" --remove_result_pickles --protein_delimiter + ;;
    cf163_stock|cf163_fast)
      for r in $(seq 1 "$REPS"); do ln -s "$BENCH/inputs/a3m/$fold.a3m" "$in/${fold}_r$r.a3m"; done
      local extra=(); [ "$ARM" = cf163_fast ] && extra=(--use-fast-kernels)
      apptainer exec --nv --cleanenv "${RUNENV[@]}" "${BINDS[@]}" "$SIF_CF163" \
        colabfold_batch "$in" "$out" "${CF_ARGS[@]}" --random-seed "$seed" "${extra[@]}" ;;
    kit_off|kit_exact|kit_fast)
      for r in $(seq 1 "$REPS"); do ln -s "$BENCH/inputs/a3m/$fold.a3m" "$in/${fold}_r$r.a3m"; done
      apptainer run --nv --cleanenv "${RUNENV[@]}" --env "COLABFOLD_OPT_JIT_ROOT=$RUN/jax" \
        --env "COLABFOLD_OPT_DATA_DIR=$BENCH/params_cf" "${BINDS[@]}" "$SIF_KIT" \
        pred --config "$(kit_config "$GPU")" --mode "${ARM#kit_}" "$in" "$out" "${CF_ARGS[@]}" --random-seed "$seed" ;;
    *) echo "unknown ARM $ARM"; return 2 ;;
  esac
}

printf "fold\tseed\ttag\trc\tstart\tend\tstatus\n" > runs.tsv
oom_at=""
for fold in $FOLDS; do
  for seed in $SEEDS; do
    tag=${fold}_s$seed
    if [ -n "$oom_at" ]; then
      printf "%s\t%s\t%s\tNA\tNA\tNA\tskipped_after_oom:%s\n" "$fold" "$seed" "$tag" "$oom_at" >> runs.tsv; continue
    fi
    t0=$(date +%s.%N)
    predict "$fold" "$seed" "$tag" > "logs/$tag.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    status=ok
    if grep -qE 'RESOURCE_EXHAUSTED|[Oo]ut of memory|OOM when allocating' "logs/$tag.log"; then status=oom; oom_at=$fold
    elif grep -q 'NOT ACTIVE' "logs/$tag.log" && [ "$ARM" != kit_off ]; then status=not_active
    elif [ $rc -ne 0 ]; then status=failed; fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$fold" "$seed" "$tag" "$rc" "$t0" "$t1" "$status" | tee -a runs.tsv
  done
done

kill $SAMPLER 2>/dev/null
{
  echo "finished=$(date -Iseconds) $(cat meta.txt)"
  echo "gpu: $(cat gpu.txt)"
  cut -f1,2,4,7 runs.tsv | tail -n +2 | awk '{print "  " $0}'
} > RESULT.txt
cat RESULT.txt
! grep -qP '\t(failed|not_active)$' runs.tsv
