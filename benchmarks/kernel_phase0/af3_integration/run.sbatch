#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=96G
set -uo pipefail
ulimit -c 0
: "${SNAPSHOT:?}" "${SUITE:?}" "${GPU:?}" "${ARM:?}" "${CAMPAIGN:?}"
source "$SNAPSHOT/harness/bench.env"
RUN=$CAMPAIGN/runs/$GPU/$SUITE/$ARM/${SLURM_JOB_ID:?}
mkdir -p "$RUN"
cd "$RUN" || exit 1
finish() {
  rc=$?
  [ -z "${sampler:-}" ] || kill "$sampler" 2>/dev/null || true
  printf 'exit_code=%s\njob=%s\nrun=%s\nsnapshot=%s\n' "$rc" "$SLURM_JOB_ID" "$RUN" "$SNAPSHOT" > RESULT.txt
}
trap finish EXIT
nvidia-smi -L > gpu.txt || exit 3
ENV=($MEM_ENV_AF3 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" --env PYTHONDONTWRITEBYTECODE=1
     --env JAX_LOG_COMPILES=true --env JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
     --env "PYTHONPATH=$SNAPSHOT/ap:$SNAPSHOT/harness/af3_integration" --env "MPLCONFIGDIR=$RUN/mpl")
BINDS=(--bind "$BENCH" --bind /g/alphafold --bind "$SNAPSHOT" --bind "$CAMPAIGN")
# The image's alphafold3 package is the fork at 86b9ea3. An arm replaces its model/ (module hooks, adapters) and jax/ (the
# fused_triangle kernels and dispatch) with one frozen source: the fork under test, or 86b9ea3 itself for `baseline`.
AF3_SITE=/opt/venv/lib/python3.12/site-packages/alphafold3
if [ "$SUITE" = layers ]; then
  src=$SNAPSHOT/fork/src/alphafold3
  apptainer exec --nv --cleanenv "${ENV[@]}" "${BINDS[@]}" --bind "$src/model:$AF3_SITE/model" --bind "$src/jax:$AF3_SITE/jax" "$SIF_AP_AF3" \
    python "$SNAPSHOT/harness/gate0/gate0_layers.py" --integrated --out "$RUN/cells.jsonl" \
    --kit "$BENCH/gate0/src/kit_f4f62fa6/common/opt_core" --tile_cc "${TILE_CC:-9.0}" \
    --sizes 256,1024 --asym_sizes 256 --dm_weights "$AF3_WEIGHTS" --dm_sizes 512 \
    --negative_control --no_timing > layers.log 2>&1 || exit $?
  "$HOST_PY" "$SNAPSHOT/harness/af3_integration/check_layers.py" cells.jsonl > checks.json || exit $?
  exit 0
fi
arms="$ARM"
[ "$SUITE" = identity ] && arms="baseline off on"
fail=0
for current in $arms; do
  src=$SNAPSHOT/fork/src/alphafold3
  mode=on
  case $current in baseline) src=$SNAPSHOT/baseline/src/alphafold3; mode=off ;; off) mode=off ;; auto) mode=auto ;; esac
  for seed in ${SEEDS:-0}; do
    for fold in $FOLDS; do
      tag=${current}_${fold}_s${seed}
      mkdir -p "$tag"
      # Every process has its own outputs; identity deliberately shares one cache.
      cache=$RUN/cache_$tag
      [ "$SUITE" = identity ] && cache=$RUN/shared_cache
      "$HOST_PY" "$SNAPSHOT/harness/af3_integration/manifest.py" "$SNAPSHOT/inputs.json" "$fold" "$RUN/$tag" "${REPS:-2}" > "$tag/manifest.jsonl" || exit $?
      extra=()
      [ "$SUITE" = identity ] && extra+=(--env XLA_FLAGS=--xla_gpu_deterministic_ops=true)
      nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$tag/gpu_memory_mib.txt" 2> "$tag/gpu_memory_error.txt" &
      sampler=$!
      apptainer exec --nv --cleanenv "${ENV[@]}" "${extra[@]}" "${BINDS[@]}" --bind "$src/model:$AF3_SITE/model" --bind "$src/jax:$AF3_SITE/jax" \
        --env "AF3_TEST_ARM=$current" --env "AF3_TEST_RECORD=$RUN/$tag/forward.jsonl" "$SIF_AP_AF3" \
        python "$SNAPSHOT/harness/af3_integration/launch.py" --manifest "$RUN/$tag/manifest.jsonl" \
        --data_directory "$AF3_WEIGHTS" --features_directory "$BENCH/inputs/features" --fold_backend alphafold3 \
        --random_seed "$seed" --protein_delimiter + --fast_kernels "$mode" \
        --jax_compilation_cache_dir "$cache" > "$tag/predict.log" 2>&1
      rc=$?
      kill "$sampler" 2>/dev/null || true
      wait "$sampler" 2>/dev/null || true
      sampler=""
      echo "$rc" > "$tag/exit_code"
      if [ "$rc" -ne 0 ]; then
        fail=1
        # Stop the remaining (larger) folds of this arm and seed; the next seed or arm still runs. The ladder is ordered by
        # size and ends with the capacity probes, so an expected OOM there costs no smaller fold.
        break
      fi
    done
  done
done
[ "$SUITE" = identity ] && "$HOST_PY" "$SNAPSHOT/harness/af3_integration/check_identity.py" "$RUN" > identity.json || { [ "$SUITE" != identity ] || fail=1; }
exit "$fail"
