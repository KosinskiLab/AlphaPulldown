#!/bin/bash
# Gate 0b on one card and one arm: end-to-end AF3 through AlphaPulldown main (a frozen export overlaid on the 2.9.0 image),
# DeepMind weights, one seed, AF3 defaults. ARM=stock runs the fork's modules; ARM=fused binds the kit's fused triangle kernels
# (gate0b/fused_af3.py, per-card dispatch). Both arms run through gate0b/launch.py with the same image, jax, tokamax and a fresh
# persistent compile cache. One process per fold, two predictions: rep 1 compiles, rep 2 is the warm call that is timed, and
# must log no compile (JAX_LOG_COMPILES). Smallest fold first; stop at the first OOM.
# Env in: CODE, GPU, ARM (stock|fused), SRC (frozen dir: ap/alphapulldown, gate0b/), FOLDS. Out: $BENCH/gate0b/runs/$GPU/$ARM/
#SBATCH -J kb0-gate0b
#SBATCH -c 8
#SBATCH --mem=96G

set -uo pipefail
source "${CODE:?}/bench.env"
: "${GPU:?}" "${ARM:?}" "${SRC:?}" "${FOLDS:?}"
case $ARM in stock) FUSED=0 ;; fused) FUSED=1 ;; *) echo "unknown arm $ARM"; exit 2 ;; esac
RUN=$BENCH/gate0b/runs${RUN_TAG:+_$RUN_TAG}/$GPU/$ARM
rm -rf "$RUN"; mkdir -p "$RUN/logs" "$RUN/out" "$RUN/in" "$RUN/jaxcache"
cd "$RUN" || exit 1
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "node_unusable node=$(hostname) job=${SLURM_JOB_ID:-none}" | tee RESULT.txt; exit 3
fi
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total --format=csv,noheader | head -1 > gpu.txt
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} arm=$ARM gpu=$GPU folds=[$FOLDS] src=$SRC ap=$(cat "$SRC/AP_COMMIT") seed=$SEED" | tee meta.txt
cat gpu.txt

KIT=$BENCH/gate0/src/kit_f4f62fa6/common/opt_core
RUNENV=($MEM_ENV_AF3 --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" --env "MPLCONFIGDIR=$RUN/mpl" --env JAX_LOG_COMPILES=true
        --env JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none --env "PYTHONPATH=$SRC/ap:$SRC/gate0b:$KIT" --env "G0B_FUSED=$FUSED")
for v in G0B_TRIMUL G0B_ATT G0B_ATT_MAX_N; do [ -n "${!v:-}" ] && RUNENV+=(--env "$v=${!v}"); done
printf "fold\tseed\trc\tstart\tend\tstatus\n" > runs.tsv
oom_at=""
for fold in $FOLDS; do
  if [ -n "$oom_at" ]; then printf "%s\t%s\tNA\tNA\tNA\tskipped_after:%s\n" "$fold" "$SEED" "$oom_at" >> runs.tsv; continue; fi
  in=$RUN/in/$fold out=$RUN/out/$fold
  mkdir -p "$in" "$out"
  spec=$("$HOST_PY" -c "import json;print(next(x['fold'] for x in json.load(open('$BENCH/inputs/folds.json')) if x['name']=='$fold'))")
  for r in 1 2; do
    printf '{"job_id": "%s_r%d", "input": "%s", "output_directory": "%s/%s_r%d"}\n' "$fold" "$r" "$spec" "$out" "$fold" "$r"
  done > "$in/manifest.jsonl"
  t0=$(date +%s.%N)
  apptainer exec --nv --cleanenv "${RUNENV[@]}" --bind "$BENCH" --bind "$CODE" --bind /g/alphafold "$SIF_AP_AF3" \
    python "$SRC/gate0b/launch.py" --manifest "$in/manifest.jsonl" --data_directory "$AF3_WEIGHTS" \
    --features_directory "$BENCH/inputs/features" --fold_backend alphafold3 --random_seed "$SEED" --protein_delimiter + \
    --jax_compilation_cache_dir "$RUN/jaxcache" > "logs/$fold.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  status=ok
  if grep -qE 'RESOURCE_EXHAUSTED|[Oo]ut of memory|OOM when allocating' "logs/$fold.log"; then status=oom; oom_at=$fold
  elif [ $rc -ne 0 ]; then status=failed; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$fold" "$SEED" "$rc" "$t0" "$t1" "$status" | tee -a runs.tsv
  if [ "$status" = failed ] && ! grep -q '	ok$' runs.tsv; then oom_at="setup_failure:$fold"; fi
done
{ echo "finished=$(date -Iseconds) $(cat meta.txt)"; echo "gpu: $(cat gpu.txt)"; tail -n +2 runs.tsv | cut -f1,3,6 | sed 's/^/  /'; } > RESULT.txt
cat RESULT.txt
