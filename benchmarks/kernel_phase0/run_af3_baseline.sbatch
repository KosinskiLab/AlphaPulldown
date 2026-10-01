#!/bin/bash
# AF3 baseline on one card: AlphaPulldown's AF3 backend with DeepMind's weights, stock
# settings (10 recycles, 5 diffusion samples, one seed), from the same AF2 feature pickles.
# No optimized arm exists yet; these numbers are what a Phase 2 port would be measured
# against, and the reach probes find the largest fold the card holds without unified memory.
# Same process-per-fold, smallest-first, stop-at-first-OOM scheme as run_af2_arm.sbatch.
#
# Env in: CODE, GPU, SUITE (speed|smoke). Out: $BENCH/runs/$SUITE/$GPU/ap_af3/
#SBATCH -J kb0-af3
#SBATCH -c 8
#SBATCH --mem=96G

set -uo pipefail
source "${CODE:?}/bench.env"
: "${GPU:?}" "${SUITE:?}"
ARM=ap_af3
RUN=$BENCH/runs/$SUITE/$GPU/$ARM
mkdir -p "$RUN/logs" "$RUN/out" "$RUN/in" "$RUN/jax"
cd "$RUN" || exit 1

if ! nvidia-smi -L > /dev/null 2>&1; then   # a broken driver must not produce numbers
  echo "node_unusable node=$(hostname) job=${SLURM_JOB_ID:-none}: $(nvidia-smi 2>&1 | head -1)" | tee RESULT.txt
  exit 3
fi
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null \
           || nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm \
  --format=csv,noheader -i "$GPU_UUID" > gpu.txt
mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_UUID")
case $SUITE in
  speed) suites="speed"; [ "${mem_mib:-0}" -ge 40000 ] && suites="speed af3reach"; REPS=$AF3_REPS ;;
  smoke) suites="smoke"; REPS=2 ;;
  *) echo "unknown SUITE $SUITE"; exit 2 ;;
esac
FOLDS=$("$HOST_PY" -c "
import json; r = json.load(open('$BENCH/inputs/folds.json')); want = '$suites'.split()
pick = [x for x in r if any(s in x['suites'] for s in want)] if want != ['smoke'] else [x for x in r if x['name'] == 's0164']
print(' '.join(x['name'] for x in sorted(pick, key=lambda x: x['tokens'])))")
echo "node=$(hostname) job=${SLURM_JOB_ID:-none} arm=$ARM gpu=$GPU suite=$SUITE folds=[$FOLDS] reps=$REPS seed=$SEED" | tee meta.txt
cat gpu.txt
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -i "$GPU_UUID" -lms 500 > gpu_mem.csv 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
sleep 2

BINDS=(--bind "$BENCH" --bind "$CODE" --bind /g/alphafold)
RUNENV=($MEM_ENV --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}")

printf "fold\tseed\ttag\trc\tstart\tend\tstatus\n" > runs.tsv
oom_at=""
for fold in $FOLDS; do
  tag=${fold}_s$SEED in=$RUN/in/${fold}_s$SEED out=$RUN/out/${fold}_s$SEED
  if [ -n "$oom_at" ]; then
    printf "%s\t%s\t%s\tNA\tNA\tNA\tskipped_after_oom:%s\n" "$fold" "$SEED" "$tag" "$oom_at" >> runs.tsv; continue
  fi
  rm -rf "$in" "$out"; mkdir -p "$in" "$out"
  spec=$("$HOST_PY" -c "import json;print(next(x['fold'] for x in json.load(open('$BENCH/inputs/folds.json')) if x['name']=='$fold'))")
  for r in $(seq 1 "$REPS"); do
    printf '{"job_id": "%s_r%d", "input": "%s", "output_directory": "%s/%s_r%d"}\n' "$fold" "$r" "$spec" "$out" "$fold" "$r"
  done > "$in/manifest.jsonl"
  t0=$(date +%s.%N)
  apptainer exec --nv --cleanenv "${RUNENV[@]}" "${BINDS[@]}" "$SIF_AP_AF3" \
    python -m alphapulldown.scripts.run_structure_prediction_batch --manifest "$in/manifest.jsonl" \
    --data_directory "$AF3_WEIGHTS" --features_directory "$BENCH/inputs/features" --fold_backend alphafold3 \
    --random_seed "$SEED" --protein_delimiter + --jax_compilation_cache_dir "$RUN/jax" > "logs/$tag.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  status=ok
  if grep -qE 'RESOURCE_EXHAUSTED|[Oo]ut of memory|OOM when allocating' "logs/$tag.log"; then status=oom; oom_at=$fold
  elif [ $rc -ne 0 ]; then status=failed; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$fold" "$SEED" "$tag" "$rc" "$t0" "$t1" "$status" | tee -a runs.tsv
done

kill $SAMPLER 2>/dev/null
{
  echo "finished=$(date -Iseconds) $(cat meta.txt)"
  echo "gpu: $(cat gpu.txt)"
  cut -f1,2,4,7 runs.tsv | tail -n +2 | awk '{print "  " $0}'
} > RESULT.txt
cat RESULT.txt
! grep -qP '\tfailed$' runs.tsv
