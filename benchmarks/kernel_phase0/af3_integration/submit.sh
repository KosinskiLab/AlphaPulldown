#!/bin/bash
# Freeze committed sources and submit all workers plus an afterany collector.
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$CODE/bench.env"
MODE=${1:-pilot}; shift || true
AP_REPO=${AP_REPO:-/scratch/dima/ap-af3-fused-triangle}
FORK_REPO=${FORK_REPO:-/scratch/dima/af3-fused-triangle}
# The fork commit every arm is compared against: fork main updated to AlphaFold 3 v3.0.3, the parent of the fused-triangle
# work and the alphafold3 that the AlphaPulldown 2.9.0 image ships (its model/ and jax/ are byte-identical to it).
BASELINE_COMMIT=${BASELINE_COMMIT:-86b9ea3}
# On an image with another alphafold3 (e.g. the AlphaFold 3 v3.0.4 / jax 0.10.2 image) set BASELINE_COMMIT to the fork commit
# that image ships without the fused-triangle work (v3.0.4: 758d767, fork main after KosinskiLab/alphafold3#3), and SIF_AP_AF3
# to that image: `baseline` must be the same alphafold3 as `off`, minus the kernels, for the identity check to mean anything.
# The layers suite runs gate0_layers.py against the kit's frozen core; gate0/submit_gate0.sh documents the export.
KIT=$BENCH/gate0/src/kit_f4f62fa6/common/opt_core
[ -d "$KIT" ] || { echo "missing frozen kit core $KIT (git archive f4f62fa6 common/opt_core, see gate0/submit_gate0.sh)" >&2; exit 1; }
HROOT=$(git -C "$CODE" rev-parse --show-toplevel)
CAMPAIGN=$BENCH/af3_integration/$(date +%Y%m%d_%H%M%S)_$MODE
SNAPSHOT=$CAMPAIGN/source
mkdir -p "$SNAPSHOT" "$CAMPAIGN/logs"
for component in ap fork harness; do
  case $component in ap) repo=$AP_REPO ;; fork) repo=$FORK_REPO ;; harness) repo=$HROOT ;; esac
  mkdir "$SNAPSHOT/$component"
  git -C "$repo" rev-parse HEAD > "$SNAPSHOT/${component}_COMMIT"
  case $component in
    ap) git -C "$repo" archive HEAD alphapulldown test/test_data/features/rna.json test/test_data/features/ligand.json test/test_data/features/af3_features/mixed/test_protein_1_af3_input.json | tar -x -C "$SNAPSHOT/ap" ;;
    fork) git -C "$repo" archive HEAD src | tar -x -C "$SNAPSHOT/fork" ;;
    harness) git -C "$repo" archive HEAD benchmarks/kernel_phase0 | tar -x --strip-components=2 -C "$SNAPSHOT/harness" ;;
  esac
done
mkdir "$SNAPSHOT/baseline"
git -C "$FORK_REPO" rev-parse "$BASELINE_COMMIT^{commit}" > "$SNAPSHOT/baseline_COMMIT"
# run.sbatch binds both directories for the `baseline` arm, so it runs stock code only.
git -C "$FORK_REPO" archive "$BASELINE_COMMIT" src/alphafold3/model src/alphafold3/jax | tar -x -C "$SNAPSHOT/baseline"
cp "$BENCH/inputs/folds.json" "$SNAPSHOT/inputs.json"
"$HOST_PY" "$SNAPSHOT/harness/af3_integration/prepare_inputs.py" "$SNAPSHOT"
# Keep the CPU work off the login node's other users; inside a SLURM allocation CPUs 0-3 may not be ours, and the
# allocation already confines it.
PIN=(); taskset -c 0-3 true 2>/dev/null && PIN=(taskset -c 0-3)
"${PIN[@]}" apptainer exec --cleanenv --bind "$BENCH" "$SIF_AP_AF3" \
  python "$SNAPSHOT/harness/af3_integration/prepare_template.py" "$SNAPSHOT" "$BENCH/inputs/natives/8B2R.cif"
find "$SNAPSHOT" -type f -not -path '*/__pycache__/*' -exec sha256sum {} + > "$CAMPAIGN/SHA256SUMS"
export CAMPAIGN SNAPSHOT
all=""
submit() {
  local gpu=$1 suite=$2 arm=$3 folds=$4 seeds=$5 reps=$6 dep=${7:-} jid tile
  case $gpu in a100) tile=8.0 ;; a40|3090) tile=8.6 ;; l40s) tile=8.9 ;; *) tile=9.0 ;; esac
  jid=$(sbatch --parsable --qos="${AF3I_QOS:-high}" --kill-on-invalid-dep=yes -p "$(gpu_partition "$gpu")" --gres="$(gpu_gres "$gpu")" -t "${WALLTIME:-05:00:00}" \
    ${dep:+--dependency=afterok:$dep} -J "af3i-$suite-$gpu-$arm" -o "$CAMPAIGN/logs/%x_%j.log" \
    --export=ALL,GPU="$gpu",SUITE="$suite",ARM="$arm",FOLDS="$folds",SEEDS="$seeds",REPS="$reps",TILE_CC="$tile" \
    "$SNAPSHOT/harness/af3_integration/run.sbatch")
  printf '%s\t%s\t%s\t%s\n' "$jid" "$gpu" "$suite" "$arm" >> "$CAMPAIGN/jobs.tsv"
  all="$all:$jid"
  LAST_JOB=$jid
  echo "$gpu/$suite/$arm: $jid"
}
for gpu in ${*:-a40 h100 a100 l40s 3090 rtx6000}; do
  submit "$gpu" layers hooks '' 0 1
  layers=$LAST_JOB
  submit "$gpu" identity paired s0164 0 2 "$layers"
  pilot=$LAST_JOB
  submit "$gpu" cli auto s0599 0 2 "$pilot"
  if [ "$MODE" = full ]; then
    for arm in off on; do
      submit "$gpu" speed "$arm" 's0164 s0357 s0599 s0896 s1309 s1792 s2546 r3584 r5376' 0 2 "$pilot"
    done
    case $gpu in a40|h100|a100)
      for arm in trimul attention; do submit "$gpu" speed "$arm" 's0896 s1792' 0 2 "$pilot"; done
      folds=$("$HOST_PY" -c 'import json,sys; print(" ".join(r["name"] for r in json.load(open(sys.argv[1])) if "accuracy" in r["suites"]))' "$SNAPSHOT/inputs.json")
      submit "$gpu" accuracy off "$folds" '0 1 2' 1 "$pilot"
      submit "$gpu" accuracy on "$folds" '0 1' 1 "$pilot"
      for arm in off on; do submit "$gpu" inputs "$arm" 'protein_rna protein_ligand populated_template' 0 2 "$pilot"; done ;;
    esac
  fi
  if [ "$MODE" = followup ]; then                # seed sweeps of folds the paired accuracy rule flagged (both arms)
    submit "$gpu" accuracy off "${FOLLOWUP_FOLDS:-acc_8JTK acc_9HH5}" "${FOLLOWUP_SEEDS:-0 1 2 3 4 5 6 7 8 9}" 1 "$pilot"
    submit "$gpu" accuracy on "${FOLLOWUP_FOLDS:-acc_8JTK acc_9HH5}" "${FOLLOWUP_SEEDS:-0 1 2 3 4 5 6 7 8 9}" 1 "$pilot"
  fi
done
collector=$(sbatch --parsable --qos=high --dependency="afterany$all" -J af3i-collect -o "$CAMPAIGN/logs/collect_%j.log" --export=ALL \
  "$SNAPSHOT/harness/af3_integration/collect.sbatch")
printf '%s\n' "$collector" > "$CAMPAIGN/COLLECTOR_JOB"
printf 'campaign=%s\ncollector=%s\nimage_af3=%s\nbaseline=%s\n' "$CAMPAIGN" "$collector" "$SIF_AP_AF3" "$BASELINE_COMMIT" | tee "$CAMPAIGN/SUBMITTED.txt"
# Retry infrastructure failures (broken node, NODE_FAIL, CUDA start-up, one timeout) and re-collect when everything is final.
"$CODE/af3_integration/start_supervisor.sh" "$CAMPAIGN" | tee -a "$CAMPAIGN/SUBMITTED.txt"
