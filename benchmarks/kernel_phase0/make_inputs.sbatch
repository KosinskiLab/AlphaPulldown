#!/bin/bash
# Freeze the inputs under $BENCH/inputs and prove ColabFold and AlphaPulldown see the same MSA.
#   1. copy the feature pickles and native structures the folds need (sources may change later)
#   2. lay out a ColabFold parameters root of symlinks to the AF2 multimer v3 params
#   3. make_inputs.py in the AlphaPulldown AF2 image: a3m + AlphaPulldown's final features
#   4. verify_inputs.py in both ColabFold images: features must match AlphaPulldown's exactly
# Writes $BENCH/inputs/RESULT.txt last; exit 1 if any fold failed or differs.
#SBATCH -J kb0-inputs
#SBATCH -p htc-el10,htc-el8
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00

set -uo pipefail
source "${CODE:?CODE must point at benchmarks/kernel_phase0}/bench.env"
IN=$BENCH/inputs
mkdir -p "$IN/features" "$IN/natives" "$BENCH/params_cf/params"
echo "node=$(hostname) start=$(date -Iseconds)"
status=0

chains=$( { grep -v '^#' "$CODE/folds.tsv" | cut -f3; cat "$SRC_ACCURACY_FOLDS"; } | tr '+ ' '\n\n' | grep . | sort -u)
for chain in $chains; do
  cp -n "$SRC_FEATURES/$chain".pkl* "$IN/features/" || { echo "missing features for $chain"; status=1; }
  cp -n "$SRC_FEATURES/${chain}"_feature_metadata_*.json* "$IN/features/" 2>/dev/null || true
done
for pdb in $(tr '+ ' '\n\n' < "$SRC_ACCURACY_FOLDS" | grep . | cut -d_ -f1 | sort -u); do
  cp -n "$SRC_NATIVES/$pdb.cif" "$IN/natives/" || { echo "missing native $pdb"; status=1; }
done
(cd "$IN" && sha256sum features/* natives/* > SHA256SUMS)

for i in 1 2 3 4 5; do
  ln -sfn "$AF2_DATA/params/params_model_${i}_multimer_v3.npz" "$BENCH/params_cf/params/params_model_${i}_multimer_v3.npz"
done
touch "$BENCH/params_cf/params/download_complexes_multimer_v3_finished.txt"

binds=(--bind "$BENCH" --bind "$CODE" --bind /g/alphafold)
apptainer exec --cleanenv --env JAX_PLATFORMS=cpu "${binds[@]}" "$SIF_AP_AF2" \
  python "$CODE/make_inputs.py" --folds-tsv "$CODE/folds.tsv" --accuracy-folds "$SRC_ACCURACY_FOLDS" \
  --features "$IN/features" --out "$IN" > "$IN/make_inputs.log" 2>&1 || status=1
tail -3 "$IN/make_inputs.log"

for pair in "cf163:$SIF_CF163" "kit:$SIF_KIT"; do
  label=${pair%%:*} sif=${pair#*:}
  apptainer exec --cleanenv --env JAX_PLATFORMS=cpu "${binds[@]}" "$sif" \
    python "$CODE/verify_inputs.py" --inputs "$IN" --label "$label" > "$IN/verify_$label.tsv" 2>"$IN/verify_$label.err" || status=1
  tail -1 "$IN/verify_$label.tsv"
done

{
  echo "exit=$status finished=$(date -Iseconds) job=${SLURM_JOB_ID:-none}"
  echo "folds: $(grep -c '"name"' "$IN/folds.json" 2>/dev/null || echo 0) in folds.json"
  grep -h 'FAILED' "$IN/make_inputs.log" 2>/dev/null
  grep -h 'differ' "$IN"/verify_*.tsv 2>/dev/null
  echo "logs: $IN/make_inputs.log $IN/verify_*.tsv"
} > "$IN/RESULT.txt"
cat "$IN/RESULT.txt"
exit $status
