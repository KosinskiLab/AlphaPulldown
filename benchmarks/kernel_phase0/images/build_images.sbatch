#!/bin/bash
# Build or pull the four images the benchmark runs. Idempotent: an image already present is
# kept, so a rerun only fills what is missing. Writes $BENCH/images/RESULT.txt last.
#SBATCH -J kb0-images
#SBATCH -p build-el8,build-el10
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00

set -uo pipefail
source "${CODE:?CODE must point at benchmarks/kernel_phase0}/bench.env"
IMG=$BENCH/images
mkdir -p "$IMG/tmp" "$IMG/cache"
export APPTAINER_TMPDIR=$IMG/tmp APPTAINER_CACHEDIR=$IMG/cache
echo "node=$(hostname) start=$(date -Iseconds)"

status=0
step() {  # $1 = output sif; rest = command that writes $1.partial
  local out=$1; shift
  if [ -s "$out" ]; then echo "have $(basename "$out")"; return 0; fi
  rm -f "$out.partial"
  echo "building $(basename "$out"): $*"
  if "$@" && [ -s "$out.partial" ]; then mv "$out.partial" "$out"; echo "built $(basename "$out")"
  else echo "FAILED $(basename "$out")"; rm -f "$out.partial"; status=1; fi
}

# The kit's source at the pinned commit.
UBM=$BENCH/src/ubm
if [ ! -d "$UBM/.git" ]; then git clone --quiet "$UBM_REPO" "$UBM" || status=1; fi
git -C "$UBM" checkout --quiet "$UBM_COMMIT" && echo "ubm at $(git -C "$UBM" rev-parse HEAD)" || status=1

step "$SIF_AP_AF2" apptainer pull "$SIF_AP_AF2.partial" "docker://kosinskilab/alphafold2:$AP_VERSION"
step "$SIF_AP_AF3" apptainer pull "$SIF_AP_AF3.partial" "docker://kosinskilab/alphafold3:$AP_VERSION"
step "$SIF_CF163" apptainer build --fakeroot "$SIF_CF163.partial" "$CODE/images/colabfold-1.6.3.def"
step "$SIF_KIT" apptainer build --fakeroot --build-arg "UBM=$UBM" "$SIF_KIT.partial" "$CODE/images/colabfold-kit.def"

{
  echo "exit=$status finished=$(date -Iseconds) node=$(hostname) job=${SLURM_JOB_ID:-none}"
  echo "ubm_commit=$(git -C "$UBM" rev-parse HEAD 2>/dev/null)"
  for sif in "$SIF_AP_AF2" "$SIF_AP_AF3" "$SIF_CF163" "$SIF_KIT"; do
    [ -s "$sif" ] && echo "$(basename "$sif") $(stat -c %s "$sif") bytes $(stat -c %y "$sif")" || echo "$(basename "$sif") MISSING"
  done
  apptainer exec "$SIF_CF163" sh -c 'grep -iE "^(colabfold|alphafold-colabfold|jax|jaxlib|jax-cuda12-plugin)==" /opt/pip-freeze.txt' 2>/dev/null
  apptainer exec "$SIF_KIT" sh -c 'grep -iE "^(colabfold|alphafold-colabfold|jax|jaxlib|jax-cuda12-plugin)==" /opt/pip-freeze.txt' 2>/dev/null
} > "$IMG/RESULT.txt"
cat "$IMG/RESULT.txt"
exit $status
