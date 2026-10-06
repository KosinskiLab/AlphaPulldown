#!/bin/bash
# Attach the infrastructure-retry supervisor to a campaign: freeze the committed harness into $CAMPAIGN/supervisor/harness,
# then queue SUPERVISORS (default 3) singleton runs of supervise.sbatch on CPU partitions (each up to 3 days).
#   start_supervisor.sh CAMPAIGN
set -euo pipefail
CAMPAIGN=$(cd "${1:?campaign directory}" && pwd)
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HROOT=$(git -C "$CODE" rev-parse --show-toplevel)
S=$CAMPAIGN/supervisor
mkdir -p "$S/harness"
if [ ! -f "$S/harness_COMMIT" ]; then
  git -C "$HROOT" archive HEAD benchmarks/kernel_phase0 | tar -x --strip-components=2 -C "$S/harness"
  git -C "$HROOT" rev-parse HEAD > "$S/harness_COMMIT"
fi
name=af3i-supervise-$(basename "$CAMPAIGN")
for i in $(seq 1 "${SUPERVISORS:-3}"); do
  jid=$(sbatch --parsable -p htc-el8,htc-el10 --dependency=singleton -J "$name" -o "$S/supervise_%j.log" \
        --export=ALL,CAMPAIGN="$CAMPAIGN" "$S/harness/af3_integration/supervise.sbatch")
  echo "$jid" >> "$S/SUPERVISOR_JOBS"
  echo "supervisor $i: $jid"
done
