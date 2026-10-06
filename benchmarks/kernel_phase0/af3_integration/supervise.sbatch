#!/bin/bash
# One supervisor run (supervise.py) for $CAMPAIGN, from the frozen copy in $CAMPAIGN/supervisor/harness. start_supervisor.sh
# queues several of these under one name with --dependency=singleton: each starts when the previous one ends (finished,
# walltime, node failure) and resumes from supervisor/tasks.json; once SUPERVISOR_RESULT.txt exists they exit at once.
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
set -uo pipefail
: "${CAMPAIGN:?}"
H=$CAMPAIGN/supervisor/harness
source "$H/bench.env"
export SUPERVISOR_SECONDS=$(( 3 * 86400 - 3600 ))
exec "$HOST_PY" "$H/af3_integration/supervise.py" "$CAMPAIGN"
