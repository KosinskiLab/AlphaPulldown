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
export BENCH HOST_PY DOCKQ   # the final collection (supervise.py finish) runs $HOST_PY collect.py CAMPAIGN $BENCH $DOCKQ
export SUPERVISOR_SECONDS=$(( 3 * 86400 - 3600 ))
# supervise.py needs only the standard library (>= 3.7): use the first interpreter this node really has. HOST_PY lives on
# /scratch now; the system pythons remain a fallback for a node without it.
for py in "$HOST_PY" /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.9 python3; do
  "$py" -c 'import sys; sys.exit(sys.version_info < (3, 7))' 2>/dev/null && exec "$py" "$H/af3_integration/supervise.py" "$CAMPAIGN"
done
echo "no usable python on $(hostname)"; exit 3
