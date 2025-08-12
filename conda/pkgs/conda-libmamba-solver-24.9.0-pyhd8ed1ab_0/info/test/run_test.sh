

set -ex



conda create -n test --dry-run scipy --solver=libmamba
CONDA_SOLVER=libmamba conda create -n test --dry-run scipy
exit 0
