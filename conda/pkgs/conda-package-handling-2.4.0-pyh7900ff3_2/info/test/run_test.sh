

set -ex



pytest -v --cov=conda_package_handling --color=yes tests/
exit 0
