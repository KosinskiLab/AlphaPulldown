

set -ex



pip check
mypy -p platformdirs --ignore-missing-imports
exit 0
