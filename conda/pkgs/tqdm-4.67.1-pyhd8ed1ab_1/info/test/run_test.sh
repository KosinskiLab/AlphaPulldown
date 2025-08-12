

set -ex



pip check
tqdm --help
tqdm -v
pytest -k "not tests_perf and not tests_tk"
exit 0
