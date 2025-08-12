

set -ex



pip check
distro --help
pytest -vvv --capture=tee-sys tests
exit 0
