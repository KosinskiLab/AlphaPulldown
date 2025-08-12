

set -ex



python -c "from importlib.util import find_spec; assert find_spec('_ruamel_yaml')"
exit 0
