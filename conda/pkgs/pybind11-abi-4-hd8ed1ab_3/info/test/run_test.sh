

set -ex



if [ $(grep "#define PYBIND11_INTERNALS_VERSION" include/pybind11/detail/internals.h | cut -d' ' -f3) != "4" ]; then exit 1; fi
exit 0
