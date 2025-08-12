#!/bin/bash

set -exuo pipefail

BROTLI_CFLAGS="-O3"

# Build both static and shared libraries
cmake ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_C_FLAGS=$BROTLI_CFLAGS \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      .

ninja
if [[ "${CONDA_BUILD_CROSS_COMPILATION:-}" != "1" ]]; then
  ctest -V
fi
