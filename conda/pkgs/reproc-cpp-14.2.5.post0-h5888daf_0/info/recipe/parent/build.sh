#!/usr/bin/env bash

set -euo pipefail

cmake -B build-dyn/ \
    -G Ninja \
    -D REPROC_TEST=OFF \
    -D BUILD_SHARED_LIBS=ON \
    -D REPROC++=ON \
    ${CMAKE_ARGS}
cmake --build build-dyn/ --parallel ${CPU_COUNT} --verbose

cmake -B build-static/ \
    -G Ninja \
    -D REPROC_TEST=OFF \
    -D BUILD_SHARED_LIBS=OFF \
    -D REPROC++=ON \
    ${CMAKE_ARGS}
cmake --build build-static/ --parallel ${CPU_COUNT} --verbose
