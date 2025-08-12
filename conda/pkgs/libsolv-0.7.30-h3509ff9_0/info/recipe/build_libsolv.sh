#!/usr/bin/env bash

set -euo pipefail

if [[ $PKG_NAME == "libsolv" ]]; then

    cmake -B build/ \
        -G Ninja \
        -D ENABLE_CONDA=ON \
        -D MULTI_SEMANTICS=ON \
        -D DISABLE_SHARED=OFF \
        -D ENABLE_STATIC=OFF \
        ${CMAKE_ARGS}
    cmake --build build/ --parallel ${CPU_COUNT} --verbose
    cmake --install build/

elif [[ $PKG_NAME == "libsolv-static" ]]; then

    cmake -B build_static/ \
        -G Ninja \
        -D ENABLE_CONDA=ON \
        -D MULTI_SEMANTICS=ON \
        -D DISABLE_SHARED=ON \
        -D ENABLE_STATIC=ON \
        ${CMAKE_ARGS}
    cmake --build build_static/ --parallel ${CPU_COUNT} --verbose
    cmake --install build_static/

fi
