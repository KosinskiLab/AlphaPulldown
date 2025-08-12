#!/usr/bin/env bash
#
cmake -B build-static/ \
    -G Ninja \
    -D BUILD_SHARED_LIBS=OFF \
    -D YAML_BUILD_SHARED_LIBS=OFF \
    -D YAML_CPP_BUILD_TESTS=OFF \
    ${CMAKE_ARGS}
cmake --build build-static/ --parallel ${CPU_COUNT} --verbose
cmake --install build-static/
