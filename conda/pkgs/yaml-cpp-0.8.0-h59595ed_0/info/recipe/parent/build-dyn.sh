#!/usr/bin/env bash

cmake -B build-dyn/ \
    -G Ninja \
    -D BUILD_SHARED_LIBS=ON \
    -D YAML_BUILD_SHARED_LIBS=ON \
    -D YAML_CPP_BUILD_TESTS=OFF \
    ${CMAKE_ARGS} 
cmake --build build-dyn/ --parallel ${CPU_COUNT} --verbose
cmake --install build-dyn/
