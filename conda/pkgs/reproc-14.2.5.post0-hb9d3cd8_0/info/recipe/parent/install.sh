#!/usr/bin/env bash

set -euo pipefail

if [[ "${PKG_NAME}" == *static ]]; then
    BUILD_DIR="build-static"
else
    BUILD_DIR="build-dyn"
fi
if [[ "${PKG_NAME}" == reproc-cpp* ]]; then
    COMPONENT="reproc++"
else
    COMPONENT="reproc"
fi

cmake --install "${BUILD_DIR}" --component ${COMPONENT}-development
cmake --install "${BUILD_DIR}" --component ${COMPONENT}-runtime
