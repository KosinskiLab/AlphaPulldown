#!/bin/bash

source ${RECIPE_DIR}/setup_compiler.sh
set -e -x

mkdir -p ${PREFIX}/lib

pushd ${PREFIX}/lib/

if [[ "${TARGET}" != *mingw* ]]; then
  ln -s libgomp.so.${libgomp_ver} libgomp.so.${libgomp_ver:0:1}
fi
popd
