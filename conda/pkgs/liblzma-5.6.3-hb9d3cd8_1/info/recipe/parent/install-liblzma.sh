#!/bin/bash

set -euxo pipefail

mkdir -p $PREFIX/lib

if [[ "${target_platform}" == osx-* ]]; then
  cp ./src/liblzma/.libs/liblzma.*.dylib $PREFIX/lib
else
  cp ./src/liblzma/.libs/liblzma.so.* $PREFIX/lib
fi
