#!/bin/bash

source ${RECIPE_DIR}/setup_compiler.sh

cd build

make -C ${triplet}/libgomp prefix=${PREFIX} install-toolexeclibLTLIBRARIES
rm ${PREFIX}/lib/libgomp.a ${PREFIX}/lib/libgomp.la

if [[ "$target_platform" == "linux-"* ]]; then
  rm ${PREFIX}/lib/libgomp.so.1
  rm ${PREFIX}/lib/libgomp.so
  ln -sf ${PREFIX}/lib/libgomp.so.${libgomp_ver} ${PREFIX}/lib/libgomp.so
else
  rm ${PREFIX}/lib/libgomp.dll.a
fi

# Install Runtime Library Exception
install -Dm644 ${SRC_DIR}/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/gcc-libs/RUNTIME.LIBRARY.EXCEPTION.gomp_copy
