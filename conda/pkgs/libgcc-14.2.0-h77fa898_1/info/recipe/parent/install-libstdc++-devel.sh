#!/bin/bash

source ${RECIPE_DIR}/setup_compiler.sh
set -e -x

export CHOST="${triplet}"

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
# export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/.build/${CHOST}/buildtools/bin:${SRC_DIR}/.build/tools/bin:${PATH}

pushd ${SRC_DIR}/build

make -C $CHOST/libstdc++-v3/src prefix=${PREFIX} install
make -C $CHOST/libstdc++-v3/include prefix=${PREFIX} install
make -C $CHOST/libstdc++-v3/libsupc++ prefix=${PREFIX} install

mkdir -p ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}
mkdir -p ${PREFIX}/${CHOST}/lib

if [[ "$target_platform" == "$cross_target_platform" ]]; then
    mv $PREFIX/lib/lib*.a ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}/
    if [[ "$target_platform" == linux-* ]]; then
        mv ${PREFIX}/lib/libstdc++.so* ${PREFIX}/${CHOST}/lib
    else
        rm ${PREFIX}/bin/libstdc++*.dll
    fi
else
    mv $PREFIX/${CHOST}/lib/lib*.a ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}/
fi

popd

