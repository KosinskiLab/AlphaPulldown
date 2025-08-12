#!/bin/bash

export CFLAGS="${CFLAGS} -fPIC"
export CXXFLAGS="${CXXFLAGS} -fPIC"

if [[ "$target_platform" == linux-* ]]; then
  export CC=$GCC
fi

export cc=$CC

./configure --prefix=${PREFIX}  \
    --shared || (cat configure.log && false)
    
cat configure.log

make -j${CPU_COUNT} ${VERBOSE_AT}
if [[ "$CONDA_BUILD_CROSS_COMPILATION" != "1" && "${CROSSCOMPILING_EMULATOR}" == "" ]]; then
    make check
fi
make install

# Remove man files.
rm -rf $PREFIX/share
