#!/bin/bash
# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/libtool/build-aux/config.* .
set -ex
./configure --prefix=${PREFIX} \
            --host=${HOST} \
            --disable-static \
            CFLAGS="${CFLAGS} -I${PREFIX}/include" \
            LDFLAGS="${LDFLAGS} -L${PREFIX}/lib"
make -j ${CPU_COUNT} ${VERBOSE_AT}
make install
if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]]; then
make check
fi
# This conflicts with a file in readline
rm -f ${PREFIX}/share/man/man3/history.3

