#!/bin/bash

if [[ $target_platform =~ linux.* ]]; then
    USE_ICONV=--without-iconv
else
    USE_ICONV=--with-iconv
fi

autoreconf -vfi
mkdir build-${HOST} && pushd build-${HOST}
${SRC_DIR}/configure --prefix=${PREFIX}  \
                     --with-zlib         \
                     --with-bz2lib       \
                     ${USE_ICONV}        \
                     --with-lz4          \
                     --with-lzma         \
                     --with-lzo2         \
                     --with-zstd         \
                     --without-cng       \
                     --with-openssl      \
                     --without-nettle    \
                     --with-xml2         \
                     --without-expat

make -j${CPU_COUNT} ${VERBOSE_AT}
make install
popd
