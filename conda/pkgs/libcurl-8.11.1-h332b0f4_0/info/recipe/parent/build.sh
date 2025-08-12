#!/bin/bash
# Get an updated config.sub and config.guess
cp $BUILD_PREFIX/share/libtool/build-aux/config.* .

# need macosx-version-min flags set in cflags and not cppflags
export CFLAGS="$CFLAGS $CPPFLAGS"

if [[ "$target_platform" == "osx-"* ]]; then
    USESSL="--with-openssl=${PREFIX} --with-secure-transport --with-default-ssl-backend=openssl"
else
    USESSL="--with-openssl=${PREFIX}"
fi

./configure \
    --prefix=${PREFIX} \
    --host=${HOST} \
    --disable-ldap \
    --enable-websockets \
    --with-ca-bundle=${PREFIX}/ssl/cacert.pem \
    $USESSL \
    --with-zlib=${PREFIX} \
    --with-zstd=${PREFIX} \
    --with-gssapi=${PREFIX} \
    --with-libssh2=${PREFIX} \
    --with-nghttp2=${PREFIX} \
    --without-libpsl \
|| cat config.log

make -j${CPU_COUNT} ${VERBOSE_AT}
# TODO :: test 1119... exit FAILED
# make test
make install

# Includes man pages and other miscellaneous.
rm -rf "${PREFIX}/share"
