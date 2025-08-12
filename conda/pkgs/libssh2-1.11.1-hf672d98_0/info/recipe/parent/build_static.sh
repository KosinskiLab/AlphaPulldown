#!/bin/bash

# copy files which are missing from the release tarball
# see: https://github.com/libssh2/libssh2/issues/379
# TODO: remove this in the 1.9.1 or later releases
cp ${RECIPE_DIR}/missing_files/*.c tests/

# We use a repackaged cmake from elsewhere to break a build cycle.
export PATH=${PREFIX}/cmake-bin/bin:${PATH}

if [[ $target_platform =~ linux.* ]]; then
  export LDFLAGS="$LDFLAGS -Wl,-rpath-link,$PREFIX/lib"
fi

cmake -D CMAKE_INSTALL_PREFIX=$PREFIX \
      -D CMAKE_PREFIX_PATH=$PREFIX \
      -D BUILD_SHARED_LIBS=OFF \
      -D BUILD_STATIC_LIBS=ON \
      -D CRYPTO_BACKEND=OpenSSL \
      -D CMAKE_INSTALL_LIBDIR=lib \
      -D ENABLE_ZLIB_COMPRESSION=ON \
      $SRC_DIR

make -j${CPU_COUNT}
# ctest  # fails on the docker image
make install