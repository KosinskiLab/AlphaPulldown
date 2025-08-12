#!/bin/bash

# copy files which are missing from the release tarball
# see: https://github.com/libssh2/libssh2/issues/379
# TODO: remove this in the 1.9.1 or later releases
cp ${RECIPE_DIR}/missing_files/*.c tests/

# We use a repackaged cmake from elsewhere to break a build cycle.
export PATH=${PREFIX}/cmake-bin/bin:${PATH}

mkdir build && cd build

cmake ${CMAKE_ARGS} \
      -DCMAKE_POLICY_DEFAULT_CMP0042=NEW \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_PREFIX_PATH=$PREFIX \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_STATIC_LIBS=OFF \
      -DCRYPTO_BACKEND=OpenSSL \
      -DENABLE_ZLIB_COMPRESSION=ON \
      $SRC_DIR

make -j${CPU_COUNT}
# ctest  # fails on the docker image
make install
