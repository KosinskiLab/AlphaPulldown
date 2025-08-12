#!/bin/bash

set -ex

IFS="." read -a VER_ARR <<<"${PKG_VERSION}"


pushd tcl${PKG_VERSION}/unix
  # autoreconf -vfi
  ./configure  --prefix="${PREFIX}"
  make -j${CPU_COUNT} ${VERBOSE_AT}
  make install install-private-headers
popd

if [[ "$target_platform" == osx-* ]]; then
  CONFIGURE_ARGS="${CONFIGURE_ARGS} --enable-aqua=yes"
elif [[ "$tk_variant" == xft ]]; then
  CONFIGURE_ARGS="${CONFIGURE_ARGS} --enable-xft"
  # Remove requires.private for now. Otherwise we need devel packages of
  # libxrender-devel and the deps in libxrender-devel is broken anyway.
  sed -i.bak 's/Requires.private: xrender, /Requires.private: /g' $BUILD_PREFIX/$HOST/sysroot/usr/lib/pkgconfig/xft.pc
  pkg-config --cflags xft fontconfig
fi

pushd tk${PKG_VERSION}/unix
  # autoreconf -vfi
  ./configure --prefix="${PREFIX}"        \
              --with-tcl="${PREFIX}"/lib  \
              ${CONFIGURE_ARGS}
  cat config.log
  make -j${CPU_COUNT} ${VERBOSE_AT}
  make install
popd

rm -rf "${PREFIX}"/{man,share}

# Link binaries to non-versioned names to make them easier to find and use.
ln -s "${PREFIX}"/bin/tclsh${VER_ARR[0]}.${VER_ARR[1]} "${PREFIX}"/bin/tclsh
ln -s "${PREFIX}"/bin/wish${VER_ARR[0]}.${VER_ARR[1]} "${PREFIX}"/bin/wish

# copy headers
cp "${SRC_DIR}"/tk${PKG_VERSION}/{unix,macosx,generic}/*.h "${PREFIX}"/include/

# Remove buildroot traces
sed -i.bak -e "s,${SRC_DIR}/tk${PKG_VERSION}/unix,${PREFIX}/lib,g" -e "s,${SRC_DIR}/tk${PKG_VERSION},${PREFIX}/include,g" ${PREFIX}/lib/tkConfig.sh
sed -i.bak -e "s,${SRC_DIR}/tcl${PKG_VERSION}/unix,${PREFIX}/lib,g" -e "s,${SRC_DIR}/tcl${PKG_VERSION},${PREFIX}/include,g" ${PREFIX}/lib/tclConfig.sh
rm -f ${PREFIX}/lib/tkConfig.sh.bak
rm -f ${PREFIX}/lib/tclConfig.sh.bak
