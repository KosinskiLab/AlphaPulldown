#!/bin/bash

set -e


#pushd ${BUILD_PREFIX}/bin
#  for fn in "${BUILD}-"*; do
#    new_fn=${fn//${BUILD}-/}
#    echo "Creating symlink from ${fn} to ${new_fn}"
#    ln -sf "${fn}" "${new_fn}"
#    varname=$(basename "${new_fn}" | tr a-z A-Z | sed "s/+/X/g" | sed "s/\./_/g" | sed "s/-/_/g")
#    echo "$varname $CC"
#    printf -v "$varname" "$BUILD_PREFIX/bin/${new_fn}"
#  done
#popd

get_cpu_arch() {
  local CPU_ARCH
  if [[ "$1" == *"-64" ]]; then
    CPU_ARCH="x86_64"
  elif [[ "$1" == *"-ppc64le" ]]; then
    CPU_ARCH="powerpc64le"
  elif [[ "$1" == *"-aarch64" ]]; then
    CPU_ARCH="aarch64"
  elif [[ "$1" == *"-s390x" ]]; then
    CPU_ARCH="s390x"
  else
    echo "Unknown architecture"
    exit 1
  fi
  echo $CPU_ARCH
}

get_triplet() {
  if [[ "$1" == linux-* ]]; then
    echo "$(get_cpu_arch $1)-conda-linux-gnu"
  elif [[ "$1" == osx-64 ]]; then
    echo "x86_64-apple-darwin13.4.0"
  elif [[ "$1" == osx-arm64 ]]; then
    echo "arm64-apple-darwin20.0.0"
  elif [[ "$1" == win-64 ]]; then
    echo "x86_64-w64-mingw32"
  else
    echo "unknown platform"
  fi
}

export BUILD="$(get_triplet $build_platform)"
export HOST="$(get_triplet $target_platform)"
export TARGET="$(get_triplet $cross_target_platform)"

# Fix permissions on license files--not sure why these are world-writable, but that's how
# they come from the upstream tarball
chmod og-w COPYING*

mkdir build
cd build

if [[ "$target_platform" == win-* ]]; then
  PREFIX=$PREFIX/Library
  export CC=$BUILD_PREFIX/bin/$HOST-cc
  export CC_FOR_BUILD=$BUILD_PREFIX/bin/$BUILD-cc
fi
TARGET_SYSROOT_DIR=$PREFIX/$TARGET/sysroot

if [[ "$target_platform" == osx-arm64 ]]; then
  OSX_ARCH="arm64"
elif [[ "$target_platform" == osx-64 ]]; then
  OSX_ARCH="x86_64"
fi
if [[ "$target_platform" == osx-* ]]; then
  export CPPFLAGS="$CPPFLAGS -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET} -arch ${OSX_ARCH}"
  export CFLAGS="$CFLAGS -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET} -arch ${OSX_ARCH}"
  export CXXFLAGS="$CXXFLAGS -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET} -arch ${OSX_ARCH}"
  export LDFLAGS="$LDFLAGS -Wl,-pie -Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -arch ${OSX_ARCH}"
fi

if [[ "$target_platform" == osx-* || "$target_platform" == linux-* ]]; then
  export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib"
fi

if [[ "$target_platform" == linux-* || "$target_platform" == win-* ]]; then
  # Since we might not have libgcc-ng packaged yet, let's statically link in libgcc
  export LDFLAGS="$LDFLAGS -static-libstdc++ -static-libgcc"
fi

if [[ "$target_platform" != win-* ]]; then
  # explicitly set c99
  export CFLAGS="$CFLAGS -std=c99"
  export CFLAGS_FOR_BUILD="$(echo $CFLAGS_FOR_BUILD | sed "s#$PREFIX#$BUILD_PREFIX#g") -std=c99"
fi

../configure \
  --prefix="$PREFIX" \
  --build=$BUILD \
  --host=$HOST \
  --target=$TARGET \
  --enable-ld=default \
  --enable-gold=yes \
  --enable-plugins \
  --disable-multilib \
  --disable-sim \
  --disable-gdb \
  --disable-nls \
  --disable-gprofng \
  --enable-default-pie \
  --with-sysroot=${TARGET_SYSROOT_DIR} \
  || (cat config.log; false)

make -j${CPU_COUNT}
make install-strip DESTDIR=$SRC_DIR/install
#exit 1
