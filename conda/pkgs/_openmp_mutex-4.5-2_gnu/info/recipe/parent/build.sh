#!/bin/bash

set -ex

# ensure patch is applied
grep 'conda-forge:: allow' gcc/gcc.c*

get_cpu_arch() {
  local CPU_ARCH
  if [[ "$1" == "linux-64" ]]; then
    CPU_ARCH="x86_64"
  elif [[ "$1" == "linux-ppc64le" ]]; then
    CPU_ARCH="powerpc64le"
  elif [[ "$1" == "linux-aarch64" ]]; then
    CPU_ARCH="aarch64"
  elif [[ "$1" == "linux-s390x" ]]; then
    CPU_ARCH="s390x"
  else
    echo "Unknown architecture"
    exit 1
  fi
  echo $CPU_ARCH
}

if [[ "$channel_targets" == *conda-forge* && "${build_platform}" == "${target_platform}" ]]; then
  # Use new compilers instead of relying on ones from the docker image
  conda create -p $SRC_DIR/cf-compilers gcc gfortran gxx binutils -c conda-forge --yes --quiet
  export PATH=$SRC_DIR/cf-compilers/bin:$PATH
fi

GCC_CONFIGURE_OPTIONS=()

if [[ "$channel_targets" == *conda-forge* ]]; then
  GCC_CONFIGURE_OPTIONS+=(--with-pkgversion="conda-forge gcc ${gcc_version}-${PKG_BUILDNUM}")
  GCC_CONFIGURE_OPTIONS+=(--with-bugurl="https://github.com/conda-forge/ctng-compilers-feedstock/issues/new/choose")
fi

export BUILD="$(get_cpu_arch $build_platform)-${gcc_vendor}-linux-gnu"
export HOST="$(get_cpu_arch $target_platform)-${gcc_vendor}-linux-gnu"
export TARGET="$(get_cpu_arch $cross_target_platform)-${gcc_vendor}-linux-gnu"

for tool in addr2line ar as c++filt gcc g++ ld nm objcopy objdump ranlib readelf size strings strip; do
  if [[ ! -f $BUILD_PREFIX/bin/$BUILD-$tool ]]; then
    ln -s $(which $tool) $BUILD_PREFIX/bin/$BUILD-$tool
  fi
  tool_upper=$(echo $tool | tr a-z A-Z | sed "s/+/X/g")
  if [[ "$tool" == gcc ]]; then
     tool_upper=CC
  elif [[ "$tool" == g++ ]]; then
     tool_upper=CXX
  fi
  eval "export ${tool_upper}_FOR_BUILD=\$BUILD_PREFIX/bin/\$BUILD-\$tool"
  eval "export ${tool_upper}_FOR_TARGET=\$BUILD_PREFIX/bin/\$TARGET-\$tool"
  eval "export ${tool_upper}=\$BUILD_PREFIX/bin/\$HOST-\$tool"
done

if [[ $build_platform != $target_platform ]]; then
  export GFORTRAN_FOR_TARGET="$BUILD_PREFIX/bin/$TARGET-gfortran"
  export GXX_FOR_TARGET="$BUILD_PREFIX/bin/$TARGET-g++"
  export FC=$GFORTRAN_FOR_TARGET
fi

# workaround a bug in gcc build files when using external binutils
# and build != host == target
export gcc_cv_objdump=$OBJDUMP_FOR_TARGET

# Workaround a problem in our gcc_bootstrap package
if [[ -d $BUILD_PREFIX/$BUILD/sysroot/usr/lib64 && ! -d $BUILD_PREFIX/$BUILD/sysroot/usr/lib ]]; then
  mkdir -p $BUILD_PREFIX/$BUILD/sysroot/usr
  ln -sf $BUILD_PREFIX/$BUILD/sysroot/usr/lib64 $BUILD_PREFIX/$BUILD/sysroot/usr/lib
fi

ls $BUILD_PREFIX/bin/

./contrib/download_prerequisites

# We want CONDA_PREFIX/usr/lib not CONDA_PREFIX/usr/lib64 and this
# is the only way. It is incompatible with multilib (obviously).
TINFO_FILES=$(find . -path "*/config/*/t-*")
for TINFO_FILE in ${TINFO_FILES}; do
  echo TINFO_FILE ${TINFO_FILE}
  sed -i.bak 's#^\(MULTILIB_OSDIRNAMES.*\)\(lib64\)#\1lib#g' ${TINFO_FILE}
  rm -f ${TINFO_FILE}.bak
  sed -i.bak 's#^\(MULTILIB_OSDIRNAMES.*\)\(libx32\)#\1lib#g' ${TINFO_FILE}
  rm -f ${TINFO_FILE}.bak
done

# workaround for https://gcc.gnu.org/bugzilla//show_bug.cgi?id=80196
if [[ "$gcc_version" == "11."* && "$build_platform" != "$target_platform" ]]; then
  sed -i.bak 's@-I$glibcxx_srcdir/libsupc++@-I$glibcxx_srcdir/libsupc++ -nostdinc++@g' libstdc++-v3/configure
fi

mkdir -p build
cd build

# We need to explicitly set the gxx include dir because previously
# with ct-ng, native build was not considered native because
# BUILD=HOST=x86_64-build_unknown-linux-gnu and TARGET=x86_64-conda-linux-gnu
# Depending on native or not, the include dir changes. Setting it explictly
# goes back to the original way.
# See https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/gcc/configure.ac#L218

../configure \
  --prefix="$PREFIX" \
  --with-slibdir="$PREFIX/lib" \
  --libdir="$PREFIX/lib" \
  --mandir="$PREFIX/man" \
  --build=$BUILD \
  --host=$HOST \
  --target=$TARGET \
  --enable-default-pie \
  --enable-languages=c,c++,fortran,objc,obj-c++ \
  --enable-__cxa_atexit \
  --disable-libmudflap \
  --enable-libgomp \
  --disable-libssp \
  --enable-libquadmath \
  --enable-libquadmath-support \
  --enable-libsanitizer \
  --enable-lto \
  --enable-threads=posix \
  --enable-target-optspace \
  --enable-plugin \
  --enable-gold \
  --disable-nls \
  --disable-bootstrap \
  --disable-multilib \
  --enable-long-long \
  --with-sysroot=${PREFIX}/${TARGET}/sysroot \
  --with-build-sysroot=${BUILD_PREFIX}/${TARGET}/sysroot \
  --with-gxx-include-dir="${PREFIX}/${TARGET}/include/c++/${gcc_version}" \
  "${GCC_CONFIGURE_OPTIONS[@]}"

make -j${CPU_COUNT} || (cat ${TARGET}/libbacktrace/config.log; false)
