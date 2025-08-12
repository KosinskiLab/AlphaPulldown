#!/bin/bash
set -xe

export CPPFLAGS="${CPPFLAGS/-DNDEBUG/}"

# https://github.com/conda-forge/bison-feedstock/issues/7
export M4="${BUILD_PREFIX}/bin/m4"

if [[ "$target_platform" == "osx-arm64" ]]; then
    # This can't be deduced when cross-compiling
    export krb5_cv_attr_constructor_destructor=yes,yes
    export ac_cv_func_regcomp=yes
    export ac_cv_printf_positional=yes
    sed -i.bak "s@mig -header@mig -cc $(which $CC) -arch arm64 -header@g" src/lib/krb5/ccache/Makefile.in
elif [[ "${target_platform}" == "linux-ppc64le" ]]; then
    # This can't be deduced when cross-compiling
    export krb5_cv_attr_constructor_destructor=yes,yes
    export ac_cv_func_regcomp=yes
    export ac_cv_printf_positional=yes
fi

pushd src
  autoreconf -i
  ./configure --prefix=${PREFIX}          \
              --host=${HOST}              \
              --build=${BUILD}            \
              --without-tcl               \
              --without-readline          \
              --with-libedit              \
              --with-crypto-impl=openssl  \
              --without-system-verto      \
              --disable-shared            \
              --with-keyutils=${PREFIX}   \
              --enable-static

  make -j${CPU_COUNT} ${VERBOSE_AT}
  make install
popd
