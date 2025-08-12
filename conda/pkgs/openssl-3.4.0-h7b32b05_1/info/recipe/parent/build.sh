#!/bin/bash

PERL=${PREFIX}/bin/perl
declare -a _CONFIG_OPTS
_CONFIG_OPTS+=(--libdir=lib)
_CONFIG_OPTS+=(--prefix=${PREFIX})
_CONFIG_OPTS+=(enable-legacy)
_CONFIG_OPTS+=(no-fips)
_CONFIG_OPTS+=(no-module)
_CONFIG_OPTS+=(no-zlib)
_CONFIG_OPTS+=(shared)
_CONFIG_OPTS+=(threads)

if [[ "$target_platform" = "linux-"* ]]; then
  _CONFIG_OPTS+=(enable-ktls)
fi

# We are cross-compiling or using a specific compiler.
# do not allow config to make any guesses based on uname.
_CONFIGURATOR="perl ./Configure"
case "$target_platform" in
  linux-64)
    _CONFIG_OPTS+=(linux-x86_64)
    CFLAGS="${CFLAGS} -Wa,--noexecstack"
    ;;
  linux-aarch64)
    _CONFIG_OPTS+=(linux-aarch64)
    CFLAGS="${CFLAGS} -Wa,--noexecstack"
    ;;
  linux-ppc64le)
    _CONFIG_OPTS+=(linux-ppc64le)
    CFLAGS="${CFLAGS} -Wa,--noexecstack"
    ;;
  osx-64)
    _CONFIG_OPTS+=(darwin64-x86_64-cc)
    ;;
  osx-arm64)
    _CONFIG_OPTS+=(darwin64-arm64-cc)
    ;;
esac

CC=${CC}" ${CPPFLAGS} ${CFLAGS}" \
  ${_CONFIGURATOR} ${_CONFIG_OPTS[@]} ${LDFLAGS}

# specify in metadata where the packaging is coming from
export OPENSSL_VERSION_BUILD_METADATA="+conda_forge"

make -j${CPU_COUNT}

if [[ "${CONDA_BUILD_CROSS_COMPILATION}" != "1" ]] || [[ "$(uname -s)" = "Linux" && "$target_platform" = "linux-"* ]]; then
  if [[ "${CONDA_BUILD_CROSS_COMPILATION}" = "1" ]]; then
      # This test fails when cross-compiling and using emulation for the tests
      # > In a cross compiled situation, there are chances that our
      # > application is linked against different C libraries than
      # > perl, and may thereby get different error messages for the
      # > same error.
      # See: https://github.com/openssl/openssl/blob/openssl-3.0.0/test/recipes/02-test_errstr.t#L20-L26
      rm ./test/recipes/02-test_errstr.t
  fi
  if [[ "$target_platform" == "linux-aarch64" ]]; then
      # https://github.com/openssl/openssl/issues/17900
      rm ./test/recipes/30-test_afalg.t
  fi
  echo "Running tests"
  make test
fi
