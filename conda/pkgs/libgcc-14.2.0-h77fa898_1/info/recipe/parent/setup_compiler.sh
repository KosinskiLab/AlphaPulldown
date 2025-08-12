#!/bin/bash
if [[ ! -d  $SRC_DIR/cf-compilers ]]; then
    extra_pkgs=()
    if [[ "$build_platform" != "$target_platform" ]]; then
      # we need a compiler to target cross_target_platform.
      # when build_platform == target_platform, the compiler
      # just built can be used.
      # when build_platform != target_platform, the compiler
      # just built cannot be used, hence we need one that
      # can be used.
      extra_pkgs+=(
        "gcc_impl_${cross_target_platform}=${gcc_version}"
        "gxx_impl_${cross_target_platform}=${gcc_version}"
        "gfortran_impl_${cross_target_platform}=${gcc_version}"
      )
    fi
    # Remove conda-forge/label/sysroot-with-crypt when GCC < 14 is dropped
    conda create -p $SRC_DIR/cf-compilers -c conda-forge/label/sysroot-with-crypt -c conda-forge --yes --quiet \
      "binutils_impl_${build_platform}" \
      "gcc_impl_${build_platform}" \
      "gxx_impl_${build_platform}" \
      "gfortran_impl_${build_platform}" \
      "binutils_impl_${target_platform}=${binutils_version}" \
      "gcc_impl_${target_platform}" \
      "gxx_impl_${target_platform}" \
      "gfortran_impl_${target_platform}" \
      "${c_stdlib}_${target_platform}=${c_stdlib_version}" \
      "binutils_impl_${cross_target_platform}=${binutils_version}" \
      "${cross_target_stdlib}_${cross_target_platform}=${cross_target_stdlib_version}" \
      ${extra_pkgs[@]}
fi

export PATH=$SRC_DIR/cf-compilers/bin:$PATH
export BUILD_PREFIX=$SRC_DIR/cf-compilers

if [[ "$target_platform" == "win-"* && "${PREFIX}" != *Library ]]; then
    export PREFIX=${PREFIX}/Library
fi

source $RECIPE_DIR/get_cpu_arch.sh

if [[ "$target_platform" == "win-64" ]]; then
  EXEEXT=".exe"
else
  EXEEXT=""
fi
SYSROOT_DIR=${PREFIX}/${TARGET}/sysroot
