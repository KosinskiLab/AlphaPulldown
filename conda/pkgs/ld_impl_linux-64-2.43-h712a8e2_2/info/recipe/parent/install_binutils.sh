#!/bin/bash

set -e

cd install

find . -type f -exec bash -c 'mkdir -p /$(dirname {}) && cp {} /{}' ';'

export TARGET="${triplet}"
export OLD_TARGET="${triplet/conda/${ctng_vendor}}"

if [[ "${target_platform}" == win-* ]]; then
  EXEEXT=".exe"
  PREFIX=${PREFIX}/Library
  symlink="cp"
else
  symlink="ln -s"
fi

SYSROOT=${PREFIX}/${TARGET}
OLD_SYSROOT=${PREFIX}/${OLD_TARGET}

mkdir -p ${PREFIX}/bin
mkdir -p ${SYSROOT}/bin
mkdir -p ${OLD_SYSROOT}/bin

TOOLS="addr2line ar as c++filt elfedit gprof ld.bfd nm objcopy objdump ranlib readelf size strings strip"

if [[ "${cross_target_platform}" == "linux-"* ]]; then
  TOOLS="${TOOLS} dwp ld.gold"
else
  TOOLS="${TOOLS} dlltool dllwrap windmc windres"
fi

# Remove hardlinks and replace them by softlinks
for tool in ${TOOLS}; do
  tool=${tool}${EXEEXT}
  if [[ "$target_platform" == "$cross_target_platform" ]]; then
      mv ${PREFIX}/bin/${tool} ${PREFIX}/bin/${TARGET}-${tool}
  fi
  rm -rf ${SYSROOT}/bin/${tool}
  $symlink ${PREFIX}/bin/${TARGET}-${tool} ${SYSROOT}/bin/${tool}
  if [[ "${TARGET}" != "${OLD_TARGET}" ]]; then
    $symlink ${PREFIX}/bin/${TARGET}-${tool} ${OLD_SYSROOT}/bin/${tool}
    $symlink ${PREFIX}/bin/${TARGET}-${tool} ${PREFIX}/bin/$OLD_TARGET-${tool}
  fi
done

rm ${PREFIX}/bin/ld${EXEEXT} || true;
rm ${PREFIX}/bin/${TARGET}-ld${EXEEXT} || true;
rm ${PREFIX}/bin/$OLD_TARGET-ld${EXEEXT} || true;
rm ${OLD_SYSROOT}/bin/ld${EXEEXT} || true;
rm ${SYSROOT}/bin/ld${EXEEXT} || true;

