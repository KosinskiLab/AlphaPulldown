#!/bin/bash
set -x

TARGET="${triplet}"

if [[ "${target_platform}" == win-* ]]; then
  EXEEXT=".exe"
  PREFIX=${PREFIX}/Library
  symlink="cp"
else
  symlink="ln -s"
fi

TOOLS="addr2line ar as c++filt elfedit gprof ld ld.bfd nm objcopy objdump ranlib readelf size strings strip"

if [[ "${cross_target_platform}" == "linux-"* ]]; then
  TOOLS="${TOOLS} dwp ld.gold"
  $symlink "${PREFIX}/bin/ld.gold${EXEEXT}" "${PREFIX}/bin/gold${EXEEXT}"
else
  TOOLS="${TOOLS} dlltool dllwrap windmc windres"
fi

for tool in ${TOOLS}; do
  rm ${PREFIX}/bin/${TARGET}-${tool}${EXEEXT}
  touch ${PREFIX}/bin/${TARGET}-${tool}${EXEEXT}
  $symlink ${PREFIX}/bin/${TARGET}-${tool}${EXEEXT} ${PREFIX}/bin/${tool}${EXEEXT}
done
