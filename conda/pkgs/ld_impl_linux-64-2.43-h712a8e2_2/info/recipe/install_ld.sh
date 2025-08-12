#!/bin/bash

set -e

TARGET="${triplet}"
OLD_TARGET="${triplet/conda/${ctng_vendor}}"

if [[ "$target_platform" == win-* ]]; then
  EXEEXT=".exe"
  PREFIX=$PREFIX/Library
  symlink="cp"
else
  symlink="ln -s"
fi

SYSROOT=$PREFIX/${TARGET}
OLD_SYSROOT=$PREFIX/${OLD_TARGET}

mkdir -p $PREFIX/bin
mkdir -p $OLD_SYSROOT/bin
mkdir -p $SYSROOT/bin

if [[ "$target_platform" == "$cross_target_platform" ]]; then
  cp $PWD/install/$PREFIX/bin/ld${EXEEXT} $PREFIX/bin/$TARGET-ld${EXEEXT}
else
  cp $PWD/install/$PREFIX/bin/$TARGET-ld${EXEEXT} $PREFIX/bin/$TARGET-ld${EXEEXT}
fi

if [[ "$TARGET" != "$OLD_TARGET" ]]; then
  $symlink $PREFIX/bin/$TARGET-ld${EXEEXT} $PREFIX/bin/$OLD_TARGET-ld${EXEEXT}
  $symlink $PREFIX/bin/$TARGET-ld${EXEEXT} $OLD_SYSROOT/bin/ld${EXEEXT}
fi
$symlink $PREFIX/bin/$TARGET-ld${EXEEXT} $SYSROOT/bin/ld${EXEEXT}
