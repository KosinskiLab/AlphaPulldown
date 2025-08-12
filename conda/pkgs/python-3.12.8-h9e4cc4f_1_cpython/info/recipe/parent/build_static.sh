#!/bin/bash
set -ex

_buildd_static=build-static
_buildd_shared=build-shared
if [[ ${PY_INTERP_DEBUG} == yes ]]; then
  DBG=d
else
  DBG=
fi
VER=${PKG_VERSION%.*}
VERABI=${VER}${DBG}


cp -pf ${_buildd_static}/libpython${VERABI}.a ${PREFIX}/lib/libpython${VERABI}.a
if [[ ${HOST} =~ .*linux.* ]]; then
  pushd ${PREFIX}/lib/python${VER}/config-${VERABI}-${HOST/-conda/}
elif [[ ${HOST} =~ .*darwin.* ]]; then
  pushd ${PREFIX}/lib/python${VER}/config-${VERABI}-darwin
fi
ln -s ../../libpython${VERABI}.a libpython${VERABI}.a
popd
# If the LTO info in the normal lib is problematic (using different compilers for example
# we also provide a 'nolto' version).
cp -pf ${_buildd_shared}/libpython${VERABI}-pic.a ${PREFIX}/lib/libpython${VERABI}.nolto.a
