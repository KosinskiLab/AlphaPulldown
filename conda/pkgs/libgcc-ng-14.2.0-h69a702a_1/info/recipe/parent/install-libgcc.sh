#!/bin/bash

source ${RECIPE_DIR}/setup_compiler.sh
set -e -x

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
#export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/.build/${TARGET}/buildtools/bin:${SRC_DIR}/.build/tools/bin:${PATH}

pushd ${SRC_DIR}/build

  if [[ "${PKG_NAME}" == "libgcc" ]]; then
    make -C ${TARGET}/libgcc prefix=${PREFIX} install-shared
    if [[ "${TARGET}" == *mingw* ]]; then
      mv $PREFIX/lib/libgcc_s*.dll $PREFIX/bin
    fi
  elif [[ "${PKG_NAME}" != "gcc_impl"* ]]; then
    # when building a cross compiler, above make line will clobber $PREFIX/lib/libgcc_s.so.1
    # and fail after some point for some architectures. To avoid that, we copy manually
    pushd ${TARGET}/libgcc
      mkdir -p ${PREFIX}/lib/gcc/${TARGET}/${gcc_version}
      install -c -m 644 libgcc_eh.a ${PREFIX}/lib/gcc/${TARGET}/${gcc_version}/libgcc_eh.a
      chmod 644 ${PREFIX}/lib/gcc/${TARGET}/${gcc_version}/libgcc_eh.a
      ${TARGET}-ranlib ${PREFIX}/lib/gcc/${TARGET}/${gcc_version}/libgcc_eh.a

      mkdir -p ${PREFIX}/${TARGET}/lib
      if [[ "${triplet}" == *linux* ]]; then
        install -c -m 644 ./libgcc_s.so.1 ${PREFIX}/${TARGET}/lib/libgcc_s.so.1
        cp $RECIPE_DIR/libgcc_s.so.ldscript ${PREFIX}/${TARGET}/lib/libgcc_s.so
      else
        # import library, not static library
        install -c -m 644 ./shlib/libgcc_s.a ${PREFIX}/${TARGET}/lib/libgcc_s.a
      fi
    popd
  fi

  # TODO :: Also do this for libgfortran (and libstdc++ too probably?)
  if [[ -f ${TARGET}/libsanitizer/libtool ]]; then
    sed -i.bak 's/.*cannot install.*/func_warning "Ignoring libtool error about cannot install to a directory not ending in"/' \
             ${TARGET}/libsanitizer/libtool
  fi
  for lib in libatomic libgomp libquadmath libitm libvtv libsanitizer/{a,l,ub,t}san; do
    # TODO :: Also do this for libgfortran (and libstdc++ too probably?)
    if [[ -f ${TARGET}/${lib}/libtool ]]; then
      sed -i.bak 's/.*cannot install.*/func_warning "Ignoring libtool error about cannot install to a directory not ending in"/' \
                 ${TARGET}/${lib}/libtool
    fi
    if [[ -d ${TARGET}/${lib} ]]; then
      make -C ${TARGET}/${lib} prefix=${PREFIX} install-toolexeclibLTLIBRARIES
      make -C ${TARGET}/${lib} prefix=${PREFIX} install-nodist_fincludeHEADERS || true
    fi
  done

  for lib in libgomp libquadmath; do
    if [[ -d ${TARGET}/${lib} ]]; then
      make -C ${TARGET}/${lib} prefix=${PREFIX} install-info
    fi
  done

popd

mkdir -p ${PREFIX}/lib

if [[ "${PKG_NAME}" != "gcc_impl"* ]]; then
  # no static libs
  find ${PREFIX}/lib -name "*\.a" -exec rm -rf {} \;
fi
# no libtool files
find ${PREFIX}/lib -name "*\.la" -exec rm -rf {} \;

if [[ "${PKG_NAME}" != gcc_impl* ]]; then
  # mv ${PREFIX}/${TARGET}/lib/* ${PREFIX}/lib
  # clean up empty folder
  rm -rf ${PREFIX}/lib/gcc
  rm -rf ${PREFIX}/lib/lib{a,l,ub,t}san.so*

  # Install Runtime Library Exception
  install -Dm644 ${SRC_DIR}/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/gcc-libs/RUNTIME.LIBRARY.EXCEPTION
fi

rm -f ${PREFIX}/share/info/dir
