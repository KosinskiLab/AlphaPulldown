set -e -x

export CHOST="${gcc_machine}-${gcc_vendor}-linux-gnu"

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
#export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/.build/${CHOST}/buildtools/bin:${SRC_DIR}/.build/tools/bin:${PATH}

pushd ${SRC_DIR}/build

  if [[ "${PKG_NAME}" == libgcc-ng ]]; then
    make -C ${CHOST}/libgcc prefix=${PREFIX} install-shared
  else
    # when building a cross compiler, above make line will clobber $PREFIX/lib/libgcc_s.so.1
    # and fail after some point for some architectures. To avoid that, we copy manually
    pushd ${CHOST}/libgcc
      mkdir -p ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}
      install -c -m 644 libgcc_eh.a ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}/libgcc_eh.a
      chmod 644 ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}/libgcc_eh.a
      ${CHOST}-ranlib ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}/libgcc_eh.a

      mkdir -p ${PREFIX}/${CHOST}/lib
      install -c -m 644 ./libgcc_s.so.1 ${PREFIX}/${CHOST}/lib/libgcc_s.so.1
      cp $RECIPE_DIR/libgcc_s.so.ldscript ${PREFIX}/${CHOST}/lib/libgcc_s.so
    popd
  fi

  # TODO :: Also do this for libgfortran (and libstdc++ too probably?)
  sed -i.bak 's/.*cannot install.*/func_warning "Ignoring libtool error about cannot install to a directory not ending in"/' \
             ${CHOST}/libsanitizer/libtool
  for lib in libatomic libgomp libquadmath libitm libvtv libsanitizer/{a,l,ub,t}san; do
    # TODO :: Also do this for libgfortran (and libstdc++ too probably?)
    if [[ -f ${CHOST}/${lib}/libtool ]]; then
      sed -i.bak 's/.*cannot install.*/func_warning "Ignoring libtool error about cannot install to a directory not ending in"/' \
                 ${CHOST}/${lib}/libtool
    fi
    if [[ -d ${CHOST}/${lib} ]]; then
      make -C ${CHOST}/${lib} prefix=${PREFIX} install-toolexeclibLTLIBRARIES
      make -C ${CHOST}/${lib} prefix=${PREFIX} install-nodist_fincludeHEADERS || true
    fi
  done

  for lib in libgomp libquadmath; do
    if [[ -d ${CHOST}/${lib} ]]; then
      make -C ${CHOST}/${lib} prefix=${PREFIX} install-info
    fi
  done

popd

mkdir -p ${PREFIX}/lib

# no static libs
find ${PREFIX}/lib -name "*\.a" -exec rm -rf {} \;
# no libtool files
find ${PREFIX}/lib -name "*\.la" -exec rm -rf {} \;

if [[ "${PKG_NAME}" != gcc_impl* ]]; then
  # mv ${PREFIX}/${CHOST}/lib/* ${PREFIX}/lib
  # clean up empty folder
  rm -rf ${PREFIX}/lib/gcc
  rm -rf ${PREFIX}/lib/lib{a,l,ub,t}san.so*

  # Install Runtime Library Exception
  install -Dm644 ${SRC_DIR}/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/gcc-libs/RUNTIME.LIBRARY.EXCEPTION
fi
