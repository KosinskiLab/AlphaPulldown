set -e -x

export CHOST="${gcc_machine}-${gcc_vendor}-linux-gnu"
_libdir=libexec/gcc/${CHOST}/${PKG_VERSION}

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
#export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/.build/${CHOST}/buildtools/bin:${SRC_DIR}/.build/tools/bin:${PATH}

pushd ${SRC_DIR}/build

make -C gcc prefix=${PREFIX} c++.install-common

# How it used to be:
# install -m755 -t ${PREFIX}/bin/ gcc/{cc1plus,lto1}
for file in cc1plus; do
  if [[ -f gcc/${file} ]]; then
    install -c gcc/${file} ${PREFIX}/${_libdir}/${file}
  fi
done

# Following 3 are in libstdcxx-devel
#make -C $CHOST/libstdc++-v3/src prefix=${PREFIX} install
#make -C $CHOST/libstdc++-v3/include prefix=${PREFIX} install
#make -C $CHOST/libstdc++-v3/libsupc++ prefix=${PREFIX} install
make -C $CHOST/libstdc++-v3/python prefix=${PREFIX} install

# Probably don't want to do this for cross-compilers
# mkdir -p ${PREFIX}/share/gdb/auto-load/usr/lib/
# cp ${SRC_DIR}/gcc_built/${CHOST}/sysroot/lib/libstdc++.so.6.*-gdb.py ${PREFIX}/share/gdb/auto-load/usr/lib/

make -C libcpp prefix=${PREFIX} install

popd

mkdir -p ${PREFIX}/lib/gcc/${CHOST}/${PKG_VERSION}
if [[ "${target_platform}" == "${cross_target_platform}" ]]; then
  ln -sf ${PREFIX}/lib/libstdc++.so ${PREFIX}/${CHOST}/lib/libstdc++.so
fi

set +x
# Strip executables, we may want to install to a different prefix
# and strip in there so that we do not change files that are not
# part of this package.
pushd ${PREFIX}
  _files=$(find . -type f)
  for _file in ${_files}; do
    _type="$( file "${_file}" | cut -d ' ' -f 2- )"
    case "${_type}" in
      *script*executable*)
      ;;
      *executable*)
       ${BUILD_PREFIX}/bin/${CHOST}-strip --strip-all -v "${_file}" || :
      ;;
    esac
  done
popd

source ${RECIPE_DIR}/make_tool_links.sh
