set -e -x

export CHOST="${gcc_machine}-${gcc_vendor}-linux-gnu"
_libdir=libexec/gcc/${CHOST}/${PKG_VERSION}

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
#export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/.build/${CHOST}/buildtools/bin:${SRC_DIR}/.build/tools/bin:${PATH}

pushd ${SRC_DIR}/build

# adapted from Arch install script from https://github.com/archlinuxarm/PKGBUILDs/blob/master/core/gcc/PKGBUILD
# We cannot make install since .la files are not relocatable so libtool deliberately prevents it:
# libtool: install: error: cannot install `libgfortran.la' to a directory not ending in ${SRC_DIR}/work/gcc_built/${CHOST}/lib/../lib
make -C ${CHOST}/libgfortran prefix=${PREFIX} all-multi libgfortran.spec ieee_arithmetic.mod ieee_exceptions.mod ieee_features.mod config.h
make -C gcc prefix=${PREFIX} fortran.install-{common,man,info}

# How it used to be:
# install -Dm755 gcc/f951 ${PREFIX}/${_libdir}/f951
for file in f951; do
  if [[ -f gcc/${file} ]]; then
    install -c gcc/${file} ${PREFIX}/${_libdir}/${file}
  fi
done

mkdir -p ${PREFIX}/${CHOST}/lib
cp ${CHOST}/libgfortran/libgfortran.spec ${PREFIX}/${CHOST}/lib

pushd ${PREFIX}/bin
  ln -sf ${CHOST}-gfortran ${CHOST}-f95
popd

make install DESTDIR=$SRC_DIR/build-finclude
mkdir -p $PREFIX/lib/gcc/${CHOST}/${gcc_version}/finclude
install -Dm644 $SRC_DIR/build-finclude/$PREFIX/lib/gcc/${CHOST}/${gcc_version}/finclude/* $PREFIX/lib/gcc/${CHOST}/${gcc_version}/finclude/

# Install Runtime Library Exception
install -Dm644 $SRC_DIR/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/gcc-fortran/RUNTIME.LIBRARY.EXCEPTION

if [[ "${target_platform}" != "${cross_target_platform}" ]]; then
  cp -f --no-dereference ${SRC_DIR}/build/${CHOST}/libgfortran/.libs/libgfortran*.so* ${PREFIX}/${CHOST}/lib/
fi
cp -f --no-dereference ${SRC_DIR}/build/${CHOST}/libgfortran/.libs/libgfortran.a ${PREFIX}/${CHOST}/lib/

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

if [[ -f ${PREFIX}/lib/libgomp.spec ]]; then
  mv ${PREFIX}/lib/libgomp.spec ${PREFIX}/${CHOST}/lib/libgomp.spec
fi

source ${RECIPE_DIR}/make_tool_links.sh
