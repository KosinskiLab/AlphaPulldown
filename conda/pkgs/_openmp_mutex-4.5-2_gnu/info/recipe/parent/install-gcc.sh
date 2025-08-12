set -e -x

export CHOST="${gcc_machine}-${gcc_vendor}-linux-gnu"
_libdir=libexec/gcc/${CHOST}/${PKG_VERSION}

# libtool wants to use ranlib that is here, macOS install doesn't grok -t etc
# .. do we need this scoped over the whole file though?
#export PATH=${SRC_DIR}/gcc_built/bin:${SRC_DIR}/.build/${CHOST}/buildtools/bin:${SRC_DIR}/.build/tools/bin:${PATH}

pushd ${SRC_DIR}/build
  # We may not have built with plugin support so failure here is not fatal:
  make prefix=${PREFIX} install-lto-plugin || true
  make -C gcc prefix=${PREFIX} install-driver install-cpp install-gcc-ar install-headers install-plugin install-lto-wrapper install-collect2
  # not sure if this is the same as the line above.  Run both, just in case
  make -C lto-plugin prefix=${PREFIX} install
  install -dm755 ${PREFIX}/lib/bfd-plugins/

  # statically linked, so this so does not exist
  # ln -s $PREFIX/lib/gcc/$CHOST/liblto_plugin.so ${PREFIX}/lib/bfd-plugins/

  make -C libcpp prefix=${PREFIX} install

  # Include languages we do not have any other place for here (and also lto1)
  for file in gnat1 brig1 cc1 go1 lto1 cc1obj cc1objplus; do
    if [[ -f gcc/${file} ]]; then
      install -c gcc/${file} ${PREFIX}/${_libdir}/${file}
    fi
  done

  # https://github.com/gcc-mirror/gcc/blob/gcc-7_3_0-release/gcc/Makefile.in#L3481-L3526
  # Could have used install-common, but it also installs cxx binaries, which we
  # don't want in this package. We could patch it, or use the loop below:
  for file in gcov{,-tool,-dump}; do
    if [[ -f gcc/${file} ]]; then
      install -c gcc/${file} ${PREFIX}/bin/${CHOST}-${file}
    fi
  done

  # make -C ${CHOST}/libgcc prefix=${PREFIX} install

  # mkdir -p $PREFIX/$CHOST/sysroot/lib

  # cp ${SRC_DIR}/gcc_built/$CHOST/sysroot/lib/libgomp.so* $PREFIX/$CHOST/sysroot/lib
  # if [ -e ${SRC_DIR}/gcc_built/$CHOST/sysroot/lib/libquadmath.so* ]; then
  #   cp ${SRC_DIR}/gcc_built/$CHOST/sysroot/lib/libquadmath.so* $PREFIX/$CHOST/sysroot/lib
  # fi

  make prefix=${PREFIX}/lib/gcc/${CHOST}/${gcc_version} install-libcc1
  install -d ${PREFIX}/share/gdb/auto-load/usr/lib

  make prefix=${PREFIX} install-fixincludes
  make -C gcc prefix=${PREFIX} install-mkheaders

  if [[ -d ${CHOST}/libgomp ]]; then
    make -C ${CHOST}/libgomp prefix=${PREFIX} install-nodist_{libsubinclude,toolexeclib}HEADERS
  fi

  if [[ -d ${CHOST}/libitm ]]; then
    make -C ${CHOST}/libitm prefix=${PREFIX} install-nodist_toolexeclibHEADERS
  fi

  if [[ -d ${CHOST}/libquadmath ]]; then
    make -C ${CHOST}/libquadmath prefix=${PREFIX} install-nodist_libsubincludeHEADERS
  fi

  if [[ -d ${CHOST}/libsanitizer ]]; then
    make -C ${CHOST}/libsanitizer prefix=${PREFIX} install-nodist_{saninclude,toolexeclib}HEADERS
  fi

  if [[ -d ${CHOST}/libsanitizer/asan ]]; then
    make -C ${CHOST}/libsanitizer/asan prefix=${PREFIX} install-nodist_toolexeclibHEADERS
  fi

  if [[ -d ${CHOST}/libsanitizer/tsan ]]; then
    make -C ${CHOST}/libsanitizer/tsan prefix=${PREFIX} install-nodist_toolexeclibHEADERS
  fi

  make -C libiberty prefix=${PREFIX} install
  # install PIC version of libiberty
  install -m644 libiberty/pic/libiberty.a ${PREFIX}/lib/gcc/${CHOST}/${gcc_version}

  make -C gcc prefix=${PREFIX} install-man install-info

  make -C gcc prefix=${PREFIX} install-po

  # many packages expect this symlink
  [[ -f ${PREFIX}/bin/${CHOST}-cc ]] && rm ${PREFIX}/bin/${CHOST}-cc
  pushd ${PREFIX}/bin
    ln -s ${CHOST}-gcc ${CHOST}-cc
  popd

  # POSIX conformance launcher scripts for c89 and c99
  cat > ${PREFIX}/bin/${CHOST}-c89 <<"EOF"
#!/bin/sh
fl="-std=c89"
for opt; do
  case "$opt" in
    -ansi|-std=c89|-std=iso9899:1990) fl="";;
    -std=*) echo "`basename $0` called with non ANSI/ISO C option $opt" >&2
      exit 1;;
  esac
done
exec gcc $fl ${1+"$@"}
EOF

  cat > ${PREFIX}/bin/${CHOST}-c99 <<"EOF"
#!/bin/sh
fl="-std=c99"
for opt; do
  case "$opt" in
    -std=c99|-std=iso9899:1999) fl="";;
    -std=*) echo "`basename $0` called with non ISO C99 option $opt" >&2
      exit 1;;
  esac
done
exec gcc $fl ${1+"$@"}
EOF

  chmod 755 ${PREFIX}/bin/${CHOST}-c{8,9}9

  rm ${PREFIX}/bin/${CHOST}-gcc-${PKG_VERSION}

popd

# generate specfile so that we can patch loader link path
# link_libgcc should have the gcc's own libraries by default (-R)
# so that LD_LIBRARY_PATH isn't required for basic libraries.
#
# GF method here to create specs file and edit it.  The other methods
# tried had no effect on the result.  including:
#   setting LINK_LIBGCC_SPECS on configure
#   setting LINK_LIBGCC_SPECS on make
#   setting LINK_LIBGCC_SPECS in gcc/Makefile
specdir=$PREFIX/lib/gcc/$CHOST/${gcc_version}
if [[ "$build_platform" == "$target_platform" ]]; then
    $PREFIX/bin/${CHOST}-gcc -dumpspecs > $specdir/specs
else
    $BUILD_PREFIX/bin/${CHOST}-gcc -dumpspecs > $specdir/specs
fi

# validate assumption that specs in build/gcc/specs are exactly the
# same as dumped specs so that I don't need to depend on gcc_impl in conda-gcc-specs subpackage
diff -s ${SRC_DIR}/build/gcc/specs $specdir/specs

# make a copy of the specs without our additions so that people can choose not to use them
# by passing -specs=builtin.specs
cp $specdir/specs $specdir/builtin.specs

# modify the default specs to only have %include_noerr that includes an optional conda.specs
# package installable via the conda-gcc-specs package where conda.specs (for $cross_target_platform
# == $target_platform) will add the minimal set of flags for the 'native' toolchains to be useable
# without anything additional set in the enviornment or extra cmdline args.
echo  "%include_noerr <conda.specs>" >> $specdir/specs

# We use double quotes here because we want $PREFIX and $CHOST to be expanded at build time
#   and recorded in the specs file.  It will undergo a prefix replacement when our compiler
#   package is installed.
sed -i -e "/\*link_command:/,+1 s+%.*+& %{!static:-rpath ${PREFIX}/lib}+" $specdir/specs


# Install Runtime Library Exception
install -Dm644 $SRC_DIR/COPYING.RUNTIME \
        ${PREFIX}/share/licenses/gcc/RUNTIME.LIBRARY.EXCEPTION

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

set -x

#${PREFIX}/bin/${CHOST}-gcc "${RECIPE_DIR}"/c11threads.c -std=c11

mkdir -p ${PREFIX}/${CHOST}/lib

if [[ "$target_platform" == "$cross_target_platform" ]]; then
  # making these this way so conda build doesn't muck with them
  pushd ${PREFIX}/${CHOST}/lib
    ln -sf ../../lib/libgomp.so libgomp.so
    for lib in libgcc_s libstdc++ libgfortran libatomic libquadmath libitm lib{a,l,ub,t}san; do
      for f in ${PREFIX}/lib/${lib}.so*; do
        ln -s ../../lib/$(basename $f) ${PREFIX}/${CHOST}/lib/$(basename $f)
      done
    done

    for f in ${PREFIX}/lib/*.spec ${PREFIX}/lib/*.o; do
      mv $f ${PREFIX}/${CHOST}/lib/$(basename $f)
    done
  popd
else
  source ${RECIPE_DIR}/install-libgcc.sh
  for lib in libgcc_s libcc1; do
    mv ${PREFIX}/lib/${lib}.so* ${PREFIX}/${CHOST}/lib/ || true
  done
  rm -f ${PREFIX}/share/info/*.info
fi

if [[ -f ${PREFIX}/lib/libgomp.spec ]]; then
  mv ${PREFIX}/lib/libgomp.spec ${PREFIX}/${CHOST}/lib/libgomp.spec
fi

source ${RECIPE_DIR}/make_tool_links.sh
