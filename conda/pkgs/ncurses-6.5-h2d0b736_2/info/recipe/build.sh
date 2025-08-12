#!/bin/bash
set -ex

# Get an updated config.sub and config.guess
# Running autoreconf messes up the build so just copy these two files
cp $BUILD_PREFIX/share/libtool/build-aux/config.* .

if [[ "$CONDA_BUILD_CROSS_COMPILATION" == "1" ]]; then
    export BUILD_CC=${CC_FOR_BUILD}
fi

for USE_WIDEC in false true;
do
    WIDEC_OPT="--disable-widec"
    w=""
    if [ "${USE_WIDEC}" = true ];
    then
        WIDEC_OPT="--enable-widec"
        w="w"
    fi

    export PKG_CONFIG_LIBDIR=$PREFIX/lib/pkgconfig

    sh ./configure \
	    --prefix=$PREFIX \
	    --without-debug \
	    --without-ada \
	    --without-manpages \
	    --with-shared \
	    --with-pkg-config \
	    --with-pkg-config-libdir=$PREFIX/lib/pkgconfig \
	    --disable-overwrite \
	    --enable-symlinks \
	    --enable-termcap \
	    --enable-pc-files \
	    --with-termlib \
	    --with-versioned-syms \
            --disable-mixed-case \
	    $WIDEC_OPT

    if [[ "$target_platform" == osx* ]]; then
        # When linking libncurses*.dylib, reexport libtinfo[w] so that later
        # client code linking against just -lncurses[w] also gets -ltinfo[w].
        sed -i.orig '/^SHLIB_LIST/s/-ltinfo/-Wl,-reexport&/' ncurses/Makefile
    fi

    make -j ${CPU_COUNT}
    make INSTALL="${BUILD_PREFIX}/bin/install -c  --strip-program=${STRIP}" install
    make clean
    make distclean

    # Provide headers in `$PREFIX/include` and
    # symlink them in `$PREFIX/include/ncurses`
    # and in `$PREFIX/include/ncursesw`.
    HEADERS_DIR="${PREFIX}/include/ncurses"
    if [ "${USE_WIDEC}" = true ];
    then
        HEADERS_DIR="${PREFIX}/include/ncursesw"
    fi
    for HEADER in $(ls $HEADERS_DIR);
    do
        mv "${HEADERS_DIR}/${HEADER}" "${PREFIX}/include/${HEADER}"
        ln -s "${PREFIX}/include/${HEADER}" "${HEADERS_DIR}/${HEADER}"
    done

    if [[ "$target_platform" != osx* ]]; then
        # Replace the installed libncurses[w].so with a linker script
        # so that linking against it also brings in -ltinfo[w].
        DEVLIB=$PREFIX/lib/libncurses$w.so
        RUNLIB=$(basename $(readlink $DEVLIB))
        rm $DEVLIB
        echo "INPUT($RUNLIB -ltinfo$w)" > $DEVLIB
    fi
done

# Explicitly delete static libraries
for LIB_NAME in libncurses libtinfo libform libmenu libpanel; do
    rm ${PREFIX}/lib/${LIB_NAME}.a
    rm ${PREFIX}/lib/${LIB_NAME}w.a
done
