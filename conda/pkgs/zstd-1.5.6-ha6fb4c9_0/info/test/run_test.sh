

set -ex



test -f ${PREFIX}/lib/libzstd.so
test ! -f ${PREFIX}/lib/libzstd.a
test -f ${PREFIX}/include/zstd.h
zstd -be -i5
test -f ${PREFIX}/lib/pkgconfig/libzstd.pc
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig
pkg-config --cflags libzstd
cd cf_test
cmake $CMAKE_ARGS .
exit 0
