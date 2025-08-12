

set -ex



test ! -f ${PREFIX}/lib/libz.a
test ! -f ${PREFIX}/lib/libz${SHLIB_EXT}
test ! -f ${PREFIX}/include/zlib.h
exit 0
