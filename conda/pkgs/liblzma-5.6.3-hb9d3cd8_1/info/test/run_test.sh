

set -ex



test ! -f ${PREFIX}/lib/liblzma.a
test ! -f ${PREFIX}/lib/liblzma${SHLIB_EXT}
test -f ${PREFIX}/lib/liblzma.so.*.*
exit 0
