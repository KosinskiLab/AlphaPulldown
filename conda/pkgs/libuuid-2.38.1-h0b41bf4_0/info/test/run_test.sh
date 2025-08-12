

set -ex



test -f ${PREFIX}/lib/libuuid.a
test -f ${PREFIX}/lib/libuuid${SHLIB_EXT}
exit 0
