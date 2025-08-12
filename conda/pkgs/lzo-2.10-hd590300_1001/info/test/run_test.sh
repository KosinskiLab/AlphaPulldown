

set -ex



test -f ${PREFIX}/include/lzo/lzoconf.h
test -f ${PREFIX}/lib/liblzo2.a
test -f ${PREFIX}/lib/liblzo2${SHLIB_EXT}
exit 0
