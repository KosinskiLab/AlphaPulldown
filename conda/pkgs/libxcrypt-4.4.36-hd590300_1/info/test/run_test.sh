

set -ex



test -f "${PREFIX}/include/crypt.h"
test -f "${PREFIX}/lib/libcrypt${SHLIB_EXT}"
test -f "${PREFIX}/lib/libcrypt.so.2"
test -f "${PREFIX}/lib/pkgconfig/libxcrypt.pc"
test ! -f "${PREFIX}/lib/libcrypt.so.1"
test ! -f "${PREFIX}/lib/libcrypt.1.dylib"
exit 0
