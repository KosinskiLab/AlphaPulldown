

set -ex



test -f "${PREFIX}/lib/libsolv${SHLIB_EXT}"
test -f "${PREFIX}/lib/libsolvext${SHLIB_EXT}"
test -f "${PREFIX}/lib/libsolv.so.1"
test -f "${PREFIX}/include/solv/repo.h"
dumpsolv -h
cmake -G Ninja -S test/ -B build/ -D LIB_NAME="libsolv${SHLIB_EXT}" ${CMAKE_ARGS}
cmake --build build/
cmake --build build --target test
exit 0
