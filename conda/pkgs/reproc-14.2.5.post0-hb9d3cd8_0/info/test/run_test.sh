

set -ex



test -f ${PREFIX}/include/reproc/run.h
test -f ${PREFIX}/lib/libreproc${SHLIB_EXT}
test -f ${PREFIX}/lib/cmake/reproc/reproc-config.cmake
test ! -f ${PREFIX}/include/reproc++/run.hpp
test ! -f ${PREFIX}/lib/libreproc++${SHLIB_EXT}
test ! -f ${PREFIX}/lib/libreproc.a
test ! -f ${PREFIX}/lib/libreproc++.a
test ! -f ${PREFIX}/lib/cmake/reproc++/reproc++-config.cmake
cmake -G Ninja -S test-c/ -B build-test-c/ ${CMAKE_ARGS}
cmake --build build-test-c/
cmake --build build-test-c/ --target test
exit 0
