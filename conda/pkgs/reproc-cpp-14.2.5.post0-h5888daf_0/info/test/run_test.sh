

set -ex



test -f ${PREFIX}/include/reproc++/run.hpp
test -f ${PREFIX}/lib/libreproc++${SHLIB_EXT}
test -f ${PREFIX}/lib/cmake/reproc++/reproc++-config.cmake
test ! -f ${PREFIX}/lib/libreproc.a
test ! -f ${PREFIX}/lib/libreproc++.a
cmake -G Ninja -S test-cpp/ -B build-test-cpp/ ${CMAKE_ARGS}
cmake --build build-test-cpp/
cmake --build build-test-cpp/ --target test
exit 0
