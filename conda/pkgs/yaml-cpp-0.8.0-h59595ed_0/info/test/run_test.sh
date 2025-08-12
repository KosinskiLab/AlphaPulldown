

set -ex



test -f "${PREFIX}/lib/libyaml-cpp${SHLIB_EXT}"
test ! -f "${PREFIX}/lib/libyaml-cpp.a"
test -f "${PREFIX}/lib/cmake/yaml-cpp/yaml-cpp-config.cmake"
cmake -G Ninja -S test/ -B build-test/ ${CMAKE_ARGS}
cmake --build build-test/
cmake --build build-test/ --target test
exit 0
