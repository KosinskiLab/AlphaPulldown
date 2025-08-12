cmake_minimum_required(VERSION 3.20)

project( test-libsolv LANGUAGES C)

add_executable( test-libsolv test.c)

find_path(libsolv_INCLUDE_DIR NAMES solv/pool.h)
find_library(libsolv_LIBRARY NAMES "${LIB_NAME}")
target_link_libraries(test-libsolv PRIVATE ${libsolv_LIBRARY})
target_include_directories(test-libsolv PRIVATE ${libsolv_INCLUDE_DIR})

enable_testing()
add_test(NAME test-libsolv COMMAND test-libsolv)
