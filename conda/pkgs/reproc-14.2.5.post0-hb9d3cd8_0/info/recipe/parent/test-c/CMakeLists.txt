cmake_minimum_required(VERSION 3.20)

project(test-reproc LANGUAGES C)

add_executable(test-reproc test.c)

find_package(reproc REQUIRED)
target_link_libraries(test-reproc PRIVATE reproc)

enable_testing()
add_test(NAME test-reproc COMMAND test-reproc)
