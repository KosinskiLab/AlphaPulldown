cmake_minimum_required(VERSION 3.20)

project(test-reproc-cpp LANGUAGES CXX)

add_executable(test-reproc-cpp test.cpp)

find_package(reproc++ REQUIRED)
target_link_libraries(test-reproc-cpp PRIVATE reproc++)

enable_testing()
add_test(NAME test-reproc-cpp COMMAND test-reproc-cpp)
