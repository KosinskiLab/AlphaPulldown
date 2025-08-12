cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

project(test_shared_cmake LANGUAGES C)
find_package(lz4 REQUIRED CONFIG)

if (NOT TARGET LZ4::lz4_shared)
    message(FATAL_ERROR "Target LZ4::lz4_shared was not defined")
endif()
