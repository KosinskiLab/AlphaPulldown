cmake -B build-static/ ^
    -G Ninja ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D YAML_BUILD_SHARED_LIBS=OFF ^
    -D YAML_CPP_BUILD_TESTS=OFF ^
    -D YAML_MSVC_SHARED_RT=ON ^
    %CMAKE_ARGS%
if errorlevel 1 exit 1

cmake --build build-static/ --parallel %CPU_COUNT% --verbose
if errorlevel 1 exit 1

cmake --install build-static/
if errorlevel 1 exit 1

