cmake -B build-dyn/ ^
    -G Ninja ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
    -D BUILD_SHARED_LIBS=ON ^
    -D YAML_BUILD_SHARED_LIBS=ON ^
    -D YAML_CPP_BUILD_TESTS=OFF ^
    -D YAML_MSVC_SHARED_RT=ON ^
    %CMAKE_ARGS%
if errorlevel 1 exit 1

cmake --build build-dyn/ --parallel %CPU_COUNT% --verbose
if errorlevel 1 exit 1

cmake --install build-dyn/
if errorlevel 1 exit 1
