cmake -B build-dyn/ ^
    -G Ninja ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
    -D REPROC_TEST=OFF ^
    -D BUILD_SHARED_LIBS=ON ^
    -D REPROC++=ON ^
    %CMAKE_ARGS%
if errorlevel 1 exit 1
cmake --build build-dyn/ --parallel %CPU_COUNT% --verbose
if errorlevel 1 exit 1

cmake -B build-static/ ^
    -G Ninja ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
    -D CMAKE_RELEASE_POSTFIX="_static" ^
    -D REPROC_TEST=OFF ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D REPROC++=ON ^
    %CMAKE_ARGS%
if errorlevel 1 exit 1
cmake --build build-static/ --parallel %CPU_COUNT% --verbose
if errorlevel 1 exit 1
