if /I "%PKG_NAME%" == "libsolv" (

    cmake -B build/ ^
        -G "Ninja" ^
        -D ENABLE_CONDA=ON ^
        -D MULTI_SEMANTICS=ON ^
        -D WITHOUT_COOKIEOPEN=ON ^
        -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
        -D DISABLE_SHARED=OFF ^
        -D ENABLE_STATIC=OFF ^
        %CMAKE_ARGS%
    if errorlevel 1 exit 1

    cmake --build build/ --parallel %CPU_COUNT%
    if errorlevel 1 exit 1

    cmake --install build/
    if errorlevel 1 exit 1

)

if /I "%PKG_NAME%" == "libsolv-static" (

    cmake -B build_static/ ^
        -G "Ninja" ^
        -D ENABLE_CONDA=ON ^
        -D MULTI_SEMANTICS=ON ^
        -D WITHOUT_COOKIEOPEN=ON ^
        -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
        -D CMAKE_RELEASE_POSTFIX="_static" ^
        -D DISABLE_SHARED=ON ^
        -D ENABLE_STATIC=ON ^
        %CMAKE_ARGS%
    if errorlevel 1 exit 1

    cmake --build build_static/ --parallel %CPU_COUNT%
    if errorlevel 1 exit 1

    cmake --install build_static/
    if errorlevel 1 exit 1

)
