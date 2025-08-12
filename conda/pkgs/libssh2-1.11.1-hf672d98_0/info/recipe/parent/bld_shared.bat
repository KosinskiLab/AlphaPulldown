:: copy files which are missing from the release tarball
:: see: https://github.com/libssh2/libssh2/issues/379
:: TODO: remove this in the 1.9.1 or later releases
copy %RECIPE_DIR%\missing_files\*.c tests\

set PATH=%PREFIX%\cmake-bin\bin;%PATH%

mkdir build_static && cd build_static
:: Build static libraries
cmake -GNinja ^
    -D CMAKE_BUILD_TYPE=Release ^
    -D BUILD_SHARED_LIBS=OFF ^
    -D BUILD_STATIC_LIBS=ON ^
    -D CMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% ^
    -D CMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ^
    -D ENABLE_ZLIB_COMPRESSION=ON ^
    -D BUILD_EXAMPLES=OFF ^
    -D BUILD_TESTING=OFF ^
	%SRC_DIR%
IF %ERRORLEVEL% NEQ 0 exit 1

ninja
IF %ERRORLEVEL% NEQ 0 exit 1

cd ..
mkdir build_shared && cd build_shared

:: Build shared libraries
cmake -GNinja ^
    -D CMAKE_BUILD_TYPE=Release ^
    -D BUILD_SHARED_LIBS=ON ^
    -D BUILD_STATIC_LIBS=OFF ^
    -D CMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% ^
    -D CMAKE_PREFIX_PATH=%LIBRARY_PREFIX% ^
    -D ENABLE_ZLIB_COMPRESSION=ON ^
    -D BUILD_EXAMPLES=OFF ^
    -D BUILD_TESTING=OFF ^
	%SRC_DIR%
IF %ERRORLEVEL% NEQ 0 exit 1

ninja
IF %ERRORLEVEL% NEQ 0 exit 1
ninja install
IF %ERRORLEVEL% NEQ 0 exit 1
