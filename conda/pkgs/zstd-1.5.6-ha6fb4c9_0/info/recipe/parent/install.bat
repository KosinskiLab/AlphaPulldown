@echo on

if "%PKG_NAME%" == "zstd-static" (
  set ZSTD_BUILD_STATIC=ON
  REM cannot build CLI without shared lib
  set ZSTD_BUILD_SHARED=ON
  echo "static build"
) else (
  set ZSTD_BUILD_STATIC=OFF
  set ZSTD_BUILD_SHARED=ON
)

pushd "%SRC_DIR%"\build\cmake
cmake -GNinja %CMAKE_ARGS% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
    -DCMAKE_INSTALL_LIBDIR="lib" ^
    -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" ^
    -DZSTD_BUILD_SHARED=%ZSTD_BUILD_SHARED% ^
    -DZSTD_BUILD_STATIC=%ZSTD_BUILD_STATIC% ^
    -DZSTD_PROGRAMS_LINK_SHARED=ON ^
    .
if %ERRORLEVEL% neq 0 exit 1

cmake --build . --target install
if %ERRORLEVEL% neq 0 exit 1

:: duplicate DLL (+ importlib) to also have files with "lib" prefix
copy %PREFIX%\Library\bin\zstd.dll %PREFIX%\Library\bin\libzstd.dll
if %ERRORLEVEL% neq 0 exit 1
copy %PREFIX%\Library\lib\zstd.lib %PREFIX%\Library\lib\libzstd.lib
if %ERRORLEVEL% neq 0 exit 1

if "%PKG_NAME%" == "zstd-static" (
  copy %PREFIX%\Library\lib\zstd_static.lib %PREFIX%\Library\lib\libzstd_static.lib
)
