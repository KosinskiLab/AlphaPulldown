@echo on
set CMAKE_CONFIG=Release

mkdir build_%CMAKE_CONFIG%
pushd build_%CMAKE_CONFIG%

if "%PKG_NAME:~-6%" == "static" (
  set CARES_STATIC=ON
  set CARES_SHARED=OFF
) else (
  set CARES_STATIC=OFF
  set CARES_SHARED=ON
)

cmake -G"NMake Makefiles" ^
      -DCARES_STATIC:BOOL=%CARES_STATIC% ^
      -DCARES_SHARED:BOOL=%CARES_SHARED% ^
      -DCMAKE_BUILD_TYPE:STRING=%CMAKE_CONFIG% ^
      -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
      "%SRC_DIR%"

nmake
if errorlevel 1 exit 1
nmake install
if errorlevel 1 exit 1

popd
