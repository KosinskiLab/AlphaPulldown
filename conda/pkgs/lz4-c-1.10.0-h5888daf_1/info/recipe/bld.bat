:: This rougly follow what projects' appveyor file does.


md build
cd build

cmake %CMAKE_ARGS% -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -GNinja ..\build\cmake
if errorlevel 1 exit 1

ninja
if errorlevel 1 exit 1

ninja install
if errorlevel 1 exit 1

:: Test.
if errorlevel 1 exit 1
lz4 -i1b lz4.exe
if errorlevel 1 exit 1
lz4 -i1b5 lz4.exe
if errorlevel 1 exit 1
lz4 -i1b10 lz4.exe
if errorlevel 1 exit 1
lz4 -i1b15 lz4.exe
if errorlevel 1 exit 1
