@echo on

cd build

# Build both, shared and static library, so that one can link against the preferred on in this setting.
cmake %CMAKE_ARGS% -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -GNinja ..\build\cmake
if errorlevel 1 exit 1

ninja
if errorlevel 1 exit 1

ninja install
if errorlevel 1 exit 1
