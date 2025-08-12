cd build

cmake --build . --target install --config Release
if %ERRORLEVEL% neq 0 (type CMakeError.log && exit 1)
