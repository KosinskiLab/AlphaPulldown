@echo ON
mkdir build
cd build
cmake %CMAKE_ARGS% ..  ^
   -DENABLE_DOC=OFF
if %ERRORLEVEL% neq 0 (type CMakeError.log && exit 1)
