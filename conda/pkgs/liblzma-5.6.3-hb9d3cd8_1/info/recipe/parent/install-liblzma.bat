@echo on

cd build

if not exist %LIBRARY_PREFIX%\bin md %LIBRARY_PREFIX%\bin
if errorlevel 1 exit 1

dir

copy liblzma.dll %LIBRARY_PREFIX%\bin
if errorlevel 1 exit 1
