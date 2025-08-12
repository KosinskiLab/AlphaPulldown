@echo on

cd build

ninja install
if errorlevel 1 exit /b 1

if %PKG_NAME% NEQ xz-tools (
    del %LIBRARY_PREFIX%\bin\xzdec.exe
    if errorlevel 1 exit /b 1

    del %LIBRARY_PREFIX%\bin\lzmadec.exe
    if errorlevel 1 exit /b 1

    del %LIBRARY_PREFIX%\bin\lzmainfo.exe
    if errorlevel 1 exit /b 1

    del %LIBRARY_PREFIX%\bin\xz.exe
    if errorlevel 1 exit /b 1
)
