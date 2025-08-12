IF "%~1"=="wapi" (
:: Compile example that links zlibwapi.lib
cl.exe /I%PREFIX%\Library\include %PREFIX%\Library\lib\zlibwapi.lib /DZLIB_WINAPI test_compile_flags.c
if errorlevel 1 exit 1
) else (
:: Compile example that links zlib.lib
cl.exe /I%PREFIX%\Library\include %PREFIX%\Library\lib\zlib.lib test_compile_flags.c
if errorlevel 1 exit 1
)

:: Run test
.\test_compile_flags.exe
if errorlevel 1 exit 1
