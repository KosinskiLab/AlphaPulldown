@echo on
setlocal enabledelayedexpansion

nmake install
if %ERRORLEVEL% neq 0 exit 1

:: don't include html docs that get installed
rd /s /q %LIBRARY_PREFIX%\html

:: install pkgconfig metadata (useful for downstream packages);
:: adapted from inspecting the conda-forge .pc files for unix, as well as
:: https://github.com/microsoft/vcpkg/blob/master/ports/openssl/install-pc-files.cmake
mkdir %LIBRARY_PREFIX%\lib\pkgconfig
for %%F in (openssl libssl libcrypto) DO (
    echo prefix=%LIBRARY_PREFIX:\=/% > %%F.pc
    type %RECIPE_DIR%\win_pkgconfig\%%F.pc.in >> %%F.pc
    echo Version: %PKG_VERSION% >> %%F.pc
    copy %%F.pc %LIBRARY_PREFIX%\lib\pkgconfig\%%F.pc
)

mkdir %LIBRARY_PREFIX%\ssl\certs
type NUL > %LIBRARY_PREFIX%\ssl\certs\.keep

REM Install step
rem copy out32dll\openssl.exe %PREFIX%\openssl.exe
rem copy out32\ssleay32.lib %LIBRARY_LIB%\ssleay32_static.lib
rem copy out32\libeay32.lib %LIBRARY_LIB%\libeay32_static.lib
rem copy out32dll\ssleay32.lib %LIBRARY_LIB%\ssleay32.lib
rem copy out32dll\libeay32.lib %LIBRARY_LIB%\libeay32.lib
rem copy out32dll\ssleay32.dll %LIBRARY_BIN%\ssleay32.dll
rem copy out32dll\libeay32.dll %LIBRARY_BIN%\libeay32.dll
rem mkdir %LIBRARY_INC%\openssl
rem xcopy /S inc32\openssl\*.* %LIBRARY_INC%\openssl\

:: Copy the [de]activate scripts to %PREFIX%\etc\conda\[de]activate.d.
:: This will allow them to be run on environment activation.
for %%F in (activate deactivate) DO (
    if not exist %PREFIX%\etc\conda\%%F.d mkdir %PREFIX%\etc\conda\%%F.d
    copy "%RECIPE_DIR%\%%F-win.bat" "%PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F-win.bat"
    copy "%RECIPE_DIR%\%%F-win.ps1" "%PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F-win.ps1"
    :: Copy unix shell activation scripts, needed by Windows Bash users
    copy "%RECIPE_DIR%\%%F-win.sh" "%PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F-win.sh"
)
