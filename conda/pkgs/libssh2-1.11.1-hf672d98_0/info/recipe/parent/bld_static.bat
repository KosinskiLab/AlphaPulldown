:: copy files which are missing from the release tarball
:: see: https://github.com/libssh2/libssh2/issues/379
:: TODO: remove this in the 1.9.1 or later releases
set PATH=%PREFIX%\cmake-bin\bin;%PATH%

cd build_static

ninja install
IF %ERRORLEVEL% NEQ 0 exit 1

:: rename lib file to *_static.lib
REN %PREFIX%\Library\lib\libssh2.lib libssh2_static.lib