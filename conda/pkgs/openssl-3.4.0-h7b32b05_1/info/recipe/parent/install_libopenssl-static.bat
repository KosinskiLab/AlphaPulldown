if not exist "%LIBRARY_LIB%" mkdir %LIBRARY_LIB%
copy libcrypto_static.lib %LIBRARY_LIB%\libcrypto_static.lib
copy libssl_static.lib %LIBRARY_LIB%\libssl_static.lib
