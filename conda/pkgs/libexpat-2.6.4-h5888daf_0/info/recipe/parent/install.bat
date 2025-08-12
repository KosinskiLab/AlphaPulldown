:: Install.
cmake --build . --config Release --target install
if errorlevel 1 exit 1

:: Workaround for package that got build with latest version that renamed these.
copy %LIBRARY_BIN%\\libexpat.dll %LIBRARY_BIN%\\expat.dll || exit 1
copy %LIBRARY_LIB%\\libexpat.lib %LIBRARY_LIB%\\expat.lib || exit 1
