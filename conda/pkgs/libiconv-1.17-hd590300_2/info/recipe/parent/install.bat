pushd build

:: Install.
ninja install
if errorlevel 1 exit 1

if %PKG_NAME%==libiconv (
    if exist %LIBRARY_PREFIX%\bin\iconv.exe DEL %LIBRARY_PREFIX%\bin\iconv.exe
) else (
  :: relying on conda-build to deduplicate files
  echo "Keeping all files, conda-build will deduplicate files"
)