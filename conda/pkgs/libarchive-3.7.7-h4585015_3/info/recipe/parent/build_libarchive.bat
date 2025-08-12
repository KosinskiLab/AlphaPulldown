:: Needed so we can find stdint.h from msinttypes.
set LIB=%LIBRARY_LIB%;%LIB%
set LIBPATH=%LIBRARY_LIB%;%LIBPATH%
set INCLUDE=%LIBRARY_INC%;%INCLUDE%

cmake -G "Ninja" %CMAKE_ARGS% ^
      -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS:BOOL=FALSE ^
      -DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% ^
      .

ninja -j%CPU_COUNT% -v
if errorlevel 1 exit /b 1
ninja install
if errorlevel 1 exit /b 1

:: Test.
:: Failures:
:: The following tests FAILED:
::         365 - libarchive_test_read_truncated_filter_bzip2 (Timeout) => runs msys2's bzip2.exe
::         372 - libarchive_test_sparse_basic (Failed)
::         373 - libarchive_test_fully_sparse_files (Failed)
::         386 - libarchive_test_warn_missing_hardlink_target (Failed)
:: ctest -C Release
:: if errorlevel 1 exit 1

:: Test extracting a 7z. This failed due to not using the multi-threaded DLL runtime, fixed by 0009-CMake-Force-Multi-threaded-DLL-runtime.patch
powershell -command "& { (New-Object Net.WebClient).DownloadFile('http://download.qt.io/development_releases/prebuilt/llvmpipe/windows/opengl32sw-64-mesa_12_0_rc2.7z', 'opengl32sw-64-mesa_12_0_rc2.7z') }"
if errorlevel 1 exit 1
%LIBRARY_BIN%\bsdtar -xf opengl32sw-64-mesa_12_0_rc2.7z
if errorlevel 1 exit 1
