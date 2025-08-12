REM edited from the one from Anaconda

REM Build step
nmake -f makefile.msc lib || exit 1

REM Install step
copy bzlib.h %LIBRARY_INC% || exit 1
copy libbz2_static.lib %LIBRARY_LIB% || exit 1
copy libbz2.lib %LIBRARY_LIB% || exit 1
copy libbz2.dll %LIBRARY_BIN% || exit 1

REM Some packages expect 'bzip2.lib', so make copies
copy libbz2_static.lib %LIBRARY_LIB%\bzip2_static.lib || exit 1
copy libbz2.lib %LIBRARY_LIB%\bzip2.lib || exit 1
