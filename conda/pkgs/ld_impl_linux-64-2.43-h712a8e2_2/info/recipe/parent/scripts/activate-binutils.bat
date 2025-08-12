@echo off
set "CONDA_BACKUP_ADDR2LINE=%ADDR2LINE%"
set "ADDR2LINE=@CHOST@-addr2line.exe"

set "CONDA_BACKUP_AR=%AR%"
set "AR=@CHOST@-ar.exe"

set "CONDA_BACKUP_AS=%AS%"
set "AS=@CHOST@-as.exe"

set "CONDA_BACKUP_CXXFILT=%CXXFILT%"
set "CXXFILT=@CHOST@-c++filt.exe"

set "CONDA_BACKUP_ELFEDIT=%ELFEDIT%"
set "ELFEDIT=@CHOST@-elfedit.exe"

set "CONDA_BACKUP_GPROF=%GPROF%"
set "GPROF=@CHOST@-gprof.exe"

set "CONDA_BACKUP_LD=%LD%"
set "LD=@CHOST@-ld.exe"

set "CONDA_BACKUP_NM=%NM%"
set "NM=@CHOST@-nm.exe"

set "CONDA_BACKUP_OBJCOPY=%OBJCOPY%"
set "OBJCOPY=@CHOST@-objcopy.exe"

set "CONDA_BACKUP_OBJDUMP=%OBJDUMP%"
set "OBJDUMP=@CHOST@-objdump.exe"

set "CONDA_BACKUP_RANLIB=%RANLIB%"
set "RANLIB=@CHOST@-ranlib.exe"

set "CONDA_BACKUP_READELF=%READELF%"
set "READELF=@CHOST@-readelf.exe"

set "CONDA_BACKUP_SIZE=%SIZE%"
set "SIZE=@CHOST@-size.exe"

set "CONDA_BACKUP_STRINGS=%STRINGS%"
set "STRINGS=@CHOST@-strings.exe"

set "CONDA_BACKUP_STRIP=%STRIP%"
set "STRIP=@CHOST@-strip.exe"

set "CONDA_BACKUP_DLLTOOL=%DLLTOOL%"
set "DLLTOOL=@CHOST@-dlltool.exe"

set "CONDA_BACKUP_DLLWRAP=%DLLWRAP%"
set "DLLWRAP=@CHOST@-dllwrap.exe"

set "CONDA_BACKUP_WINDMC=%WINDMC%"
set "WINDMC=@CHOST@-windmc.exe"

set "CONDA_BACKUP_WINDRES=%WINDRES%"
set "WINDRES=@CHOST@-windres.exe"
