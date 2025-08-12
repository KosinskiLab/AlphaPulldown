

set -ex



test -e $PREFIX/lib/libffi${SHLIB_EXT}
test -e $PREFIX/lib/libffi.a
test -e $PREFIX/include/ffi.h
test -e $PREFIX/include/ffitarget.h
exit 0
