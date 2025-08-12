

set -ex



test -f $PREFIX/include/libssh2.h
test -f $PREFIX/include/libssh2_publickey.h
test -f $PREFIX/include/libssh2_sftp.h
test ! -f $PREFIX/lib/libssh2.a
test -f $PREFIX/lib/libssh2${SHLIB_EXT}
exit 0
