

set -ex



test -f ${PREFIX}/lib/libkeyutils.so
keyctl --version | grep '1.6.1'
exit 0
