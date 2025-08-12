

set -ex



touch checksum.txt
$PREFIX/bin/openssl sha256 checksum.txt
pkg-config --print-errors --exact-version "3.4.0" openssl
if [[ "$(pkg-config --variable=prefix openssl)" == "" ]]; then exit 1; fi
exit 0
