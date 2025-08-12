

set -ex



curl-config --features
curl-config --protocols
test -f ${PREFIX}/lib/libcurl${SHLIB_EXT}
test ! -f ${PREFIX}/lib/libcurl.a
CURL_SSL_BACKENDS=$(curl-config --ssl-backends)
if ! echo $CURL_SSL_BACKENDS | grep -q "OpenSSL"; then exit 1; fi
exit 0
