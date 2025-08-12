

set -ex



test -f "${PREFIX}/ssl/cacert.pem"
test -f "${PREFIX}/ssl/cert.pem"
curl --cacert "${PREFIX}/ssl/cacert.pem" https://www.google.com
exit 0
