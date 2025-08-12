if [[ "${SSL_CERT_FILE:-}" == "" ]]; then
    export SSL_CERT_FILE="${CONDA_PREFIX}\\Library\\ssl\\cacert.pem"
    export __CONDA_OPENSSL_CERT_FILE_SET="1"
fi
if [[ "${SSL_CERT_DIR:-}" == "" ]]; then
    export SSL_CERT_DIR="${CONDA_PREFIX}\\Library\\ssl\\certs"
    export __CONDA_OPENSSL_CERT_DIR_SET="1"
fi
