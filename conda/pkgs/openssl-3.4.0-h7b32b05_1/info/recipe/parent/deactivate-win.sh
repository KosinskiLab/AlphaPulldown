if [[ "${__CONDA_OPENSSL_CERT_FILE_SET:-}" == "1" ]]; then
    unset SSL_CERT_FILE
    unset __CONDA_OPENSSL_CERT_FILE_SET
fi
if [[ "${__CONDA_OPENSSL_CERT_DIR_SET:-}" == "1" ]]; then
    unset SSL_CERT_DIR
    unset __CONDA_OPENSSL_CERT_DIR_SET
fi
