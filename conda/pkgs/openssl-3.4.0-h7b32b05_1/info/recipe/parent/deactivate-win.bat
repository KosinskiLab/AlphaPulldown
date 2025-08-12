@echo off
if "%__CONDA_OPENSSL_CERT_FILE_SET%" == "1" (
    set SSL_CERT_FILE=
    set __CONDA_OPENSSL_CERT_FILE_SET=
)
if "%__CONDA_OPENSSL_CERT_DIR_SET%" == "1" (
    set SSL_CERT_DIR=
    set __CONDA_OPENSSL_CERT_DIR_SET=
)
