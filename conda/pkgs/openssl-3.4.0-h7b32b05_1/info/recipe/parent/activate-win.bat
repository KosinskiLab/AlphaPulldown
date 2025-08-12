@echo off
if "%SSL_CERT_FILE%"=="" (
    set SSL_CERT_FILE=%CONDA_PREFIX%\Library\ssl\cacert.pem
    set "__CONDA_OPENSSL_CERT_FILE_SET=1"
)
if "%SSL_CERT_DIR%"=="" (
    set SSL_CERT_DIR=%CONDA_PREFIX%\Library\ssl\certs
    set "__CONDA_OPENSSL_CERT_DIR_SET=1"
)
