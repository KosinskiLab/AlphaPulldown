if ($Env:__CONDA_OPENSSL_CERT_FILE_SET -eq "1") {
    Remove-Item -Path Env:\SSL_CERT_FILE
    Remove-Item -Path Env:\__CONDA_OPENSSL_CERT_FILE_SET
}
if ($Env:__CONDA_OPENSSL_CERT_DIR_SET -eq "1") {
    Remove-Item -Path Env:\SSL_CERT_DIR
    Remove-Item -Path Env:\__CONDA_OPENSSL_CERT_DIR_SET
}
