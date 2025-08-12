if (-not $Env:SSL_CERT_FILE) {
    $Env:SSL_CERT_FILE = "$Env:CONDA_PREFIX\Library\ssl\cacert.pem"
    $Env:__CONDA_OPENSSL_CERT_FILE_SET = "1"
}
if (-not $Env:SSL_CERT_DIR) {
    $Env:SSL_CERT_DIR = "$Env:CONDA_PREFIX\Library\ssl\certs"
    $Env:__CONDA_OPENSSL_CERT_DIR_SET = "1"
}
