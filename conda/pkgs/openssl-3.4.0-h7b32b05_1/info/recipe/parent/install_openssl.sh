#!/bin/bash

PERL=${PREFIX}/bin/perl
make install_sw install_ssldirs

# https://github.com/ContinuumIO/anaconda-issues/issues/6424
if [[ ${HOST} =~ .*linux.* ]]; then
  if execstack -q "${PREFIX}"/lib/libcrypto.so.3.0 | grep -e '^X '; then
    echo "Error, executable stack found in libcrypto.so.3.0"
    exit 1
  fi
fi

# Make sure ${PREFIX}/ssl/certs directory exists
# Otherwise SSL_ERROR_SYSCALL is returned instead of SSL_ERROR_SLL
mkdir -p "${PREFIX}/ssl/certs"
touch "${PREFIX}/ssl/certs/.keep"

# remove the static libraries
rm ${PREFIX}/lib/libcrypto.a
rm ${PREFIX}/lib/libssl.a

