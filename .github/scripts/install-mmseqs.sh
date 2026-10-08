#!/usr/bin/env bash
# Install the MMseqs2 release the prediction images bundle, for the local-MMseqs2
# integration tests, and point MMSEQS_INTEGRATION_BINARY at it.
set -euo pipefail
dockerfile="$(dirname "$0")/../../docker/alphafold2.dockerfile"
version="$(sed -n 's/^ARG MMSEQS_VERSION=//p' "$dockerfile")"
sha256="$(sed -n 's/^ARG MMSEQS_GPU_SHA256=//p' "$dockerfile")"
commit="$(sed -n 's/^ARG MMSEQS_COMMIT=//p' "$dockerfile")"
archive="$(mktemp)"
curl -fsSL "https://github.com/soedinglab/MMseqs2/releases/download/${version}/mmseqs-linux-gpu.tar.gz" -o "$archive"
echo "${sha256}  ${archive}" | sha256sum -c -
tar -xzf "$archive" -C "$HOME"
rm -f "$archive"
test "$("$HOME/mmseqs/bin/mmseqs" version)" = "$commit"
echo "MMSEQS_INTEGRATION_BINARY=$HOME/mmseqs/bin/mmseqs" >> "$GITHUB_ENV"
