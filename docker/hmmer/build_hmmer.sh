#!/usr/bin/env bash
# Build HMMER 3.4 with AlphaFold 3's jackhmmer patches and install it into PREFIX.
#
# The patches are docker/*.patch from google-deepmind/alphafold3@03f4f48 (Apache-2.0):
#   jackhmmer_seq_limit.patch       adds --seq_limit, used by AlphaFold 3's jackhmmer wrapper
#   easel_fast_fasta_parsing.patch  faster FASTA parsing (EddyRivasLab/easel#83)
#   jackhmmer_parallel_fasta.patch  every worker reads its own part of the database
#                                   (EddyRivasLab/hmmer#353)
set -euo pipefail

prefix="${1:?usage: build_hmmer.sh PREFIX}"
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

url=https://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
if command -v curl >/dev/null; then
  curl -fsSL "$url" -o hmmer-3.4.tar.gz
else
  wget -q "$url" -O hmmer-3.4.tar.gz
fi
cp "$here"/*.patch .
sha256sum -c "$here/SHA256SUMS"

tar -xzf hmmer-3.4.tar.gz
# In AlphaFold 3's order: both jackhmmer patches edit src/jackhmmer.c.
for p in jackhmmer_seq_limit easel_fast_fasta_parsing jackhmmer_parallel_fasta; do
  patch -p0 --forward --fuzz=0 < "$p.patch"
done
(
  cd hmmer-3.4
  ./configure --prefix="$prefix"
  make -j"$(nproc)"
  make install
  make -C easel install
)

# --cpu 0 reads the database serially; --cpu 4 takes the patched path where each
# worker reads its own 512 KB chunks. Copies of the query at the start, middle and
# end of a ~6 MB database must give the same hits and target count both ways.
bin="$prefix/bin"
"$bin/esl-shuffle" -G --amino -N 1 -L 400 --seed 1 | sed '1s/.*/>query/' > query.fa
"$bin/esl-shuffle" -G --amino -N 10000 -L 300 --seed 2 > a.fa
"$bin/esl-shuffle" -G --amino -N 10000 -L 300 --seed 3 > b.fa
{
  sed '1s/.*/>q_start first copy/' query.fa
  cat a.fa
  sed '1s/.*/>q_mid second copy/' query.fa
  cat b.fa
  sed '1s/.*/>q_end third copy/' query.fa
} > db.fa
for cpu in 0 4; do
  "$bin/jackhmmer" --cpu "$cpu" -N 2 -o "out.$cpu" --tblout "tbl.$cpu" -A "sto.$cpu" \
    query.fa db.fa
  grep -v '^#' "tbl.$cpu" > "hits.$cpu"
  grep -E '^(Target sequences|Passed .* filter)' "out.$cpu" >> "hits.$cpu"
done
grep -q '^Target sequences: *20003 ' hits.4
test "$(grep -c '^q_' hits.4)" -eq 3
cmp hits.0 hits.4
cmp sto.0 sto.4
echo "HMMER 3.4 with jackhmmer patches installed in $prefix; threaded and serial searches agree"
