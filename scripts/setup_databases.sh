#!/bin/bash
#
# Copyright (c) 2026 European Molecular Biology Laboratory
#
# Set up the sequence databases AlphaPulldown needs.
#
#   alphafold2  delegates to AlphaFold's own download_all_data.sh
#   alphafold3  delegates to AlphaFold 3's own fetch_databases.sh
#   mmseqs      builds GPU-padded MMseqs2 databases from the AlphaFold 3 FASTAs
#
# Only the third does work no upstream script does. AlphaPulldown's local MMseqs2
# feature stage requires padded databases (mmseqs makepaddedseqdb) and there was
# previously no tooling to produce them, so every site had to work it out by hand.
#
# Nothing here is copied from other projects: the download steps call the upstream
# scripts vendored in AlphaPulldown, and the MMseqs steps are the documented
# createdb/makepaddedseqdb calls.

set -euo pipefail

readonly PROGRAM=${0##*/}
readonly MMSEQS_RELEASE_DEFAULT="18-8cc5c"

# The four protein databases AlphaFold 3 searches, and the AlphaFold 3 FASTA that
# each is built from. Keep in step with mmseqs2_features.databases in config.yaml.
readonly MMSEQS_DATABASES=(
  "uniref90:uniref90_2022_05.fa"
  "mgnify:mgy_clusters_2022_05.fa"
  "small_bfd:bfd-first_non_consensus_sequences.fasta"
  "uniprot:uniprot_all_2021_04.fa"
)

usage() {
  cat <<EOF
Usage: $PROGRAM --dest DIR [--alphafold2] [--alphafold3] [--mmseqs] [options]

At least one of --alphafold2, --alphafold3, --mmseqs (or --all).

  --dest DIR           where databases live. Required.
  --all                everything below
  --alphafold2         AlphaFold 2 databases (~2.6 TB, or ~600 GB with --reduced)
  --alphafold3         AlphaFold 3 databases (~630 GB)
  --mmseqs             GPU-padded MMseqs2 databases built from the AF3 FASTAs
  --reduced            AlphaFold 2 in reduced_dbs mode (small_bfd instead of BFD)
  --mmseqs-source DIR  AF3 FASTAs to build from (default: DIR passed to --dest)
  --mmseqs-binary PATH mmseqs executable (default: download release $MMSEQS_RELEASE_DEFAULT)
  --threads N          threads for MMseqs2 (default: all available)
  --keep-unpadded      keep the intermediate unpadded databases (roughly doubles size)
  --dry-run            print what would happen and exit
  -h, --help           this message

Sizes, measured on the AlphaFold 3 database set:

  database    FASTA    padded    build time
  uniref90     67 GB     79 GB      ~35 min
  mgnify      120 GB    170 GB     ~100 min
  small_bfd    17 GB     22 GB      ~16 min
  uniprot     102 GB    117 GB      ~55 min

Peak host RAM when searching tracks the LARGEST database rather than their total,
because they are searched one after another: budget about 0.85x the padded size of
the biggest one. Building needs far less; ~120 GB is comfortable.

Example:
  $PROGRAM --dest /scratch/AlphaFold_DBs/3.0.0 --alphafold3 --mmseqs
EOF
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf '%s: %s\n' "$PROGRAM" "$*" >&2; exit 1; }

DEST=""; DO_AF2=0; DO_AF3=0; DO_MMSEQS=0; REDUCED=0; DRY_RUN=0
MMSEQS_SOURCE=""; MMSEQS_BINARY=""; THREADS=""; KEEP_UNPADDED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST=${2:?--dest needs a directory}; shift 2 ;;
    --all) DO_AF2=1; DO_AF3=1; DO_MMSEQS=1; shift ;;
    --alphafold2) DO_AF2=1; shift ;;
    --alphafold3) DO_AF3=1; shift ;;
    --mmseqs) DO_MMSEQS=1; shift ;;
    --reduced) REDUCED=1; shift ;;
    --mmseqs-source) MMSEQS_SOURCE=${2:?}; shift 2 ;;
    --mmseqs-binary) MMSEQS_BINARY=${2:?}; shift 2 ;;
    --threads) THREADS=${2:?}; shift 2 ;;
    --keep-unpadded) KEEP_UNPADDED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

[[ -n "$DEST" ]] || { usage >&2; die "--dest is required"; }
(( DO_AF2 || DO_AF3 || DO_MMSEQS )) || { usage >&2; die "nothing to do"; }
: "${MMSEQS_SOURCE:=$DEST}"
: "${THREADS:=$(nproc 2>/dev/null || echo 8)}"

run() {
  if (( DRY_RUN )); then printf '  would run: %s\n' "$*" >&2; else "$@"; fi
}

# --- the vendored upstream download scripts ----------------------------------
# AlphaPulldown vendors AlphaFold 2 and 3, each with its own downloader, and this
# script lives beside them. Use those rather than reimplementing URLs that upstream
# changes without notice.
readonly REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

find_upstream() {
  local relative=$1
  local candidate="${ALPHAPULLDOWN_SRC:-$REPO_ROOT}/$relative"
  [[ -f "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; }
  return 1
}

setup_alphafold2() {
  local script
  script=$(find_upstream "alphafold/scripts/download_all_data.sh") || die \
"cannot find alphafold/scripts/download_all_data.sh.
AlphaFold 2 is a git submodule; run 'git submodule update --init' in the
AlphaPulldown checkout, or download the databases following
https://github.com/google-deepmind/alphafold"
  local mode="full_dbs"; (( REDUCED )) && mode="reduced_dbs"
  log "AlphaFold 2 databases -> $DEST ($mode)"
  run bash "$script" "$DEST" "$mode"
}

setup_alphafold3() {
  local script
  script=$(find_upstream "alphafold3/fetch_databases.sh") || die \
"cannot find alphafold3/fetch_databases.sh.
AlphaFold 3 is a git submodule; run 'git submodule update --init' in the
AlphaPulldown checkout, or download the databases following
https://github.com/google-deepmind/alphafold3"
  log "AlphaFold 3 databases -> $DEST"
  run bash "$script" "$DEST"
}

# --- MMseqs2 ------------------------------------------------------------------
ensure_mmseqs() {
  if [[ -n "$MMSEQS_BINARY" ]]; then
    [[ -x "$MMSEQS_BINARY" ]] || die "not executable: $MMSEQS_BINARY"
    return
  fi
  # The prediction container bundles one; prefer it when this runs inside.
  if [[ -x /opt/mmseqs/bin/mmseqs ]]; then
    MMSEQS_BINARY=/opt/mmseqs/bin/mmseqs
    log "using the MMseqs2 bundled in this image: $MMSEQS_BINARY"
    return
  fi
  local tools="$DEST/tools"
  MMSEQS_BINARY="$tools/mmseqs/bin/mmseqs"
  [[ -x "$MMSEQS_BINARY" ]] && { log "using $MMSEQS_BINARY"; return; }
  log "downloading MMseqs2 $MMSEQS_RELEASE_DEFAULT (GPU build) -> $tools"
  run mkdir -p "$tools"
  run curl -fsSL -o "$tools/mmseqs.tar.gz" \
    "https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS_RELEASE_DEFAULT}/mmseqs-linux-gpu.tar.gz"
  run tar -xzf "$tools/mmseqs.tar.gz" -C "$tools"
  run rm -f "$tools/mmseqs.tar.gz"
  (( DRY_RUN )) || [[ -x "$MMSEQS_BINARY" ]] || die "MMseqs2 download did not produce $MMSEQS_BINARY"
}

setup_mmseqs() {
  ensure_mmseqs
  local out="$DEST/mmseqs"
  run mkdir -p "$out"
  log "building GPU-padded MMseqs2 databases -> $out (threads=$THREADS)"

  local entry name fasta src padded plain
  for entry in "${MMSEQS_DATABASES[@]}"; do
    name=${entry%%:*}; fasta=${entry#*:}
    src="$MMSEQS_SOURCE/$fasta"
    plain="$out/$name"
    padded="${plain}_gpu"

    if [[ -e "${padded}.dbtype" ]]; then
      log "$name: already built, skipping"
      continue
    fi
    if [[ ! -f "$src" ]] && (( ! DRY_RUN )); then
      log "$name: SKIPPED, no source FASTA at $src"
      continue
    fi

    log "$name: createdb"
    run "$MMSEQS_BINARY" createdb "$src" "$plain" --threads "$THREADS"
    log "$name: makepaddedseqdb"
    run "$MMSEQS_BINARY" makepaddedseqdb "$plain" "$padded" --threads "$THREADS"

    # makepaddedseqdb can fail and still leave files behind, which then look like a
    # database until a search reports "Input ... does not exist". Check the marker.
    if (( ! DRY_RUN )) && [[ ! -e "${padded}.dbtype" ]]; then
      die "$name: makepaddedseqdb produced no ${padded}.dbtype - the build failed"
    fi
    if (( ! KEEP_UNPADDED )) && (( ! DRY_RUN )); then
      log "$name: removing the intermediate unpadded database"
      rm -f "$plain" "$plain".* "${plain}_h" "${plain}_h".*
    fi
    log "$name: done ($(du -shc "${padded}"* 2>/dev/null | tail -1 | cut -f1))"
  done

  cat <<EOF

Add to config.yaml under mmseqs2_features (see the ADVANCED section):

  databases:
EOF
  for entry in "${MMSEQS_DATABASES[@]}"; do
    name=${entry%%:*}
    printf '    %s: {path: %s/%s_gpu, identifier: %s-CHANGE_ME}\n' \
      "$name" "$out" "$name" "$name"
  done
  cat <<EOF

Set each identifier to something that changes whenever you rebuild that database:
cache validity depends on it, so reusing an identifier after a rebuild silently
serves stale MSAs.
EOF
}

(( DO_AF2 )) && setup_alphafold2
(( DO_AF3 )) && setup_alphafold3
(( DO_MMSEQS )) && setup_mmseqs
log "done"
