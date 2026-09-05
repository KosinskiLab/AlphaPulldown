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
# Runs standalone: fetch it with curl and run it. Everything happens inside the
# prediction container, which already carries the AlphaFold downloaders and a pinned
# MMseqs2, so no AlphaPulldown checkout and no host-side tooling are needed. Users who
# deployed only the Snakemake workflow (snakedeploy materialises workflow/ and config/
# and nothing else) can therefore still use it.
#
# Nothing here is copied from other projects: the download steps call the upstream
# scripts shipped in the container, and the MMseqs steps are the documented
# createdb/makepaddedseqdb calls.

set -euo pipefail

readonly PROGRAM=${0##*/}
readonly MMSEQS_RELEASE_DEFAULT="18-8cc5c"
readonly CONTAINER_DEFAULT="docker://kosinskilab/alphafold3:latest"

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
  --container URI      prediction container providing the downloaders and MMseqs2
                       (default: $CONTAINER_DEFAULT)
  --mmseqs-binary PATH host mmseqs to use instead of the container's
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
MMSEQS_SOURCE=""; MMSEQS_BINARY=""; THREADS=""; KEEP_UNPADDED=0; CONTAINER="$CONTAINER_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST=${2:?--dest needs a directory}; shift 2 ;;
    --all) DO_AF2=1; DO_AF3=1; DO_MMSEQS=1; shift ;;
    --alphafold2) DO_AF2=1; shift ;;
    --alphafold3) DO_AF3=1; shift ;;
    --mmseqs) DO_MMSEQS=1; shift ;;
    --reduced) REDUCED=1; shift ;;
    --mmseqs-source) MMSEQS_SOURCE=${2:?}; shift 2 ;;
    --container) CONTAINER=${2:?}; shift 2 ;;
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

# --- running things inside the prediction container ---------------------------
# The container ships the AlphaFold 2 and 3 download scripts under /app/AlphaPulldown
# and a pinned MMseqs2 at /opt/mmseqs/bin/mmseqs, so nothing has to be installed here.
CONTAINER_RUNTIME=""
ensure_container() {
  [[ -n "$CONTAINER_RUNTIME" ]] && return
  CONTAINER_RUNTIME=$(command -v apptainer || command -v singularity) || die \
"neither apptainer nor singularity found. One of them is needed to run the
prediction container; install it, or pass --mmseqs-binary and download the
AlphaFold databases yourself."
  log "using $CONTAINER_RUNTIME with $CONTAINER"
}

in_container() {  # in_container [--nv] -- command...
  local nv=()
  [[ ${1:-} == "--nv" ]] && { nv=(--nv); shift; }
  ensure_container
  run "$CONTAINER_RUNTIME" exec "${nv[@]}" \
    --bind "$DEST:$DEST" --bind "$MMSEQS_SOURCE:$MMSEQS_SOURCE" \
    "$CONTAINER" "$@"
}

setup_alphafold2() {
  # AlphaFold 2's downloader needs aria2c and rsync, which the container does not
  # carry, so this one genuinely runs on the host.
  local missing=()
  for tool in aria2c rsync; do command -v "$tool" >/dev/null || missing+=("$tool"); done
  if (( ${#missing[@]} )); then
    die "AlphaFold 2 database download needs ${missing[*]} on this machine \
(the container does not provide them). Install them, or download the databases
following https://github.com/google-deepmind/alphafold"
  fi
  local script="$DEST/.alphafold2_download_all_data.sh"
  log "extracting AlphaFold 2 downloader from the container"
  ensure_container
  run mkdir -p "$DEST"
  (( DRY_RUN )) || "$CONTAINER_RUNTIME" exec "$CONTAINER" \
    cat /app/AlphaPulldown/alphafold/scripts/download_all_data.sh > "$script"
  local mode="full_dbs"; (( REDUCED )) && mode="reduced_dbs"
  log "AlphaFold 2 databases -> $DEST ($mode)"
  run bash "$script" "$DEST" "$mode"
}

setup_alphafold3() {
  log "AlphaFold 3 databases -> $DEST (inside the container)"
  run mkdir -p "$DEST"
  in_container bash /app/AlphaPulldown/alphafold3/fetch_databases.sh "$DEST"
}

# --- MMseqs2 ------------------------------------------------------------------
# Runs one mmseqs command, in the container by default.
mmseqs_run() {
  if [[ -n "$MMSEQS_BINARY" ]]; then
    run "$MMSEQS_BINARY" "$@"
  else
    in_container /opt/mmseqs/bin/mmseqs "$@"
  fi
}

ensure_mmseqs() {
  if [[ -n "$MMSEQS_BINARY" ]]; then
    [[ -x "$MMSEQS_BINARY" ]] || die "not executable: $MMSEQS_BINARY"
    log "using host MMseqs2: $MMSEQS_BINARY"
  else
    ensure_container
    log "using the MMseqs2 bundled in $CONTAINER (pinned $MMSEQS_RELEASE_DEFAULT)"
  fi
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
    mmseqs_run createdb "$src" "$plain" --threads "$THREADS"
    log "$name: makepaddedseqdb"
    mmseqs_run makepaddedseqdb "$plain" "$padded" --threads "$THREADS"

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
