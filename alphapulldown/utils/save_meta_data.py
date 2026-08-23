#
# Author: Dingquan Yu
# A script containing utility functions
#
from alphapulldown import __version__ as AP_VERSION
from alphafold.version import __version__ as AF_VERSION
import os
from absl import logging
import subprocess
import datetime
import re
import hashlib
import glob


COMMON_PATTERNS = [
    r"[Vv]ersion\s*(\d+\.\d+(?:\.\d+)?)",  # version 1.0 or version 1.0.0
    r"\b(\d+\.\d+(?:\.\d+)?)\b"  # just the version number 1.0 or 1.0.0
]
BFD_HASH_HHM_FFINDEX = "799f308b20627088129847709f1abed6"

DB_NAME_TO_URL = {
    'UniRef90' : ["ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"],
    'UniRef30' : ["https://storage.googleapis.com/alphafold-databases/v2.3/UniRef30_{release_date}.tar.gz"],
    'MGnify' : ["https://storage.googleapis.com/alphafold-databases/v2.3/mgy_clusters_{release_date}.fa.gz"],
    'BFD' : ["https://storage.googleapis.com/alphafold-databases/casp14_versions/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"],
    'Reduced BFD' : ["https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz"],
    'PDB70' : ["http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz"],
    'UniProt' : [
        "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz",
        "ftp://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
        ],
    'PDB seqres' : ["ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"],
    'ColabFold' : ["https://wwwuser.gwdg.de/~compbiol/colabfold/colabfold_envdb_202108.tar.gz"],
    'NT-RNA': ["https://storage.googleapis.com/alphafold-databases/v3.0/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst"],
    'Rfam': ["https://storage.googleapis.com/alphafold-databases/v3.0/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst"],
    'RNAcentral': ["https://storage.googleapis.com/alphafold-databases/v3.0/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst"],
    'PDB mmCIF': ["https://storage.googleapis.com/alphafold-databases/v3.0/pdb_2022_09_28_mmcif_files.tar.zst"],
}

AF3_BUNDLE_SIGNATURE_FILES = (
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
    "pdb_seqres_2022_09_28.fasta",
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
)


def _looks_like_official_af3_database_bundle(path):
    """Return whether ``path`` belongs to the standard AF3 3.0 DB bundle."""
    path = os.path.abspath(os.fspath(path))
    expected_basenames = {*AF3_BUNDLE_SIGNATURE_FILES, "mmcif_files"}
    if os.path.basename(path.rstrip(os.sep)) not in expected_basenames:
        return False
    bundle_root = os.path.dirname(path.rstrip(os.sep))
    return all(
        os.path.isfile(os.path.join(bundle_root, filename))
        for filename in AF3_BUNDLE_SIGNATURE_FILES
    )


def get_program_version(binary_path):
    """Get version information for a given binary."""
    for cmd_suffix in ["--help", "-h"]:
        cmd = [binary_path, cmd_suffix]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            version = parse_version(result.stdout + result.stderr)
            if version:
                return version
        except Exception as e:
            logging.debug(f"Error while processing {cmd}: {e}")

    logging.warning(f"Cannot parse version from {binary_path}")
    return None


def get_metadata_for_binary(k, v):
    name = k.replace("_binary_path", "")
    return {name: {"version": get_program_version(v)}}


def resolve_database_path(v):
    """Follow symlinks to the file a database flag actually reads.

    Several AF3 database versions are encoded in the *filename*, and AF3's own
    ``fetch_databases.sh`` pins ``pdb_seqres_2022_09_28.fasta``. Sites that
    refresh a database in place commonly keep the pinned name as a symlink to
    the current file, so the configured path and the bytes actually searched
    disagree. Recording the configured name would then put a date in the
    metadata -- and from there into a methods section -- that is simply wrong.

    Returns ``(resolved_path, was_symlink)``.
    """
    try:
        path = os.fspath(v)
    except TypeError:
        return str(v), False
    try:
        resolved = os.path.realpath(path)
    except OSError:
        return path, False
    return resolved, os.path.abspath(path) != resolved


def get_metadata_for_database(k, v):
    name = k.replace("_database_path", "").replace("_dir", "")
    # Version strings are parsed out of file names, so follow symlinks first:
    # a refreshed database hidden behind a pinned name must not be reported
    # under the pinned name's date.
    resolved, via_symlink = resolve_database_path(v)

    if name == "pdb_seqres":
        af3_seqres_release = re.search(
            r"pdb_seqres_(\d{4}_\d{2}_\d{2})", str(resolved)
        )
        if af3_seqres_release:
            version = af3_seqres_release.group(1)
            entry = {
                "release_date": version.replace("_", "-"),
                "version": version,
                "location_url": [
                    "https://storage.googleapis.com/alphafold-databases/"
                    f"v3.0/pdb_seqres_{version}.fasta.zst"
                ],
            }
            if via_symlink:
                entry["configured_path"] = str(v)
                entry["resolved_path"] = resolved
            return {"PDB seqres": entry}

    if name == "rna_central":
        official_bundle = _looks_like_official_af3_database_bundle(v)
        version_match = re.search(
            r"(?:rnacentral|rna_central).*?(\d+_\d+)", str(v), re.IGNORECASE
        )
        if version_match:
            version = version_match.group(1)
        elif official_bundle:
            version = "21_0"
        else:
            version = None
        return {
            "RNAcentral": {
                "release_date": get_last_modified_date(v),
                "version": version,
                "location_url": (
                    DB_NAME_TO_URL["RNAcentral"] if official_bundle else []
                ),
            }
        }

    if name == "template_mmcif":
        official_bundle = _looks_like_official_af3_database_bundle(v)
        release_match = re.search(
            r"(\d{4}[-_]\d{2}[-_]\d{2})", str(v)
        )
        if release_match:
            release_date = release_match.group(1).replace("_", "-")
        elif official_bundle:
            release_date = "2022-09-28"
        else:
            release_date = None
        return {
            "PDB mmCIF": {
                "release_date": release_date,
                "version": release_date,
                "location_url": (
                    DB_NAME_TO_URL["PDB mmCIF"] if official_bundle else []
                ),
            }
        }

    af3_databases = {
        "ntrna": ("NT-RNA", r"(\d{4}_\d{2}_\d{2})", None, None),
        "rfam": ("Rfam", r"rfam_(\d+_\d+)", None, None),
    }
    if name in af3_databases:
        (
            display_name,
            version_pattern,
            default_version,
            default_release_date,
        ) = af3_databases[name]
        version = default_version
        if version_pattern:
            match = re.search(version_pattern, str(v))
            if match:
                version = match.group(1)
        return {
            display_name: {
                "release_date": default_release_date or get_last_modified_date(v),
                "version": version,
                "location_url": DB_NAME_TO_URL[display_name],
            }
        }

    specific_databases = ["pdb70", "bfd"]
    if name in specific_databases:
        name = name.upper()
        url = DB_NAME_TO_URL[name]
        fn = v + "_hhm.ffindex"
        hash_value = get_hash(fn)
        release_date = get_last_modified_date(fn)
        if release_date == "NA":
            release_date = None
        if hash_value == BFD_HASH_HHM_FFINDEX:
            release_date = "AF2"
        return {name: {"release_date": release_date, "version": hash_value, "location_url": url}}

    other_databases = ["small_bfd", "uniprot", "uniref90", "pdb_seqres"]
    if name in other_databases:
        if name == "small_bfd":
            name = "Reduced BFD"
        elif name == "uniprot":
            name = "UniProt"
        elif name == "uniref90":
            name = "UniRef90"
        elif name == "pdb_seqres":
            name = "PDB seqres"
        url = DB_NAME_TO_URL[name]
        # here we ignore pdb_mmcif assuming it's version is identical to pdb_seqres
        return {name: {"release_date": get_last_modified_date(v),
                       "version": None if name != "PDB seqres" else get_hash(v), "location_url": url}}

    if name in ["uniref30", "mgnify"]:
        if name == "uniref30":
            name = "UniRef30"
        elif name == "mgnify":
            name = "MGnify"
        hash_value = None
        release_date = None
        match = re.search(r"(\d{4}_\d{2})", v)
        if match:
            #release_date = match.group(1)
            url_release_date = match.group(1)
            url = [DB_NAME_TO_URL[name][0].format(release_date=url_release_date)]
            if name == "UniRef30":
                hash_value = get_hash(v + "_hhm.ffindex")
                if not hash_value:
                    hash_value = url_release_date
            if name == "MGnify":
                hash_value = url_release_date
        return {name: {"release_date": release_date, "version": hash_value, "location_url": url}}
    return {}


def get_meta_dict(flag_dict):
    """Save metadata in JSON format."""
    metadata = {
        "databases": {},
        "software": {"AlphaPulldown": {"version": AP_VERSION},
                     "AlphaFold": {"version": AF_VERSION}},
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "other": {},
    }

    for k, v in flag_dict.items():
        if v is None:
            continue
        if k == "use_cprofile_for_profiling" or k.startswith("test_") or k.startswith("help"):
            continue
        metadata["other"][k] = str(v)
        if "_binary_path" in k:
            metadata["software"].update(get_metadata_for_binary(k, v))
        elif "_database_path" in k or "template_mmcif_dir" in k:
            metadata["databases"].update(get_metadata_for_database(k, v))
        elif k == "use_mmseqs2" and v:
            url = DB_NAME_TO_URL["ColabFold"]
            metadata["databases"].update({"ColabFold":
                                              {"version": datetime.datetime.now().strftime('%Y-%m-%d'),
                                               "release_date": None,
                                               "location_url": url}
                                          })

    return metadata


def get_last_modified_date(path):
    """
    Get the last modified date of a file or the most recently modified file in a directory.
    """
    try:
        if not os.path.exists(path):
            logging.warning(f"Path does not exist: {path}")
            return None

        if os.path.isfile(path):
            return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Getting last modified date for {path}")
        most_recent_timestamp = max(
            (
                os.path.getmtime(entry)
                for entry in glob.glob(os.path.join(path, "*"))
                if os.path.isfile(entry)
            ),
            default=0.0,
        )

        return datetime.datetime.fromtimestamp(most_recent_timestamp).strftime(
            '%Y-%m-%d %H:%M:%S') if most_recent_timestamp else None

    except Exception as e:
        logging.warning(f"Error processing {path}: {e}")
        return None


def parse_version(output):
    """Parse version information from a given output string."""
    for pattern in COMMON_PATTERNS:
        match = re.search(pattern, output)
        if match:
            return match.group(1)

    match = re.search(r"Kalign\s+version\s+(\d+\.\d+)", output)
    if match:
        return match.group(1)

    return None


def get_hash(filename):
    """Get the md5 hash of a file."""
    md5_hash = hashlib.md5()
    try:
        with open(filename, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
    except OSError as exc:
        logging.warning(f"Cannot hash database file {filename}: {exc}")
        return None
    return md5_hash.hexdigest()
