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
}

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


def get_metadata_for_database(k, v):
    name = k.replace("_database_path", "").replace("_dir", "")

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
        elif k == "use_mmseqs2":
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
        most_recent_timestamp = max((entry.stat().st_mtime for entry in glob.glob(path + '*') if entry.is_file()),
                                    default=0.0)

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
    with open(filename, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
        return (md5_hash.hexdigest())
