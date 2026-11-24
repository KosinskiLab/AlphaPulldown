"""AF2 provenance for selected resources and reused alignments."""
from pathlib import Path
from alphapulldown.utils import save_meta_data
from alphapulldown.utils.feature_metadata import retain_msa_provenance


MSA_DATABASES = {
    "uniref90_database_path": "UniRef90", "mgnify_database_path": "MGnify",
    "bfd_database_path": "BFD", "uniref30_database_path": "UniRef30",
    "small_bfd_database_path": "Reduced BFD", "uniprot_database_path": "UniProt",
}
CACHE_DATABASES = {
    "uniref90_hits.sto": ("UniRef90",), "mgnify_hits.sto": ("MGnify",),
    "bfd_uniref_hits.a3m": ("BFD", "UniRef30"),
    "small_bfd_hits.sto": ("Reduced BFD",), "uniprot_hits.sto": ("UniProt",),
}


def flag_enabled(value):
    return value is True or str(value).lower() == "true"


def cached_msa_databases(directory):
    """Call after cache validation/unpacking, before AF2 consumes the files."""
    return sorted({name for file, names in CACHE_DATABASES.items()
                   if (Path(directory) / file).is_file() for name in names})


def filter_af2_metadata_flags(flags, *, reuse_features=False,
                              cached_databases=(), direct_template=False):
    """Select the configured resources this AF2 execution can actually use."""
    selected = dict(flags)
    mmseqs = flag_enabled(flags.get("use_mmseqs2"))
    msa = not flag_enabled(flags.get("skip_msa")) and not reuse_features
    databases = set()
    binaries = set()
    if not mmseqs:
        if not direct_template:
            databases.add("template_mmcif_dir")
            binaries.add("kalign_binary_path")
            if flag_enabled(flags.get("use_hhsearch")):
                databases.add("pdb70_database_path")
                binaries.add("hhsearch_binary_path")
            else:
                databases.add("pdb_seqres_database_path")
                binaries.update(("hmmsearch_binary_path", "hmmbuild_binary_path"))
        if msa:
            databases.update(("uniref90_database_path", "mgnify_database_path", "uniprot_database_path"))
            if flags.get("db_preset") == "reduced_dbs":
                databases.add("small_bfd_database_path")
            else:
                databases.update(("bfd_database_path", "uniref30_database_path"))
            databases = {key for key in databases if MSA_DATABASES.get(key) not in cached_databases}
            if databases & {"uniref90_database_path", "mgnify_database_path", "uniprot_database_path", "small_bfd_database_path"}:
                binaries.add("jackhmmer_binary_path")
            if "bfd_database_path" in databases:
                binaries.add("hhblits_binary_path")
    for key in selected:
        if (key.endswith("_database_path") or key == "template_mmcif_dir") and key not in databases:
            selected[key] = None
        elif key.endswith("_binary_path") and key not in binaries:
            selected[key] = None
    selected["use_mmseqs2"] = mmseqs and msa and "ColabFold" not in cached_databases
    return selected


def get_af2_feature_metadata(flags, *, previous=None, reuse_features=False,
                             cached_databases=(), direct_template=None):
    # A cache directory may contain alignments from a different preset. Record
    # only the cache files the selected execution can consume.
    active = filter_af2_metadata_flags(flags, reuse_features=reuse_features,
                                       direct_template=direct_template is not None)
    active_names = {name for key, name in MSA_DATABASES.items() if active.get(key)}
    if active.get("use_mmseqs2"):
        active_names.add("ColabFold")
    cached_databases = sorted(set(cached_databases) & active_names)
    selected = filter_af2_metadata_flags(flags, reuse_features=reuse_features,
                                         cached_databases=cached_databases,
                                         direct_template=direct_template is not None)
    metadata = save_meta_data.get_meta_dict(selected)
    metadata.setdefault("other", {})
    metadata.setdefault("databases", {})
    metadata["other"]["use_mmseqs2"] = str(flags.get("use_mmseqs2", False))
    if direct_template is not None:
        metadata["template_provenance"] = {"mode": "direct_mmcif", "path": str(direct_template)}
    elif flags.get("path_to_mmt"):
        for name in ("PDB seqres", "PDB70", "PDB mmCIF"):
            if name in metadata["databases"]:
                metadata["databases"][name]["location_url"] = []
        metadata["template_provenance"] = {"mode": "custom_templates"}
    if reuse_features:
        metadata = retain_msa_provenance(metadata, previous)
    elif cached_databases:
        metadata["msa_provenance"] = {
            "mode": "cached_alignments", "unknown_source": True,
            "cached_databases": sorted(cached_databases),
        }
    else:
        metadata["msa_provenance"] = {
            "mode": "query_only" if flag_enabled(flags.get("skip_msa")) else "generated",
        }
    return metadata
