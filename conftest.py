import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_itemcollected(item):
    try:
        par = getattr(item.parent, "obj", None)
        node = getattr(item, "obj", None)
        pref = (par.__doc__.strip() if getattr(par, "__doc__", None) else par.__class__.__name__) if par else ""
        suf = (node.__doc__.strip() if getattr(node, "__doc__", None) else node.__name__) if node else ""
        if pref or suf:
            item._nodeid = " ".join(x for x in (pref, suf) if x)
    except Exception:
        pass

@pytest.fixture
def tmp_flags(monkeypatch, tmp_path):
    """
    Provide a parsed absl FLAGS with sane defaults.
    Values are applied via the underlying flag parsers (strings),
    None values are left at their declared defaults.
    """
    import alphapulldown.scripts.create_individual_features as cf
    F = cf.FLAGS

    # for save_meta_data.get_meta_dict
    if not hasattr(F, "flag_values_dict"):
        def _flag_values_dict():
            return {k: getattr(F, k, None) for k in dir(F) if not k.startswith("_")}
        F.flag_values_dict = _flag_values_dict

    defaults = dict(
        data_pipeline="alphafold2",
        fasta_paths=[str(tmp_path / "a.fasta")],
        data_dir="/db",
        output_dir=str(tmp_path / "out"),
        max_template_date="2021-09-30",
        use_mmseqs2=False,
        use_precomputed_msas=False,
        save_msa_files=False,
        skip_existing=False,
        compress_features=False,
        db_preset="full_dbs",
        use_hhsearch=False,
        # db flags we want to stay None (let code fill from data_dir):
        uniref90_database_path=None,
        uniref30_database_path=None,
        mgnify_database_path=None,
        bfd_database_path=None,
        small_bfd_database_path=None,
        pdb70_database_path=None,
        uniprot_database_path=None,
        pdb_seqres_database_path=None,
        template_mmcif_dir=None,
        obsolete_pdbs_path=None,
        # misc
        seq_index=None,
        path_to_mmt=None,
        description_file=None,
        multiple_mmts=False,
        # binaries
        jackhmmer_binary_path="jackhmmer",
        hhblits_binary_path="hhblits",
        hhsearch_binary_path="hhsearch",
        hmmsearch_binary_path="hmmsearch",
        hmmbuild_binary_path="hmmbuild",
        nhmmer_binary_path="nhmmer",
        hmmalign_binary_path="hmmalign",
        kalign_binary_path="kalign",
    )

    # helper: stringify for absl parser
    def _to_arg_str(name, val):
        if val is None:
            return None
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, (int, float, str)):
            return str(val)
        if isinstance(val, (list, tuple)):
            # absl list flags expect comma-separated string
            return ",".join(str(x) for x in val)
        # fallback
        return str(val)

    flags_dict = F._flags()  # name -> Flag
    for name, value in defaults.items():
        fl = flags_dict.get(name)
        if fl is None:
            # not an absl flag on this FLAGS; attach as attribute for completeness
            setattr(F, name, value)
            continue
        arg = _to_arg_str(name, value)
        if arg is None:
            # leave at declared default (often None); don't parse
            continue
        fl.parse(arg)  # sets .value and marks present

    # mark the whole set parsed to allow reads of flags we left at default
    if hasattr(F, "mark_as_parsed"):
        F.mark_as_parsed()
    else:
        F(["pytest"])

    # ensure module under test uses this FLAGS instance
    monkeypatch.setattr(cf, "FLAGS", F)
    return F
