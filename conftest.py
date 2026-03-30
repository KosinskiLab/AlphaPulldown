import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent
TEST_IMPORT_PATHS = (
    REPO_ROOT,
    REPO_ROOT / "ColabFold",
    REPO_ROOT / "alphafold",
    REPO_ROOT / "alphafold3" / "src",
    REPO_ROOT / "AlphaLink2",
)

for import_path in TEST_IMPORT_PATHS:
    import_path_str = str(import_path)
    if import_path.exists() and import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


def _install_jax_tree_stub() -> None:
    try:
        import jax  # noqa: F401
        return
    except Exception:
        for module_name in list(sys.modules):
            if module_name == "jax" or module_name.startswith("jax."):
                sys.modules.pop(module_name, None)

    import tree as dm_tree

    jax_stub = types.ModuleType("jax")
    jax_numpy_stub = types.ModuleType("jax.numpy")
    jax_nn_stub = types.ModuleType("jax.nn")
    jax_lax_stub = types.ModuleType("jax.lax")
    tree_stub = types.ModuleType("jax.tree")
    tree_util_stub = types.ModuleType("jax.tree_util")
    version_stub = types.ModuleType("jax.version")
    version_stub.__version__ = "0.0-test"

    def _tree_map(func, *structures):
        return dm_tree.map_structure(func, *structures)

    def _tree_flatten(structure):
        return dm_tree.flatten(structure), structure

    def _tree_unflatten(treedef, leaves):
        return dm_tree.unflatten_as(treedef, leaves)

    def _tree_leaves(structure):
        return dm_tree.flatten(structure)

    def _register_pytree_node(*args, **kwargs):
        return None

    def _register_pytree_node_class(cls):
        return cls

    tree_stub.map = _tree_map
    tree_stub.flatten = _tree_leaves
    tree_stub.unflatten = _tree_unflatten

    tree_util_stub.tree_map = _tree_map
    tree_util_stub.tree_flatten = _tree_flatten
    tree_util_stub.tree_unflatten = _tree_unflatten
    tree_util_stub.tree_leaves = _tree_leaves
    tree_util_stub.tree_structure = lambda structure: structure
    tree_util_stub.register_pytree_node = _register_pytree_node
    tree_util_stub.register_pytree_node_class = _register_pytree_node_class

    jax_numpy_stub.__dict__.update(np.__dict__)
    jax_nn_stub.softmax = lambda x, axis=-1: np.exp(x - np.max(x, axis=axis, keepdims=True)) / np.sum(
        np.exp(x - np.max(x, axis=axis, keepdims=True)),
        axis=axis,
        keepdims=True,
    )
    jax_lax_stub.stop_gradient = lambda x: x

    jax_stub.tree = tree_stub
    jax_stub.tree_map = _tree_map
    jax_stub.tree_util = tree_util_stub
    jax_stub.numpy = jax_numpy_stub
    jax_stub.nn = jax_nn_stub
    jax_stub.lax = jax_lax_stub
    jax_stub.Array = np.ndarray
    jax_stub.local_devices = lambda: [types.SimpleNamespace(platform="cpu")]
    jax_stub.devices = lambda *_args, **_kwargs: [types.SimpleNamespace(platform="cpu")]
    jax_stub.default_backend = lambda: "cpu"
    jax_stub.version = version_stub

    sys.modules["jax"] = jax_stub
    sys.modules["jax.numpy"] = jax_numpy_stub
    sys.modules["jax.nn"] = jax_nn_stub
    sys.modules["jax.lax"] = jax_lax_stub
    sys.modules["jax.tree"] = tree_stub
    sys.modules["jax.tree_util"] = tree_util_stub
    sys.modules["jax.version"] = version_stub


_install_jax_tree_stub()


def pytest_addoption(parser):
    parser.addoption(
        "--use-temp-dir",
        action="store_true",
        default=False,
        help="Run functional test suites with isolated temporary output directories.",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath))
        parts = set(path.parts)
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        if "functional" in parts:
            item.add_marker(pytest.mark.functional)
        if "cluster" in parts:
            item.add_marker(pytest.mark.cluster)


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
        fl.unparse()
        arg = _to_arg_str(name, value)
        if arg is None:
            # reset to the declared default and leave it unset
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
