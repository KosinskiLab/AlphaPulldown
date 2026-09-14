"""Exercise template construction and finalization through their caller interfaces."""
from unittest.mock import Mock, patch

import pytest

from alphapulldown.features.af2_feature_finalizer import build_af2_template_stack
from alphapulldown.scripts import create_individual_features as legacy
from alphapulldown.scripts import finalize_batch_features as cli
from alphapulldown.scripts._mmseqs2_cli import (
    require_template_flags,
    required_template_flag_names,
)
from absl.testing import flagsaver
from types import SimpleNamespace


@pytest.fixture
def finalization_flags(tmp_flags, tmp_path):
    with flagsaver.flagsaver():
        tmp_flags.msa_input_dir = str(tmp_path / "msas")
        tmp_flags.template_seqres_database_id = "seqres-v1"
        tmp_flags.template_pdb70_database_id = "pdb70-v1"
        tmp_flags.template_mmcif_database_id = "mmcif-v1"
        tmp_flags.keep_msas = False
        (tmp_path / "a.fasta").write_text(">query\nACDE\n")
        yield tmp_flags


@pytest.mark.parametrize("hhsearch", [False, True])
def test_template_stack_resolves_defaults_without_mutating_flags(finalization_flags, hhsearch):
    finalization_flags.use_hhsearch = hhsearch
    before = finalization_flags.flag_values_dict()
    settings = legacy.af2_template_stack_settings()
    assert settings.searcher_name == ("hhsearch" if hhsearch else "hmmsearch")
    assert settings.pdb70_database_path == "/db/pdb70/pdb70"
    assert settings.pdb_seqres_database_path == "/db/pdb_seqres/pdb_seqres.txt"
    assert settings.template_mmcif_dir == "/db/pdb_mmcif/mmcif_files"
    assert finalization_flags.flag_values_dict() == before


def test_template_stack_honors_overrides_without_a_data_root(finalization_flags):
    finalization_flags.data_dir = None
    finalization_flags.pdb70_database_path = "/custom/pdb70"
    settings = legacy.af2_template_stack_settings()
    assert settings.pdb70_database_path == "/custom/pdb70"
    assert settings.pdb_seqres_database_path is None


@pytest.mark.parametrize("hhsearch", [False, True])
def test_template_factory_builds_the_selected_search_and_featurizer(finalization_flags, hhsearch):
    finalization_flags.use_hhsearch = hhsearch
    settings = legacy.af2_template_stack_settings()
    with patch("alphafold.data.tools.hhsearch.HHSearch") as hh, \
         patch("alphafold.data.tools.hmmsearch.Hmmsearch") as hmm, \
         patch("alphafold.data.templates.HhsearchHitFeaturizer") as hh_features, \
         patch("alphafold.data.templates.HmmsearchHitFeaturizer") as hmm_features:
        searcher, featurizer = build_af2_template_stack(settings)
    selected_search, selected_features = (hh, hh_features) if hhsearch else (hmm, hmm_features)
    assert searcher is selected_search.return_value
    assert featurizer is selected_features.return_value
    if hhsearch:
        hh.assert_called_once_with(binary_path="hhsearch", databases=[settings.pdb70_database_path])
        hmm.assert_not_called()
    else:
        hmm.assert_called_once_with(binary_path="hmmsearch", hmmbuild_binary_path="hmmbuild",
                                    database_path=settings.pdb_seqres_database_path)
        hh.assert_not_called()
    assert selected_features.call_args.kwargs == dict(
        mmcif_dir=settings.template_mmcif_dir, max_template_date=settings.max_template_date,
        max_hits=20, kalign_binary_path="kalign", release_dates_path=None,
        obsolete_pdbs_path=settings.obsolete_pdbs_path,
    )


@pytest.mark.parametrize("hhsearch", [False, True])
def test_metadata_contains_only_the_template_resources_used(finalization_flags, hhsearch):
    finalization_flags.use_hhsearch = hhsearch
    stack = legacy.af2_template_stack_settings()
    with patch.object(cli.save_meta_data, "get_meta_dict", return_value={"sentinel": 1}) as metadata:
        assert cli.af2_template_metadata(stack) == {"sentinel": 1}
    supplied = metadata.call_args.args[0]
    assert supplied["kalign_binary_path"] == "kalign"
    assert supplied["template_mmcif_dir"] == stack.template_mmcif_dir
    if hhsearch:
        assert supplied["pdb70_database_path"] == stack.pdb70_database_path
        assert "pdb_seqres_database_path" not in supplied
        assert "hmmsearch_binary_path" not in supplied
    else:
        assert supplied["pdb_seqres_database_path"] == stack.pdb_seqres_database_path
        assert "pdb70_database_path" not in supplied
        assert "hhsearch_binary_path" not in supplied
    assert "jackhmmer_binary_path" not in supplied
    assert "hhblits_binary_path" not in supplied


@pytest.mark.parametrize("backend,hhsearch,required", [
    ("alphafold2", True, "template_pdb70_database_id"),
    ("alphafold2", False, "template_seqres_database_id"),
    ("alphafold3", True, "template_seqres_database_id"),
])
def test_template_identity_validation_selects_the_database(finalization_flags, backend, hhsearch, required):
    finalization_flags.data_pipeline = backend
    finalization_flags.use_hhsearch = hhsearch
    assert required_template_flag_names(data_pipeline=backend, use_hhsearch=hhsearch) == (
        required, "template_mmcif_database_id"
    )
    require_template_flags(finalization_flags)
    setattr(finalization_flags, required, "  ")
    with pytest.raises(ValueError, match=required):
        require_template_flags(finalization_flags)


@pytest.mark.parametrize("hhsearch", [False, True])
def test_af2_finalization_cli_passes_selected_identity_and_requests(finalization_flags, hhsearch):
    finalization_flags.use_hhsearch = hhsearch
    if hhsearch:
        finalization_flags.template_seqres_database_id = None
    result = SimpleNamespace(written=(), reused=(), failures=())
    with patch.object(cli, "build_af2_template_stack", return_value=("searcher", "featurizer")), \
         patch.object(cli, "af2_template_metadata", return_value={"software": {}}), \
         patch.object(cli, "Af2FeatureFinalizer") as factory:
        factory.return_value.generate.return_value = result
        cli.main([])
    settings = factory.call_args.kwargs["settings"]
    assert settings.template_searcher == ("hhsearch" if hhsearch else "hmmsearch")
    assert settings.template_pdb70_database_id == "pdb70-v1"
    assert settings.template_mmcif_database_id == "mmcif-v1"
    assert factory.call_args.kwargs["template_searcher"] == "searcher"
    [request] = factory.return_value.generate.call_args.args[0]
    assert (request.name, request.sequence) == ("query", "ACDE")


def test_af3_finalization_cli_keeps_its_seqres_identity(finalization_flags):
    finalization_flags.data_pipeline = "alphafold3"
    result = SimpleNamespace(written=(), reused=(), failures=())
    with patch.object(legacy, "create_pipeline_af3", return_value="af3-pipeline"), \
         patch.object(legacy, "get_af3_feature_metadata", return_value={}) as metadata, \
         patch.object(cli, "FeatureFinalizer") as factory:
        factory.return_value.generate.return_value = result
        cli.main([])
    settings = factory.call_args.kwargs["settings"]
    assert settings.template_seqres_database_id == "seqres-v1"
    assert settings.template_mmcif_database_id == "mmcif-v1"
    assert factory.call_args.kwargs["af3_pipeline"] == "af3-pipeline"
    assert metadata.call_args.kwargs["skip_msa"] is True
    [request] = factory.return_value.generate.call_args.args[0]
    assert request.sequence == "ACDE"


def test_cli_rejects_missing_pdb70_identity_before_loading_tools(finalization_flags):
    finalization_flags.use_hhsearch = True
    finalization_flags.template_pdb70_database_id = None
    with patch.object(cli, "build_af2_template_stack") as factory:
        with pytest.raises(ValueError, match="template_pdb70_database_id"):
            cli.main([])
    factory.assert_not_called()


def test_cli_reports_per_sequence_failures(finalization_flags):
    result = SimpleNamespace(written=(), reused=(), failures=(SimpleNamespace(name="query", error="damaged bundle"),))
    with patch.object(cli, "_finalize_alphafold2", return_value=result):
        with pytest.raises(RuntimeError, match=r"query \(damaged bundle\)"):
            cli.main([])


@pytest.mark.parametrize("flag", ["skip_msa", "keep_msas", "use_mmseqs2", "path_to_mmt"])
def test_cli_rejects_incompatible_feature_modes(finalization_flags, flag):
    setattr(finalization_flags, flag, "/tmp/mmt" if flag == "path_to_mmt" else True)
    with pytest.raises(ValueError, match="cannot be combined"):
        cli.main([])
