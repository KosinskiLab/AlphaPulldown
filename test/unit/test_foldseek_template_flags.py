"""The structural-template flag schema: validation, defaults, and metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from alphapulldown.scripts._foldseek_cli import (
    STRUCTURAL_TEMPLATE_FLAG_NAMES,
    build_foldseek_template_searcher,
    cache_dir,
    foldseek_search_settings,
    structural_template_metadata_flags,
    validate_structural_template_flags,
)
from alphapulldown.structural_templates import FoldseekTemplateSearcher


def _flags(**overrides) -> SimpleNamespace:
    values = dict(
        use_foldseek_templates=True,
        foldseek_binary_path="/usr/local/bin/foldseek",
        foldseek_database_path="/databases/foldseek/pdb",
        foldseek_database_id="pdb-2024-01",
        foldseek_temp_dir=None,
        foldseek_e_value=1e-3,
        foldseek_max_hits=100,
        foldseek_min_alignment_tm_score=0.0,
        foldseek_alignment_type=2,
        foldseek_threads=8,
        structural_template_cache_dir=None,
        esmfold_model_dir="/weights/esmfold_v1",
        esmfold_device="cuda",
        esmfold_chunk_size=None,
        esmfold_max_sequence_length=1500,
        use_mmseqs2=False,
        data_pipeline="alphafold2",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------- validation

def test_a_disabled_feature_needs_no_configuration():
    validate_structural_template_flags(_flags(use_foldseek_templates=False))


def test_enabling_without_a_database_names_every_missing_flag():
    with pytest.raises(ValueError) as error:
        validate_structural_template_flags(
            _flags(foldseek_database_path=None, esmfold_model_dir=None)
        )

    message = str(error.value)
    assert "--foldseek_database_path" in message
    assert "--esmfold_model_dir" in message


def test_the_remote_mmseqs2_path_has_no_template_search_to_replace():
    with pytest.raises(ValueError, match="--use_mmseqs2"):
        validate_structural_template_flags(_flags(use_mmseqs2=True))


def test_alphafold3_searches_its_own_templates():
    with pytest.raises(ValueError, match="AlphaFold 2 data pipeline"):
        validate_structural_template_flags(_flags(data_pipeline="alphafold3"))


# ------------------------------------------------------------------ defaults

def test_the_cache_lives_under_the_output_directory_by_default():
    assert cache_dir(_flags(), output_dir="/out") == Path(
        "/out/structural_templates"
    )


def test_an_explicit_cache_directory_wins():
    assert cache_dir(
        _flags(structural_template_cache_dir="/scratch/templates"), output_dir="/out"
    ) == Path("/scratch/templates")


def test_a_cache_directory_is_required_when_there_is_no_output_directory():
    with pytest.raises(ValueError, match="structural_template_cache_dir"):
        cache_dir(_flags(), output_dir=None)


def test_foldseek_scratch_defaults_beneath_the_cache():
    settings = foldseek_search_settings(_flags(), output_dir="/out")

    assert settings.temp_dir == Path("/out/structural_templates/tmp")
    assert settings.database.identifier == "pdb-2024-01"
    assert settings.database.path == Path("/databases/foldseek/pdb")


def test_the_searcher_is_built_without_touching_foldseek_or_the_weights():
    # Construction must stay cheap: nothing is run and no checkpoint is loaded
    # until a query actually arrives.
    searcher = build_foldseek_template_searcher(_flags(), output_dir="/out")

    assert isinstance(searcher, FoldseekTemplateSearcher)
    assert searcher.input_format == "a3m"


# ------------------------------------------------------------------ metadata

def test_metadata_omits_the_feature_entirely_when_it_is_off():
    flag_dict = {"use_hhsearch": False, **vars(_flags(use_foldseek_templates=False))}

    filtered = structural_template_metadata_flags(flag_dict)

    # Exactly the structural-template flags go, so a default run's metadata is
    # the same dict it was before this feature existed.
    assert set(flag_dict) - set(filtered) == set(STRUCTURAL_TEMPLATE_FLAG_NAMES)
    assert filtered["use_hhsearch"] is False


def test_metadata_records_the_feature_when_it_is_used():
    filtered = structural_template_metadata_flags(vars(_flags()))

    assert filtered["foldseek_database_id"] == "pdb-2024-01"
    assert filtered["use_foldseek_templates"] is True


def test_metadata_filtering_does_not_mutate_the_caller_s_flags():
    flag_dict = vars(_flags(use_foldseek_templates=False))

    structural_template_metadata_flags(flag_dict)

    assert "foldseek_database_id" in flag_dict
