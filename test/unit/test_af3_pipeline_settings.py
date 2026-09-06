"""AF3 pipeline settings are resolved explicitly, not by rewriting a global.

The batch scripts used to call `create_arguments()` -- whose job is to mutate the
module-global FLAGS -- and then call two functions that read that mutation back.
The interface between the scripts was the shape of a global, and it was only
correct if the three calls happened in that order.
"""

from __future__ import annotations

import dataclasses

from alphapulldown.af3_pipeline import (
    BINARY_FIELDS,
    DATABASE_FIELDS,
    AF3PipelineSettings,
    build_pipeline,
)


class _Flags:
    """Just enough of a parsed invocation, with nothing resolved."""

    def __init__(self, **overrides):
        self.max_template_date = "2050-01-01"
        for field in BINARY_FIELDS:
            setattr(self, field, f"/usr/bin/{field}")
        for _flag_name, _key in DATABASE_FIELDS.values():
            setattr(self, _flag_name, None)
        for key, value in overrides.items():
            setattr(self, key, value)

    def flag_values_dict(self):
        return {k: v for k, v in vars(self).items()}


def _resolver(key):
    return f"/db/{key}"


def test_settings_resolve_every_database_without_touching_the_flags():
    flags = _Flags()

    settings = AF3PipelineSettings.from_flags(flags, _resolver)

    assert set(settings.databases) == set(DATABASE_FIELDS)
    assert settings.databases["uniref90_database_path"] == "/db/uniref90"
    # The uniprot triple is the one whose three names all differ.
    assert settings.databases["uniprot_cluster_annot_database_path"] == "/db/uniprot"
    # Nothing was written back.
    for flag_name, _key in DATABASE_FIELDS.values():
        assert getattr(flags, flag_name) is None


def test_an_explicit_path_wins_over_the_default():
    flags = _Flags(uniref90_database_path="/mine/uniref90.fa")

    settings = AF3PipelineSettings.from_flags(flags, _resolver)

    assert settings.databases["uniref90_database_path"] == "/mine/uniref90.fa"


def test_metadata_sees_the_paths_the_run_actually_used():
    """Provenance used to come from FLAGS *after* create_arguments() rewrote it."""
    flags = _Flags()

    settings = AF3PipelineSettings.from_flags(flags, _resolver)
    merged = settings.flag_values_with_resolved_paths(flags.flag_values_dict())

    assert merged["uniref90_database_path"] == "/db/uniref90"
    assert merged["template_mmcif_dir"] == "/db/template_mmcif_dir"
    # The unrelated keys survive.
    assert merged["max_template_date"] == "2050-01-01"


def test_settings_are_frozen_so_they_cannot_drift_after_resolution():
    settings = AF3PipelineSettings.from_flags(_Flags(), _resolver)

    try:
        settings.max_template_date = "1999-01-01"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("settings must be frozen")


def test_build_pipeline_passes_binaries_databases_and_date_through():
    captured = {}

    class _Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _Pipeline:
        def __init__(self, config):
            self.config = config

    settings = AF3PipelineSettings.from_flags(_Flags(), _resolver)
    pipeline = build_pipeline(settings, pipeline_cls=_Pipeline, config_cls=_Config)

    assert isinstance(pipeline, _Pipeline)
    assert captured["uniref90_database_path"] == "/db/uniref90"
    assert captured["jackhmmer_binary_path"] == "/usr/bin/jackhmmer_binary_path"
    assert captured["max_template_date"].isoformat() == "2050-01-01"
    assert captured["jackhmmer_n_cpu"] == 8
