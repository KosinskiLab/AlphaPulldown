from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import alphapulldown.objects as objects_mod
from alphapulldown.objects import ChoppedObject, MonomericObject, MultimericObject
from alphapulldown.utils import mmseqs_species_identifiers


def _feature_dict(
    sequence: str = "ABCDEFGHIJ",
    *,
    msa_rows: int = 2,
    all_seq_rows: int = 3,
    template_count: int = 1,
):
    length = len(sequence)
    feature_dict = {
        "aatype": np.arange(length * 22, dtype=np.float32).reshape(length, 22),
        "between_segment_residues": np.zeros(length, dtype=np.int32),
        "residue_index": np.arange(length, dtype=np.int32),
        "seq_length": np.full(length, length, dtype=np.int32),
        "sequence": np.array([sequence.encode()], dtype=object),
        "deletion_matrix_int": np.arange(msa_rows * length, dtype=np.int32).reshape(
            msa_rows, length
        ),
        "msa": np.arange(msa_rows * length, dtype=np.int32).reshape(msa_rows, length),
        "num_alignments": np.full(length, msa_rows, dtype=np.int32),
        "msa_species_identifiers": np.array([b"9606"] * msa_rows, dtype=object),
        "deletion_matrix_int_all_seq": np.arange(
            all_seq_rows * length, dtype=np.int32
        ).reshape(all_seq_rows, length),
        "msa_all_seq": np.arange(all_seq_rows * length, dtype=np.int32).reshape(
            all_seq_rows, length
        )
        + 100,
        "msa_species_identifiers_all_seq": np.array(
            [b"9606"] * all_seq_rows, dtype=object
        ),
        "domain_name": np.array([b"domain"], dtype=object),
    }
    if template_count:
        feature_dict.update(
            {
                "template_aatype": np.arange(
                    template_count * length * 22, dtype=np.float32
                ).reshape(template_count, length, 22),
                "template_all_atom_masks": np.ones(
                    (template_count, length, 37), dtype=np.float32
                ),
                "template_all_atom_positions": np.ones(
                    (template_count, length, 37, 3), dtype=np.float32
                ),
                "template_domain_names": np.array(
                    [b"1abc_A"] * template_count, dtype=object
                ),
                "template_sequence": np.array(
                    [sequence.encode()] * template_count, dtype=object
                ),
                "template_sum_probs": np.full(template_count, 0.5, dtype=np.float32),
                "template_confidence_scores": np.full(
                    (template_count, length), 0.75, dtype=np.float32
                ),
                "template_release_date": np.array(
                    ["2024-01-01"] * template_count, dtype=object
                ),
            }
        )
    return feature_dict


def test_make_features_populates_defaults_and_removes_msa_when_not_saved(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    monomer.uniprot_runner = "runner"

    class FakePipeline:
        def process(self, fasta_file, msa_output_dir):
            text = Path(fasta_file).read_text(encoding="utf-8")
            assert text == ">proteinA\nACDE"
            assert msa_output_dir.endswith("proteinA")
            return {"existing": "value"}

    remove_calls = []
    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        monomer,
        "all_seq_msa_features",
        lambda *_args, **_kwargs: {"msa_all_seq": np.asarray([[1, 2]], dtype=np.int32)},
    )
    monkeypatch.setattr(
        MonomericObject,
        "remove_msa_files",
        staticmethod(lambda msa_output_path: remove_calls.append(msa_output_path)),
    )
    monkeypatch.setattr(
        MonomericObject, "zip_msa_files", staticmethod(lambda _path: None)
    )

    monomer.make_features(
        pipeline=FakePipeline(),
        output_dir=str(tmp_path),
        use_precomputed_msa=False,
        save_msa=False,
        compress_msa_files=False,
    )

    assert monomer.feature_dict["existing"] == "value"
    assert np.array_equal(
        monomer.feature_dict["template_confidence_scores"], np.array([[1, 1, 1, 1]])
    )
    assert np.array_equal(
        monomer.feature_dict["template_release_date"], np.array(["none"])
    )
    assert remove_calls == [str(tmp_path / "proteinA")]


def test_make_features_rezips_when_inputs_were_zipped_and_compression_is_enabled(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    zip_calls = []

    class FakePipeline:
        def process(self, *_args, **_kwargs):
            return {
                "template_confidence_scores": np.array([[0.1, 0.2, 0.3, 0.4]]),
                "template_release_date": np.array(["2024-01-01"]),
            }

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: True)
    )
    monkeypatch.setattr(
        monomer, "all_seq_msa_features", lambda *_args, **_kwargs: {"extra": 1}
    )
    monkeypatch.setattr(
        MonomericObject,
        "zip_msa_files",
        staticmethod(lambda path: zip_calls.append(path)),
    )
    monkeypatch.setattr(
        MonomericObject,
        "remove_msa_files",
        staticmethod(lambda msa_output_path=None, **_kwargs: None),
    )

    monomer.make_features(
        pipeline=FakePipeline(),
        output_dir=str(tmp_path),
        save_msa=True,
        compress_msa_files=True,
    )

    assert zip_calls == [str(tmp_path / "proteinA"), str(tmp_path / "proteinA")]
    assert np.array_equal(
        monomer.feature_dict["template_release_date"], np.array(["2024-01-01"])
    )


def test_make_features_removes_msa_when_precomputed_inputs_are_not_saved(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    remove_calls = []

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        monomer, "all_seq_msa_features", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        MonomericObject,
        "remove_msa_files",
        staticmethod(lambda msa_output_path: remove_calls.append(msa_output_path)),
    )
    monkeypatch.setattr(
        MonomericObject, "zip_msa_files", staticmethod(lambda _path: None)
    )

    class FakePipeline:
        def process(self, *_args, **_kwargs):
            return {}

    monomer.make_features(
        pipeline=FakePipeline(),
        output_dir=str(tmp_path),
        use_precomputed_msa=True,
        save_msa=False,
    )

    assert remove_calls == [str(tmp_path / "proteinA")]


def test_make_features_skip_msa_builds_query_only_features_and_templates(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    calls = {}

    class FakeTemplateSearcher:
        input_format = "a3m"
        output_format = "hhr"

        def query(self, alignment):
            calls["template_query"] = alignment
            return "template_hits"

        def get_template_hits(self, output_string, input_sequence):
            calls["template_hits"] = (output_string, input_sequence)
            return ["hitA"]

    class FakeTemplateFeaturizer:
        def get_templates(self, query_sequence, hits):
            calls["template_features"] = (query_sequence, hits)
            return SimpleNamespace(
                features={
                    "template_aatype": np.ones((1, 4, 22), dtype=np.float32),
                    "template_all_atom_masks": np.ones((1, 4, 37), dtype=np.float32),
                    "template_all_atom_positions": np.ones(
                        (1, 4, 37, 3), dtype=np.float32
                    ),
                    "template_domain_names": np.asarray([b"1abc_A"], dtype=object),
                    "template_sequence": np.asarray([b"ACDE"], dtype=object),
                    "template_sum_probs": np.asarray([0.5], dtype=np.float32),
                }
            )

    class FakePipeline:
        template_searcher = FakeTemplateSearcher()
        template_featurizer = FakeTemplateFeaturizer()

        def process(self, *_args, **_kwargs):
            raise AssertionError("skip_msa should bypass pipeline.process")

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        monomer,
        "all_seq_msa_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("skip_msa should bypass all_seq_msa_features")
        ),
    )
    monkeypatch.setattr(
        MonomericObject,
        "remove_msa_files",
        staticmethod(lambda msa_output_path=None, **_kwargs: None),
    )
    monkeypatch.setattr(
        MonomericObject, "zip_msa_files", staticmethod(lambda _path: None)
    )

    monomer.make_features(
        pipeline=FakePipeline(),
        output_dir=str(tmp_path),
        save_msa=False,
        skip_msa=True,
    )

    assert calls["template_query"] == ">query\nACDE\n"
    assert calls["template_hits"] == ("template_hits", "ACDE")
    assert calls["template_features"] == ("ACDE", ["hitA"])
    assert monomer.skip_msa is True
    assert monomer.feature_dict["msa"].shape == (1, 4)
    assert monomer.feature_dict["msa_all_seq"].shape == (1, 4)
    assert np.array_equal(
        monomer.feature_dict["num_alignments"], np.asarray([1, 1, 1, 1], dtype=np.int32)
    )
    assert monomer.feature_dict["msa_species_identifiers_all_seq"].tolist() == [b""]
    assert monomer.feature_dict["template_domain_names"].tolist() == [b"1abc_A"]


def test_make_mmseq_features_builds_all_seq_features_and_writes_a3m(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    calls = {}

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    def fake_get_msa_and_templates(**kwargs):
        calls["get_msa_and_templates"] = kwargs
        return (["UNPAIRED"], ["PAIRED"], ["UNIQUE"], ["CARD"], ["TEMPLATE"])

    monkeypatch.setattr(objects_mod, "get_msa_and_templates", fake_get_msa_and_templates)
    monkeypatch.setattr(
        objects_mod,
        "msa_to_str",
        lambda *args: ">101\nACDE\n>hit\nAC-E\n",
    )

    def fake_build(sequence, msa, template_features):
        calls["build_monomer_feature"] = (sequence, msa, template_features)
        return {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.asarray([[0, 0, 1, 0]], dtype=np.int32),
            "template_confidence_scores": None,
            "template_release_date": None,
        }

    monkeypatch.setattr(objects_mod, "build_monomer_feature", fake_build)

    def fake_enrich(feature_dict, a3m, **kwargs):
        calls["enrich"] = {"a3m": a3m, "kwargs": kwargs}
        feature_dict["msa_species_identifiers"] = np.asarray([b"", b"562"], dtype=object)
        feature_dict["msa_uniprot_accession_identifiers"] = np.asarray(
            [b"", b"A0A123"], dtype=object
        )

    monkeypatch.setattr(
        objects_mod,
        "enrich_mmseq_feature_dict_with_identifiers",
        fake_enrich,
    )
    monkeypatch.setattr(
        MonomericObject, "zip_msa_files", staticmethod(lambda _path: None)
    )

    monomer.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        compress_msa_files=False,
        use_precomputed_msa=False,
        use_templates=False,
    )

    assert calls["build_monomer_feature"] == ("ACDE", "UNPAIRED", "TEMPLATE")
    assert calls["enrich"] == {
        "a3m": ">101\nACDE\n>hit\nAC-E",
        "kwargs": {"cache_path": str(tmp_path / "proteinA.mmseq_ids.json")},
    }
    assert (tmp_path / "proteinA.a3m").read_text(encoding="utf-8").startswith(">101")
    assert "msa_all_seq" in monomer.feature_dict
    assert "deletion_matrix_int_all_seq" in monomer.feature_dict
    assert "msa_species_identifiers_all_seq" in monomer.feature_dict
    assert "msa_uniprot_accession_identifiers_all_seq" in monomer.feature_dict
    assert isinstance(monomer.feature_dict["template_confidence_scores"], np.ndarray)
    assert monomer.feature_dict["template_release_date"] == ["none"]


def test_make_mmseq_features_skip_msa_uses_single_sequence_mode(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    calls = {}

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )

    def fake_get_msa_and_templates(**kwargs):
        calls["get_msa_and_templates"] = kwargs
        return (["UNPAIRED"], [""], ["UNIQUE"], ["CARD"], ["TEMPLATE"])

    monkeypatch.setattr(objects_mod, "get_msa_and_templates", fake_get_msa_and_templates)
    monkeypatch.setattr(
        objects_mod,
        "build_monomer_feature",
        lambda sequence, msa, template_features: {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
            "template_confidence_scores": None,
            "template_release_date": None,
        },
    )

    def fake_enrich(feature_dict, a3m, **kwargs):
        calls["enrich"] = {"a3m": a3m, "kwargs": kwargs}
        feature_dict["msa_species_identifiers"] = np.asarray([b""], dtype=object)
        feature_dict["msa_uniprot_accession_identifiers"] = np.asarray(
            [b""], dtype=object
        )

    monkeypatch.setattr(
        objects_mod,
        "enrich_mmseq_feature_dict_with_identifiers",
        fake_enrich,
    )
    monkeypatch.setattr(
        MonomericObject, "zip_msa_files", staticmethod(lambda _path: None)
    )

    monomer.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        use_templates=True,
        skip_msa=True,
    )

    assert calls["get_msa_and_templates"]["msa_mode"] == "single_sequence"
    assert calls["get_msa_and_templates"]["pair_mode"] == "none"
    assert calls["get_msa_and_templates"]["a3m_lines"] == [">101\nACDE"]
    assert calls["get_msa_and_templates"]["use_templates"] is True
    assert calls["enrich"]["a3m"] == ">101\nACDE"
    assert monomer.skip_msa is True
    assert monomer.feature_dict["msa"].shape == (1, 4)
    assert monomer.feature_dict["msa_all_seq"].shape == (1, 4)
    assert monomer.feature_dict["msa_uniprot_accession_identifiers_all_seq"].tolist() == [
        b""
    ]


def test_make_mmseq_features_compresses_fresh_mmseq_result_dir(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    zip_calls = []

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        objects_mod,
        "get_msa_and_templates",
        lambda **_kwargs: (["UNPAIRED"], ["PAIRED"], ["UNIQUE"], ["CARD"], ["TEMPLATE"]),
    )
    monkeypatch.setattr(objects_mod, "msa_to_str", lambda *args: ">101\nACDE\n")
    monkeypatch.setattr(
        objects_mod,
        "build_monomer_feature",
        lambda *_args, **_kwargs: {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
            "template_confidence_scores": None,
            "template_release_date": None,
        },
    )
    monkeypatch.setattr(
        objects_mod,
        "enrich_mmseq_feature_dict_with_identifiers",
        lambda feature_dict, *_args, **_kwargs: feature_dict.update(
            {
                "msa_species_identifiers": np.asarray([b""], dtype=object),
                "msa_uniprot_accession_identifiers": np.asarray([b""], dtype=object),
            }
        ),
    )
    monkeypatch.setattr(
        MonomericObject,
        "zip_msa_files",
        staticmethod(lambda path: zip_calls.append(path)),
    )

    monomer.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        compress_msa_files=True,
        use_precomputed_msa=False,
    )

    assert zip_calls == [str(tmp_path / "proteinA")]


def test_make_mmseq_features_uses_precomputed_a3m_without_template_research(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    calls = {"get_msa_and_templates": 0}
    precomputed_dir = tmp_path / "proteinA"
    precomputed_dir.mkdir()
    (precomputed_dir / ".result.zip").write_text("zip", encoding="utf-8")
    (tmp_path / "proteinA.a3m").write_text(">101\nACDE\n", encoding="utf-8")

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        objects_mod,
        "unserialize_msa",
        lambda a3m_lines, sequence: (
            ["PRECOMP_MSA"],
            ["PRECOMP_PAIRED"],
            ["UNIQUE"],
            ["CARD"],
            ["PRECOMP_TEMPLATE"],
        ),
    )

    def fake_get_msa_and_templates(**_kwargs):
        calls["get_msa_and_templates"] += 1
        return None

    monkeypatch.setattr(objects_mod, "get_msa_and_templates", fake_get_msa_and_templates)
    monkeypatch.setattr(
        objects_mod,
        "build_monomer_feature",
        lambda *_args, **_kwargs: {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
            "template_confidence_scores": np.asarray([[0.9, 0.9, 0.9, 0.9]]),
            "template_release_date": ["2024-01-01"],
        },
    )
    monkeypatch.setattr(
        objects_mod,
        "enrich_mmseq_feature_dict_with_identifiers",
        lambda feature_dict, *_args, **_kwargs: feature_dict.update(
            {
                "msa_species_identifiers": np.asarray([b"562"], dtype=object),
                "msa_uniprot_accession_identifiers": np.asarray([b"A0A123"], dtype=object),
            }
        ),
    )

    monomer.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        use_precomputed_msa=True,
        use_templates=False,
    )

    assert calls["get_msa_and_templates"] == 0
    assert np.array_equal(
        monomer.feature_dict["template_confidence_scores"],
        np.asarray([[0.9, 0.9, 0.9, 0.9]]),
    )
    assert monomer.feature_dict["template_release_date"] == ["2024-01-01"]


def test_make_mmseq_features_researches_templates_without_rerunning_msa(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    calls = {}
    (tmp_path / "proteinA.a3m").write_text(">101\nACDE\n", encoding="utf-8")

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        objects_mod,
        "unserialize_msa",
        lambda a3m_lines, sequence: (
            ["PRECOMP_MSA"],
            ["PRECOMP_PAIRED"],
            ["UNIQUE"],
            ["CARD"],
            ["PRECOMP_TEMPLATE"],
        ),
    )

    def fake_get_msa_and_templates(**kwargs):
        calls["get_msa_and_templates"] = kwargs
        return (
            ["IGNORED_UNPAIRED"],
            ["IGNORED_PAIRED"],
            ["IGNORED_UNIQUE"],
            ["IGNORED_CARD"],
            ["RESEARCHED_TEMPLATE"],
        )

    monkeypatch.setattr(objects_mod, "get_msa_and_templates", fake_get_msa_and_templates)

    def fake_build(sequence, msa, template_features):
        calls["build_monomer_feature"] = (sequence, msa, template_features)
        return {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
            "template_confidence_scores": None,
            "template_release_date": None,
        }

    monkeypatch.setattr(objects_mod, "build_monomer_feature", fake_build)
    monkeypatch.setattr(
        objects_mod,
        "enrich_mmseq_feature_dict_with_identifiers",
        lambda feature_dict, *_args, **_kwargs: feature_dict.update(
            {
                "msa_species_identifiers": np.asarray([b"562"], dtype=object),
                "msa_uniprot_accession_identifiers": np.asarray([b"A0A123"], dtype=object),
            }
        ),
    )

    monomer.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        use_precomputed_msa=True,
        use_templates=True,
    )

    assert calls["get_msa_and_templates"]["msa_mode"] == "single_sequence"
    assert calls["get_msa_and_templates"]["a3m_lines"] is False
    assert calls["get_msa_and_templates"]["use_templates"] is True
    assert calls["build_monomer_feature"] == (
        "ACDE",
        "PRECOMP_MSA",
        "RESEARCHED_TEMPLATE",
    )


def test_make_mmseq_features_uses_custom_template_path_for_precomputed_msa(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    calls = {}
    (tmp_path / "proteinA.a3m").write_text(">101\nACDE\n", encoding="utf-8")
    custom_template_path = str(tmp_path / "custom_templates")

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        objects_mod,
        "unserialize_msa",
        lambda a3m_lines, sequence: (
            ["PRECOMP_MSA"],
            ["PRECOMP_PAIRED"],
            ["UNIQUE"],
            ["CARD"],
            ["PRECOMP_TEMPLATE"],
        ),
    )

    def fake_get_msa_and_templates(**kwargs):
        calls["get_msa_and_templates"] = kwargs
        return (
            ["IGNORED_UNPAIRED"],
            ["IGNORED_PAIRED"],
            ["IGNORED_UNIQUE"],
            ["IGNORED_CARD"],
            ["CUSTOM_TEMPLATE"],
        )

    monkeypatch.setattr(objects_mod, "get_msa_and_templates", fake_get_msa_and_templates)
    monkeypatch.setattr(
        objects_mod,
        "build_monomer_feature",
        lambda *_args, **_kwargs: {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
            "template_confidence_scores": None,
            "template_release_date": None,
        },
    )
    monkeypatch.setattr(
        objects_mod,
        "enrich_mmseq_feature_dict_with_identifiers",
        lambda feature_dict, *_args, **_kwargs: feature_dict.update(
            {
                "msa_species_identifiers": np.asarray([b"562"], dtype=object),
                "msa_uniprot_accession_identifiers": np.asarray([b"A0A123"], dtype=object),
            }
        ),
    )

    monomer.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        use_precomputed_msa=True,
        use_templates=False,
        custom_template_path=custom_template_path,
    )

    assert calls["get_msa_and_templates"]["use_templates"] is True
    assert calls["get_msa_and_templates"]["custom_template_path"] == custom_template_path


def test_make_mmseq_features_reuses_identifier_sidecar_on_precomputed_run(
    monkeypatch, tmp_path
):
    a3m_text = "\n".join(
        [
            "# mmseqs header",
            ">101",
            "ACDE",
            ">UniRef100_A0A636IKY3\t136\t0.883",
            "ACDF",
            "",
        ]
    )
    feature_rows = {
        "msa": np.asarray([[1, 2, 3, 4], [1, 2, 3, 5]], dtype=np.int32),
        "deletion_matrix_int": np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
        "template_confidence_scores": None,
        "template_release_date": None,
    }

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: False)
    )
    monkeypatch.setattr(
        MonomericObject, "zip_msa_files", staticmethod(lambda _path: None)
    )
    monkeypatch.setattr(
        objects_mod,
        "get_msa_and_templates",
        lambda **_kwargs: (
            ["UNPAIRED"],
            ["PAIRED"],
            ["UNIQUE"],
            ["CARD"],
            ["TEMPLATE"],
        ),
    )
    monkeypatch.setattr(objects_mod, "msa_to_str", lambda *args: a3m_text)
    monkeypatch.setattr(
        objects_mod,
        "build_monomer_feature",
        lambda *_args, **_kwargs: dict(feature_rows),
    )
    monkeypatch.setattr(
        objects_mod,
        "unserialize_msa",
        lambda a3m_lines, sequence: (
            ["PRECOMP_MSA"],
            ["PRECOMP_PAIRED"],
            ["UNIQUE"],
            ["CARD"],
            ["PRECOMP_TEMPLATE"],
        ),
    )

    first_calls = []

    def fake_uniprot_batch(accessions, *, urlopen):
        first_calls.append(tuple(accessions))
        return {
            "results": [
                {
                    "primaryAccession": "A0A636IKY3",
                    "organism": {"taxonId": 562},
                }
            ]
        }

    monkeypatch.setattr(
        mmseqs_species_identifiers,
        "_query_uniprot_batch",
        fake_uniprot_batch,
    )
    monkeypatch.setattr(
        mmseqs_species_identifiers,
        "_query_uniparc_batch",
        lambda accessions, *, urlopen: {"results": []},
    )

    first = MonomericObject("proteinA", "ACDE")
    first.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        use_precomputed_msa=False,
    )

    assert first_calls == [("A0A636IKY3",)]
    assert (tmp_path / "proteinA.mmseq_ids.json").exists()
    assert first.feature_dict["msa_species_identifiers"].tolist() == [b"", b"562"]

    mmseqs_species_identifiers._SPECIES_ID_CACHE.clear()
    second_calls = []

    def fail_uniprot_batch(accessions, *, urlopen):
        second_calls.append(tuple(accessions))
        raise AssertionError("expected sidecar cache to skip UniProt lookups")

    monkeypatch.setattr(
        mmseqs_species_identifiers,
        "_query_uniprot_batch",
        fail_uniprot_batch,
    )

    second = MonomericObject("proteinA", "ACDE")
    second.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
        use_precomputed_msa=True,
    )

    assert second_calls == []
    assert second.feature_dict["msa_species_identifiers"].tolist() == [b"", b"562"]
    assert second.feature_dict["msa_uniprot_accession_identifiers"].tolist() == [
        b"",
        b"A0A636IKY3",
    ]
    assert "msa_all_seq" in second.feature_dict
    assert "msa_species_identifiers_all_seq" in second.feature_dict
    assert "msa_uniprot_accession_identifiers_all_seq" in second.feature_dict


def test_make_mmseq_features_rezips_output_dir_when_original_msas_were_zipped(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("proteinA", "ACDE")
    zip_calls = []

    monkeypatch.setattr(
        MonomericObject, "unzip_msa_files", staticmethod(lambda _path: True)
    )
    monkeypatch.setattr(
        objects_mod,
        "get_msa_and_templates",
        lambda **_kwargs: (["UNPAIRED"], ["PAIRED"], ["UNIQUE"], ["CARD"], ["TEMPLATE"]),
    )
    monkeypatch.setattr(objects_mod, "msa_to_str", lambda *args: ">101\nACDE\n")
    monkeypatch.setattr(
        objects_mod,
        "build_monomer_feature",
        lambda *_args, **_kwargs: {
            "msa": np.asarray([[1, 2, 3, 4]], dtype=np.int32),
            "deletion_matrix_int": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
            "template_confidence_scores": None,
            "template_release_date": None,
        },
    )
    monkeypatch.setattr(
        objects_mod,
        "enrich_mmseq_feature_dict_with_identifiers",
        lambda feature_dict, *_args, **_kwargs: feature_dict.update(
            {
                "msa_species_identifiers": np.asarray([b""], dtype=object),
                "msa_uniprot_accession_identifiers": np.asarray([b""], dtype=object),
            }
        ),
    )
    monkeypatch.setattr(
        MonomericObject,
        "zip_msa_files",
        staticmethod(lambda path: zip_calls.append(path)),
    )

    monomer.make_mmseq_features(
        DEFAULT_API_SERVER="https://fake.server",
        output_dir=str(tmp_path),
    )

    assert zip_calls == [str(tmp_path)]


def test_zip_msa_files_is_noop_when_no_supported_files_exist(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        "alphapulldown.objects.subprocess.run",
        lambda cmd, **kwargs: commands.append((cmd, kwargs)),
    )

    MonomericObject.zip_msa_files(str(tmp_path))

    assert commands == []


def test_remove_msa_files_returns_when_path_is_not_a_directory(tmp_path):
    missing = tmp_path / "missing"

    MonomericObject.remove_msa_files(str(missing))

    assert not missing.exists()


def test_chopped_object_initialization_preserves_source_inputs_and_updates_description():
    feature_dict = _feature_dict()

    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", feature_dict, [(2, 4), (7, 8)])

    assert chopped.description == "proteinA_2-4_7-8"
    assert chopped.source_sequence == "ABCDEFGHIJ"
    assert chopped.source_feature_dict is feature_dict
    assert chopped.regions == [(2, 4), (7, 8)]


def test_prepare_new_msa_feature_slices_sequence_and_alignment_matrices():
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", _feature_dict(), [(2, 4)])

    sliced, sequence = chopped.prepare_new_msa_feature(chopped.feature_dict, 2, 4)

    assert sequence == "BCD"
    assert sliced["aatype"].shape == (3, 22)
    assert sliced["deletion_matrix_int"].shape == (2, 3)
    assert sliced["deletion_matrix_int_all_seq"].shape == (3, 3)
    assert sliced["msa"].shape == (2, 3)
    assert sliced["msa_all_seq"].shape == (3, 3)
    assert np.array_equal(sliced["sequence"], np.array([b"BCD"]))
    assert np.array_equal(sliced["seq_length"], np.array([3, 3, 3]))
    assert np.array_equal(sliced["num_alignments"], np.array([2, 2, 2]))


def test_prepare_new_template_feature_returns_empty_arrays_when_templates_missing():
    feature_dict = _feature_dict()
    for key in list(feature_dict):
        if key.startswith("template_"):
            feature_dict.pop(key)
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", feature_dict, [(1, 3)])

    sliced = chopped.prepare_new_template_feature(chopped.feature_dict, 1, 3)

    assert sliced["template_aatype"].shape == (0, 3, 22)
    assert sliced["template_all_atom_masks"].shape == (0, 3, 37)
    assert sliced["template_all_atom_positions"].shape == (0, 3, 37, 3)
    assert sliced["template_confidence_scores"].shape == (0, 3)
    assert sliced["template_release_date"].size == 0


def test_prepare_new_template_feature_slices_sequences_and_defaults_confidence():
    feature_dict = _feature_dict(sequence="ABCDEFGHIJ")
    feature_dict.pop("template_confidence_scores")
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", feature_dict, [(3, 6)])

    sliced = chopped.prepare_new_template_feature(chopped.feature_dict, 3, 6)

    assert sliced["template_aatype"].shape == (1, 4, 22)
    assert np.array_equal(sliced["template_sequence"], np.array([b"CDEF"], dtype=object))
    assert np.array_equal(sliced["template_confidence_scores"], np.ones((1, 4)))


def test_prepare_individual_sliced_feature_dict_combines_msa_and_template_features():
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", _feature_dict(), [(1, 2)])

    sliced = chopped.prepare_individual_sliced_feature_dict(chopped.feature_dict, 1, 2)

    assert chopped.new_sequence == "AB"
    assert "template_aatype" in sliced
    assert "msa_all_seq" in sliced


def test_concatenate_sliced_feature_dict_joins_regions_on_expected_axes():
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", _feature_dict(), [(1, 2), (5, 6)])
    slice_a = chopped.prepare_individual_sliced_feature_dict(chopped.feature_dict, 1, 2)
    slice_b = chopped.prepare_individual_sliced_feature_dict(chopped.feature_dict, 5, 6)

    merged = chopped.concatenate_sliced_feature_dict([slice_a, slice_b])

    assert merged["sequence"][0] == b"ABEF"
    assert merged["aatype"].shape == (4, 22)
    assert merged["msa"].shape == (2, 4)
    assert merged["msa_all_seq"].shape == (3, 4)
    assert merged["template_aatype"].shape == (1, 4, 22)
    assert np.array_equal(merged["seq_length"], np.array([4, 4, 4, 4]))
    assert np.array_equal(merged["num_alignments"], np.array([2, 2, 2, 2]))


def test_concatenate_sliced_feature_dict_handles_missing_template_confidence_scores():
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", _feature_dict(), [(1, 2), (3, 4)])
    slice_a = chopped.prepare_individual_sliced_feature_dict(chopped.feature_dict, 1, 2)
    slice_b = chopped.prepare_individual_sliced_feature_dict(chopped.feature_dict, 3, 4)
    slice_a.pop("template_confidence_scores")
    slice_b.pop("template_confidence_scores")

    merged = chopped.concatenate_sliced_feature_dict([slice_a, slice_b])

    assert "template_confidence_scores" not in merged
    assert merged["template_aatype"].shape == (1, 4, 22)


def test_prepare_final_sliced_feature_dict_updates_sequence_for_single_region():
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", _feature_dict(), [(4, 7)])

    chopped.prepare_final_sliced_feature_dict()

    assert chopped.sequence == "DEFG"
    assert chopped.feature_dict["sequence"][0] == b"DEFG"
    assert np.array_equal(chopped.feature_dict["domain_name"], np.array([b"domain"]))
    assert chopped.new_feature_dict == {}


def test_split_into_individual_region_objects_returns_prepared_region_objects():
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", _feature_dict(), [(1, 2), (7, 8)])

    split = chopped.split_into_individual_region_objects()

    assert [obj.description for obj in split] == ["proteinA_1-2", "proteinA_7-8"]
    assert [obj.sequence for obj in split] == ["AB", "GH"]
    assert [obj.regions for obj in split] == [[(1, 2)], [(7, 8)]]


def test_split_into_individual_region_objects_returns_self_for_single_region():
    chopped = ChoppedObject("proteinA", "ABCDEFGHIJ", _feature_dict(), [(1, 2)])

    assert chopped.split_into_individual_region_objects() == [chopped]


def test_multimeric_object_init_calls_template_setup_and_feature_creation(monkeypatch):
    calls = []
    interactor = SimpleNamespace(description="proteinA", sequence="ACDE", feature_dict={})

    def fake_prepare(path, template_dir):
        calls.append(("prepare_meta", path, template_dir))
        return {"proteinA": [("file.cif", "A")]}

    monkeypatch.setattr(objects_mod, "prepare_multimeric_template_meta_info", fake_prepare)
    monkeypatch.setattr(
        MultimericObject,
        "build_description_monomer_mapping",
        lambda self: setattr(self, "monomers_mapping", {"proteinA": interactor}),
    )
    monkeypatch.setattr(
        MultimericObject,
        "create_multimeric_template_features",
        lambda self: calls.append("templates"),
    )
    monkeypatch.setattr(
        MultimericObject,
        "create_all_chain_features",
        lambda self: calls.append("features"),
    )

    multimer = MultimericObject(
        [interactor],
        pair_msa=False,
        multimeric_template=True,
        multimeric_template_meta_data="meta.csv",
        multimeric_template_dir="/tmp/templates",
    )

    assert multimer.description == "proteinA"
    assert multimer.pair_msa is False
    assert multimer.multimeric_template is True
    assert multimer.multimeric_template_meta_data == {"proteinA": [("file.cif", "A")]}
    assert calls == [
        ("prepare_meta", "meta.csv", "/tmp/templates"),
        "templates",
        "features",
    ]


def test_multimeric_object_init_without_template_flags_still_creates_features(monkeypatch):
    calls = []
    interactor = SimpleNamespace(description="proteinA", sequence="ACDE", feature_dict={})

    monkeypatch.setattr(
        MultimericObject,
        "build_description_monomer_mapping",
        lambda self: setattr(self, "monomers_mapping", {"proteinA": interactor}),
    )
    monkeypatch.setattr(
        MultimericObject,
        "create_all_chain_features",
        lambda self: calls.append("features"),
    )

    multimer = MultimericObject([interactor], multimeric_template=False)

    assert not hasattr(multimer, "multimeric_template_meta_data")
    assert calls == ["features"]


def test_build_description_monomer_mapping_and_create_output_name():
    interactors = [
        SimpleNamespace(description="proteinA"),
        SimpleNamespace(description="proteinB"),
    ]
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = interactors
    multimer.description = ""

    multimer.build_description_monomer_mapping()
    multimer.create_output_name()

    assert multimer.monomers_mapping == {
        "proteinA": interactors[0],
        "proteinB": interactors[1],
    }
    assert multimer.description == "proteinA_and_proteinB"


def test_create_chain_id_map_uses_parsed_fasta_and_chain_builder(monkeypatch):
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = [
        SimpleNamespace(description="proteinA", sequence="ACD"),
        SimpleNamespace(description="proteinB", sequence="WXY"),
    ]

    def fake_make_chain_id_map(*, sequences, descriptions):
        assert sequences == ["ACD", "WXY"]
        assert descriptions == ["proteinA", "proteinB"]
        return {"A": SimpleNamespace(sequence="ACD"), "B": SimpleNamespace(sequence="WXY")}

    monkeypatch.setattr(
        objects_mod.pipeline_multimer,
        "_make_chain_id_map",
        fake_make_chain_id_map,
    )

    multimer.create_chain_id_map()

    assert multimer.input_seqs == ["ACD", "WXY"]
    assert sorted(multimer.chain_id_map) == ["A", "B"]


def test_save_binary_matrix_writes_image_file(tmp_path):
    multimer = MultimericObject.__new__(MultimericObject)
    output = tmp_path / "mask.png"

    multimer.save_binary_matrix(np.array([[0, 1], [1, 0]], dtype=int), output)

    assert output.is_file()


def test_save_binary_matrix_falls_back_to_default_font_and_draws_labels(monkeypatch, tmp_path):
    multimer = MultimericObject.__new__(MultimericObject)
    saved = {}
    draw_calls = []

    class FakeImage:
        width = 4
        height = 4

        def save(self, path):
            saved["path"] = path

    class FakeDraw:
        def textsize(self, text, font):
            return (3, 2)

        def text(self, xy, text, font, fill):
            draw_calls.append((xy, text, fill))

    monkeypatch.setattr("PIL.Image.fromarray", lambda _array: FakeImage())
    monkeypatch.setattr("PIL.ImageDraw.Draw", lambda image: FakeDraw())
    monkeypatch.setattr("PIL.ImageFont.truetype", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no arial")))
    monkeypatch.setattr("PIL.ImageFont.load_default", lambda: "default-font")

    output = tmp_path / "mask.png"
    multimer.save_binary_matrix(np.array([[0, 0], [0, 1]], dtype=int), output)

    assert saved["path"] == output
    assert draw_calls


def test_create_multichain_mask_groups_positions_by_template_prefix():
    interactors = [
        SimpleNamespace(
            sequence="AA",
            feature_dict={
                "template_domain_names": np.array([b"1abc_A"], dtype=object),
                "template_sequence": np.array([b"AA"], dtype=object),
            },
        ),
        SimpleNamespace(
            sequence="BB",
            feature_dict={
                "template_domain_names": np.array([b"1abc_B"], dtype=object),
                "template_sequence": np.array([b"B-"], dtype=object),
            },
        ),
        SimpleNamespace(
            sequence="CC",
            feature_dict={
                "template_domain_names": np.array([b"2def_C"], dtype=object),
                "template_sequence": np.array([b"CC"], dtype=object),
            },
        ),
    ]
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = interactors

    mask = multimer.create_multichain_mask()

    assert mask.shape == (6, 6)
    assert np.all(mask[:4, :4] == 1)
    assert np.all(mask[:4, 4:] == 0)
    assert np.all(mask[4:, :4] == 0)
    assert np.all(mask[4:, 4:] == 1)


def test_create_multimeric_template_features_warns_without_metadata(caplog):
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.multimeric_template_dir = None

    with caplog.at_level("WARNING"):
        multimer.create_multimeric_template_features()

    assert "did not give path to multimeric_template_dir" in caplog.text


def test_create_multimeric_template_features_updates_matching_monomer(monkeypatch, tmp_path):
    template_file = tmp_path / "1abc.cif"
    template_file.write_text("data_1abc", encoding="utf-8")
    monomer = SimpleNamespace(sequence="ACDE", feature_dict={})
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = [SimpleNamespace(description="proteinA", sequence="ACDE", feature_dict=monomer.feature_dict)]
    multimer.multimeric_template_dir = str(tmp_path)
    multimer.multimeric_template_meta_data = {"proteinA": [("1abc.cif", "B")]}
    multimer.monomers_mapping = {"proteinA": monomer}
    multimer.threshold_clashes = 12.5
    multimer.hb_allowance = 0.7
    multimer.plddt_threshold = 42.0
    extractor_calls = []

    monkeypatch.setattr(
        objects_mod,
        "extract_multimeric_template_features_for_single_chain",
        lambda **kwargs: extractor_calls.append(kwargs)
        or SimpleNamespace(features={"templated": kwargs["chain_id"]}),
    )

    multimer.create_multimeric_template_features()

    assert monomer.feature_dict["templated"] == "B"
    assert extractor_calls == [{
        "query_seq": "ACDE",
        "pdb_id": "1abc",
        "chain_id": "B",
        "mmcif_file": str(template_file),
        "threshold_clashes": 12.5,
        "hb_allowance": 0.7,
        "plddt_threshold": 42.0,
    }]


def test_create_multimeric_template_features_assigns_duplicate_rows_to_homo_oligomers(monkeypatch, tmp_path):
    template_file = tmp_path / "templ.cif"
    template_file.write_text("data_templ", encoding="utf-8")
    monomer_a = SimpleNamespace(description="proteinA", sequence="AAAA", feature_dict={})
    monomer_b = SimpleNamespace(description="proteinA", sequence="BBBB", feature_dict={})
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = [monomer_a, monomer_b]
    multimer.multimeric_template_dir = str(tmp_path)
    multimer.multimeric_template_meta_data = {
        "proteinA": [("templ.cif", "A"), ("templ.cif", "B")]
    }
    multimer.threshold_clashes = 1000
    multimer.hb_allowance = 0.4
    multimer.plddt_threshold = 0
    calls = []

    monkeypatch.setattr(
        objects_mod,
        "extract_multimeric_template_features_for_single_chain",
        lambda **kwargs: calls.append((kwargs["query_seq"], kwargs["chain_id"]))
        or SimpleNamespace(features={"templated": kwargs["chain_id"]}),
    )

    multimer.create_multimeric_template_features()

    assert calls == [("AAAA", "A"), ("BBBB", "B")]
    assert monomer_a.feature_dict["templated"] == "A"
    assert monomer_b.feature_dict["templated"] == "B"


def test_create_multimeric_template_features_matches_chopped_objects_by_base_description(
    monkeypatch,
    tmp_path,
):
    template_file = tmp_path / "1abc.cif"
    template_file.write_text("data_1abc", encoding="utf-8")
    chopped = ChoppedObject("P04051", "ACDE", {}, [(1, 2), (3, 4)])
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = [chopped]
    multimer.multimeric_template_dir = str(tmp_path)
    multimer.multimeric_template_meta_data = {"P04051": [("1abc.cif", "B")]}
    multimer.threshold_clashes = 1000
    multimer.hb_allowance = 0.4
    multimer.plddt_threshold = 0
    calls = []

    monkeypatch.setattr(
        objects_mod,
        "extract_multimeric_template_features_for_single_chain",
        lambda **kwargs: calls.append((kwargs["query_seq"], kwargs["chain_id"]))
        or SimpleNamespace(features={"templated": kwargs["chain_id"]}),
    )

    multimer.create_multimeric_template_features()

    assert calls == [("ACDE", "B")]
    assert chopped.feature_dict["templated"] == "B"


def test_create_multimeric_template_features_rejects_non_mmcif_files(tmp_path):
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = [SimpleNamespace(description="proteinA", sequence="ACDE", feature_dict={})]
    multimer.multimeric_template_dir = str(tmp_path)
    multimer.multimeric_template_meta_data = {"proteinA": [("bad.pdb", "A")]}
    multimer.monomers_mapping = {"proteinA": SimpleNamespace(sequence="ACDE", feature_dict={})}

    with pytest.raises(AssertionError, match="does not seem to be a mmcif file"):
        multimer.create_multimeric_template_features()


def test_remove_all_seq_features_drops_pairing_keys():
    features = [
        {"msa": 1, "msa_all_seq": 2, "foo_all_seq": 3, "bar": 4},
        {"deletion_matrix_int_all_seq": 5, "x": 6},
    ]

    stripped = MultimericObject.remove_all_seq_features(features)

    assert stripped == [{"msa": 1, "bar": 4}, {"x": 6}]


def test_pair_and_merge_pairs_and_deduplicates_for_heteromer(monkeypatch):
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.pair_msa = True
    calls = {}

    monkeypatch.setattr(
        objects_mod.feature_processing,
        "process_unmerged_features",
        lambda features: calls.setdefault("process_unmerged_features", features),
    )
    monkeypatch.setattr(
        objects_mod.feature_processing,
        "_is_homomer_or_monomer",
        lambda _chains: False,
    )
    def fake_create_paired_features(*, chains):
        calls["create_paired_features"] = chains
        return ["paired"]

    def fake_deduplicate(chains):
        calls["deduplicate_unpaired_sequences"] = chains
        return ["deduped"]

    def fake_crop_chains(chains, **kwargs):
        calls["crop_chains"] = (chains, kwargs)
        return "cropped"

    def fake_merge_chain_features(**kwargs):
        calls["merge_chain_features"] = kwargs
        return {"merged": True}

    monkeypatch.setattr(
        objects_mod.msa_pairing,
        "create_paired_features",
        fake_create_paired_features,
    )
    monkeypatch.setattr(
        objects_mod.msa_pairing,
        "deduplicate_unpaired_sequences",
        fake_deduplicate,
    )
    monkeypatch.setattr(
        objects_mod.feature_processing,
        "crop_chains",
        fake_crop_chains,
    )
    monkeypatch.setattr(
        objects_mod.msa_pairing,
        "merge_chain_features",
        fake_merge_chain_features,
    )
    monkeypatch.setattr(
        objects_mod.feature_processing,
        "process_final",
        lambda example: {"processed": example},
    )

    output = multimer.pair_and_merge({"A": {"chain": "A"}, "B": {"chain": "B"}})

    assert "create_paired_features" in calls
    assert "deduplicate_unpaired_sequences" in calls
    cropped_chains, crop_kwargs = calls["crop_chains"]
    assert cropped_chains == ["deduped"]
    assert crop_kwargs["pair_msa_sequences"] is True
    assert calls["merge_chain_features"]["pair_msa_sequences"] is True
    assert output == {"processed": {"merged": True}}


def test_pair_and_merge_removes_all_seq_features_when_pairing_disabled(monkeypatch):
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.pair_msa = False
    calls = {}

    monkeypatch.setattr(
        objects_mod.feature_processing,
        "process_unmerged_features",
        lambda _features: None,
    )
    def fake_remove_all_seq_features(chains):
        calls["remove_all_seq_features"] = chains
        return ["stripped"]

    def fake_crop_chains(chains, **kwargs):
        calls["crop_chains"] = (chains, kwargs)
        return "cropped"

    def fake_merge_chain_features(**kwargs):
        calls["merge_chain_features"] = kwargs
        return {"merged": False}

    monkeypatch.setattr(
        MultimericObject,
        "remove_all_seq_features",
        staticmethod(fake_remove_all_seq_features),
    )
    monkeypatch.setattr(
        objects_mod.feature_processing,
        "crop_chains",
        fake_crop_chains,
    )
    monkeypatch.setattr(
        objects_mod.msa_pairing,
        "merge_chain_features",
        fake_merge_chain_features,
    )
    monkeypatch.setattr(
        objects_mod.feature_processing,
        "process_final",
        lambda example: example,
    )

    result = multimer.pair_and_merge({"A": {"chain": "A"}})

    assert calls["remove_all_seq_features"] == [{"chain": "A"}]
    assert calls["crop_chains"][0] == ["stripped"]
    assert calls["merge_chain_features"]["pair_msa_sequences"] is False
    assert result == {"merged": False}


def test_create_all_chain_features_converts_assembly_and_injects_multimer_mask(monkeypatch):
    interactors = [
        SimpleNamespace(
            description="proteinA",
            sequence="AC",
            feature_dict={"raw": 1, "template_domain_names": np.array([b"1abc_A"])},
        ),
        SimpleNamespace(
            description="proteinB",
            sequence="WX",
            feature_dict={"raw": 2, "template_domain_names": np.array([b"1abc_B"])},
        ),
    ]
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = interactors
    multimer.multimeric_template = True
    multimer.multichain_mask = None

    monkeypatch.setattr(
        multimer,
        "create_multichain_mask",
        lambda: np.array([[1, 0], [0, 1]], dtype=int),
    )

    def fake_create_chain_id_map():
        multimer.chain_id_map = {
            "A": SimpleNamespace(sequence="AC"),
            "B": SimpleNamespace(sequence="WX"),
        }

    monkeypatch.setattr(multimer, "create_chain_id_map", fake_create_chain_id_map)
    monkeypatch.setattr(
        objects_mod.pipeline_multimer,
        "convert_monomer_features",
        lambda chain_features, chain_id: {"converted": chain_features["raw"], "chain_id": chain_id},
    )
    monkeypatch.setattr(
        objects_mod.pipeline_multimer,
        "add_assembly_features",
        lambda all_chain_features: {"assembled": all_chain_features},
    )
    monkeypatch.setattr(
        multimer,
        "pair_and_merge",
        lambda all_chain_features: {"template_sequence": ["old"], "pair_input": all_chain_features},
    )
    monkeypatch.setattr(
        objects_mod.pipeline_multimer,
        "pad_msa",
        lambda feature_dict, size: {**feature_dict, "padded_to": size},
    )

    multimer.create_all_chain_features()

    assert multimer.multichain_mask.shape == (2, 2)
    assert multimer.all_chain_features == {
        "assembled": {
            "A": {"converted": 1, "chain_id": "A"},
            "B": {"converted": 2, "chain_id": "B"},
        }
    }
    assert multimer.feature_dict["padded_to"] == 512
    assert multimer.feature_dict["template_sequence"] == []
    assert np.array_equal(
        multimer.feature_dict["multichain_mask"], np.array([[1, 0], [0, 1]], dtype=int)
    )


def test_create_all_chain_features_skips_multimeric_template_postprocessing_when_disabled(
    monkeypatch,
):
    interactors = [
        SimpleNamespace(description="proteinA", sequence="AC", feature_dict={"raw": 1}),
    ]
    multimer = MultimericObject.__new__(MultimericObject)
    multimer.interactors = interactors
    multimer.multimeric_template = False

    monkeypatch.setattr(
        multimer,
        "create_chain_id_map",
        lambda: setattr(
            multimer,
            "chain_id_map",
            {"A": SimpleNamespace(sequence="AC")},
        ),
    )
    monkeypatch.setattr(
        objects_mod.pipeline_multimer,
        "convert_monomer_features",
        lambda chain_features, chain_id: {"converted": chain_features["raw"], "chain_id": chain_id},
    )
    monkeypatch.setattr(
        objects_mod.pipeline_multimer,
        "add_assembly_features",
        lambda all_chain_features: {"assembled": all_chain_features},
    )
    monkeypatch.setattr(
        multimer,
        "pair_and_merge",
        lambda all_chain_features: {"template_sequence": ["kept"], "pair_input": all_chain_features},
    )
    monkeypatch.setattr(
        objects_mod.pipeline_multimer,
        "pad_msa",
        lambda feature_dict, size: {**feature_dict, "padded_to": size},
    )

    multimer.create_all_chain_features()

    assert multimer.feature_dict["template_sequence"] == ["kept"]
    assert "multichain_mask" not in multimer.feature_dict
