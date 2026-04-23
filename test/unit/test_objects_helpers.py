from pathlib import Path

import numpy as np

from alphapulldown.objects import MonomericObject


def test_monomeric_object_initializes_and_exposes_uniprot_runner():
    monomer = MonomericObject("desc", "ACDE")

    assert monomer.description == "desc"
    assert monomer.sequence == "ACDE"
    assert monomer.feature_dict == {}
    assert monomer.uniprot_runner is None

    monomer.uniprot_runner = "runner"

    assert monomer.uniprot_runner == "runner"


def test_zip_msa_files_invokes_gzip_for_supported_suffixes(monkeypatch, tmp_path):
    for name in ("hits.a3m", "query.fasta", "align.sto", "profile.hmm", "ignore.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return None

    monkeypatch.setattr("alphapulldown.objects.subprocess.run", fake_run)

    MonomericObject.zip_msa_files(str(tmp_path))

    assert sorted(cmd for cmd, _ in commands) == sorted([
        f"gzip {tmp_path / 'hits.a3m'}",
        f"gzip {tmp_path / 'query.fasta'}",
        f"gzip {tmp_path / 'align.sto'}",
        f"gzip {tmp_path / 'profile.hmm'}",
    ])


def test_unzip_msa_files_returns_true_and_unzips_gz_files(monkeypatch, tmp_path):
    (tmp_path / "hits.a3m.gz").write_text("x", encoding="utf-8")
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return None

    monkeypatch.setattr("alphapulldown.objects.subprocess.run", fake_run)

    used_zipped = MonomericObject.unzip_msa_files(str(tmp_path))

    assert used_zipped is True
    assert [cmd for cmd, _ in commands] == [f"gunzip {tmp_path / 'hits.a3m.gz'}"]


def test_unzip_msa_files_returns_false_when_no_gz_files(tmp_path):
    (tmp_path / "hits.a3m").write_text("x", encoding="utf-8")

    used_zipped = MonomericObject.unzip_msa_files(str(tmp_path))

    assert used_zipped is False


def test_remove_msa_files_removes_only_msa_files_and_empty_directory(tmp_path):
    for name in ("hits.a3m", "query.fasta", "align.sto", "profile.hmm"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    MonomericObject.remove_msa_files(str(tmp_path))

    assert not tmp_path.exists()


def test_remove_msa_files_keeps_directory_if_non_msa_files_remain(tmp_path):
    (tmp_path / "hits.a3m").write_text("x", encoding="utf-8")
    keep_file = tmp_path / "keep.txt"
    keep_file.write_text("x", encoding="utf-8")

    MonomericObject.remove_msa_files(str(tmp_path))

    assert tmp_path.is_dir()
    assert keep_file.is_file()


def test_all_seq_msa_features_keeps_only_pairing_related_keys(monkeypatch, tmp_path):
    monomer = MonomericObject("desc", "ACDE")
    input_fasta_path = str(tmp_path / "input.fasta")
    Path(input_fasta_path).write_text(">x\nACDE\n", encoding="utf-8")
    calls = {}

    class FakeMsa:
        def truncate(self, max_seqs):
            assert max_seqs == 50000
            return self

    def fake_run_msa_tool(*args, **kwargs):
        calls["run_msa_tool"] = (args, kwargs)
        return {"sto": "fake"}

    monkeypatch.setattr("alphapulldown.objects.pipeline.run_msa_tool", fake_run_msa_tool)
    monkeypatch.setattr("alphapulldown.objects.parsers.parse_stockholm", lambda sto: FakeMsa())
    monkeypatch.setattr(
        "alphapulldown.objects.pipeline.make_msa_features",
        lambda _msas: {
            "msa": np.asarray([[1, 2]], dtype=np.int32),
            "msa_species_identifiers": np.asarray([b"9606"]),
            "msa_uniprot_accession_identifiers": np.asarray([b"P12345"]),
            "deletion_matrix_int": np.asarray([[0, 0]], dtype=np.int32),
            "extra_feature": np.asarray([1], dtype=np.int32),
        },
    )

    features = monomer.all_seq_msa_features(
        input_fasta_path=input_fasta_path,
        uniprot_msa_runner="runner",
        output_dir=str(tmp_path),
        use_precomputed_msa=True,
    )

    assert set(features) == {
        "msa_all_seq",
        "msa_species_identifiers_all_seq",
        "msa_uniprot_accession_identifiers_all_seq",
        "deletion_matrix_int_all_seq",
    }
    run_args, run_kwargs = calls["run_msa_tool"]
    assert run_args == (
        "runner",
        input_fasta_path,
        f"{tmp_path}/uniprot_hits.sto",
        "sto",
        True,
    )
    assert run_kwargs == {}


def test_all_seq_msa_features_backfills_missing_uniprot_accession_identifiers(
    monkeypatch, tmp_path
):
    monomer = MonomericObject("desc", "ACDE")
    input_fasta_path = str(tmp_path / "input.fasta")
    Path(input_fasta_path).write_text(">x\nACDE\n", encoding="utf-8")

    class FakeMsa:
        def truncate(self, max_seqs):
            assert max_seqs == 50000
            return self

    monkeypatch.setattr(
        "alphapulldown.objects.pipeline.run_msa_tool",
        lambda *args, **kwargs: {"sto": "fake"},
    )
    monkeypatch.setattr(
        "alphapulldown.objects.parsers.parse_stockholm",
        lambda sto: FakeMsa(),
    )
    monkeypatch.setattr(
        "alphapulldown.objects.pipeline.make_msa_features",
        lambda _msas: {
            "msa": np.asarray([[1, 2], [1, 3]], dtype=np.int32),
            "msa_species_identifiers": np.asarray([b"", b"9606"], dtype=object),
            "deletion_matrix_int": np.asarray([[0, 0], [0, 0]], dtype=np.int32),
        },
    )

    features = monomer.all_seq_msa_features(
        input_fasta_path=input_fasta_path,
        uniprot_msa_runner="runner",
        output_dir=str(tmp_path),
        use_precomputed_msa=False,
    )

    assert features["msa_species_identifiers_all_seq"].tolist() == [b"", b"9606"]
    assert features["msa_uniprot_accession_identifiers_all_seq"].tolist() == [
        b"",
        b"",
    ]
