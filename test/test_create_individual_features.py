import json
from pathlib import Path
import pytest

import alphapulldown.scripts.create_individual_features as cf

class DummyMonomer:
    def __init__(self, description, seq):
        self.description = description
        self.seq = seq
        self.feature_dict = {}
        self.uniprot_runner = None
    def make_features(self, *a, **k):
        return

def write_fasta(p: Path, header: str, seq: str):
    p.write_text(f">{header}\n{seq}\n")

def test_af2_creates_pickled_features(tmp_flags, tmp_path, monkeypatch):
    out = Path(tmp_flags.output_dir)
    write_fasta(Path(tmp_flags.fasta_paths[0]), "A0A024R1R8", "ACDEFGHIKLMNPQRSTVWY")

    monkeypatch.setattr(cf, "MonomericObject", DummyMonomer)
    # uniprot runner not needed; create_pipeline_af2 called but its return not pickled
    monkeypatch.setattr(cf, "create_pipeline_af2", lambda: object())
    monkeypatch.setattr(cf, "create_uniprot_runner", lambda *a, **k: None)
    monkeypatch.setattr(cf.save_meta_data, "get_meta_dict", lambda *_: {"ok": True})

    cf.create_individual_features()

    pkl = out / "A0A024R1R8.pkl"
    meta = list(out.glob("A0A024R1R8_feature_metadata_*.json"))
    assert pkl.exists()
    assert meta and json.loads(meta[0].read_text()) == {"ok": True}

@pytest.mark.parametrize("desc,seq,expect", [
    ("PROT", "ACDEFGHIKLMNPQRSTVWY", "PROT_af3_input.json"),
    ("RNA_TEST", "AUGGCUACG", "RNA_TEST_af3_input.json"),
    ("DNA_TEST", "ATGGCATCG", "DNA_TEST_af3_input.json"),
])
def test_af3_writes_json_per_chain(tmp_flags, tmp_path, monkeypatch, desc, seq, expect):
    tmp_flags.data_pipeline = "alphafold3"
    tmp_flags.fasta_paths = [str(tmp_path / "x.fasta")]
    write_fasta(Path(tmp_flags.fasta_paths[0]), desc, seq)

    class DummyFeat:
        def to_json(self): return '{"test":"features"}'

    class DummyPipe:
        def process(self, *_): return DummyFeat()

    monkeypatch.setattr(cf, "create_pipeline_af3", lambda: DummyPipe())

    cf.create_af3_individual_features()
    out = Path(tmp_flags.output_dir) / expect
    assert out.exists()
    assert json.loads(out.read_text()) == {"test": "features"}

def test_af3_import_failure(tmp_flags, monkeypatch):
    monkeypatch.setattr(cf, "AF3DataPipeline", None)
    monkeypatch.setattr(cf, "AF3DataPipelineConfig", None)
    with pytest.raises(ImportError):
        cf.create_pipeline_af3()

def test_get_database_path_rules(tmp_flags):
    # non-MMseqs2 without data_dir -> ValueError
    tmp_flags.use_mmseqs2 = False
    tmp_flags.data_dir = None
    with pytest.raises(ValueError):
        cf.get_database_path("uniref90")

    # MMseqs2 without data_dir -> None
    tmp_flags.use_mmseqs2 = True
    assert cf.get_database_path("uniref90") is None

def test_db_mapping_representative_keys(tmp_flags):
    tmp_flags.use_mmseqs2 = False
    tmp_flags.data_dir = "/base"
    tmp_flags.data_pipeline = "alphafold2"
    assert cf.get_database_path("uniref90").endswith("uniref90/uniref90.fasta")
    tmp_flags.data_pipeline = "alphafold3"
    assert cf.get_database_path("uniref90").endswith("uniref90_2022_05.fa")

def test_main_required_flags_validation_exits(tmp_flags, monkeypatch):
    tmp_flags.fasta_paths = None
    monkeypatch.setattr(cf.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as e:
        cf.main([])
    assert e.value.code == 1


def test_skip_existing_actually_skips(tmp_flags, tmp_path, monkeypatch):
    write_fasta(Path(tmp_flags.fasta_paths[0]), "A0A024R1R8", "ACDEFGHIKL")
    out = Path(tmp_flags.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "A0A024R1R8.pkl").write_bytes(b"x")
    tmp_flags.skip_existing = True

    called = {"features": 0}
    class DummyMonomer:
        def __init__(self, d, s): self.description = d; self.uniprot_runner=None
        def make_features(self, *a, **k): called["features"] += 1
    monkeypatch.setattr(cf, "MonomericObject", DummyMonomer)
    monkeypatch.setattr(cf, "create_pipeline_af2", lambda: object())
    monkeypatch.setattr(cf, "create_uniprot_runner", lambda *a, **k: None)
    monkeypatch.setattr(cf.save_meta_data, "get_meta_dict", lambda *_: {})

    cf.create_individual_features()
    assert called["features"] == 0  # nothing recomputed

def test_af2_override_respected(tmp_flags):
    # AF2 honors explicit flag values in create_arguments()
    tmp_flags.use_mmseqs2 = False
    tmp_flags.data_pipeline = "alphafold2"
    tmp_flags.data_dir = "/base"
    tmp_flags.uniref90_database_path = "/override/uniref90.fa"  # explicit override
    # clear other DB flags to avoid accidental pass-through
    tmp_flags.uniref30_database_path = None
    tmp_flags.mgnify_database_path = None
    tmp_flags.bfd_database_path = None
    tmp_flags.small_bfd_database_path = None
    tmp_flags.pdb70_database_path = None
    tmp_flags.uniprot_database_path = None
    tmp_flags.pdb_seqres_database_path = None
    tmp_flags.template_mmcif_dir = None
    tmp_flags.obsolete_pdbs_path = None

    cf.create_arguments()
    assert tmp_flags.uniref90_database_path == "/override/uniref90.fa"


def test_af3_override_respected_after_refactor(tmp_flags, monkeypatch):
    tmp_flags.data_pipeline = "alphafold3"
    tmp_flags.use_mmseqs2 = False
    tmp_flags.data_dir = "/db3"
    tmp_flags.uniref90_database_path = "/override/uniref90.fa"

    captured = {}
    class DummyCfg:
        def __init__(self, **kw): captured.update(kw)
    class DummyPipe: pass

    monkeypatch.setattr(cf, "AF3DataPipelineConfig", lambda **kw: DummyCfg(**kw))
    monkeypatch.setattr(cf, "AF3DataPipeline", lambda cfg: DummyPipe())

    cf.create_pipeline_af3()
    assert captured["uniref90_database_path"] == "/override/uniref90.fa"

