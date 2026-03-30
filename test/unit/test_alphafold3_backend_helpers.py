import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alphapulldown.objects import MonomericObject


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "alphapulldown"
    / "folding_backend"
    / "alphafold3_backend.py"
)


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_alphafold3_backend_stubs(tmp_path: Path) -> None:
    for module_name in list(sys.modules):
        if module_name == "alphafold3" or module_name.startswith("alphafold3."):
            sys.modules.pop(module_name, None)

    try:
        import jax  # type: ignore

        if not hasattr(jax, "Device"):
            jax.Device = type("Device", (), {})
        if not hasattr(jax, "tree_map"):
            jax.tree_map = jax.tree_util.tree_map
        if not hasattr(jax, "device_put"):
            jax.device_put = lambda value, device=None: value
        if not hasattr(jax, "jit"):
            jax.jit = lambda func, device=None: func
        if not hasattr(jax, "random"):
            jax.random = SimpleNamespace(PRNGKey=lambda seed: seed)
    except Exception:  # pragma: no cover - conftest already installs a stub
        pass

    if "haiku" not in sys.modules:
        haiku = types.ModuleType("haiku")
        haiku.Params = dict

        def _transform(fn):
            return SimpleNamespace(apply=fn)

        haiku.transform = _transform
        sys.modules["haiku"] = haiku

    terms_dir = tmp_path / "af3cpp"
    terms_dir.mkdir(parents=True, exist_ok=True)
    (terms_dir / "OUTPUT_TERMS_OF_USE.md").write_text("stub terms", encoding="utf-8")

    alphafold3_pkg = _package("alphafold3")
    cpp_mod = types.ModuleType("alphafold3.cpp")
    cpp_mod.__file__ = str(terms_dir / "cpp_stub.so")

    common_pkg = _package("alphafold3.common")
    base_config_mod = types.ModuleType("alphafold3.common.base_config")
    base_config_mod.BaseConfig = type("BaseConfig", (), {})
    folding_input_mod = types.ModuleType("alphafold3.common.folding_input")

    @dataclasses.dataclass(frozen=True)
    class StubTemplate:
        mmcif: str
        query_to_template_map: dict[int, int]

    class StubProteinChain:
        def __init__(
            self,
            *,
            id,
            sequence,
            ptms=None,
            residue_ids=None,
            description=None,
            paired_msa="",
            unpaired_msa="",
            templates=None,
        ):
            self.id = id
            self.sequence = sequence
            self.ptms = [] if ptms is None else list(ptms)
            self.residue_ids = None if residue_ids is None else list(residue_ids)
            self.description = description
            self.paired_msa = paired_msa
            self.unpaired_msa = unpaired_msa
            self.templates = None if templates is None else list(templates)

    class StubRnaChain:
        def __init__(
            self,
            *,
            id,
            sequence,
            modifications=None,
            residue_ids=None,
            description=None,
            unpaired_msa="",
        ):
            self.id = id
            self.sequence = sequence
            self.modifications = [] if modifications is None else list(modifications)
            self.residue_ids = None if residue_ids is None else list(residue_ids)
            self.description = description
            self.unpaired_msa = unpaired_msa

    class StubDnaChain:
        def __init__(
            self,
            *,
            id,
            sequence,
            modifications=None,
            residue_ids=None,
            description=None,
        ):
            self.id = id
            self.sequence = sequence
            self._modifications = [] if modifications is None else list(modifications)
            self.residue_ids = None if residue_ids is None else list(residue_ids)
            self.description = description

        def modifications(self):
            return list(self._modifications)

    class StubLigand:
        def __init__(self, *, id, ccd_ids=None, smiles=None, description=None):
            self.id = id
            self.ccd_ids = None if ccd_ids is None else list(ccd_ids)
            self.smiles = smiles
            self.description = description

    @dataclasses.dataclass(frozen=True)
    class StubInput:
        name: str
        chains: tuple
        rng_seeds: tuple[int, ...]
        user_ccd: object | None = None

        def __post_init__(self):
            object.__setattr__(self, "chains", tuple(self.chains))
            object.__setattr__(self, "rng_seeds", tuple(self.rng_seeds))

        def sanitised_name(self) -> str:
            return self.name.replace(" ", "_")

        def to_json(self) -> str:
            def _serialize_chain(chain):
                if isinstance(chain, (str, int, float, bool)) or chain is None:
                    return chain
                payload = {"type": chain.__class__.__name__, "id": getattr(chain, "id", None)}
                if hasattr(chain, "sequence"):
                    payload["sequence"] = chain.sequence
                if hasattr(chain, "paired_msa"):
                    payload["paired_msa"] = chain.paired_msa
                if hasattr(chain, "unpaired_msa"):
                    payload["unpaired_msa"] = chain.unpaired_msa
                return payload

            return json.dumps(
                {"name": self.name, "chains": [_serialize_chain(chain) for chain in self.chains]}
            )

        @classmethod
        def from_json(cls, json_str: str):
            payload = json.loads(json_str)
            if isinstance(payload, list):
                payload = payload[0]
            if not isinstance(payload, dict):
                raise ValueError("Unsupported AF3 JSON payload")

            def _parse_template(template_payload):
                return StubTemplate(
                    mmcif=template_payload["mmcif"],
                    query_to_template_map={
                        int(key): int(value)
                        for key, value in template_payload.get(
                            "query_to_template_map", {}
                        ).items()
                    },
                )

            chains = []
            for chain_payload in payload.get("chains", []):
                chain_type = chain_payload["type"]
                common_kwargs = {
                    "id": chain_payload["id"],
                    "sequence": chain_payload.get("sequence", ""),
                    "description": chain_payload.get("description"),
                    "residue_ids": chain_payload.get("residue_ids"),
                }
                if chain_type == "protein":
                    chains.append(
                        StubProteinChain(
                            **common_kwargs,
                            ptms=chain_payload.get("ptms", []),
                            paired_msa=chain_payload.get("paired_msa", ""),
                            unpaired_msa=chain_payload.get("unpaired_msa", ""),
                            templates=[
                                _parse_template(template_payload)
                                for template_payload in chain_payload.get("templates", [])
                            ],
                        )
                    )
                elif chain_type == "rna":
                    chains.append(
                        StubRnaChain(
                            **common_kwargs,
                            modifications=chain_payload.get("modifications", []),
                            unpaired_msa=chain_payload.get("unpaired_msa", ""),
                        )
                    )
                elif chain_type == "dna":
                    chains.append(
                        StubDnaChain(
                            **common_kwargs,
                            modifications=chain_payload.get("modifications", []),
                        )
                    )
                elif chain_type == "ligand":
                    chains.append(
                        StubLigand(
                            id=chain_payload["id"],
                            ccd_ids=chain_payload.get("ccd_ids"),
                            smiles=chain_payload.get("smiles"),
                            description=chain_payload.get("description"),
                        )
                    )
                else:
                    raise ValueError(f"Unsupported chain type: {chain_type}")
            return cls(
                name=payload.get("name", "json_input"),
                chains=tuple(chains),
                rng_seeds=tuple(payload.get("rng_seeds", [0])),
            )

        def with_multiple_seeds(self, num_seeds: int):
            return dataclasses.replace(self, rng_seeds=tuple(range(1, num_seeds + 1)))

    folding_input_mod.Input = StubInput
    folding_input_mod.Template = StubTemplate
    folding_input_mod.ProteinChain = StubProteinChain
    folding_input_mod.RnaChain = StubRnaChain
    folding_input_mod.DnaChain = StubDnaChain
    folding_input_mod.Ligand = StubLigand

    constants_pkg = _package("alphafold3.constants")
    chemical_components_mod = types.ModuleType("alphafold3.constants.chemical_components")
    chemical_components_mod.Ccd = lambda user_ccd=None: SimpleNamespace(user_ccd=user_ccd)

    data_pkg = _package("alphafold3.data")
    featurisation_mod = types.ModuleType("alphafold3.data.featurisation")
    featurisation_mod.featurise_input = lambda **_kwargs: []
    parsers_mod = types.ModuleType("alphafold3.data.parsers")

    def _parse_fasta(text: str):
        descriptions = []
        sequences = []
        current_description = None
        current_sequence = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_description is not None:
                    descriptions.append(current_description)
                    sequences.append("".join(current_sequence))
                current_description = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        if current_description is not None:
            descriptions.append(current_description)
            sequences.append("".join(current_sequence))
        return sequences, descriptions

    parsers_mod.parse_fasta = _parse_fasta

    af3_jax_pkg = _package("alphafold3.jax")
    attention_pkg = _package("alphafold3.jax.attention")
    attention_mod = types.ModuleType("alphafold3.jax.attention.attention")
    attention_mod.Implementation = str

    model_pkg = _package("alphafold3.model")
    features_mod = types.ModuleType("alphafold3.model.features")
    features_mod.BatchDict = dict
    params_mod = types.ModuleType("alphafold3.model.params")
    params_mod.get_model_haiku_params = (
        lambda model_dir=None: {"__meta__": {"__identifier__": np.asarray([1], dtype=np.uint8)}}
    )
    post_processing_mod = types.ModuleType("alphafold3.model.post_processing")
    post_processing_mod.write_output = lambda **_kwargs: None
    post_processing_mod.write_embeddings = lambda **_kwargs: None
    model_model_mod = types.ModuleType("alphafold3.model.model")
    model_model_mod.ModelResult = dict

    @dataclasses.dataclass(frozen=True)
    class StubInferenceResult:
        predicted_structure: object
        metadata: dict

    class StubModel:
        class Config:
            pass

    model_model_mod.InferenceResult = StubInferenceResult
    model_model_mod.Model = StubModel

    components_pkg = _package("alphafold3.model.components")
    utils_mod = types.ModuleType("alphafold3.model.components.utils")
    utils_mod.remove_invalidly_typed_feats = lambda feats: feats

    structure_pkg = _package("alphafold3.structure")
    mmcif_mod = types.ModuleType("alphafold3.structure.mmcif")
    mmcif_mod.from_string = (
        lambda mmcif_string: (_ for _ in ()).throw(ValueError("invalid mmcif"))
        if mmcif_string == "INVALID"
        else {"mmcif": mmcif_string}
    )

    modules = {
        "alphafold3": alphafold3_pkg,
        "alphafold3.cpp": cpp_mod,
        "alphafold3.common": common_pkg,
        "alphafold3.common.base_config": base_config_mod,
        "alphafold3.common.folding_input": folding_input_mod,
        "alphafold3.constants": constants_pkg,
        "alphafold3.constants.chemical_components": chemical_components_mod,
        "alphafold3.data": data_pkg,
        "alphafold3.data.featurisation": featurisation_mod,
        "alphafold3.data.parsers": parsers_mod,
        "alphafold3.jax": af3_jax_pkg,
        "alphafold3.jax.attention": attention_pkg,
        "alphafold3.jax.attention.attention": attention_mod,
        "alphafold3.model": model_pkg,
        "alphafold3.model.features": features_mod,
        "alphafold3.model.params": params_mod,
        "alphafold3.model.post_processing": post_processing_mod,
        "alphafold3.model.model": model_model_mod,
        "alphafold3.model.components": components_pkg,
        "alphafold3.model.components.utils": utils_mod,
        "alphafold3.structure": structure_pkg,
        "alphafold3.structure.mmcif": mmcif_mod,
    }

    for name, module in modules.items():
        sys.modules[name] = module

    alphafold3_pkg.cpp = cpp_mod
    alphafold3_pkg.common = common_pkg
    alphafold3_pkg.constants = constants_pkg
    alphafold3_pkg.data = data_pkg
    alphafold3_pkg.jax = af3_jax_pkg
    alphafold3_pkg.model = model_pkg
    alphafold3_pkg.structure = structure_pkg
    common_pkg.base_config = base_config_mod
    common_pkg.folding_input = folding_input_mod
    constants_pkg.chemical_components = chemical_components_mod
    data_pkg.featurisation = featurisation_mod
    data_pkg.parsers = parsers_mod
    af3_jax_pkg.attention = attention_pkg
    attention_pkg.attention = attention_mod
    model_pkg.features = features_mod
    model_pkg.params = params_mod
    model_pkg.post_processing = post_processing_mod
    model_pkg.model = model_model_mod
    model_pkg.components = components_pkg
    components_pkg.utils = utils_mod
    structure_pkg.mmcif = mmcif_mod


@pytest.fixture(scope="module")
def af3_backend_module(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("af3_backend_stubs")
    _install_alphafold3_backend_stubs(tmp_path)
    sys.modules.pop("alphapulldown.folding_backend.alphafold3_backend", None)
    spec = importlib.util.spec_from_file_location(
        "alphapulldown.folding_backend.alphafold3_backend",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeChainsTable:
    def __init__(self, chain_ids):
        self._chain_ids = list(chain_ids)

    def apply_array_to_column(self, column_name, arr):
        assert column_name == "id"
        return np.asarray([self._chain_ids[index] for index in arr], dtype=object)


class FakeResiduesTable:
    def __init__(self, auth_seq_id, residue_ids, insertion_codes, chain_keys):
        self.auth_seq_id = np.asarray(auth_seq_id, dtype=object)
        self.id = np.asarray(residue_ids, dtype=np.int32)
        self.insertion_code = np.asarray(insertion_codes, dtype=object)
        self.chain_key = np.asarray(chain_keys, dtype=np.int32)


class FakeStructure:
    def __init__(self, auth_seq_id, residue_ids, insertion_codes, chain_keys, chain_ids):
        self.residues_table = FakeResiduesTable(
            auth_seq_id, residue_ids, insertion_codes, chain_keys
        )
        self.chains_table = FakeChainsTable(chain_ids)
        self.last_update = None

    def copy_and_update_residues(self, **kwargs):
        self.last_update = kwargs
        return SimpleNamespace(updated_residues=kwargs)


def test_output_name_helpers_compact_and_normalise_fragments(af3_backend_module):
    assert af3_backend_module._normalise_output_name_fragment(" A/B : C ") == "A_B___C"
    assert af3_backend_module._collapse_repeated_name_fragments(
        ["a", "a", "b", "b", "b", "c"]
    ) == ["a__x2", "b__x3", "c"]
    compacted = af3_backend_module._compact_output_job_name("x" * 260, max_chars=40)
    assert compacted.startswith("x")
    assert len(compacted) <= 40


def test_compound_and_region_name_helpers_handle_existing_suffixes(af3_backend_module):
    assert af3_backend_module._compact_existing_compound_name(
        "protA_and_protA_and_protB"
    ) == "protA__x2_and_protB"
    assert af3_backend_module._regions_to_name_fragment([(1, 10), (20, 25)]) == "1-10_20-25"
    assert af3_backend_module._json_input_basename("/tmp/A0ABD7FQG0_af3_input.json") == "A0ABD7FQG0"
    assert af3_backend_module._json_input_basename("/tmp/job_input.json") == "job"


def test_object_and_output_job_name_helpers_cover_json_monomer_and_multimer_inputs(
    af3_backend_module,
):
    monomer = MonomericObject("protA", "ACDE")
    multimer = object.__new__(af3_backend_module.MultimericObject)
    multimer.description = "protA_and_protA_and_protB"
    json_entry = {"json_input": "/tmp/sample_af3_input.json", "regions": [(5, 9)]}

    assert af3_backend_module._object_name_fragment(json_entry) == "sample__5-9"
    assert af3_backend_module._object_name_fragment(monomer) == "protA"
    assert af3_backend_module._object_name_fragment(multimer) == "protA__x2_and_protB"

    job_name = af3_backend_module._build_output_job_name(
        [{"object": [json_entry, monomer, monomer, multimer]}]
    )
    assert job_name == "sample__5-9_and_protA__x2_and_protA__x2_and_protB"


def test_duplicate_occurrence_and_author_id_helpers_handle_overflow(af3_backend_module):
    assert af3_backend_module._duplicate_occurrence_to_insertion_code(1) == "."
    assert af3_backend_module._duplicate_occurrence_to_insertion_code(2) == "A"
    assert af3_backend_module._duplicate_occurrence_to_insertion_code(27) == "Z"
    with pytest.raises(ValueError):
        af3_backend_module._duplicate_occurrence_to_insertion_code(28)
    assert af3_backend_module._duplicate_occurrence_to_insertion_code(28, strict=False) == "."

    author_ids, insertion_codes, labels = af3_backend_module._author_ids_with_insertion_codes(
        ["A", "A", "A"],
        ["10", "10", "10"],
        strict=False,
    )
    assert author_ids == ["10", "10", "10"]
    assert insertion_codes == [".", "A", "B"]
    assert labels == ["10", "10A", "10B"]


def test_author_id_helpers_respect_existing_insertion_codes_and_overflow_labels(
    af3_backend_module,
):
    author_ids, insertion_codes, labels = af3_backend_module._author_ids_with_insertion_codes(
        ["A"] * 28,
        ["5"] * 28,
        strict=False,
    )
    assert author_ids[0] == "5"
    assert insertion_codes[1] == "A"
    assert insertion_codes[26] == "Z"
    assert insertion_codes[27] == "."
    assert labels[27] == "5[28]"

    author_ids, insertion_codes, labels = af3_backend_module._author_ids_with_insertion_codes(
        ["A", "A"],
        ["5", "5"],
        existing_insertion_codes=[".", "C"],
    )
    assert insertion_codes == [".", "C"]
    assert labels == ["5", "5C"]


def test_residue_author_and_existing_insertion_code_helpers_normalise_structure_tables(
    af3_backend_module,
):
    fallback_structure = FakeStructure(
        auth_seq_id=[".", "?"], residue_ids=[101, 102], insertion_codes=["?", ""], chain_keys=[0, 0], chain_ids=["A"]
    )
    assert af3_backend_module._residue_author_ids(fallback_structure) == ["101", "102"]
    assert af3_backend_module._existing_insertion_codes(fallback_structure) == [".", "."]

    explicit_structure = FakeStructure(
        auth_seq_id=["7", "8A"], residue_ids=[1, 2], insertion_codes=[".", "B"], chain_keys=[0, 0], chain_ids=["A"]
    )
    assert af3_backend_module._residue_author_ids(explicit_structure) == ["7", "8A"]
    assert af3_backend_module._existing_insertion_codes(explicit_structure) == [".", "B"]


def test_coerce_json_scalar_and_sequential_residue_ids(af3_backend_module):
    assert af3_backend_module._coerce_json_scalar("17") == 17
    assert af3_backend_module._coerce_json_scalar("A17") == "A17"
    assert af3_backend_module._sequential_residue_ids_per_chain(["A", "A", "B", "A", "B"]) == [1, 2, 1, 3, 2]


def test_augment_confidence_json_with_author_numbering_updates_sidecar(af3_backend_module, tmp_path):
    confidences_path = tmp_path / "confidences.json"
    confidences_path.write_text(
        json.dumps({"token_res_ids": ["1", "2"]}),
        encoding="utf-8",
    )
    inference_result = af3_backend_module.model.InferenceResult(
        predicted_structure=None,
        metadata={
            "token_auth_res_ids": ["10", "11"],
            "token_pdb_ins_codes": [".", "A"],
            "token_auth_res_labels": ["10", "11A"],
        },
    )

    af3_backend_module._augment_confidence_json_with_author_numbering(
        confidences_path,
        inference_result,
    )

    payload = json.loads(confidences_path.read_text(encoding="utf-8"))
    assert payload["token_label_seq_ids"] == [1, 2]
    assert payload["token_auth_res_ids"] == [10, 11]
    assert payload["token_pdb_ins_codes"] == [".", "A"]
    assert payload["token_auth_res_labels"] == ["10", "11A"]


def test_augment_confidence_json_with_author_numbering_is_noop_without_metadata(
    af3_backend_module,
    tmp_path,
):
    confidences_path = tmp_path / "confidences.json"
    original = {"token_res_ids": ["1", "2"], "other": "value"}
    confidences_path.write_text(json.dumps(original), encoding="utf-8")
    inference_result = af3_backend_module.model.InferenceResult(
        predicted_structure=None,
        metadata={},
    )

    af3_backend_module._augment_confidence_json_with_author_numbering(
        confidences_path,
        inference_result,
    )

    assert json.loads(confidences_path.read_text(encoding="utf-8")) == original


def test_make_viewer_compatible_inference_result_rewrites_residue_and_token_numbering(
    af3_backend_module,
):
    structure = FakeStructure(
        auth_seq_id=["10", "10", "7"],
        residue_ids=[101, 102, 201],
        insertion_codes=[".", ".", "C"],
        chain_keys=[0, 0, 1],
        chain_ids=["A", "B"],
    )
    inference_result = af3_backend_module.model.InferenceResult(
        predicted_structure=structure,
        metadata={
            "token_chain_ids": ["A", "A", "B", "B"],
            "token_res_ids": [10, 10, 7, 7],
        },
    )

    viewer_result = af3_backend_module._make_viewer_compatible_inference_result(
        inference_result
    )

    assert structure.last_update is not None
    np.testing.assert_array_equal(structure.last_update["res_id"], np.asarray([1, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        structure.last_update["res_auth_seq_id"],
        np.asarray(["10", "10", "7"], dtype=object),
    )
    np.testing.assert_array_equal(
        structure.last_update["res_insertion_code"],
        np.asarray([".", "A", "C"], dtype=object),
    )
    assert viewer_result.metadata["token_res_ids"] == [1, 2, 1, 2]
    assert viewer_result.metadata["token_auth_res_ids"] == ["10", "10", "7", "7"]
    assert viewer_result.metadata["token_pdb_ins_codes"] == [".", "A", ".", "A"]
    assert viewer_result.metadata["token_auth_res_labels"] == ["10", "10A", "7", "7A"]


def test_write_outputs_writes_per_sample_final_outputs_and_ranking_csv(
    af3_backend_module,
    monkeypatch,
    tmp_path,
):
    output_calls = []
    embedding_calls = []
    augment_calls = []

    monkeypatch.setattr(
        af3_backend_module,
        "_make_viewer_compatible_inference_result",
        lambda result: result,
    )
    monkeypatch.setattr(
        af3_backend_module.post_processing,
        "write_output",
        lambda **kwargs: output_calls.append(kwargs),
    )
    monkeypatch.setattr(
        af3_backend_module.post_processing,
        "write_embeddings",
        lambda **kwargs: embedding_calls.append(kwargs),
    )
    monkeypatch.setattr(
        af3_backend_module,
        "_augment_confidence_json_with_author_numbering",
        lambda path, result: augment_calls.append((Path(path).name, result.metadata["ranking_score"])),
    )

    result_low = af3_backend_module.model.InferenceResult(
        predicted_structure=None,
        metadata={"ranking_score": 0.2},
    )
    result_high = af3_backend_module.model.InferenceResult(
        predicted_structure=None,
        metadata={"ranking_score": 0.9},
    )
    fold_input = af3_backend_module.folding_input.Input(
        name="job",
        chains=("A",),
        rng_seeds=(7,),
    )
    results = [
        af3_backend_module.ResultsForSeed(
            seed=7,
            inference_results=[result_low, result_high],
            full_fold_input=fold_input,
            embeddings={"single_embeddings": np.asarray([[1.0]], dtype=np.float32)},
            distogram=np.asarray([[0.5]], dtype=np.float32),
        )
    ]

    af3_backend_module.write_outputs(results, tmp_path, "job_name")

    assert len(output_calls) == 3
    assert output_calls[-1]["name"] == "job_name"
    assert output_calls[-1]["terms_of_use"] == "stub terms"
    assert embedding_calls[0]["name"] == "job_name_seed-7"
    assert (tmp_path / "seed-7_distogram" / "job_name_seed-7_distogram.npz").is_file()

    ranking_csv = (tmp_path / "ranking_scores.csv").read_text(encoding="utf-8")
    assert "seed,sample,ranking_score" in ranking_csv
    assert "7,1,0.9" in ranking_csv
    assert augment_calls == [
        ("confidences.json", 0.2),
        ("confidences.json", 0.9),
        ("job_name_confidences.json", 0.9),
    ]


def test_process_fold_input_rejects_missing_chains(af3_backend_module, tmp_path):
    fold_input = af3_backend_module.folding_input.Input(
        name="broken",
        chains=(),
        rng_seeds=(1,),
    )
    with pytest.raises(ValueError, match="no chains"):
        af3_backend_module.process_fold_input(
            fold_input=fold_input,
            model_runner=None,
            output_dir=tmp_path,
        )


def test_process_fold_input_writes_json_and_skips_inference_without_model_runner(
    af3_backend_module,
    tmp_path,
):
    fold_input = af3_backend_module.folding_input.Input(
        name="job name",
        chains=("A",),
        rng_seeds=(1,),
    )

    returned = af3_backend_module.process_fold_input(
        fold_input=fold_input,
        model_runner=None,
        output_dir=tmp_path,
    )

    assert returned is fold_input
    prepared_path = tmp_path / "job_name_data.json"
    assert prepared_path.is_file()
    assert json.loads(prepared_path.read_text(encoding="utf-8")) == {
        "name": "job name",
        "chains": ["A"],
    }


def test_process_fold_input_checks_model_params_and_calls_predict_structure(
    af3_backend_module,
    monkeypatch,
    tmp_path,
):
    fold_input = af3_backend_module.folding_input.Input(
        name="job",
        chains=("A",),
        rng_seeds=(42,),
    )
    predict_calls = []

    class FakeRunner:
        @property
        def model_params(self):
            return {"ok": True}

    fake_results = ["result"]
    monkeypatch.setattr(
        af3_backend_module,
        "predict_structure",
        lambda **kwargs: predict_calls.append(kwargs) or fake_results,
    )

    returned = af3_backend_module.process_fold_input(
        fold_input=fold_input,
        model_runner=FakeRunner(),
        output_dir=tmp_path,
        buckets=[256],
        resolve_msa_overlaps=False,
        debug_msas=True,
    )

    assert returned == fake_results
    assert predict_calls[0]["fold_input"] is fold_input
    assert predict_calls[0]["output_dir"] == tmp_path
    assert predict_calls[0]["buckets"] == [256]
    assert predict_calls[0]["resolve_msa_overlaps"] is False
    assert predict_calls[0]["debug_msas"] is True


def test_predict_structure_writes_final_msa_and_collects_optional_outputs(
    af3_backend_module,
    monkeypatch,
    tmp_path,
):
    fold_input = af3_backend_module.folding_input.Input(
        name="job name",
        chains=("A",),
        rng_seeds=(7,),
    )
    example = {
        "msa": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "num_alignments": 1,
    }
    inference_result = af3_backend_module.model.InferenceResult(
        predicted_structure=None,
        metadata={"token_chain_ids": ["A", "A"]},
    )

    monkeypatch.setattr(
        af3_backend_module.featurisation,
        "featurise_input",
        lambda **kwargs: [example],
    )
    monkeypatch.setattr(
        af3_backend_module,
        "ids_to_a3m_af3",
        lambda rows: ">query\nABC\n",
    )

    class FakeRunner:
        def run_inference(self, batch, rng_key):
            return {
                "single_embeddings": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
                "pair_embeddings": np.ones((3, 3), dtype=np.float32),
                "distogram": {"distogram": np.full((3, 3), 0.5, dtype=np.float32)},
            }

        def extract_structures(self, batch, result, target_name):
            return [inference_result]

    results = af3_backend_module.predict_structure(
        fold_input=fold_input,
        model_runner=FakeRunner(),
        buckets=[128],
        output_dir=tmp_path,
        resolve_msa_overlaps=False,
        debug_msas=True,
    )

    assert len(results) == 1
    assert results[0].seed == 7
    assert results[0].embeddings["single_embeddings"].shape == (2, 1)
    assert results[0].embeddings["pair_embeddings"].shape == (2, 2)
    assert results[0].distogram.shape == (2, 2)
    final_msa = tmp_path / "job_name_seed-7_final_complex_msa.a3m"
    assert final_msa.read_text(encoding="utf-8") == ">query\nABC\n"


def test_af3_setup_builds_model_runner_and_validates_gpu_capability(
    af3_backend_module,
    monkeypatch,
    tmp_path,
):
    class FakeConfig:
        def __init__(self):
            self.global_config = SimpleNamespace(
                flash_attention_implementation=None
            )
            self.heads = SimpleNamespace(
                diffusion=SimpleNamespace(eval=SimpleNamespace(num_samples=None))
            )
            self.num_recycles = None
            self.return_embeddings = False
            self.return_distogram = False

    class FakeModel:
        Config = FakeConfig

    cache_updates = []
    monkeypatch.setattr(
        sys.modules["alphafold3.model.model"],
        "Model",
        FakeModel,
    )
    monkeypatch.setattr(
        af3_backend_module.jax,
        "config",
        SimpleNamespace(update=lambda key, value: cache_updates.append((key, value))),
        raising=False,
    )
    monkeypatch.setattr(
        af3_backend_module.jax,
        "local_devices",
        lambda backend="gpu": [SimpleNamespace(compute_capability=8.0)],
    )

    configured = af3_backend_module.AlphaFold3Backend.setup(
        num_diffusion_samples=8,
        flash_attention_implementation="triton",
        buckets=[128],
        jax_compilation_cache_dir=str(tmp_path / "jax-cache"),
        model_dir=str(tmp_path / "models"),
        num_recycles=12,
        return_embeddings=True,
        return_distogram=True,
    )

    runner = configured["model_runner"]
    assert runner.device.compute_capability == 8.0
    assert runner.model_dir == tmp_path / "models"
    assert runner.config.global_config.flash_attention_implementation == "triton"
    assert runner.config.heads.diffusion.eval.num_samples == 8
    assert runner.config.num_recycles == 12
    assert runner.config.return_embeddings is True
    assert runner.config.return_distogram is True
    assert cache_updates == [("jax_compilation_cache_dir", str(tmp_path / "jax-cache"))]

    monkeypatch.setattr(
        af3_backend_module.jax,
        "local_devices",
        lambda backend="gpu": [SimpleNamespace(compute_capability=5.0)],
    )
    with pytest.raises(ValueError, match="requires at least GPU compute capability 6.0"):
        af3_backend_module.AlphaFold3Backend.setup(
            num_diffusion_samples=1,
            flash_attention_implementation="triton",
            buckets=[128],
            jax_compilation_cache_dir=None,
            model_dir=str(tmp_path / "models"),
        )


def test_af3_predict_expands_num_seeds_and_calls_process_fold_input(
    af3_backend_module,
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeFoldInput:
        def __init__(self, name):
            self.name = name
            self.rng_seeds = (3,)

        def with_multiple_seeds(self, num_seeds):
            return FakeFoldInput(f"{self.name}__{num_seeds}")

    fake_input = FakeFoldInput("job")
    monkeypatch.setattr(
        af3_backend_module.AlphaFold3Backend,
        "prepare_input",
        staticmethod(lambda **kwargs: [{fake_input: (tmp_path, False)}]),
    )
    monkeypatch.setattr(
        af3_backend_module,
        "process_fold_input",
        lambda **kwargs: calls.append(kwargs) or ["predicted"],
    )

    results = list(
        af3_backend_module.AlphaFold3Backend.predict(
            model_runner=object(),
            objects_to_model=[{"object": object(), "output_dir": str(tmp_path)}],
            random_seed=7,
            buckets=256,
            num_seeds=4,
            debug_msas=True,
        )
    )

    assert len(results) == 1
    assert results[0]["prediction_results"] == ["predicted"]
    assert calls[0]["buckets"] == (256,)
    assert calls[0]["resolve_msa_overlaps"] is False
    assert calls[0]["debug_msas"] is True
    assert calls[0]["fold_input"].name == "job__4"


def test_af3_postprocess_skips_missing_args_and_writes_outputs(
    af3_backend_module,
    monkeypatch,
    tmp_path,
):
    calls = []
    monkeypatch.setattr(
        af3_backend_module,
        "write_outputs",
        lambda **kwargs: calls.append(kwargs),
    )

    af3_backend_module.AlphaFold3Backend.postprocess(output_dir=tmp_path)
    assert calls == []

    fold_input = af3_backend_module.folding_input.Input(
        name="job",
        chains=("A",),
        rng_seeds=(1,),
    )
    af3_backend_module.AlphaFold3Backend.postprocess(
        prediction_results=["result"],
        output_dir=tmp_path,
        multimeric_object=fold_input,
    )

    assert calls == [
        {
            "all_inference_results": ["result"],
            "output_dir": tmp_path,
            "job_name": "job",
        }
    ]


def _write_stub_af3_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_prepare_input_merges_json_chain_and_rewrites_duplicate_ids(
    af3_backend_module,
    tmp_path,
):
    json_path = _write_stub_af3_json(
        tmp_path / "input.json",
        {
            "name": "json job",
            "rng_seeds": [99],
            "chains": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": "GG",
                    "unpaired_msa": ">query\nGG\n",
                    "paired_msa": "",
                }
            ],
        },
    )
    monomer = MonomericObject("protA", "ACDE")

    prepared_inputs = af3_backend_module.AlphaFold3Backend.prepare_input(
        objects_to_model=[
            {
                "object": [monomer, {"json_input": str(json_path)}],
                "output_dir": str(tmp_path),
            }
        ],
        random_seed=7,
    )

    assert len(prepared_inputs) == 1
    fold_input, (output_dir, resolve_msa_overlaps) = next(
        iter(prepared_inputs[0].items())
    )
    assert output_dir == str(tmp_path)
    assert resolve_msa_overlaps is True
    assert [chain.id for chain in fold_input.chains] == ["A", "B"]
    assert fold_input.chains[0].sequence == "ACDE"
    assert fold_input.chains[1].sequence == "GG"
    assert fold_input.chains[1].paired_msa == ">query\nGG\n"
    assert fold_input.chains[1].unpaired_msa == ""


def test_prepare_input_rejects_regions_for_multi_chain_json_inputs(
    af3_backend_module,
    tmp_path,
):
    json_path = _write_stub_af3_json(
        tmp_path / "multi_chain.json",
        {
            "name": "multi job",
            "chains": [
                {"type": "protein", "id": "A", "sequence": "AAAA"},
                {"type": "protein", "id": "B", "sequence": "BBBB"},
            ],
        },
    )

    with pytest.raises(ValueError, match="exactly one chain per file"):
        af3_backend_module.AlphaFold3Backend.prepare_input(
            objects_to_model=[
                {
                    "object": {"json_input": str(json_path), "regions": [(1, 2)]},
                    "output_dir": str(tmp_path),
                }
            ],
            random_seed=11,
        )


def test_prepare_input_slices_single_chain_json_regions_and_promotes_msa(
    af3_backend_module,
    tmp_path,
):
    json_path = _write_stub_af3_json(
        tmp_path / "single_chain.json",
        {
            "name": "single job",
            "chains": [
                {
                    "type": "protein",
                    "id": "Q",
                    "sequence": "ABCDEFGH",
                    "residue_ids": [10, 11, 12, 13, 14, 15, 16, 17],
                    "ptms": [["phospho", 2], ["methyl", 6]],
                    "unpaired_msa": ">query\nABCDEFGH\n",
                    "paired_msa": "",
                    "templates": [
                        {
                            "mmcif": "data_valid",
                            "query_to_template_map": {"0": 0, "1": 1, "4": 2, "5": 3},
                        }
                    ],
                }
            ],
        },
    )

    prepared_inputs = af3_backend_module.AlphaFold3Backend.prepare_input(
        objects_to_model=[
            {
                "object": {
                    "json_input": str(json_path),
                    "regions": [(1, 2), (5, 6)],
                },
                "output_dir": str(tmp_path),
            }
        ],
        random_seed=13,
    )

    fold_input, (_output_dir, resolve_msa_overlaps) = next(iter(prepared_inputs[0].items()))
    chain = fold_input.chains[0]

    assert resolve_msa_overlaps is True
    assert chain.id == "Q"
    assert chain.sequence == "ABEF"
    assert chain.residue_ids == [10, 11, 14, 15]
    assert chain.ptms == [("phospho", 2), ("methyl", 4)]
    assert chain.paired_msa == ">query\nABEF\n"
    assert chain.unpaired_msa == ""
    assert len(chain.templates) == 1
    assert chain.templates[0].query_to_template_map == {0: 0, 1: 1, 2: 2, 3: 3}


def test_prepare_input_drops_invalid_json_templates_but_keeps_valid_ones(
    af3_backend_module,
    tmp_path,
):
    json_path = _write_stub_af3_json(
        tmp_path / "templates.json",
        {
            "name": "template job",
            "chains": [
                {
                    "type": "protein",
                    "id": "A",
                    "sequence": "ACDE",
                    "templates": [
                        {"mmcif": "data_valid", "query_to_template_map": {"0": 0}},
                        {"mmcif": "INVALID", "query_to_template_map": {"1": 1}},
                    ],
                }
            ],
        },
    )

    prepared_inputs = af3_backend_module.AlphaFold3Backend.prepare_input(
        objects_to_model=[
            {
                "object": {"json_input": str(json_path)},
                "output_dir": str(tmp_path),
            }
        ],
        random_seed=17,
    )

    fold_input, _ = next(iter(prepared_inputs[0].items()))
    chain = fold_input.chains[0]
    assert len(chain.templates) == 1
    assert chain.templates[0].mmcif == "data_valid"
    assert chain.templates[0].query_to_template_map == {0: 0}


def test_prepare_input_normalises_adjacent_duplicate_residues_for_monomers(
    af3_backend_module,
    monkeypatch,
    tmp_path,
):
    monomer = MonomericObject("protA", "ABCD")
    monomer.feature_dict = {
        "residue_index": np.asarray([0, 0, 1, 2], dtype=np.int32),
        "msa": np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
        "deletion_matrix_int": np.asarray(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32
        ),
        "sequence": np.asarray([b"ABCD"], dtype=object),
        "seq_length": np.full(4, 4, dtype=np.int32),
        "num_alignments": np.full(4, 2, dtype=np.int32),
    }
    captured = {}

    def _fake_msa_rows_and_deletions_to_a3m(
        *,
        msa_rows,
        deletion_rows,
        query_sequence,
    ):
        captured["msa_rows"] = np.asarray(msa_rows)
        captured["deletion_rows"] = np.asarray(deletion_rows)
        captured["query_sequence"] = query_sequence
        return ">query\nACD\n"

    monkeypatch.setattr(
        af3_backend_module,
        "msa_rows_and_deletions_to_a3m",
        _fake_msa_rows_and_deletions_to_a3m,
    )

    prepared_inputs = af3_backend_module.AlphaFold3Backend.prepare_input(
        objects_to_model=[
            {
                "object": monomer,
                "output_dir": str(tmp_path),
            }
        ],
        random_seed=19,
    )

    fold_input, (_output_dir, resolve_msa_overlaps) = next(iter(prepared_inputs[0].items()))
    chain = fold_input.chains[0]

    assert resolve_msa_overlaps is True
    assert chain.sequence == "ACD"
    assert chain.residue_ids == [1, 2, 3]
    assert chain.paired_msa == ">query\nACD\n"
    assert chain.unpaired_msa == ""
    assert captured["query_sequence"] == "ACD"
    np.testing.assert_array_equal(
        captured["msa_rows"],
        np.asarray([[1, 3, 4], [5, 7, 8]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        captured["deletion_rows"],
        np.asarray([[0, 2, 3], [4, 6, 7]], dtype=np.int32),
    )
