"""CLI wiring for ``--keep_msas``: rewrite existing features, templates only.

The mode exists because template search is minutes and MSA search is hours, so
when a template database is refreshed underneath a set of features the MSAs are
still good. These tests pin the behaviour that makes that safe: MSAs must come
through untouched, stale templates must be gone, and anything that cannot be
updated in place must fall back to normal generation rather than half-update.
"""

import json
import lzma
import pickle
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from absl import flags

import alphapulldown.scripts.create_individual_features as create_features
from alphapulldown.objects import MonomericObject
from alphapulldown.utils.template_reuse import (
    SOURCE_RECONSTRUCTED,
    SOURCE_UNIREF90_FILE,
)

FLAGS = flags.FLAGS

REAL_STO = (
    "# STOCKHOLM 1.0\n\n"
    "query    ACDE\n"
    "hit1     ACDF\n"
    "#=GC RF  xxxx\n"
    "//\n"
)

OLD_TEMPLATES = {
    "template_aatype": np.zeros((1, 4, 22), dtype=np.float32),
    "template_domain_names": np.asarray([b"stale_2022"], dtype=object),
    "template_sequence": np.asarray([b"OLD"], dtype=object),
    "template_sum_probs": np.asarray([0.5], dtype=np.float32),
}

NEW_TEMPLATES = {
    "template_aatype": np.ones((2, 4, 22), dtype=np.float32),
    "template_domain_names": np.asarray([b"fresh_A", b"fresh_B"], dtype=object),
    "template_sequence": np.asarray([b"NEWA", b"NEWB"], dtype=object),
    "template_sum_probs": np.asarray([0.9, 0.8], dtype=np.float32),
}

MSA_KEYS = ("msa", "deletion_matrix_int", "num_alignments", "msa_species_identifiers")


def _make_monomer(sequence="ACDE"):
    monomer = MonomericObject("proteinA", sequence)
    monomer.feature_dict = {
        "msa": np.asarray([[0, 1, 2, 3], [0, 1, 2, 2]], dtype=np.int32),
        "deletion_matrix_int": np.zeros((2, 4), dtype=np.int32),
        "num_alignments": np.asarray([2, 2, 2, 2], dtype=np.int32),
        "msa_species_identifiers": np.asarray([b"", b"9606"], dtype=object),
        **OLD_TEMPLATES,
    }
    return monomer


@pytest.fixture
def flagged(tmp_path):
    """A flag state with --keep_msas on and an output dir to work in."""
    FLAGS(["test"])
    FLAGS.output_dir = str(tmp_path)
    FLAGS.keep_msas = True
    FLAGS.use_mmseqs2 = False
    FLAGS.skip_msa = False
    FLAGS.skip_existing = False
    FLAGS.compress_features = False
    yield tmp_path
    FLAGS.keep_msas = False


def _write_monomer(output_dir, monomer, *, compressed=False):
    path = Path(output_dir) / (
        f"{monomer.description}.pkl.xz" if compressed else f"{monomer.description}.pkl"
    )
    opener = lzma.open if compressed else open
    with opener(path, "wb") as handle:
        pickle.dump(monomer, handle)
    return path


def _fake_pipeline():
    return types.SimpleNamespace(
        template_searcher=object(), template_featurizer=object()
    )


def _load_written(output_dir, name="proteinA"):
    with open(Path(output_dir) / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


# ------------------------------------------------------------------- AF2

@pytest.mark.parametrize("compressed", [False, True])
def test_keep_msas_replaces_templates_and_preserves_msas(flagged, compressed):
    original = _make_monomer()
    _write_monomer(flagged, original, compressed=compressed)
    (flagged / "proteinA").mkdir()
    (flagged / "proteinA" / "uniref90_hits.sto").write_text(REAL_STO)
    FLAGS.compress_features = compressed

    with patch.object(
        create_features, "search_templates", return_value=NEW_TEMPLATES
    ) as mock_search, patch(
        "alphapulldown.utils.save_meta_data.get_meta_dict", return_value={"s": "t"}
    ):
        updated = create_features._update_templates_keeping_msas(
            "proteinA", "ACDE", _fake_pipeline()
        )

    assert updated is True
    mock_search.assert_called_once()
    # The alignment on disk is what AF2 itself searches with, so it wins.
    assert mock_search.call_args.kwargs["stockholm_msa"] == REAL_STO

    if compressed:
        with lzma.open(flagged / "proteinA.pkl.xz", "rb") as handle:
            written = pickle.load(handle)
    else:
        written = _load_written(flagged)

    assert written.feature_dict["template_domain_names"].tolist() == [
        b"fresh_A", b"fresh_B",
    ]
    assert b"stale_2022" not in written.feature_dict["template_domain_names"].tolist()
    for key in MSA_KEYS:
        np.testing.assert_array_equal(
            written.feature_dict[key], original.feature_dict[key]
        )
    assert written.template_msa_source == SOURCE_UNIREF90_FILE


def test_keep_msas_reconstructs_the_alignment_when_msa_files_were_deleted(flagged):
    """--save_msa_files=False removes alignments once features are written."""
    _write_monomer(flagged, _make_monomer())

    with patch.object(
        create_features, "search_templates", return_value=NEW_TEMPLATES
    ) as mock_search, patch(
        "alphapulldown.utils.save_meta_data.get_meta_dict", return_value={}
    ):
        assert create_features._update_templates_keeping_msas(
            "proteinA", "ACDE", _fake_pipeline()
        )

    # Rebuilt from the stored features, and labelled as such: the merged MSA is
    # not the uniref90 alignment, so hits may legitimately differ.
    assert "# STOCKHOLM" in mock_search.call_args.kwargs["stockholm_msa"]
    assert _load_written(flagged).template_msa_source == SOURCE_RECONSTRUCTED


def test_keep_msas_survives_a_search_that_finds_no_templates(flagged):
    """A restrictive --max_template_date can legitimately match nothing.

    The features must stay usable: stale templates gone, MSAs intact, and the
    template_* block well-formed rather than absent.
    """
    original = _make_monomer()
    _write_monomer(flagged, original)

    with patch.object(create_features, "search_templates", return_value={}), \
         patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={}):
        assert create_features._update_templates_keeping_msas(
            "proteinA", "ACDE", _fake_pipeline()
        )

    written = _load_written(flagged)
    assert "stale_2022" not in str(written.feature_dict.get("template_domain_names", ""))
    for key in MSA_KEYS:
        np.testing.assert_array_equal(
            written.feature_dict[key], original.feature_dict[key]
        )
    # Downstream code indexes these unconditionally.
    assert "template_sum_probs" in written.feature_dict
    assert "template_confidence_scores" in written.feature_dict
    assert "template_release_date" in written.feature_dict


def test_keep_msas_declines_when_no_features_exist(flagged):
    assert create_features._update_templates_keeping_msas(
        "proteinA", "ACDE", _fake_pipeline()
    ) is False


def test_keep_msas_declines_on_a_sequence_mismatch(flagged):
    """Same name, different protein: updating in place would corrupt it."""
    _write_monomer(flagged, _make_monomer(sequence="ACDE"))

    with patch.object(create_features, "search_templates") as mock_search:
        result = create_features._update_templates_keeping_msas(
            "proteinA", "WWWW", _fake_pipeline()
        )

    assert result is False
    mock_search.assert_not_called()


def test_keep_msas_requires_a_template_stack(flagged):
    _write_monomer(flagged, _make_monomer())
    empty_pipeline = types.SimpleNamespace(
        template_searcher=None, template_featurizer=None
    )
    with pytest.raises(RuntimeError, match="template searcher"):
        create_features._update_templates_keeping_msas(
            "proteinA", "ACDE", empty_pipeline
        )


def test_create_individual_features_updates_instead_of_regenerating(flagged):
    """The AF2 loop must not run MSA generation when an update succeeded."""
    _write_monomer(flagged, _make_monomer())
    fasta = flagged / "in.fasta"
    fasta.write_text(">proteinA\nACDE\n")
    FLAGS.fasta_paths = [str(fasta)]

    with patch.object(create_features, "create_arguments"), \
         patch.object(create_features, "create_pipeline_af2",
                      return_value=_fake_pipeline()), \
         patch.object(create_features, "create_uniprot_runner") as mock_runner, \
         patch.object(create_features, "create_and_save_monomer_objects") as mock_make, \
         patch.object(create_features, "search_templates",
                      return_value=NEW_TEMPLATES), \
         patch("alphapulldown.utils.save_meta_data.get_meta_dict", return_value={}):
        create_features.create_individual_features()

    mock_make.assert_not_called()
    assert mock_runner.called, "uniprot runner is still built for any misses"


def test_create_individual_features_falls_through_for_new_proteins(flagged):
    """A protein with no stored features has nothing to keep; generate it."""
    fasta = flagged / "in.fasta"
    fasta.write_text(">brand_new\nACDE\n")
    FLAGS.fasta_paths = [str(fasta)]

    with patch.object(create_features, "create_arguments"), \
         patch.object(create_features, "create_pipeline_af2",
                      return_value=_fake_pipeline()), \
         patch.object(create_features, "create_uniprot_runner"), \
         patch.object(create_features, "create_and_save_monomer_objects") as mock_make:
        create_features.create_individual_features()

    mock_make.assert_called_once()


# ------------------------------------------------------- flag combinations

@pytest.mark.parametrize(
    ("conflicting", "message"),
    [("use_mmseqs2", "MMseqs2"), ("skip_msa", "skip_msa")],
)
def test_keep_msas_rejects_incompatible_flags(conflicting, message):
    FLAGS(["test"])
    FLAGS.keep_msas = True
    FLAGS.data_pipeline = "alphafold2"
    setattr(FLAGS, conflicting, True)
    try:
        with pytest.raises(ValueError, match=message):
            create_features.validate_data_pipeline_flags()
    finally:
        FLAGS.keep_msas = False
        setattr(FLAGS, conflicting, False)


# ------------------------------------------------------------------- AF3
#
# AlphaFold3 is optional and its compiled parts are absent in CI, so the AF3
# API is stubbed the way the rest of this suite stubs it. What is under test is
# AlphaPulldown's own logic -- which fields are carried over and which are
# cleared -- not AF3's JSON parser.


class _StubProteinChain:
    def __init__(self, id, sequence, ptms=(), residue_ids=None, description=None,
                 paired_msa=None, unpaired_msa=None, templates=None):
        self.id = id
        self.sequence = sequence
        self.ptms = list(ptms)
        self.residue_ids = residue_ids
        self.description = description
        self.paired_msa = paired_msa
        self.unpaired_msa = unpaired_msa
        self.templates = templates


class _StubRnaChain:
    def __init__(self, id, sequence, unpaired_msa=None):
        self.id = id
        self.sequence = sequence
        self.unpaired_msa = unpaired_msa


class _StubInput:
    def __init__(self, name, chains, rng_seeds):
        self.name = name
        self.chains = list(chains)
        self.rng_seeds = list(rng_seeds)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if "sequences" not in data or "name" not in data:
            raise ValueError("not an AF3 input")
        chains = []
        for entry in data["sequences"]:
            protein = entry["protein"]
            chains.append(_StubProteinChain(
                id=protein["id"],
                sequence=protein["sequence"],
                description=protein.get("description"),
                paired_msa=protein.get("pairedMsa"),
                unpaired_msa=protein.get("unpairedMsa"),
                templates=protein.get("templates"),
            ))
        return cls(data["name"], chains, data.get("modelSeeds", [42]))


def _stub_folding_input():
    module = types.SimpleNamespace(
        ProteinChain=_StubProteinChain,
        RnaChain=_StubRnaChain,
        Input=_StubInput,
    )
    return module


def _af3_json(*, name="proteinA", unpaired=">q\nACDE\n", paired=">q\nACDE\n",
              templates=None):
    protein = {"id": "A", "sequence": "ACDE", "description": name}
    if unpaired is not None:
        protein["unpairedMsa"] = unpaired
    if paired is not None:
        protein["pairedMsa"] = paired
    if templates is not None:
        protein["templates"] = templates
    return json.dumps(
        {"name": name, "modelSeeds": [42], "sequences": [{"protein": protein}]}
    )


@pytest.fixture
def af3_stub():
    with patch.object(create_features, "folding_input", _stub_folding_input()):
        yield


def test_af3_chain_without_templates_keeps_msas_and_clears_templates(af3_stub):
    """AF3 searches templates only when both MSAs are set and templates is None.

    templates=[] means "no templates, do not search"; a partially populated
    chain is rejected by AF3 outright. So None is the specific signal needed.
    """
    chain = _StubProteinChain(
        id="A", sequence="ACDE",
        unpaired_msa=">q\nACDE\n", paired_msa=">q\nACDE\n",
        templates=[{"mmcif": "data_x"}],
    )

    stripped = create_features._af3_chain_without_templates(chain)

    assert stripped.templates is None, "None means 'search'; [] means 'do not'"
    assert stripped.unpaired_msa == ">q\nACDE\n"
    assert stripped.paired_msa == ">q\nACDE\n"
    assert stripped.sequence == "ACDE"
    assert stripped.id == "A"


def test_af3_chain_carries_over_description_and_residue_ids(af3_stub):
    """The description holds AlphaPulldown's metadata envelope; losing it would
    discard the feature provenance."""
    chain = _StubProteinChain(
        id="B", sequence="ACDE", description="prot\n__META__=1",
        residue_ids=[1, 2, 3, 4],
        unpaired_msa=">q\nACDE\n", paired_msa=">q\nACDE\n", templates=[],
    )

    stripped = create_features._af3_chain_without_templates(chain)

    assert stripped.description == "prot\n__META__=1"
    assert stripped.residue_ids == [1, 2, 3, 4]


def test_af3_chain_without_msas_is_left_alone(af3_stub):
    """Nothing to keep: let the normal path search MSAs and templates."""
    chain = _StubProteinChain(id="A", sequence="ACDE")
    assert create_features._af3_chain_without_templates(chain) is chain


def test_af3_non_protein_chains_pass_through(af3_stub):
    """RNA/DNA chains have no template search to re-run."""
    chain = _StubRnaChain(id="R", sequence="ACGU")
    assert create_features._af3_chain_without_templates(chain) is chain


def test_af3_input_keeping_msas_round_trips_existing_features(tmp_path, af3_stub):
    path = tmp_path / "proteinA_af3_input.json"
    path.write_text(_af3_json(templates=[{"mmcif": "data_stale"}]))

    rebuilt = create_features._af3_input_keeping_msas(str(path), "proteinA")

    assert rebuilt is not None
    assert rebuilt.name == "proteinA"
    (chain,) = rebuilt.chains
    assert chain.templates is None, "stale templates must be dropped and re-searched"
    assert chain.unpaired_msa == ">q\nACDE\n", "the expensive MSA must survive"
    assert chain.paired_msa == ">q\nACDE\n"


def test_af3_input_keeping_msas_declines_on_a_sequence_mismatch(tmp_path, af3_stub):
    """Same description, different protein: the AF2 path refuses, so must AF3.

    Features are matched to inputs by description alone. If a FASTA is edited
    but keeps its name, reusing the cached chain would re-run template search
    against the old sequence and then overwrite the file with features for the
    wrong protein.
    """
    path = tmp_path / "proteinA_af3_input.json"
    path.write_text(_af3_json())          # cached chain is "ACDE"

    assert create_features._af3_input_keeping_msas(
        str(path), "proteinA", "WWWWWW"
    ) is None


def test_af3_input_keeping_msas_accepts_a_matching_sequence(tmp_path, af3_stub):
    path = tmp_path / "proteinA_af3_input.json"
    path.write_text(_af3_json())

    rebuilt = create_features._af3_input_keeping_msas(
        str(path), "proteinA", "ACDE"
    )

    assert rebuilt is not None
    assert rebuilt.chains[0].unpaired_msa == ">q\nACDE\n"


def test_af3_input_keeping_msas_skips_the_check_without_a_sequence(tmp_path, af3_stub):
    """The sequence is optional so the helper stays usable on its own."""
    path = tmp_path / "proteinA_af3_input.json"
    path.write_text(_af3_json())
    assert create_features._af3_input_keeping_msas(str(path), "proteinA") is not None


def test_af3_loop_regenerates_rather_than_overwriting_a_renamed_protein(
    tmp_path, af3_stub, monkeypatch
):
    """End to end: an edited FASTA under an old name must not be overwritten."""
    FLAGS(["test"])
    FLAGS.output_dir = str(tmp_path)
    FLAGS.keep_msas = True
    FLAGS.skip_existing = False
    FLAGS.compress_features = False
    FLAGS.data_pipeline = "alphafold3"
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">proteinA\nWWWWWW\n")      # edited: no longer ACDE
    FLAGS.fasta_paths = [str(fasta)]
    (tmp_path / "proteinA_af3_input.json").write_text(_af3_json())

    seen = {}

    class _Pipeline:
        def process(self, input_obj):
            seen["chains"] = [c.sequence for c in input_obj.chains]
            seen["msa"] = [getattr(c, "unpaired_msa", None) for c in input_obj.chains]
            return {"sequences": []}

    with patch.object(create_features, "create_arguments"), \
         patch.object(create_features, "create_pipeline_af3", return_value=_Pipeline()), \
         patch.object(create_features, "validate_data_pipeline_flags"), \
         patch.object(create_features, "get_af3_feature_metadata", return_value={}):
        create_features.create_af3_individual_features()

    assert seen["chains"] == ["WWWWWW"], "must build a fresh chain from the FASTA"
    assert seen["msa"] == [None], "stale MSAs must not be carried over"
    FLAGS.keep_msas = False
    FLAGS.data_pipeline = "alphafold2"


def test_af3_input_keeping_msas_reads_a_compressed_file(tmp_path, af3_stub):
    """AF3 feature JSONs are published .xz; the reuse path must read them."""
    path = tmp_path / "proteinA_af3_input.json.xz"
    with lzma.open(path, "wt", encoding="utf-8") as handle:
        handle.write(_af3_json())

    rebuilt = create_features._af3_input_keeping_msas(str(path), "proteinA")

    assert rebuilt is not None
    assert rebuilt.chains[0].templates is None


def test_af3_input_keeping_msas_declines_a_malformed_file(tmp_path, af3_stub):
    """A partial or foreign JSON must fall back, not abort the whole run."""
    path = tmp_path / "broken_af3_input.json"
    path.write_text('{"not": "an af3 input"}')
    assert create_features._af3_input_keeping_msas(str(path), "broken") is None


def test_af3_input_keeping_msas_declines_when_there_are_no_msas(tmp_path, af3_stub):
    path = tmp_path / "proteinA_af3_input.json"
    path.write_text(_af3_json(unpaired=None, paired=None))
    assert create_features._af3_input_keeping_msas(str(path), "proteinA") is None
