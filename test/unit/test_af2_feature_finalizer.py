"""AlphaFold 2 features from a local MMseqs2 bundle, through AlphaFold 2's own code.

Every property asserted here is one that fails silently when wrong: a feature key
another source lacks crashes multimer pairing only when that chain happens to come
first; a template profile built from the merged alignment returns different hits
without complaint; a species lost from a header simply pairs nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

pipeline = pytest.importorskip(
    "alphafold.data.pipeline", reason="needs the AlphaFold 2 data pipeline"
)
from alphafold.common import residue_constants  # noqa: E402

from alphapulldown.af2_feature_finalizer import (  # noqa: E402
    Af2FeatureFinalizationSettings,
    Af2FeatureFinalizer,
)
from alphapulldown.feature_batch import PROTEIN, RNA, FeatureRequest  # noqa: E402


FIXTURES = Path(__file__).resolve().parents[1] / "test_data" / "features" / "af2_features"

# Measured on a natively generated (jackhmmer) pickle from the shared feature
# store. Local features must carry every one of these, or a multimer mixing the
# two sources fails on the missing key.
NATIVE_KEYS = {
    "aatype", "between_segment_residues", "deletion_matrix_int",
    "deletion_matrix_int_all_seq", "domain_name", "msa", "msa_all_seq",
    "msa_species_identifiers", "msa_species_identifiers_all_seq", "num_alignments",
    "residue_index", "seq_length", "sequence", "template_aatype",
    "template_all_atom_masks", "template_all_atom_positions",
    "template_confidence_scores", "template_domain_names", "template_release_date",
    "template_sequence", "template_sum_probs",
}
ACCESSION_KEYS = {
    "msa_uniprot_accession_identifiers",
    "msa_uniprot_accession_identifiers_all_seq",
}

QUERY = "MKTAYIAKQRQISFVKSHFSRQ"


class RecordingSearcher:
    """Stands in for hmmsearch and remembers the profile it was handed."""

    input_format = "sto"
    output_format = "sto"

    def __init__(self):
        self.queries: list[str] = []

    def query(self, text):
        self.queries.append(text)
        return "RAW_HITS"

    def get_template_hits(self, output_string, input_sequence):
        return []


class NoHitFeaturizer:
    """What AlphaFold 2's HmmsearchHitFeaturizer returns when nothing is found."""

    def get_templates(self, query_sequence, hits):
        num_res = len(query_sequence)

        class Result:
            features = {
                "template_aatype": np.zeros(
                    (1, num_res, len(residue_constants.restypes_with_x_and_gap)),
                    np.float32,
                ),
                "template_all_atom_masks": np.zeros(
                    (1, num_res, residue_constants.atom_type_num), np.float32
                ),
                "template_all_atom_positions": np.zeros(
                    (1, num_res, residue_constants.atom_type_num, 3), np.float32
                ),
                "template_domain_names": np.array([b""], dtype=object),
                "template_sequence": np.array([b""], dtype=object),
                "template_sum_probs": np.array([0], dtype=np.float32),
            }

        return Result()


def _a3m(records):
    return "".join(f">{description}\n{sequence}\n" for description, sequence in records)


def _write_bundle(msa_dir: Path, name: str, sequence: str, *, uniref90, mgnify,
                  small_bfd, paired) -> None:
    msa_dir.mkdir(parents=True, exist_ok=True)
    unpaired = [("query", sequence), *uniref90, *mgnify, *small_bfd]
    paired_records = [("query", sequence), *paired]
    (msa_dir / f"{name}_mmseqs_msa.json").write_text(
        json.dumps(
            {
                "schemaVersion": 3,
                "name": name,
                "sequence": sequence,
                "unpairedMsa": _a3m(unpaired),
                "pairedMsa": _a3m(paired_records),
                "unpairedDepth": len(unpaired),
                "pairedDepth": len(paired_records),
                "unpairedDatabaseRows": [
                    {"name": "uniref90", "rows": len(uniref90)},
                    {"name": "mgnify", "rows": len(mgnify)},
                    {"name": "small_bfd", "rows": len(small_bfd)},
                ],
                "provenance": {"schema_version": 5, "fixture": name},
            }
        ),
        encoding="utf-8",
    )


def _mutate(sequence: str, position: int, residue: str) -> str:
    return sequence[:position] + residue + sequence[position + 1 :]


def _insert(sequence: str, position: int, insertion: str) -> str:
    return sequence[:position] + insertion + sequence[position:]


def _standard_bundle(msa_dir: Path, name: str = "alpha", sequence: str = QUERY):
    """Rows named by database, one uniref90 hit with an insertion, UniProt species."""
    _write_bundle(
        msa_dir,
        name,
        sequence,
        uniref90=[
            ("UniRef90_U1 uniref hit", _mutate(sequence, 3, "W")),
            # A three-residue insertion before position 5 -- lowercase, as stitched.
            # It also carries its own substitution: a row whose match columns equal
            # the query's is a duplicate of it, and AlphaFold 2 drops it as one.
            ("UniRef90_U2 inserted", _insert(_mutate(sequence, 11, "W"), 5, "ggg")),
        ],
        mgnify=[("MGYP000000001", _mutate(sequence, 7, "W"))],
        small_bfd=[("BFD_B1", _mutate(sequence, 9, "W"))],
        paired=[
            ("sp|P12345|KIN1_HUMAN kinase OS=Homo sapiens OX=9606", _mutate(sequence, 2, "V")),
            ("tr|Q67890|Q67890_MOUSE kinase OS=Mus musculus OX=10090", _mutate(sequence, 4, "V")),
        ],
    )


def _finalizer(tmp_path: Path, **overrides):
    settings = dict(
        output_dir=tmp_path / "features",
        msa_input_dir=tmp_path / "msas",
        max_template_date="2050-01-01",
        template_seqres_database_id="pdb-seqres-2050",
        template_mmcif_database_id="mmcif-2050",
    )
    settings.update(overrides)
    searcher = RecordingSearcher()
    finalizer = Af2FeatureFinalizer(
        settings=Af2FeatureFinalizationSettings(**settings),
        template_searcher=searcher,
        template_featurizer=NoHitFeaturizer(),
    )
    return finalizer, searcher


def _features(tmp_path: Path, name: str = "alpha") -> dict:
    with open(tmp_path / "features" / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle).feature_dict


def _generate(tmp_path, name="alpha", sequence=QUERY, **overrides):
    finalizer, searcher = _finalizer(tmp_path, **overrides)
    result = finalizer.generate([FeatureRequest(name=name, sequence=sequence)])
    assert result.failures == (), result.failures
    return result, searcher


def test_features_carry_the_native_key_set_plus_accession_identifiers(tmp_path):
    _standard_bundle(tmp_path / "msas")
    _generate(tmp_path)
    assert set(_features(tmp_path)) == NATIVE_KEYS | ACCESSION_KEYS


def test_insertions_reach_the_deletion_matrix(tmp_path):
    """The point of recovering insertions: native features have them, and until
    now every local MMseqs2 alignment produced an all-zero deletion matrix."""
    _standard_bundle(tmp_path / "msas")
    _generate(tmp_path)
    features = _features(tmp_path)

    deletions = features["deletion_matrix_int"]
    assert deletions.shape == features["msa"].shape
    assert deletions.sum() == 3, "three inserted residues, counted once"
    # Recorded against the residue that follows the insertion.
    row = int(np.nonzero(deletions.sum(axis=1))[0][0])
    assert deletions[row, 5] == 3


def test_rows_follow_alphafold2s_merge_order(tmp_path):
    _standard_bundle(tmp_path / "msas")
    _generate(tmp_path)
    msa = _features(tmp_path)["msa"]
    mutated_at = [int(np.nonzero(row != msa[0])[0][0]) if (row != msa[0]).any() else None
                  for row in msa]
    # query, uniref90 (3, 11), then BFD (9) before MGnify (7), as AlphaFold 2 merges.
    assert mutated_at == [None, 3, 11, 9, 7]


def test_templates_are_searched_from_uniref90_alone(tmp_path):
    _standard_bundle(tmp_path / "msas")
    _, searcher = _generate(tmp_path)
    [profile] = searcher.queries
    assert "UniRef90_U1" in profile
    assert "MGYP000000001" not in profile and "BFD_B1" not in profile


def test_pairing_features_come_from_uniprot_with_parsed_species(tmp_path):
    _standard_bundle(tmp_path / "msas")
    _generate(tmp_path)
    features = _features(tmp_path)

    assert list(features["msa_species_identifiers_all_seq"]) == [b"", b"HUMAN", b"MOUSE"]
    assert list(features["msa_uniprot_accession_identifiers_all_seq"]) == [
        b"", b"P12345", b"Q67890",
    ]
    # Not a copy of the unpaired features, which is what the remote path uses.
    assert features["msa_all_seq"].shape[0] == 3
    assert features["msa"].shape[0] == 5


def test_no_network_is_ever_touched(tmp_path, monkeypatch):
    """A batch of thousands of chains must not start calling UniProt REST."""
    import urllib.request

    def refuse(*args, **kwargs):
        raise AssertionError("the local AlphaFold 2 path queried the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    _write_bundle(
        tmp_path / "msas",
        "alpha",
        QUERY,
        uniref90=[("UniRef90_A0A0A0A0A0 no species in this header", _mutate(QUERY, 3, "W"))],
        mgnify=[],
        small_bfd=[],
        # A bare accession: exactly the case the shared helper would look up.
        paired=[("P12345", _mutate(QUERY, 2, "V"))],
    )
    _generate(tmp_path)


def test_published_pickle_is_a_monomeric_object_and_is_reused(tmp_path):
    from alphapulldown.objects import MonomericObject

    _standard_bundle(tmp_path / "msas")
    first, searcher = _generate(tmp_path)
    assert [artifact.name for artifact in first.written] == ["alpha"]
    with open(tmp_path / "features" / "alpha.pkl", "rb") as handle:
        monomer = pickle.load(handle)
    assert isinstance(monomer, MonomericObject)
    assert monomer.sequence == QUERY and monomer.skip_msa is False
    assert list((tmp_path / "features").glob("alpha_feature_metadata_*.json"))

    again, searcher_again = _generate(tmp_path)
    assert [artifact.name for artifact in again.reused] == ["alpha"]
    assert searcher_again.queries == [], "a reused artifact must not search again"


def test_changed_template_settings_are_not_served_from_cache(tmp_path):
    _standard_bundle(tmp_path / "msas")
    _generate(tmp_path)
    moved, _ = _generate(tmp_path, max_template_date="2020-01-01")
    assert [artifact.name for artifact in moved.written] == ["alpha"]


def test_compressed_output_round_trips(tmp_path):
    import lzma

    _standard_bundle(tmp_path / "msas")
    _generate(tmp_path, compress=True)
    with lzma.open(tmp_path / "features" / "alpha.pkl.xz", "rb") as handle:
        assert pickle.load(handle).sequence == QUERY


def test_rna_is_refused_by_name(tmp_path):
    finalizer, _ = _finalizer(tmp_path)
    result = finalizer.generate(
        [FeatureRequest(name="trna", sequence="ACGU", molecule_type=RNA)]
    )
    assert [failure.name for failure in result.failures] == ["trna"]
    assert "protein" in result.failures[0].error


def _local_monomer(tmp_path: Path, name: str, sequence: str):
    _standard_bundle(tmp_path / "msas", name, sequence)
    _generate(tmp_path, name=name, sequence=sequence)
    with open(tmp_path / "features" / f"{name}.pkl", "rb") as handle:
        return pickle.load(handle)


def _native_like_monomer():
    """An older pickle with 21 keys and no accession identifiers at all."""
    with open(FIXTURES / "protein" / "P61626.pkl", "rb") as handle:
        return pickle.load(handle)


@pytest.mark.parametrize("local_first", (True, False))
def test_local_chain_pairs_with_a_pickle_from_another_source(tmp_path, local_first):
    """AlphaFold 2 pairing takes its key set from the FIRST chain and indexes every
    other chain with it, so an extra key crashes only in one order (issue #619).
    Both orders, through the real pairing code."""
    from alphapulldown.objects import MultimericObject

    local = _local_monomer(tmp_path, "alpha", QUERY)
    native = _native_like_monomer()
    chains = [local, native] if local_first else [native, local]

    merged = MultimericObject(interactors=chains, pair_msa=True).feature_dict

    total = len(local.sequence) + len(native.sequence)
    assert merged["aatype"].shape == (total,)
    assert merged["msa"].shape[1] == total


def test_two_local_chains_pair_the_species_they_share(tmp_path):
    """Which rows pair, not merely that pairing ran: HUMAN and MOUSE occur in both
    chains' UniProt alignments, so those rows line up in the paired block."""
    from alphapulldown.objects import MultimericObject

    first = _local_monomer(tmp_path, "alpha", QUERY)
    second_sequence = _mutate(QUERY, 12, "L")
    second = _local_monomer(tmp_path, "beta", second_sequence)

    merged = MultimericObject(interactors=[first, second], pair_msa=True).feature_dict

    # The merged multimer MSA is in the model's residue order, not HHblits'.
    to_id = residue_constants.restype_order_with_x
    human_row = [to_id[r] for r in _mutate(QUERY, 2, "V") + _mutate(second_sequence, 2, "V")]
    mouse_row = [to_id[r] for r in _mutate(QUERY, 4, "V") + _mutate(second_sequence, 4, "V")]
    # An unpaired row carries one chain's residues and gaps across the other, so
    # both chains' hits can only sit on one row if pairing put them there.
    rows = [list(row) for row in merged["msa"]]
    assert human_row in rows, "the HUMAN hits of both chains must share one row"
    assert mouse_row in rows, "the MOUSE hits of both chains must share one row"
    # And a hit present in only one chain's alignment is never paired.
    unpaired_hit_a = [to_id[r] for r in _mutate(QUERY, 3, "W") + second_sequence]
    assert unpaired_hit_a not in rows


def test_a_homomer_of_a_local_chain_assembles(tmp_path):
    """Two copies of one chain: AF2 merges identical entities into a dense MSA
    instead of pairing them, a different code path from the heteromer."""
    from alphapulldown.objects import MultimericObject

    first = _local_monomer(tmp_path, "alpha", QUERY)
    with open(tmp_path / "features" / "alpha.pkl", "rb") as handle:
        second = pickle.load(handle)

    merged = MultimericObject(interactors=[first, second], pair_msa=True).feature_dict

    assert merged["aatype"].shape == (2 * len(QUERY),)
    assert merged["msa"].shape[1] == 2 * len(QUERY)


def test_local_chains_assemble_without_pairing(tmp_path):
    """pair_msa=False skips species pairing and block-diagonalises the MSAs."""
    from alphapulldown.objects import MultimericObject

    first = _local_monomer(tmp_path, "alpha", QUERY)
    second = _local_monomer(tmp_path, "beta", _mutate(QUERY, 12, "L"))

    merged = MultimericObject(interactors=[first, second], pair_msa=False).feature_dict

    to_id = residue_constants.restype_order_with_x
    # Unpaired, a HUMAN hit shares a row with no other chain's residues.
    human_row = [to_id[r] for r in _mutate(QUERY, 2, "V") + _mutate(QUERY, 2, "V")]
    assert human_row not in [list(row) for row in merged["msa"]]
    assert merged["msa"].shape[1] == 2 * len(QUERY)


def test_a_chopped_local_chain_pairs_with_a_full_one(tmp_path):
    """Issue #619's shape: a full chain first, a chopped chain second. The chopped
    chain must keep the accession identifiers, row-aligned, or pairing crashes."""
    from alphapulldown.objects import ChoppedObject, MultimericObject

    full = _local_monomer(tmp_path, "alpha", QUERY)
    source = _local_monomer(tmp_path, "beta", _mutate(QUERY, 12, "L"))
    region = (3, 15)
    chopped = ChoppedObject(
        source.description, source.sequence, source.feature_dict, [region]
    )
    chopped.prepare_final_sliced_feature_dict()
    assert (
        chopped.feature_dict["msa_uniprot_accession_identifiers_all_seq"].shape[0]
        == chopped.feature_dict["msa_all_seq"].shape[0]
    )

    merged = MultimericObject(interactors=[full, chopped], pair_msa=True).feature_dict

    region_length = region[1] - region[0] + 1
    assert merged["aatype"].shape == (len(QUERY) + region_length,)


@pytest.mark.parametrize("damage", ("spans_do_not_add_up", "no_spans"))
def test_a_bundle_unusable_to_alphafold2_is_deleted_so_it_is_rebuilt(tmp_path, damage):
    """Rejecting such a bundle and leaving it in place fails the same finalization on
    every retry: its Shard completion still validates, so no repair is scheduled.
    Deleting it is what gets it rebuilt."""
    _standard_bundle(tmp_path / "msas")
    bundle_path = tmp_path / "msas" / "alpha_mmseqs_msa.json"
    bundle = json.loads(bundle_path.read_text())
    if damage == "no_spans":
        del bundle["unpairedDatabaseRows"]
    else:
        bundle["unpairedDatabaseRows"][0]["rows"] += 1
    bundle_path.write_text(json.dumps(bundle))

    finalizer, _ = _finalizer(tmp_path)
    result = finalizer.generate([FeatureRequest(name="alpha", sequence=QUERY)])

    assert [failure.name for failure in result.failures] == ["alpha"]
    assert "row spans" in result.failures[0].error
    assert not bundle_path.exists()
