import numpy as np

from alphafold.data import msa_pairing
from alphafold.data import parsers
from alphafold.data import pipeline
from alphapulldown.utils import mmseqs_species_identifiers


def _feature_dict_from_a3m(
    sequence: str,
    a3m: str,
    *,
    species_resolver,
) -> dict[str, np.ndarray]:
  feature_dict = {
      **pipeline.make_sequence_features(sequence, 'none', len(sequence)),
      **pipeline.make_msa_features([parsers.parse_a3m(a3m)]),
  }
  mmseqs_species_identifiers.enrich_mmseq_feature_dict_with_identifiers(
      feature_dict,
      a3m,
      species_resolver=species_resolver,
  )
  valid_feats = msa_pairing.MSA_FEATURES + (
      'msa_species_identifiers',
      'msa_uniprot_accession_identifiers',
  )
  feature_dict.update(
      {
          f'{key}_all_seq': value
          for key, value in feature_dict.items()
          if key in valid_feats
      }
  )
  return feature_dict


def test_make_msa_features_resolves_mmseqs_species_identifiers(monkeypatch):
  monkeypatch.setattr(
      mmseqs_species_identifiers,
      'resolve_species_ids_by_accession',
      lambda accessions, **_: {
          'A0A636IKY3': '108619',
          'UPI001118B830': '562',
      },
  )

  a3m = '\n'.join([
      '>101',
      'ACDE',
      '>UniRef100_A0A636IKY3\t136\t0.883',
      'ACDF',
      '>UniRef100_UPI001118B830\t855\t0.990',
      'AC-E',
      '',
  ])

  features = mmseqs_species_identifiers.build_mmseq_identifier_features(a3m)

  assert features['msa_species_identifiers'].tolist() == [
      b'',
      b'108619',
      b'562',
  ]
  assert features['msa_uniprot_accession_identifiers'].tolist() == [
      b'',
      b'A0A636IKY3',
      b'UPI001118B830',
  ]


def test_pair_sequences_works_with_mmseqs_accession_species_resolution(
    monkeypatch,
):
  monkeypatch.setattr(
      mmseqs_species_identifiers,
      'resolve_species_ids_by_accession',
      lambda accessions, **_: {
          'A0A636IKY3': '562',
          'A0A743YDY2': '573',
          'UPI001118B830': '562',
          'UPI00101273C6': '573',
      },
  )

  chain_a = _feature_dict_from_a3m(
      'ACDE',
      '\n'.join([
          '>101',
          'ACDE',
          '>UniRef100_A0A636IKY3\t136\t0.883',
          'ACDF',
          '>UniRef100_A0A743YDY2\t134\t0.932',
          'AC-E',
          '',
      ]),
      species_resolver=mmseqs_species_identifiers.resolve_species_ids_by_accession,
  )
  chain_b = _feature_dict_from_a3m(
      'WXYZ',
      '\n'.join([
          '>101',
          'WXYZ',
          '>UniRef100_UPI001118B830\t855\t0.990',
          'WXYW',
          '>UniRef100_UPI00101273C6\t833\t0.919',
          'WX-Z',
          '',
      ]),
      species_resolver=mmseqs_species_identifiers.resolve_species_ids_by_accession,
  )

  paired_rows = msa_pairing.pair_sequences([chain_a, chain_b])[2]

  assert paired_rows.shape == (3, 2)
  assert tuple(paired_rows[0]) == (0, 0)
  assert {tuple(row) for row in paired_rows[1:]} == {(1, 1), (2, 2)}
