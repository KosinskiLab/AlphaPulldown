from urllib import error

import numpy as np
import pytest

from alphafold.data import msa_pairing
from alphafold.data import parsers
from alphafold.data import pipeline
from alphapulldown.objects import MonomericObject
from alphapulldown.utils import mmseqs_species_identifiers


@pytest.fixture(autouse=True)
def clear_species_id_cache():
  mmseqs_species_identifiers._SPECIES_ID_CACHE.clear()
  yield
  mmseqs_species_identifiers._SPECIES_ID_CACHE.clear()


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


def test_make_mmseq_features_researches_templates_for_precomputed_msa(
    monkeypatch,
    tmp_path,
):
  import alphapulldown.objects as objects_mod

  a3m_path = tmp_path / 'dummy.a3m'
  a3m_text = '\n'.join([
      '# header line that should be ignored later',
      '>101',
      'ACDE',
      '',
  ])
  a3m_path.write_text(a3m_text, encoding='utf-8')

  calls = {}

  def fake_unserialize_msa(a3m_lines, sequence):
    calls['unserialize_msa'] = {
        'a3m_lines': a3m_lines,
        'sequence': sequence,
    }
    return (
        ['PRECOMPUTED_UNPAIRED'],
        ['PRECOMPUTED_PAIRED'],
        ['PRECOMPUTED_UNIQUE'],
        ['PRECOMPUTED_CARDINALITY'],
        ['PRECOMPUTED_TEMPLATE'],
    )

  def fake_get_msa_and_templates(**kwargs):
    calls['get_msa_and_templates'] = kwargs
    return (
        ['IGNORED_UNPAIRED'],
        ['IGNORED_PAIRED'],
        ['IGNORED_UNIQUE'],
        ['IGNORED_CARDINALITY'],
        ['TEMPLATE_FROM_RESEARCH'],
    )

  def fake_build_monomer_feature(sequence, msa, template_feature):
    calls['build_monomer_feature'] = {
        'sequence': sequence,
        'msa': msa,
        'template_feature': template_feature,
    }
    return {
        'template_confidence_scores': None,
        'template_release_date': None,
    }

  def fake_enrich(feature_dict, a3m, **_kwargs):
    calls['enrich_mmseq_feature_dict_with_identifiers'] = a3m
    feature_dict['msa_species_identifiers'] = np.asarray([b''])
    feature_dict['msa_uniprot_accession_identifiers'] = np.asarray([b''])

  monkeypatch.setattr(objects_mod, 'unserialize_msa', fake_unserialize_msa)
  monkeypatch.setattr(
      objects_mod,
      'get_msa_and_templates',
      fake_get_msa_and_templates,
  )
  monkeypatch.setattr(
      objects_mod,
      'build_monomer_feature',
      fake_build_monomer_feature,
  )
  monkeypatch.setattr(
      objects_mod,
      'enrich_mmseq_feature_dict_with_identifiers',
      fake_enrich,
  )

  monomer = MonomericObject('dummy', 'ACDE')
  monomer.make_mmseq_features(
      DEFAULT_API_SERVER='https://fake.server',
      output_dir=str(tmp_path),
      use_precomputed_msa=True,
      use_templates=True,
  )

  assert calls['unserialize_msa']['sequence'] == 'ACDE'
  assert calls['unserialize_msa']['a3m_lines'] == ['>101\nACDE']
  assert calls['get_msa_and_templates'] == {
      'jobname': 'dummy',
      'query_sequences': 'ACDE',
      'a3m_lines': False,
      'result_dir': tmp_path,
      'msa_mode': 'single_sequence',
      'use_templates': True,
      'custom_template_path': None,
      'pair_mode': 'none',
      'host_url': 'https://fake.server',
      'user_agent': 'alphapulldown',
  }
  assert calls['build_monomer_feature'] == {
      'sequence': 'ACDE',
      'msa': 'PRECOMPUTED_UNPAIRED',
      'template_feature': 'TEMPLATE_FROM_RESEARCH',
  }
  assert (
      calls['enrich_mmseq_feature_dict_with_identifiers']
      == 'PRECOMPUTED_UNPAIRED'
  )
  assert isinstance(monomer.feature_dict['template_confidence_scores'], np.ndarray)
  assert monomer.feature_dict['template_release_date'] == ['none']


def test_resolve_species_ids_by_accession_retries_after_transport_failure(
    monkeypatch,
):
  calls = []

  def fake_query(accessions, *, urlopen):
    calls.append(tuple(accessions))
    if len(calls) == 1:
      raise error.URLError('temporary outage')
    return {
        'results': [
            {
                'primaryAccession': 'A0A636IKY3',
                'organism': {'taxonId': 562},
            }
        ]
    }

  monkeypatch.setattr(
      mmseqs_species_identifiers,
      '_query_uniprot_batch',
      fake_query,
  )

  first = mmseqs_species_identifiers.resolve_species_ids_by_accession(
      ['A0A636IKY3']
  )
  second = mmseqs_species_identifiers.resolve_species_ids_by_accession(
      ['A0A636IKY3']
  )

  assert first == {'A0A636IKY3': ''}
  assert second == {'A0A636IKY3': '562'}
  assert calls == [('A0A636IKY3',), ('A0A636IKY3',)]


def test_resolve_species_ids_by_accession_skips_single_accession_fallback_after_transport_failure(
    monkeypatch,
):
  calls = []

  def fake_query(accessions, *, urlopen):
    calls.append(tuple(accessions))
    raise error.URLError('offline')

  monkeypatch.setattr(
      mmseqs_species_identifiers,
      '_query_uniprot_batch',
      fake_query,
  )

  resolved = mmseqs_species_identifiers.resolve_species_ids_by_accession(
      ['A0A636IKY3', 'A0A743YDY2']
  )

  assert resolved == {
      'A0A636IKY3': '',
      'A0A743YDY2': '',
  }
  assert calls == [('A0A636IKY3', 'A0A743YDY2')]
