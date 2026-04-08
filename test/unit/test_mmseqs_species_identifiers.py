import json
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


def _write_identifier_sidecar(
    cache_path,
    a3m: str,
    *,
    species_ids: list[str],
    accessions: list[str],
    source_sha256: str | None = None,
):
  if source_sha256 is None:
    source_sha256 = mmseqs_species_identifiers._calculate_mmseq_source_sha256(
        mmseqs_species_identifiers.strip_mmseq_comment_lines(a3m)
    )
  cache_path.write_text(
      json.dumps(
          {
              'source_sha256': source_sha256,
              'msa_species_identifiers': species_ids,
              'msa_uniprot_accession_identifiers': accessions,
          }
      ),
      encoding='utf-8',
  )


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


def test_build_mmseq_identifier_features_uses_matching_sidecar_without_resolver(
    tmp_path,
):
  a3m = '\n'.join([
      '# mmseqs header',
      '>101',
      'ACDE',
      '>UniRef100_A0A636IKY3\t136\t0.883',
      'ACDF',
      '',
  ])
  cache_path = tmp_path / 'protein.mmseq_ids.json'
  _write_identifier_sidecar(
      cache_path,
      a3m,
      species_ids=['', '108619'],
      accessions=['', 'A0A636IKY3'],
  )

  calls = []
  features = mmseqs_species_identifiers.build_mmseq_identifier_features(
      a3m,
      species_resolver=lambda accessions: calls.append(tuple(accessions)),
      cache_path=str(cache_path),
      expected_rows=2,
  )

  assert features['msa_species_identifiers'].tolist() == [b'', b'108619']
  assert features['msa_uniprot_accession_identifiers'].tolist() == [
      b'',
      b'A0A636IKY3',
  ]
  assert calls == []


def test_build_mmseq_identifier_features_writes_sidecar_on_cache_miss(tmp_path):
  a3m = '\n'.join([
      '>101',
      'ACDE',
      '>UniRef100_A0A636IKY3\t136\t0.883',
      'ACDF',
      '',
  ])
  cache_path = tmp_path / 'protein.mmseq_ids.json'
  calls = []

  features = mmseqs_species_identifiers.build_mmseq_identifier_features(
      a3m,
      species_resolver=lambda accessions: calls.append(tuple(accessions)) or {
          'A0A636IKY3': '108619'
      },
      cache_path=str(cache_path),
      expected_rows=2,
  )

  assert calls == [('A0A636IKY3',)]
  assert features['msa_species_identifiers'].tolist() == [b'', b'108619']
  payload = json.loads(cache_path.read_text(encoding='utf-8'))
  assert payload['msa_species_identifiers'] == ['', '108619']
  assert payload['msa_uniprot_accession_identifiers'] == ['', 'A0A636IKY3']


def test_build_mmseq_identifier_features_refreshes_stale_checksum(tmp_path):
  a3m = '\n'.join([
      '>101',
      'ACDE',
      '>UniRef100_A0A636IKY3\t136\t0.883',
      'ACDF',
      '',
  ])
  cache_path = tmp_path / 'protein.mmseq_ids.json'
  _write_identifier_sidecar(
      cache_path,
      a3m,
      species_ids=['', 'stale'],
      accessions=['', 'A0A636IKY3'],
      source_sha256='wrong',
  )
  calls = []

  features = mmseqs_species_identifiers.build_mmseq_identifier_features(
      a3m,
      species_resolver=lambda accessions: calls.append(tuple(accessions)) or {
          'A0A636IKY3': '108619'
      },
      cache_path=str(cache_path),
      expected_rows=2,
  )

  assert calls == [('A0A636IKY3',)]
  assert features['msa_species_identifiers'].tolist() == [b'', b'108619']
  payload = json.loads(cache_path.read_text(encoding='utf-8'))
  assert payload['source_sha256'] != 'wrong'
  assert payload['msa_species_identifiers'] == ['', '108619']


def test_build_mmseq_identifier_features_refreshes_row_count_mismatch(tmp_path):
  a3m = '\n'.join([
      '>101',
      'ACDE',
      '>UniRef100_A0A636IKY3\t136\t0.883',
      'ACDF',
      '',
  ])
  cache_path = tmp_path / 'protein.mmseq_ids.json'
  _write_identifier_sidecar(
      cache_path,
      a3m,
      species_ids=[''],
      accessions=[''],
  )
  calls = []

  features = mmseqs_species_identifiers.build_mmseq_identifier_features(
      a3m,
      species_resolver=lambda accessions: calls.append(tuple(accessions)) or {
          'A0A636IKY3': '108619'
      },
      cache_path=str(cache_path),
      expected_rows=2,
  )

  assert calls == [('A0A636IKY3',)]
  assert features['msa_species_identifiers'].tolist() == [b'', b'108619']
  payload = json.loads(cache_path.read_text(encoding='utf-8'))
  assert payload['msa_species_identifiers'] == ['', '108619']


def test_build_mmseq_identifier_features_reuses_blank_species_ids_from_sidecar(
    tmp_path,
):
  a3m = '\n'.join([
      '>101',
      'ACDE',
      '>UniRef100_A0A636IKY3\t136\t0.883',
      'ACDF',
      '',
  ])
  cache_path = tmp_path / 'protein.mmseq_ids.json'
  first_calls = []
  first = mmseqs_species_identifiers.build_mmseq_identifier_features(
      a3m,
      species_resolver=lambda accessions: first_calls.append(tuple(accessions)) or {},
      cache_path=str(cache_path),
      expected_rows=2,
  )

  mmseqs_species_identifiers._SPECIES_ID_CACHE.clear()
  second_calls = []
  second = mmseqs_species_identifiers.build_mmseq_identifier_features(
      a3m,
      species_resolver=lambda accessions: second_calls.append(tuple(accessions)),
      cache_path=str(cache_path),
      expected_rows=2,
  )

  assert first_calls == [('A0A636IKY3',)]
  assert first['msa_species_identifiers'].tolist() == [b'', b'']
  assert second['msa_species_identifiers'].tolist() == [b'', b'']
  assert second_calls == []


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

  def fake_enrich(feature_dict, a3m, **kwargs):
    calls['enrich_mmseq_feature_dict_with_identifiers'] = {
        'a3m': a3m,
        'kwargs': kwargs,
    }
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
  assert calls['enrich_mmseq_feature_dict_with_identifiers'] == {
      'a3m': '>101\nACDE',
      'kwargs': {'cache_path': str(tmp_path / 'dummy.mmseq_ids.json')},
  }
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


def test_build_mmseq_identifier_features_skips_non_uniprot_identifiers(
    monkeypatch,
):
  calls = []

  def fake_resolver(accessions):
    calls.append(tuple(accessions))
    return {'A0A636IKY3': '108619'}

  a3m = '\n'.join([
      '>101',
      'ACDE',
      '>MGYP000264027769',
      'ACDF',
      '>UniRef100_MGYP000264027769',
      'ACDG',
      '>UniRef100_A0A636IKY3',
      'ACDH',
      '',
  ])

  features = mmseqs_species_identifiers.build_mmseq_identifier_features(
      a3m,
      species_resolver=fake_resolver,
  )

  assert calls == [('A0A636IKY3',)]
  assert features['msa_species_identifiers'].tolist() == [
      b'',
      b'',
      b'',
      b'108619',
  ]
  assert features['msa_uniprot_accession_identifiers'].tolist() == [
      b'',
      b'',
      b'',
      b'A0A636IKY3',
  ]


def test_resolve_species_ids_by_accession_skips_unsupported_accessions(
    monkeypatch,
):
  uniprot_calls = []
  uniparc_calls = []

  def fake_uniprot_query(accessions, *, urlopen):
    uniprot_calls.append(tuple(accessions))
    return {
        'results': [
            {
                'primaryAccession': 'A0A636IKY3',
                'organism': {'taxonId': 562},
            }
        ]
    }

  def fake_uniparc_query(accessions, *, urlopen):
    uniparc_calls.append(tuple(accessions))
    return {
        'results': [
            {
                'uniParcId': 'UPI001118B830',
                'organisms': [{'taxonId': 83333}],
            }
        ]
    }

  monkeypatch.setattr(
      mmseqs_species_identifiers,
      '_query_uniprot_batch',
      fake_uniprot_query,
  )
  monkeypatch.setattr(
      mmseqs_species_identifiers,
      '_query_uniparc_batch',
      fake_uniparc_query,
  )

  resolved = mmseqs_species_identifiers.resolve_species_ids_by_accession(
      ['A0A636IKY3', 'MGYP000264027769', 'UPI001118B830']
  )

  assert resolved == {
      'A0A636IKY3': '562',
      'MGYP000264027769': '',
      'UPI001118B830': '83333',
  }
  assert uniprot_calls == [('A0A636IKY3',)]
  assert uniparc_calls == [('UPI001118B830',)]
