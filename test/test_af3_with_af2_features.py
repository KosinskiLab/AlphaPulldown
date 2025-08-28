#!/usr/bin/env python
from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from absl import logging
from absl.testing import absltest

import tempfile
from alphapulldown.objects import MultimericObject, MonomericObject
from alphapulldown import __path__ as apd_path
from alphapulldown.folding_backend.alphafold3_backend import AlphaFold3Backend, predict_structure
from alphafold.common import residue_constants
from alphapulldown.utils.msa_encoding import a3m_to_ids, ids_to_a3m 


class TestAF3WithAF2Features(absltest.TestCase):
  def test_msa_identity_after_slicing_and_runtime_dump(self):
    # Load monomeric pickles
    repo_root = Path(__file__).resolve().parents[1]
    features_dir = repo_root / 'test' / 'test_data' / 'features'
    pkl_a = features_dir / '3L4Q_A.3L4Q.cif.A.pkl'
    pkl_c = features_dir / '3L4Q_C.3L4Q.pdb.C.pkl'

    with open(pkl_a, 'rb') as fa, open(pkl_c, 'rb') as fc:
      mono_a: MonomericObject = pickle.load(fa)
      mono_c: MonomericObject = pickle.load(fc)

    # Build multimer and trigger feature creation as in pipeline
    multi = MultimericObject(interactors=[mono_a, mono_c], pair_msa=True)
    # After construction, multi.feature_dict should contain 'msa' combined
    combined = np.asarray(multi.feature_dict.get('msa'))
    self.assertIsNotNone(combined)
    self.assertGreater(combined.size, 0)


    # Prepare AF3 input via backend and call predict_structure directly with a fake runner
    out_dir = Path(tempfile.mkdtemp(prefix='af3_msa_debug_'))
    objects_to_model = [{'object': multi, 'output_dir': str(out_dir)}]
    mappings = AlphaFold3Backend.prepare_input(objects_to_model=objects_to_model, random_seed=1234)
    self.assertEqual(len(mappings), 1)
    fold_input_obj, mapping_val = next(iter(mappings[0].items()))

    class _FakeRunner:
      def run_inference(self, featurised_example, rng_key):
        return {}
      def extract_structures(self, batch, result, target_name):
        return []

    # This should dump A3M/NPZ before attempting inference
    predict_structure(
      fold_input=fold_input_obj,
      model_runner=_FakeRunner(),
      buckets=(512,),
      output_dir=str(out_dir),
      resolve_msa_overlaps=False,
      debug_msas=True,
    )

    # Compare MultimericObject MSA (ids) to backend pre-pairing A3M (chars)
    a3m_path = next(out_dir.glob('*final_complex_msa.a3m'))
    #skip all lines that start with '>'
    lines = a3m_path.read_text().splitlines()
    a3m_text = "\n".join([line for line in lines if not line.startswith('>')])
    a3m_ids = a3m_to_ids(a3m_text)
    r = min(a3m_ids.shape[0], combined.shape[0])
    c = min(a3m_ids.shape[1], combined.shape[1])
    # Convert to chars using ids_to_a3m
    print(ids_to_a3m(a3m_ids[:r, :c])[:10])
    print(ids_to_a3m(combined[:r, :c])[:10])
    # If not equal, print the difference
    if not np.array_equal(a3m_ids[:r, :c], combined[:r, :c]):
      logging.error(f"Difference at index {r}, {c}: {a3m_ids[r, c]} != {combined[r, c]}")
    self.assertTrue(np.array_equal(a3m_ids[:r, :c], combined[:r, :c]))


if __name__ == '__main__':
  absltest.main()


