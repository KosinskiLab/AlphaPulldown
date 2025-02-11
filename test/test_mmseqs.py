import os
import tempfile
import numpy as np

from absl.testing import absltest
from alphapulldown.objects import MonomericObject

# Dummy implementations for the functions used by make_mmseq_features.
def fake_build_monomer_feature(sequence, msa, template_features):
    # Return a dummy dictionary that will be later updated.
    return {"dummy_feature": 42,
            "template_confidence_scores": None,
            "template_release_date": None}

# Define fake MSA_FEATURES as a tuple.
class FakeMSAPairing:
    MSA_FEATURES = ("dummy_feature",)

# Save originals to restore later.
_original_build_monomer_feature = None
_original_get_msa_and_templates = None
_original_unserialize_msa = None

def fake_get_msa_and_templates(jobname, query_sequences, a3m_lines, result_dir,
                               msa_mode, use_templates, custom_template_path,
                               pair_mode, host_url, user_agent):
    # Return fake tuple values.
    fake_unpaired = ["FAKE_UNPAIRED"]
    fake_paired = ["FAKE_PAIRED"]
    fake_unique = ["FAKE_UNIQUE"]
    fake_card = ["FAKE_CARDINALITY"]
    fake_template = ["FAKE_TEMPLATE"]
    return (fake_unpaired, fake_paired, fake_unique, fake_card, fake_template)

def fake_unserialize_msa(a3m_lines, sequence):
    # Return fake tuple values based solely on the precomputed file.
    fake_unpaired = ["PRECOMPUTED_UNPAIRED"]
    fake_paired = ["PRECOMPUTED_PAIRED"]
    fake_unique = ["PRECOMPUTED_UNIQUE"]
    fake_card = ["PRECOMPUTED_CARDINALITY"]
    fake_template = ["PRECOMPUTED_TEMPLATE"]
    return (fake_unpaired, fake_paired, fake_unique, fake_card, fake_template)

class MmseqFeaturesTest(absltest.TestCase):

    def setUp(self):
        super(MmseqFeaturesTest, self).setUp()
        # Create a dummy MonomericObject with a known description and sequence.
        self.monomer = MonomericObject("dummy", "ACDE")
        # Create a temporary output directory.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

        # Monkey-patch the functions used inside make_mmseq_features.
        import alphapulldown.objects as objects_mod
        self._original_build_monomer_feature = objects_mod.build_monomer_feature
        self._original_get_msa_and_templates = objects_mod.get_msa_and_templates
        self._original_unserialize_msa = objects_mod.unserialize_msa

        objects_mod.build_monomer_feature = fake_build_monomer_feature
        objects_mod.get_msa_and_templates = fake_get_msa_and_templates
        objects_mod.unserialize_msa = fake_unserialize_msa

        # Override msa_pairing.MSA_FEATURES with a tuple.
        objects_mod.msa_pairing.MSA_FEATURES = FakeMSAPairing.MSA_FEATURES

    def tearDown(self):
        # Restore originals.
        import alphapulldown.objects as objects_mod
        objects_mod.build_monomer_feature = self._original_build_monomer_feature
        objects_mod.get_msa_and_templates = self._original_get_msa_and_templates
        objects_mod.unserialize_msa = self._original_unserialize_msa
        self.temp_dir.cleanup()
        super(MmseqFeaturesTest, self).tearDown()

    def test_use_precomputed_msa(self):
        """Test that if a precomputed MSA exists and use_precomputed_msa is True,
        the branch using unserialize_msa is taken."""
        # Create a dummy precomputed a3m file.
        a3m_path = os.path.join(self.output_dir, self.monomer.description + ".a3m")
        precomputed_content = ">dummy\nPRECOMPUTED_CONTENT\n"
        with open(a3m_path, "w") as f:
            f.write(precomputed_content)

        # Call the method with use_precomputed_msa=True.
        self.monomer.make_mmseq_features(
            DEFAULT_API_SERVER="http://fake.api",
            output_dir=self.output_dir,
            use_precomputed_msa=True
        )
        # Our fake_unserialize_msa returns fake values that we check:
        self.assertEqual(self.monomer.feature_dict["dummy_feature"], 42)
        # Check that template_confidence_scores and template_release_date got set.
        self.assertTrue(isinstance(self.monomer.feature_dict["template_confidence_scores"], np.ndarray))
        self.assertEqual(self.monomer.feature_dict["template_release_date"], ['none'])

    def test_api_generation(self):
        """Test that if no precomputed MSA exists (or use_precomputed_msa is False),
        the API branch is taken and a new a3m file is created."""
        a3m_path = os.path.join(self.output_dir, self.monomer.description + ".a3m")
        # Ensure the file does not exist.
        if os.path.exists(a3m_path):
            os.remove(a3m_path)
        # Call the method with use_precomputed_msa=False.
        self.monomer.make_mmseq_features(
            DEFAULT_API_SERVER="http://fake.api",
            output_dir=self.output_dir,
            use_precomputed_msa=False
        )
        # The fake_get_msa_and_templates returns known fake values.
        self.assertEqual(self.monomer.feature_dict["dummy_feature"], 42)
        # The a3m file should now exist.
        self.assertTrue(os.path.isfile(a3m_path))
        # Check that the file contains our dummy content from the fake_get_msa_and_templates branch.
        with open(a3m_path) as f:
            msa_content = f.read()
        self.assertIn("FAKE_UNPAIRED", msa_content)
        # Check that default template values were added.
        self.assertTrue(isinstance(self.monomer.feature_dict["template_confidence_scores"], np.ndarray))
        self.assertEqual(self.monomer.feature_dict["template_release_date"], ['none'])


if __name__ == '__main__':
    absltest.main()
