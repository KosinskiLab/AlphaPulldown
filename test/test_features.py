import numpy as np
from absl.testing import parameterized, absltest

from alphapulldown.features import (
    ProteinSequence, MSAFeatures, TemplateFeatures, ProteinFeatures, InvalidRegionError
)

class FeatureClassesTest(parameterized.TestCase):
    def setUp(self):
        L, N, C = 10, 4, 3
        # aatype: (N, L, 21)
        self.aatype = np.zeros((N, L, 21), dtype=int)
        # positions: (N, L, C, 3)
        self.positions = np.zeros((N, L, C, 3), dtype=float)
        # masks: (N, L, C)
        self.masks = np.zeros((N, L, C), dtype=bool)

        self.tpl_feats = TemplateFeatures(
            aatype=self.aatype,
            all_atom_positions=self.positions,
            all_atom_mask=self.masks,
            template_domain_names=[f"dom{i}" for i in range(N)]
        )

        # build a minimal ProteinFeatures:
        from alphapulldown.features import MSAFeatures
        seq = ProteinSequence("id", "A"*L)
        msa = MSAFeatures(
            msa=np.zeros((2, L), int),
            deletion_matrix=np.zeros((2, L), int),
            species_identifiers=["x","y"]
        )
        self.prot_feats = ProteinFeatures(
            sequence=seq,
            msa=msa,
            template=self.tpl_feats,
        )

    @parameterized.parameters((1,3), (5,8))
    def test_template_features_get_region(self, start, end):
        sub = self.tpl_feats.get_region(start, end)
        # aatype should slice [ :, start-1:end, : ]
        np.testing.assert_array_equal(sub.aatype, self.aatype[:, start-1:end, :])
        # positions should slice [ :, start-1:end, :, : ]
        np.testing.assert_array_equal(
            sub.all_atom_positions,
            self.positions[:, start-1:end, :, :]
        )
        # masks are 3D, so slice [ :, start-1:end, : ]
        np.testing.assert_array_equal(
            sub.all_atom_mask,
            self.masks[:, start-1:end, :]
        )

    def test_protein_sequence_get_region_invalid(self):
        p = ProteinSequence("x","ABC")
        with self.assertRaises(InvalidRegionError):
            p.get_region(0,2)

    def test_protein_sequence_get_region_valid(self):
        p = ProteinSequence("id","ABCDE")
        sub = p.get_region(2,4)
        self.assertEqual(sub.sequence, "BCD")
        self.assertTrue(sub.identifier.endswith("_2-4"))

    def test_msa_features_get_region(self):
        msa = np.arange(20).reshape(4,5)
        delmat = np.zeros_like(msa)
        mf = MSAFeatures(msa, delmat, ["a","b","c","d"])
        sub = mf.get_region(2,4)
        np.testing.assert_array_equal(sub.msa, msa[:,1:4])
        np.testing.assert_array_equal(sub.deletion_matrix, delmat[:,1:4])
        self.assertListEqual(sub.species_identifiers, ["a","b","c","d"])

    def test_protein_features_get_region_and_get_regions(self):
        # single region
        subpf = self.prot_feats.get_region(2,5)
        self.assertEqual(len(subpf.sequence.sequence), 4)
        # multi-region
        multi = self.prot_feats.get_regions([(1,2),(5,7)])
        self.assertEqual(len(multi.sequence.sequence), (2 + 3))

if __name__ == "__main__":
    absltest.main()
