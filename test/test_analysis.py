import math
import numpy as np
from absl.testing import absltest, parameterized
from typing import List

# --- Dummy Classes to Simulate Bio.PDB Objects ---

class DummyAtom:
    def __init__(self, coord, bfactor, parent):
        self.coord = np.array(coord)
        self._bfactor = bfactor
        self.parent = parent

    def get_coord(self):
        return self.coord

    def get_bfactor(self):
        return self._bfactor

    def get_parent(self):
        return self.parent

    def __sub__(self, other):
        # Return the Euclidean distance between atom coordinates.
        return np.linalg.norm(self.coord - other.coord)

class DummyParent:
    def __init__(self, chain_id):
        self.id = chain_id

class DummyResidue:
    def __init__(self, residue_id, resname, chain_id,
                 use_cb=True, coord=(0, 0, 0), bfactor=80.0):
        self.id = residue_id
        self.resname = resname
        self.parent = DummyParent(chain_id)
        if use_cb:
            self.CB = DummyAtom(coord, bfactor, self)
        else:
            self.CA = DummyAtom(coord, bfactor, self)

    def get_parent(self):
        return self.parent

    def get_resname(self):
        return self.resname

    def __getitem__(self, key):
        if key == "CB":
            if hasattr(self, "CB"):
                return self.CB
            else:
                raise KeyError("No CB")
        elif key == "CA":
            if hasattr(self, "CA"):
                return self.CA
            else:
                raise KeyError("No CA")
        raise KeyError(key)

    def __contains__(self, key):
        if key == "CB":
            return hasattr(self, "CB")
        elif key == "CA":
            return hasattr(self, "CA")
        return False

    # NEW: Make iteration yield each atom that exists
    def __iter__(self):
        if hasattr(self, "CB"):
            yield self.CB
        if hasattr(self, "CA"):
            yield self.CA

# --- Dummy Data Builders ---
def create_dummy_chains():
    # Create two chains:
    # Chain A: two residues. Residue A1 is near chain B's residue B1.
    # Chain B: two residues. Residue B1 is near; B2 is far.
    chain_A = [
        DummyResidue("1", "ALA", "A", use_cb=True,  coord=(0, 0, 0),   bfactor=80),
        DummyResidue("2", "GLY", "A", use_cb=False, coord=(10, 10, 10), bfactor=90),
    ]
    chain_B = [
        DummyResidue("1", "LYS", "B", use_cb=True,  coord=(1, 0, 0),   bfactor=70),
        DummyResidue("2", "SER", "B", use_cb=True,  coord=(50, 50, 50), bfactor=60),
    ]
    return chain_A, chain_B

def create_dummy_res_index_map(chain_A, chain_B):
    # Assume order: chain A residues then chain B residues.
    index_map = {}
    index = 0
    for res in chain_A:
        index_map[(res.get_parent().id, res.id)] = index
        index += 1
    for res in chain_B:
        index_map[(res.get_parent().id, res.id)] = index
        index += 1
    return index_map

def create_dummy_pae_matrix():
    # 4x4 matrix for 4 residues.
    # Let indices: A1=0, A2=1, B1=2, B2=3.
    # Only A1 and B1 are close: set their PAE to 8.
    return [
        [0,   5,   8,  20],
        [5,   0,   25, 25],
        [8,   10,  0,  6 ],
        [20,  25,  6,  0 ]
    ]

# Import your real analysis classes from your main code
from alphapulldown.analysis_pipeline.run_analysis import InterfaceAnalysis, ComplexAnalysis, D0

# --- Test Suite for InterfaceAnalysis ---
class InterfaceAnalysisTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.chain_A, self.chain_B = create_dummy_chains()
        self.res_index_map = create_dummy_res_index_map(self.chain_A, self.chain_B)
        self.pae_matrix = create_dummy_pae_matrix()
        self.contact_thresh = 12.0

    def test_interface_contact_pairs(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # Only residue A1 and B1 are within 12 Å.
        self.assertEqual(iface.contact_pairs, 1)

    def test_average_interface_plddt(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # Expected interface residues: A1 (bfactor 80) and B1 (bfactor 70)
        expected = (80 + 70) / 2.0  # 75.0
        self.assertAlmostEqual(iface.average_interface_plddt, expected, places=5)

    def test_average_interface_pae(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # Only one contact pair: (A1,B1) => indices (0,2): 8 and (2,0): 8.
        expected = (8 + 8) / 2.0  # 8.0
        self.assertAlmostEqual(iface.average_interface_pae, expected, places=5)

    def test_pDockQ(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # With one contact, log10(1)=0, so pDockQ = 0.724/(1+exp(0.052*152.611)) + 0.018.
        expected = 0.724 / (1 + math.exp(0.052 * 152.611)) + 0.018
        self.assertAlmostEqual(iface.pDockQ(), expected, places=6)

    def test_pDockQ2_and_ipsae(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # For the only contact pair, ptm = 1/(1+(8/D0)^2)
        expected_ptm = 1.0 / (1 + (8 / D0) ** 2)
        pDockQ2_val, mean_ptm = iface.pDockQ2()
        self.assertAlmostEqual(mean_ptm, expected_ptm, places=5)
        x = iface.average_interface_plddt * expected_ptm  # 75 * expected_ptm
        expected_pDockQ2 = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
        self.assertAlmostEqual(pDockQ2_val, expected_pDockQ2, places=5)
        self.assertAlmostEqual(iface.ipsae(), expected_ptm, places=5)

    def test_lis(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # For chain A and B, only residue A1 and B1 are interface.
        # From dummy PAE matrix, value at (0,2) is 8, so LIS = (12 - 8)/12 = 0.33333.
        self.assertAlmostEqual(iface.lis(), 0.33333, places=5)


# --- Test Suite for ComplexAnalysis ---
class DummyModel:
    """A dummy model that returns a list of dummy chains."""
    def __init__(self, chains: List[List[DummyResidue]]):
        self._chains = chains

    def get_chains(self):
        return self._chains

class DummyStructure:
    """A dummy structure that yields a dummy model and a list of residues."""
    def __init__(self, model: DummyModel):
        self._model = model

    def get_models(self):
        yield self._model

    def get_residues(self):
        residues = []
        for chain in self._model.get_chains():
            residues.extend(chain)
        return residues

class ComplexAnalysisTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.chain_A, self.chain_B = create_dummy_chains()
        self.dummy_model = DummyModel([self.chain_A, self.chain_B])
        self.dummy_structure = DummyStructure(self.dummy_model)
        self.res_index_map = create_dummy_res_index_map(self.chain_A, self.chain_B)
        self.pae_matrix = create_dummy_pae_matrix()

        # Create a dummy ranking metric.
        self.ranking_metric = {
            "multimer": True,
            "iptm+ptm": 0.09161740784114437,
            "iptm": 0.08377959013960515
        }

        # Create a dummy ComplexAnalysis instance by subclassing to bypass file I/O.
        class DummyComplexAnalysis(ComplexAnalysis):
            def __init__(self, structure_file, pae_file, ranking_metric, contact_thresh):
                # Bypass real file parsing; inject dummy data directly.
                self.chain_A, self.chain_B = create_dummy_chains()
                self.res_index_map = create_dummy_res_index_map(self.chain_A, self.chain_B)
                self.pae_matrix = create_dummy_pae_matrix()
                self.ranking_metric = ranking_metric
                self.contact_thresh = contact_thresh
                self.structure = DummyStructure(DummyModel([self.chain_A, self.chain_B]))
                self.interfaces = self._create_interfaces()

            def _build_residue_index_map(self):
                return self.res_index_map

            def _create_interfaces(self):
                iface = InterfaceAnalysis(self.chain_A, self.chain_B, 12.0,
                                          self.pae_matrix, self.res_index_map)
                return [iface] if iface.num_intf_residues > 0 else []

        self.comp = DummyComplexAnalysis("dummy.pdb", "dummy.pae", self.ranking_metric, 12.0)
        # Inject the same dummy structure references so everything matches.
        self.comp.structure = self.dummy_structure
        self.comp.pae_matrix = self.pae_matrix
        self.comp.res_index_map = self.res_index_map
        self.comp.interfaces = self.comp._create_interfaces()

    def test_interface_creation(self):
        self.assertEqual(len(self.comp.interfaces), 1)
        iface = self.comp.interfaces[0]
        self.assertAlmostEqual(iface.average_interface_plddt, 75.0)
        self.assertAlmostEqual(iface.average_interface_pae, 8.0)

    def test_global_metrics(self):
        self.assertGreater(self.comp.contact_pairs_global, 0)
        self.assertIsInstance(self.comp.pdockq, float)
        self.assertIsInstance(self.comp.mpdockq, float)


if __name__ == '__main__':
    absltest.main()
