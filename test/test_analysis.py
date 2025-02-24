import math
import numpy as np
from absl.testing import absltest, parameterized
from typing import List
from alphapulldown.analysis_pipeline.run_analysis import (
    InterfaceAnalysis,
    ComplexAnalysis,
    D0
)

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
        # Euclidean distance between atom coordinates.
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

    # Yield each atom in this residue
    def __iter__(self):
        if hasattr(self, "CB"):
            yield self.CB
        if hasattr(self, "CA"):
            yield self.CA

# --- Dummy Data Builders ---
def create_dummy_chains():
    # Two chains, each with two residues:
    # A1 near B1; A2 far from B1 & B2.
    chain_A = [
        DummyResidue("1", "ALA", "A", use_cb=True,  coord=(0, 0, 0),   bfactor=80),
        DummyResidue("2", "GLY", "A", use_cb=False, coord=(10,10,10), bfactor=90),
    ]
    chain_B = [
        DummyResidue("1", "LYS", "B", use_cb=True,  coord=(1,0,0),    bfactor=70),
        DummyResidue("2", "SER", "B", use_cb=True,  coord=(50,50,50), bfactor=60),
    ]
    return chain_A, chain_B

def create_dummy_res_index_map(chain_A, chain_B):
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
    # 4x4 matrix for 4 residues: A1=0, A2=1, B1=2, B2=3
    # A1-B1 is close => PAE=8 in (0,2) & (2,0).
    return [
        [0,   5,   8,  20],
        [5,   0,   25, 25],
        [8,   10,  0,   6],
        [20, 25,   6,   0]
    ]

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
        # A1 (bfactor=80) and B1 (bfactor=70) => average=75
        expected = 75.0
        self.assertAlmostEqual(iface.average_interface_plddt, expected, places=5)

    def test_average_interface_pae(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # Single contact pair => (A1,B1)=8, (B1,A1)=8 => average=8
        expected = 8.0
        self.assertAlmostEqual(iface.average_interface_pae, expected, places=5)

    def test_pDockQ(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # One contact => log10(1)=0 => pDockQ = 0.724/(1+exp(0.052*152.611)) + 0.018
        expected = 0.724 / (1 + math.exp(0.052 * 152.611)) + 0.018
        self.assertAlmostEqual(iface.pDockQ, expected, places=6)

    def test_pDockQ2_and_ipsae(self):
        iface = InterfaceAnalysis(
            self.chain_A, self.chain_B,
            self.contact_thresh, self.pae_matrix,
            self.res_index_map
        )
        # PTM = 1 / [1 + (8/D0)^2]
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
        # Only A1-B1 => PAE=8 => LIS=(12-8)/12=0.3333
        self.assertAlmostEqual(iface.lis(), 0.33333, places=5)

# --- Dummy Model/Structure for ComplexAnalysis ---
class DummyModel:
    """A dummy model that returns lists of dummy chains and atoms."""
    def __init__(self, chains: List[List[DummyResidue]]):
        self._chains = chains

    def get_chains(self):
        return self._chains

    def get_atoms(self):
        """Flatten all residues -> atoms."""
        all_atoms = []
        for chain in self._chains:
            for residue in chain:
                for atom in residue:
                    all_atoms.append(atom)
        return all_atoms

class DummyStructure:
    """A dummy structure that yields a single dummy model."""
    def __init__(self, model: DummyModel):
        self._model = model

    def get_models(self):
        yield self._model

    def get_residues(self):
        residues = []
        for chain in self._model.get_chains():
            residues.extend(chain)
        return residues

# --- Test Suite for ComplexAnalysis ---
class ComplexAnalysisTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.chain_A, self.chain_B = create_dummy_chains()
        self.dummy_model = DummyModel([self.chain_A, self.chain_B])
        self.dummy_structure = DummyStructure(self.dummy_model)
        self.res_index_map = create_dummy_res_index_map(self.chain_A, self.chain_B)
        self.pae_matrix = create_dummy_pae_matrix()

        # Create a dummy ranking metric
        self.ranking_metric = {
            "multimer": True,
            "iptm+ptm": 0.09161740784114437,
            "iptm": 0.08377959013960515
        }

        # Create a dummy ComplexAnalysis instance by subclassing to bypass file I/O
        class DummyComplexAnalysis(ComplexAnalysis):
            def __init__(self, structure_file, pae_file, ranking_metric, contact_thresh):
                # Bypass real file parsing; inject dummy data.
                self.chain_A, self.chain_B = create_dummy_chains()
                self.res_index_map = create_dummy_res_index_map(self.chain_A, self.chain_B)
                self.pae_matrix = create_dummy_pae_matrix()
                self.ranking_metric = ranking_metric
                self.contact_thresh = contact_thresh
                self.structure = DummyStructure(DummyModel([self.chain_A, self.chain_B]))
                # Create the interface analyses
                self.interfaces = []
                model = next(self.structure.get_models())
                chains = list(model.get_chains())
                for i in range(len(chains)):
                    for j in range(i + 1, len(chains)):
                        iface = InterfaceAnalysis(chains[i],
                                                 chains[j],
                                                 self.contact_thresh,
                                                 self.pae_matrix,
                                                 self.res_index_map)
                        if iface.num_intf_residues > 0:
                            self.interfaces.append(iface)

            @property
            def num_chains(self):
                model = next(self.structure.get_models())
                return len(list(model.get_chains()))

        self.comp = DummyComplexAnalysis("dummy.pdb", "dummy.pae", self.ranking_metric, 12.0)
        # Inject the same dummy structure references so everything matches
        self.comp.structure = self.dummy_structure
        self.comp.pae_matrix = self.pae_matrix
        self.comp.res_index_map = self.res_index_map

    def test_interface_creation(self):
        # We expect exactly one interface: A1 with B1
        self.assertEqual(len(self.comp.interfaces), 1)
        iface = self.comp.interfaces[0]
        self.assertAlmostEqual(iface.average_interface_plddt, 75.0)
        self.assertAlmostEqual(iface.average_interface_pae, 8.0)

    def test_global_metrics(self):
        # Should have a nonzero global contact count
        self.assertGreater(self.comp.contact_pairs_global, 0)
        # And mpDockQ should be a float
        self.assertIsInstance(self.comp.mpDockQ, float)

if __name__ == '__main__':
    absltest.main()
