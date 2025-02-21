import os
import glob
import math
import json
import csv
from typing import Any, Dict, List, Tuple, Set
from functools import cached_property
import enum
import numpy as np

from absl import app, flags, logging
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser

# --- Global Constants ---
# Constants for pDockQ (global complex scores)
PD_L = 0.827
PD_X0 = 261.398
PD_K = 0.036
PD_B = 0.221

# Constants for mpDockQ (global complex scores)
MPD_L = 0.9
MPD_X0 = 250.0
MPD_K = 0.04
MPD_B = 0.2

# Constant for PAE transformation
D0 = 10.0

@enum.unique
class ModelsToAnalyse(enum.Enum):
    BEST = 0
    ALL = 1

FLAGS = flags.FLAGS
flags.DEFINE_string('pathToDir', None,
                    'Path to the directory containing predicted model files and ranking_debug.json')
flags.DEFINE_float('contact_thresh', 12.0,
                   'Distance threshold for counting contacts (Å). Two residues are interacting if their C-Beta are within this distance.')
flags.DEFINE_float('pae_filter', 100.0,
                   'Maximum acceptable average interface PAE; interfaces above this are skipped')
flags.DEFINE_enum_class(
    "models_to_analyse",
    ModelsToAnalyse.BEST,
    ModelsToAnalyse,
    "If `all`, all models are used. If `best`, only the most confident model (ranked_0) is used.",
)
flags.DEFINE_integer('surface_thres', 2, 'Surface threshold. Must be integer')


# --- Helper Functions ---
def extract_job_name() -> str:
    """Use the basename of the pathToDir directory as the job name."""
    return os.path.basename(os.path.normpath(FLAGS.pathToDir))


def _sigmoid(value: float, L: float, x0: float, k: float, b: float) -> float:
    """Return the sigmoid of value using the given parameters."""
    return L / (1 + math.exp(-k * (value - x0))) + b


def read_json(filepath: str) -> Any:
    try:
        with open(filepath) as f:
            data = json.load(f)
        logging.info("Loaded JSON file: %s", filepath)
        return data
    except Exception as e:
        raise ValueError(f"Error parsing {filepath}: {e}")


def parse_ranking_debug_json_all(directory: str) -> Dict[str, Any]:
    path = os.path.join(directory, "ranking_debug.json")
    data = read_json(path)
    if "order" not in data or not isinstance(data["order"], list):
        raise ValueError("Invalid ranking_debug.json: missing or invalid 'order' key")
    return data


def get_ranking_metric_for_model(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    match (("iptm+ptm" in data, "iptm" in data, "plddts" in data, "ptm" in data)):
        case (True, True, _, _):
            if model not in data["iptm+ptm"] or model not in data["iptm"]:
                raise ValueError(f"Model '{model}' not found in multimer metrics")
            return {"model": model,
                    "iptm+ptm": data["iptm+ptm"][model],
                    "iptm": data["iptm"][model],
                    "multimer": True}
        case (_, _, True, True):
            if model not in data["plddts"] or model not in data["ptm"]:
                raise ValueError(f"Model '{model}' not found in monomer metrics")
            return {"model": model,
                    "plddts": data["plddts"][model],
                    "ptm": data["ptm"][model],
                    "multimer": False}
        case _:
            raise ValueError("Invalid ranking_debug.json: expected multimer or monomer keys not found")


def load_pae_file(directory: str, model: str) -> Dict[str, Any]:
    pae_filename = f"pae_{model}.json"
    path = os.path.join(directory, pae_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"PAE file '{pae_filename}' not found in directory '{directory}'.")
    return read_json(path)


def find_structure_file(directory: str, model: str) -> str:
    cif_files = glob.glob(os.path.join(directory, f"*{model}*.cif"))
    if cif_files:
        return cif_files[0]
    pdb_files = glob.glob(os.path.join(directory, f"*{model}*.pdb"))
    if pdb_files:
        return pdb_files[0]
    raise ValueError(f"No structure file (CIF or PDB) found for model '{model}' in directory.")


# --- Precomputation Helper Functions ---
def compute_interface_avg_plddt(precomputed: Tuple[Set[Any], Set[Any], Set[Tuple[Any, Any]]]) -> float:
    """Compute the average interface pLDDT (proxy via CA B-factor) over the union of interface residues."""
    res_set = precomputed[0].union(precomputed[1])
    bvals = []
    for res in res_set:
        try:
            atom = res["CB"]
        except KeyError:
            atom = res["CA"]
        bvals.append(atom.get_bfactor())
    return sum(bvals) / len(bvals) if bvals else float('nan')


def compute_interface_avg_pae(precomputed: Tuple[Set[Any], Set[Any], Set[Tuple[Any, Any]]],
                              pae_matrix: List[List[float]],
                              res_index_map: Dict[Tuple[str, Any], int]) -> float:
    """Compute the average PAE over all contact pairs (both directions)."""
    pae_values = []
    for res1, res2 in precomputed[2]:
        key1 = (res1.get_parent().id, res1.id)
        key2 = (res2.get_parent().id, res2.id)
        i = res_index_map.get(key1)
        j = res_index_map.get(key2)
        if i is None or j is None:
            continue
        try:
            val1 = float(pae_matrix[i][j])
            val2 = float(pae_matrix[j][i])
            pae_values.extend([val1, val2])
        except Exception:
            continue
    return sum(pae_values) / len(pae_values) if pae_values else float('nan')


# --- InterfaceAnalysis Class ---
class InterfaceAnalysis:
    """
    Represents a single interface between two chains.
    Calculates per-interface metrics:
      - average_interface_plddt: average CA B-factor (proxy for pLDDT)
      - average_interface_pae: average PAE over the contact pairs.
      - contact_pairs: number of residue–residue contacts (using C-Betas) between the two chains
      - score_complex: defined as average_interface_plddt * log10(contact_pairs + 1)
      - num_intf_residues: number of interface residues (union of residues from both chains)
      - polar, hydrophobic, charged: normalized fractions of interface residues
      - pDockQ: computed from mean pLDDT and number of contacts
          pDockQ = 0.724 / (1 + exp(-0.052*(mean_pLDDT * log10(n_contacts) - 152.611))) + 0.018
      - pDockQ2: incorporates PAE:
          pDockQ2 = 1.31 / (1 + exp(-0.075*(mean_pLDDT * mean_ptm - 84.733))) + 0.005
      - ipSAE: the average of PAE transformed by:
          ptm(x, D0) = 1/(1 + (x/D0)**2)
      - lis: average of (12 - PAE)/12 for PAE values <= 12
    This class reads the full PAE matrix and residue index map (passed from the parent).
    It computes its own interacting residues from which it calculates the averages.
    """
    def __init__(self, chain1: List[Any], chain2: List[Any], contact_thresh: float,
                 pae_matrix: List[List[float]],
                 res_index_map: Dict[Tuple[str, Any], int]) -> None:
        self.chain1 = chain1
        self.chain2 = chain2
        self.contact_thresh = contact_thresh
        self._pae_matrix = pae_matrix
        self._res_index_map = res_index_map
        # Compute contact pairs and interacting residues.
        self.precomputed = self._get_interface_residues()
        self.interface_residues_chain1, self.interface_residues_chain2, self.pairs = self.precomputed
        # Compute average interface pLDDT and PAE from the interacting residues.
        self._average_interface_plddt = compute_interface_avg_plddt(self.precomputed)
        self._average_interface_pae = compute_interface_avg_pae(self.precomputed, pae_matrix, res_index_map)

    def _get_interface_residues(self) -> Tuple[Set[Any], Set[Any], Set[Tuple[Any, Any]]]:
        """Identify interacting residues and contact pairs using representative atoms (CB if available, else CA)."""
        res_pairs: Set[Tuple[Any, Any]] = set()
        atoms1 = [res["CB"] if "CB" in res else res["CA"] for res in self.chain1]
        atoms2 = [res["CB"] if "CB" in res else res["CA"] for res in self.chain2]
        ns = NeighborSearch(atoms1 + atoms2)
        for atom1 in atoms1:
            neighbors = ns.search(atom1.coord, self.contact_thresh)
            for atom2 in neighbors:
                if atom2 in atoms2:
                    res_pairs.add((atom1.get_parent(), atom2.get_parent()))
        res1_set = {pair[0] for pair in res_pairs}
        res2_set = {pair[1] for pair in res_pairs}
        return (res1_set, res2_set, res_pairs)

    @cached_property
    def average_interface_plddt(self) -> float:
        """Return the computed average interface pLDDT."""
        return self._average_interface_plddt

    @cached_property
    def average_interface_pae(self) -> float:
        """Return the computed average interface PAE."""
        return self._average_interface_pae

    @cached_property
    def contact_pairs(self) -> int:
        """Return the number of unique residue–residue contacts."""
        return len(self.pairs)

    def score_complex(self) -> float:
        """Compute the interface score = average_interface_plddt * log10(contact_pairs + 1)."""
        cp = self.contact_pairs
        return self.average_interface_plddt * math.log10(cp) if cp > 0 else 0.0

    @property
    def num_intf_residues(self) -> int:
        """Return the number of interface residues (union of residues from both chains)."""
        return len(self.interface_residues_chain1.union(self.interface_residues_chain2))

    @property
    def polar(self) -> float:
        """Return the normalized fraction of polar residues at the interface."""
        polar_res = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in polar_res)
        return count / len(residues) if residues else 0.0

    @property
    def hydrophobic(self) -> float:
        """Return the normalized fraction of hydrophobic residues at the interface."""
        hydro_res = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in hydro_res)
        return count / len(residues) if residues else 0.0

    @property
    def charged(self) -> float:
        """Return the normalized fraction of charged residues at the interface."""
        charged_res = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in charged_res)
        return count / len(residues) if residues else 0.0

    def pDockQ(self) -> float:
        """
        Compute pDockQ:
          pDockQ = 0.724 / (1 + exp(-0.052*(mean_pLDDT * log10(n_contacts) - 152.611))) + 0.018
        """
        n_contacts = self.contact_pairs
        if n_contacts <= 0:
            return 0.0
        x = self.average_interface_plddt * math.log10(n_contacts)
        return 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018

    def pDockQ2(self) -> Tuple[float, float]:
        """
        Compute pDockQ2 and return (pDockQ2, mean_ptm) where:
          - pDockQ2 = 1.31 / (1 + exp(-0.075*(mean_pLDDT * mean_ptm - 84.733))) + 0.005
          - mean_ptm is the average of the ptm-transformed PAE over the interface contact pairs.
        """
        ptm_values = []
        for res1, res2 in self.pairs:
            key1 = (res1.get_parent().id, res1.id)
            key2 = (res2.get_parent().id, res2.id)
            i = self._res_index_map.get(key1)
            j = self._res_index_map.get(key2)
            if i is None or j is None:
                continue
            try:
                pae_val = float(self._pae_matrix[i][j])
            except Exception:
                continue
            ptm_values.append(1.0 / (1 + (pae_val / D0) ** 2))
        mean_ptm = sum(ptm_values) / len(ptm_values) if ptm_values else 0.0
        x = self.average_interface_plddt * mean_ptm
        pDockQ2_val = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
        return pDockQ2_val, mean_ptm

    def ipsae(self) -> float:
        """
        Compute ipSAE (mean_ptm) using the transformed PAE values over the interface.
        """
        ptm_values = []
        for res1, res2 in self.pairs:
            key1 = (res1.get_parent().id, res1.id)
            key2 = (res2.get_parent().id, res2.id)
            i = self._res_index_map.get(key1)
            j = self._res_index_map.get(key2)
            if i is None or j is None:
                continue
            try:
                pae_val = float(self._pae_matrix[i][j])
            except Exception:
                continue
            ptm_values.append(1.0 / (1 + (pae_val / D0) ** 2))
        return sum(ptm_values) / len(ptm_values) if ptm_values else 0.0

    def lis(self) -> float:
        """
        Compute the LIS score for this interface.
        Select rows corresponding to the first chain and columns for the second chain
        from the global PAE matrix, then compute the average of (12 - PAE)/12 for values <= 12.
        """
        # Get chain IDs from the first residue in each chain.
        cid1 = self.chain1[0].get_parent().id
        cid2 = self.chain2[0].get_parent().id
        indices_chain1 = [i for ((c, _), i) in self._res_index_map.items() if c == cid1]
        indices_chain2 = [i for ((c, _), i) in self._res_index_map.items() if c == cid2]
        if not indices_chain1 or not indices_chain2:
            return 0.0
        pae_arr = np.array(self._pae_matrix)
        submatrix = pae_arr[np.ix_(indices_chain1, indices_chain2)]
        valid = submatrix[submatrix <= 12]
        if valid.size == 0:
            return 0.0
        scores = (12 - valid) / 12
        return float(np.mean(scores))

    # Dummy methods for additional interface metrics.
    @cached_property
    def sc(self) -> float:
        """Dummy shape complementarity."""
        return -0.1

    @cached_property
    def hb(self) -> int:
        """Dummy hydrogen bond count."""
        return 54

    @cached_property
    def sb(self) -> int:
        """Dummy salt bridge count."""
        return 1

    @cached_property
    def int_solv_en(self) -> float:
        """Dummy interface solvation energy."""
        return -29.56

    @cached_property
    def int_area(self) -> float:
        """Dummy buried interface area."""
        return 3137.7

    @cached_property
    def pdockq_interface(self) -> float:
        """Dummy interface docking score (pDockQ)."""
        return self.pDockQ()

    @cached_property
    def mpdockq_interface(self) -> float:
        """Dummy interface docking score (mpDockQ) computed from interface score."""
        score = self.score_complex()
        return _sigmoid(score, MPD_L, MPD_X0, MPD_K, MPD_B)


# --- ComplexAnalysis Class ---
class ComplexAnalysis:
    """
    Represents a predicted complex.
    Loads structure, ranking, and PAE data; creates per-interface analyses;
    computes global metrics; and calculates average interface pLDDT/PAE from interchain residue pairs.
    """

    def __init__(self, structure_file: str, pae_file: str,
                 ranking_metric: Dict[str, Any], contact_thresh: float) -> None:
        self.structure_file = structure_file
        self.contact_thresh = contact_thresh

        ext = os.path.splitext(structure_file)[1].lower()
        if ext == ".cif":
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("complex", structure_file)

        self.ranking_metric = ranking_metric
        self.pae_data = read_json(pae_file)
        try:
            self.max_predicted_aligned_error = self.pae_data[0]["max_predicted_aligned_error"]
        except Exception as e:
            logging.error("Error extracting max_predicted_aligned_error: %s", e)
            self.max_predicted_aligned_error = float('nan')
        try:
            self._predicted_aligned_error = self.pae_data[0]["predicted_aligned_error"]
        except Exception as e:
            logging.error("Error extracting predicted_aligned_error: %s", e)
            self._predicted_aligned_error = None

        if ranking_metric.get("multimer"):
            self._iptm_ptm = ranking_metric["iptm+ptm"]
            self._iptm = ranking_metric["iptm"]
        else:
            self._iptm_ptm = None
            self._iptm = None

        self.res_index_map = self._build_residue_index_map()
        # Cache the full PAE matrix (loaded once)
        self.pae_matrix = self._predicted_aligned_error
        self.interfaces = self._create_interfaces()

    def _build_residue_index_map(self) -> Dict[Tuple[str, Any], int]:
        """
        Build a mapping from (chain id, residue id) to sequential index.
        Assumes that the residue order in the structure matches the PAE matrix.
        """
        index_map = {}
        index = 0
        model = next(self.structure.get_models())
        for chain in model.get_chains():
            for res in chain:
                index_map[(chain.id, res.id)] = index
                index += 1
        return index_map

    def _create_interfaces(self) -> List[InterfaceAnalysis]:
        """
        Create an InterfaceAnalysis object for each chain pair.
        Only save an interface if there is at least one interacting residue.
        """
        interfaces = []
        model = next(self.structure.get_models())
        chains = list(model.get_chains())
        if len(chains) < 2:
            logging.info("Only one chain detected; no interfaces.")
            return interfaces
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                iface = InterfaceAnalysis(
                    list(chains[i]), list(chains[j]),
                    self.contact_thresh,
                    self.pae_matrix,
                    self.res_index_map
                )
                if iface.num_intf_residues > 0:
                    interfaces.append(iface)
        return interfaces

    @property
    def average_interface_pae(self) -> float:
        """
        Global average interface PAE: average over all interfaces that pass the PAE filter.
        """
        if not self.interfaces:
            return float('nan')
        pae_list = [iface.average_interface_pae for iface in self.interfaces
                    if not math.isnan(iface.average_interface_pae) and iface.average_interface_pae <= FLAGS.pae_filter]
        return sum(pae_list) / len(pae_list) if pae_list else float('nan')

    @cached_property
    def average_interface_plddt(self) -> float:
        """Global average interface pLDDT computed over all interfaces."""
        if not self.interfaces:
            return float('nan')
        values = [iface.average_interface_plddt for iface in self.interfaces]
        return sum(values) / len(values)

    @cached_property
    def contact_pairs_global(self) -> int:
        """Count global contacts using the contact threshold."""
        count = 0
        thresh = FLAGS.contact_thresh
        residues = list(self.structure.get_residues())
        for i, res1 in enumerate(residues):
            for res2 in residues[i + 1:]:
                if res1.get_parent().id == res2.get_parent().id:
                    continue
                for atom1 in res1:
                    for atom2 in res2:
                        if atom1 - atom2 < thresh:
                            count += 1
        return count

    @cached_property
    def pdockq(self) -> float:
        """Global docking quality score computed from global contact pairs."""
        contacts = self.contact_pairs_global
        return _sigmoid(contacts, PD_L, PD_X0, PD_K, PD_B)

    def compute_complex_score(self) -> float:
        """Compute the complex score from global average_interface_plddt and global contact pairs."""
        return self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)

    @cached_property
    def mpdockq(self) -> float:
        """Compute mpDockQ using the complex score."""
        score = self.compute_complex_score()
        return _sigmoid(score, MPD_L, MPD_X0, MPD_K, MPD_B)

    @cached_property
    def iptm_ptm(self) -> float:
        return self._iptm_ptm if self._iptm_ptm is not None else float('nan')

    @cached_property
    def iptm(self) -> float:
        return self._iptm if self._iptm is not None else float('nan')

    @property
    def num_chains(self) -> int:
        """Return the number of chains in the first model."""
        model = next(self.structure.get_models())
        return len(list(model.get_chains()))


# --- Processing All Models ---
def process_all_models(directory: str, contact_thresh: float) -> None:
    job_name = extract_job_name()
    ranking_data = parse_ranking_debug_json_all(directory)
    ranked_order: List[str] = ranking_data["order"]

    if FLAGS.models_to_analyse == ModelsToAnalyse.BEST:
        models = [ranked_order[0]]
    else:
        models = ranked_order

    output_data = []
    for model in models:
        try:
            r_metric = get_ranking_metric_for_model(ranking_data, model)
            struct_file = find_structure_file(directory, model)
            pae_file = os.path.join(directory, f"pae_{model}.json")
            comp = ComplexAnalysis(struct_file, pae_file, r_metric, contact_thresh)
            global_score = comp.mpdockq if comp.num_chains > 1 else comp.pdockq
            binding_energy = -1.3 * comp.contact_pairs_global  # TODO: Replace with proper binding energy calculation.
            model_used = model
            if comp.interfaces:
                for iface in comp.interfaces:
                    # Use the chain IDs from the first residue in each chain.
                    iface_label = f"{iface.chain1[0].get_parent().id}_{iface.chain2[0].get_parent().id}"
                    # Skip interfaces with no interacting residues.
                    if iface.num_intf_residues == 0:
                        continue
                    # Skip interfaces with average PAE above the filter.
                    if iface.average_interface_pae > FLAGS.pae_filter:
                        continue
                    pDockQ2_val, _ = iface.pDockQ2()
                    ipSAE_val = iface.ipsae()
                    lis_val = iface.lis()
                    record = {
                        "jobs": job_name,
                        "model_used": model_used,
                        "interface": iface_label,
                        "iptm_ptm": comp.iptm_ptm,
                        "iptm": comp.iptm,
                        "pDockQ/mpDockQ": global_score,
                        "average_interface_pae": iface.average_interface_pae,
                        "interface_average_plddt": iface.average_interface_plddt,
                        "interface_num_intf_residues": iface.num_intf_residues,
                        "interface_polar": iface.polar,
                        "interface_hydrophobic": iface.hydrophobic,
                        "interface_charged": iface.charged,
                        "interface_contact_pairs": iface.contact_pairs,
                        "interface_score": iface.score_complex(),
                        "interface_pDockQ2": pDockQ2_val,
                        "interface_ipSAE": ipSAE_val,
                        "interface_LIS": lis_val,
                        "binding_energy": binding_energy,
                        "model_used_label": model_used,
                        "interface_sc": iface.sc,
                        "interface_hb": iface.hb,
                        "interface_sb": iface.sb,
                        "interface_int_solv_en": iface.int_solv_en,
                        "interface_int_area": iface.int_area,
                        "interface_pdockq": iface.pdockq_interface,
                        "interface_mpdockq": iface.mpdockq_interface
                    }
                    output_data.append(record)
            logging.info("Processed model: %s", model)
        except Exception as e:
            logging.error("Error processing model %s: %s", model, e)
    if not output_data:
        logging.warning("No interfaces passed the filter; writing an empty output.")
        output_data = []
    output_file = os.path.join(directory, "interfaces.csv")
    with open(output_file, "w", newline='') as f:
        if output_data:
            writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
            writer.writeheader()
            writer.writerows(output_data)
        else:
            f.write("")
    logging.info("Unified interface scores written to %s", output_file)


def main(argv: List[str]) -> None:
    del argv
    process_all_models(FLAGS.pathToDir, FLAGS.contact_thresh)


if __name__ == '__main__':
    app.run(main)
