import os
import glob
import math
import json
import csv
from typing import Any, Dict, List, Tuple, Set
from functools import cached_property
import enum

from absl import app, flags, logging
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser


@enum.unique
class ModelsToAnalyse(enum.Enum):
    BEST = 0
    ALL = 1


FLAGS = flags.FLAGS
flags.DEFINE_string('pathToDir', None,
                    'Path to the directory containing predicted model files and ranking_debug.json')
# Use one flag for all distance‐based filtering.
flags.DEFINE_float('contact_thresh', 4.0,
                   'Distance threshold for both residue identification and counting contacts (Å)')
# Interfaces with average PAE above this value will be skipped.
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


def _sigmoid(value: float, L: float = 0.827, x0: float = 261.398, k: float = 0.036, b: float = 0.221) -> float:
    """Return the sigmoid of value using the given parameters."""
    return L / (1 + math.exp(-k * (value - x0))) + b


# --- File Handling Utilities ---
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


# --- InterfaceAnalysis Class ---
class InterfaceAnalysis:
    """
    Represents a single interface between two chains.
    Calculates per-interface metrics:
      - average_interface_plddt: average CA B-factor (proxy for pLDDT)
      - contact_pairs: number of atom–atom contacts between the two chains (using contact_thresh)
      - score_complex: defined as average_interface_plddt * log10(contact_pairs + 1)
      - num_intf_residues: number of interface residues (union of residues from both chains)
      - polar, hydrophobic, charged: normalized fractions of interface residues
    """

    def __init__(self, chain1, chain2, contact_thresh: float) -> None:
        self.chain1 = chain1
        self.chain2 = chain2
        self.contact_thresh = contact_thresh  # used for both residue identification and counting contacts
        self.interface_residues_chain1, self.interface_residues_chain2 = self._get_interface_residues()

    def _get_interface_residues(self) -> Tuple[Set, Set]:
        """Identify residues that are part of the interface using contact_thresh."""
        res1_set: Set = set()
        res2_set: Set = set()
        for res1 in self.chain1:
            for res2 in self.chain2:
                for atom1 in res1:
                    for atom2 in res2:
                        if atom1 - atom2 < self.contact_thresh:
                            res1_set.add(res1)
                            res2_set.add(res2)
                            break
                    else:
                        continue
                    break
        return res1_set, res2_set

    def average_interface_plddt(self) -> float:
        """Return the average CA B-factor for interface residues (proxy for pLDDT)."""
        bvals = []
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        for res in residues:
            try:
                atom = res["CA"]
            except KeyError:
                atom = list(res.child_dict.values())[0]
            bvals.append(atom.get_bfactor())
        return sum(bvals) / len(bvals) if bvals else float('nan')

    @cached_property
    def contact_pairs(self) -> int:
        """Count the number of atom–atom contacts (using all atoms) between the two chains."""
        count = 0
        thresh = FLAGS.contact_thresh
        for res1 in self.interface_residues_chain1:
            for res2 in self.interface_residues_chain2:
                for atom1 in res1:
                    for atom2 in res2:
                        if atom1 - atom2 < thresh:
                            count += 1
        return count

    def score_complex(self) -> float:
        """Compute the interface score as: average_interface_plddt * log10(contact_pairs + 1)"""
        cp = self.contact_pairs
        return self.average_interface_plddt() * math.log10(cp + 1) if cp > 0 else 0.0

    @property
    def num_intf_residues(self) -> int:
        """Return the number of interface residues (union of both chains)."""
        return len(self.interface_residues_chain1.union(self.interface_residues_chain2))

    @property
    def polar(self) -> float:
        polar_res = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in polar_res)
        return count / len(residues) if residues else 0.0

    @property
    def hydrophobic(self) -> float:
        hydro_res = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in hydro_res)
        return count / len(residues) if residues else 0.0

    @property
    def charged(self) -> float:
        charged_res = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        count = sum(1 for res in residues if res.get_resname() in charged_res)
        return count / len(residues) if residues else 0.0


# --- ComplexAnalysis Class ---
class ComplexAnalysis:
    """
    Represents a predicted complex.
    Loads structure, ranking, and PAE data; creates per-interface analyses;
    computes global metrics; and calculates average interface PAE for interchain residue pairs.

    Global metrics are computed but only interface-specific values are output.

    TODO: Replace dummy methods (sc, hb, sb, int_solv_en, int_area) with proper algorithms.
    TODO: Refine binding energy calculation.
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
            self.predicted_aligned_error = self.pae_data[0]["predicted_aligned_error"]
        except Exception as e:
            logging.error("Error extracting predicted_aligned_error: %s", e)
            self.predicted_aligned_error = None

        if ranking_metric.get("multimer"):
            self._iptm_ptm = ranking_metric["iptm+ptm"]
            self._iptm = ranking_metric["iptm"]
        else:
            self._iptm_ptm = None
            self._iptm = None

        self.interfaces = self._create_interfaces()
        self.res_index_map = self._build_residue_index_map()

    def _create_interfaces(self) -> List[InterfaceAnalysis]:
        """Create an InterfaceAnalysis object for each pairwise combination of chains."""
        interfaces = []
        model = next(self.structure.get_models())
        chains = list(model.get_chains())
        if len(chains) < 2:
            logging.info("Only one chain detected; no interfaces.")
            return interfaces
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                interfaces.append(InterfaceAnalysis(chains[i], chains[j], FLAGS.contact_thresh))
        return interfaces

    def _build_residue_index_map(self) -> Dict[Tuple[str, Any], int]:
        """
        Build a mapping from (chain id, residue id) to sequential index.
        TODO: Verify that the residue order in the structure matches the PAE matrix.
        """
        index_map = {}
        index = 0
        model = next(self.structure.get_models())
        for chain in model.get_chains():
            for res in chain:
                index_map[(chain.id, res.id)] = index
                index += 1
        return index_map

    def _average_interface_pae_for_interface(self, iface: InterfaceAnalysis) -> float:
        """
        Compute the average PAE for a single interface, considering only interchain residue pairs.
        """
        if self.predicted_aligned_error is None or not self.res_index_map:
            return float('nan')
        pae_vals = []
        for res in iface.interface_residues_chain1:
            key = (res.get_parent().id, res.id)
            i = self.res_index_map.get(key)
            if i is None:
                continue
            for res2 in iface.interface_residues_chain2:
                key2 = (res2.get_parent().id, res2.id)
                j = self.res_index_map.get(key2)
                if j is None:
                    continue
                try:
                    pae_vals.append(float(self.predicted_aligned_error[i][j]))
                    pae_vals.append(float(self.predicted_aligned_error[j][i]))
                except Exception as e:
                    logging.error("Error extracting PAE for indices %s, %s: %s", i, j, e)
        return sum(pae_vals) / len(pae_vals) if pae_vals else float('nan')

    @property
    def average_interface_pae(self) -> float:
        """
        Compute the global average interface PAE as the average over all interfaces,
        considering only interchain residue pairs.
        Interfaces with average PAE above FLAGS.pae_filter are skipped.
        """
        if not self.interfaces:
            return float('nan')
        pae_list = [self._average_interface_pae_for_interface(iface) for iface in self.interfaces]
        pae_list = [val for val in pae_list if not math.isnan(val) and val <= FLAGS.pae_filter]
        return sum(pae_list) / len(pae_list) if pae_list else float('nan')

    @cached_property
    def average_interface_plddt(self) -> float:
        """Global average interface pLDDT computed over all interfaces."""
        if not self.interfaces:
            return float('nan')
        # Call the interface method (as a function) for each interface.
        values = [iface.average_interface_plddt() for iface in self.interfaces]
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
        return _sigmoid(contacts)

    def compute_complex_score(self) -> float:
        """Compute the complex score from global average_interface_plddt and global contact pairs."""
        return self.average_interface_plddt * math.log10(self.contact_pairs_global + 1)

    @cached_property
    def mpdockq(self) -> float:
        """Compute mpDockQ using the complex score."""
        score = self.compute_complex_score()
        return _sigmoid(score)

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

    @cached_property
    def sc(self) -> float:
        # TODO: Replace this dummy constant with a proper shape complementarity calculation.
        return -0.1

    @cached_property
    def hb(self) -> int:
        # TODO: Replace this dummy constant with an actual hydrogen bond count.
        return 54

    @cached_property
    def sb(self) -> int:
        # TODO: Replace this dummy constant with an actual salt bridge count.
        return 1

    @cached_property
    def int_solv_en(self) -> float:
        # TODO: Replace this dummy constant with a proper interface solvation energy calculation.
        return -29.56

    @cached_property
    def int_area(self) -> float:
        # TODO: Replace this dummy constant with an accurate buried interface area calculation.
        return 3137.7


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
                    iface_label = f"{iface.chain1.id}_{iface.chain2.id}"
                    interface_pae = comp._average_interface_pae_for_interface(iface)
                    if interface_pae > FLAGS.pae_filter:
                        continue  # Skip interfaces with average PAE above the filter.
                    record = {
                        "jobs": job_name,
                        "model_used": model_used,
                        "interface": iface_label,
                        "iptm_ptm": comp.iptm_ptm,
                        "iptm": comp.iptm,
                        "pDockQ/mpDockQ": global_score,
                        "average_interface_pae": interface_pae,
                        "interface_average_plddt": iface.average_interface_plddt(),
                        "interface_num_intf_residues": iface.num_intf_residues,
                        "interface_polar": iface.polar,
                        "interface_hydrophobic": iface.hydrophobic,
                        "interface_charged": iface.charged,
                        "interface_contact_pairs": iface.contact_pairs,
                        "interface_score": iface.score_complex(),
                        "binding_energy": binding_energy,
                        "sc": comp.sc,
                        "hb": comp.hb,
                        "sb": comp.sb,
                        "int_solv_en": comp.int_solv_en,
                        "int_area": comp.int_area,
                        "model_used_label": model_used
                    }
                    output_data.append(record)
            else:
                record = {
                    "jobs": job_name,
                    "model_used": model_used,
                    "iptm_ptm": comp.iptm_ptm,
                    "iptm": comp.iptm,
                    "pDockQ/mpDockQ": global_score,
                    "average_interface_pae": comp.average_interface_pae,
                    "binding_energy": binding_energy,
                    "interface": "",
                    "interface_average_plddt": "",
                    "interface_num_intf_residues": "",
                    "interface_polar": "",
                    "interface_hydrophobic": "",
                    "interface_charged": "",
                    "interface_contact_pairs": "",
                    "interface_score": "",
                    "sc": comp.sc,
                    "hb": comp.hb,
                    "sb": comp.sb,
                    "int_solv_en": comp.int_solv_en,
                    "int_area": comp.int_area,
                }
                output_data.append(record)
            logging.info("Processed model: %s", model)
        except Exception as e:
            logging.error("Error processing model %s: %s", model, e)
    if not output_data:
        logging.warning("No interfaces passed the PAE filter; writing an empty output.")
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
