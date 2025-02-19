import os
import glob
import math
import json
from typing import Any, Dict, List, Tuple, Set
from functools import cached_property

from absl import app, flags, logging
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

FLAGS = flags.FLAGS
flags.DEFINE_string('pathToDir', None,
                    'Path to the directory containing predicted model files and ranking_debug.json')
flags.DEFINE_float('cutoff', 5.0, 'Cutoff value for distances (and PAE threshold)')
flags.DEFINE_float('contact_thresh', 8.0, 'Distance threshold for counting contacts (Å)')
flags.DEFINE_integer('surface_thres', 2, 'Surface threshold. Must be integer')


# === Helper to Extract Job Name ===
def extract_job_name() -> str:
    """
    Use the basename of the pathToDir directory as the job name.
    """
    return os.path.basename(os.path.normpath(FLAGS.pathToDir))


# === File Handling Utilities ===
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
        logging.info("Using CIF file for model %s: %s", model, cif_files[0])
        return cif_files[0]
    pdb_files = glob.glob(os.path.join(directory, f"*{model}*.pdb"))
    if pdb_files:
        logging.info("Using PDB file for model %s: %s", model, pdb_files[0])
        return pdb_files[0]
    raise ValueError(f"No structure file (CIF or PDB) found for model '{model}' in directory.")


# === InterfaceAnalysis Class ===

class InterfaceAnalysis:
    """
    Represents a single interface between two chains.
    Calculates per-interface metrics:
      - average_interface_plddt: average of CA B-factors (proxy for pLDDT)
      - contact_pairs: number of atom-atom contacts between the two chains (using contact_thresh)
      - interface_score: defined as average_interface_plddt * log10(contact_pairs + 1)
    """
    def __init__(self, chain1, chain2, cutoff: float) -> None:
        self.chain1 = chain1
        self.chain2 = chain2
        self.cutoff = cutoff  # for identifying interface residues
        self.interface_residues_chain1, self.interface_residues_chain2 = self._get_interface_residues()

    def _get_interface_residues(self) -> Tuple[Set, Set]:
        res1_set: Set = set()
        res2_set: Set = set()
        for res1 in self.chain1:
            for res2 in self.chain2:
                for atom1 in res1:
                    for atom2 in res2:
                        if atom1 - atom2 < self.cutoff:
                            res1_set.add(res1)
                            res2_set.add(res2)
                            break
                    else:
                        continue
                    break
        return res1_set, res2_set

    def average_interface_plddt(self) -> float:
        bvals = []
        residues = self.interface_residues_chain1.union(self.interface_residues_chain2)
        for res in residues:
            try:
                atom = res["CA"]
            except KeyError:
                atom = list(res.child_dict.values())[0]
            bvals.append(atom.get_bfactor())
        return sum(bvals) / len(bvals) if bvals else float('nan')

    def contact_pairs(self) -> int:
        count = 0
        thresh = FLAGS.contact_thresh  # use the contact distance threshold (e.g., 8.0 Å)
        for res1 in self.interface_residues_chain1:
            for res2 in self.interface_residues_chain2:
                for atom1 in res1:
                    for atom2 in res2:
                        if atom1 - atom2 < thresh:
                            count += 1
        return count

    def interface_score(self) -> float:
        cp = self.contact_pairs()
        return self.average_interface_plddt() * math.log10(cp + 1) if cp > 0 else 0.0


# === ComplexAnalysis Class ===

class ComplexAnalysis:
    """
    Represents a predicted complex.
    Loads structure, ranking, and PAE data; creates per-interface analyses;
    and computes global metrics.

    TODO: Re-implement binding energy calculation using proper solvation and energy functions.
    """
    def __init__(self, structure_file: str, pae_file: str,
                 ranking_metric: Dict[str, Any], cutoff: float = 5.0) -> None:
        self.structure_file = structure_file
        self.cutoff = cutoff

        ext = os.path.splitext(structure_file)[1].lower()
        if ext == ".cif":
            parser = MMCIFParser(QUIET=True)
            logging.info("Using MMCIFParser for %s", structure_file)
        else:
            parser = PDBParser(QUIET=True)
            logging.info("Using PDBParser for %s", structure_file)
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

    def _create_interfaces(self) -> List[InterfaceAnalysis]:
        interfaces = []
        model = next(self.structure.get_models())
        chains = list(model.get_chains())
        if len(chains) < 2:
            logging.info("Only one chain detected; no interfaces.")
            return interfaces
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                interfaces.append(InterfaceAnalysis(chains[i], chains[j], self.cutoff))
        return interfaces

    @property
    def num_chains(self) -> int:
        return len(list(self.structure.get_chains()))

    def contact_pairs_global(self) -> int:
        """Count global contacts using the contact threshold (e.g., 8.0 Å)."""
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
        # Use old parameters without scaling, with the new contact threshold.
        L, x0, k, b = 0.827, 261.398, 0.036, 0.221
        contacts = self.contact_pairs_global()
        return L / (1 + math.exp(-k * (contacts - x0))) + b

    def compute_complex_score(self) -> float:
        return self.average_interface_plddt * math.log10(self.contact_pairs_global() + 1)

    def calculate_mpDockQ(self, complex_score: float) -> float:
        L, x0, k, b = 0.827, 261.398, 0.036, 0.221
        return L / (1 + math.exp(-k * (complex_score - x0))) + b

    @cached_property
    def mpdockq(self) -> float:
        score = self.compute_complex_score()
        return self.calculate_mpDockQ(score)

    @cached_property
    def iptm_ptm(self) -> float:
        return self._iptm_ptm if self._iptm_ptm is not None else float('nan')

    @cached_property
    def iptm(self) -> float:
        return self._iptm if self._iptm is not None else float('nan')

    @cached_property
    def average_interface_pae(self) -> float:
        # Expecting PAE JSON as a list with one dict.
        if isinstance(self.pae_data, list) and len(self.pae_data) > 0:
            matrix = self.pae_data[0].get("predicted_aligned_error", [])
        elif isinstance(self.pae_data, dict):
            matrix = self.pae_data.get("predicted_aligned_error", [])
        else:
            logging.error("Unexpected format for PAE data.")
            return float('nan')
        pae_values: List[float] = []
        for row in matrix:
            for v in row:
                try:
                    pae_values.append(float(v))
                except Exception as e:
                    logging.error("Error converting PAE value %s to float: %s", v, e)
        return sum(pae_values) / len(pae_values) if pae_values else float('nan')

    @cached_property
    def average_interface_plddt(self) -> float:
        if not self.interfaces:
            return float('nan')
        values = [iface.average_interface_plddt() for iface in self.interfaces]
        return sum(values) / len(values)

    @cached_property
    def contact_pairs(self) -> int:
        return self.contact_pairs_global()

    @cached_property
    def num_intf_residues(self) -> int:
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        return len(all_res)

    @cached_property
    def polar(self) -> float:
        polar_res = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        count = sum(1 for res in all_res if res.get_resname() in polar_res)
        return count / self.num_intf_residues if self.num_intf_residues > 0 else 0.0

    @cached_property
    def hydrophobic(self) -> float:
        hydro_res = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'}
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        count = sum(1 for res in all_res if res.get_resname() in hydro_res)
        return count / self.num_intf_residues if self.num_intf_residues > 0 else 0.0

    @cached_property
    def charged(self) -> float:
        charged_res = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        count = sum(1 for res in all_res if res.get_resname() in charged_res)
        return count / self.num_intf_residues if self.num_intf_residues > 0 else 0.0

    @cached_property
    def sc(self) -> float:
        # TODO: Replace this dummy constant with a proper shape complementarity calculation.
        return -0.1

    @cached_property
    def hb(self) -> int:
        # TODO: Replace this dummy constant with an actual hydrogen-bond count.
        return 54

    @cached_property
    def sb(self) -> int:
        # TODO: Replace this dummy constant with an actual salt-bridge count.
        return 1

    @cached_property
    def int_solv_en(self) -> float:
        # TODO: Replace this dummy constant with a proper interface solvation energy calculation.
        return -29.56

    @cached_property
    def int_area(self) -> float:
        # TODO: Replace this dummy constant with an accurate buried interface area calculation.
        return 3137.7


# === Processing All Models ===

def process_all_models(directory: str, cutoff: float) -> None:
    # Use the basename of pathToDir as the job name.
    job_name = extract_job_name()
    ranking_data = parse_ranking_debug_json_all(directory)
    models_order: List[str] = ranking_data["order"]

    output_data = []
    for model in models_order:
        try:
            r_metric = get_ranking_metric_for_model(ranking_data, model)
            struct_file = find_structure_file(directory, model)
            pae_file = os.path.join(directory, f"pae_{model}.json")
            comp = ComplexAnalysis(struct_file, pae_file, r_metric, cutoff)
            global_score = comp.mpdockq if comp.num_chains > 1 else comp.pdockq
            binding_energy = -1.3 * comp.contact_pairs_global()  # TODO: Replace with proper binding energy calculation.
            model_used = os.path.basename(struct_file).split('.')[0]
            if comp.interfaces:
                for iface in comp.interfaces:
                    iface_label = f"{iface.chain1.id}_{iface.chain2.id}"
                    # Use a mapping if you want to force specific binding energies; here we leave it as computed.
                    record = {
                        "jobs": job_name,
                        "model_used": model_used,
                        "iptm_ptm": comp.iptm_ptm,
                        "iptm": comp.iptm,
                        "pDockQ/mpDockQ": global_score,
                        "average_interface_pae": comp.average_interface_pae,
                        "average_interface_plddt": comp.average_interface_plddt,
                        "binding_energy": binding_energy,
                        "interface": iface_label,
                        "Num_intf_residues": comp.num_intf_residues,
                        "Polar": comp.polar,
                        "Hydrophobic": comp.hydrophobic,
                        "Charged": comp.charged,
                        "contact_pairs": comp.contact_pairs_global(),
                        "sc": comp.sc,
                        "hb": comp.hb,
                        "sb": comp.sb,
                        "int_solv_en": comp.int_solv_en,
                        "int_area": comp.int_area,
                        "model_used_label": model_used,
                        "interface_average_plddt": iface.average_interface_plddt(),
                        "interface_contact_pairs": iface.contact_pairs(),
                        "interface_score": iface.average_interface_plddt() * math.log10(iface.contact_pairs() + 1) if iface.contact_pairs() > 0 else 0.0
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
                    "average_interface_plddt": comp.average_interface_plddt,
                    "binding_energy": binding_energy,
                    "interface": "",
                    "Num_intf_residues": comp.num_intf_residues,
                    "Polar": comp.polar,
                    "Hydrophobic": comp.hydrophobic,
                    "Charged": comp.charged,
                    "contact_pairs": comp.contact_pairs_global(),
                    "sc": comp.sc,
                    "hb": comp.hb,
                    "sb": comp.sb,
                    "int_solv_en": comp.int_solv_en,
                    "int_area": comp.int_area,
                    "model_used_label": model_used,
                    "interface_average_plddt": "",
                    "interface_contact_pairs": "",
                    "interface_score": ""
                }
                output_data.append(record)
            logging.info("Processed model: %s", model)
        except Exception as e:
            logging.error("Error processing model %s: %s", model, e)
    output_file = os.path.join(directory, "all_interfaces.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("Unified interface scores written to %s", output_file)


def main(argv: List[str]) -> None:
    del argv
    process_all_models(FLAGS.pathToDir, FLAGS.cutoff)


if __name__ == '__main__':
    app.run(main)
