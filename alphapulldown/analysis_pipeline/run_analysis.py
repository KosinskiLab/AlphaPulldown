import os
import glob
import math
import json
import csv
from typing import Any, Dict, List, Tuple, Set
from functools import cached_property

from absl import app, flags, logging
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

FLAGS = flags.FLAGS
flags.DEFINE_string('pathToDir', None,
                    'Path to the directory containing predicted model files and ranking_debug.json')
flags.DEFINE_float('cutoff', 5.0, 'Cutoff value for distances (and PAE threshold)')
flags.DEFINE_integer('surface_thres', 2, 'Surface threshold. Must be integer')


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
    Calculates per-interface metrics, e.g.:
      - average interface pLDDT (using B-factors as a proxy)
      - number of atom-atom contact pairs between the two chains.
    """
    def __init__(self, chain1, chain2, cutoff: float) -> None:
        """
        Args:
            chain1: First chain (a Bio.PDB.Chain object)
            chain2: Second chain (a Bio.PDB.Chain object)
            cutoff: Distance cutoff for interface determination.
        """
        self.chain1 = chain1
        self.chain2 = chain2
        self.cutoff = cutoff
        self.interface_residues_chain1, self.interface_residues_chain2 = self._get_interface_residues()

    def _get_interface_residues(self) -> Tuple[Set, Set]:
        """
        Identify residues in chain1 and chain2 that are at the interface.
        Returns:
            A tuple (set_of_residues_chain1, set_of_residues_chain2)
        """
        res1_set = set()
        res2_set = set()
        for res1 in self.chain1:
            for res2 in self.chain2:
                # Check if any atom in res1 is within cutoff of any atom in res2.
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
        """
        Calculates the average pLDDT over the interface residues.
        Here, the B-factor is used as a proxy for pLDDT.
        """
        bvals = []
        # Combine interface residues from both chains.
        for res in self.interface_residues_chain1.union(self.interface_residues_chain2):
            # Use CA atom if available.
            atom = res["CA"] if "CA" in res.child_dict else list(res.child_dict.values())[0]
            bvals.append(atom.get_bfactor())
        return sum(bvals) / len(bvals) if bvals else float('nan')

    def contact_pairs(self) -> int:
        """
        Counts the number of atom-atom contacts between the two chains at the interface.
        """
        count = 0
        for res1 in self.interface_residues_chain1:
            for res2 in self.interface_residues_chain2:
                for atom1 in res1:
                    for atom2 in res2:
                        if atom1 - atom2 < self.cutoff:
                            count += 1
        return count


# === ComplexAnalysis Class ===

class ComplexAnalysis:
    """
    Represents a predicted complex.
    Loads structure, ranking, and PAE data and creates per-interface analyses.
    """
    def __init__(self, structure_file: str, pae_file: str,
                 ranking_metric: Dict[str, Any], cutoff: float = 5.0) -> None:
        """
        Args:
            structure_file: Path to the CIF/PDB file.
            pae_file: Path to the PAE JSON file.
            ranking_metric: Dictionary containing ranking metrics for the model.
            cutoff: Distance cutoff for interface calculations.
        """
        self.structure_file = structure_file
        self.cutoff = cutoff

        # Choose the appropriate parser.
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

        # Create an InterfaceAnalysis object for every pair of chains.
        self.interfaces = self._create_interfaces()

    def _create_interfaces(self) -> List[InterfaceAnalysis]:
        """
        For each pairwise combination of chains in the first model, create an InterfaceAnalysis.
        """
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

    @cached_property
    def iptm_ptm(self) -> float:
        """Combined AlphaFold ipTM and pTM score (multimer only)."""
        return self._iptm_ptm if self._iptm_ptm is not None else float('nan')

    @cached_property
    def iptm(self) -> float:
        """AlphaFold ipTM score (multimer only)."""
        return self._iptm if self._iptm is not None else float('nan')

    @cached_property
    def pdockq(self) -> float:
        """Global docking quality score computed via a sigmoid formulation."""
        L, x0, k, b = 0.827, 261.398, 0.036, 0.221
        contacts = self.contact_pairs_global()
        return L / (1 + math.exp(-k * (contacts - x0))) + b

    def contact_pairs_global(self) -> int:
        """
        Count atom-atom contacts over the entire complex.
        """
        count = 0
        residues = list(self.structure.get_residues())
        for i, res1 in enumerate(residues):
            for res2 in residues[i + 1:]:
                if res1.get_parent().id == res2.get_parent().id:
                    continue
                for atom1 in res1:
                    for atom2 in res2:
                        if atom1 - atom2 < self.cutoff:
                            count += 1
        return count

    @cached_property
    def average_interface_pae(self) -> float:
        """Global average PAE for the complex."""
        pae_values: List[float] = [value for row in self.pae_data for value in row]
        good_values = [v for v in pae_values if v < self.cutoff]
        return sum(good_values) / len(good_values) if good_values else float('nan')

    @cached_property
    def average_interface_plddt(self) -> float:
        """
        Global average interface pLDDT.
        Calculated as the average of per-interface average pLDDT values.
        """
        if not self.interfaces:
            return float('nan')
        values = [iface.average_interface_plddt() for iface in self.interfaces]
        return sum(values) / len(values)

    @cached_property
    def num_intf_residues(self) -> int:
        """Global number of unique interface residues (across all interfaces)."""
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        return len(all_res)

    @cached_property
    def polar(self) -> int:
        """Global count of polar interface residues."""
        polar_res = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        return sum(1 for res in all_res if res.get_resname() in polar_res)

    @cached_property
    def hydrophobic(self) -> int:
        """Global count of hydrophobic interface residues."""
        hydro_res = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'}
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        return sum(1 for res in all_res if res.get_resname() in hydro_res)

    @cached_property
    def charged(self) -> int:
        """Global count of charged interface residues."""
        charged_res = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        all_res = set()
        for iface in self.interfaces:
            all_res.update(iface.interface_residues_chain1)
            all_res.update(iface.interface_residues_chain2)
        return sum(1 for res in all_res if res.get_resname() in charged_res)

    @cached_property
    def contact_pairs(self) -> int:
        """Global contact pairs count over the entire complex."""
        return self.contact_pairs_global()

    @cached_property
    def sc(self) -> float:
        """Global shape complementarity score (dummy implementation)."""
        return 0.65

    @cached_property
    def hb(self) -> int:
        """Global hydrogen bonds count (dummy implementation)."""
        return 5

    @cached_property
    def sb(self) -> int:
        """Global salt bridges count (dummy implementation)."""
        return 2

    @cached_property
    def int_solv_en(self) -> float:
        """Global interface solvation energy (dummy implementation)."""
        return -15.0

    @cached_property
    def int_area(self) -> float:
        """Global interface area (dummy implementation)."""
        return 1200.0


# === Processing All Models ===

def process_all_models(directory: str, cutoff: float) -> None:
    ranking_data = parse_ranking_debug_json_all(directory)
    models_order: List[str] = ranking_data["order"]

    output_file = os.path.join(directory, "scores.csv")
    headers = [
        "model", "iptm_ptm", "iptm", "pdockq", "mpdockq", "average_interface_pae",
        "average_interface_plddt", "num_intf_residues", "polar", "hydrophobic",
        "charged", "contact_pairs", "sc", "hb", "sb", "int_solv_en", "int_area",
        "max_predicted_aligned_error", "predicted_aligned_error"
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for model in models_order:
            try:
                r_metric = get_ranking_metric_for_model(ranking_data, model)
                struct_file = find_structure_file(directory, model)
                pae_file = os.path.join(directory, f"pae_{model}.json")
                complex_analysis = ComplexAnalysis(struct_file, pae_file, r_metric, cutoff)
                row = {
                    "model": model,
                    "iptm_ptm": complex_analysis.iptm_ptm,
                    "iptm": complex_analysis.iptm,
                    "pdockq": complex_analysis.pdockq,
                    "mpdockq": complex_analysis.mpdockq,
                    "average_interface_pae": complex_analysis.average_interface_pae,
                    "average_interface_plddt": complex_analysis.average_interface_plddt,
                    "num_intf_residues": complex_analysis.num_intf_residues,
                    "polar": complex_analysis.polar,
                    "hydrophobic": complex_analysis.hydrophobic,
                    "charged": complex_analysis.charged,
                    "contact_pairs": complex_analysis.contact_pairs,
                    "sc": complex_analysis.sc,
                    "hb": complex_analysis.hb,
                    "sb": complex_analysis.sb,
                    "int_solv_en": complex_analysis.int_solv_en,
                    "int_area": complex_analysis.int_area,
                    "max_predicted_aligned_error": complex_analysis.max_predicted_aligned_error,
                    "predicted_aligned_error": complex_analysis.predicted_aligned_error
                }
                writer.writerow(row)
                logging.info("Processed model: %s", model)
                # Write per-interface metrics to a separate CSV if interfaces exist.
                if complex_analysis.interfaces:
                    interface_file = os.path.join(directory, f"{model}_interface_metrics.csv")
                    with open(interface_file, "w", newline="") as ifile:
                        iface_headers = ["interface", "average_interface_plddt", "contact_pairs"]
                        iface_writer = csv.DictWriter(ifile, fieldnames=iface_headers)
                        iface_writer.writeheader()
                        for iface in complex_analysis.interfaces:
                            iface_writer.writerow({
                                "interface": f"{iface.chain1.id}_{iface.chain2.id}",
                                "average_interface_plddt": iface.average_interface_plddt(),
                                "contact_pairs": iface.contact_pairs()
                            })
                    logging.info("Interface metrics written to %s", interface_file)
            except Exception as e:
                logging.error("Error processing model %s: %s", model, e)
    logging.info("Global scores written to %s", output_file)


def main(argv: List[str]) -> None:
    del argv
    process_all_models(FLAGS.pathToDir, FLAGS.cutoff)


if __name__ == '__main__':
    app.run(main)
