"""
    A script that surveys all interfaces and obtains all the residues on the interface

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""

from absl import logging
import subprocess
import os
import tempfile
from typing import Any, List, Dict
import numpy as np
from itertools import combinations
import pandas as pd
from Bio.PDB import PDBParser, PDBIO
from biopandas.pdb import PandasPdb
from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta.core.scoring import get_score_function
import pyrosetta

pyrosetta.init()
logging.set_verbosity(logging.INFO)


class PDBAnalyser:
    """
    A class that stores a pandas dataframe of all the information
    of the residues in a PDB file.
    """

    def __init__(self, pdb_file_path: str) -> None:
        self.pdb_file_path = pdb_file_path
        self.pdb_pandas = PandasPdb().read_pdb(pdb_file_path)
        self.pdb = PDBParser().get_structure("ranked_0", pdb_file_path)[0]
        self.pdb_df = self.pdb_pandas.df['ATOM']
        self.chain_combinations = {}
        self.get_all_combinations_of_chains()

    def get_all_combinations_of_chains(self) -> None:
        """Get all pairwise combinations of chains."""
        unique_chain_ids = pd.unique(self.pdb_df.chain_id)
        if len(unique_chain_ids) > 1:
            chain_combinations = combinations(unique_chain_ids, 2)
            for idx, combo in enumerate(chain_combinations):
                self.chain_combinations[f"interface_{idx + 1}"] = combo
        else:
            self.chain_combinations = unique_chain_ids

    def retrieve_C_beta_coords(self, chain_df: pd.DataFrame) -> np.ndarray:
        """
        Retrieve the x, y, z coords of the C beta atoms of one chain.

        Args:
            chain_df: A pandas dataframe that belongs to one chain.

        Returns:
            A numpy array of all C-beta atoms' x, y, z coordinates.

        Note:
            Will retrieve C-alpha atom coords if the residue is glycine.
        """
        mask = (chain_df['atom_name'] == 'CB') | (
            (chain_df['residue_name'] == 'GLY') & (chain_df['atom_name'] == 'CA')
        )
        subdf = chain_df[mask]
        return subdf[['x_coord', 'y_coord', 'z_coord']].values

    def obtain_interface_residues(self, chain_df_1: pd.DataFrame, chain_df_2: pd.DataFrame, cutoff: float = 12):
        """
        Get all the residues on the interface within the cutoff.

        Args:
            chain_df_1: Pandas dataframe for the first chain.
            chain_df_2: Pandas dataframe for the second chain.
            cutoff: Distance cutoff for defining interactions.

        Returns:
            A tuple of arrays containing the indices of interface residues,
            or None if no interface is detected.
        """
        chain_1_CB_coords = self.retrieve_C_beta_coords(chain_df_1)
        chain_2_CB_coords = self.retrieve_C_beta_coords(chain_df_2)
        distance_mtx = np.sqrt(
            np.sum((chain_1_CB_coords[:, np.newaxis] - chain_2_CB_coords)**2, axis=-1)
        )
        satisfied_residues_chain_1, satisfied_residues_chain_2 = np.where(distance_mtx < cutoff)

        if satisfied_residues_chain_1.size > 0 and satisfied_residues_chain_2.size > 0:
            return satisfied_residues_chain_1, satisfied_residues_chain_2
        else:
            print("No interface residues are found.")
            return None

    def calculate_average_pae(self, pae_mtx: np.ndarray, chain_1_residues: np.ndarray, chain_2_residues: np.ndarray) -> float:
        """
        Calculate the average interface PAE.

        Args:
            pae_mtx: An array with PAE values from AlphaFold predictions (num_res * num_res).
            chain_1_residues: Row vector of selected residues in chain 1 (1 * num_selected_res_chain_1).
            chain_2_residues: Row vector of selected residues in chain 2 (1 * num_selected_res_chain_2).

        Returns:
            The average PAE value of the interface residues.
        """
        pae_sum = sum(pae_mtx[i, j] + pae_mtx[j, i] for i, j in zip(chain_1_residues, chain_2_residues))
        return pae_sum / (2 * len(chain_1_residues))

    def calculate_average_plddt(self, chain_1_plddt: List[float], chain_2_plddt: List[float],
                                chain_1_residues: np.ndarray, chain_2_residues: np.ndarray) -> float:
        """
        Calculate the average interface pLDDT.

        Args:
            chain_1_plddt: List of pLDDT values for residues in chain 1.
            chain_2_plddt: List of pLDDT values for residues in chain 2.
            chain_1_residues: Row vector of selected residues in chain 1 (1 * num_selected_res_chain_1).
            chain_2_residues: Row vector of selected residues in chain 2 (1 * num_selected_res_chain_2).

        Returns:
            The average pLDDT value of the interface residues.
        """
        plddt_sum = sum(chain_1_plddt[i] + chain_2_plddt[j] for i, j in zip(set(chain_1_residues), set(chain_2_residues)))
        total_num = len(set(chain_1_residues)) + len(set(chain_2_residues))
        return plddt_sum / total_num

    def update_df(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Update the DataFrame with reversed interfaces."""
        reversed_rows = []
        for interface in input_df['interface'].unique():
            try:
                chain_1, chain_2 = interface.split("_")
                reversed_interface = f"{chain_2}_{chain_1}"
                if reversed_interface not in input_df['interface'].values:
                    subdf = input_df[input_df['interface'] == interface].copy()
                    subdf['interface'] = reversed_interface
                    reversed_rows.append(subdf)
            except ValueError:
                logging.warning(f"Skipping interface due to unexpected format: {interface}")

        if reversed_rows:
            reversed_df = pd.concat(reversed_rows, ignore_index=True)
            output_df = pd.concat([input_df, reversed_df], ignore_index=True)
        else:
            output_df = input_df

        return output_df

    def calculate_binding_energy(self, chain_1_id: str, chain_2_id: str) -> float:
        """Calculate the binding energy of two chains using pyRosetta."""
        chain_1_structure, chain_2_structure = self.pdb[chain_1_id], self.pdb[chain_2_id]

        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp:
            pdbio = PDBIO()
            pdbio.set_structure(chain_1_structure)
            pdbio.set_structure(chain_2_structure)
            pdbio.save(temp.name)
            complex_pose = pose_from_pdb(temp.name)

        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp1, tempfile.NamedTemporaryFile(suffix='.pdb') as temp2:
            pdbio.set_structure(chain_1_structure)
            pdbio.save(temp1.name)
            chain_1_pose = pose_from_pdb(temp1.name)

            pdbio.set_structure(chain_2_structure)
            pdbio.save(temp2.name)
            chain_2_pose = pose_from_pdb(temp2.name)

        sfxn = get_score_function(True)
        complex_energy = sfxn(complex_pose)
        chain_1_energy = sfxn(chain_1_pose)
        chain_2_energy = sfxn(chain_2_pose)

        return complex_energy - chain_1_energy - chain_2_energy

    def run_and_summarize_pi_score(self, work_dir, pdb_path: str, surface_thres: int = 2, interface_name: str = "") -> pd.DataFrame:
        """Calculate all predicted models' PI scores and make a pandas DataFrame of the results."""
        try:
            command = (
                f"source activate pi_score && "
                f"export PYTHONPATH=/software:$PYTHONPATH && "
                f"python /software/pi_score/run_piscore_wc.py -p {pdb_path} "
                f"-o {work_dir} -s {surface_thres} -ps 10"
            )
            subprocess.run(command, shell=True, executable='/bin/bash', check=True)

            csv_files = [f for f in os.listdir(work_dir) if 'filter_intf_features' in f]
            pi_score_files = [f for f in os.listdir(work_dir) if 'pi_score_' in f]

            if not csv_files or not pi_score_files:
                raise FileNotFoundError("Required output files not found.")

            filtered_df = pd.read_csv(os.path.join(work_dir, csv_files[0]))

        except (subprocess.CalledProcessError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            logging.warning(f"PI score calculation has failed: {e}. Proceeding with the rest of the jobs")
            columns = [
                "pdb", "chains", "Num_intf_residues", "Polar", "Hydrophobhic", "Charged",
                "contact_pairs", "sc", "hb", "sb", "int_solv_en", "int_area", "pvalue", "pi_score"
            ]
            filtered_df = pd.DataFrame([{col: "None" for col in columns}])
            filtered_df['pi_score'] = "No interface detected"
            filtered_df['interface'] = interface_name

        filtered_df = self.update_df(filtered_df)

        try:
            with open(os.path.join(work_dir, pi_score_files[0]), 'r') as f:
                lines = [l for l in f.readlines() if "#" not in l]

            if lines:
                pi_score_df = pd.read_csv(os.path.join(work_dir, pi_score_files[0]))
            else:
                pi_score_df = pd.DataFrame.from_dict({"pi_score": ['SC: mds: too many atoms']})

            pi_score_df['interface'] = pi_score_df.get('chains', 'interface')
            filtered_df = pd.merge(filtered_df, pi_score_df, on=['interface'], how='left')

            filtered_df = filtered_df.drop(columns=["#PDB", "pdb", " pvalue", "chains", "predicted_class"], errors='ignore')

        except Exception as e:
            logging.warning(f"Error while merging PI score data: {e}")

        return filtered_df

    def calculate_pi_score(self, interface: int = "") -> pd.DataFrame:
        """Run the PI-score pipeline between the two chains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pi_score_output_dir = os.path.join(tmpdir, "pi_score_outputs")
            pi_score_df = self.run_and_summarize_pi_score(pi_score_output_dir, self.pdb_file_path, interface_name=interface)
        return pi_score_df

    def __call__(self, pae_mtx: np.ndarray, plddt: Dict[str, List[float]], cutoff: float = 12) -> Any:
        """
        Obtain interface residues and calculate average PAE and pLDDT of the interface residues.

        Args:
            pae_mtx: An array with PAE values from AlphaFold predictions (num_res * num_res).
            plddt: A dictionary containing pLDDT scores for each chain.
            cutoff: Cutoff value for determining whether two residues are interacting.

        Returns:
            A pandas DataFrame containing the calculated data.
        """
        output_df = pd.DataFrame()
        if not isinstance(self.chain_combinations, dict):
            print("Your PDB structure seems to be a monomeric structure. The program will stop.")
            import sys
            sys.exit()
        else:
            for k, v in self.chain_combinations.items():
                chain_1_id, chain_2_id = v
                chain_1_df = self.pdb_df[self.pdb_df['chain_id'] == chain_1_id]
                chain_2_df = self.pdb_df[self.pdb_df['chain_id'] == chain_2_id]

                pi_score_df = self.calculate_pi_score(interface=f"{chain_1_id}_{chain_2_id}")
                pi_score_df = self.update_df(pi_score_df)

                chain_1_plddt, chain_2_plddt = plddt[chain_1_id], plddt[chain_2_id]
                interface_residues = self.obtain_interface_residues(chain_1_df, chain_2_df, cutoff=cutoff)

                if interface_residues is not None:
                    average_interface_pae = self.calculate_average_pae(pae_mtx, interface_residues[0], interface_residues[1])
                    average_interface_plddt = self.calculate_average_plddt(chain_1_plddt, chain_2_plddt, interface_residues[0], interface_residues[1])
                    binding_energy = self.calculate_binding_energy(chain_1_id, chain_2_id)
                else:
                    average_interface_pae = "No interface residues detected"
                    average_interface_plddt = "No interface residues detected"
                    binding_energy = "None"

                other_measurements_df = pd.DataFrame.from_dict({
                    "average_interface_pae": [average_interface_pae],
                    "average_interface_plddt": [average_interface_plddt],
                    "binding_energy": [binding_energy],
                    "interface": [f"{chain_1_id}_{chain_2_id}"]
                })
                output_df = pd.concat([output_df, other_measurements_df])

        return pd.merge(output_df, pi_score_df, how='left', on='interface')
