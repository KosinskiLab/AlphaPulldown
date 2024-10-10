""" 
    A script that survey all interfaces and obtain all the residues on the interface

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author:Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
from absl import logging
import subprocess
import os
import tempfile
from typing import Any, List, Dict, Union
import numpy as np
from itertools import combinations, accumulate
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
    A class that store pandas dataframe of all the information 
    of the residues in a PDB file 
    """

    def __init__(self, pdb_file_path: str) -> None:
        self.pdb_file_path = pdb_file_path
        if not os.path.exists(os.path.join(pdb_file_path)):
            raise FileNotFoundError(f"ranked_0.pdb not found in this folder. Please remove this folder and rerun the pipeline again.")
        else:
            self.pdb_pandas = PandasPdb().read_pdb(pdb_file_path)
            self.pdb = PDBParser().get_structure("ranked_0", pdb_file_path)[0]
        
        self.pdb_df = self.pdb_pandas.df['ATOM']
        self.chain_combinations = {}
        self.get_all_combinations_of_chains()
        self.calculate_padding_of_chains()

    def __repr__(self) -> str:
        return ', '.join(f'{key}: {value}' for key, value in self.__dict__.items())

    def calculate_padding_of_chains(self):
        """
        A method that calculate how many residues need to be padded

        e.g. all residue indexes in the 1st chain do not need to be padded
        all residue indexes in the 2nd chain need to be padded with len(1st chain)
        all residue indexes in the 3nd chain need to be padded with len(1st chain) + len(2nd chain)
        """
        self.chain_cumsum = dict()
        residue_counts = dict()
        for chain in self.pdb:
            residue_counts[chain.id] = sum(1 for _ in chain.get_residues())
        cum_sum = list(accumulate([0] + list(residue_counts.values())[:-1]))
        for chain_id, cs in zip(residue_counts.keys(), cum_sum):
            self.chain_cumsum[chain_id] = cs

    def get_all_combinations_of_chains(self):
        """A method that get all the paiwise combination of chains"""
        unique_chain_ids = pd.unique(self.pdb_df.chain_id)
        if len(unique_chain_ids) > 1:
            chain_combinations = combinations(unique_chain_ids, 2)
            for idx, combo in enumerate(chain_combinations):
                self.chain_combinations.update(
                    {f"interface_{idx+1}": combo}
                )
        else:
            self.chain_combinations = unique_chain_ids

    def retrieve_C_beta_coords(self, chain_df: pd.DataFrame) -> np.ndarray:
        """
        A method that retrieve the x,y,z coords of the C beta atoms of one chain

        Args:
            chain_df : a pandas dataframe that belongs to one chain

        Return:
            a numpy array of all C-beta atoms x,y,z coordinates

        Note:
        Will retrieve C-alpha atom coords if the residue is a glycine
        """
        mask = (chain_df['atom_name'] == 'CB') | (
            (chain_df['residue_name'] == 'GLY') & (chain_df['atom_name'] == 'CA'))
        subdf = chain_df[mask]
        return subdf[['x_coord', 'y_coord', 'z_coord']].values

    def obtain_interface_residues(self, chain_df_1: pd.DataFrame, chain_df_2: pd.DataFrame, cutoff: int = 5) -> Union[None, set]:
        """A method that get all the residues on the interface within the cutoff"""
        chain_1_CB_coords = self.retrieve_C_beta_coords(chain_df_1)
        chain_2_CB_coords = self.retrieve_C_beta_coords(chain_df_2)

        distance_mtx = np.sqrt(
            np.sum((chain_1_CB_coords[:, np.newaxis] - chain_2_CB_coords)**2, axis=-1))
        satisfied_residues_chain_1, satisfied_residues_chain_2 = np.where(
            distance_mtx < cutoff)
        if (len(satisfied_residues_chain_1) > 0) and (len(satisfied_residues_chain_2) > 0):
            return (satisfied_residues_chain_1, satisfied_residues_chain_2)

        else:
            print(f"No interface residues are found.")
            return None

    def calculate_average_pae(self, pae_mtx: np.ndarray, chain_id_1: str, chain_id_2: str,
                              chain_1_residues: np.ndarray, chain_2_residues: np.ndarray) -> float:
        """
        A method to calculate average interface pae 

        Args:
        pae_mtx: An array with PAE values from AlphaFold predictions. shape: num_res * num_res
        chain_1_residues: a row vector. shape: 1 * num_selected_res_chain_1
        chain_2_residues: a row vector. shape: 1 * num_selected_res_chain_2

        Return:
        float: average PAE value of the interface residues
        """
        if chain_id_1 not in self.chain_cumsum or chain_id_2 not in self.chain_cumsum:
            raise KeyError(f"Chain ID {chain_id_1} or {chain_id_2} not found in chain_cumsum: {self.chain_cumsum}")

        pae_sum = 0
        chain_1_pad_num = self.chain_cumsum[chain_id_1]
        chain_2_pad_num = self.chain_cumsum[chain_id_2]
        satisfied_residues_chain_1 = [
            i + chain_1_pad_num for i in chain_1_residues]
        satisfied_residues_chain_2 = [
            i + chain_2_pad_num for i in chain_2_residues]
        total_num = len(chain_1_residues)
        for i, j in zip(satisfied_residues_chain_1, satisfied_residues_chain_2):
            pae_sum += pae_mtx[i, j]
            pae_sum += pae_mtx[j, i]

        return pae_sum / (2*total_num)

    def calculate_average_plddt(self, chain_1_plddt: List[float], chain_2_plddt: List[float],
                                chain_1_residues: np.ndarray, chain_2_residues: np.ndarray) -> float:
        """
        A method to calculate average interface plddt 

        Args:
        chain_1_plddt: a list of plddt values of the residues on chain_1
        chain_2_plddt: a list of plddt values of the residues on chain_2 
        chain_1_residues: a row vector. shape: 1 * num_selected_res_chain_1
        chain_2_residues: a row vector. shape: 1 * num_selected_res_chain_2

        Return:
        float: average PAE value of the interface residues
        """
        plddt_sum = 0
        total_num = len(set(chain_1_residues)) + len(set(chain_2_residues))
        for i, j in zip(set(chain_1_residues), set(chain_2_residues)):
            plddt_sum += chain_1_plddt[i]
            plddt_sum += chain_2_plddt[j]

        return plddt_sum / total_num
    
    def _default_dataframe(self) -> pd.DataFrame:
        """
        Returns a default DataFrame when PI score calculation fails.

        Returns:
            pd.DataFrame: Default DataFrame.
        """
        return pd.DataFrame({
            "pdb": ["None"], "Num_intf_residues": ["None"],
            "Polar": ["None"], "Hydrophobhic": ["None"], "Charged": ["None"],
            "contact_pairs": ["None"], " sc": ["None"], " hb": ["None"],
            " sb": ["None"], " int_solv_en": ["None"], " int_area": ["None"],
            "pvalue": ["None"], "pi_score": ["Calculation failed"], "interface": ["None"],
        })

    def _handle_pi_score_error(self, exception: Exception, command: List[str], error_message: str) -> pd.DataFrame:
        """Helper method for handling PI score errors with appropriate logging."""
        logging.error(f"PI score calculation failed: {exception}. Command: {command}. Error: {error_message}")
        return self._default_dataframe()

    def update_df(self, input_df):
        """
        This method update the dataframe with reversed interfaces
        
        e.g. the calculated columns of interface B_C will be copied to C_B
        """
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
        """Calculate binding energer of 2 chains using pyRosetta"""
        chain_1_structure, chain_2_structure = self.pdb[chain_1_id], self.pdb[chain_2_id]
        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp:
            pdbio = PDBIO()
            pdbio.set_structure(chain_1_structure)
            pdbio.set_structure(chain_2_structure)
            pdbio.save(temp.name)
            complex_pose = pose_from_pdb(temp.name)

        with tempfile.NamedTemporaryFile(suffix='.pdb') as temp1, tempfile.NamedTemporaryFile(suffix='.pdb') as temp2:
            pdbio = PDBIO()
            pdbio.set_structure(chain_1_structure)
            pdbio.save(file=temp1.name)
            chain_1_pose = pose_from_pdb(temp1.name)
            pdbio = PDBIO()
            pdbio.set_structure(chain_2_structure)
            pdbio.save(file=temp2.name)
            chain_2_pose = pose_from_pdb(temp2.name)

        sfxn = get_score_function(True)
        complex_energy = sfxn(complex_pose)
        chain_1_energy, chain_2_energy = sfxn(chain_1_pose), sfxn(chain_2_pose)
        return complex_energy - chain_1_energy - chain_2_energy

    def run_and_summarise_pi_score(
            self,
            work_dir: str,
            pdb_path: str,
            surface_thres: int = 2,
            interface_name: str = "",
            python_env: str = "pi_score",
            piscore_script_path: str = "/software/pi_score/run_piscore_wc.py",
            software_path: str = "/software",
    ) -> pd.DataFrame:
        """
        Calculates all predicted models' pi_scores and creates a pandas DataFrame of the results.

        Args:
            work_dir (str): Directory to store results.
            pdb_path (str): Path to the PDB file.
            surface_thres (int, optional): Surface threshold. Defaults to 2.
            interface_name (str, optional): Interface name. Defaults to "".
            python_env (str, optional): Python environment name. Defaults to "pi_score".
            piscore_script_path (str, optional): Path to the pi_score script. Defaults to "/software/pi_score/run_piscore_wc.py".
            software_path (str, optional): Path to the software directory. Defaults to "/software".

        Returns:
            pd.DataFrame: DataFrame containing the results.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use temp_dir for temporary storage
                # Set environment variables
                env = os.environ.copy()
                env["TMPDIR"] = temp_dir
                env["PISA_TMPDIR"] = temp_dir
                env["PISA_SESSIONDIR"] = temp_dir
                env["SC_TMPDIR"] = temp_dir
                env["SC_SESSIONDIR"] = temp_dir
                env["HOME"] = temp_dir  # Some utilities may use HOME
                # Ensure /tmp/root exists and is writable (alternative to setting envs)
                os.makedirs('/tmp/root', exist_ok=True)
                os.chmod('/tmp/root', 0o777)
                command = [
                    "bash", "-c",
                    f"source activate {python_env} && "
                    f"export PYTHONPATH={software_path}:$PYTHONPATH && "
                    f"python {piscore_script_path} -p {pdb_path} -o {temp_dir} -s {surface_thres}"
                ]
                # We decided to remove pi_score from the pipeline
                #result = subprocess.run(command, capture_output=True, text=True)
                #if result.returncode != 0:
                #    return self._handle_pi_score_error(result, command, result.stderr)

                # Read the results from the temporary directory
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                if not csv_files:
                    return self._default_dataframe()

                df_list = [pd.read_csv(os.path.join(temp_dir, csv_file)) for csv_file in csv_files]
                result_df = pd.concat(df_list, ignore_index=True)
                return result_df

        except Exception as e:
            logging.error(f"Error running pi_score: {e}")
            return self._default_dataframe()

    def calculate_pi_score(self, pi_score_output_dir: str, interface: int = "") -> pd.DataFrame:
        """Run the PI-score pipeline between the 2 chains"""
        pi_score_df = self.run_and_summarise_pi_score(
                pi_score_output_dir, self.pdb_file_path, interface_name=interface)
        return pi_score_df

    def __call__(self, pi_score_output_dir: str, pae_mtx: np.ndarray, plddt: Dict[str, List[float]], cutoff: float = 12) -> Any:
        """
        Obtain interface residues and calculate average PAE, average plDDT of the interface residues

        Args:
        pae_mtx: An array with PAE values from AlphaFold predictions. shape: num_res * num_res
        plddt: A dictionary that contains plddt score of each chain. 
        cutoff: cutoff value when determining whether 2 residues are interacting. deemed to be interacting 
        if the distance between two C-Beta is smaller than cutoff

        """
        output_df = pd.DataFrame()
        if type(self.chain_combinations) != dict:
            print(
                f"Your PDB structure seems to be a monomeric structure. The programme will stop.")
            import sys
            sys.exit()
        else:
            # Make sure pi_score_output_dir exists
            os.makedirs(pi_score_output_dir, exist_ok=True)
            for k, v in self.chain_combinations.items():
                chain_1_id, chain_2_id = v
                chain_1_df, chain_2_df = self.pdb_df[self.pdb_df['chain_id'] ==
                                                     chain_1_id], self.pdb_df[self.pdb_df['chain_id'] == chain_2_id]
                pi_score_df = self.calculate_pi_score(pi_score_output_dir,
                                                      interface=f"{chain_1_id}_{chain_2_id}")
                pi_score_df = self.update_df(pi_score_df)
                chain_1_plddt, chain_2_plddt = plddt[chain_1_id], plddt[chain_2_id]
                interface_residues = self.obtain_interface_residues(
                    chain_1_df, chain_2_df, cutoff=cutoff)
                if interface_residues is not None:
                    try:
                        average_interface_pae = self.calculate_average_pae(pae_mtx, chain_1_id, chain_2_id,
                                                                       interface_residues[0], interface_residues[1])
                    except Exception as e:
                        logging.error(f"Error while calculating the average PAE values of the interface residues: {e}")
                        average_interface_pae = 'None'
                    try:
                        binding_energy = self.calculate_binding_energy(
                        chain_1_id, chain_2_id)
                    except Exception as e:
                        logging.error(f"Error while calculating the binding energy using pyRosetta: {e}")
                        binding_energy = 'None'
                    try:
                        average_interface_plddt = self.calculate_average_plddt(chain_1_plddt, chain_2_plddt,
                                                                           interface_residues[0], interface_residues[1])
                    except Exception as e:
                        logging.error(f"Error while calculating the average pLDDT values of the interface residues: {e}")
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
