""" 
    A script that survey all interfaces and obtain all the residues on the interface

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author:Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
from biopandas.pdb import PandasPdb
from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta.core.scoring import get_score_function
import pyrosetta; pyrosetta.init()
from Bio.PDB import PDBParser, PDBIO
import pandas as pd
from itertools import combinations
import numpy as np
from typing import Any, List, Dict
import tempfile 
import os
import subprocess

class PDBAnalyser:
    """
    A class that store pandas dataframe of all the information 
    of the residues in a PDB file 
    """

    def __init__(self, pdb_file_path: str) -> None:
        self.pdb_pandas = PandasPdb().read_pdb(pdb_file_path)
        self.pdb = PDBParser().get_structure("ranked_0",pdb_file_path)[0]
        self.pdb_df = self.pdb_pandas.df['ATOM']
        self.chain_combinations = {}
        self.get_all_combinations_of_chains()
        pass

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

    def obtain_interface_residues(self, chain_df_1: pd.DataFrame, chain_df_2: pd.DataFrame, cutoff: int = 12):
        """A method that get all the residues on the interface within the cutoff"""
        chain_1_CB_coords = self.retrieve_C_beta_coords(chain_df_1)
        chain_2_CB_coords = self.retrieve_C_beta_coords(chain_df_2)
        distance_mtx = np.sqrt(
            np.sum((chain_1_CB_coords[:, np.newaxis] - chain_2_CB_coords)**2, axis=-1))
        satisfied_residues_chain_1, satisfied_residues_chain_2 = np.where(
            distance_mtx < cutoff)
        if (len(satisfied_residues_chain_1) > 0) and (len(satisfied_residues_chain_2) > 0):
            return satisfied_residues_chain_1, satisfied_residues_chain_2

        else:
            print(f"No interface residues are found.")
            return None

    def calculate_average_pae(self, pae_mtx: np.ndarray,
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
        pae_sum = 0
        total_num = len(chain_1_residues)
        for i, j in zip(chain_1_residues, chain_2_residues):
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
            pdbio.save(file = temp1.name)
            chain_1_pose = pose_from_pdb(temp1.name)
            pdbio = PDBIO()
            pdbio.set_structure(chain_2_structure)
            pdbio.save(file = temp2.name)
            chain_2_pose = pose_from_pdb(temp2.name)
        
        sfxn = get_score_function(True)
        complex_energy = sfxn(complex_pose)
        chain_1_energy, chain_2_energy = sfxn(chain_1_pose), sfxn(chain_2_pose)
        return complex_energy - chain_1_energy - chain_2_energy 
    
    def run_and_summarise_pi_score(self, work_dir, pdb_path, 
                                   surface_thres: int = 2) -> pd.DataFrame:
        
        """A function to calculate all predicted models' pi_scores and make a pandas df of the results"""

        subprocess.run(
        f"source activate pi_score && export PYTHONPATH=/software:$PYTHONPATH && python /software/pi_score/run_piscore_wc.py -p {pdb_path} -o {work_dir} -s {surface_thres} -ps 10", shell=True, executable='/bin/bash')
        
        csv_files = [f for f in os.listdir(
            work_dir) if 'filter_intf_features' in f]
        pi_score_files = [f for f in os.listdir(work_dir) if 'pi_score_' in f]
        filtered_df = pd.read_csv(os.path.join(work_dir, csv_files[0]))

        if filtered_df.shape[0] == 0:
            for column in filtered_df.columns:
                filtered_df[column] = ["None"]
            filtered_df['pi_score'] = "No interface detected"
        else:
            with open(os.path.join(work_dir, pi_score_files[0]), 'r') as f:
                lines = [l for l in f.readlines() if "#" not in l]
                if len(lines) > 0:
                    pi_score = pd.read_csv(
                        os.path.join(work_dir, pi_score_files[0]))
                else:
                    pi_score = pd.DataFrame.from_dict(
                        {"pi_score": ['SC:  mds: too many atoms']})
                f.close()
            pi_score['interface'] = pi_score['chains']
            filtered_df = pd.merge(filtered_df, pi_score, on=['interface'])
            try:
                filtered_df = filtered_df.drop(
                    columns=["#PDB", "pdb", " pvalue", "chains", "predicted_class"])
            except:
                pass

        return filtered_df

    def calculate_pi_score(self, chain_1_id: str, chain_2_id: str) -> pd.DataFrame:
        """Run the PI-score pipeline between the 2 chains"""
        chain_1_structure, chain_2_structure = self.pdb[chain_1_id], self.pdb[chain_2_id]
        with tempfile.TemporaryDirectory() as tmpdir:
            pi_score_output_dir = os.path.join(tmpdir, "pi_score_outputs")
            pdb_output_file = os.path.join(tmpdir, "complex.pdb")
            pdbio = PDBIO()
            pdbio.set_structure(chain_1_structure)
            pdbio.set_structure(chain_2_structure)
            pdbio.save(pdb_output_file)
            pi_score_df = self.run_and_summarise_pi_score(pi_score_output_dir, pdb_output_file)
            pi_score_df['interface'] = f"{chain_1_id}_{chain_2_id}"
        return pi_score_df
    
    def __call__(self, pae_mtx: np.ndarray, plddt: Dict[str, List[float]], cutoff: float = 12) -> Any:
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
            for k, v in self.chain_combinations.items():
                chain_1_id, chain_2_id = v
                chain_1_df, chain_2_df = self.pdb_df[self.pdb_df['chain_id'] ==
                                                     chain_1_id], self.pdb_df[self.pdb_df['chain_id'] == chain_2_id]
                chain_1_plddt, chain_2_plddt = plddt[chain_1_id], plddt[chain_2_id]
                interface_residues = self.obtain_interface_residues(
                    chain_1_df, chain_2_df, cutoff=cutoff)
                if interface_residues is not None:
                    average_interface_pae = self.calculate_average_pae(pae_mtx,
                                                                       interface_residues[0], interface_residues[1])
                    average_interface_plddt = self.calculate_average_plddt(chain_1_plddt, chain_2_plddt,
                                                                           interface_residues[0], interface_residues[1])
                else:
                    average_interface_pae = "None"
                    average_interface_plddt = "None"
                binding_energy = self.calculate_binding_energy(chain_1_id, chain_2_id)
                other_measurements_df = pd.DataFrame.from_dict({
                    "average_interface_pae": [average_interface_pae],
                    "average_interface_plddt": [average_interface_plddt],
                    "binding_energy": [binding_energy]
                })
                pi_score_df = self.calculate_pi_score(chain_1_id, chain_2_id)
                output_df = pd.concat([output_df, pd.concat([other_measurements_df, pi_score_df], axis=1)])
        return output_df