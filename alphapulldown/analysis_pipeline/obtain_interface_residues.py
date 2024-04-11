""" 
    A script that survey all interfaces and obtain all the residues on the interface

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author:Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
from biopandas.pdb import PandasPdb
import pandas as pd
from itertools import combinations
import numpy as np
from typing import Tuple

class PDBAnalyser:
    """
    A class that store pandas dataframe of all the information 
    of the residues in a PDB file 
    """

    def __init__(self, pdb_file_path: str) -> None:
        self.pdb = PandasPdb().read_pdb(pdb_file_path)
        self.pdb_df = self.pdb.df['ATOM']
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

    def retrieve_C_beta_coords(self, chain_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        A method that retrieve the x,y,z coords of the C beta atoms of one chain

        Args:
            chain_df : a pandas dataframe that belongs to one chain

        Return:
            a numpy array of all C-beta atoms x,y,z coordinates
            a sub-dataframe of all corresponding residue_name

        Note:
        Will retrieve C-alpha atom coords if the residue is a glycine
        """
        subdf = chain_df.loc[(chain_df['atom_name'] == 'CB') or (
            (chain_df['residue_name'] == 'GLY') and (chain_df['atom_name'] == 'CA'))]
        return subdf[['x_coord', 'y_coord', 'z_coord']].values, subdf['residue_number']

    def obtain_interface_residues(self, chain_df_1: pd.DataFrame, chain_df_2: pd.DataFrame, cutoff: int = 12):
        """A method that get all the residues on the interface within the cutoff"""
        chain_1_CB_coords, chain_1_residue_indexes = self.retrieve_C_beta_coords(chain_df_1)
        chain_2_CB_coords, chain_2_residue_indexes = self.retrieve_C_beta_coords(chain_df_2)
        distance_mtx = np.sqrt(
            np.sum((chain_1_CB_coords[:, np.newaxis] - chain_2_CB_coords)**2, axis=-1))
        