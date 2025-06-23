""" Create monomeric or multimeric objects with corresponding import features for the modelling backends

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
from absl import logging
logging.set_verbosity(logging.INFO)
import os
import subprocess
import numpy as np
from alphafold.data.tools import jackhmmer
from alphafold.data import parsers
from alphafold.data import pipeline_multimer
from alphafold.data import pipeline
from alphafold.data import msa_pairing
from alphafold.data import feature_processing
from pathlib import Path as plPath
from typing import List, Dict, Any, Tuple
from colabfold.batch import get_msa_and_templates, msa_to_str, build_monomer_feature, unserialize_msa
from alphapulldown.utils.multimeric_template_utils import (extract_multimeric_template_features_for_single_chain,
                                                     prepare_multimeric_template_meta_info)
from alphapulldown.utils.file_handling import temp_fasta_file

class MonomericObject:
    """
    monomeric objects

    Args
    description: description of the protein. By default is everything after the ">" symbol
    sequence: Sequence of the protein. String
    uniprot_runner: a uniprot_runner object
    """

    def __init__(self, description, sequence) -> None:
        self.description = description
        self.sequence = sequence
        self.feature_dict = dict()
        self._uniprot_runner = None
        pass

    @property
    def uniprot_runner(self):
        return self._uniprot_runner

    @uniprot_runner.setter
    def uniprot_runner(self, uniprot_runner):
        self._uniprot_runner = uniprot_runner

    @staticmethod
    def zip_msa_files(msa_output_path: str):
        """
        A static method that zip individual msa files within the given msa_output_path folder
        """
        def zip_individual_file(msa_file: plPath):
            assert os.path.exists(msa_file)
            cmd = f"gzip {msa_file}"
            _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        msa_file_endings = ['.a3m', '.fasta', '.sto', '.hmm']
        msa_files = [i for i in plPath(
            msa_output_path).iterdir() if i.suffix in msa_file_endings]
        if len(msa_files) > 0:
            for msa_file in msa_files:
                zip_individual_file(msa_file)

    @staticmethod
    def unzip_msa_files(msa_output_path: str):
        """
        A static method that unzip msa files in a folder if they exist
        """
        def unzip_individual_file(msa_file: plPath):
            assert os.path.exists(msa_file)
            cmd = f"gunzip {msa_file}"
            _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        zipped_files = [i for i in plPath(
            msa_output_path).iterdir() if i.suffix == '.gz']
        if len(zipped_files) > 0:
            for zipped_file in zipped_files:
                unzip_individual_file(zipped_file)
            return True  # means it has used zipped msa files
        else:
            return False

    @staticmethod
    def remove_msa_files(msa_output_path: str):
        """Remove MSA files and delete the directory if it’s empty."""
        msa_dir = plPath(msa_output_path)
        if not msa_dir.is_dir():
            return

        msa_suffixes = {'.a3m', '.fasta', '.sto', '.hmm'}

        # delete all matching MSA files
        for f in msa_dir.iterdir():
            if f.is_file() and f.suffix in msa_suffixes:
                f.unlink()

        # if the directory’s empty now, remove it
        if not any(msa_dir.iterdir()):
            msa_dir.rmdir()

    def all_seq_msa_features(
            self,
            input_fasta_path: str,
            uniprot_msa_runner: jackhmmer.Jackhmmer,
            output_dir: str = None,
            use_precomputed_msa: bool = False
    ) -> None:
        """Get MSA features for unclustered uniprot, for pairing later on."""

        logging.info(
            f"now going to run uniprot runner and save uniprot alignment in {output_dir}"
        )
        result = pipeline.run_msa_tool(
            uniprot_msa_runner,
            input_fasta_path,
            f"{output_dir}/uniprot_hits.sto",
            "sto",
            use_precomputed_msa,
        )

        msa = parsers.parse_stockholm(result["sto"])
        msa = msa.truncate(max_seqs=50000)
        all_seq_features = pipeline.make_msa_features([msa])
        valid_feats = msa_pairing.MSA_FEATURES + (
            "msa_species_identifiers",
            "msa_uniprot_accession_identifiers",
        )
        feats = {
            f"{k}_all_seq": v for k, v in all_seq_features.items() if k in valid_feats
        }
        return feats

    def make_features(
            self, pipeline, output_dir: str,
            use_precomputed_msa: bool = False,
            save_msa: bool = True, compress_msa_files: bool = False
    ):
        """a method that make msa and template features"""
        os.makedirs(os.path.join(output_dir, self.description), exist_ok=True)

        # firstly check if there are zipped msa files. unzip it if there is zipped msa files
        using_zipped_msa_files = MonomericObject.unzip_msa_files(
            os.path.join(output_dir, self.description))

        # Then start creating msa features
        msa_output_dir = os.path.join(output_dir, self.description)
        sequence_str = f">{self.description}\n{self.sequence}"
        logging.info(
            "will save msa files in :{}".format(msa_output_dir))
        plPath(msa_output_dir).mkdir(parents=True, exist_ok=True)
        with temp_fasta_file(sequence_str) as fasta_file:
            self.feature_dict = pipeline.process(
                fasta_file, msa_output_dir)
            pairing_results = self.all_seq_msa_features(
                fasta_file, self._uniprot_runner, msa_output_dir, use_precomputed_msa
            )
            self.feature_dict.update(pairing_results)
        
        # Add extra features to make it compatible with pickle features obtaiend from mmseqs2
        template_confidence_scores = self.feature_dict.get('template_confidence_scores', None)
        template_release_date = self.feature_dict.get('template_release_date', None)
        if template_confidence_scores is None:
            self.feature_dict.update(
                {'template_confidence_scores': np.array([[1] * len(self.sequence)])}
            )
        if template_release_date is None:
            self.feature_dict.update({"template_release_date" : np.array(['none'])})

        # post processing
        if (not save_msa) and (not use_precomputed_msa):
            logging.info("You chose not to save MSA files and not use precomputed MSA files.")     
            MonomericObject.remove_msa_files(msa_output_path=msa_output_dir)
        elif (not save_msa) and use_precomputed_msa:
            logging.info("You chose not to save MSA files but still want to use precomputed MSA files.")
            logging.info("MSA files are deleted because features were successfully generated.")
            MonomericObject.remove_msa_files(msa_output_path=msa_output_dir)
        if compress_msa_files:
            MonomericObject.zip_msa_files(msa_output_dir)
        if using_zipped_msa_files:
            MonomericObject.zip_msa_files(
                os.path.join(output_dir, self.description))


    def make_mmseq_features(
            self, DEFAULT_API_SERVER,
            output_dir=None,
            compress_msa_files=False,
            use_precomputed_msa=False,
    ):
        """
        A method to use mmseq_remote to calculate MSA.
        Modified from ColabFold to allow reusing precomputed MSAs if available.
        """
        os.makedirs(output_dir, exist_ok=True)
        using_zipped_msa_files = MonomericObject.unzip_msa_files(output_dir)

        msa_mode = "mmseqs2_uniref_env"
        keep_existing_results = True
        result_dir = output_dir
        use_templates = True
        result_zip = os.path.join(result_dir, self.description, ".result.zip")
        if keep_existing_results and plPath(result_zip).is_file():
            logging.info(f"Skipping {self.description} (result.zip)")

        a3m_path = os.path.join(result_dir, self.description + ".a3m")
        if use_precomputed_msa and os.path.isfile(a3m_path):
            logging.info(f"Using precomputed MSA from {a3m_path}")
            a3m_lines = [plPath(a3m_path).read_text()]
            (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality,
             template_features) = unserialize_msa(a3m_lines, self.sequence)
        else:
            logging.info("You chose to calculate MSA with mmseqs2.\nPlease also cite: "
                         "Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S and Steinegger M. "
                         "ColabFold: Making protein folding accessible to all. "
                         "Nature Methods (2022) doi: 10.1038/s41592-022-01488-1")
            (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality,
             template_features) = get_msa_and_templates(
                jobname=self.description,
                query_sequences=self.sequence,
                a3m_lines=None,
                result_dir=plPath(result_dir),
                msa_mode=msa_mode,
                use_templates=use_templates,
                custom_template_path=None,
                pair_mode="none",
                host_url=DEFAULT_API_SERVER,
                user_agent='alphapulldown'
            )
            msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
            plPath(a3m_path).write_text(msa)
            a3m_lines = [plPath(a3m_path).read_text()]
            if compress_msa_files:
                MonomericObject.zip_msa_files(os.path.join(result_dir, self.description))

        # Remove header lines starting with '#' if present.
        a3m_lines[0] = "\n".join([line for line in a3m_lines[0].splitlines() if not line.startswith("#")])
        self.feature_dict = build_monomer_feature(self.sequence, unpaired_msa[0], template_features[0])

        # Fix: Change tuple to list so that we can concatenate with msa_pairing.MSA_FEATURES.
        valid_feats = msa_pairing.MSA_FEATURES + ("msa_species_identifiers", "msa_uniprot_accession_identifiers")
        feats = {f"{k}_all_seq": v for k, v in self.feature_dict.items() if k in valid_feats}

        # Add default template confidence and release date if missing.
        if self.feature_dict.get('template_confidence_scores', None) is None:
            self.feature_dict.update({'template_confidence_scores': np.array([[1] * len(self.sequence)])})
        if self.feature_dict.get('template_release_date', None) is None:
            self.feature_dict.update({"template_release_date": ['none']})
        self.feature_dict.update(feats)

        if using_zipped_msa_files:
            MonomericObject.zip_msa_files(output_dir)


class ChoppedObject(MonomericObject):
    """A monomeric object chopped into specified regions."""

    def __init__(
        self,
        description: str,
        sequence: str,
        feature_dict: Dict[str, Any],
        regions: List[Tuple[int, int]],
    ) -> None:
        super().__init__(description, sequence)
        self.monomeric_description = description
        self.feature_dict = feature_dict
        self.regions = regions
        self.new_sequence = ""
        self.new_feature_dict: Dict[str, Any] = {}
        self.prepare_new_self_description()

    def prepare_new_self_description(self) -> None:
        """Builds a description string with appended region tags."""
        self.description = self.monomeric_description
        for start, end in self.regions:
            self.description += f"_{start}-{end}"

    def prepare_new_msa_feature(
        self, msa: Dict[str, Any], start: int, end: int
    ) -> Tuple[Dict[str, Any], str]:
        """
        Slice MSA features to [start:end) (1-based coords).
        Returns (new_feature_dict, sliced_sequence_str).
        """
        i0, i1 = start - 1, end
        length = i1 - i0

        aatype = msa["aatype"][i0:i1, :]
        bsr = msa["between_segment_residues"][i0:i1]
        residue_index = np.arange(1, length + 1, dtype=np.int32)

        seq0 = msa["sequence"][0]
        full_seq = seq0.decode() if isinstance(seq0, (bytes, bytearray)) else str(seq0)
        sliced_seq = full_seq[i0:i1]
        sequence_arr = np.array([sliced_seq.encode()])

        deletions       = msa["deletion_matrix_int"][ :, i0:i1]
        deletions_all   = msa["deletion_matrix_int_all_seq"][ :, i0:i1]
        msa_arr         = msa["msa"][ :, i0:i1]
        msa_all         = msa["msa_all_seq"][ :, i0:i1]
        num_alignments  = np.full(length, msa_arr.shape[0], dtype=np.int32)
        species         = msa["msa_species_identifiers"]
        species_all     = msa["msa_species_identifiers_all_seq"]
        domain_name     = msa["domain_name"]

        new_msa_feature = {
            "aatype": aatype,
            "between_segment_residues": bsr,
            "domain_name": domain_name,
            "residue_index": residue_index,
            "seq_length": np.full(length, length, dtype=np.int32),
            "sequence": sequence_arr,
            "deletion_matrix_int": deletions,
            "msa": msa_arr,
            "num_alignments": num_alignments,
            "msa_species_identifiers": species,
            "msa_all_seq": msa_all,
            "deletion_matrix_int_all_seq": deletions_all,
            "msa_species_identifiers_all_seq": species_all,
        }
        return new_msa_feature, sliced_seq

    def prepare_new_template_feature(
        self, tmpl: Dict[str, Any], start: int, end: int
    ) -> Dict[str, Any]:
        """
        Slice template features to [start:end) (1-based coords).
        If no templates are present, returns empty arrays.
        """
        i0, i1 = start - 1, end
        length = i1 - i0

        if "template_aatype" not in tmpl:
            logging.info("No templates for %s", self.description)
            return {
                "template_aatype":             np.empty((0, length, 22), dtype=np.float32),
                "template_all_atom_masks":     np.empty((0, length, 37), dtype=np.float32),
                "template_all_atom_positions": np.empty((0, length, 37, 3), dtype=np.float32),
                "template_domain_names":       np.array([], dtype=object),
                "template_sequence":           np.array([], dtype=object),
                "template_sum_probs":          np.empty((0,), dtype=np.float32),
                "template_confidence_scores":  np.empty((0, length), dtype=np.float32),
                "template_release_date":       np.array([], dtype=object),
            }

        new_taa   = tmpl["template_aatype"][:, i0:i1, :]
        new_masks = tmpl["template_all_atom_masks"][ :, i0:i1, :]
        new_pos   = tmpl["template_all_atom_positions"][ :, i0:i1, :, :]

        domains      = tmpl.get("template_domain_names", np.array([], dtype=object))
        release_dates= tmpl.get("template_release_date", np.array([], dtype=object))
        sum_probs    = tmpl.get("template_sum_probs", np.empty((0,), dtype=np.float32))

        if "template_confidence_scores" in tmpl:
            conf = tmpl["template_confidence_scores"][ :, i0:i1]
        else:
            conf = np.ones((new_taa.shape[0], length), dtype=np.float32)

        seqs = []
        for s in tmpl.get("template_sequence", []):
            s_str = s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
            seqs.append(s_str[i0:i1].encode())
        seqs_arr = np.array(seqs, dtype=object)

        return {
            "template_aatype":             new_taa,
            "template_all_atom_masks":     new_masks,
            "template_all_atom_positions": new_pos,
            "template_domain_names":       domains,
            "template_sequence":           seqs_arr,
            "template_sum_probs":          sum_probs,
            "template_confidence_scores":  conf,
            "template_release_date":       release_dates,
        }

    def prepare_individual_sliced_feature_dict(
        self, feats: Dict[str, Any], start: int, end: int
    ) -> Dict[str, Any]:
        """Combine MSA- and template-based slicing for one region."""
        msa_feat, seq_fragment = self.prepare_new_msa_feature(feats, start, end)
        tmpl_feat              = self.prepare_new_template_feature(feats, start, end)
        self.new_sequence     += seq_fragment
        return {**tmpl_feat, **msa_feat}

    def concatenate_sliced_feature_dict(
        self, slice_dicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge sliced feature dicts into one continuous feature_dict,
        using explicit axis mapping per key.
        """
        out = slice_dicts[0].copy()
        num_align = out["num_alignments"][0]

        # specify concat axis for particular 2D/ND keys
        axis_map = {
            # MSA features (seqs x residues)
            "deletion_matrix_int":            1,
            "deletion_matrix_int_all_seq":    1,
            "msa":                            1,
            "msa_all_seq":                    1,
            # template confidence (templates x residues)
            "template_confidence_scores":     1,
        }

        # merge template confidence separately
        if "template_confidence_scores" in out:
            out["template_confidence_scores"] = np.concatenate(
                [sd["template_confidence_scores"] for sd in slice_dicts],
                axis=axis_map["template_confidence_scores"],
            )

        skip = {
            "template_domain_names", "template_sequence",
            "template_sum_probs", "template_release_date",
            "msa_species_identifiers", "msa_species_identifiers_all_seq",
        }

        for key, base_arr in list(out.items()):
            if key in skip or key == "template_confidence_scores":
                continue
            arrs = [sd[key] for sd in slice_dicts]
            ndim = base_arr.ndim
            if ndim == 1:
                out[key] = np.concatenate(arrs, axis=0)
            elif ndim == 2:
                ax = axis_map.get(key, 0)
                out[key] = np.concatenate(arrs, axis=ax)
            else:
                # for 3D+ arrays (e.g., template_aatype, masks, positions)
                # residues axis=1, so concat on axis 1
                out[key] = np.concatenate(arrs, axis=1)

        total_len = len(self.new_sequence)
        out["residue_index"]    = np.arange(1, total_len + 1, dtype=np.int32)
        out["seq_length"]       = np.full(total_len, total_len, dtype=np.int32)
        out["num_alignments"]   = np.full(total_len, num_align, dtype=np.int32)
        out["sequence"]         = np.array([self.new_sequence.encode()])

        return out

    def prepare_final_sliced_feature_dict(self) -> None:
        """
        Chop self.feature_dict into regions, update self.sequence & feature_dict.
        """
        self.new_sequence = ""
        slices = [
            self.prepare_individual_sliced_feature_dict(
                self.feature_dict, start, end
            ) for start, end in self.regions
        ]

        final = slices[0] if len(slices) == 1 else self.concatenate_sliced_feature_dict(slices)

        self.sequence = self.new_sequence
        self.feature_dict = final
        self.new_feature_dict = {}



class MultimericObject:
    """
    multimeric objects with features for multimer

    Args
    index: assign a unique index ranging from 0 just to identify different multimer jobs
    interactors: individual interactors that are to be concatenated
    pair_msa: boolean, tells the programme whether to pair MSA or not
    multimeric_template: boolean, tells the programme whether use multimeric templates or not
    multimeric_template_meta_data: a csv with the format {"monomer_A":{"xxx.cif":"chainID"},"monomer_B":{"yyy.cif":"chainID"}}
    multimeric_template_dir: a directory where all the multimeric templates mmcifs files are stored
    """

    def __init__(self, interactors: list, pair_msa: bool = True, 
                 multimeric_template: bool = False,
                 multimeric_template_meta_data: str = None,
                 multimeric_template_dir:str = None) -> None:
        self.description = ""
        self.interactors = interactors
        self.build_description_monomer_mapping()
        self.pair_msa = pair_msa
        self.multimeric_template = multimeric_template
        self.chain_id_map = dict()
        self.input_seqs = []
        self.multimeric_template_dir = multimeric_template_dir
        self.create_output_name()

        if multimeric_template_meta_data is not None:
            self.multimeric_template_meta_data = prepare_multimeric_template_meta_info(multimeric_template_meta_data,
                                                                                       self.multimeric_template_dir)
            
        if self.multimeric_template:
            self.create_multimeric_template_features()
        self.create_all_chain_features()
        pass
    
    def build_description_monomer_mapping(self):
        """This method constructs a dictionary {description: monomer}"""
        self.monomers_mapping = {m.description: m for m in self.interactors}

    def create_output_name(self):
        """a method to create output name"""
        for i in range(len(self.interactors)):
            if i == 0:
                self.description += f"{self.interactors[i].description}"
            else:
                self.description += f"_and_{self.interactors[i].description}"

    def create_chain_id_map(self):
        """a method to create chain id"""
        multimer_sequence_str = ""
        for interactor in self.interactors:
            multimer_sequence_str = (
                multimer_sequence_str
                + f">{interactor.description}\n{interactor.sequence}\n"
            )
        self.input_seqs, input_descs = parsers.parse_fasta(
            multimer_sequence_str)
        self.chain_id_map = pipeline_multimer._make_chain_id_map(
            sequences=self.input_seqs, descriptions=input_descs
        )

    def save_binary_matrix(self, matrix, file_path):
        from PIL import Image, ImageDraw, ImageFont
        height, width = matrix.shape
        image_data = np.zeros((height, width, 3), dtype=np.uint8)
        image_data[matrix == 1] = [255, 0, 0]  # Set ones as red
        image_data[matrix == 0] = [0, 0, 255]  # Set zeros as blue

        image = Image.fromarray(image_data)

        draw = ImageDraw.Draw(image)
        font_size = 16
        # Try to use Arial font, if not available, use default font
        try:
            font = ImageFont.truetype("Arial", font_size)
        except OSError:
            font = ImageFont.load_default()
        for col in range(width - 1):
            if matrix[:, col].any() != matrix[:, col + 1].any():
                text = str(col + 1)
                text_width, text_height = draw.textsize(text, font=font)
                x = (col + 0.5) * image.width / width - text_width / 2
                y = image.height - text_height
                # Set text fill color to black
                draw.text((x, y), text, font=font, fill=(0, 0, 0))

        image.save(file_path)

    def create_multichain_mask(self):
        """a method to create pdb_map for further multitemplate modeling"""
        pdb_map = []
        no_gap_map = []
        for interactor in self.interactors:
            temp_length = len(interactor.sequence)
            pdb_map.extend(
                [interactor.feature_dict['template_domain_names'][0]] * temp_length)
            has_no_gaps = [True] * temp_length
            # for each template in the interactor, check for gaps in sequence
            for template_sequence in interactor.feature_dict['template_sequence']:
                is_not_gap = [
                    s != '-' for s in template_sequence.decode("utf-8").strip()]
                # False if any of the templates has a gap in this position
                has_no_gaps = [a and b for a,
                               b in zip(has_no_gaps, is_not_gap)]
            no_gap_map.extend(has_no_gaps)
        multichain_mask = np.zeros((len(pdb_map), len(pdb_map)), dtype=int)
        for index1, id1 in enumerate(pdb_map):
            for index2, id2 in enumerate(pdb_map):
                # and (no_gap_map[index1] and no_gap_map[index2]):
                if (id1[:4] == id2[:4]):
                    multichain_mask[index1, index2] = 1
        # DEBUG
        #self.save_binary_matrix(multichain_mask, "multichain_mask.png")
        return multichain_mask
    
    def create_multimeric_template_features(self):
        """A method of creating multimeric template features"""
        if self.multimeric_template_dir is None or not hasattr(self,"multimeric_template_meta_data"):
            logging.warning(f"""
You chose to use multimeric template modelling 
but did not give path to multimeric_template_dir or the descrption File. 
This suggests you have already created template features from your desired multimeric models when runnign 
create_individual_features.py 
                            """)
            pass
        else:
            for monomer_name in self.multimeric_template_meta_data:
                for k,v in self.multimeric_template_meta_data[monomer_name].items():
                    curr_monomer = self.monomers_mapping[monomer_name]
                    assert k.endswith(".cif"), "The multimeric template file you provided does not seem to be a mmcif file. Please check your format and make sure it ends with .cif"
                    assert os.path.exists(os.path.join(self.multimeric_template_dir,k)), f"Your provided {k} cannot be found in: {self.multimeric_template_dir}. Abort"
                    pdb_id = k.split('.cif')[0]
                    multimeric_template_features = extract_multimeric_template_features_for_single_chain(query_seq=curr_monomer.sequence,
                                                                                                        pdb_id=pdb_id,chain_id=v,
                                                                                                        mmcif_file=os.path.join(self.multimeric_template_dir,k))
                    curr_monomer.feature_dict.update(multimeric_template_features.features)
            

    @staticmethod
    def remove_all_seq_features(np_chain_list: List[Dict]) -> List[Dict]:
        """
        Because AlphaPulldown will calculate Uniprot MSA during the feature creating step automatically,
        thus, if the user wants to model multimeric structures without paired MSAs, this method will be called 
        within the pair_and_merge method 

        Args:
        np_chain_list: A list of dictionary that corresponds to individual chain's feature matrices


        Return:
        A new list of chain features without all these xxx_all_seq features
        """
        output_list = []
        for feat_dict in np_chain_list:
            new_chain = {k: v for k, v in feat_dict.items()
                         if '_all_seq' not in k}
            output_list.append(new_chain)
        return output_list

    def pair_and_merge(self, all_chain_features):
        """merge all chain features"""
        feature_processing.process_unmerged_features(all_chain_features)
        MAX_TEMPLATES = 4
        MSA_CROP_SIZE = 2048
        np_chains_list = list(all_chain_features.values())
        pair_msa_sequences = self.pair_msa and not feature_processing._is_homomer_or_monomer(
            np_chains_list)
        logging.debug(f"pair_msa_sequences is type : {type(pair_msa_sequences)} value: {pair_msa_sequences}")
        if pair_msa_sequences:
            np_chains_list = msa_pairing.create_paired_features(
                chains=np_chains_list)
            np_chains_list = msa_pairing.deduplicate_unpaired_sequences(
                np_chains_list)
        else:
            np_chains_list = MultimericObject.remove_all_seq_features(
                np_chains_list)
        np_chains_list = feature_processing.crop_chains(
            np_chains_list,
            msa_crop_size=MSA_CROP_SIZE,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=MAX_TEMPLATES,
        )
        np_example = msa_pairing.merge_chain_features(
            np_chains_list=np_chains_list,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=MAX_TEMPLATES,
        )
        np_example = feature_processing.process_final(np_example)
        return np_example

    def create_all_chain_features(self):
        """
        concatenate and create all chain features

        Args
        uniprot_runner: a jackhammer runner with path to the uniprot database
        msa_pairing: boolean pairs msas or not
        """
        if self.multimeric_template:
            logging.info("Running in TrueMultimer mode")
            self.multichain_mask = self.create_multichain_mask()
        self.create_chain_id_map()
        all_chain_features = {}
        sequence_features = {}
        count = 0  # keep the index of input_seqs
        for chain_id, fasta_chain in self.chain_id_map.items():
            chain_features = self.interactors[count].feature_dict
            chain_features = pipeline_multimer.convert_monomer_features(
                chain_features, chain_id
            )
            all_chain_features[chain_id] = chain_features
            sequence_features[fasta_chain.sequence] = chain_features
            count += 1
        self.all_chain_features = pipeline_multimer.add_assembly_features(
            all_chain_features
        )

        self.feature_dict = self.pair_and_merge(
            all_chain_features=self.all_chain_features
        )
        self.feature_dict = pipeline_multimer.pad_msa(self.feature_dict, 512)

        ## make integer to np.array
        #for k in ['num_alignments']:
        #    self.feature_dict[k] = np.array([self.feature_dict[k]])
        if self.multimeric_template:
            self.feature_dict['template_sequence'] = []
            self.feature_dict['multichain_mask'] = self.multichain_mask
            # save used templates
            for i in self.interactors:
                logging.info(
                    "Used multimeric templates for protein {}".format(i.description))
                logging.info(i.feature_dict['template_domain_names'])
