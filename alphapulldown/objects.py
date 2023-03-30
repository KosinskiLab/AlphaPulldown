#
# Author Dingquan Yu
# scripts to create objects (e.g. monomeric object, multimeric objects)
#
# #
import logging
import tempfile
import os
import contextlib
from tracemalloc import start
import numpy as np
from alphafold.data import parsers
from alphafold.data import pipeline_multimer
from alphafold.data import pipeline
from alphafold.data import msa_pairing
from alphafold.data import feature_processing
from alphafold.data import templates
from pathlib import Path as plPath
from alphafold.data.tools import hhsearch
from colabfold.batch import get_queries,unserialize_msa,get_msa_and_templates,msa_to_str,build_monomer_feature,parse_fasta

@contextlib.contextmanager
def temp_fasta_file(sequence_str):
    """function that create temp file"""
    with tempfile.NamedTemporaryFile("w", suffix=".fasta") as fasta_file:
        fasta_file.write(sequence_str)
        fasta_file.seek(0)
        yield fasta_file.name


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

    def all_seq_msa_features(
        self,
        input_fasta_path,
        uniprot_msa_runner,
        save_msa,
        output_dir=None,
        use_precomuted_msa=False,
    ):
        """Get MSA features for unclustered uniprot, for pairing later on."""
        if not use_precomuted_msa:
            if not save_msa:
                with tempfile.TemporaryDirectory() as tempdir:
                    logging.info("now going to run uniprot runner")
                    result = pipeline.run_msa_tool(
                        uniprot_msa_runner,
                        input_fasta_path,
                        f"{tempdir}/uniprot.sto",
                        "sto",
                        use_precomuted_msa,
                    )
            elif save_msa and (output_dir is not None):
                logging.info(
                    f"now going to run uniprot runner and save uniprot alignment in {output_dir}"
                )
                result = pipeline.run_msa_tool(
                    uniprot_msa_runner,
                    input_fasta_path,
                    f"{output_dir}/uniprot.sto",
                    "sto",
                    use_precomuted_msa,
                )
        else:
            result = pipeline.run_msa_tool(
                uniprot_msa_runner,
                input_fasta_path,
                f"{output_dir}/uniprot.sto",
                "sto",
                use_precomuted_msa,
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
        self, pipeline, output_dir=None, use_precomputed_msa=False, save_msa=True
    ):
        """a method that make msa and template features"""
        if not use_precomputed_msa:
            if not save_msa:
                """this means no msa files are going to be saved"""
                logging.info("You have chosen not to save msa output files")
                sequence_str = f">{self.description}\n{self.sequence}"
                with temp_fasta_file(
                    sequence_str
                ) as fasta_file, tempfile.TemporaryDirectory() as tmpdirname:
                    self.feature_dict = pipeline.process(
                        input_fasta_path=fasta_file, msa_output_dir=tmpdirname
                    )
                    pairing_results = self.all_seq_msa_features(
                        fasta_file, self._uniprot_runner, save_msa
                    )
                    self.feature_dict.update(pairing_results)

            else:
                """this means no precomputed msa available and will save output msa files"""
                msa_output_dir = os.path.join(output_dir, self.description)
                sequence_str = f">{self.description}\n{self.sequence}"
                logging.info("will save msa files in :{}".format(msa_output_dir))
                plPath(msa_output_dir).mkdir(parents=True, exist_ok=True)
                with temp_fasta_file(sequence_str) as fasta_file:
                    self.feature_dict = pipeline.process(fasta_file, msa_output_dir)
                    pairing_results = self.all_seq_msa_features(
                        fasta_file, self._uniprot_runner, save_msa, msa_output_dir
                    )
                    self.feature_dict.update(pairing_results)
        else:
            """This means precomputed msa files are available"""
            msa_output_dir = os.path.join(output_dir, self.description)
            plPath(msa_output_dir).mkdir(parents=True, exist_ok=True)
            logging.info(
                "use precomputed msa. Searching for msa files in :{}".format(
                    msa_output_dir
                )
            )
            sequence_str = f">{self.description}\n{self.sequence}"
            with temp_fasta_file(sequence_str) as fasta_file:
                self.feature_dict = pipeline.process(fasta_file, msa_output_dir)
                pairing_results = self.all_seq_msa_features(
                    fasta_file,
                    self._uniprot_runner,
                    save_msa,
                    msa_output_dir,
                    use_precomuted_msa=True,
                )
                self.feature_dict.update(pairing_results)

    def mk_template(self,a3m_lines,pdb70_database_path,template_mmcif_dir,query_sequence,max_template_date, obsolete_pdbs_path=None):
        """
        Overwrite ColabFold's original mk_template to incorporate max_template data argument
        from the command line input.
        Modified from ColabFold: https://github.com/sokrypton/ColabFold

        Args
        template_path should be the same as FLAG.data_dir
        """
        template_featuriser = templates.HhsearchHitFeaturizer(
        mmcif_dir=f"{template_mmcif_dir}",
        max_template_date=max_template_date,
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=obsolete_pdbs_path,
        )
        hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{pdb70_database_path}"]
    )

        hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
        hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
        templates_result = template_featuriser.get_templates(
            query_sequence=query_sequence, hits=hhsearch_hits
        )
        return dict(templates_result.features)

    def make_mmseq_features(
        self,DEFAULT_API_SERVER,pdb70_database_path,template_mmcif_dir,max_template_date,output_dir=None,obsolete_pdbs_path=None
    ):
        """
        A method to use mmseq_remote to calculate msa
        Modified from ColabFold: https://github.com/sokrypton/ColabFold
        """


        logging.info("You chose to calculate MSA with mmseq2")
        logging.info("Debug version")
        msa_mode = "MMseqs2 (UniRef+Environmental)"
        keep_existing_results=True
        result_dir = output_dir
        use_templates=False
        result_zip = os.path.join(result_dir,self.description,".result.zip")
        if keep_existing_results and plPath(result_zip).is_file():
            logging.info(f"Skipping {self.description} (result.zip)")

        logging.info(f"looking for possible precomputed a3m at {os.path.join(result_dir,self.description+'.a3m')}")
        try:
            logging.info(f"input is {os.path.join(result_dir,self.description+'.a3m')}")
            input_path=os.path.join(result_dir,self.description+'.a3m')
            a3m_lines = [plPath(input_path).read_text()]
            logging.info(f"Finished parsing the precalculated a3m_file\nNow will search for template")
        except:
            a3m_lines=None

        if a3m_lines is not None:
                (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features,
                ) = unserialize_msa(a3m_lines, self.sequence)

        else:
            (
                unpaired_msa,
                paired_msa,
                query_seqs_unique,
                query_seqs_cardinality,
                template_features,
            ) = get_msa_and_templates(
                self.description,
                self.sequence,
                plPath(result_dir),
                msa_mode,
                use_templates,
                custom_template_path=None,
                pair_mode="none",
                host_url=DEFAULT_API_SERVER,
            )
            msa = msa_to_str(
                unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality
            )
            plPath(os.path.join(result_dir,self.description + ".a3m")).write_text(msa)
            a3m_lines=[plPath(os.path.join(result_dir,self.description + ".a3m")).read_text()]
        # unserialize_msa was from colabfold.batch and originally will only create mock template features
        # below will search against pdb70 database using hhsearch and create real template features
        logging.info("will search for templates in local template database")
        template_features = [self.mk_template(a3m_lines[0],
        pdb70_database_path,template_mmcif_dir,query_sequence=self.sequence,max_template_date=max_template_date, obsolete_pdbs_path=obsolete_pdbs_path)]
        self.feature_dict = build_monomer_feature(self.sequence,unpaired_msa[0],template_features[0])


        # update feature_dict with
        valid_feats = msa_pairing.MSA_FEATURES + (
            "msa_species_identifiers",
            "msa_uniprot_accession_identifiers",
        )
        feats = {
            f"{k}_all_seq": v for k, v in self.feature_dict.items() if k in valid_feats
        }
        self.feature_dict.update(feats)

class ChoppedObject(MonomericObject):
    """chopped monomeric objects"""

    def __init__(self, description, sequence, feature_dict, regions) -> None:
        super().__init__(description, sequence)
        self.feature_dict = feature_dict
        self.regions = regions
        self.new_sequence = ""
        self.new_feature_dict = dict()
        self.description = description
        self.prepare_new_self_description()

    def prepare_new_self_description(self):
        """prepare self_description of chopped proteins"""
        for r in self.regions:
            self.description += f"_{r[0]}-{r[1]}"

    def prepare_new_msa_feature(self, msa_feature, start_point, end_point):
        """
        prepare msa features

        Args:
        msa_feature is actually the full feature_dict
        """
        start_point = start_point - 1
        length = end_point - start_point
        new_seq_length = np.array([length] * length)
        new_aa_type = msa_feature["aatype"][start_point:end_point, :]
        new_between_segment_residue = msa_feature["between_segment_residues"][
            start_point:end_point
        ]
        new_domain_name = msa_feature["domain_name"]
        new_residue_index = msa_feature["residue_index"][start_point:end_point]
        new_sequence = np.array([msa_feature["sequence"][0][start_point:end_point]])
        new_deletion_mtx = msa_feature["deletion_matrix_int"][:, start_point:end_point]
        new_deletion_mtx_all_seq = msa_feature["deletion_matrix_int_all_seq"][
            :, start_point:end_point
        ]
        new_msa = msa_feature["msa"][:, start_point:end_point]
        new_msa_all_seq = msa_feature["msa_all_seq"][:, start_point:end_point]
        new_num_alignments = np.array([msa_feature["msa"].shape[0]] * length)
        new_uniprot_species = msa_feature["msa_species_identifiers"]
        new_uniprot_species_all_seq = msa_feature["msa_species_identifiers_all_seq"]

        new_msa_feature = {
            "aatype": new_aa_type,
            "between_segment_residues": new_between_segment_residue,
            "domain_name": new_domain_name,
            "residue_index": new_residue_index,
            "seq_length": new_seq_length,
            "sequence": new_sequence,
            "deletion_matrix_int": new_deletion_mtx,
            "msa": new_msa,
            "num_alignments": new_num_alignments,
            "msa_species_identifiers": new_uniprot_species,
            "msa_all_seq": new_msa_all_seq,
            "deletion_matrix_int_all_seq": new_deletion_mtx_all_seq,
            "msa_species_identifiers_all_seq": new_uniprot_species_all_seq,
        }

        return new_msa_feature, new_sequence[0].decode("utf-8")

    def prepare_new_template_feature_dict(
        self, template_feature, start_point, end_point
    ):
        """
        prepare template  features

        Args:
        template_feature is actually the full feature_dict
        """
        start_point = start_point - 1
        new_template_aatype = template_feature["template_aatype"][
            :, start_point:end_point, :
        ]
        new_template_all_atom_masks = template_feature["template_all_atom_masks"][
            :, start_point:end_point, :
        ]
        new_template_all_atom_positions = template_feature[
            "template_all_atom_positions"
        ][:, start_point:end_point, :, :]
        new_template_domain_names = template_feature["template_domain_names"]
        new_template_sequence = template_feature["template_sequence"]
        new_template_sum_probs = template_feature["template_sum_probs"]

        new_template_feature = {
            "template_aatype": new_template_aatype,
            "template_all_atom_masks": new_template_all_atom_masks,
            "template_all_atom_positions": new_template_all_atom_positions,
            "template_domain_names": new_template_domain_names,
            "template_sequence": new_template_sequence,
            "template_sum_probs": new_template_sum_probs,
        }
        return new_template_feature

    def prepare_individual_sliced_feature_dict(
        self, feature_dict, start_point, end_point
    ):
        """combine prepare_new_template_feature_dict and prepare_new_template_feature_dict"""
        new_msa_feature, new_sequence = self.prepare_new_msa_feature(
            feature_dict, start_point, end_point
        )
        sliced_feature_dict = {
            **self.prepare_new_template_feature_dict(
                feature_dict, start_point, end_point
            ),
            **new_msa_feature,
        }
        self.new_sequence = self.new_sequence + new_sequence
        return sliced_feature_dict

    def concatenate_sliced_feature_dict(self, feature_dicts: list):
        """concatenate regions such as 1-200 + 500-600"""
        output_dict = feature_dicts[0]
        new_sequence_length = feature_dicts[0]["seq_length"][0]
        num_alignment = feature_dicts[0]["num_alignments"][0]
        for sub_dict in feature_dicts[1:]:
            new_sequence_length += sub_dict["seq_length"][0]
            for k in feature_dicts[0].keys():
                if sub_dict[k].ndim > 1:
                    if k == "aatype":
                        output_dict[k] = np.concatenate(
                            (output_dict[k], sub_dict[k]), axis=0
                        )
                    elif "msa_species_identifiers" in k:
                        pass
                    else:
                        output_dict[k] = np.concatenate(
                            (output_dict[k], sub_dict[k]), axis=1
                        )
                elif sub_dict[k].ndim == 1:
                    if "msa_species_identifiers" in k:
                        pass
                    else:
                        output_dict[k] = np.concatenate(
                            (output_dict[k], sub_dict[k]), axis=0
                        )

        update_dict = {
            "seq_length": np.array([len(self.new_sequence)] * len(self.new_sequence)),
            "num_alignments": np.array([num_alignment] * len(self.new_sequence)),
            "sequence": np.array([self.new_sequence.encode()]),
        }
        output_dict.update(update_dict)
        return output_dict

    def prepare_final_sliced_feature_dict(self):
        """prepare final features for the corresponding region"""
        if len(self.regions) == 1:
            start_point = self.regions[0][0]
            end_point = self.regions[0][1]
            self.new_feature_dict = self.prepare_individual_sliced_feature_dict(
                self.feature_dict, start_point, end_point
            )
            self.sequence = self.new_sequence
            self.feature_dict = self.new_feature_dict
            self.new_feature_dict = dict()
        elif len(self.regions) > 1:
            temp_feature_dicts = []
            for sub_region in self.regions:
                start_point = sub_region[0]
                end_point = sub_region[1]
                curr_feature_dict = self.prepare_individual_sliced_feature_dict(
                    self.feature_dict, start_point, end_point
                )
                temp_feature_dicts.append(curr_feature_dict)
            self.sequence = self.new_sequence
            self.new_feature_dict = self.concatenate_sliced_feature_dict(
                temp_feature_dicts
            )
            self.feature_dict = self.new_feature_dict
            self.new_feature_dict = dict()
            del temp_feature_dicts


class MultimericObject:
    """
    multimeric objects with features for multimer

    Args
    index: assign a unique index ranging from 0 just to identify different multimer jobs
    interactors: individual interactors that are to be concatenated
    """

    def __init__(self, interactors: list, pair_msa: bool = True) -> None:
        self.description = ""
        self.interactors = interactors
        self.pair_msa = pair_msa
        self.chain_id_map = dict()
        self.input_seqs = []
        self.get_all_residue_index()
        self.create_output_name()
        self.create_all_chain_features()
        pass

    def get_all_residue_index(self):
        """get all residue indexes from subunits"""
        self.res_indexes=[]
        for i in self.interactors:
            curr_res_idx = i.feature_dict['residue_index']
            self.res_indexes.append([curr_res_idx[0],curr_res_idx[-1]])

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
        self.input_seqs, input_descs = parsers.parse_fasta(multimer_sequence_str)
        self.chain_id_map = pipeline_multimer._make_chain_id_map(
            sequences=self.input_seqs, descriptions=input_descs
        )

    def pair_and_merge(self, all_chain_features):
        """merge all chain features"""
        MAX_TEMPLATES = 4
        MSA_CROP_SIZE = 2048
        feature_processing.process_unmerged_features(all_chain_features)
        np_chains_list = list(all_chain_features.values())
        pair_msa_sequences = self.pair_msa and not feature_processing._is_homomer_or_monomer(np_chains_list)
        if pair_msa_sequences:
            np_chains_list = msa_pairing.create_paired_features(chains=np_chains_list)
            np_chains_list = msa_pairing.deduplicate_unpaired_sequences(np_chains_list)
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
