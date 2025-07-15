#!/usr/bin/env python3
# coding: utf-8
"""
Feature generator for AlphaFold 2 and AlphaFold 3, supporting classic Hmmer, MMseqs2, and truemultimer modes.

"""

import json
import lzma
import os
import pickle
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from absl import logging, app, flags
from colabfold.utils import DEFAULT_API_SERVER

# AlphaFold2 imports
from alphafold.data import templates
from alphafold.data.pipeline import DataPipeline as AF2DataPipeline
from alphafold.data.tools import hmmsearch, hhsearch

# AlphaPulldown helpers
from alphapulldown.utils.create_custom_template_db import create_db
from alphapulldown.objects import MonomericObject
from alphapulldown.utils.file_handling import iter_seqs, parse_csv_file
from alphapulldown.utils.modelling_setup import create_uniprot_runner
from alphapulldown.utils import save_meta_data

# Try to import AlphaFold3, but it's optional
try:
    from alphafold3.data.pipeline import DataPipeline as AF3DataPipeline, DataPipelineConfig as AF3DataPipelineConfig
    from alphafold3.common import folding_input
except ImportError:
    AF3DataPipeline = None
    AF3DataPipelineConfig = None
    folding_input = None

# =================== Database Maps ===================
AF2_DATABASES = {
    "uniref90": "uniref90/uniref90.fasta",
    "uniref30": "uniref30/UniRef30_2023_02",
    "mgnify": "mgnify/mgy_clusters_2022_05.fa",
    "bfd": "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
    "small_bfd": "small_bfd/bfd-first_non_consensus_sequences.fasta",
    "pdb70": "pdb70/pdb70",
    "uniprot": "uniprot/uniprot.fasta",
    "pdb_seqres": "pdb_seqres/pdb_seqres.txt",
    "template_mmcif_dir": "pdb_mmcif/mmcif_files",
    "obsolete_pdbs": "pdb_mmcif/obsolete.dat",
}

AF3_DATABASES = {
    "uniref90": "uniref90_2022_05.fa",
    "uniref30": "uniref30/UniRef30_2023_02",
    "mgnify": "mgy_clusters_2022_05.fa",
    "bfd": "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
    "small_bfd": "bfd-first_non_consensus_sequences.fasta",
    "pdb_seqres": "pdb_seqres_2022_09_28.fasta",
    "template_mmcif_dir": "mmcif_files",
    "obsolete_pdbs": "obsolete.dat",
    "pdb70": "pdb70/pdb70",
    "uniprot": "uniprot_all_2021_04.fa",
    "ntrna": "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rfam": "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
    "rna_central": "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
}

# =================== Flags ===================
flags.DEFINE_enum(
    'data_pipeline', 'alphafold2', ['alphafold2', 'alphafold3'],
    'Choose pipeline: alphafold2 or alphafold3'
)
flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing a prediction target.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to output directory.')
flags.DEFINE_string('jackhmmer_binary_path', shutil.which('jackhmmer'), '')
flags.DEFINE_string('hhblits_binary_path', shutil.which('hhblits'), '')
flags.DEFINE_string('hhsearch_binary_path', shutil.which('hhsearch'), '')
flags.DEFINE_string('hmmsearch_binary_path', shutil.which('hmmsearch'), '')
flags.DEFINE_string('hmmbuild_binary_path', shutil.which('hmmbuild'), '')
flags.DEFINE_string('nhmmer_binary_path', shutil.which('nhmmer'), '')
flags.DEFINE_string('hmmalign_binary_path', shutil.which('hmmalign'), '')
flags.DEFINE_string('kalign_binary_path', shutil.which('kalign'), '')
flags.DEFINE_string('uniref90_database_path', None, '')
flags.DEFINE_string('mgnify_database_path', None, '')
flags.DEFINE_string('bfd_database_path', None, '')
flags.DEFINE_string('small_bfd_database_path', None, '')
flags.DEFINE_string('uniref30_database_path', None, '')
flags.DEFINE_string('uniprot_database_path', None, '')
flags.DEFINE_string('pdb70_database_path', None, '')
flags.DEFINE_string('pdb_seqres_database_path', None, '')
flags.DEFINE_string('template_mmcif_dir', None, '')
flags.DEFINE_string('max_template_date', None, 'Max template release date.')
flags.DEFINE_string('obsolete_pdbs_path', None, '')
flags.DEFINE_enum('db_preset', 'full_dbs', ['full_dbs', 'reduced_dbs'], '')
flags.DEFINE_boolean('use_precomputed_msas', False, '')
flags.DEFINE_bool("use_mmseqs2", False, "")
flags.DEFINE_bool("save_msa_files", False, "")
flags.DEFINE_bool("skip_existing", False, "")
flags.DEFINE_string("new_uniclust_dir", None, "")
flags.DEFINE_integer("seq_index", None, "")
flags.DEFINE_boolean("use_hhsearch", False, "")
flags.DEFINE_boolean("compress_features", False, "")
flags.DEFINE_string("path_to_mmt", None, "")
flags.DEFINE_string("description_file", None, "")
flags.DEFINE_float("threshold_clashes", 1000, "")
flags.DEFINE_float("hb_allowance", 0.4, "")
flags.DEFINE_float("plddt_threshold", 0, "")
flags.DEFINE_boolean("multiple_mmts", False, "")

FLAGS = flags.FLAGS

# =================== Helper Functions ===================

def get_database_path(key):
    """Return the absolute path for a given database key, depending on pipeline."""
    db_map = AF3_DATABASES if FLAGS.data_pipeline == 'alphafold3' else AF2_DATABASES
    default_subpath = db_map[key]
    return os.path.join(FLAGS.data_dir, default_subpath)

def create_arguments(local_custom_template_db=None):
    """Set all database paths in FLAGS for the selected AlphaFold version.
    Optionally override template paths with a local custom template DB."""
    FLAGS.uniref90_database_path = get_database_path("uniref90")
    FLAGS.uniref30_database_path = get_database_path("uniref30")
    FLAGS.mgnify_database_path = get_database_path("mgnify")
    FLAGS.bfd_database_path = get_database_path("bfd")
    FLAGS.small_bfd_database_path = get_database_path("small_bfd")
    FLAGS.pdb70_database_path = get_database_path("pdb70")
    FLAGS.uniprot_database_path = get_database_path("uniprot")
    FLAGS.pdb_seqres_database_path = get_database_path("pdb_seqres")
    FLAGS.template_mmcif_dir = get_database_path("template_mmcif_dir")
    FLAGS.obsolete_pdbs_path = get_database_path("obsolete_pdbs")
    if local_custom_template_db:
        FLAGS.pdb_seqres_database_path = os.path.join(local_custom_template_db, "pdb_seqres.txt")
        FLAGS.template_mmcif_dir = os.path.join(local_custom_template_db, "pdb_mmcif", "mmcif_files")
        FLAGS.obsolete_pdbs_path = os.path.join(local_custom_template_db, "pdb_mmcif", "obsolete.dat")

def check_template_date():
    """Check if the max_template_date is provided."""
    if not FLAGS.max_template_date:
        logging.error("You have not provided a max_template_date. Please specify a date and run again.")
        sys.exit(1)

# =================== AlphaFold 2 Feature Creation ===================

def create_pipeline_af2():
    """Create and configure the AlphaFold2 data pipeline."""
    use_small_bfd = FLAGS.db_preset == "reduced_dbs"
    if FLAGS.use_hhsearch:
        template_searcher = hhsearch.HHSearch(
            binary_path=FLAGS.hhsearch_binary_path, databases=[FLAGS.pdb70_database_path]
        )
        template_featuriser = templates.HhsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir, max_template_date=FLAGS.max_template_date,
            max_hits=20, kalign_binary_path=FLAGS.kalign_binary_path,
            release_dates_path=None, obsolete_pdbs_path=FLAGS.obsolete_pdbs_path
        )
    else:
        template_featuriser = templates.HmmsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir, max_template_date=FLAGS.max_template_date,
            max_hits=20, kalign_binary_path=FLAGS.kalign_binary_path,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path, release_dates_path=None
        )
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=FLAGS.hmmsearch_binary_path,
            hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
            database_path=FLAGS.pdb_seqres_database_path
        )
    return AF2DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        use_small_bfd=use_small_bfd,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        template_searcher=template_searcher,
        template_featurizer=template_featuriser
    )

def create_individual_features():
    """Generate AlphaFold2 features for each monomer sequence."""
    create_arguments()
    pipeline = create_pipeline_af2()
    uniprot_runner = None if FLAGS.use_mmseqs2 else create_uniprot_runner(
        FLAGS.jackhmmer_binary_path, FLAGS.uniprot_database_path
    )
    for seq_idx, (seq, desc) in enumerate(iter_seqs(FLAGS.fasta_paths), 1):
        if FLAGS.seq_index is None or seq_idx == FLAGS.seq_index:
            monomer = MonomericObject(desc, seq)
            monomer.uniprot_runner = uniprot_runner
            create_and_save_monomer_objects(monomer, pipeline)

def create_and_save_monomer_objects(monomer, pipeline):
    """Save a MonomericObject after feature creation (pickled, optionally compressed)."""
    # Ensure output directory exists
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    pickle_path = os.path.join(FLAGS.output_dir, f"{monomer.description}.pkl")
    if FLAGS.compress_features:
        pickle_path += ".xz"
    if FLAGS.skip_existing and os.path.exists(pickle_path):
        logging.info(f"Feature file for {monomer.description} already exists. Skipping...")
        return
    meta_dict = save_meta_data.get_meta_dict(FLAGS.flag_values_dict())
    metadata_output_path = os.path.join(
        FLAGS.output_dir, f"{monomer.description}_feature_metadata_{datetime.now().date()}.json"
    )
    if FLAGS.compress_features:
        with lzma.open(metadata_output_path + '.xz', "wt") as meta_data_outfile:
            json.dump(meta_dict, meta_data_outfile)
    else:
        with open(metadata_output_path, "w") as meta_data_outfile:
            json.dump(meta_dict, meta_data_outfile)
    if FLAGS.use_mmseqs2:
        monomer.make_mmseq_features(DEFAULT_API_SERVER=DEFAULT_API_SERVER, output_dir=FLAGS.output_dir, use_precomputed_msa=FLAGS.use_precomputed_msas)
    else:
        monomer.make_features(
            pipeline=pipeline, output_dir=FLAGS.output_dir,
            use_precomputed_msa=FLAGS.use_precomputed_msas,
            save_msa=FLAGS.save_msa_files)
    if FLAGS.compress_features:
        with lzma.open(pickle_path, "wb") as pickle_file:
            pickle.dump(monomer, pickle_file)
    else:
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(monomer, pickle_file)

def create_individual_features_truemultimer():
    """Generate features in TrueMultimer mode, one set per entry in the description CSV."""
    feats = parse_csv_file(
        FLAGS.description_file, FLAGS.fasta_paths, FLAGS.path_to_mmt, FLAGS.multiple_mmts
    )
    for idx, feat in enumerate(feats, 1):
        if FLAGS.seq_index is None or idx == FLAGS.seq_index:
            process_multimeric_features(feat, idx)

def process_multimeric_features(feat, idx):
    """Process a multimeric feature from a parsed CSV entry."""
    for temp_path in feat["templates"]:
        if not os.path.isfile(temp_path):
            raise FileNotFoundError(f"Template file {temp_path} does not exist.")
    protein, chains, template_paths = feat["protein"], feat["chains"], feat["templates"]
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path_to_custom_db = create_custom_db(temp_dir, protein, template_paths, chains)
        create_arguments(local_path_to_custom_db)
        uniprot_runner = None if FLAGS.use_mmseqs2 else create_uniprot_runner(
            FLAGS.jackhmmer_binary_path, FLAGS.uniprot_database_path
        )
        pipeline = create_pipeline_af2()
        monomer = MonomericObject(protein, feat['sequence'])
        monomer.uniprot_runner = uniprot_runner
        create_and_save_monomer_objects(monomer, pipeline)

def create_custom_db(temp_dir, protein, template_paths, chains):
    """Create a local custom template DB for TrueMultimer/AF2."""
    local_path_to_custom_template_db = Path(temp_dir) / "custom_template_db" / protein
    create_db(
        local_path_to_custom_template_db, template_paths, chains,
        FLAGS.threshold_clashes, FLAGS.hb_allowance, FLAGS.plddt_threshold
    )
    return local_path_to_custom_template_db

# =================== AlphaFold 3 Feature Creation ===================

def create_pipeline_af3():
    """Create the AlphaFold3 pipeline. Raises if AF3 not available."""
    if AF3DataPipeline is None or AF3DataPipelineConfig is None:
        raise ImportError("alphafold3.data.pipeline not available")
    
    # Convert max_template_date string to datetime.date object
    import datetime
    max_template_date = datetime.date.fromisoformat(FLAGS.max_template_date)
    
    config = AF3DataPipelineConfig(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        nhmmer_binary_path=FLAGS.nhmmer_binary_path,
        hmmalign_binary_path=FLAGS.hmmalign_binary_path,
        hmmsearch_binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        small_bfd_database_path=get_database_path("small_bfd"),
        mgnify_database_path=get_database_path("mgnify"),
        uniprot_cluster_annot_database_path=get_database_path("uniprot"),
        uniref90_database_path=get_database_path("uniref90"),
        ntrna_database_path=get_database_path("ntrna"),
        rfam_database_path=get_database_path("rfam"),
        rna_central_database_path=get_database_path("rna_central"),
        pdb_database_path=get_database_path("template_mmcif_dir"),
        seqres_database_path=get_database_path("pdb_seqres"),
        jackhmmer_n_cpu=8,
        nhmmer_n_cpu=8,
        max_template_date=max_template_date
    )
    return AF3DataPipeline(config)

def create_af3_individual_features():
    """Generate AlphaFold3 features, one .json per chain."""
    # Ensure output directory exists
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    pipeline = create_pipeline_af3()
    for seq_idx, (seq, desc) in enumerate(iter_seqs(FLAGS.fasta_paths), 1):
        if FLAGS.seq_index is None or seq_idx == FLAGS.seq_index:
            # Check if output file already exists and skip if requested
            outpath = Path(FLAGS.output_dir) / f"{desc}_af3_input.json"
            if FLAGS.skip_existing and outpath.exists():
                logging.info(f"Feature file for {desc} already exists. Skipping...")
                continue
            
            # Create AlphaFold3 input object with proper chain structure
            try:
                # Generate proper chain ID using AlphaFold3's int_id_to_str_id function
                try:
                    from alphafold3.structure.mmcif import int_id_to_str_id
                    chain_id = int_id_to_str_id(seq_idx)
                except ImportError:
                    # Fallback if mmcif_lib is not available
                    chain_id = chr(ord('A') + (seq_idx - 1) % 26)
                    if seq_idx > 26:
                        # For sequences beyond 26, use AA, BB, etc.
                        chain_id = chain_id + chain_id
                
                # Determine chain type based on sequence content
                if all(c in 'ACGTN' for c in seq.upper()):
                    # DNA sequence
                    from alphafold3.common.folding_input import DnaChain
                    chain = DnaChain(sequence=seq, id=chain_id, modifications=[])
                elif all(c in 'ACGUN' for c in seq.upper()):
                    # RNA sequence
                    from alphafold3.common.folding_input import RnaChain
                    chain = RnaChain(sequence=seq, id=chain_id, modifications=[])
                elif all(c in 'ACDEFGHIKLMNPQRSTVWYX' for c in seq.upper()):
                    # Protein sequence
                    from alphafold3.common.folding_input import ProteinChain
                    chain = ProteinChain(sequence=seq, id=chain_id, ptms=[])
                else:
                    raise ValueError(f"Invalid sequence: {seq}")
                
                input_obj = folding_input.Input(
                    name=desc,
                    chains=[chain],
                    rng_seeds=[42]
                )
                
                features = pipeline.process(input_obj)
                if hasattr(features, "to_json"):
                    outpath.write_text(features.to_json())
                else:
                    outpath.write_text(json.dumps(features))
                    
            except Exception as e:
                logging.error(f"Failed to create AlphaFold3 input object for {desc}: {e}")
                continue

# =================== Main Entry Point ===================

def main(argv):
    """Main entry: dispatch to AF2 or AF3, truemultimer or not."""
    del argv
    Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)
    if FLAGS.data_pipeline == "alphafold3":
        create_af3_individual_features()
    else:
        check_template_date()
        if FLAGS.path_to_mmt:
            create_individual_features_truemultimer()
        else:
            create_individual_features()

if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["fasta_paths", "output_dir", "max_template_date", "data_dir"]
    )
    app.run(main)
