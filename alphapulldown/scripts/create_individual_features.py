#!/usr/bin/env python

""" CLI inferface for creating AlphaFold features from FASTA files, with optional MMseqs2 and HHsearch support.
    Allows to create features with custom multimeric templates using PDB/mmCIF and CSV files with descriptions.
    Copyright (c) 2025 European Molecular Biology Laboratory
    Author: Dmitry Molodenskiy <dmitry.molodenskiy@embl-hamburg.de>
"""
import os
import json
import lzma
import pickle
import tempfile
from datetime import date
from pathlib import Path

from absl import app, flags, logging
from alphafold.data.pipeline import DataPipeline
from alphafold.data import templates
from alphafold.data.tools import hmmsearch, hhsearch

from alphapulldown.utils.create_custom_template_db import create_db
from alphapulldown.objects import MonomericObject
from alphapulldown.utils.file_handling import iter_seqs, parse_csv_file
from alphapulldown.utils.modelling_setup import create_uniprot_runner
from alphapulldown.utils import save_meta_data

# Flags
flags.DEFINE_list('fasta_paths', None, 'Comma-separated FASTA paths.')
flags.DEFINE_string('data_dir', None, 'Directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Directory for results.')
flags.DEFINE_string('jackhmmer_binary_path', None, 'JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', None, 'HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', None, 'HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', None, 'HMMER "hmmsearch" executable.')
flags.DEFINE_string('hmmbuild_binary_path', None, 'HMMER "hmmbuild" executable.')
flags.DEFINE_string('kalign_binary_path', None, 'Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Uniref90 database path.')
flags.DEFINE_string('mgnify_database_path', None, 'MGnify database path.')
flags.DEFINE_string('bfd_database_path', None, 'BFD database path.')
flags.DEFINE_string('small_bfd_database_path', None, 'Small BFD database path.')
flags.DEFINE_string('uniref30_database_path', None, 'UniRef30 database path.')
flags.DEFINE_string('uniprot_database_path', None, 'Uniprot database path.')
flags.DEFINE_string('pdb70_database_path', None, 'PDB70 database path.')
flags.DEFINE_string('pdb_seqres_database_path', None, 'PDB seqres database path.')
flags.DEFINE_string('template_mmcif_dir', None, 'Directory of template mmCIF files.')
flags.DEFINE_string('max_template_date', '2050-10-10', 'Max template release date (default far future).')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Obsolete PDBs mapping.')
flags.DEFINE_enum('db_preset', 'full_dbs', ['full_dbs', 'reduced_dbs'], 'MSA database preset.')
flags.DEFINE_bool('use_precomputed_msas', False, 'Reuse existing MSAs.')
flags.DEFINE_bool('use_mmseqs2', False, 'Use MMseqs2 instead of HMMER/MSA tools.')
flags.DEFINE_bool('save_msa_files', False, 'Save MSA files.')
flags.DEFINE_bool('skip_existing', False, 'Skip if feature pickle exists.')
flags.DEFINE_integer('seq_index', None, '1-based sequence index to process.')
flags.DEFINE_bool('use_hhsearch', False, 'Use HHsearch for templates.')
flags.DEFINE_bool('compress_features', False, 'Compress outputs with LZMA.')
flags.DEFINE_string('path_to_mmt', None, 'Directory of multimeric templates.')
flags.DEFINE_string('description_file', None, 'CSV file of multimer descriptions.')
flags.DEFINE_float('threshold_clashes', 1000.0, 'VDW clash threshold.')
flags.DEFINE_float('hb_allowance', 0.4, 'Hydrogen bond allowance.')
flags.DEFINE_float('plddt_threshold', 0.0, 'pLDDT threshold.')
flags.DEFINE_bool('multiple_mmts', False, 'Allow multiple templates per protein.')

FLAGS = flags.FLAGS
MAX_TEMPLATE_HITS = 20

# Default database subpaths under data_dir
DB_SUBPATHS = {
    'uniref30': 'uniref30/UniRef30_2023_02',
    'uniref90': 'uniref90/uniref90.fasta',
    'mgnify': 'mgnify/mgy_clusters_2022_05.fa',
    'bfd': 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
    'small_bfd': 'small_bfd/bfd-first_non_consensus_sequences.fasta',
    'pdb70': 'pdb70/pdb70',
}

def resolve_path(flag_val, key):
    return flag_val or os.path.join(FLAGS.data_dir, DB_SUBPATHS[key])

def init_args(custom_db=None):
    """Set up database and template paths."""
    FLAGS.uniref30_database_path = resolve_path(FLAGS.uniref30_database_path, 'uniref30')
    FLAGS.uniref90_database_path = resolve_path(FLAGS.uniref90_database_path, 'uniref90')
    FLAGS.mgnify_database_path = resolve_path(FLAGS.mgnify_database_path, 'mgnify')
    FLAGS.bfd_database_path = resolve_path(FLAGS.bfd_database_path, 'bfd')
    FLAGS.small_bfd_database_path = resolve_path(FLAGS.small_bfd_database_path, 'small_bfd')
    FLAGS.pdb70_database_path = resolve_path(FLAGS.pdb70_database_path, 'pdb70')

    if custom_db:
        base = Path(custom_db)
        FLAGS.pdb_seqres_database_path = str(base / 'pdb_seqres' / 'pdb_seqres.txt')
        FLAGS.template_mmcif_dir = str(base / 'pdb_mmcif' / 'mmcif_files')
        FLAGS.obsolete_pdbs_path = str(base / 'pdb_mmcif' / 'obsolete.dat')
    else:
        FLAGS.pdb_seqres_database_path = FLAGS.pdb_seqres_database_path or os.path.join(FLAGS.data_dir, 'pdb_seqres/pdb_seqres.txt')
        FLAGS.template_mmcif_dir = FLAGS.template_mmcif_dir or os.path.join(FLAGS.data_dir, 'pdb_mmcif/mmcif_files')
        FLAGS.obsolete_pdbs_path = FLAGS.obsolete_pdbs_path or os.path.join(FLAGS.data_dir, 'pdb_mmcif/obsolete.dat')

    return FLAGS.flag_values_dict()

def create_pipeline():
    """Return DataPipeline or None for MMseqs2."""
    if FLAGS.use_mmseqs2:
        return None

    if FLAGS.use_hhsearch:
        logging.info('Using HHsearch for templates')
        searcher = hhsearch.HHSearch(FLAGS.hhsearch_binary_path, [FLAGS.pdb70_database_path])
        featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path
        )
    else:
        logging.info('Using HMMER hmmsearch for templates')
        searcher = hmmsearch.Hmmsearch(
            binary_path=FLAGS.hmmsearch_binary_path,
            hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
            database_path=FLAGS.pdb_seqres_database_path
        )
        featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=FLAGS.template_mmcif_dir,
            max_template_date=FLAGS.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=FLAGS.kalign_binary_path,
            obsolete_pdbs_path=FLAGS.obsolete_pdbs_path
        )

    return DataPipeline(
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        hhblits_binary_path=FLAGS.hhblits_binary_path,
        uniref90_database_path=FLAGS.uniref90_database_path,
        mgnify_database_path=FLAGS.mgnify_database_path,
        bfd_database_path=FLAGS.bfd_database_path,
        uniref30_database_path=FLAGS.uniref30_database_path,
        small_bfd_database_path=FLAGS.small_bfd_database_path,
        use_small_bfd=FLAGS.db_preset == 'reduced_dbs',
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        template_searcher=searcher,
        template_featurizer=featurizer
    )

def save_pickle(obj, path):
    open_fn = lzma.open if FLAGS.compress_features else open
    ext = '.xz' if FLAGS.compress_features else ''
    with open_fn(path + ext, 'wb') as f:
        pickle.dump(obj, f)

def save_meta(outdir, desc):
    meta = save_meta_data.get_meta_dict(FLAGS.flag_values_dict())
    fname = f"{desc}_feature_metadata_{date.today()}.json"
    open_fn = lzma.open if FLAGS.compress_features else open
    mode = 'wt' if FLAGS.compress_features else 'w'
    ext = '.xz' if FLAGS.compress_features else ''
    with open_fn(os.path.join(outdir, fname) + ext, mode) as f:
        json.dump(meta, f)

def process_monomer(mono, pipeline):
    pkl_base = os.path.join(FLAGS.output_dir, f"{mono.description}.pkl")
    if FLAGS.skip_existing and os.path.exists(pkl_base + ('.xz' if FLAGS.compress_features else '')):
        logging.info(f"Skipping {mono.description}")
        return
    save_meta(FLAGS.output_dir, mono.description)
    mono.make_features(
        pipeline=pipeline,
        output_dir=FLAGS.output_dir,
        use_precomputed_msa=FLAGS.use_precomputed_msas,
        save_msa=FLAGS.save_msa_files,
        use_mmseqs2=FLAGS.use_mmseqs2
    )
    save_pickle(mono, pkl_base)

def process_individual():
    init_args()
    uni_runner = None if FLAGS.use_mmseqs2 else create_uniprot_runner(FLAGS.jackhmmer_binary_path, FLAGS.uniprot_database_path)
    pipeline = create_pipeline()
    for idx, (seq, desc) in enumerate(iter_seqs(FLAGS.fasta_paths), 1):
        if not FLAGS.seq_index or FLAGS.seq_index == idx:
            mono = MonomericObject(desc, seq)
            mono.uniprot_runner = uni_runner
            process_monomer(mono, pipeline)

def process_multimeric():
    feats = parse_csv_file(FLAGS.description_file, FLAGS.fasta_paths, FLAGS.path_to_mmt, FLAGS.multiple_mmts)
    for idx, feat in enumerate(feats, 1):
        if not FLAGS.seq_index or FLAGS.seq_index == idx:
            for t in feat['templates']:
                if not os.path.isfile(t):
                    raise FileNotFoundError(f"Missing template: {t}")
            with tempfile.TemporaryDirectory() as tmp:
                create_db(tmp, feat['templates'], feat['chains'], FLAGS.threshold_clashes, FLAGS.hh_allowance, FLAGS.plddt_threshold)
                init_args(tmp)
                uni_runner = None if FLAGS.use_mmseqs2 else create_uniprot_runner(FLAGS.jackhmmer_binary_path, FLAGS.uniprot_database_path)
                mono = MonomericObject(feat['protein'], feat['sequence'])
                mono.uniprot_runner = uni_runner
                process_monomer(mono, create_pipeline())

def main(_argv):
    if FLAGS.use_mmseqs2 and FLAGS.path_to_mmt:
        raise ValueError("Cannot use MMseqs2 with multimeric templates.")
    Path(FLAGS.output_dir).mkdir(parents=True, exist_ok=True)

    # max_template_date has a sensible default, so no error needed

    if FLAGS.path_to_mmt:
        process_multimeric()
    else:
        process_individual()


if __name__ == '__main__':
    flags.mark_flags_as_required(['fasta_paths', 'output_dir', 'data_dir'])
    app.run(main)
