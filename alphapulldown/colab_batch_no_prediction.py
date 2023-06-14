from pathlib import Path
from ColabFold.colabfold.utils import (
    ACCEPT_DEFAULT_TERMS,
    DEFAULT_API_SERVER,
    NO_GPU_FOUND,
    CIF_REVISION_DATE,
    get_commit,
    safe_filename,
    setup_logging,
    CFMMCIFIO,
)
from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict
import os

from ColabFold.colabfold.batch import get_queries,unserialize_msa,get_msa_and_templates,msa_to_str,build_monomer_feature
from absl import logging,app
from pathlib import Path



def run_msa(queries,DEFAULT_API_SERVER,result_dir):
    # firstly declare some global variables 
    msa_mode = "MMseqs2 (UniRef+Environmental)"
    keep_existing_results=True
    result_dir = result_dir
    use_templates = True
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        jobname=safe_filename(raw_jobname)
        result_zip = os.path.join(result_dir,jobname,".result.zip")
        #result_zip = Path(result_dir).joinpath(jobname).with_suffix(".result.zip")
        if keep_existing_results and Path(result_zip).is_file():
            logging.info(f"Skipping {jobname} (result.zip)")
            continue
        # In the local version we use a marker file
        is_done_marker = os.path.join(result_dir,jobname + ".done.txt")
        if keep_existing_results and Path(is_done_marker).is_file():
            logging.info(f"Skipping {jobname} (already done)")
            continue

        query_sequence_len = (
            len(query_sequence)
            if isinstance(query_sequence, str)
            else sum(len(s) for s in query_sequence)
        )
        logging.info(
            f"Query {job_number + 1}/{len(queries)}: {jobname} (length {query_sequence_len})"
        )

        try:
            if a3m_lines is not None:
                (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features,
                ) = unserialize_msa(a3m_lines, query_sequence)
            else:
                (
                    unpaired_msa,
                    paired_msa,
                    query_seqs_unique,
                    query_seqs_cardinality,
                    template_features,
                ) = get_msa_and_templates(
                    jobname,
                    query_sequence,
                    result_dir,
                    msa_mode,
                    use_templates,
                    custom_template_path=None,
                    pair_mode="none",
                    host_url=DEFAULT_API_SERVER,
                )
            msa = msa_to_str(
                unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality
            )
            result_dir.joinpath(jobname + ".a3m").write_text(msa)
        except Exception as e:
            logging.info(f"Could not get MSA/templates for {jobname}: {e}")
            continue
        logging.info(f"unpaired msa is {unpaired_msa}")
        logging.info(f"paired msa is {paired_msa}")
        logging.info(f"template feature keys is {template_features[0].keys()}")
        monomeric_feature_dict = build_monomer_feature(query_sequence,unpaired_msa[0],template_features[0])
        logging.info(f"monomeric_feature_dict has keys: {monomeric_feature_dict.keys()}")

        # It seems that colabfold only pair msas using the same a3m file. They didn't extract uniprot 
        # ids and pair uniprot ids only. If so, I only have to copy the feautres from monomeric_dict
        # and make them {}_all_seq
def main(argv):
    input_dir = "/media/geoffrey/bigdata/g/kosinski/geoffrey/alphapulldown/example_data/colabfold_batch/"
    queries,is_complex = get_queries(input_dir)
    logging.info(f"is_complex is {is_complex}")
    result_dir='/media/geoffrey/bigdata/scratch/gyu/colabfold_batch_test'
    run_msa(queries,DEFAULT_API_SERVER,Path(result_dir))
if __name__=="__main__":
    app.run(main)