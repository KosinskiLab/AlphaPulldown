"""
A unittest script that test if creating MonomericObject 
or MultimericObject works
"""
from absl.testing import absltest
from alphapulldown.objects import MonomericObject
import shutil
from alphafold.data.pipeline import DataPipeline
from alphafold.data.tools import hmmsearch
from alphapulldown.utils.file_handling import parse_fasta
from alphapulldown.folding_backend.alphafold_backend import AlphaFoldBackend
from alphapulldown.utils.modelling_setup import create_uniprot_runner, templates
import os
from colabfold.utils import DEFAULT_API_SERVER

class TestCreateObjects(absltest.TestCase):
    def setUp(self) -> None:
        self.jackhmmer_binary_path = shutil.which("jackhmmer")
        self.hmmsearch_binary_path = shutil.which("hmmsearch")
        self.hhblits_binary_path = shutil.which("hhblits")
        self.kalign_binary_path = shutil.which("kalign")
        self.hmmbuild_binary_path = shutil.which("hmmbuild")
        self.fasta_paths = "./test/test_data/fastas/test.fasta"
        self.monomer_object_dir = "./test/test_data/features"
        self.output_dir = "./test/test_data/features"
        self.data_dir = "/scratch/AlphaFold_DBs/2.3.2/"
        self.max_template_date = "2200-01-01"
        self.uniref30_database_path = os.path.join(self.data_dir, "uniref30", "UniRef30_2021_03")
        self.uniref90_database_path = os.path.join(self.data_dir, "uniref90", "uniref90.fasta")
        self.mgnify_database_path = os.path.join(self.data_dir, "mgnify", "mgy_clusters_2022_05.fa")
        self.bfd_database_path = os.path.join(self.data_dir, "bfd",
            "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt")
        self.small_bfd_database_path = os.path.join(self.data_dir,
                                                    "small_bfd", "bfd-first_non_consensus_sequences.fasta")
        self.pdb_seqres_database_path = os.path.join(self.data_dir, "pdb_seqres", "pdb_seqres.txt")
        self.template_mmcif_dir = os.path.join(self.data_dir, "pdb_mmcif", "mmcif_files")
        self.obsolete_pdbs_path = os.path.join(self.data_dir, "pdb_mmcif", "obsolete.dat")
        self.pdb70_database_path = os.path.join(self.data_dir, "pdb70", "pdb70")
        self.uniprot_database_path = os.path.join(self.data_dir, "uniprot/uniprot.fasta")

        self.num_cycle = 1
        self.random_seed = None
        self.num_predictions_per_model = 20
        self.msa_depth = 128
        self.model_names = "model_2_multimer_v3"
        self.gradient_msa_depth = True

    def test_1_initialise_MonomericObject(self):
        """Test initialise a monomeric object"""
        sequences, descriptions = parse_fasta(open(self.fasta_paths,'r').read())
        monomer_object = MonomericObject(description=descriptions[0],sequence=sequences[0])
        assert monomer_object.description == descriptions[0]
        assert monomer_object.sequence == sequences[0]
        return monomer_object
    
    def test_2_initialise_datapipeline(self):
        """Test setting up datapipelines"""
        monomer_data_pipeline = DataPipeline(
        jackhmmer_binary_path=self.jackhmmer_binary_path,
        hhblits_binary_path=self.hhblits_binary_path,
        uniref90_database_path=self.uniref90_database_path,
        mgnify_database_path=self.mgnify_database_path,
        bfd_database_path=self.bfd_database_path,
        uniref30_database_path=self.uniref30_database_path,
        small_bfd_database_path=self.small_bfd_database_path,
        use_small_bfd=False,
        use_precomputed_msas=False,
        template_searcher=hmmsearch.Hmmsearch(
            binary_path=self.hmmsearch_binary_path,
            hmmbuild_binary_path=self.hmmbuild_binary_path,
            database_path=self.pdb_seqres_database_path,
        ),
        template_featurizer=templates.HmmsearchHitFeaturizer(
            mmcif_dir=self.template_mmcif_dir,
            max_template_date=self.max_template_date,
            max_hits=20,
            kalign_binary_path=self.kalign_binary_path,
            obsolete_pdbs_path=self.obsolete_pdbs_path,
            release_dates_path=None,
        ),)
        return monomer_data_pipeline

    @absltest.skip("Skip the test for now")
    def test_3_create_model_runner_gradient_msa_depth(self):
        msa_range = [16,19,23,28,33,40,48,57,69,82,99,118,142,170,204,245,294,353,423,508]
        extra_msa_ranges = [i*4 for i in msa_range]
        model_runners, random_seed = AlphaFoldBackend.setup(
            "multimer",
            self.num_cycle,
            self.random_seed,
            self.data_dir,
            self.num_predictions_per_model,
            self.gradient_msa_depth,
            self.model_names,
        )
        for num, model_runner in enumerate(model_runners):
            self.assertEqual(model_runner, f"model_2_multimer_v3_pred_{num}_msa_{msa_range[num]}")

    @absltest.skip("Skip the test for now")
    def test_4_create_model_runner_one_model_msa_depth(self):
        model_runners, random_seed = create_model_runners_and_random_seed(
            "multimer",
            self.num_cycle,
            self.random_seed,
            self.data_dir,
            self.num_predictions_per_model,
            self.gradient_msa_depth,
            self.model_names,
            self.msa_depth,
        )
        for num, model_runner in enumerate(model_runners):
            self.assertEqual(model_runner, f"model_2_multimer_v3_pred_{num}_msa_128")

    def test_5_run_alignments(self):
        monomer_obj = self.test_1_initialise_MonomericObject()
        monomer_pipeline = self.test_2_initialise_datapipeline()
        uniprot_runner = create_uniprot_runner(self.jackhmmer_binary_path, self.uniprot_database_path)
        monomer_obj.uniprot_runner = uniprot_runner
        monomer_obj.make_features(monomer_pipeline,self.output_dir,
                                  use_precomputed_msa=False,save_msa=True)

    @absltest.skip("Skip the test for now")
    def test_5_run_alignments_with_mmseqs2(self):
        # Firstly test mmseqs2 remote mode 
        monomer_obj = self.test_1_initialise_MonomericObject()
        if os.path.exists(os.path.join("./test/test_data",f"{monomer_obj.description}.a3m")):
            os.remove(os.path.join("./test/test_data",f"{monomer_obj.description}.a3m"))
        monomer_obj.make_mmseq_features(
            DEFAULT_API_SERVER=DEFAULT_API_SERVER,
            output_dir=self.output_dir
        )
        # then test running mmseqs2 if the a3m file already exists
        monomer_obj.make_mmseq_features(
            DEFAULT_API_SERVER=DEFAULT_API_SERVER,
            output_dir=self.output_dir
        )

    @absltest.skip("Skip the test for now")
    def test_5_run_alignments_with_mmseqs2_zipped_msa(self):
        # Firstly test mmseqs2 remote mode 
        monomer_obj = self.test_1_initialise_MonomericObject()
        if os.path.exists(os.path.join("./test/test_data",f"{monomer_obj.description}.a3m")):
            os.remove(os.path.join("./test/test_data",f"{monomer_obj.description}.a3m"))
        if os.path.exists(os.path.join("./test/test_data",f"{monomer_obj.description}.a3m.gz")):
            os.remove(os.path.join("./test/test_data",f"{monomer_obj.description}.a3m.gz"))
        monomer_obj.make_mmseq_features(
            DEFAULT_API_SERVER=DEFAULT_API_SERVER,
            output_dir=self.output_dir,compress_msa_files=True
        )
        # then test running mmseqs2 if the a3m file already exists
        monomer_obj.make_mmseq_features(
            DEFAULT_API_SERVER=DEFAULT_API_SERVER,
            output_dir=self.output_dir,compress_msa_files=True
        )

    def test_6_make_features_from_precomputed_msa(self):
        monomer_obj = self.test_1_initialise_MonomericObject()
        monomer_pipeline = self.test_2_initialise_datapipeline()
        uniprot_runner = create_uniprot_runner(self.jackhmmer_binary_path, self.uniprot_database_path)
        monomer_obj.uniprot_runner = uniprot_runner
        monomer_obj.make_features(monomer_pipeline,self.output_dir,
                                  use_precomputed_msa=True,save_msa=True)
    
    def test_7_make_features_and_zip_msa(self):
        monomer_obj = self.test_1_initialise_MonomericObject()
        monomer_pipeline = self.test_2_initialise_datapipeline()
        uniprot_runner = create_uniprot_runner(self.jackhmmer_binary_path, self.uniprot_database_path)
        monomer_obj.uniprot_runner = uniprot_runner
        monomer_obj.make_features(monomer_pipeline,self.output_dir,
                                  use_precomputed_msa=False,save_msa=True,compress_msa_files=True)
        # Then test whether the zipped msa files can be reused
        monomer_obj.make_features(monomer_pipeline,self.output_dir,
                                  use_precomputed_msa=True,save_msa=True)

if __name__ == "__main__":
    absltest.main()