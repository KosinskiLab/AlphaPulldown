import subprocess

class TestPip:
    def __init__(self) -> None:
        self.fasta_paths = "/g/kosinski/geoffrey/translation_initiation_apms"
        self.image_path = "/g/kosinski/geoffrey/alphapulldown"
        self.monomer_object_dir = "/scratch/gyu/test_monomer_objects/test_image"
        self.output_dir = "/scratch/gyu/test_predicted_structures/test_image"
        self.data_dir = "/scratch/AlphaFold_DBs/2.2.0/"
        self.max_template_date = "2200-01-01"
        self.oligomer_state_file = "/g/kosinski/geoffrey/alphapulldown/"
        self.protein_lists = "/g/kosinski/geoffrey/alphapulldown/"
        self.custom = "/g/kosinski/geoffrey/alphapulldown/"
    def test_create_features_no_precomuted_msa(self):
        bind_cmd = f"singularity exec --no-home --bind {self.fasta_paths}:/input_data,{self.data_dir}:/data_dir,{self.monomer_object_dir}:/output_dir"
        for i in [3,4,5,6]:

            exec_cmd = f"{self.image_path}/alphapulldown.sif create_individual_features.py --fasta_paths=/input_data/protein_sequences.fasta --data_dir=/data_dir --output_dir=output_dir --max_template_date={self.max_template_date} --seq_index={i}"
            print("will create features first")
            print(f"cmd looks like: {bind_cmd} {exec_cmd}")
            subprocess.run(f"{bind_cmd} {exec_cmd}",shell=True,executable="/bin/bash")

    
    def test_homooligomer(self):
        bind_cmd = f"singularity exec --no-home --bind {self.oligomer_state_file}:/input_data,{self.data_dir}:/data_dir,{self.output_dir}:/output_dir,{self.monomer_object_dir}:/monomer_object_dir"
        
        for i in [1,2]:
            exec_cmd = f"{self.image_path}/alphapulldown.sif run_multimer_jobs.py --mode=homo-oligomer --output_path=/output_dir/test_image/homooligomer --monomer_objects_dir=/monomer_object_dir --data_dir=/data_dir --job_index={i} --oligomer_state_file=/input_data/test_homooligomer_state.txt"
            subprocess.run(f"{bind_cmd} {exec_cmd}",shell=True,executable="/bin/bash")

    def test_all_vs_all(self):
        bind_cmd = f"singularity exec --no-home --bind {self.protein_lists}:/input_data,{self.data_dir}:/data_dir,{self.output_dir}:/output_dir,{self.monomer_object_dir}:/monomer_object_dir"
        for i in [1,2]:

            exec_cmd = f"{self.image_path}/alphapulldown.sif run_multimer_jobs.py --mode=all_vs_all --output_path=/output_dir/test_image/all_vs_all --monomer_objects_dir=/monomer_object_dir --data_dir=/data_dir --job_index={i} --protein_lists=/input_data/test_protein_list.txt"
            subprocess.run(f"{bind_cmd} {exec_cmd}",shell=True,executable="/bin/bash")

    def test_custom(self):
        bind_cmd = f"singularity exec --no-home --bind {self.custom}:/input_data,{self.data_dir}:/data_dir,{self.output_dir}:/output_dir,{self.monomer_object_dir}:/monomer_object_dir"
        for i in [1,2]:
            exec_cmd = f"{self.image_path}/alphapulldown.sif run_multimer_jobs.py --mode=custom --output_path=/output_dir/test_image/custom --monomer_objects_dir=/monomer_object_dir --data_dir=/data_dir --job_index={i} --protein_lists=/input_data/test_custom.txt"
            subprocess.run(f"{bind_cmd} {exec_cmd}",shell=True,executable="/bin/bash")
    
    def run_tests(self):
        self.test_create_features_no_precomuted_msa()
        self.test_homooligomer()
        self.test_all_vs_all()
        self.test_custom()

if __name__ == '__main__':
    test_object = TestPip()
    test_object.run_tests()