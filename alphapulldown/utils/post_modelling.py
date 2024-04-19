import subprocess,json,os

def zip_result_pickles(output_path):
    """A function that remove results pickles in the output directory"""
    cmd = f"cd {output_path} && gzip --force --verbose *.pkl"
    try:
        results = subprocess.run(cmd,shell=True,capture_output=True,text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while compressing result pickles: {e.returncode}")
        print(f"Command output: {e.output}")

def post_prediction_process(output_path,zip_pickles = False,remove_pickles = False):
    """A function to process resulted files after the prediction"""
    if remove_pickles and zip_pickles:
        remove_irrelavent_pickles(output_path)
    else:
        if zip_pickles:
            zip_result_pickles(output_path)
        if remove_pickles:
            remove_irrelavent_pickles(output_path)

def remove_irrelavent_pickles(output_path):
    """Remove result pickles that do not belong to the best model"""
    try:
        best_model = json.load(open(os.path.join(output_path,"ranking_debug.json"),'rb'))['order'][0]
        pickle_to_remove = [os.path.join(output_path,i) for i in os.listdir(output_path) if (i.endswith('pkl')) and (best_model not in i)]
        cmd = ['rm'] + pickle_to_remove
        results = subprocess.run(cmd)
    except FileNotFoundError:
        print(f"ranking_debug.json does not exist in : {output_path}. Please check your inputs.")
    except subprocess.CalledProcessError as e:
        print(f"Error while removing result pickles: {e.returncode}")
        print(f"Command output: {e.output}")      