def create_individual_features():
    from absl import app
    from alphapulldown.scripts import create_individual_features
    app.run(create_individual_features.main)

def run_multimer_jobs():
    from absl import app
    from alphapulldown.scripts import run_multimer_jobs
    app.run(run_multimer_jobs.main)

def run_structure_prediction():
    from absl import app
    from alphapulldown.scripts import run_structure_prediction
    app.run(run_structure_prediction.main)

def convert_to_modelcif():
    from absl import app
    from alphapulldown.scripts import convert_to_modelcif
    app.run(convert_to_modelcif.main)

def truncate_pickles():
    from absl import app
    from alphapulldown.scripts import truncate_pickles
    app.run(truncate_pickles.main)

def create_notebook():
    from absl import app
    from alphapulldown.analysis_pipeline import create_notebook
    app.run(create_notebook.main)

def get_good_interpae():
    from absl import app
    from alphapulldown.analysis_pipeline import get_good_inter_pae
    app.run(get_good_inter_pae.main)
