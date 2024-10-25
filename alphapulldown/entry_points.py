from absl import app
from alphapulldown.scripts import (
    create_individual_features,
    run_multimer_jobs,
    convert_to_modelcif,
    truncate_pickles
)
from alphapulldown.analysis_pipeline import (
    create_notebook,
)

def create_individual_features_entry():
    app.run(create_individual_features.main)

def run_multimer_jobs_entry():
    app.run(run_multimer_jobs.main)

def convert_to_modelcif_entry():
    app.run(convert_to_modelcif.main)

def truncate_pickles_entry():
    app.run(truncate_pickles.main)

def create_notebook_entry():
    app.run(create_notebook.main)