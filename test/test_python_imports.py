import os, logging, csv,sys
from alphafold.data.templates import (_read_file, 
                                      _extract_template_features,
                                      _build_query_to_hit_index_mapping)
from alphafold.data.templates import SingleHitResult
from alphafold.data import mmcif_parsing
from alphafold.data.mmcif_parsing import ParsingResult
from alphafold.data.parsers import TemplateHit
from typing import Optional
import shutil
import logging
import tempfile
import os
import contextlib
import numpy as np
from alphafold.data import parsers
from alphafold.data import pipeline_multimer
from alphafold.data import pipeline
from alphafold.data import msa_pairing
from alphafold.data import feature_processing
from pathlib import Path as plPath
from colabfold.batch import unserialize_msa, get_msa_and_templates, msa_to_str, build_monomer_feature
from alphapulldown.multimeric_template_utils import (extract_multimeric_template_features_for_single_chain,
                                                     prepare_multimeric_template_meta_info)

import itertools
from absl import app, logging
from alphapulldown.utils import (create_interactors, read_all_proteins, read_custom, make_dir_monomer_dictionary,
                                 load_monomer_objects, check_output_dir, create_model_runners_and_random_seed,
                                 create_and_save_pae_plots, post_prediction_process)
from alphapulldown.multimeric_template_utils import prepare_multimeric_template_meta_info
from itertools import combinations
from alphapulldown.objects import MultimericObject
import os
from pathlib import Path
from alphapulldown.predict_structure import predict, ModelsToRelax
from alphapulldown.utils import get_run_alphafold
from alphapulldown import __version__ as ap_version
from alphafold.version import __version__ as af_version
import json
from datetime import datetime
from alphafold.data.tools import jackhmmer
from alphapulldown.objects import ChoppedObject
from alphapulldown import __version__ as AP_VERSION
from alphafold.version import __version__ as AF_VERSION
import json
import os
import pickle
import logging
from alphapulldown.plot_pae import plot_pae
import alphafold
from alphafold.model import config
from alphafold.model import model
from alphafold.model import data
from alphafold.data import templates
import random
import subprocess
from alphafold.data import parsers
from pathlib import Path
import numpy as np
import sys
import datetime
import re
import hashlib
import glob
import importlib.util
