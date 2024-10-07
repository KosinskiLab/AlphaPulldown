from alphapulldown.utils import *
import io
import warnings
import subprocess
from absl import app
import os
from alphapulldown.utils.create_combinations import process_files

import argparse
from os import makedirs
from typing import Dict, List
from os.path import exists, join

from alphapulldown.folding_backend import FoldingBackendManager
from alphapulldown.objects import MultimericObject, MonomericObject
from alphapulldown.utils.modelling_setup import create_interactors
from absl import logging
logging.set_verbosity(logging.INFO)
import tempfile
import os
import subprocess
import numpy as np
from alphafold.data.tools import jackhmmer
from alphafold.data import parsers
from alphafold.data import pipeline_multimer
from alphafold.data import pipeline
from alphafold.data import msa_pairing
from alphafold.data import feature_processing
from pathlib import Path as plPath
from typing import List, Dict
from colabfold.batch import unserialize_msa, get_msa_and_templates, msa_to_str, build_monomer_feature
from alphapulldown.utils.multimeric_template_utils import (extract_multimeric_template_features_for_single_chain,
                                                     prepare_multimeric_template_meta_info)
from alphapulldown.utils.file_handling import temp_fasta_file
import json
import os
import pickle,gzip
import time
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.relax import relax
import numpy as np
import jax.numpy as jnp

