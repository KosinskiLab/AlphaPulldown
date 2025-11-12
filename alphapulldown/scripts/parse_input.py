#!/usr/bin/env python
from absl import flags, app, logging
import json
from alphapulldown.utils.modelling_setup import parse_fold, create_custom_info
from alphapulldown_input_parser import generate_fold_specifications

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'input_list', None,
    'Path to input file list.')
flags.DEFINE_list(
    'features_directory', None,
    'Path to computed monomer features.')
flags.DEFINE_string(
    'protein_delimiter', '+',
    'Delimiter for proteins.')
flags.DEFINE_string(
    'output_prefix', None,
    'Prefix for output JSON files.')    

FLAGS = flags.FLAGS

def main(argv):
    specifications = generate_fold_specifications(
        input_files=FLAGS.input_list,
        delimiter=FLAGS.protein_delimiter,
        exclude_permutations=True,
    )
    parsed = parse_fold(specifications, FLAGS.features_directory, FLAGS.protein_delimiter)
    data = create_custom_info(parsed)

    with open(FLAGS.output_prefix + "data.json", 'w') as out_f:
        json.dump(data, out_f, indent=1)

app.run(main)
