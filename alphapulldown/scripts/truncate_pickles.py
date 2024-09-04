#!/usr/bin/env python3

"""
Copies all contents from a source directory to a destination directory,
excluding specific keys from .pkl files if those keys are present.

Example usage: python truncate_pickles.py --src_dir=/path/to/source --dst_dir=/path/to/destination --number_of_threads=16
--keys_to_exclude=aligned_confidence_probs,distogram,masked_msa"""

import os
import shutil
import sys
import pickle
from concurrent.futures import ThreadPoolExecutor
from absl import app, flags, logging

# Define flags with default values and descriptions
flags.DEFINE_string('src_dir', None, 'Path to the source directory from which to copy files.', required=True)
flags.DEFINE_string('dst_dir', None, 'Path to the destination directory where files will be copied.', required=True)
flags.DEFINE_string('keys_to_exclude', 'aligned_confidence_probs,distogram,masked_msa',
                    'Comma-separated list of keys to exclude from .pkl files. Default keys are "aligned_confidence_probs,distogram,masked_msa".')
flags.DEFINE_integer('number_of_threads', 1, 'Number of threads to use for execution. Default is 1 (serial execution).')

from absl.flags import FLAGS

def copy_file(src_file_path, dst_file_path, keys_to_exclude):
    if src_file_path.endswith('.pkl'):
        try:
            with open(src_file_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and any(key in data for key in keys_to_exclude):
                for key in keys_to_exclude:
                    data.pop(key, None)

                with open(dst_file_path, 'wb') as f:
                    pickle.dump(data, f)
                    logging.info(f'{src_file_path} Copied without keys {keys_to_exclude} to {dst_file_path}')
            else:
                shutil.copy2(src_file_path, dst_file_path)
        except Exception as e:
            logging.error(f"Error processing {src_file_path}: {e}")
    else:
        shutil.copy2(src_file_path, dst_file_path)

def copy_contents(src_dir, dst_dir, keys_to_exclude, number_of_threads=1):
    keys_to_exclude = keys_to_exclude.split(",")
    tasks = []

    with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        for root, _, files in os.walk(src_dir):
            rel_path = os.path.relpath(root, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(dst_path, exist_ok=True)

            for file in files:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_path, file)

                if os.path.exists(dst_file_path):
                    continue

                if number_of_threads > 1:
                    task = executor.submit(copy_file, src_file_path, dst_file_path, keys_to_exclude)
                    tasks.append(task)
                else:
                    copy_file(src_file_path, dst_file_path, keys_to_exclude)

    # If running with multiple threads, wait for all tasks to complete
    if number_of_threads > 1:
        for task in tasks:
            task.result()  # This will re-raise any exception that occurred in the task

def main(argv):
    del argv  # Unused by main, but required by app.run.
    if not os.path.isdir(FLAGS.src_dir):
        logging.error(f"Input directory '{FLAGS.src_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(FLAGS.dst_dir, exist_ok=True)
    copy_contents(FLAGS.src_dir, FLAGS.dst_dir, FLAGS.keys_to_exclude, FLAGS.number_of_threads)

if __name__ == '__main__':
    app.run(main)
