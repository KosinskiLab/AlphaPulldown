"""Instrument the public AF3 batch CLI; no replacement model classes or kernels."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time

import jax
import numpy as np
from alphapulldown.folding_backend import alphafold3_backend as backend
from alphapulldown.prediction import fast_kernels

arm = os.environ['AF3_TEST_ARM']
original_settings = fast_kernels._af3_settings

def settings(device):
    result = original_settings(device)
    if arm == 'trimul':
        result['fused_triangle_attention'] = 'off'
    elif arm == 'attention':
        result['fused_triangle_multiplication'] = False
    return result

fast_kernels._af3_settings = settings
original_inference = backend.ModelRunner.run_inference
record_path = Path(os.environ['AF3_TEST_RECORD'])

def inference(self, example, key):
    start = time.perf_counter()
    output = original_inference(self, example, key)
    elapsed = time.perf_counter() - start
    digest = hashlib.sha256()
    finite = True
    for path, leaf in jax.tree_util.tree_flatten_with_path(output)[0]:
        digest.update(str(path).encode())
        if isinstance(leaf, bytes):
            digest.update(leaf)
        else:
            value = np.asarray(leaf)
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
            if np.issubdtype(value.dtype, np.inexact):
                finite &= bool(np.isfinite(value).all())
    tokens = int(example['token_index'].shape[0])
    record = dict(arm=arm, seconds=elapsed, hash=digest.hexdigest(), finite=finite,
                  compile_logging=bool(jax.config.jax_log_compiles),
                  device=self.device.device_kind, cc=str(self.device.compute_capability),
                  memory=self.device.memory_stats(),
                  versions={p: importlib.metadata.version(p) for p in ('jax', 'jaxlib', 'tokamax')},
                  dispatch=fast_kernels.af3_metadata(self.config.global_config,
                      requested_mode=self.kernel_mode, tokens=tokens, reason=self.kernel_reason))
    with record_path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    return output

backend.ModelRunner.run_inference = inference
sys.argv = ['run_structure_prediction_batch'] + sys.argv[1:]
runpy.run_module('alphapulldown.scripts.run_structure_prediction_batch', run_name='__main__', alter_sys=True)
