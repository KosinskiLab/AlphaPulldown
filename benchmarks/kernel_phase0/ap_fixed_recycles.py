#!/usr/bin/env python
"""Run AlphaPulldown's resident batch command with recycle early-stopping disabled.

Benchmark harness only; AlphaPulldown's code is untouched. Two wrappers, both outside it:

- AlphaFold-Multimer stops recycling inside the graph once structures move less than
  0.5 A, so the work done depends on the input. The ColabFold arms get
  --recycle-early-stop-tolerance 0; this sets the same value in every multimer config
  AlphaPulldown builds, so every arm runs exactly --num_cycle recycles.
- Each RunModel.predict call prints one `[bench] predict` line with its wall time (the
  call returns materialised arrays) and the recycles it ran, so result pickles, which
  reach gigabytes at large sizes, need not be kept.
"""

import os
import runpy
import sys
import time

import numpy as np
from alphafold.model import config as af_config
from alphafold.model import model as af_model

_model_config = af_config.model_config
_predict = af_model.RunModel.predict


# AP_FUSED_KERNELS=1 (arm ap_fast): switch on the fork's colabfold-kernels hooks, which the
# job puts ahead of the image's alphafold on PYTHONPATH (exp/af2-fused-kernels).
FUSED = os.environ.get("AP_FUSED_KERNELS") == "1"
if FUSED:
    import jax

    _cc = jax.devices()[0].compute_capability
    _cc = int(round(float(_cc) * 10)) if "." in str(_cc) else int(_cc)
    print(f"[bench] fused_kernels=on compute_capability={_cc}", flush=True)


def _fixed_recycles(name, *args, **kwargs):
    cfg = _model_config(name, *args, **kwargs)
    if "multimer" in name:
        cfg.model.recycle_early_stop_tolerance = 0.0
    if FUSED:
        cfg.model.global_config.use_pallas = True
        cfg.model.global_config.compute_capability = _cc
    return cfg


def _timed_predict(self, *args, **kwargs):
    start = time.perf_counter()
    out = _predict(self, *args, **kwargs)
    seconds = time.perf_counter() - start
    result = out[0] if isinstance(out, tuple) else out
    recycles = result.get("num_recycles") if hasattr(result, "get") else None
    recycles = "NA" if recycles is None else int(np.asarray(recycles))
    print(f"[bench] predict seconds={seconds:.3f} num_recycles={recycles}", flush=True)
    return out


af_config.model_config = _fixed_recycles
af_model.RunModel.predict = _timed_predict
sys.argv[0] = "run_structure_prediction_batch"
runpy.run_module("alphapulldown.scripts.run_structure_prediction_batch", run_name="__main__")
