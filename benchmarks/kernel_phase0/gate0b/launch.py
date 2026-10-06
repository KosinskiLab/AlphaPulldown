"""Gate 0b launcher: AlphaPulldown's batch predictor, with the fused triangle kernels bound in when G0B_FUSED=1.

Both arms go through this file, so they share the same process setup; only the fused arm calls fused_af3.install().
  python launch.py <run_structure_prediction_batch arguments>
"""
import os
import runpy
import sys

if __name__ == "__main__":
    if os.environ.get("G0B_FUSED") == "1":
        import fused_af3
        fused_af3.install()
    else:
        print("G0B_CONFIG stock", file=sys.stderr, flush=True)
    sys.argv = ["run_structure_prediction_batch"] + sys.argv[1:]
    runpy.run_module("alphapulldown.scripts.run_structure_prediction_batch", run_name="__main__", alter_sys=True)
