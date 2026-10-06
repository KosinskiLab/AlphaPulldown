import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('integrated_report', HERE/'collect.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_speed_requires_complete_compile_free_matching_device(tmp_path):
    bench = tmp_path/'bench'
    bench.mkdir()
    def trial(arm, device='H100', compile_s=0, rc=0):
        run = tmp_path/f'runs/h100/speed/{arm}/123/{arm}_s0896_s0'
        run.mkdir(parents=True, exist_ok=True)
        record = dict(seconds=10 if arm == 'off' else 8, finite=True, compile_logging=True, device=device, memory={}, dispatch={})
        (run/'forward.jsonl').write_text((json.dumps(record)+'\n')*2)
        (run/'exit_code').write_text(str(rc))
        log = 'Running model inference for seed 0\nModel inference for seed 0 took 10.00 seconds.\n'
        warm = f'Finished XLA compilation of jit(apply_fn) in {compile_s:.2f} sec\n' if compile_s else ''
        (run/'predict.log').write_text(log+'Running model inference for seed 0\n'+warm+'Model inference for seed 0 took 8.00 seconds.\n')
    trial('off'); trial('on')
    assert report.collect(tmp_path, bench, 'unused')['speedups'][0]['speedup'] == 1.25
    trial('on', compile_s=1)
    assert report.collect(tmp_path, bench, 'unused')['speedups'] == []
    trial('on', device='A100')
    assert report.collect(tmp_path, bench, 'unused')['speedups'] == []
    trial('on', rc=1)
    assert report.collect(tmp_path, bench, 'unused')['speedups'] == []


def test_identity_requires_full_outputs_and_enabled_fused_path(tmp_path):
    for arm in ('baseline', 'off', 'on'):
        folder = tmp_path/f'{arm}_s0164_s0'
        folder.mkdir()
        row = dict(finite=True, hash='same' if arm != 'on' else 'fused', dispatch={'fused_kernels':arm == 'on'})
        (folder/'forward.jsonl').write_text((json.dumps(row)+'\n')*2)
    assert subprocess.run([sys.executable, str(HERE/'check_identity.py'), str(tmp_path)], capture_output=True).returncode == 0
    (tmp_path/'on_s0164_s0/forward.jsonl').write_text('')
    assert subprocess.run([sys.executable, str(HERE/'check_identity.py'), str(tmp_path)], capture_output=True).returncode == 1


def test_layer_gate_rejects_missing_leak_check(tmp_path):
    row = dict(arm='fused', status='ok', gate=True, det_ok=True, real_finite=True,
               mask='pad', pad_finite=True, leak_ok=True)
    negative = dict(row, arm='fused_biasT', gate=False)
    cells = tmp_path/'cells.jsonl'
    cells.write_text((json.dumps(row)+'\n')*66+json.dumps(negative)+'\n')
    assert subprocess.run([sys.executable, str(HERE/'check_layers.py'), str(cells)], capture_output=True).returncode == 0
    del row['leak_ok']
    cells.write_text((json.dumps(row)+'\n')*66+json.dumps(negative)+'\n')
    assert subprocess.run([sys.executable, str(HERE/'check_layers.py'), str(cells)], capture_output=True).returncode == 1
