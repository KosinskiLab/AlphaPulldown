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


def fused_dispatch(attention='pallas', fused=True):
    """af3_fused_triangles.metadata() for a 256-token bucket."""
    if not fused:
        return {'fused_kernels': False}
    ops = {f'{op}_c{c}': {'implementation': attention if op == 'triangle_attention' else 'pallas', 'reason': ''}
           for op in ('triangle_multiplication', 'triangle_attention') for c in (128, 64)}
    return {'fused_kernels': True, 'operations': ops}


def identity_run(tmp_path, on_dispatch):
    for arm in ('baseline', 'off', 'on'):
        folder = tmp_path/f'{arm}_s0164_s0'
        folder.mkdir(exist_ok=True)
        row = dict(finite=True, hash='same' if arm != 'on' else 'fused',
                   dispatch=on_dispatch if arm == 'on' else fused_dispatch(fused=False))
        (folder/'forward.jsonl').write_text((json.dumps(row)+'\n')*2)
    return subprocess.run([sys.executable, str(HERE/'check_identity.py'), str(tmp_path)], capture_output=True).returncode


def test_identity_requires_full_outputs_and_enabled_fused_path(tmp_path):
    assert identity_run(tmp_path, fused_dispatch()) == 0
    assert identity_run(tmp_path, fused_dispatch('pallas_tokamax_core')) == 0
    (tmp_path/'on_s0164_s0/forward.jsonl').write_text('')
    assert subprocess.run([sys.executable, str(HERE/'check_identity.py'), str(tmp_path)], capture_output=True).returncode == 1


def test_identity_rejects_a_partly_dispatched_fused_path(tmp_path):
    partial = fused_dispatch()
    partial['operations']['triangle_attention_c64'] = {'implementation': 'default', 'reason': 'size_limit'}
    assert identity_run(tmp_path, partial) == 1


def layer_cells(tmp_path, row, negative):
    cells = tmp_path/'cells.jsonl'
    cells.write_text((json.dumps(row)+'\n')*66+json.dumps(negative)+'\n')
    return subprocess.run([sys.executable, str(HERE/'check_layers.py'), str(cells)], capture_output=True).returncode


def test_layer_gate_rejects_missing_leak_check(tmp_path):
    row = dict(arm='fused', status='ok', gate=True, det_ok=True, real_finite=True, mask='pad', pad_finite=True, leak_ok=True,
               integrated=True, fork_requested='pallas', fork_implementation='pallas', fork_reason='')
    negative = dict(row, arm='fused_biasT', gate=False)
    assert layer_cells(tmp_path, row, negative) == 0
    del row['leak_ok']
    assert layer_cells(tmp_path, row, negative) == 1


def test_layer_gate_rejects_the_fork_falling_back_to_stock(tmp_path):
    row = dict(arm='fused', status='ok', gate=True, det_ok=True, real_finite=True, mask='asym',
               integrated=True, fork_requested='pallas', fork_implementation='default', fork_reason='size_limit')
    negative = dict(row, arm='fused_biasT', gate=False)
    assert layer_cells(tmp_path, row, negative) == 1                            # stock output passes the gate trivially
    assert layer_cells(tmp_path, dict(row, fork_implementation='pallas', integrated=False), negative) == 1   # prototype classes
    assert layer_cells(tmp_path, dict(row, fork_implementation='pallas'), negative) == 0
    tokcore = dict(row, arm='fused_tokcore', fork_requested='pallas_tokamax_core', fork_implementation='pallas_tokamax_core')
    assert layer_cells(tmp_path, tokcore, negative) == 0
