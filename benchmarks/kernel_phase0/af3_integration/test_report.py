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


# ---------------------------------------------------------------------------------------------- paired accuracy rule
def arm_rows(rankings, dockqs, seeds=None):
    return {s: dict(ranking=r, dockq=d) for s, r, d in zip(seeds or range(len(rankings)), rankings, dockqs)}


def verdict(stock, fused):
    return report.accuracy_verdict('a40', 'acc_X', arm_rows(*stock), arm_rows(*fused))


def test_confident_fold_passes_within_paired_thresholds():
    a = verdict(([0.90, 0.91, 0.92], [0.80, 0.81, 0.79]), ([0.905, 0.915], [0.81, 0.80]))
    assert a['confident'] and a['paired_pass'] and a['verdict'] == 'pass' and not a['follow_up']
    assert abs(a['mean_abs_delta_ranking'] - 0.005) < 1e-9 and abs(a['mean_abs_delta_dockq'] - 0.01) < 1e-9


def test_confident_fold_over_threshold_names_a_stock_seed_outlier():
    # A40 9HH5: stock seed 1 ranks 0.853, the other stock seeds 0.754-0.770; fused seed 1 sits among them.
    a = verdict(([0.770, 0.853, 0.754], [0.007] * 3), ([0.769, 0.755], [0.007] * 2))
    assert a['confident'] and not a['paired_pass'] and a['verdict'] == 'follow_up' and a['outlier'] == 'stock'
    assert [(t['trigger'], t['seed'], t['outlier']) for t in a['triggers']] == [('mean_delta_ranking', 1, 'stock')]


def test_confident_fold_with_a_fused_outlier():
    a = verdict(([0.90, 0.90, 0.91], [0.80, 0.80, 0.81]), ([0.90, 0.90], [0.80, 0.50]))
    assert a['verdict'] == 'follow_up' and a['outlier'] == 'fused'
    assert {t['trigger'] for t in a['triggers']} == {'dockq_jump', 'mean_delta_dockq'}


def test_low_confidence_flip_is_a_follow_up_on_the_stock_side():
    # A40 8JTK: stock seed 1 collapsed (0.359 / DockQ 0.008), fused seed 1 matches the other stock seeds.
    a = verdict(([0.919, 0.359, 0.902], [0.692, 0.008, 0.779]), ([0.916, 0.917], [0.691, 0.682]))
    assert not a['confident'] and a['fused_within_stock_range'] == {'ranking': True, 'dockq': True}
    assert [(t['trigger'], t['seed'], t['outlier']) for t in a['triggers']] == [
        ('dockq_jump', 1, 'stock'), ('confidence_crossing', 1, 'stock')]
    assert a['verdict'] == 'follow_up' and a['outlier'] == 'stock'


def test_non_confident_fold_reports_the_stock_range_only():
    inside = verdict(([0.55, 0.61, 0.53], [0.88, 0.90, 0.85]), ([0.56, 0.60], [0.89, 0.86]))
    assert inside['verdict'] == 'within_stock_range' and not inside['follow_up']
    outside = verdict(([0.55, 0.61, 0.53], [0.88, 0.90, 0.85]), ([0.56, 0.62], [0.89, 0.86]))
    assert outside['verdict'] == 'outside_stock_range' and outside['fused_within_stock_range'] == {'ranking': False, 'dockq': True}


def test_pairs_need_three_stock_and_two_paired_seeds():
    assert verdict(([0.9, 0.9], [0.8, 0.8]), ([0.9, 0.9], [0.8, 0.8]))['verdict'] == 'incomplete'
    a = report.accuracy_verdict('a40', 'acc_X', arm_rows([0.9] * 3, [0.8] * 3), arm_rows([0.9], [0.8], seeds=[2]))
    assert a['verdict'] == 'incomplete' and a['paired_seeds'] == [2]


# ---------------------------------------------------------------------------------------------- synthetic campaigns
def trial(root, gpu, suite, arm, job, fold, seed=0, rc=0, preds=2, log_tail='', ranking=None):
    run = root/f'runs/{gpu}/{suite}/{arm}/{job}/{arm}_{fold}_s{seed}'
    run.mkdir(parents=True)
    record = dict(seconds=10, finite=True, compile_logging=True, device='NVIDIA A40', memory={}, dispatch={})
    (run/'forward.jsonl').write_text((json.dumps(record)+'\n') * (preds if rc == 0 else 0))
    (run/'exit_code').write_text(str(rc))
    log = ''.join(f'Running model inference for seed {seed}\nModel inference for seed {seed} took 10.00 seconds.\n'
                  for _ in range(preds if rc == 0 else 0))
    (run/'predict.log').write_text(log + log_tail)
    if ranking is not None:
        (run/'rep1').mkdir()
        (run/'rep1/ranking_scores.csv').write_text(f'seed,sample,ranking_score\n{seed},0,{ranking}\n')
        (run/'rep1/x_model.cif').write_text('data_x\n')
    return run


def test_collect_pairs_accuracy_by_seed_and_writes_the_fold_table(tmp_path):
    cache = {}
    for arm, values in (('off', [(0.919, 0.692), (0.359, 0.008), (0.902, 0.779)]), ('on', [(0.916, 0.691), (0.917, 0.682)])):
        for seed, (ranking, dockq) in enumerate(values):
            run = trial(tmp_path, 'a40', 'accuracy', arm, 10 + (arm == 'on'), 'acc_8JTK', seed, preds=1, ranking=ranking)
            cache[str(run/'rep1/x_model.cif')] = dockq
    (tmp_path/'dockq_cache.json').write_text(json.dumps(cache))           # no DockQ call
    result = report.collect(tmp_path, tmp_path, 'unused')
    [fold] = result['accuracy']
    assert fold['verdict'] == 'follow_up' and fold['outlier'] == 'stock' and fold['paired_seeds'] == [0, 1]
    md = (tmp_path/'REPORT.md').read_text()
    assert '| a40 | acc_8JTK | False | 0,1 | 0.281 | 0.338 | follow_up |' in md
    assert 'follow-up: acc_8JTK (stock outlier)' in md


OOM_TAIL = 'E1006 run_structure_prediction_batch.py:34] Prediction job r3584_r1 failed: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 31.89GiB.\n'


def test_capacity_limit_is_not_a_failure(tmp_path):
    for arm, job in (('off', 20), ('on', 21)):
        for fold in ('s0164', 's2546'):
            trial(tmp_path, 'a40', 'speed', arm, job, fold)
        trial(tmp_path, 'a40', 'speed', arm, job, 'r3584', rc=1, log_tail=OOM_TAIL)
    trial(tmp_path, 'h100', 'speed', 'off', 30, 's0164')
    trial(tmp_path, 'h100', 'speed', 'off', 30, 's0357', rc=1, log_tail='Traceback: ValueError: something else\n')
    trial(tmp_path, 'h100', 'speed', 'on', 31, 's0164')
    trial(tmp_path, 'h100', 'speed', 'on', 31, 's0357', rc=1, log_tail=OOM_TAIL)
    result = report.collect(tmp_path, tmp_path, 'unused')
    arms = {(a['gpu'], a['arm']): a for a in result['capacity']['arms']}
    assert arms['a40', 'off'] == dict(gpu='a40', arm='off', largest_completed='s2546', first_oom='r3584', failed=[])
    assert arms['h100', 'off']['failed'] == ['s0357'] and arms['h100', 'off']['first_oom'] is None
    cards = {c['gpu']: c['unchanged'] for c in result['capacity']['cards']}
    assert cards == {'a40': True, 'h100': False}
    md = (tmp_path/'REPORT.md').read_text()
    assert '- a40 on: largest completed s2546; capacity limit at r3584 (out of memory)' in md
    assert '**a40: capacity unchanged** (stock and fused both complete s2546 and run out of memory at r3584)' in md
    assert 'FAILED (not out of memory): s0357' in md and '**h100: capacity differs**' in md
    assert '3 reached a capacity limit (out of memory); 1 failed or incomplete otherwise' in md


def test_gates_report_the_latest_job_per_card(tmp_path):
    def gate(suite, arm, job, name, checks, rc):
        run = tmp_path/f'runs/a40/{suite}/{arm}/{job}'
        run.mkdir(parents=True)
        (run/name).write_text(json.dumps(checks))
        (run/'RESULT.txt').write_text(f'exit_code={rc}\njob={job}\n')
    gate('layers', 'hooks', 9, 'checks.json', dict(fused_cells=60, failed=6), 1)
    gate('layers', 'hooks', 12, 'checks.json', dict(fused_cells=66, failed=0), 0)
    gate('identity', 'paired', 13, 'identity.json', dict(complete=True, fused_active=False), 1)
    g = report.collect(tmp_path, tmp_path, 'unused')['gates']['a40']
    assert g['layers']['job'] == '12' and g['layers']['passed'] and not g['identity']['passed']
    md = (tmp_path/'REPORT.md').read_text()
    assert '| a40 | pass (job 12, exit 0; fused_cells=66, failed=0) | FAIL (job 13, exit 1; complete=True, fused_active=False) |' in md
