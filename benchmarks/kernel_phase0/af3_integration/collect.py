"""Collect completed trials only; retain failed/unfinished trials without speed claims."""
import csv
import json
from pathlib import Path
import re
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect import parse_af3, score_dockq


def latest_attempts(rows):
    """One row per trial (card, suite, requested arm, arm, fold, seed). A supervisor retry runs in a new job directory, so a
    trial can appear twice: an ok row beats a failed one, then the later job wins. Superseded rows are returned apart."""
    best = {}
    for r in rows:
        key = (r['gpu'], r['suite'], r['requested_arm'], r['arm'], r['fold'], r['seed'])
        rank = (r['status'] == 'ok', int(r['job']) if r['job'].isdigit() else 0)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, r)
    kept = {id(r) for _, r in best.values()}
    return [r for r in rows if id(r) in kept], [r for r in rows if id(r) not in kept]


def collect(root, bench, dockq):
    rows = []
    for log in sorted(root.glob('runs/*/*/*/*/*/predict.log')):
        tag = log.parent
        job = tag.parent
        gpu, suite, requested_arm = job.parts[-4:-1]
        arm, fold, seed = re.fullmatch(r'([^_]+)_(.+)_s(\d+)', tag.name).groups()
        records = [json.loads(line) for line in (tag/'forward.jsonl').read_text().splitlines()] if (tag/'forward.jsonl').exists() else []
        parsed = parse_af3(log.read_text(errors='replace'))
        rc = (tag/'exit_code').read_text().strip() if (tag/'exit_code').exists() else None
        expected = 1 if suite == 'accuracy' else 2
        ok = rc == '0' and len(records) == len(parsed) == expected and all(r['finite'] for r in records)
        warm_ok = ok and len(parsed) == 2 and parsed[1]['jit_s'] == 0 and all(r['compile_logging'] for r in records)
        models = sorted((tag/'rep1').glob('*_model.cif'))
        rankings = list((tag/'rep1').glob('ranking_scores.csv'))
        ranking = max((float(r['ranking_score']) for r in csv.DictReader(rankings[0].open())), default=None) if rankings else None
        memory_file = tag/'gpu_memory_mib.txt'
        memory = [int(v) for v in memory_file.read_text().splitlines() if v.strip().isdigit()] if memory_file.exists() else []
        row = dict(gpu=gpu, suite=suite, arm=arm, fold=fold, seed=int(seed), job=job.name,
                   status='ok' if ok else 'failed_or_incomplete', warm_valid=warm_ok,
                   warm_seconds=records[1]['seconds'] if warm_ok else None,
                   first_seconds=records[0]['seconds'] if records else None,
                   compile_seconds=[r['jit_s'] for r in parsed],
                   device=records[0]['device'] if records else None,
                   peak_allocator_bytes=max((r['memory'].get('peak_bytes_in_use',0) for r in records if r['memory']), default=0),
                   peak_gpu_mib=max(memory) if memory else None,
                   dispatch=records[-1]['dispatch'] if records else None,
                   ranking=ranking, model=str(models[0]) if models else None)
        row['requested_arm'] = requested_arm
        rows.append(row)
    rows, superseded = latest_attempts(rows)
    score_dockq(rows, bench/'inputs/natives', dockq, root/'dockq_cache.json')
    pairs = []
    for fast in rows:
        if fast['suite'] != 'speed' or fast['arm'] not in ('on', 'trimul', 'attention') or not fast['warm_valid']:
            continue
        stock = [r for r in rows if r['suite'] == 'speed' and r['arm'] == 'off' and r['gpu'] == fast['gpu']
                 and r['fold'] == fast['fold'] and r['seed'] == fast['seed'] and r['device'] == fast['device'] and r['warm_valid']]
        if len(stock) == 1:
            pairs.append(dict(gpu=fast['gpu'], fold=fast['fold'], arm=fast['arm'],
                              speedup=stock[0]['warm_seconds']/fast['warm_seconds']))
    accuracy = []
    for gpu, fold in sorted({(r['gpu'], r['fold']) for r in rows if r['suite'] == 'accuracy'}):
        group = [r for r in rows if r['gpu'] == gpu and r['fold'] == fold and r['suite'] == 'accuracy']
        base = [r for r in group if r['arm'] == 'off' and r['status'] == 'ok' and r.get('dockq') is not None]
        fast = [r for r in group if r['arm'] == 'on' and r['status'] == 'ok' and r.get('dockq') is not None]
        complete = len(base) >= 3 and len(fast) >= 2
        checks = {}
        if complete:
            for metric in ('ranking', 'dockq'):
                lo, hi = min(r[metric] for r in base), max(r[metric] for r in base)
                checks[metric+'_within_seed_range'] = all(lo <= r[metric] <= hi for r in fast)
        accuracy.append(dict(gpu=gpu, fold=fold, complete=complete, **checks))
    result = dict(rows=rows, speedups=pairs, accuracy=accuracy, superseded_attempts=superseded)
    (root/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    md = ['# AF3 integrated validation', '', 'Only complete, finite runs with zero warm compile time and the same GPU model form speed ratios.', '',
          '| Card | Fold | Arm | Warm speed-up |', '| --- | --- | --- | ---: |']
    md += [f"| {r['gpu']} | {r['fold']} | {r['arm']} | {r['speedup']:.3f} |" for r in pairs]
    md += ['', f'{len(rows)} prediction trials; {sum(r["status"] != "ok" for r in rows)} failed or incomplete.',
           'Memory, dispatch, compilation, DockQ and seed comparisons are in REPORT.json.',
           'A missing comparison is not a pass. Queue state and exit codes are in sacct.txt and per-job RESULT.txt.']
    (root/'REPORT.md').write_text('\n'.join(md)+'\n')
    return result

if __name__ == '__main__':
    collect(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
