"""Collect an af3_integration campaign into REPORT.md and REPORT.json. Safe to re-run; it never edits run directories.

- Gates: the latest layer-gate (runs/<gpu>/layers/hooks/<job>/checks.json) and bit-identity
  (runs/<gpu>/identity/paired/<job>/identity.json) job per card.
- Accuracy, the pre-registered paired rule. Per (card, fold), stock (`off`) and fused (`on`) are paired by seed. A fold is
  confident if its ranking score is >= 0.7 in every stock seed. Confident folds pass if the mean |delta ranking| over the
  paired seeds is <= 0.02 and the mean |delta DockQ| <= 0.05. For other folds the report says whether every fused value lies
  within the stock seeds' range. A fold is flagged follow_up when one paired |delta DockQ| > 0.2, when a paired ranking is
  >= 0.7 in one arm only, or when a confident fold misses the paired thresholds. Each trigger names its outlier side: `stock`
  (that stock seed lies outside the other stock seeds' range: seed-to-seed variation), `fused` (the fused value lies outside
  the stock seeds' range), `both` or `neither`.
- Capacity: per card and speed arm, the largest fold that completed and the first fold that ran out of memory. The ladder
  ends with capacity probes, so an out-of-memory fold there is a capacity limit, not a failure.
- Speed: warm speed-ups from complete, finite runs with zero warm compilation on the same GPU model.
  collect.py CAMPAIGN BENCH DOCKQ
"""
import csv
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect import parse_af3, score_dockq  # noqa: E402  the harness root's collect.py

STOCK, FUSED = 'off', 'on'
CONFIDENT = 0.7              # ranking score every stock seed must reach for a fold to be confident
MAX_MEAN_DELTA_RANKING = 0.02
MAX_MEAN_DELTA_DOCKQ = 0.05
FOLLOW_UP_DELTA_DOCKQ = 0.2  # any single paired |delta DockQ| above this is a follow-up
MIN_STOCK_SEEDS, MIN_PAIRED_SEEDS = 3, 2
OOM = re.compile(r'RESOURCE_EXHAUSTED|Out of memory')


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


def fold_tokens(fold):
    """Ladder order: s0164 -> 164, r3584 -> 3584."""
    digits = re.sub(r'\D', '', fold)
    return int(digits) if digits else 0


def outside(value, values):
    return bool(values) and not min(values) <= value <= max(values)


def outlier_side(stock, fused, seed, metric):
    """Which member of the pair (seed) is the outlier on metric: the stock seed against the other stock seeds' range, the fused
    value against the range of all stock seeds."""
    others = [r[metric] for s, r in stock.items() if s != seed]
    stock_out = outside(stock[seed][metric], others)
    fused_out = outside(fused[seed][metric], [r[metric] for r in stock.values()])
    return {(True, False): 'stock', (False, True): 'fused', (True, True): 'both'}.get((stock_out, fused_out), 'neither')


def accuracy_verdict(gpu, fold, stock, fused):
    """stock, fused: {seed: row} of ok rows with ranking and DockQ."""
    seeds = sorted(set(stock) & set(fused))
    entry = dict(gpu=gpu, fold=fold, stock_seeds=sorted(stock), fused_seeds=sorted(fused), paired_seeds=seeds,
                 complete=len(stock) >= MIN_STOCK_SEEDS and len(seeds) >= MIN_PAIRED_SEEDS)
    if not entry['complete']:
        return dict(entry, verdict='incomplete')
    pairs = [dict(seed=s, **{f'{arm}_{m}': rows[s][m] for arm, rows in (('stock', stock), ('fused', fused))
                              for m in ('ranking', 'dockq')}) for s in seeds]
    for p in pairs:
        p.update(delta_ranking=p['fused_ranking'] - p['stock_ranking'], delta_dockq=p['fused_dockq'] - p['stock_dockq'])
    mean = {m: sum(abs(p[f'delta_{m}']) for p in pairs) / len(pairs) for m in ('ranking', 'dockq')}
    confident = all(r['ranking'] >= CONFIDENT for r in stock.values())
    entry.update(confident=confident, pairs=pairs, mean_abs_delta_ranking=mean['ranking'], mean_abs_delta_dockq=mean['dockq'],
                 stock_range={m: [min(r[m] for r in stock.values()), max(r[m] for r in stock.values())] for m in ('ranking', 'dockq')})
    triggers = []
    for p in pairs:
        if abs(p['delta_dockq']) > FOLLOW_UP_DELTA_DOCKQ:
            triggers.append(dict(trigger='dockq_jump', seed=p['seed'], metric='dockq', delta=p['delta_dockq'],
                                 outlier=outlier_side(stock, fused, p['seed'], 'dockq')))
        if (p['stock_ranking'] >= CONFIDENT) != (p['fused_ranking'] >= CONFIDENT):
            triggers.append(dict(trigger='confidence_crossing', seed=p['seed'], metric='ranking', delta=p['delta_ranking'],
                                 outlier=outlier_side(stock, fused, p['seed'], 'ranking')))
    if confident:
        limits = dict(ranking=MAX_MEAN_DELTA_RANKING, dockq=MAX_MEAN_DELTA_DOCKQ)
        entry['paired_pass'] = all(mean[m] <= limits[m] for m in limits)
        for m in (m for m in limits if mean[m] > limits[m]):       # attributed to the pair that moved most
            worst = max(pairs, key=lambda p: abs(p[f'delta_{m}']))
            triggers.append(dict(trigger=f'mean_delta_{m}', seed=worst['seed'], metric=m, delta=worst[f'delta_{m}'],
                                 outlier=outlier_side(stock, fused, worst['seed'], m)))
        verdict = 'pass'
    else:
        entry['fused_within_stock_range'] = {m: all(not outside(r[m], [s[m] for s in stock.values()]) for r in fused.values())
                                             for m in ('ranking', 'dockq')}
        verdict = 'within_stock_range' if all(entry['fused_within_stock_range'].values()) else 'outside_stock_range'
    sides = {t['outlier'] for t in triggers}
    entry.update(follow_up=bool(triggers), triggers=triggers, verdict='follow_up' if triggers else verdict,
                 outlier=(sides.pop() if len(sides) == 1 else 'mixed') if triggers else None)
    return entry


def accuracy_verdicts(rows):
    acc = [r for r in rows if r['suite'] == 'accuracy']
    out = []
    for gpu, fold in sorted({(r['gpu'], r['fold']) for r in acc}):
        arm = {a: {r['seed']: r for r in acc if r['gpu'] == gpu and r['fold'] == fold and r['arm'] == a and r['status'] == 'ok'
                   and r.get('ranking') is not None and r.get('dockq') is not None} for a in (STOCK, FUSED)}
        out.append(accuracy_verdict(gpu, fold, arm[STOCK], arm[FUSED]))
    return out


def capacity(rows):
    """Per card and speed arm: the largest fold that completed and the first fold that ran out of memory."""
    speed = [r for r in rows if r['suite'] == 'speed']
    arms = []
    for gpu, arm in sorted({(r['gpu'], r['arm']) for r in speed}):
        trials = sorted((r for r in speed if r['gpu'] == gpu and r['arm'] == arm), key=lambda r: fold_tokens(r['fold']))
        done = [r['fold'] for r in trials if r['status'] == 'ok']
        oom = [r['fold'] for r in trials if r['oom']]
        arms.append(dict(gpu=gpu, arm=arm, largest_completed=done[-1] if done else None, first_oom=oom[0] if oom else None,
                         failed=[r['fold'] for r in trials if r['status'] != 'ok' and not r['oom']]))
    cards = []
    for gpu in sorted({a['gpu'] for a in arms}):
        by = {a['arm']: a for a in arms if a['gpu'] == gpu}
        if STOCK in by and FUSED in by:
            s, f = by[STOCK], by[FUSED]
            cards.append(dict(gpu=gpu, unchanged=(s['largest_completed'], s['first_oom']) == (f['largest_completed'], f['first_oom']),
                              largest_completed=s['largest_completed'], first_oom=s['first_oom']))
    return dict(arms=arms, cards=cards)


def gates(root):
    """Latest layer-gate and identity job per card, with its exit code (RESULT.txt) and checks."""
    out = {}
    for suite, arm, name in (('layers', 'hooks', 'checks.json'), ('identity', 'paired', 'identity.json')):
        for card in sorted(p for p in (root/'runs').glob('*') if p.is_dir()):
            jobs = sorted((p for p in (card/suite/arm).glob('*') if p.is_dir()), key=lambda p: int(p.name) if p.name.isdigit() else 0)
            if not jobs:
                continue
            job = jobs[-1]
            result = dict(line.split('=', 1) for line in (job/'RESULT.txt').read_text().splitlines() if '=' in line) \
                if (job/'RESULT.txt').exists() else {}
            try:
                checks = json.loads((job/name).read_text())
            except (OSError, ValueError):                           # not written, or the check crashed mid-output
                checks = None
            out.setdefault(card.name, {})[suite] = dict(job=job.name, exit_code=result.get('exit_code'), checks=checks,
                                                        passed=checks is not None and result.get('exit_code') == '0')
    return out


def collect(root, bench, dockq):
    rows = []
    for log in sorted(root.glob('runs/*/*/*/*/*/predict.log')):
        tag = log.parent
        job = tag.parent
        gpu, suite, requested_arm = job.parts[-4:-1]
        arm, fold, seed = re.fullmatch(r'([^_]+)_(.+)_s(\d+)', tag.name).groups()
        records = [json.loads(line) for line in (tag/'forward.jsonl').read_text().splitlines()] if (tag/'forward.jsonl').exists() else []
        text = log.read_text(errors='replace')
        parsed = parse_af3(text)
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
                   status='ok' if ok else 'failed_or_incomplete', oom=not ok and bool(OOM.search(text)), warm_valid=warm_ok,
                   warm_seconds=records[1]['seconds'] if warm_ok else None,
                   first_seconds=records[0]['seconds'] if records else None,
                   compile_seconds=[r['jit_s'] for r in parsed],
                   device=records[0]['device'] if records else None,
                   peak_allocator_bytes=max((r['memory'].get('peak_bytes_in_use', 0) for r in records if r['memory']), default=0),
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
    accuracy = accuracy_verdicts(rows)
    limits = capacity(rows)
    gate = gates(root)
    result = dict(gates=gate, accuracy=accuracy, capacity=limits, speedups=pairs, rows=rows, superseded_attempts=superseded)
    (root/'REPORT.json').write_text(json.dumps(result, indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(report_md(rows, pairs, accuracy, limits, gate))+'\n')
    return result


def fmt(value, digits=3):
    return '–' if value is None else f'{value:.{digits}f}'


def report_md(rows, pairs, accuracy, limits, gate):
    md = ['# AF3 integrated validation', '', '## Gates (latest job per card)', '',
          '| Card | Layer gate | Bit identity |', '| --- | --- | --- |']
    def cell(g):
        if not g:
            return 'not run'
        checks = ', '.join(f'{k}={v}' for k, v in (g['checks'] or {}).items()) or 'no checks file'
        return f"{'pass' if g['passed'] else 'FAIL'} (job {g['job']}, exit {g['exit_code']}; {checks})"
    md += [f"| {card} | {cell(g.get('layers'))} | {cell(g.get('identity'))} |" for card, g in sorted(gate.items())]
    md += ['', '## Accuracy (pre-registered paired rule)', '',
           f'Stock and fused paired by seed. Confident = ranking >= {CONFIDENT} in every stock seed; a confident fold passes '
           f'if mean |Δranking| <= {MAX_MEAN_DELTA_RANKING} and mean |ΔDockQ| <= {MAX_MEAN_DELTA_DOCKQ}. Other folds report '
           f'whether every fused value lies within the stock seed range. follow_up: a paired |ΔDockQ| > {FOLLOW_UP_DELTA_DOCKQ}, a '
           f'paired ranking >= {CONFIDENT} in one arm only, or a confident fold over the thresholds. Outlier `stock`: that stock '
           'seed lies outside the other stock seeds\' range (seed-to-seed variation); `fused`: the fused value lies outside the '
           'stock seeds\' range.', '']
    for gpu in sorted({a['gpu'] for a in accuracy}):
        mine = [a for a in accuracy if a['gpu'] == gpu]
        confident = [a for a in mine if a.get('confident')]
        passed = [a['fold'] for a in confident if a['verdict'] == 'pass']
        follow = [f"{a['fold']} ({a['outlier']} outlier)" for a in mine if a['verdict'] == 'follow_up']
        other = [a for a in mine if a['complete'] and not a['confident'] and a['verdict'] != 'follow_up']
        incomplete = [a['fold'] for a in mine if not a['complete']]
        md.append(f"- **{gpu}**: {len(passed)}/{len(confident)} confident folds pass; follow-up: {', '.join(follow) or 'none'}; "
                  f"non-confident within stock range: {sum(a['verdict'] == 'within_stock_range' for a in other)}/{len(other)}"
                  + (f"; incomplete: {', '.join(incomplete)}" if incomplete else ''))
    md += ['', '| Card | Fold | Confident | Seeds | mean abs Δranking | mean abs ΔDockQ | Verdict | Triggers (seed: outlier) |',
           '| --- | --- | --- | --- | ---: | ---: | --- | --- |']
    for a in accuracy:
        triggers = '; '.join(f"{t['trigger']} s{t['seed']} Δ{t['delta']:+.3f}: {t['outlier']}" for t in a.get('triggers', []))
        verdict = a['verdict']
        if verdict == 'outside_stock_range':
            verdict += f" ({', '.join(m for m, ok in a['fused_within_stock_range'].items() if not ok)})"
        md.append(f"| {a['gpu']} | {a['fold']} | {a.get('confident', '–')} | {','.join(map(str, a['paired_seeds']))} | "
                  f"{fmt(a.get('mean_abs_delta_ranking'))} | {fmt(a.get('mean_abs_delta_dockq'))} | {verdict} | {triggers} |")
    md += ['', '## Capacity (speed ladder)', '',
           'The ladder ends with capacity probes (3,584 and 5,376 tokens); a fold that runs out of memory there is the card\'s '
           'capacity limit, not a failure. A failed fold stops the larger folds of its arm.', '']
    for a in limits['arms']:
        line = f"- {a['gpu']} {a['arm']}: largest completed {a['largest_completed'] or 'none'}"
        line += f"; capacity limit at {a['first_oom']} (out of memory)" if a['first_oom'] else '; no capacity limit reached'
        md.append(line + (f"; FAILED (not out of memory): {', '.join(a['failed'])}" if a['failed'] else ''))
    for c in limits['cards']:
        if c['unchanged']:
            md.append(f"- **{c['gpu']}: capacity unchanged** (stock and fused both complete {c['largest_completed']}"
                      + (f" and run out of memory at {c['first_oom']})" if c['first_oom'] else ')'))
        else:
            md.append(f"- **{c['gpu']}: capacity differs** between stock and fused (see the lines above)")
    md += ['', '## Speed', '', 'Only complete, finite runs with zero warm compile time and the same GPU model form speed ratios.', '',
           '| Card | Fold | Arm | Warm speed-up |', '| --- | --- | --- | ---: |']
    md += [f"| {r['gpu']} | {r['fold']} | {r['arm']} | {r['speedup']:.3f} |" for r in pairs]
    limit = [r for r in rows if r['suite'] == 'speed' and r['oom']]          # elsewhere an OOM is an ordinary failure
    md += ['', f'{len(rows)} prediction trials; {len(limit)} reached a capacity limit (speed ladder, out of memory); '
           f'{sum(r["status"] != "ok" for r in rows) - len(limit)} failed or incomplete otherwise.',
           'Memory, dispatch, compilation, DockQ, per-seed pairs and gate checks are in REPORT.json.',
           'A missing comparison is not a pass. Queue state and exit codes are in sacct.txt and per-job RESULT.txt.']
    return md


if __name__ == '__main__':
    collect(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
