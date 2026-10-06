"""Bit identity on one fold: 86b9ea3 (baseline) and the fork with the kernels off give the same forward results; the fused
path is repeatable and dispatches every triangle operation it records (AlphaPulldown's af3_fused_triangles.metadata:
operations triangle_{multiplication,attention}_c{128,64}, each with the `implementation` that ran)."""
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
rows = {}
for arm in ('baseline', 'off', 'on'):
    paths = list(root.glob(f'{arm}_*/forward.jsonl'))
    rows[arm] = [json.loads(line) for path in paths for line in path.read_text().splitlines()]

def all_fused(dispatch):
    operations = dispatch.get('operations') or {}
    return bool(dispatch.get('fused_kernels')) and bool(operations) and all(
        op.get('implementation', 'default') != 'default' for op in operations.values())

result = dict(
    complete=all(len(v) == 2 for v in rows.values()),
    finite=all(r['finite'] for v in rows.values() for r in v),
    baseline_off_identical=bool(rows['baseline'] and rows['off']) and {r['hash'] for r in rows['baseline']+rows['off']}.__len__() == 1,
    stock_not_fused=not any(r['dispatch']['fused_kernels'] for r in rows['baseline'] + rows['off']),
    fused_deterministic=bool(rows['on']) and len({r['hash'] for r in rows['on']}) == 1,
    fused_active=bool(rows['on']) and all(all_fused(r['dispatch']) for r in rows['on']))
print(json.dumps(result, indent=2))
sys.exit(0 if all(result.values()) else 1)
