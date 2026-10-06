import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
rows = {}
for arm in ('baseline', 'off', 'on'):
    paths = list(root.glob(f'{arm}_*/forward.jsonl'))
    rows[arm] = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
result = dict(
    complete=all(len(v) == 2 for v in rows.values()),
    finite=all(r['finite'] for v in rows.values() for r in v),
    baseline_off_identical=bool(rows['baseline'] and rows['off']) and {r['hash'] for r in rows['baseline']+rows['off']}.__len__() == 1,
    fused_deterministic=bool(rows['on']) and len({r['hash'] for r in rows['on']}) == 1,
    fused_active=bool(rows['on']) and all(r['dispatch']['fused_kernels'] for r in rows['on']))
print(json.dumps(result, indent=2))
sys.exit(0 if all(result.values()) else 1)
