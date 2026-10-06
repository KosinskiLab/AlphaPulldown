import json
from pathlib import Path
import sys
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
fused = [r for r in rows if r['arm'] in ('fused', 'fused_tokcore')]
negative = [r for r in rows if r['arm'] == 'fused_biasT']
def valid(r):
    return (r['status'] == 'ok' and r.get('gate') is True and r.get('det_ok') is True
            and r.get('real_finite') is True and (r['mask'] != 'pad' or
            (r.get('pad_finite') is True and r.get('leak_ok') is True)))
errors = [r for r in fused if not valid(r)]
failed_other = [r for r in rows if r['status'] != 'ok']
negative_ok = bool(negative) and all(r.get('gate') is False for r in negative)
result = dict(fused_cells=len(fused), failed=len(errors), failed_other=len(failed_other), negative_control=negative_ok)
print(json.dumps(result, indent=2))
sys.exit(0 if len(fused) == 66 and not errors and not failed_other and negative_ok else 1)
