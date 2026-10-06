"""Gate the integrated layer sweep (gate0_layers.py --integrated): every fused record must pass Gate 0's accuracy gate,
padding/leak, finiteness and repeat checks, and must have run the fork's fused kernel rather than its stock fallback."""
import json
from pathlib import Path
import sys
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
fused = [r for r in rows if r['arm'] in ('fused', 'fused_tokcore')]
negative = [r for r in rows if r['arm'] == 'fused_biasT']
def dispatched(r):
    """A layer the fork does not dispatch runs the stock body, which passes the accuracy gate trivially."""
    want = 'pallas_tokamax_core' if r['arm'] == 'fused_tokcore' else 'pallas'
    return r.get('integrated') is True and r.get('fork_implementation') == want
def valid(r):
    return (r['status'] == 'ok' and r.get('gate') is True and r.get('det_ok') is True
            and r.get('real_finite') is True and dispatched(r) and (r['mask'] != 'pad' or
            (r.get('pad_finite') is True and r.get('leak_ok') is True)))
errors = [r for r in fused if not valid(r)]
failed_other = [r for r in rows if r['status'] != 'ok']
negative_ok = bool(negative) and all(r.get('gate') is False for r in negative)
result = dict(fused_cells=len(fused), failed=len(errors), not_dispatched=sum(not dispatched(r) for r in fused),
              failed_other=len(failed_other), negative_control=negative_ok)
print(json.dumps(result, indent=2))
sys.exit(0 if len(fused) == 66 and not errors and not failed_other and negative_ok else 1)
