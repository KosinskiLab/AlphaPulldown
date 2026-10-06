import json
from pathlib import Path
import sys
rows, name, output, reps = sys.argv[1:]
row = next(r for r in json.loads(Path(rows).read_text()) if r['name'] == name)
for rep in range(1, int(reps)+1):
    print(json.dumps(dict(job_id=f'{name}_r{rep}', input=row['fold'], output_directory=f'{output}/rep{rep}')))
