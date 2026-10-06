"""Freeze small protein/RNA and protein/ligand integration inputs; not accuracy benchmarks."""
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
fixtures = root/'ap/test/test_data/features'
protein = json.loads((fixtures/'af3_features/mixed/test_protein_1_af3_input.json').read_text())['sequences'][0]
rows = json.loads((root/'inputs.json').read_text())
for kind in ('rna', 'ligand'):
    name = 'protein_'+kind
    other = json.loads((fixtures/f'{kind}.json').read_text())['sequences'][0]
    data = dict(dialect='alphafold3', version=1, name=name, modelSeeds=[0], sequences=[protein, other])
    path = root/f'{name}_af3_input.json'
    path.write_text(json.dumps(data))
    rows.append(dict(name=name, fold=str(path), suites=['inputs']))
(root/'inputs.json').write_text(json.dumps(rows, indent=2)+'\n')
