"""Make an explicitly populated template fixture from the frozen native structure."""
import json
from pathlib import Path
import sys
from alphafold3 import structure

root, native = map(Path, sys.argv[1:])
model = structure.from_mmcif(native.read_text())
sequences = model.chain_single_letter_sequence()
chain, sequence = next((k, v) for k, v in sequences.items() if len(v) > 40 and set(v) <= set('ACDEFGHIKLMNPQRSTVWYX'))
template = model.filter(chain_id=chain)
indices = list(range(len(sequence)))
data = dict(dialect='alphafold3', version=1, name='populated_template', modelSeeds=[0],
            sequences=[{'protein':dict(id='A', sequence=sequence, unpairedMsa='>query\n'+sequence+'\n', pairedMsa='',
                       templates=[dict(mmcif=template.to_mmcif(), queryIndices=indices, templateIndices=indices)])}])
path = root/'populated_template_af3_input.json'
path.write_text(json.dumps(data))
rows = json.loads((root/'inputs.json').read_text())
rows.append(dict(name='populated_template', fold=str(path), suites=['inputs']))
(root/'inputs.json').write_text(json.dumps(rows, indent=2)+'\n')
