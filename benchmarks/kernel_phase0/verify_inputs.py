#!/usr/bin/env python
"""Prove that ColabFold folds the same input AlphaPulldown does.

Runs inside a ColabFold image (1.6.3, and the kit's 1.6.1). For every fold it parses the
exported a3m with ColabFold's own unserialize_msa + generate_input_feature, the calls
colabfold_batch makes, and compares the resulting features with AlphaPulldown's final
feature dict (ref/<name>.ref.npz). Any difference fails the fold; exit 1 if any failed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from colabfold.batch import generate_input_feature, unserialize_msa

MAX_SEQ = 508  # colabfold_batch's max_seq for alphafold2_multimer_v3; pads the MSA to 512 as AlphaPulldown does


def compare(name: str, inputs: Path) -> list[str]:
    a3m = (inputs / "a3m" / f"{name}.a3m").read_text()
    unpaired, paired, seqs, cardinality, templates = unserialize_msa([a3m], None)
    features, _ = generate_input_feature(seqs, cardinality, unpaired, paired, templates,
                                         True, "alphafold2_multimer_v3", MAX_SEQ)
    ref = np.load(inputs / "ref" / f"{name}.ref.npz")
    problems = []
    for key in ref.files:
        got, want = np.asarray(features.get(key)), ref[key]
        if got.shape != want.shape:
            problems.append(f"{key}: shape {got.shape} != {want.shape}")
        elif not np.array_equal(got, want):
            problems.append(f"{key}: {int(np.sum(got != want))} of {want.size} values differ")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    from importlib.metadata import version as dist_version
    version = f"colabfold {dist_version('colabfold')} / alphafold-colabfold {dist_version('alphafold-colabfold')}"
    failed = 0
    for record in json.loads((args.inputs / "folds.json").read_text()):
        problems = compare(record["name"], args.inputs)
        failed += bool(problems)
        print(f"{args.label}\t{version}\t{record['name']}\t{'OK' if not problems else 'DIFF'}\t{'; '.join(problems)}")
    print(f"{args.label}: {failed} fold(s) differ")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
