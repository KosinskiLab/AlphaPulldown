#!/usr/bin/env python
"""Build every benchmark input from AlphaPulldown's own feature pipeline.

Runs inside the AlphaPulldown AF2 image. For each fold it builds the MultimericObject
exactly as `run_structure_prediction` would (parse_fold -> create_interactors ->
MultimericObject), captures the per-chain MSAs at the moment AlphaPulldown merges them
(after species pairing, unpaired de-duplication and cropping), and writes them as a
ColabFold complex a3m. ColabFold re-merges those rows with the same AlphaFold code, so
both tools fold the same MSA; verify_inputs.py proves it per fold against AlphaPulldown's
final feature dict, which is saved here as <name>.ref.npz.

Outputs in --out: a3m/<name>.a3m, ref/<name>.ref.npz, folds.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from alphafold.data import msa_pairing
from alphapulldown.objects import MultimericObject
from alphapulldown.utils.af2_to_af3_msa import AF2_ID_TO_A3M, aligned_row_and_deletions_to_a3m
from alphapulldown.utils.modelling_setup import create_custom_info, create_interactors, parse_fold

REF_KEYS = ("msa", "deletion_matrix", "aatype", "residue_index", "asym_id", "entity_id", "sym_id")

_captured = []
_merge = msa_pairing.merge_chain_features


def _capture_merge(np_chains_list, pair_msa_sequences, max_templates):
    _captured.append(([dict(c) for c in np_chains_list], pair_msa_sequences))
    return _merge(np_chains_list=np_chains_list, pair_msa_sequences=pair_msa_sequences,
                  max_templates=max_templates)


msa_pairing.merge_chain_features = _capture_merge


def read_folds(folds_tsv: Path, accuracy_folds: Path):
    folds = {}
    for line in folds_tsv.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, suites, spec = line.split("\t")
        folds[name] = {"name": name, "suites": suites.split(","), "fold": spec}
    for spec in accuracy_folds.read_text().split():
        name = "acc_" + spec.split("_")[0]
        folds[name] = {"name": name, "suites": ["accuracy"], "fold": spec}
    return folds


def row_to_a3m(row, deletions) -> str:
    return aligned_row_and_deletions_to_a3m(np.asarray(row), np.asarray(deletions).astype(np.int64))


def build(spec: str, features_dir: str):
    parsed = parse_fold([spec], [features_dir], "+")
    interactors = create_interactors(create_custom_info(parsed), [features_dir])[0]
    _captured.clear()
    obj = MultimericObject(interactors=interactors, pair_msa=True)
    if len(_captured) != 1:
        raise RuntimeError(f"{spec}: expected one merge, saw {len(_captured)}")
    chains, paired = _captured[0]
    return obj, [i.sequence for i in interactors], chains, paired


def to_colabfold_a3m(sequences, chains, paired) -> tuple[str, dict]:
    if not paired:
        raise RuntimeError("AlphaPulldown did not pair this fold; the a3m export covers heteromers only")
    lengths = [len(s) for s in sequences]
    gaps = ["-" * n for n in lengths]
    n_paired = {int(c["msa_all_seq"].shape[0]) for c in chains}
    if len(n_paired) != 1:
        raise RuntimeError(f"paired MSAs differ in depth across chains: {n_paired}")
    n_paired = n_paired.pop()
    for k, (seq, chain) in enumerate(zip(sequences, chains)):
        if "".join(AF2_ID_TO_A3M[int(t)] for t in chain["msa_all_seq"][0]) != seq:
            raise RuntimeError(f"paired row 0 of chain {k} is not the query")

    # Row 0 of the paired block (all queries) is the header entry; ColabFold reads it as the
    # first paired row. Every row gets a unique header: ColabFold drops repeated (header, row).
    lines = ["#" + ",".join(map(str, lengths)) + "\t" + ",".join("1" for _ in lengths),
             ">" + "\t".join(str(101 + k) for k in range(len(sequences))),
             "".join(sequences)]
    for r in range(1, n_paired):
        row = "".join(row_to_a3m(c["msa_all_seq"][r], c["deletion_matrix_all_seq"][r]) for c in chains)
        lines += [">" + "\t".join(f"p{r}" for _ in chains), row]
    # Unpaired rows exactly as AlphaPulldown merges them. deduplicate_unpaired_sequences has
    # already removed the query (it is paired row 0), so unlike ColabFold's own a3m files these
    # blocks do not open with a padded query; adding one would add rows AlphaPulldown lacks.
    for k, chain in enumerate(chains):
        for r in range(chain["msa"].shape[0]):
            body = row_to_a3m(chain["msa"][r], chain["deletion_matrix"][r])
            lines += [f">u{k}_{r}", "".join(body if j == k else gaps[j] for j in range(len(chains)))]
    stats = {"paired_rows": n_paired, "unpaired_rows": [int(c["msa"].shape[0]) for c in chains]}
    return "\n".join(lines) + "\n", stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds-tsv", type=Path, required=True)
    ap.add_argument("--accuracy-folds", type=Path, required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (args.out / "a3m").mkdir(parents=True, exist_ok=True)
    (args.out / "ref").mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    for name, fold in read_folds(args.folds_tsv, args.accuracy_folds).items():
        try:
            obj, sequences, chains, paired = build(fold["fold"], args.features)
            a3m, stats = to_colabfold_a3m(sequences, chains, paired)
        except Exception as exc:  # one bad fold must not hide the others
            failures.append(name)
            print(f"FAILED {name} {fold['fold']}: {exc!r}", file=sys.stderr)
            continue
        (args.out / "a3m" / f"{name}.a3m").write_text(a3m)
        np.savez_compressed(args.out / "ref" / f"{name}.ref.npz",
                            **{k: np.asarray(obj.feature_dict[k]) for k in REF_KEYS})
        record = {**fold, "tokens": sum(map(len, sequences)), "chain_lengths": list(map(len, sequences)),
                  "msa_rows": int(obj.feature_dict["msa"].shape[0]), **stats}
        records.append(record)
        print(json.dumps(record))
    (args.out / "folds.json").write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} folds, {len(failures)} failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
