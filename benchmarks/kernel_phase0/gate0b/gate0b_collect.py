#!/usr/bin/env python3
"""Collect Gate 0b: $BENCH/gate0b/runs[_TAG]/<gpu>/<arm>/ -> GATE0B.md.

Per fold: rep 1 (compiles) and rep 2 (warm) inference seconds from AlphaPulldown's log, the compile logged inside rep 2 (must be 0),
the best ranking_score of each rep, and the fused arm's served/fallback census. Speed-up = stock warm s / fused warm s (> 1 = faster).
  gate0b_collect.py [TAG]
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from collect import parse_af3  # noqa: E402  the Phase 0 AF3 log parser (same log lines on AlphaPulldown main)

BENCH = os.environ.get("BENCH", "/scratch/dima/kernel_bench_phase0")


def best_ranking(out_dir):
    f = next(iter(glob.glob(os.path.join(out_dir, "**", "*ranking_scores.csv"), recursive=True)), None)
    if not f:
        return None
    return max(float(r["ranking_score"]) for r in csv.DictReader(open(f)))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    root = os.path.join(BENCH, "gate0b", "runs" + (f"_{tag}" if tag else ""))
    folds = {x["name"]: x["tokens"] for x in json.load(open(os.path.join(BENCH, "inputs", "folds.json")))}
    rows = {}
    for log in sorted(glob.glob(os.path.join(root, "*", "*", "logs", "*.log"))):
        parts = log.split(os.sep)
        gpu, arm, fold = parts[-4], parts[-3], os.path.basename(log)[:-4]
        text = open(log, errors="replace").read()
        reps = parse_af3(text)
        served = re.findall(r"G0B_SERVED (\{.*\})", text)
        cfg = re.findall(r"G0B_CONFIG (.*)", text)
        run = os.path.dirname(os.path.dirname(log))
        status = "ok"
        for line in open(os.path.join(run, "runs.tsv")).read().splitlines()[1:]:
            f = line.split("\t")
            if f[0] == fold:
                status = f[-1]
        rows[(gpu, fold, arm)] = dict(
            status=status, n_reps=len(reps), first_s=reps[0]["call_s"] if reps else None,
            warm_s=reps[1]["call_s"] if len(reps) > 1 else None, warm_jit_s=reps[1]["jit_s"] if len(reps) > 1 else None,
            rank1=best_ranking(os.path.join(run, "out", fold, f"{fold}_r1")), rank2=best_ranking(os.path.join(run, "out", fold, f"{fold}_r2")),
            served=json.loads(served[-1]) if served else None, cfg=cfg[0] if cfg else None)
    gpus = sorted({k[0] for k in rows})
    md = ["# Gate 0b: end-to-end AF3 through AlphaPulldown main, stock vs the kit's fused triangle kernels", "",
          "Warm = rep 2 of one process (rep 1 compiles); `warm compile` must be 0. Speed-up = stock warm / fused warm (> 1 = faster).",
          "Ranking = best ranking_score of the 5 samples, seed 0, rep 1.", "",
          "| card | fold | tokens | stock warm s | fused warm s | speed-up | warm compile s (stock/fused) | first call s (stock/fused) | "
          "ranking stock / fused | status |", "|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    f1 = lambda x: "–" if x is None else f"{x:.1f}"   # noqa: E731
    census = []
    for gpu in gpus:
        for fold in sorted({k[1] for k in rows if k[0] == gpu}, key=lambda f: folds.get(f, 0)):
            s, f = rows.get((gpu, fold, "stock"), {}), rows.get((gpu, fold, "fused"), {})
            sp = s["warm_s"] / f["warm_s"] if s.get("warm_s") and f.get("warm_s") else None
            md.append(f"| {gpu} | {fold} | {folds.get(fold)} | {f1(s.get('warm_s'))} | {f1(f.get('warm_s'))} | {f2(sp)} | "
                      f"{f1(s.get('warm_jit_s'))} / {f1(f.get('warm_jit_s'))} | {f1(s.get('first_s'))} / {f1(f.get('first_s'))} | "
                      f"{f2(s.get('rank1'))} / {f2(f.get('rank1'))} | {s.get('status', '–')} / {f.get('status', '–')} |")
            if f.get("served"):
                census.append(f"- {gpu} {fold}: {f.get('cfg')} — " + ", ".join(f"{k}: {v}" for k, v in f["served"].items()))
    md += ["", "## Fused arm: where the kernels engaged (trace-time calls)", ""] + census
    out = os.path.join(BENCH, "gate0b", "GATE0B" + (f"_{tag}" if tag else "") + ".md")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
