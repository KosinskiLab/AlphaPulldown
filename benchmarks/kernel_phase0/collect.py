#!/usr/bin/env python3
"""Turn $BENCH/runs into tables, REPORT.md and RESULT.txt. Safe to re-run at any time.

Reads every run directory written by run_af2_arm.sbatch / run_af3_baseline.sbatch:
  runs/<suite>/<gpu>/<arm>/{runs.tsv,gpu.txt,gpu_mem.csv,logs/<tag>.log,out/<tag>/}
Standard library only; DockQ (accuracy suite) is called as an external program.
"""

import argparse
import datetime as dt
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

CF_QUERY = re.compile(r"Query \d+/\d+: (\S+) \(length (\d+)\)")
CF_TOOK = re.compile(r"took ([0-9.]+)s \((\d+) recycles\)")
AP_PREDICT = re.compile(r"\[bench\] predict seconds=([0-9.]+) num_recycles=(\S+)")
AF3_INFER = re.compile(r"Model inference for seed \d+ took ([0-9.]+) seconds")
AF3_START = re.compile(r"Running model inference for seed \d+")
# JAX_LOG_COMPILES lines for the model's forward function; AlphaPulldown re-jits it on every call.
AF3_JIT = re.compile(r"Finished (?:tracing \+ transforming apply_fn for pjit|jaxpr to MLIR module conversion "
                     r"jit\(apply_fn\)|XLA compilation of jit\(apply_fn\)) in ([0-9.]+) sec")
KIT_ACTIVE = re.compile(r"\[colabfold-opt\] (NOT ACTIVE|ACTIVE)\b(.*)")
KIT_LEVER = re.compile(r"LEVER name=(\S+) state=(\S+)(.*)")
DOCKQ_SCORE = re.compile(r"DockQ:?\s+([0-9.]+)")

COMPARISONS = [  # (label, baseline arm, arm): ratio = baseline time / arm time, >1 means faster
    ("ColabFold 1.6.3 fast kernels vs its stock", "cf163_stock", "cf163_fast"),
    ("Anthropic kit exact vs its stock (1.6.1)", "kit_off", "kit_exact"),
    ("Anthropic kit fast vs its stock (1.6.1)", "kit_off", "kit_fast"),
    ("Anthropic kit fast vs ColabFold 1.6.3 fast", "cf163_fast", "kit_fast"),
    ("ColabFold 1.6.3 stock vs AlphaPulldown stock", "ap_stock", "cf163_stock"),
    ("ColabFold 1.6.1 stock vs AlphaPulldown stock", "ap_stock", "kit_off"),
    # What a port would gain: each kernel arm against the code AlphaPulldown runs today.
    ("ColabFold 1.6.3 fast kernels vs AlphaPulldown stock", "ap_stock", "cf163_fast"),
    ("Anthropic kit fast vs AlphaPulldown stock", "ap_stock", "kit_fast"),
]
SAME_TOOL_STOCK = {"cf163_fast": "cf163_stock", "kit_exact": "kit_off", "kit_fast": "kit_off"}
ARM_ORDER = ["ap_stock", "cf163_stock", "cf163_fast", "kit_off", "kit_exact", "kit_fast", "ap_af3", "ap_af3_cache"]
# Every CUDA GPU type the cluster offers for compute (V100 and MI210 sit only in build-el10), in
# the order of the summary table: label as in bench.env, name, compute capability.
CLUSTER_GPUS = [("3090", "RTX 3090", "sm_86"), ("a40", "A40", "sm_86"), ("l40s", "L40S", "sm_89"),
                ("a100", "A100", "sm_80"), ("h100pcie", "H100 PCIe", "sm_90"), ("h100", "H100 SXM", "sm_90"),
                ("h200", "H200", "sm_90"),
                ("b200", "B200", "sm_100"), ("rtx6000", "RTX Pro 6000 Blackwell", "sm_120"),
                ("b4500", "RTX Pro 4500 Blackwell (MIG half, 16 GB)", "sm_120")]
NOT_TESTABLE = {"b200": "reserved"}  # bgx1, B200's only node: reservation vLLMs for another user until 2026-12-31
AF3_CACHE_LABEL = "AF3 persistent compilation cache vs AlphaPulldown default (per prediction call)"
SUMMARY_ROWS = [  # (row label, comparison label in speedups.tsv)
    ("AF2: ColabFold 1.6.3 fused kernels vs AlphaPulldown stock", "ColabFold 1.6.3 fast kernels vs AlphaPulldown stock"),
    ("AF2: Anthropic kit fast vs AlphaPulldown stock", "Anthropic kit fast vs AlphaPulldown stock"),
    ("AF2: Anthropic kit fast vs ColabFold 1.6.3 fused kernels", "Anthropic kit fast vs ColabFold 1.6.3 fast"),
    ("AF3: persistent compilation cache vs AlphaPulldown default", AF3_CACHE_LABEL),
]


def read_tsv(path: Path):
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:] if line.strip()]


def gpu_memory(run: Path):
    samples = []
    path = run / "gpu_mem.csv"
    if not path.exists():
        return samples
    for line in path.read_text().splitlines():
        try:
            stamp, used = [x.strip() for x in line.split(",")]
            when = dt.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp()
            samples.append((when, int(used)))
        except ValueError:
            continue
    return samples


def peak_mib(samples, start, end):
    if not samples or start in ("NA", None):
        return None
    start, end = float(start), float(end)
    window = [m for t, m in samples if start <= t <= end]
    before = [m for t, m in samples if t < start]
    idle = min(before) if before else min(m for _, m in samples)
    return max(window) - idle if window else None


def parse_colabfold(log: str, out: Path):
    """Per rep: forward seconds, recycles, and confidences from the scores json."""
    reps, current = {}, None
    for line in log.splitlines():
        if m := CF_QUERY.search(line):
            current = m.group(1)
        elif (m := CF_TOOK.search(line)) and current:
            reps[current] = {"seconds": float(m.group(1)), "recycles": int(m.group(2))}
    for job, rec in reps.items():
        scores = sorted(out.glob(f"{job}_scores_rank_001_*.json"))
        if scores:
            s = json.loads(scores[0].read_text())
            rec["ptm"], rec["iptm"] = s.get("ptm"), s.get("iptm")
            rec["plddt"] = statistics.fmean(s["plddt"]) if s.get("plddt") else None
            if rec["ptm"] is not None and rec["iptm"] is not None:
                rec["ranking_confidence"] = 0.8 * rec["iptm"] + 0.2 * rec["ptm"]
        models = sorted(out.glob(f"{job}_unrelaxed_rank_001_*.pdb"))
        rec["model"] = str(models[0]) if models else None
    return [reps[k] for k in sorted(reps, key=lambda j: int(j.rsplit("_r", 1)[1]))]


def mean_plddt_from_pdb(path: Path):
    values = [float(line[60:66]) for line in path.read_text().splitlines()
              if line.startswith("ATOM") and line[12:16].strip() == "CA"]
    return statistics.fmean(values) if values else None


def parse_alphapulldown(log: str, out: Path, fold: str):
    reps = []
    for m in AP_PREDICT.finditer(log):
        recycles = m.group(2)
        reps.append({"seconds": float(m.group(1)), "recycles": int(recycles) if recycles.isdigit() else None})
    for i, rec in enumerate(reps, start=1):
        job = out / f"{fold}_r{i}"
        ranking = next(job.rglob("ranking_debug.json"), None)
        if ranking:
            r = json.loads(ranking.read_text())
            rec["ranking_confidence"] = next(iter(r.get("iptm+ptm", {}).values()), None)
            rec["iptm"] = next(iter(r.get("iptm", {}).values()), None) if isinstance(r.get("iptm"), dict) else None
        model = next(job.rglob("unrelaxed_model_1_multimer_v3*.pdb"), None)
        rec["model"] = str(model) if model else None
        rec["plddt"] = mean_plddt_from_pdb(model) if model else None
    return reps


def parse_af3(log: str):
    """Per inference call: total seconds, the jit(apply_fn) trace + lowering + compile logged inside
    it, and the forward time left after removing that. jit_logged is False for runs made without
    JAX_LOG_COMPILES, whose seconds include an unknown compile and are not a forward time."""
    reps = []
    for chunk in AF3_START.split(log)[1:]:
        m = AF3_INFER.search(chunk)
        if not m:
            continue
        call = float(m.group(1))
        jit = [float(x) for x in AF3_JIT.findall(chunk[:m.start()])]
        reps.append({"call_s": call, "jit_s": sum(jit), "jit_logged": bool(jit),
                     "seconds": call - sum(jit) if jit else None})
    return reps


def af3_reps_identical(out: Path, fold: str):
    """True when rep 2 (served from the persistent cache) ranks its samples exactly as rep 1 did."""
    scores = [next((out / f"{fold}_r{i}").rglob("*ranking_scores.csv"), None) for i in (1, 2)]
    if not all(scores):
        return None
    return scores[0].read_text() == scores[1].read_text()


def parse_kit(log: str):
    active = [f"{m.group(1)}{m.group(2)}".strip() for m in KIT_ACTIVE.finditer(log)]
    levers = {}
    for m in KIT_LEVER.finditer(log):
        fields = dict(kv.split("=", 1) for kv in m.group(3).split() if "=" in kv)
        lever = {"state": m.group(2), **{k: fields[k] for k in ("calls", "served", "rows", "fallbacks", "reason") if k in fields}}
        # A kernel lever can be "on" yet serve every call with the stock XLA op (no kernel table
        # for this card), either as fallbacks or as rows/served entries naming xla only.
        served = ",".join(lever.get(k, "") for k in ("served", "rows")).strip(",")
        impls = {part.split(":")[-2] for part in served.split(",") if part.count(":") >= 1 and part.split(":")[-1].isdigit()}
        calls, fallbacks = fields.get("calls", ""), fields.get("fallbacks", "0")
        lever["stock_only"] = (impls == {"xla"}) or (calls.isdigit() and int(calls) > 0 and fallbacks == calls) \
            or (calls == "0" and fallbacks.isdigit() and int(fallbacks) > 0)
        levers[m.group(1)] = lever
    return {"active": active[-1] if active else None, "levers": levers}


def collect_runs(bench: Path, folds: dict):
    rows, kits = [], []
    for run in sorted(bench.glob("runs/*/*/*")):
        if not (run / "runs.tsv").exists():
            continue
        suite, gpu, arm = run.parts[-3:]
        arm = arm.split("#")[0]  # chunked arms run as <arm>#<tag>, one directory per chunk
        samples = gpu_memory(run)
        gpu_name = (run / "gpu.txt").read_text().split(",")[0].strip() if (run / "gpu.txt").exists() else gpu
        if not gpu_name.startswith("NVIDIA"):  # MIG runs before the label fix wrote "No devices were found"
            gpu_name = next((name for label, name, _ in CLUSTER_GPUS if label == gpu), gpu)
        for r in read_tsv(run / "runs.tsv"):
            fold, tag = r["fold"], r["tag"]
            log_path = run / "logs" / f"{tag}.log"
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
            out = run / "out" / tag
            if arm.startswith(("cf163", "kit")):
                reps = parse_colabfold(log, out)
            elif arm.startswith("ap_af3"):
                reps = parse_af3(log)
            else:
                reps = parse_alphapulldown(log, out, fold)
            times = [x["seconds"] for x in reps if x["seconds"] is not None]
            row = {"suite": suite, "gpu": gpu, "gpu_name": gpu_name, "arm": arm, "fold": fold,
                   "tokens": folds.get(fold, {}).get("tokens"), "seed": r["seed"], "status": r["status"],
                   "reps": len(times), "first_s": times[0] if times else None,
                   "fwd_s": statistics.median(times[1:]) if len(times) > 1 else None,
                   "recycles": reps[-1].get("recycles") if reps else None,
                   "peak_mib": peak_mib(samples, r["start"], r["end"]),
                   "wall_s": float(r["end"]) - float(r["start"]) if r["start"] != "NA" else None}
            if arm.startswith("ap_af3"):  # compile is measured per call, not as first rep minus the rest
                row["call_s"] = statistics.median([x["call_s"] for x in reps[1:]]) if len(reps) > 1 else None
                row["first_s"] = reps[0]["call_s"] if reps else None
                # Timed reps only: for the cache arm rep 1 compiles and rep 2 loads from the cache.
                jit = [x["jit_s"] for x in reps[1:] if x["jit_logged"]]
                row["compile_s"] = statistics.median(jit) if jit else None
                if row["status"] == "ok" and reps and not any(x["jit_logged"] for x in reps):
                    row["status"] = "ok_no_jit_log"
                if arm == "ap_af3_cache":
                    row["outputs_identical"] = af3_reps_identical(out, fold)
            elif row["fwd_s"] is not None:
                row["compile_s"] = row["first_s"] - row["fwd_s"]
                if row["recycles"] is not None:
                    row["per_pass_s"] = row["fwd_s"] / (row["recycles"] + 1)
            if reps:
                for key in ("ranking_confidence", "iptm", "ptm", "plddt", "model"):
                    row[key] = reps[0].get(key)
            if row["status"] == "ok" and not times and not reps:
                row["status"] = "no_timings"
            if arm.startswith("kit"):
                k = parse_kit(log)
                row["kit_active"] = k["active"]
                kits.append({"suite": suite, "gpu": gpu, "arm": arm, "fold": fold, **k})
            rows.append(row)
    return rows, kits


def score_dockq(rows, natives: Path, dockq: str, cache_path: Path):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for row in rows:
        if row["suite"] != "accuracy" or not row.get("model"):
            continue
        model = row["model"]
        if model not in cache:
            native = natives / f"{row['fold'].removeprefix('acc_')}.cif"
            run = subprocess.run([dockq, model, str(native), "--short"], capture_output=True, text=True, timeout=900)
            found = [float(x) for x in DOCKQ_SCORE.findall(run.stdout)]
            cache[model] = max(found) if found else None
        row["dockq"] = cache[model]
    cache_path.write_text(json.dumps(cache, indent=1))


def geomean(values):
    values = [v for v in values if v and v > 0]
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def fmt(v, nd=2):
    if v is None:
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_tsv(path: Path, rows, columns):
    with path.open("w") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in columns) + "\n")


def speed_section(rows):
    speed = [r for r in rows if r["suite"] == "speed" and not r["arm"].startswith("ap_af3")]
    by = {(r["gpu"], r["arm"], r["fold"]): r for r in speed}
    gpus = sorted({r["gpu"] for r in speed})
    lines = ["## AF2-Multimer speed (forward call, compile excluded)", "",
             f"Ratio = baseline time / arm time, so >1 means faster. Geometric mean over the folds both arms completed.", ""]
    lines.append("| comparison | " + " | ".join(gpus) + " |")
    lines.append("|---|" + "---|" * len(gpus))
    ratios_out = []
    mixed_models = 0
    for label, base, arm in COMPARISONS:
        cells = []
        for gpu in gpus:
            ratios = []
            for (g, a, fold), r in by.items():
                b = by.get((g, base, fold))
                if g == gpu and a == arm and b and r.get("fwd_s") and b.get("fwd_s") and r["status"] == b["status"] == "ok":
                    if r["gpu_name"] != b["gpu_name"]:  # e.g. H100 SXM vs PCIe from different partitions
                        mixed_models += 1
                        continue
                    ratios.append(b["fwd_s"] / r["fwd_s"])
                    ratios_out.append({"gpu": g, "comparison": label, "baseline": base, "arm": arm, "fold": fold,
                                       "tokens": r["tokens"], "ratio": b["fwd_s"] / r["fwd_s"]})
            gm = geomean(ratios)
            cells.append(f"{gm:.2f}× (n={len(ratios)})" if gm else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if mixed_models:
        lines += ["", f"{mixed_models} fold pair(s) left out: the two arms ran on different GPU models."]
    lines += ["", "### Largest fold completed (tokens) and its peak GPU memory (MiB above idle)", ""]
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in speed)]
    lines.append("| gpu | " + " | ".join(arms) + " |")
    lines.append("|---|" + "---|" * len(arms))
    for gpu in gpus:
        cells = []
        for arm in arms:
            ok = [r for r in speed if r["gpu"] == gpu and r["arm"] == arm and r["status"] == "ok"]
            best = max(ok, key=lambda r: r["tokens"] or 0, default=None)
            cells.append(f"{best['tokens']} ({fmt(best['peak_mib'], 0)})" if best else "–")
        lines.append(f"| {gpu} | " + " | ".join(cells) + " |")
    lines += ["", "### Forward seconds per fold", ""]
    for gpu in gpus:
        lines += [f"**{gpu}** ({next(r['gpu_name'] for r in speed if r['gpu'] == gpu)})", "",
                  "| fold | tokens | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
        for fold in sorted({r["fold"] for r in speed if r["gpu"] == gpu}, key=lambda f: next(r["tokens"] or 0 for r in speed if r["fold"] == f)):
            cells = []
            for arm in arms:
                r = by.get((gpu, arm, fold))
                cells.append("–" if not r else (fmt(r.get("fwd_s")) if r["status"] == "ok" else r["status"]))
            lines.append(f"| {fold} | {next(r['tokens'] for r in speed if r['fold'] == fold)} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines, ratios_out


def af3_ratios(rows):
    """Per fold: default call time / cached call time, both second reps on the same GPU model."""
    by = {(r["gpu"], r["arm"], r["fold"]): r for r in rows if r["suite"] == "speed"}
    out = []
    for (gpu, arm, fold), c in by.items():
        d = by.get((gpu, "ap_af3", fold))
        if arm == "ap_af3_cache" and d and c["status"] == d["status"] == "ok" and c.get("call_s") and d.get("call_s") \
                and c["gpu_name"] == d["gpu_name"]:
            out.append({"gpu": gpu, "comparison": AF3_CACHE_LABEL, "baseline": "ap_af3", "arm": arm, "fold": fold,
                        "tokens": c["tokens"], "ratio": d["call_s"] / c["call_s"]})
    return out


def af3_section(rows):
    af3 = [r for r in rows if r["arm"].startswith("ap_af3") and r["suite"] == "speed"]
    if not af3:
        return []
    by = {(r["gpu"], r["arm"], r["fold"]): r for r in af3}
    lines = ["## AF3 (AlphaPulldown, DeepMind weights, second rep of each fold)", "",
             "AlphaPulldown's AF3 backend re-traces and recompiles the model on every predict call, even for an "
             "identical input in the same process, so by default every call pays the compile. `fwd s` is the "
             "default call minus the jit(apply_fn) trace, lowering and compile logged inside it "
             "(JAX_LOG_COMPILES). `call s` is what one prediction costs: by default, and with a persistent "
             "compilation cache (`--jax_compilation_cache_dir` plus `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`; "
             "rep 1 compiles and stores, rep 2 is served from the cache, as every prediction after the first in "
             "each token bucket would be). `same output` checks that the cached rep ranks its 5 samples exactly "
             "as the compiling rep did. Status `ok_no_jit_log` marks runs made before compiles were logged.", "",
             "| gpu | fold | tokens | status | fwd s | compile s | call s | cached: call s | cached: compile s | "
             "speed-up | same output | peak MiB |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["gpu"], r["fold"]) for r in af3}, key=lambda k: (k[0], next(r["tokens"] or 0 for r in af3 if r["fold"] == k[1])))
    for gpu, fold in keys:
        d, c = by.get((gpu, "ap_af3", fold), {}), by.get((gpu, "ap_af3_cache", fold), {})
        tokens = (d or c).get("tokens")
        status = d.get("status", "–") if not c or c.get("status") == d.get("status") else f"{d.get('status', '–')} / cached {c.get('status')}"
        speedup = d["call_s"] / c["call_s"] if d.get("call_s") and c.get("call_s") and d["status"] == c["status"] == "ok" else None
        same = {True: "yes", False: "NO", None: "–"}[c.get("outputs_identical")] if c else "–"
        lines.append(f"| {gpu} | {fold} | {tokens} | {status} | {fmt(d.get('fwd_s'))} | {fmt(d.get('compile_s'))} | "
                     f"{fmt(d.get('call_s'))} | {fmt(c.get('call_s'))} | {fmt(c.get('compile_s'))} | "
                     f"{fmt(speedup) + '×' if speedup else '–'} | {same} | {fmt((d or c).get('peak_mib'), 0)} |")
    return lines + [""]


def summary_section(ratios, rows):
    """One table over every cluster GPU type: geometric mean of the per-fold ratios, with their range."""
    done = {r["gpu"] for r in rows if r["suite"] == "speed" and r["status"] == "ok"}
    names = {r["gpu"]: r["gpu_name"] for r in rows if r["suite"] == "speed" and r.get("gpu_name")}
    lines = ["## Speed-up on every GPU type in the cluster", "",
             "Geometric mean over the folds both arms completed (164 to 2,546 tokens for AF2, 164 tokens up to "
             "the card's limit for AF3), with the per-fold range in brackets; >1× means faster. AF2 rows compare "
             "forward time with the model compiled; AF3 compares the whole prediction call. A kernel's gain "
             "grows with fold size, and the AF3 cache's gain shrinks with it (the compile it saves is a fixed "
             "~50 s): see the per-fold tables below. No AF3 kernel arm exists: Anthropic's AF3 kit runs only "
             "sokrypton's fork with OpenFold3 weights.", ""]
    head = [f"{name} ({cc})" for label, name, cc in CLUSTER_GPUS]
    lines.append("| | " + " | ".join(head) + " |")
    lines.append("|---|" + "---|" * len(CLUSTER_GPUS))
    for row_label, comparison in SUMMARY_ROWS:
        cells = []
        for label, _, _ in CLUSTER_GPUS:
            vals = [x["ratio"] for x in ratios if x["gpu"] == label and x["comparison"] == comparison]
            if vals:
                cells.append(f"{geomean(vals):.2f}× ({min(vals):.1f}–{max(vals):.1f})")
            else:
                cells.append(NOT_TESTABLE.get(label) or ("not run" if label not in done else "–"))
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    lines += ["", "Cards measured: " + (", ".join(f"{g} = {names[g]}" for g in sorted(done) if g in names) or "none") + ". "
              "`not run`: no result yet (queued or not submitted). `reserved`: B200's only node (bgx1) is reserved "
              "for another user until 2026-12-31.", ""]
    return lines


def accuracy_section(rows):
    acc = [r for r in rows if r["suite"] == "accuracy"]
    if not acc:
        return []
    by = {(r["gpu"], r["arm"], r["seed"], r["fold"]): r for r in acc}
    lines = ["## Accuracy sanity check (12 heterodimers, model_1, one seed per row)", "",
             "Stock seed-to-seed spread is the yardstick: a kernel that changes results by much more than a seed does is broken. "
             "Each card is compared only with itself; a kit kernel is only checked on a card where it engages "
             "(see Kit activation).", "",
             "| gpu | arm | seed | folds | mean DockQ | DockQ ≥ 0.23 | mean ranking conf. | mean \\|Δ ranking conf.\\| vs stock seed 0 | mean \\|Δ DockQ\\| vs stock seed 0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = lambda x: (x[0], ARM_ORDER.index(x[1]) if x[1] in ARM_ORDER else 99, x[2])
    for gpu, arm, seed in sorted({(r["gpu"], r["arm"], r["seed"]) for r in acc}, key=order):
        mine = [r for r in acc if r["gpu"] == gpu and r["arm"] == arm and r["seed"] == seed and r["status"] == "ok"]
        stock = SAME_TOOL_STOCK.get(arm, arm)
        ref_seed = "0" if (stock != arm or seed != "0") else None
        d_conf, d_dockq = [], []
        if ref_seed is not None:
            for r in mine:
                b = by.get((gpu, stock, ref_seed, r["fold"]))
                if b and r.get("ranking_confidence") is not None and b.get("ranking_confidence") is not None:
                    d_conf.append(abs(r["ranking_confidence"] - b["ranking_confidence"]))
                if b and r.get("dockq") is not None and b.get("dockq") is not None:
                    d_dockq.append(abs(r["dockq"] - b["dockq"]))
        dq = [r["dockq"] for r in mine if r.get("dockq") is not None]
        rc = [r["ranking_confidence"] for r in mine if r.get("ranking_confidence") is not None]
        lines.append(f"| {gpu} | {arm} | {seed} | {len(mine)} | {fmt(statistics.fmean(dq) if dq else None, 3)} | "
                     f"{sum(d >= 0.23 for d in dq)}/{len(dq)} | {fmt(statistics.fmean(rc) if rc else None, 3)} | "
                     f"{fmt(statistics.fmean(d_conf) if d_conf else None, 3)} | {fmt(statistics.fmean(d_dockq) if d_dockq else None, 3)} |")
    return lines + [""]


def kit_section(kits):
    if not kits:
        return []
    lines = ["## Kit activation (did the optimizations actually run?)", "",
             "A lever can be on yet serve every call with the stock XLA op when the card has no kernel table; "
             "those are listed as stock-only. Folds that ran out of memory before any model call (the kit "
             "then reports partial activation) are left out; `folds` counts the ones that ran. "
             "Per-lever counters: `report/kit_levers.json`.", "",
             "| gpu | arm | folds | ACTIVE (mode) | levers on / reported | stock-only on every fold | some fallbacks |",
             "|---|---|---|---|---|---|---|"]
    grouped = defaultdict(list)
    for k in kits:
        grouped[(k["gpu"], k["arm"], k["suite"])].append(k)
    for (gpu, arm, suite), all_ks in sorted(grouped.items()):
        ks = [k for k in all_ks if k["active"] and "no model call went through" not in k["active"]] or all_ks
        last = ks[-1]
        on = sum(v["state"] == "on" for v in last["levers"].values())
        names = {n for k in ks for n in k["levers"]}
        stock_only = sorted(n for n in names if all(k["levers"].get(n, {}).get("stock_only") for k in ks))
        fell = sorted({n for k in ks for n, v in k["levers"].items()
                       if v.get("fallbacks", "0") not in ("0", "") and n not in stock_only})
        active = (last["active"] or "–").split(" levers=")[0][:40]
        folds = str(len(ks)) if len(ks) == len(all_ks) else f"{len(ks)} of {len(all_ks)}"
        lines.append(f"| {gpu} | {arm} ({suite}) | {folds} | {active} | {on}/{len(last['levers'])} | "
                     f"{', '.join(stock_only) or '–'} | {', '.join(fell) or '–'} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--dockq", required=True)
    ap.add_argument("--expected", type=Path, help="jobs.tsv from submit.sh, to report missing runs")
    args = ap.parse_args()
    bench = args.bench
    folds_json = bench / "inputs" / "folds.json"
    folds = {f["name"]: f for f in json.loads(folds_json.read_text())} if folds_json.exists() else {}
    if not folds:
        print(f"no inputs: {folds_json} is missing; see {bench}/inputs/RESULT.txt")
    rows, kits = collect_runs(bench, folds)
    report = bench / "report"
    report.mkdir(exist_ok=True)
    score_dockq(rows, bench / "inputs" / "natives", args.dockq, report / "dockq_cache.json")

    columns = ["suite", "gpu", "gpu_name", "arm", "fold", "tokens", "seed", "status", "reps", "first_s", "fwd_s",
               "compile_s", "call_s", "recycles", "per_pass_s", "peak_mib", "wall_s", "ranking_confidence", "iptm", "ptm", "plddt",
               "dockq", "kit_active", "outputs_identical", "model"]
    write_tsv(report / "runs.tsv", rows, columns)
    speed_lines, ratios = speed_section(rows)
    ratios += af3_ratios(rows)
    write_tsv(report / "speedups.tsv", ratios, ["gpu", "comparison", "baseline", "arm", "fold", "tokens", "ratio"])
    (report / "kit_levers.json").write_text(json.dumps(kits, indent=1))

    missing = []
    if args.expected and args.expected.exists():
        for job in read_tsv(args.expected):
            result = bench / "runs" / job["suite"] / job["gpu"] / job["arm"] / "RESULT.txt"
            if not result.exists():
                missing.append(job)
            elif result.read_text().startswith("node_unusable"):
                missing.append({**job, "job": f"{job['job']}, {result.read_text().split(':')[0]}"})
    statuses = defaultdict(int)
    for r in rows:
        statuses[r["status"].split(":")[0]] += 1

    md = ["# Phase 0 kernel benchmark", "",
          f"Generated {dt.datetime.now().isoformat(timespec='seconds')} from `{bench}/runs`. "
          f"Per-rep rows: `report/runs.tsv`; per-fold ratios: `report/speedups.tsv`; kit lever counters: `report/kit_levers.json`.", "",
          "Every AF2 arm folds the same MSA (verified per fold, `inputs/verify_*.tsv`), with model_1_multimer_v3, "
          "a fixed number of recycles (early stopping off), no templates, no relaxation, no unified memory.", "",
          f"Fold outcomes: {dict(statuses)}. Runs not finished: {len(missing)}.", ""]
    md += summary_section(ratios, rows) + speed_lines + af3_section(rows) + accuracy_section(rows) + kit_section(kits)
    if missing:
        md += ["## Runs without a RESULT.txt", ""] + [f"- {m['suite']}/{m['gpu']}/{m['arm']} (job {m['job']})" for m in missing] + [""]
    (bench / "REPORT.md").write_text("\n".join(md))

    (bench / "RESULT.txt").write_text(
        f"collected={dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"fold_outcomes={dict(statuses)}\nruns_missing={len(missing)}\n"
        f"report={bench / 'REPORT.md'}\ntables={report}\nlogs={bench / 'runs'}/<suite>/<gpu>/<arm>/logs\n")
    print((bench / "RESULT.txt").read_text())


if __name__ == "__main__":
    main()
