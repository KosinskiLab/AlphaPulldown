#!/usr/bin/env python3
"""Supervise an af3_integration campaign: retry infrastructure failures, never scientific ones, then re-collect.

A watcher, not a worker: it only reads SLURM accounting and the campaign's files, resubmits jobs, and writes state under
$CAMPAIGN/supervisor/. Killing it never touches a running job, and a new instance resumes from tasks.json.

Infrastructure failure (retried, up to MAX_ATTEMPTS per task, with the failing node excluded):
  - SLURM states NODE_FAIL, BOOT_FAIL, PREEMPTED, DEADLINE;
  - FAILED with exit code 3 (run.sbatch: `nvidia-smi -L` failed, the node's GPU is unusable);
  - TIMEOUT (once; the retry gets 1.5x the walltime, capped at 12 h);
  - any non-zero end whose logs show the GPU or container could not start (CUDA init, no device, container or I/O errors).
Everything else is final: a gate or identity check that fails, an expected OOM in a capacity probe, a user cancel. A child
cancelled because its parent failed is resubmitted only after the parent's retry, with the dependency rewired to it.

Resubmission reuses the job's own SubmitLine from sacct (same partition, gres, walltime, name, log path, exports, script),
drops the old --dependency and adds the new one and --exclude. When every task is final, it writes sacct.txt for all
attempts, runs the frozen collector (one row per trial, latest attempt wins) and SUPERVISOR_RESULT.txt.
  supervise.py CAMPAIGN [--once] [--dry-run]
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MAX_ATTEMPTS = 3
POLL_S = 300
INFRA_STATES = {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "DEADLINE"}
FINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"} | INFRA_STATES
INFRA_LOG = re.compile(r"cuInit|CUDA_ERROR_(NO_DEVICE|NOT_INITIALIZED|UNKNOWN|SYSTEM_DRIVER_MISMATCH)|Unknown CUDA error|"
                       r"no CUDA-capable device|Failed to initialize CUDA|NVIDIA-SMI has failed|No devices were found|"
                       r"container creation failed|FATAL: +(?:while|container)|Transport endpoint is not connected|"
                       r"Stale file handle|Input/output error|Unable to load CUDA|jax_plugins.xla_cuda12.initialize|"
                       r"/bin/python[0-9.]*: No such file or directory|apptainer: command not found")


def sh(args, check=True):
    p = subprocess.run(args, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({p.returncode}): {p.stderr.strip()[:400]}")
    return p.stdout


def sacct(ids):
    out = {}
    for chunk in range(0, len(ids), 200):
        text = sh(["sacct", "-X", "-P", "-n", "-j", ",".join(ids[chunk:chunk + 200]),
                   "--format=JobID,State,ExitCode,NodeList,Timelimit,Reason,SubmitLine%4000"], check=False)
        for line in text.splitlines():
            f = line.split("|", 6)
            if len(f) == 7:
                out[f[0]] = dict(state=f[1].split()[0], exit=f[2], node=f[3], timelimit=f[4], reason=f[5], submit=f[6])
    return out


def parse_submit(line):
    """sbatch argv from a SubmitLine: single-token options, one --export=... argument (values may contain spaces), the script."""
    head, script = line.rsplit(" ", 1)
    i = head.index("--export=")
    return head[:i].split(), head[i:], script


class Supervisor:
    def __init__(self, campaign, dry_run=False):
        self.c = Path(campaign)
        self.dir = self.c / "supervisor"
        self.dir.mkdir(exist_ok=True)
        self.state_file = self.dir / "tasks.json"
        self.dry = dry_run
        if self.state_file.exists():
            self.tasks = json.loads(self.state_file.read_text())
        else:                                                           # one task per jobs.tsv line; the parent from its dependency
            self.tasks = {}
            lines = [ln.split("\t") for ln in (self.c / "jobs.tsv").read_text().splitlines() if ln.strip()]
            info = sacct([ln[0] for ln in lines])
            for jid, gpu, suite, arm in (ln[:4] for ln in lines):
                dep = re.search(r"--dependency=afterok:(\d+)", info.get(jid, {}).get("submit", ""))
                self.tasks[f"{gpu}/{suite}/{arm}"] = dict(jobs=[jid], parent_job=dep.group(1) if dep else None, exclude=[], status="active",
                                                         history=[])
            for t in self.tasks.values():                               # parent job id -> parent task key
                t["parent"] = next((k for k, u in self.tasks.items() if t["parent_job"] in u["jobs"]), None)
            self.save()

    def save(self):
        if self.dry:
            return
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.tasks, indent=1))
        os.replace(tmp, self.state_file)

    def log(self, msg):
        line = time.strftime("%Y-%m-%dT%H:%M:%S ") + msg
        print(line, flush=True)
        if not self.dry:
            with open(self.dir / "supervisor.log", "a") as f:
                f.write(line + "\n")

    def infra_in_logs(self, key, jid):
        gpu, suite, arm = key.split("/")
        texts = []
        for p in list((self.c / "logs").glob(f"*_{jid}.log")) + list((self.c / "runs" / gpu / suite / arm / jid).rglob("*.log")):
            try:
                with open(p, errors="replace") as f:
                    f.seek(max(0, os.path.getsize(p) - 200_000))
                    texts.append(f.read())
            except OSError:
                pass
        m = INFRA_LOG.search("\n".join(texts))
        return m.group(0) if m else None

    def classify(self, key, jid, a):
        state, code = a["state"], a["exit"].split(":")[0]
        if state == "COMPLETED":
            return "done", None
        if state in INFRA_STATES:
            return "infra", state
        if state == "FAILED" and code == "3":
            return "infra", "gpu_unusable(exit 3)"
        if state == "FAILED" and code == "127":
            return "infra", "command_not_found(exit 127)"                # e.g. /home (the host python's target) not mounted
        if state == "TIMEOUT":
            return ("infra", "timeout") if sum(h.get("why") == "timeout" for h in self.tasks[key]["history"]) == 0 else ("final", "timeout")
        if state == "CANCELLED" and self.tasks[key]["parent"] and (a["node"] in ("None assigned", "") or "Dependency" in a["reason"]):
            return "dep_cancelled", a["reason"]                         # never started: --kill-on-invalid-dep after its parent failed
        if state in ("FAILED", "OUT_OF_MEMORY"):
            why = self.infra_in_logs(key, jid)
            return ("infra", f"log:{why}") if why else ("final", f"{state} {a['exit']}")
        return "final", f"{state} {a['exit']} {a['reason']}"

    def resubmit(self, key, a, why, parent_jid=None):
        t = self.tasks[key]
        opts, export, script = parse_submit(a["submit"])
        opts = [o for o in opts if not o.startswith(("--dependency", "--exclude", "--parsable"))]
        if why == "timeout" and "-t" in opts:
            d, _, hms = a["timelimit"].rpartition("-")
            h, m, sec = (int(x) for x in hms.split(":"))
            new = min(int(((int(d or 0) * 24 + h) * 3600 + m * 60 + sec) * 1.5), 12 * 3600)
            opts[opts.index("-t") + 1] = f"{new // 3600:02d}:{new % 3600 // 60:02d}:00"
        if a.get("node") and a["node"] not in ("None assigned", "") and why != "timeout":
            t["exclude"] = sorted(set(t["exclude"]) | {a["node"]})
        argv = [opts[0], "--parsable"] + opts[1:] + ([f"--exclude={','.join(t['exclude'])}"] if t["exclude"] else []) \
            + ([f"--dependency=afterok:{parent_jid}"] if parent_jid else []) + [export, script]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_", "SALLOC_", "SRUN_"))}
        env.update(CAMPAIGN=str(self.c), SNAPSHOT=str(self.c / "source"))   # what submit.sh exported; --export=ALL carries it
        if self.dry:
            self.log(f"DRY resubmit {key} ({why}): {' '.join(argv)[:300]}")
            return None
        jid = subprocess.run(argv, capture_output=True, text=True, env=env, check=True).stdout.strip().split(";")[0]
        t["jobs"].append(jid)
        t["history"].append(dict(why=why, old=a.get("jobid"), new=jid, node=a.get("node")))
        t["status"] = "active"
        with open(self.c / "jobs.tsv", "a") as f:                       # the original collector's sacct and this record see it
            f.write("\t".join([jid] + key.split("/")) + "\tretry\n")
        self.log(f"resubmitted {key}: {a.get('jobid')} -> {jid} ({why}; exclude={t['exclude']}; parent={parent_jid})")
        self.save()
        return jid

    def step(self):
        current = {k: t["jobs"][-1] for k, t in self.tasks.items()}
        acct = sacct(sorted(set(current.values())))
        pending = 0
        for key in sorted(self.tasks, key=lambda k: (self.tasks[k]["parent"] is not None, k)):   # parents first
            t, jid = self.tasks[key], current[key]
            a = acct.get(jid)
            if t["status"] in ("done", "final") or a is None:
                pending += a is None and t["status"] == "active"
                continue
            a["jobid"] = jid
            if a["state"] not in FINAL_STATES:
                pending += 1
                continue
            kind, why = self.classify(key, jid, a)
            if kind == "done":
                t["status"] = "done"
            elif kind == "infra" and len(t["jobs"]) < MAX_ATTEMPTS:
                parent = self.tasks.get(t["parent"]) if t["parent"] else None
                self.resubmit(key, a, why, parent["jobs"][-1] if parent and parent["status"] != "done" else None)
                pending += 1
            elif kind == "dep_cancelled":
                parent = self.tasks.get(t["parent"])
                if parent and parent["status"] == "active" and len(parent["jobs"]) > 1:   # the parent is being retried: follow it
                    if parent["jobs"][-1] != t.get("parent_job"):
                        self.resubmit(key, a, "parent_retried", parent["jobs"][-1])
                        t["parent_job"] = parent["jobs"][-1]
                    pending += 1
                elif parent and parent["status"] == "done" and len(t["jobs"]) < MAX_ATTEMPTS:  # cancelled before the retry finished
                    self.resubmit(key, a, "parent_retried", None)
                    pending += 1
                elif parent and parent["status"] == "active":
                    pending += 1
                else:
                    t["status"] = "final"
                    t["history"].append(dict(why=f"final: parent {t['parent']} did not succeed"))
            else:
                t["status"] = "final"
                t["history"].append(dict(why=f"final: {why}", job=jid))
                self.log(f"final {key} job {jid}: {why}")
        self.save()
        hb = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), job=os.environ.get("SLURM_JOB_ID"), pending=pending,
                  done=sum(t["status"] == "done" for t in self.tasks.values()), final=sum(t["status"] == "final" for t in self.tasks.values()),
                  retries=sum(len(t["jobs"]) - 1 for t in self.tasks.values()))
        if not self.dry:
            (self.dir / "HEARTBEAT.json").write_text(json.dumps(hb) + "\n")
        return pending, hb

    def finish(self, hb):
        ids = [j for t in self.tasks.values() for j in t["jobs"]]
        if self.dry:
            self.log(f"DRY finish: would collect {len(ids)} jobs")
            return
        (self.c / "sacct.txt").write_text(sh(["sacct", "-X", "-P", "-j", ",".join(ids),
                                              "--format=JobID,JobName%40,State,ExitCode,NodeList,Elapsed"], check=False))
        bench = os.environ.get("BENCH", "/scratch/dima/kernel_bench_phase0")
        py = os.environ.get("HOST_PY", "/scratch/dima/af2_mmseqs_bench/dockq_venv/bin/python")
        dockq = os.environ.get("DOCKQ", "/scratch/dima/af2_mmseqs_bench/dockq_venv/bin/DockQ")
        collector = self.dir / "harness" / "af3_integration" / "collect.py"
        if not os.path.exists(py):                                      # this node lacks the interpreter: leave it to a successor
            self.log(f"finish deferred: {py} not found on {os.uname().nodename}")
            sys.exit(3)
        p = subprocess.run([py, str(collector), str(self.c), bench, dockq], capture_output=True, text=True)
        (self.dir / "collect.log").write_text(p.stdout + p.stderr)
        lines = [f"supervisor_finished={time.strftime('%Y-%m-%dT%H:%M:%S')}", f"collector_exit={p.returncode}",
                 f"campaign={self.c}", f"report={self.c}/REPORT.md", f"tasks_done={hb['done']}", f"tasks_final_not_ok={hb['final']}",
                 f"retries={hb['retries']}"]
        lines += [f"  {k}: {t['status']} jobs={','.join(t['jobs'])}" + (f" last={t['history'][-1].get('why')}" if t["history"] else "")
                  for k, t in sorted(self.tasks.items())]
        (self.c / "SUPERVISOR_RESULT.txt").write_text("\n".join(lines) + "\n")
        self.log(f"finished: collector exit {p.returncode}, {hb}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    once, dry = "--once" in sys.argv, "--dry-run" in sys.argv
    s = Supervisor(args[0], dry_run=dry)
    if (s.c / "SUPERVISOR_RESULT.txt").exists() and not dry:
        print("campaign already finished; nothing to do")
        return 0
    deadline = time.time() + float(os.environ.get("SUPERVISOR_SECONDS", 2.9 * 86400))
    while True:
        pending, hb = s.step()
        s.log(f"step: {hb}")
        if pending == 0:
            s.finish(hb)
            return 0
        if once or time.time() + POLL_S > deadline:
            return 0                                                    # a queued successor (singleton) resumes from tasks.json
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
