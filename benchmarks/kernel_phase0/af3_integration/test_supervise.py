"""Unit tests for the campaign supervisor's parsing, failure classes and the collector's one-row-per-trial rule."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util  # noqa: E402
import supervise  # noqa: E402

_spec = importlib.util.spec_from_file_location("af3_collect", Path(__file__).resolve().parent / "collect.py")
_collect = importlib.util.module_from_spec(_spec)   # its own name: the harness root has a collect.py too
_spec.loader.exec_module(_collect)
latest_attempts = _collect.latest_attempts

LINE = ("sbatch --parsable --qos=high --kill-on-invalid-dep=yes -p gpu-el10 --gres=gpu:nvidia_a40:1 -t 05:00:00 "
        "--dependency=afterok:11 -J af3i-speed-a40-off -o /c/logs/%x_%j.log "
        "--export=ALL,GPU=a40,SUITE=speed,ARM=off,FOLDS=s0164 s0357,SEEDS=0 1,REPS=2,TILE_CC=8.6 /c/source/run.sbatch")


def make(tmp_path):
    c = tmp_path / "campaign"
    (c / "logs").mkdir(parents=True)
    (c / "supervisor").mkdir()
    tasks = {"a40/layers/hooks": dict(jobs=["10"], parent=None, parent_job=None, exclude=[], status="done", history=[]),
             "a40/identity/paired": dict(jobs=["11"], parent="a40/layers/hooks", parent_job="10", exclude=[], status="active", history=[]),
             "a40/speed/off": dict(jobs=["12"], parent="a40/identity/paired", parent_job="11", exclude=[], status="active", history=[])}
    (c / "supervisor" / "tasks.json").write_text(json.dumps(tasks))
    return supervise.Supervisor(c, dry_run=True)


def acct(**kw):
    a = dict(state="FAILED", exit="1:0", node="gpu40", timelimit="05:00:00", reason="None", submit=LINE, jobid="12")
    a.update(kw)
    return a


def test_parse_submit_keeps_multiword_exports():
    opts, export, script = supervise.parse_submit(LINE)
    assert export == "--export=ALL,GPU=a40,SUITE=speed,ARM=off,FOLDS=s0164 s0357,SEEDS=0 1,REPS=2,TILE_CC=8.6"
    assert script == "/c/source/run.sbatch" and opts[0] == "sbatch" and "--dependency=afterok:11" in opts


def test_failure_classes(tmp_path):
    s = make(tmp_path)
    assert s.classify("a40/speed/off", "12", acct(state="NODE_FAIL")) == ("infra", "NODE_FAIL")
    assert s.classify("a40/speed/off", "12", acct(exit="3:0"))[0] == "infra"
    assert s.classify("a40/speed/off", "12", acct(exit="1:0")) == ("final", "FAILED 1:0")      # a gate or capacity failure stays final
    assert s.classify("a40/speed/off", "12", acct(state="TIMEOUT", exit="0:0")) == ("infra", "timeout")
    s.tasks["a40/speed/off"]["history"].append(dict(why="timeout"))
    assert s.classify("a40/speed/off", "12", acct(state="TIMEOUT", exit="0:0"))[0] == "final"  # only one timeout retry
    assert s.classify("a40/speed/off", "12", acct(state="CANCELLED", exit="0:0", node="None assigned"))[0] == "dep_cancelled"
    assert s.classify("a40/speed/off", "12", acct(exit="127:0"))[0] == "infra"                  # /home missing on the node
    (s.c / "logs" / "af3i-speed-a40-off_12.log").write_text("RuntimeError: operation cuInit(0) failed")
    assert s.classify("a40/speed/off", "12", acct(exit="1:0"))[0] == "infra"


def test_latest_attempt_wins():
    base = dict(gpu="a40", suite="speed", requested_arm="off", arm="off", fold="s0164", seed=0)
    rows = [dict(base, job="100", status="ok"), dict(base, job="200", status="failed_or_incomplete"), dict(base, job="300", status="ok"),
            dict(base, fold="s0357", job="100", status="failed_or_incomplete")]
    kept, superseded = latest_attempts(rows)
    assert [(r["fold"], r["job"]) for r in kept] == [("s0164", "300"), ("s0357", "100")]
    assert len(superseded) == 2
