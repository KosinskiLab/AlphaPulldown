"""A crashed MMseqs2 must say so, because a crash leaves stderr empty.

Measured on real databases: `mmseqs result2msa` segfaults on one RNA query against
the 37M-entry nt_rna database, while the same command succeeds for other queries on
that same database, and for that query on rfam and rnacentral. Not memory (it died
identically at 200, 275 and 700 GB), not database size, not hit count (the crashing
query returned FEWER result bytes than the one that worked).

Before this, the user saw only "MMseqs2 command failed: result2msa ..." with nothing
after it, which reads as a configuration mistake and invites pointless retries.
"""

from __future__ import annotations

import signal
import subprocess

import pytest

from alphapulldown.feature_batch import SubprocessMmseqsProcess, _describe_exit


def test_a_segfault_is_named_not_left_blank():
    message = _describe_exit(-signal.SIGSEGV)

    assert "SIGSEGV" in message
    assert "crashed" in message


def test_a_shell_wrapped_signal_is_recognised_too():
    """A shell between us and mmseqs reports the same death as 128+N."""
    assert _describe_exit(128 + int(signal.SIGSEGV)) == _describe_exit(-signal.SIGSEGV)


def test_an_ordinary_failure_is_not_dressed_up_as_a_crash():
    message = _describe_exit(1)

    assert message == "MMseqs2 exited with status 1."
    assert "crash" not in message.lower()


def test_a_missing_binary_is_distinguished_from_a_crash():
    assert _describe_exit(None) == "MMseqs2 could not be executed."


def test_the_crash_explanation_reaches_the_raised_error(tmp_path, monkeypatch):
    """The point of the diagnostic is that a user actually sees it."""
    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=-signal.SIGSEGV, cmd=["mmseqs"], stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _boom)
    process = SubprocessMmseqsProcess(tmp_path / "mmseqs")

    with pytest.raises(RuntimeError) as caught:
        process._run(("result2msa", "q", "db", "res", "msa"))

    text = str(caught.value)
    assert "SIGSEGV" in text
    assert "Retrying will not help" in text
