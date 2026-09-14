"""The MSA stage must not drag AlphaFold or JAX in with it.

`create_batch_msas.py` runs as a GPU rule in AlphaPulldownSnakemake. Importing
JAX would make it preallocate GPU memory on a node whose GPU is meant for the
MMseqs2 search -- which is why its CPU sibling has to set `JAX_PLATFORMS=cpu`.
The saving is a GPU node's memory, not milliseconds, and nothing else asserts it,
so a stray top-level import would regress it silently.

Checked in a subprocess: whatever the test session already imported would
otherwise mask the very thing being measured.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Heavy in different ways: jax and torch claim GPU memory, alphafold and
# tensorflow are slow and pull the rest of the stack behind them.
FORBIDDEN = ("jax", "jaxlib", "alphafold", "torch", "tensorflow")

LIGHT_MODULES = (
    # The script the Snakemake GPU rule actually runs.
    "alphapulldown.scripts.create_batch_msas",
    "alphapulldown.feature_batch",
    "alphapulldown.scripts._mmseqs2_cli",
    "alphapulldown.inference_flags",
)

PROBE = """
import importlib, sys
importlib.import_module({module!r})
leaked = sorted(
    name for name in sys.modules
    if any(name == f or name.startswith(f + ".") for f in {forbidden!r})
)
print(",".join(leaked))
"""


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_the_light_path_pulls_in_neither_alphafold_nor_jax(module):
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module, forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    leaked = [name for name in completed.stdout.strip().split(",") if name]
    assert not leaked, (
        f"{module} now imports {leaked}. This stage shares a GPU node with the "
        "MMseqs2 search; importing JAX there preallocates GPU memory. Import it "
        "inside the function that needs it instead."
    )


# Importing the module was never the hard part. Converting a search result used
# `alphafold3.cpp.msa_conversion`, so the stage imported cleanly in the AlphaFold 2
# image -- which installs AlphaPulldown without the `alphafold3` extra -- and then
# died on the first result. Exercise the conversion itself, not just the import.
RESULT_PROBE = """
import importlib, sys
feature_batch = importlib.import_module("alphapulldown.feature_batch")
from alphapulldown.utils.msa_formats import stitch_headers_and_insertions
headers = [
    ("query_0", "MKTAYIAKQRQ"),
    ("sp|P00001|EXACT_HUMAN exact OS=Homo sapiens OX=9606", "MKTAYIAKQRQ"),
    ("tr|P00003|DELETE_YEAST deletion OS=Saccharomyces OX=4932", "MKT---AKQRQ"),
]
insertions = [
    ("query_0", "MKTAYIAKQRQ"),
    ("P00001", "MKTAYIAKQRQ"),
    ("P00003", "MKT---AKwwQRQ"),
]
a3m = feature_batch._stitched_to_a3m(
    stitch_headers_and_insertions(headers, insertions), "MKTAYIAKQRQ"
)
assert "sp|P00001|EXACT_HUMAN" in a3m, "full headers must survive conversion"
assert "MKT---AKwwQRQ" in a3m, "deletions and insertions must survive conversion"
leaked = sorted(
    name for name in sys.modules
    if any(name == f or name.startswith(f + ".") for f in {forbidden!r})
)
print(",".join(leaked))
"""


def test_converting_a_search_result_pulls_in_neither_alphafold_nor_jax():
    completed = subprocess.run(
        [sys.executable, "-c", RESULT_PROBE.format(forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    leaked = [name for name in completed.stdout.strip().split(",") if name]
    assert not leaked, (
        f"converting a search result now imports {leaked}. The MSA stage has to "
        "run in the AlphaFold 2 image, which has no AlphaFold 3 at all."
    )
