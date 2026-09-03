from pathlib import Path

import pytest

from alphapulldown.scripts.create_batch_features import _feature_requests


def test_cli_adapter_rejects_non_protein_af3_fasta(tmp_path: Path):
    fasta = tmp_path / "dna.fasta"
    fasta.write_text(">DNA example\nACGT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="protein"):
        _feature_requests([str(fasta)])
