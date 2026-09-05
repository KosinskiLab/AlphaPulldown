"""Throwaway: stub just enough AlphaFold 3 to exercise the RNA tests locally.

Not committed. `align_sequence_to_gapless_query` is reimplemented from the behaviour
the existing committed tests pin (query "AC-DE" + hit "ACXDE" -> "ACxDE").
"""

import json
import sys
import types


def _install():
    try:
        import alphafold3.cpp.msa_conversion  # noqa: F401

        return
    except ImportError:
        pass

    af3 = sys.modules.get("alphafold3") or types.ModuleType("alphafold3")
    sys.modules["alphafold3"] = af3

    cpp = types.ModuleType("alphafold3.cpp")
    msa_conversion = types.ModuleType("alphafold3.cpp.msa_conversion")

    def align_sequence_to_gapless_query(*, sequence, query_sequence):
        out = []
        for residue, query_residue in zip(sequence, query_sequence):
            if query_residue in "-.":
                out.append("." if residue in "-." else residue.lower())
            else:
                out.append(residue.upper())
        return "".join(out)

    msa_conversion.align_sequence_to_gapless_query = align_sequence_to_gapless_query
    cpp.msa_conversion = msa_conversion

    common = types.ModuleType("alphafold3.common")
    folding_input = types.ModuleType("alphafold3.common.folding_input")

    class Input:
        def __init__(self, payload):
            self._payload = payload

        @classmethod
        def from_json(cls, text):
            return cls(json.loads(text))

        def to_json(self):
            return json.dumps(self._payload)

    folding_input.Input = Input
    common.folding_input = folding_input

    data = types.ModuleType("alphafold3.data")
    pipeline = types.ModuleType("alphafold3.data.pipeline")

    class DataPipelineConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DataPipeline:
        def __init__(self, config):
            self._config = config

        def process(self, fold_input):
            return fold_input

    pipeline.DataPipelineConfig = DataPipelineConfig
    pipeline.DataPipeline = DataPipeline
    data.pipeline = pipeline

    af3.cpp = cpp
    af3.common = common
    af3.data = data
    sys.modules["alphafold3.cpp"] = cpp
    sys.modules["alphafold3.cpp.msa_conversion"] = msa_conversion
    sys.modules["alphafold3.common"] = common
    sys.modules["alphafold3.common.folding_input"] = folding_input
    sys.modules["alphafold3.data"] = data
    sys.modules["alphafold3.data.pipeline"] = pipeline


_install()
