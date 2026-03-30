import gzip
import importlib.util
import json
import pickle
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import alphapulldown.scripts.generate_crosslink_pickle as crosslink_pickle
import alphapulldown.scripts.rename_colab_search_a3m as rename_a3m
import alphapulldown.scripts.truncate_pickles as truncate_pickles


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SEQ_NAMES_PATH = (
    REPO_ROOT / "alphapulldown" / "scripts" / "prepare_seq_names.py"
)
PARSE_INPUT_PATH = REPO_ROOT / "alphapulldown" / "scripts" / "parse_input.py"
SPLIT_JOBS_PATH = (
    REPO_ROOT / "alphapulldown" / "scripts" / "split_jobs_into_clusters.py"
)


def _load_module_from_path(module_name: str, module_path: Path):
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_parse_input_module(monkeypatch):
    flags_mod = types.ModuleType("absl.flags")
    flags_mod.FLAGS = SimpleNamespace()

    def _define_list(name, default, help_text):
        del help_text
        setattr(flags_mod.FLAGS, name, default)

    def _define_string(name, default, help_text):
        del help_text
        setattr(flags_mod.FLAGS, name, default)

    flags_mod.DEFINE_list = _define_list
    flags_mod.DEFINE_string = _define_string

    app_calls = []
    app_mod = types.ModuleType("absl.app")
    app_mod.run = lambda fn: app_calls.append(fn)

    logging_mod = types.ModuleType("absl.logging")
    logging_mod.INFO = 20
    logging_mod.set_verbosity = lambda *_args, **_kwargs: None

    absl_pkg = types.ModuleType("absl")
    absl_pkg.app = app_mod
    absl_pkg.flags = flags_mod
    absl_pkg.logging = logging_mod

    parser_calls = {}
    parser_mod = types.ModuleType("alphapulldown_input_parser")

    def _generate_fold_specifications(**kwargs):
        parser_calls["generate"] = kwargs
        return ["foldA"]

    parser_mod.generate_fold_specifications = _generate_fold_specifications

    modelling_calls = {}
    modelling_setup_mod = types.ModuleType("alphapulldown.utils.modelling_setup")

    def _parse_fold(specifications, features_directory, delimiter):
        modelling_calls["parse_fold"] = (
            specifications,
            features_directory,
            delimiter,
        )
        return specifications

    def _create_custom_info(parsed):
        modelling_calls["create_custom_info"] = parsed
        return {"parsed": parsed}

    modelling_setup_mod.parse_fold = _parse_fold
    modelling_setup_mod.create_custom_info = _create_custom_info

    monkeypatch.setitem(sys.modules, "absl", absl_pkg)
    monkeypatch.setitem(sys.modules, "absl.app", app_mod)
    monkeypatch.setitem(sys.modules, "absl.flags", flags_mod)
    monkeypatch.setitem(sys.modules, "absl.logging", logging_mod)
    monkeypatch.setitem(sys.modules, "alphapulldown_input_parser", parser_mod)
    monkeypatch.setitem(
        sys.modules,
        "alphapulldown.utils.modelling_setup",
        modelling_setup_mod,
    )

    module = _load_module_from_path("test_parse_input_module", PARSE_INPUT_PATH)
    return module, app_calls, parser_calls, modelling_calls


def _load_split_jobs_module(monkeypatch):
    parser_mod = types.ModuleType("alphapulldown_input_parser")
    parser_mod.generate_fold_specifications = lambda **kwargs: ["A,B", "C;D"]

    modelling_setup_mod = types.ModuleType("alphapulldown.utils.modelling_setup")
    modelling_setup_mod.parse_fold = lambda args: args
    modelling_setup_mod.create_custom_info = lambda parsed_input: parsed_input
    modelling_setup_mod.create_interactors = (
        lambda data, features_directory, index: [[data, features_directory, index]]
    )

    objects_mod = types.ModuleType("alphapulldown.objects")

    class StubMultimericObject:
        def __init__(self, interactors):
            self.interactors = interactors
            self.feature_dict = {"msa": np.zeros((4, 12), dtype=np.int32)}

    objects_mod.MultimericObject = StubMultimericObject

    monkeypatch.setitem(sys.modules, "alphapulldown_input_parser", parser_mod)
    monkeypatch.setitem(
        sys.modules,
        "alphapulldown.utils.modelling_setup",
        modelling_setup_mod,
    )
    monkeypatch.setitem(sys.modules, "alphapulldown.objects", objects_mod)

    return _load_module_from_path("test_split_jobs_module", SPLIT_JOBS_PATH)


def test_prepare_seq_names_rewrites_headers_from_uniprot_style_fasta(
    monkeypatch,
    tmp_path,
    capsys,
):
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(
        ">sp|Q9H9K5|Protein alpha OS=Test\nACDE\n>tr|P12345|Protein beta\nFGHI\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", [str(PREPARE_SEQ_NAMES_PATH), str(fasta_path)])
    runpy.run_path(str(PREPARE_SEQ_NAMES_PATH), run_name="__main__")

    assert capsys.readouterr().out.strip().splitlines() == [
        ">Q9H9K5",
        "ACDE",
        ">P12345",
        "FGHI",
    ]


def test_rename_colab_search_a3m_renames_legacy_search_outputs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "0.a3m").write_text(">protA\nACDE\n>hit\nACDE\n", encoding="utf-8")

    rename_a3m.main()

    renamed = tmp_path / "protA.a3m"
    assert renamed.read_text(encoding="utf-8") == ">protA\nACDE\n>hit\nACDE\n"
    assert not (tmp_path / "0.a3m").exists()


def test_rename_colab_search_a3m_requires_input_fasta_for_new_colabfold(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "0.a3m").write_text(">101\nACDE\n>hit\nACDE\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Please provide the input FASTA file"):
        rename_a3m.main()


def test_rename_colab_search_a3m_uses_input_fasta_names_for_new_colabfold(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "0.a3m").write_text(">101\nACDE\n>hit\nACDE\n", encoding="utf-8")
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">queryA\nACDE\n", encoding="utf-8")

    rename_a3m.main(str(fasta_path))

    renamed = tmp_path / "queryA.a3m"
    assert renamed.read_text(encoding="utf-8") == ">queryA\nACDE\n>hit\nACDE\n"
    assert not (tmp_path / "0.a3m").exists()


def test_generate_crosslink_pickle_parses_single_row_input(tmp_path, monkeypatch):
    links_path = tmp_path / "links.txt"
    links_path.write_text("5 A 9 B 0.05\n", encoding="utf-8")
    output_path = tmp_path / "crosslinks.pkl.gz"
    monkeypatch.setattr(
        crosslink_pickle,
        "parse_arguments",
        lambda: SimpleNamespace(csv=str(links_path), output=str(output_path)),
    )

    crosslink_pickle.main()

    with gzip.open(output_path, "rb") as handle:
        assert pickle.load(handle) == {"A": {"B": [(4, 8, 0.05)]}}


def test_generate_crosslink_pickle_parse_arguments_reads_cli_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--csv", "links.txt", "--output", "crosslinks.pkl.gz"],
    )

    args = crosslink_pickle.parse_arguments()

    assert args.csv == "links.txt"
    assert args.output == "crosslinks.pkl.gz"


def test_parse_input_main_writes_fold_spec_json(monkeypatch, tmp_path):
    module, app_calls, parser_calls, modelling_calls = _load_parse_input_module(
        monkeypatch
    )
    module.FLAGS = SimpleNamespace(
        input_list=["folds.txt"],
        protein_delimiter="+",
        features_directory=["/features"],
        output_prefix=str(tmp_path / "parsed_"),
    )

    module.main([])

    assert app_calls == [module.main]
    assert parser_calls["generate"] == {
        "input_files": ["folds.txt"],
        "delimiter": "+",
        "exclude_permutations": True,
    }
    assert modelling_calls["parse_fold"] == (
        ["foldA"],
        ["/features"],
        "+",
    )
    assert json.loads((tmp_path / "parsed_data.json").read_text(encoding="utf-8")) == {
        "parsed": ["foldA"]
    }


def test_split_jobs_cluster_jobs_writes_cluster_files(monkeypatch, tmp_path):
    module = _load_split_jobs_module(monkeypatch)
    plot_calls = []
    monkeypatch.setattr(
        module,
        "profile_all_jobs_and_cluster",
        lambda all_folds, args: pd.DataFrame(
            {
                "name": ["job_a", "job_b", "job_c"],
                "msa_depth": [20, 40, 60],
                "seq_length": [100, 120, 320],
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "plot_clustering_result",
        lambda X, labels, num_cluster, output_dir: plot_calls.append(
            (X.copy(), np.asarray(labels), num_cluster, output_dir)
        ),
    )

    module.cluster_jobs(["job_a", "job_b", "job_c"], SimpleNamespace(output_dir=str(tmp_path)))

    assert (tmp_path / "job_cluster1_120_40.txt").read_text(encoding="utf-8").splitlines() == [
        "job_a",
        "job_b",
    ]
    assert (tmp_path / "job_cluster2_320_60.txt").read_text(encoding="utf-8").splitlines() == [
        "job_c",
    ]
    assert plot_calls[0][2] == 2
    np.testing.assert_array_equal(
        plot_calls[0][0],
        np.asarray([[100, 20], [120, 40], [320, 60]], dtype=np.int64),
    )
    np.testing.assert_array_equal(plot_calls[0][1], np.asarray([0, 0, 1]))


def test_split_jobs_main_normalises_generated_specs_and_all_vs_all_input(
    monkeypatch,
    tmp_path,
):
    module = _load_split_jobs_module(monkeypatch)
    cluster_calls = []
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            protein_lists=["proteins.txt"],
            protein_delimiter="+",
            mode="all_vs_all",
            features_directory=["/features"],
            output_dir=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        module,
        "generate_fold_specifications",
        lambda **kwargs: ["A,B", "C;D"],
    )
    monkeypatch.setattr(
        module,
        "cluster_jobs",
        lambda all_folds, args: cluster_calls.append((all_folds, args)),
    )

    module.main()

    assert cluster_calls[0][0] == ["A:B", "C+D"]
    assert cluster_calls[0][1].protein_lists == ["proteins.txt"]


def test_truncate_pickles_main_copies_tree_and_removes_selected_pickle_keys(
    monkeypatch,
    tmp_path,
):
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    nested_src = src_dir / "nested"
    nested_dst = dst_dir / "nested"
    nested_src.mkdir(parents=True)
    nested_dst.mkdir(parents=True)
    with open(nested_src / "result.pkl", "wb") as handle:
        pickle.dump(
            {
                "keep": 1,
                "aligned_confidence_probs": [1, 2],
                "distogram": [3, 4],
            },
            handle,
        )
    (nested_src / "notes.txt").write_text("copied\n", encoding="utf-8")
    (nested_dst / "notes.txt").write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(
        truncate_pickles,
        "FLAGS",
        SimpleNamespace(
            src_dir=str(src_dir),
            dst_dir=str(dst_dir),
            keys_to_exclude="aligned_confidence_probs,distogram",
            number_of_threads=2,
        ),
    )

    truncate_pickles.main([])

    with open(nested_dst / "result.pkl", "rb") as handle:
        assert pickle.load(handle) == {"keep": 1}
    assert (nested_dst / "notes.txt").read_text(encoding="utf-8") == "existing\n"


def test_truncate_pickles_main_exits_when_source_dir_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        truncate_pickles,
        "FLAGS",
        SimpleNamespace(
            src_dir=str(tmp_path / "missing"),
            dst_dir=str(tmp_path / "dst"),
            keys_to_exclude="aligned_confidence_probs,distogram",
            number_of_threads=1,
        ),
    )

    with pytest.raises(SystemExit, match="1"):
        truncate_pickles.main([])
