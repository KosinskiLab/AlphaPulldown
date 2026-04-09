from pathlib import Path

from alphapulldown.analysis_pipeline.diagnostics import (
    plot_inputs,
    save_msa_coverage_plot,
    save_prediction_plots,
)


TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_save_prediction_plots_writes_core_diagnostics(tmp_path):
    written_paths = save_prediction_plots(
        TEST_DATA / "predictions" / "TEST_homo_2er",
        tmp_path,
    )

    assert [path.name for path in written_paths] == [
        "TEST_homo_2er_pae.png",
        "TEST_homo_2er_plddt.png",
        "TEST_homo_2er_distogram.png",
    ]
    for path in written_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_save_msa_coverage_plot_supports_monomer_pickles(tmp_path):
    output_path = save_msa_coverage_plot(
        TEST_DATA / "features" / "A0A024R1R8.pkl",
        tmp_path,
    )

    assert output_path.name == "A0A024R1R8_msa_coverage.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_inputs_accepts_feature_directories_and_prediction_dirs(tmp_path):
    written_paths = plot_inputs(
        [
            TEST_DATA / "predictions" / "af_vs_ap" / "A0A024R1R8",
            TEST_DATA / "predictions" / "TEST_homo_2er",
        ],
        output_dir=tmp_path,
    )

    assert sorted(path.name for path in written_paths) == [
        "A0A024R1R8_msa_coverage.png",
        "TEST_homo_2er_distogram.png",
        "TEST_homo_2er_pae.png",
        "TEST_homo_2er_plddt.png",
    ]
