import numpy as np

from alphapulldown.utils.plotting import plot_pae_from_matrix


def test_plot_pae_from_matrix_writes_png_file(tmp_path):
    output = tmp_path / "pae.png"

    plot_pae_from_matrix(
        seqs=["AA", "BBB"],
        pae_matrix=np.arange(25, dtype=float).reshape(5, 5),
        figure_name=str(output),
        ranking=2,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
