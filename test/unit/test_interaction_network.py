from pathlib import Path

import pandas as pd

from alphapulldown.analysis_pipeline.interaction_network import (
    build_interaction_edge_table,
    plot_interaction_network,
    summarise_nodes,
    write_table,
)


def test_build_interaction_edge_table_prefers_interface_pairs():
    table = pd.DataFrame(
        [
            {
                "jobs": "A_and_B_and_C",
                "iptm_ptm": 0.91,
                "average_interface_pae": 4.0,
                "interface": "A_C",
            },
            {
                "jobs": "A_and_B",
                "iptm_ptm": 0.72,
                "average_interface_pae": 2.0,
                "interface": "A_B",
            },
            {
                "jobs": "B_and_B",
                "iptm_ptm": 0.88,
                "average_interface_pae": 3.0,
                "interface": "A_B",
            },
        ]
    )

    edge_table = build_interaction_edge_table(table, min_score=0.7, max_pae=5.0)

    assert edge_table.to_dict(orient="records") == [
        {
            "source": "A",
            "target": "C",
            "score": 0.91,
            "support": 1,
            "jobs": "A_and_B_and_C",
            "self_interaction": False,
        },
        {
            "source": "A",
            "target": "B",
            "score": 0.72,
            "support": 1,
            "jobs": "A_and_B",
            "self_interaction": False,
        },
        {
            "source": "B",
            "target": "B",
            "score": 0.88,
            "support": 1,
            "jobs": "B_and_B",
            "self_interaction": True,
        },
    ]


def test_plot_interaction_network_and_tables_write_outputs(tmp_path):
    edge_table = pd.DataFrame(
        [
            {
                "source": "A",
                "target": "B",
                "score": 0.82,
                "support": 2,
                "jobs": "A_and_B",
                "self_interaction": False,
            },
            {
                "source": "B",
                "target": "B",
                "score": 0.77,
                "support": 1,
                "jobs": "B_and_B",
                "self_interaction": True,
            },
        ]
    )

    plot_path = plot_interaction_network(edge_table, tmp_path / "network.png", seed=7)
    node_table = summarise_nodes(edge_table)
    node_path = write_table(node_table, tmp_path / "nodes.csv")
    edge_path = write_table(edge_table, tmp_path / "edges.csv")

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert node_path.read_text(encoding="utf-8").startswith("node,degree,best_score")
    assert edge_path.read_text(encoding="utf-8").startswith("source,target,score")
