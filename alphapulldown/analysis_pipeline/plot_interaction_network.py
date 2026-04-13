#!/usr/bin/env python3
"""Plot an interaction network from AlphaPulldown score tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from alphapulldown.analysis_pipeline.interaction_network import (
    build_interaction_edge_table,
    plot_interaction_network,
    summarise_nodes,
    write_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an interaction-network plot from an AlphaPulldown score CSV "
            "(for example good_interpae or pi_score outputs)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_csv", help="Path to the score CSV.")
    parser.add_argument("output_plot", help="Path to the output PNG/PDF plot.")
    parser.add_argument(
        "--score_column",
        default="iptm_ptm",
        help="Numeric column used for edge strength filtering and styling.",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.7,
        help=(
            "Minimum score required for an interaction to be plotted. "
            "Use 0.0 to include every aggregated edge."
        ),
    )
    parser.add_argument(
        "--max_pae",
        type=float,
        default=None,
        help="Optional maximum average_interface_pae filter.",
    )
    parser.add_argument(
        "--label_top_n",
        type=int,
        default=60,
        help="Number of top-ranked nodes to label.",
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=300,
        help=(
            "Maximum number of strongest aggregated edges to plot after filtering. "
            "Set to 0 to disable this cap."
        ),
    )
    parser.add_argument(
        "--title",
        default="Interaction Network",
        help="Plot title.",
    )
    parser.add_argument(
        "--edges_out",
        default=None,
        help="Optional CSV path for the aggregated edge table.",
    )
    parser.add_argument(
        "--nodes_out",
        default=None,
        help="Optional CSV path for the node summary table.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Matplotlib DPI for raster outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the force layout.",
    )
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    parser = build_parser()
    args = parser.parse_args(argv)

    score_table = pd.read_csv(args.input_csv)
    edge_table = build_interaction_edge_table(
        score_table,
        score_column=args.score_column,
        min_score=args.min_score,
        max_pae=args.max_pae,
    )
    if args.max_edges > 0 and len(edge_table) > args.max_edges:
        print(
            f"Keeping the top {args.max_edges} of {len(edge_table)} aggregated edges.",
            file=sys.stderr,
        )
        edge_table = edge_table.head(args.max_edges).reset_index(drop=True)

    plot_path = plot_interaction_network(
        edge_table,
        args.output_plot,
        title=args.title,
        label_top_n=args.label_top_n,
        dpi=args.dpi,
        seed=args.seed,
        score_label=args.score_column,
    )

    written_paths = [plot_path]
    if args.edges_out:
        written_paths.append(write_table(edge_table, args.edges_out))
    if args.nodes_out:
        node_table = summarise_nodes(edge_table)
        written_paths.append(write_table(node_table, args.nodes_out))

    for path in written_paths:
        print(path)
    return written_paths


if __name__ == "__main__":
    main()
