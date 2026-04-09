#!/usr/bin/env python3
"""Generate AlphaPulldown diagnostic plots similar to ColabFold outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphapulldown.analysis_pipeline.diagnostics import plot_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write MSA coverage, pLDDT, PAE, and distogram plots from "
            "AlphaPulldown feature pickles or prediction directories."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Feature pickles, directories containing features.pkl, or "
            "prediction directories containing result*.pkl files."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write plots into. Defaults to the parent directory of each input.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Matplotlib DPI to use for saved plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    parser = build_parser()
    args = parser.parse_args(argv)
    written_paths = plot_inputs(
        args.inputs,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
    for path in written_paths:
        print(path)
    return written_paths


if __name__ == "__main__":
    main()
