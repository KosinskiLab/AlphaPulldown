#!/usr/bin/env python3
"""Generate AlphaFold Server batch JSON files from AlphaPulldown job inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphapulldown.utils.alphafold_server_json import (
    build_alphafold_server_jobs,
    write_jobs_to_json_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AlphaPulldown protein list jobs into AlphaFold Server batch JSON files."
        )
    )
    parser.add_argument(
        "--protein_lists",
        required=True,
        nargs="+",
        help="One or more AlphaPulldown protein list files.",
    )
    parser.add_argument(
        "--monomer_objects_dir",
        required=True,
        nargs="+",
        help="One or more directories containing AlphaPulldown monomer feature pickles.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output JSON path. Files are automatically split if more than --jobs_per_file jobs are generated.",
    )
    parser.add_argument(
        "--mode",
        default="pulldown",
        choices=["pulldown", "all_vs_all", "homo-oligomer", "custom"],
        help="Job generation mode, matching run_multimer_jobs.py.",
    )
    parser.add_argument(
        "--oligomer_state_file",
        default=None,
        help="Path to the oligomer-state file used for mode=homo-oligomer.",
    )
    parser.add_argument(
        "--protein_delimiter",
        default="+",
        help="Protein delimiter used by alphapulldown-input-parser.",
    )
    parser.add_argument(
        "--model_seeds",
        default="",
        help="Comma-separated list of AlphaFold Server seeds. Leave empty to request automatic seed assignment.",
    )
    parser.add_argument(
        "--job_index",
        type=int,
        default=None,
        help="1-based job index to export. Export all jobs when omitted.",
    )
    parser.add_argument(
        "--jobs_per_file",
        type=int,
        default=100,
        help="Maximum number of jobs per output JSON file. AlphaFold Server currently accepts up to 100 jobs per file.",
    )
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_seeds = [seed.strip() for seed in args.model_seeds.split(",") if seed.strip()]
    jobs = build_alphafold_server_jobs(
        protein_lists=args.protein_lists,
        monomer_directories=args.monomer_objects_dir,
        mode=args.mode,
        oligomer_state_file=args.oligomer_state_file,
        protein_delimiter=args.protein_delimiter,
        model_seeds=model_seeds,
        job_index=args.job_index,
    )
    written_paths = write_jobs_to_json_files(
        jobs,
        args.output_path,
        jobs_per_file=args.jobs_per_file,
    )
    for path in written_paths:
        print(path)
    return written_paths


if __name__ == "__main__":
    main()
