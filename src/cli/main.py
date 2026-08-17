"""Command-line entry point for genomic-variant-prioritizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cli.score import FineTunedScorer, ScoringError, run_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="variant-prioritizer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="Rank SNVs from a VCF")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--reference-dir", type=Path, required=True)
    score.add_argument("--cosmic", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        scorer = FineTunedScorer(args.checkpoint, args.batch_size, args.device)
        report = run_score(args.input, args.output, args.reference_dir, args.cosmic, scorer)
    except ScoringError as error:
        raise SystemExit(f"Error: {error}") from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
