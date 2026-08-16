"""Validate the final Phase 2 dataset and its leakage-safe gene split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.pipeline.extract_sequences import VARIANT_INDEX, WINDOW_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def run(input_path: Path, report_path: Path) -> dict[str, object]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    variant_ids: set[str] = set()
    gene_splits: dict[str, str] = {}

    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counts["rows"] += 1
            variant_id = row["Variant_ID"]
            if variant_id in variant_ids:
                raise ValueError(f"Duplicate Variant_ID: {variant_id}")
            variant_ids.add(variant_id)

            reference = row["Reference_Sequence"]
            alternate = row["Alternate_Sequence"]
            if len(reference) != WINDOW_SIZE or len(alternate) != WINDOW_SIZE:
                raise ValueError(f"Invalid sequence length: {variant_id}")
            if reference[VARIANT_INDEX] != row["Reference_Allele"]:
                raise ValueError(f"Reference center mismatch: {variant_id}")
            if alternate[VARIANT_INDEX] != row["Alternate_Allele"]:
                raise ValueError(f"Alternate center mismatch: {variant_id}")
            differences = sum(left != right for left, right in zip(reference, alternate))
            if differences != 1:
                raise ValueError(f"Sequence pair differs at {differences} bases: {variant_id}")

            gene = row["Hugo_Symbol"]
            split = row["Split"]
            if gene in gene_splits and gene_splits[gene] != split:
                raise ValueError(f"Gene split leakage: {gene}")
            gene_splits[gene] = split
            counts[f"split_{split}"] += 1
            counts[f"label_{row['Label_Name']}"] += 1
            counts[f"strand_{row['Transcript_Strand_Check']}"] += 1

    report: dict[str, object] = {
        "counts": dict(sorted(counts.items())),
        "unique_variant_ids": len(variant_ids),
        "unique_genes": len(gene_splits),
        "gene_split_leakage_count": 0,
        "sequence_length": WINDOW_SIZE,
        "variant_index": VARIANT_INDEX,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.input, args.report), indent=2))


if __name__ == "__main__":
    main()
