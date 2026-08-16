"""Apply the COSMIC Cancer Gene Census membership proxy label."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cgc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def load_cgc(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Gene Symbol" not in rows[0]:
        raise ValueError("CGC export is empty or lacks the Gene Symbol column")
    return {row["Gene Symbol"].strip(): row for row in rows}


def run(
    input_path: Path, cgc_path: Path, output_path: Path, report_path: Path
) -> Counter[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cgc = load_cgc(cgc_path)
    counts: Counter[str] = Counter()
    driver_genes: set[str] = set()
    passenger_genes: set[str] = set()

    with input_path.open(newline="") as source, output_path.open(
        "w", newline=""
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or "Hugo_Symbol" not in reader.fieldnames:
            raise ValueError("Input lacks Hugo_Symbol")
        writer = csv.DictWriter(
            destination,
            fieldnames=[*reader.fieldnames, "Label", "Label_Name", "CGC_Tier"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            gene = row["Hugo_Symbol"].strip()
            cgc_row = cgc.get(gene)
            is_driver = cgc_row is not None
            label_name = "driver" if is_driver else "passenger"
            writer.writerow(
                {
                    **row,
                    "Label": "1" if is_driver else "0",
                    "Label_Name": label_name,
                    "CGC_Tier": cgc_row["Tier"] if cgc_row else "",
                }
            )
            counts["total"] += 1
            counts[label_name] += 1
            (driver_genes if is_driver else passenger_genes).add(gene)

    counts["driver_genes"] = len(driver_genes)
    counts["passenger_genes"] = len(passenger_genes)
    report_path.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n")
    return counts


def main() -> None:
    args = parse_args()
    counts = run(args.input, args.cgc, args.output, args.report)
    for key in sorted(counts):
        print(f"{key}\t{counts[key]}")


if __name__ == "__main__":
    main()
