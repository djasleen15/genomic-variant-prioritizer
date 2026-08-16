"""Run the mandatory GRCh38 reference-allele concordance gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from pyfaidx import Fasta


REQUIRED_COLUMNS = {
    "Chromosome",
    "Start_Position",
    "Reference_Allele",
    "Liftover_Strand",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--concordant", required=True, type=Path)
    parser.add_argument("--mismatches", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-rate", default=0.99, type=float)
    return parser.parse_args()


class ChromosomeReferences:
    def __init__(self, reference_dir: Path):
        self.reference_dir = reference_dir
        self._references: dict[str, Fasta] = {}

    def base(self, chromosome: str, one_based_position: int) -> str:
        return self.sequence(chromosome, one_based_position - 1, one_based_position)

    def sequence(self, chromosome: str, start: int, end: int) -> str:
        if chromosome not in self._references:
            path = self.reference_dir / f"chr{chromosome}.fa"
            if not path.exists():
                raise FileNotFoundError(f"Missing reference FASTA: {path}")
            self._references[chromosome] = Fasta(str(path), as_raw=True)
        reference = self._references[chromosome]
        return str(reference[f"chr{chromosome}"][start:end]).upper()

    def close(self) -> None:
        for reference in self._references.values():
            reference.close()


def calculate_rates(counts: Counter[str]) -> dict[str, dict[str, int | float]]:
    report: dict[str, dict[str, int | float]] = {}
    for group in ("overall", "forward", "reverse"):
        total = counts[f"{group}_total"]
        matched = counts[f"{group}_matched"]
        mismatched = total - matched
        report[group] = {
            "total": total,
            "matched": matched,
            "mismatched": mismatched,
            "rate": matched / total if total else 0.0,
        }
    return report


def run(
    input_path: Path,
    reference_dir: Path,
    concordant_path: Path,
    mismatches_path: Path,
    report_path: Path,
) -> dict[str, dict[str, int | float]]:
    concordant_path.parent.mkdir(parents=True, exist_ok=True)
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    references = ChromosomeReferences(reference_dir)
    counts: Counter[str] = Counter()

    try:
        with (
            input_path.open(newline="") as source,
            concordant_path.open("w", newline="") as concordant_handle,
            mismatches_path.open("w", newline="") as mismatch_handle,
        ):
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("Input MAF has no header")
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"Input MAF is missing columns: {sorted(missing)}")
            writer = csv.DictWriter(
                mismatch_handle,
                fieldnames=[*reader.fieldnames, "Observed_Reference_Base"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            concordant_writer = csv.DictWriter(
                concordant_handle,
                fieldnames=reader.fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            concordant_writer.writeheader()

            for row in reader:
                group = "forward" if row["Liftover_Strand"] == "+" else "reverse"
                observed = references.base(
                    row["Chromosome"], int(row["Start_Position"])
                )
                expected = row["Reference_Allele"].upper()
                counts["overall_total"] += 1
                counts[f"{group}_total"] += 1
                if observed == expected:
                    counts["overall_matched"] += 1
                    counts[f"{group}_matched"] += 1
                    concordant_writer.writerow(row)
                else:
                    writer.writerow({**row, "Observed_Reference_Base": observed})
    finally:
        references.close()

    report = calculate_rates(counts)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    report = run(
        args.input,
        args.reference_dir,
        args.concordant,
        args.mismatches,
        args.report,
    )
    print(json.dumps(report, indent=2))
    failed_groups = [
        group for group, metrics in report.items() if metrics["rate"] < args.minimum_rate
    ]
    if failed_groups:
        raise SystemExit(
            "Reference concordance gate failed for: " + ", ".join(failed_groups)
        )


if __name__ == "__main__":
    main()
