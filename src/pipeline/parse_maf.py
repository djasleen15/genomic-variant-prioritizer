"""Defensively validate and deduplicate the post-liftover BRCA MAF."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "Hugo_Symbol",
    "NCBI_Build",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "Liftover_Strand",
}
PRIMARY_CHROMOSOMES = {str(number) for number in range(1, 23)} | {"X", "Y"}
DNA_BASES = set("ACGT")
DEDUPLICATION_COLUMNS = (
    "Tumor_Sample_Barcode",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def validate_snv(row: dict[str, str]) -> str | None:
    if row["NCBI_Build"] != "GRCh38":
        return "unexpected_genome_build"
    if row["Chromosome"] not in PRIMARY_CHROMOSOMES:
        return "non_primary_chromosome"
    if row["Variant_Type"] != "SNP":
        return "not_snp"
    if row["Liftover_Strand"] not in {"+", "-"}:
        return "invalid_liftover_strand"

    reference = row["Reference_Allele"].upper()
    alternate = row["Tumor_Seq_Allele2"].upper()
    if len(reference) != 1 or reference not in DNA_BASES:
        return "invalid_reference_allele"
    if len(alternate) != 1 or alternate not in DNA_BASES:
        return "invalid_alternate_allele"
    if reference == alternate:
        return "reference_equals_alternate"

    try:
        start = int(row["Start_Position"])
        end = int(row["End_Position"])
    except ValueError:
        return "invalid_position"
    if start < 1 or end < 1:
        return "invalid_position"
    if start != end:
        return "snv_interval_not_one_base"
    return None


def deduplication_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in DEDUPLICATION_COLUMNS)


def clean_rows(
    rows: Iterable[dict[str, str]],
) -> tuple[list[dict[str, str]], list[tuple[dict[str, str], str]], Counter[str]]:
    retained: list[dict[str, str]] = []
    rejected: list[tuple[dict[str, str], str]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, ...]] = set()

    for source_row in rows:
        row = source_row.copy()
        counts["input"] += 1
        reason = validate_snv(row)
        if reason is None:
            key = deduplication_key(row)
            if key in seen:
                reason = "duplicate_variant_call"
            else:
                seen.add(key)
        if reason is not None:
            rejected.append((row, reason))
            counts[reason] += 1
            continue

        row["Reference_Allele"] = row["Reference_Allele"].upper()
        row["Tumor_Seq_Allele2"] = row["Tumor_Seq_Allele2"].upper()
        retained.append(row)
        counts["retained"] += 1

    return retained, rejected, counts


def run(
    input_path: Path, output_path: Path, rejected_path: Path, report_path: Path
) -> Counter[str]:
    for path in (output_path, rejected_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Input MAF has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input MAF is missing columns: {sorted(missing)}")
        fieldnames = reader.fieldnames
        retained, rejected, counts = clean_rows(reader)

    with output_path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(retained)

    with rejected_path.open("w", newline="") as rejected_handle:
        writer = csv.DictWriter(
            rejected_handle,
            fieldnames=[*fieldnames, "Phase2_Rejection_Reason"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row, reason in rejected:
            writer.writerow({**row, "Phase2_Rejection_Reason": reason})

    report_path.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n")
    return counts


def main() -> None:
    args = parse_args()
    counts = run(args.input, args.output, args.rejected, args.report)
    for key in sorted(counts):
        print(f"{key}\t{counts[key]}")


if __name__ == "__main__":
    main()
