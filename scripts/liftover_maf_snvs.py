#!/usr/bin/env python3
"""Lift a GRCh37 MAF's SNVs to GRCh38 with auditable rejection output.

Coordinates in MAF files are 1-based and inclusive. ``pyliftover`` accepts
0-based positions, so each SNV position is decremented before conversion and
incremented when written. Non-SNV records are outside the v1 project scope and
are recorded in the rejection file rather than silently discarded.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from pyliftover import LiftOver


REQUIRED_COLUMNS = {
    "NCBI_Build",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Strand",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
}
BASE_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
PRIMARY_CHROMOSOMES = {str(number) for number in range(1, 23)} | {"X", "Y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lift GRCh37 MAF SNVs to GRCh38."
    )
    parser.add_argument("--input", required=True, type=Path, help="GRCh37 MAF")
    parser.add_argument("--chain", required=True, type=Path, help="UCSC chain.gz")
    parser.add_argument("--output", required=True, type=Path, help="GRCh38 MAF")
    parser.add_argument(
        "--rejected", required=True, type=Path, help="Rejected records TSV"
    )
    return parser.parse_args()


def normalize_chromosome(chromosome: str) -> str:
    return chromosome if chromosome.startswith("chr") else f"chr{chromosome}"


def maf_chromosome(chromosome: str) -> str:
    return chromosome.removeprefix("chr")


def complement(allele: str) -> str:
    return allele.translate(BASE_COMPLEMENT)


def reject(writer: csv.DictWriter, row: dict[str, str], reason: str) -> None:
    writer.writerow({**row, "Liftover_Rejection_Reason": reason})


def convert_snv_record(
    source_row: dict[str, str], lifter: LiftOver
) -> tuple[dict[str, str] | None, str | None, bool]:
    """Convert one MAF record, returning output, rejection reason, and reversal."""
    row = source_row.copy()
    if row["NCBI_Build"] != "GRCh37":
        return None, "unexpected_source_build", False
    if row["Variant_Type"] != "SNP":
        return None, "not_snv", False

    try:
        start_position = int(row["Start_Position"])
        end_position = int(row["End_Position"])
    except ValueError:
        return None, "invalid_position", False
    if start_position < 1 or end_position < 1:
        return None, "invalid_position", False
    if start_position != end_position:
        return None, "snv_interval_not_one_base", False

    # MAF coordinates are 1-based; pyliftover coordinates are 0-based.
    source_position = start_position - 1
    mappings = lifter.convert_coordinate(
        normalize_chromosome(row["Chromosome"]), source_position
    )
    if not mappings:
        return None, "unmapped", False
    if len(mappings) != 1:
        return None, "multiple_mappings", False

    target_chromosome, target_position, target_strand, _ = mappings[0]
    if target_strand not in {"+", "-"}:
        return None, "invalid_target_strand", False
    target_maf_chromosome = maf_chromosome(target_chromosome)
    if target_maf_chromosome not in PRIMARY_CHROMOSOMES:
        return None, "non_primary_target_contig", False

    row["NCBI_Build"] = "GRCh38"
    row["Chromosome"] = target_maf_chromosome
    # pyliftover returns a 0-based target coordinate; MAF is 1-based.
    row["Start_Position"] = str(target_position + 1)
    row["End_Position"] = str(target_position + 1)
    reverse_strand = target_strand == "-"
    if reverse_strand:
        for column in (
            "Reference_Allele",
            "Tumor_Seq_Allele1",
            "Tumor_Seq_Allele2",
        ):
            row[column] = complement(row[column])
    # Alleles are emitted relative to the target reference's forward strand.
    row["Strand"] = "+"
    return row, None, reverse_strand


def process_maf(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    lifter: LiftOver,
) -> Counter[str]:
    """Convert a MAF and return mutually auditable processing counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    with (
        input_path.open(newline="") as source,
        output_path.open("w", newline="") as destination,
        rejected_path.open("w", newline="") as rejected,
    ):
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Input MAF has no header")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input MAF is missing columns: {sorted(missing)}")

        output_writer = csv.DictWriter(
            destination,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        rejected_writer = csv.DictWriter(
            rejected,
            fieldnames=[*reader.fieldnames, "Liftover_Rejection_Reason"],
            delimiter="\t",
            lineterminator="\n",
        )
        output_writer.writeheader()
        rejected_writer.writeheader()

        for row in reader:
            counts["input"] += 1
            converted, reason, reverse_strand = convert_snv_record(row, lifter)
            if reason is not None:
                reject(rejected_writer, row, reason)
                counts[reason] += 1
                continue
            if reverse_strand:
                counts["reverse_strand"] += 1
            assert converted is not None
            output_writer.writerow(converted)
            counts["lifted"] += 1

    return counts


def main() -> None:
    args = parse_args()
    lifter = LiftOver(str(args.chain))
    counts = process_maf(args.input, args.output, args.rejected, lifter)
    for key in sorted(counts):
        print(f"{key}\t{counts[key]}")


if __name__ == "__main__":
    main()
