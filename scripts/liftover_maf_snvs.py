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


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)
    lifter = LiftOver(str(args.chain))
    counts: Counter[str] = Counter()

    with (
        args.input.open(newline="") as source,
        args.output.open("w", newline="") as destination,
        args.rejected.open("w", newline="") as rejected,
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
            if row["NCBI_Build"] != "GRCh37":
                reject(rejected_writer, row, "unexpected_source_build")
                counts["unexpected_source_build"] += 1
                continue
            if row["Variant_Type"] != "SNP":
                reject(rejected_writer, row, "not_snv")
                counts["not_snv"] += 1
                continue
            if row["Start_Position"] != row["End_Position"]:
                reject(rejected_writer, row, "snv_interval_not_one_base")
                counts["snv_interval_not_one_base"] += 1
                continue

            try:
                source_position = int(row["Start_Position"]) - 1
            except ValueError:
                reject(rejected_writer, row, "invalid_position")
                counts["invalid_position"] += 1
                continue

            mappings = lifter.convert_coordinate(
                normalize_chromosome(row["Chromosome"]), source_position
            )
            if not mappings:
                reject(rejected_writer, row, "unmapped")
                counts["unmapped"] += 1
                continue
            if len(mappings) != 1:
                reject(rejected_writer, row, "multiple_mappings")
                counts["multiple_mappings"] += 1
                continue

            target_chromosome, target_position, target_strand, _ = mappings[0]
            if target_strand not in {"+", "-"}:
                reject(rejected_writer, row, "invalid_target_strand")
                counts["invalid_target_strand"] += 1
                continue
            target_maf_chromosome = maf_chromosome(target_chromosome)
            if target_maf_chromosome not in PRIMARY_CHROMOSOMES:
                reject(rejected_writer, row, "non_primary_target_contig")
                counts["non_primary_target_contig"] += 1
                continue

            row["NCBI_Build"] = "GRCh38"
            row["Chromosome"] = target_maf_chromosome
            row["Start_Position"] = str(target_position + 1)
            row["End_Position"] = str(target_position + 1)
            if target_strand == "-":
                for column in (
                    "Reference_Allele",
                    "Tumor_Seq_Allele1",
                    "Tumor_Seq_Allele2",
                ):
                    row[column] = complement(row[column])
                counts["reverse_strand"] += 1
            # Alleles are emitted relative to the target reference's forward strand.
            row["Strand"] = "+"

            output_writer.writerow(row)
            counts["lifted"] += 1

    for key in sorted(counts):
        print(f"{key}\t{counts[key]}")


if __name__ == "__main__":
    main()
