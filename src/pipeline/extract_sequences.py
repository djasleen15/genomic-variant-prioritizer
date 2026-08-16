"""Extract centered GRCh38 reference/alternate windows and check orientation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from src.pipeline.sanity_checks import ChromosomeReferences


WINDOW_SIZE = 512
VARIANT_INDEX = WINDOW_SIZE // 2
DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")
HGVSC_SUBSTITUTION = re.compile(r"([ACGT])>([ACGT])$")
OUTPUT_COLUMNS = [
    "Variant_ID",
    "Chromosome",
    "Position",
    "Reference_Allele",
    "Alternate_Allele",
    "Hugo_Symbol",
    "Tumor_Sample_Barcode",
    "Transcript_ID",
    "Liftover_Strand",
    "Transcript_Strand_Check",
    "Reference_Sequence",
    "Alternate_Sequence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def complement(base: str) -> str:
    return base.translate(DNA_COMPLEMENT)


def transcript_strand_check(row: dict[str, str]) -> str:
    """Compare genomic alleles with transcript-oriented alleles encoded in HGVSc."""
    match = HGVSC_SUBSTITUTION.search(row.get("HGVSc", "").upper())
    if match is None:
        return "not_applicable"
    transcript_ref, transcript_alt = match.groups()
    genomic_ref = row["Reference_Allele"].upper()
    genomic_alt = row["Tumor_Seq_Allele2"].upper()
    if (transcript_ref, transcript_alt) == (genomic_ref, genomic_alt):
        return "plus"
    if (transcript_ref, transcript_alt) == (
        complement(genomic_ref),
        complement(genomic_alt),
    ):
        return "minus"
    return "mismatch"


def variant_id(row: dict[str, str]) -> str:
    identity = ":".join(
        (
            row["Tumor_Sample_Barcode"],
            row["Chromosome"],
            row["Start_Position"],
            row["Reference_Allele"],
            row["Tumor_Seq_Allele2"],
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def extract_window(
    row: dict[str, str], references: ChromosomeReferences
) -> tuple[str, str]:
    chromosome = row["Chromosome"]
    position = int(row["Start_Position"])
    start = position - 1 - VARIANT_INDEX
    end = start + WINDOW_SIZE
    if start < 0:
        raise ValueError("window_crosses_chromosome_start")
    sequence = references.sequence(chromosome, start, end)
    if len(sequence) != WINDOW_SIZE:
        raise ValueError("window_crosses_chromosome_end")
    expected = row["Reference_Allele"].upper()
    if sequence[VARIANT_INDEX] != expected:
        raise ValueError("center_reference_mismatch")
    alternate = row["Tumor_Seq_Allele2"].upper()
    alternate_sequence = (
        sequence[:VARIANT_INDEX] + alternate + sequence[VARIANT_INDEX + 1 :]
    )
    return sequence, alternate_sequence


def run(
    input_path: Path,
    reference_dir: Path,
    output_path: Path,
    rejected_path: Path,
    report_path: Path,
) -> Counter[str]:
    for path in (output_path, rejected_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    references = ChromosomeReferences(reference_dir)

    try:
        with (
            input_path.open(newline="") as source,
            output_path.open("w", newline="") as destination,
            rejected_path.open("w", newline="") as rejected_handle,
        ):
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("Input MAF has no header")
            writer = csv.DictWriter(
                destination,
                fieldnames=OUTPUT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            rejected_writer = csv.DictWriter(
                rejected_handle,
                fieldnames=[*reader.fieldnames, "Sequence_Rejection_Reason"],
                delimiter="\t",
                lineterminator="\n",
            )
            rejected_writer.writeheader()

            for row in reader:
                counts["input"] += 1
                strand_check = transcript_strand_check(row)
                counts[f"strand_{strand_check}"] += 1
                if strand_check == "mismatch":
                    rejected_writer.writerow(
                        {**row, "Sequence_Rejection_Reason": "transcript_strand_mismatch"}
                    )
                    counts["rejected"] += 1
                    continue
                try:
                    reference_sequence, alternate_sequence = extract_window(
                        row, references
                    )
                except ValueError as error:
                    rejected_writer.writerow(
                        {**row, "Sequence_Rejection_Reason": str(error)}
                    )
                    counts[str(error)] += 1
                    counts["rejected"] += 1
                    continue
                writer.writerow(
                    {
                        "Variant_ID": variant_id(row),
                        "Chromosome": row["Chromosome"],
                        "Position": row["Start_Position"],
                        "Reference_Allele": row["Reference_Allele"],
                        "Alternate_Allele": row["Tumor_Seq_Allele2"],
                        "Hugo_Symbol": row["Hugo_Symbol"],
                        "Tumor_Sample_Barcode": row["Tumor_Sample_Barcode"],
                        "Transcript_ID": row.get("Transcript_ID", ""),
                        "Liftover_Strand": row["Liftover_Strand"],
                        "Transcript_Strand_Check": strand_check,
                        "Reference_Sequence": reference_sequence,
                        "Alternate_Sequence": alternate_sequence,
                    }
                )
                counts["retained"] += 1
    finally:
        references.close()

    report_path.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n")
    return counts


def main() -> None:
    args = parse_args()
    counts = run(
        args.input, args.reference_dir, args.output, args.rejected, args.report
    )
    for key in sorted(counts):
        print(f"{key}\t{counts[key]}")
    if counts["rejected"]:
        raise SystemExit("Sequence extraction rejected one or more records")


if __name__ == "__main__":
    main()
