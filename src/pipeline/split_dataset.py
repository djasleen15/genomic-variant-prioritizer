"""Create and persist a deterministic, label-stratified gene-level split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_VERSION = "gene-split-v1"
SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gene-map", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def assign_gene_splits(
    gene_labels: dict[str, str], seed: int = SPLIT_SEED
) -> dict[str, str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for gene, label in gene_labels.items():
        by_label[label].append(gene)

    assignments: dict[str, str] = {}
    randomizer = random.Random(seed)
    for label in sorted(by_label):
        genes = sorted(by_label[label])
        randomizer.shuffle(genes)
        validation_count = round(len(genes) * VALIDATION_FRACTION)
        test_count = round(len(genes) * (1 - TRAIN_FRACTION - VALIDATION_FRACTION))
        if len(genes) >= 3:
            validation_count = max(1, validation_count)
            test_count = max(1, test_count)
        train_end = len(genes) - validation_count - test_count
        validation_end = train_end + validation_count
        for gene in genes[:train_end]:
            assignments[gene] = "train"
        for gene in genes[train_end:validation_end]:
            assignments[gene] = "validation"
        for gene in genes[validation_end:]:
            assignments[gene] = "test"
    return assignments


def patient_id(sample_barcode: str) -> str:
    return "-".join(sample_barcode.split("-")[:3])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DisjointSet:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def gene_patient_components(rows: list[dict[str, str]]) -> dict[str, int]:
    graph = DisjointSet()
    for row in rows:
        graph.union(
            "gene:" + row["Hugo_Symbol"].strip(),
            "patient:" + patient_id(row["Tumor_Sample_Barcode"]),
        )
    components: dict[str, set[str]] = defaultdict(set)
    for node in graph.parent:
        components[graph.find(node)].add(node)
    largest = max(components.values(), key=len)
    return {
        "component_count": len(components),
        "largest_component_genes": sum(node.startswith("gene:") for node in largest),
        "largest_component_patients": sum(
            node.startswith("patient:") for node in largest
        ),
    }


def run(
    input_path: Path, output_path: Path, gene_map_path: Path, report_path: Path
) -> dict[str, object]:
    for path in (output_path, gene_map_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Input dataset has no header")
        rows = list(reader)
        fieldnames = reader.fieldnames

    gene_labels: dict[str, str] = {}
    for row in rows:
        gene = row["Hugo_Symbol"].strip()
        label = row["Label"]
        if gene in gene_labels and gene_labels[gene] != label:
            raise ValueError(f"Gene has inconsistent labels: {gene}")
        gene_labels[gene] = label
    assignments = assign_gene_splits(gene_labels)

    with gene_map_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Split_Version", "Gene", "Label", "Split"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for gene in sorted(assignments):
            writer.writerow(
                {
                    "Split_Version": SPLIT_VERSION,
                    "Gene": gene,
                    "Label": gene_labels[gene],
                    "Split": assignments[gene],
                }
            )

    variant_counts: Counter[str] = Counter()
    gene_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    with output_path.open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[*fieldnames, "Split_Version", "Split"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            split = assignments[row["Hugo_Symbol"].strip()]
            writer.writerow({**row, "Split_Version": SPLIT_VERSION, "Split": split})
            variant_counts[split] += 1
            label_counts[f"{split}_{row['Label_Name']}"] += 1
            patient_splits[patient_id(row["Tumor_Sample_Barcode"])].add(split)
    for split in assignments.values():
        gene_counts[split] += 1

    split_gene_sets = {
        split: {gene for gene, assigned in assignments.items() if assigned == split}
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(split_gene_sets["train"] & split_gene_sets["validation"]),
        "train_test": len(split_gene_sets["train"] & split_gene_sets["test"]),
        "validation_test": len(
            split_gene_sets["validation"] & split_gene_sets["test"]
        ),
    }
    train_patients = {
        patient for patient, splits in patient_splits.items() if "train" in splits
    }
    test_rows = [row for row in rows if assignments[row["Hugo_Symbol"].strip()] == "test"]
    test_variants_from_train_patients = sum(
        patient_id(row["Tumor_Sample_Barcode"]) in train_patients for row in test_rows
    )
    report: dict[str, object] = {
        "split_version": SPLIT_VERSION,
        "seed": SPLIT_SEED,
        "gene_map_sha256": sha256(gene_map_path),
        "variant_counts": dict(sorted(variant_counts.items())),
        "gene_counts": dict(sorted(gene_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "gene_overlap_counts": overlaps,
        "patient_count": len(patient_splits),
        "patients_present_in_multiple_splits": sum(
            len(splits) > 1 for splits in patient_splits.values()
        ),
        "test_variant_patient_overlap": {
            "test_variants": len(test_rows),
            "test_variants_from_patients_in_train": test_variants_from_train_patients,
            "fraction": (
                test_variants_from_train_patients / len(test_rows) if test_rows else 0.0
            ),
        },
        "gene_patient_components": gene_patient_components(rows),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    report = run(args.input, args.output, args.gene_map, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
