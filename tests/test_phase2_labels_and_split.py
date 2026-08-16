import csv

from src.pipeline.label_variants import run as run_labels
from src.pipeline.split_dataset import assign_gene_splits, run as run_split


def write_tsv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def test_cgc_membership_labels_driver_and_passenger(tmp_path):
    input_path = tmp_path / "sequences.tsv"
    write_tsv(
        input_path,
        [
            {"Hugo_Symbol": "DRIVER1", "Variant_ID": "v1"},
            {"Hugo_Symbol": "OTHER", "Variant_ID": "v2"},
        ],
    )
    cgc_path = tmp_path / "cgc.csv"
    with cgc_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Gene Symbol", "Tier"])
        writer.writeheader()
        writer.writerow({"Gene Symbol": "DRIVER1", "Tier": "1"})

    counts = run_labels(
        input_path, cgc_path, tmp_path / "labeled.tsv", tmp_path / "report.json"
    )

    assert counts["driver"] == 1
    assert counts["passenger"] == 1
    labeled = list(csv.DictReader((tmp_path / "labeled.tsv").open(), delimiter="\t"))
    assert [(row["Label"], row["CGC_Tier"]) for row in labeled] == [
        ("1", "1"),
        ("0", ""),
    ]


def test_gene_split_is_deterministic_stratified_and_disjoint():
    labels = {f"driver_{index}": "1" for index in range(10)}
    labels.update({f"passenger_{index}": "0" for index in range(20)})

    first = assign_gene_splits(labels)
    second = assign_gene_splits(labels)

    assert first == second
    for label in {"0", "1"}:
        splits = {first[gene] for gene, gene_label in labels.items() if gene_label == label}
        assert splits == {"train", "validation", "test"}
    assert set(first) == set(labels)


def test_split_keeps_each_gene_wholly_in_one_split(tmp_path):
    input_path = tmp_path / "labeled.tsv"
    rows = []
    for index in range(20):
        for variant in range(2):
            rows.append(
                {
                    "Variant_ID": f"v{index}_{variant}",
                    "Hugo_Symbol": f"gene_{index}",
                    "Label": str(index % 2),
                    "Label_Name": "driver" if index % 2 else "passenger",
                    "Tumor_Sample_Barcode": "TCGA-AA-0001-01",
                }
            )
    write_tsv(input_path, rows)

    report = run_split(
        input_path,
        tmp_path / "split.tsv",
        tmp_path / "gene_map.tsv",
        tmp_path / "report.json",
    )

    split_rows = list(csv.DictReader((tmp_path / "split.tsv").open(), delimiter="\t"))
    gene_splits = {}
    for row in split_rows:
        gene_splits.setdefault(row["Hugo_Symbol"], set()).add(row["Split"])
    assert all(len(splits) == 1 for splits in gene_splits.values())
    assert all(count == 0 for count in report["gene_overlap_counts"].values())
    assert report["test_variant_patient_overlap"]["test_variants"] > 0
    assert (
        report["test_variant_patient_overlap"][
            "test_variants_from_patients_in_train"
        ]
        == report["test_variant_patient_overlap"]["test_variants"]
    )
    assert report["test_variant_patient_overlap"]["fraction"] == 1.0
