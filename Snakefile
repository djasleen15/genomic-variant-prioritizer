configfile: "config/config.yaml"

PHASE2 = config["paths"]["phase2_dir"]
PHASE3 = config["paths"]["phase3_dir"]


rule all:
    input:
        f"{PHASE2}/labeled_split_dataset.tsv",
        f"{PHASE2}/phase2_validation.json",
        f"{PHASE3}/baseline_report.json",


rule liftover:
    input:
        maf=config["paths"]["source_maf"],
        chain=config["paths"]["liftover_chain"],
        script="scripts/liftover_maf_snvs.py",
    output:
        maf=config["paths"]["lifted_maf"],
        rejected=config["paths"]["liftover_rejected"],
    shell:
        "python {input.script} --input {input.maf} --chain {input.chain} "
        "--output {output.maf} --rejected {output.rejected}"


rule parse_maf:
    input:
        maf=rules.liftover.output.maf,
    output:
        clean=f"{PHASE2}/clean_variants.tsv",
        rejected=f"{PHASE2}/parse_rejected.tsv",
        report=f"{PHASE2}/parse_report.json",
    shell:
        "python -m src.pipeline.parse_maf --input {input.maf} --output {output.clean} "
        "--rejected {output.rejected} --report {output.report}"


rule reference_concordance:
    input:
        clean=rules.parse_maf.output.clean,
    output:
        concordant=f"{PHASE2}/concordant_variants.tsv",
        mismatches=f"{PHASE2}/reference_mismatches.tsv",
        report=f"{PHASE2}/reference_concordance.json",
    params:
        reference_dir=config["paths"]["reference_dir"],
    shell:
        "python -m src.pipeline.sanity_checks --input {input.clean} "
        "--reference-dir {params.reference_dir} --concordant {output.concordant} "
        "--mismatches {output.mismatches} --report {output.report}"


rule extract_sequences:
    input:
        concordant=rules.reference_concordance.output.concordant,
    output:
        pairs=f"{PHASE2}/sequence_pairs.tsv",
        rejected=f"{PHASE2}/sequence_rejected.tsv",
        report=f"{PHASE2}/sequence_report.json",
    params:
        reference_dir=config["paths"]["reference_dir"],
    shell:
        "python -m src.pipeline.extract_sequences --input {input.concordant} "
        "--reference-dir {params.reference_dir} --output {output.pairs} "
        "--rejected {output.rejected} --report {output.report}"


rule label_variants:
    input:
        pairs=rules.extract_sequences.output.pairs,
        cgc=config["paths"]["cgc"],
    output:
        labeled=f"{PHASE2}/labeled_variants.tsv",
        report=f"{PHASE2}/label_report.json",
    shell:
        "python -m src.pipeline.label_variants --input {input.pairs} --cgc {input.cgc} "
        "--output {output.labeled} --report {output.report}"


rule split_dataset:
    input:
        labeled=rules.label_variants.output.labeled,
    output:
        dataset=f"{PHASE2}/labeled_split_dataset.tsv",
        gene_map=f"{PHASE2}/gene_split_v1.tsv",
        report=f"{PHASE2}/split_report.json",
    shell:
        "python -m src.pipeline.split_dataset --input {input.labeled} "
        "--output {output.dataset} --gene-map {output.gene_map} --report {output.report}"


rule validate_phase2:
    input:
        dataset=rules.split_dataset.output.dataset,
    output:
        report=f"{PHASE2}/phase2_validation.json",
    shell:
        "python -m src.pipeline.validate_phase2 --input {input.dataset} "
        "--report {output.report}"


rule baseline:
    input:
        dataset=rules.split_dataset.output.dataset,
    output:
        report=f"{PHASE3}/baseline_report.json",
        sequence_model=f"{PHASE3}/sequence_only_logistic.joblib",
        forest_model=f"{PHASE3}/sequence_only_random_forest.joblib",
        diagnostic_model=f"{PHASE3}/cgc_inclusive_diagnostic_logistic.joblib",
    params:
        output_dir=PHASE3,
        mlflow_dir=config["paths"]["mlflow_dir"],
    shell:
        "python -m src.models.baseline --input {input.dataset} "
        "--output-dir {params.output_dir} --mlflow-dir {params.mlflow_dir} "
        "--report {output.report}"
