"""Phase 9 utilities for validation-only model-improvement experiments.

The dataset, labels, and ``gene-split-v1`` assignments are immutable in this
phase. Test rows must not be evaluated while configurations are being chosen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


POOLING_STRATEGIES = ("mutation", "mean", "attention")
EXPECTED_SPLITS = frozenset({"train", "validation", "test"})
DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
DEFAULT_VARIANT_KEY = (
    "Chromosome",
    "Position",
    "Reference_Allele",
    "Alternate_Allele",
)


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement and reject non-DNA characters."""
    invalid = set(sequence.upper()) - set("ACGTN")
    if invalid:
        raise ValueError(f"Sequence contains unsupported bases: {sorted(invalid)}")
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def reverse_complement_training_rows(
    frame: pd.DataFrame,
    *,
    reference_column: str = "Reference_Sequence",
    alternate_column: str = "Alternate_Sequence",
    split_column: str = "Split",
) -> pd.DataFrame:
    """Append reverse-complement copies of training rows only.

    Validation and test rows remain untouched so evaluation examples are never
    duplicated or altered. The returned frame records augmented rows explicitly.
    """
    required = {reference_column, alternate_column, split_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reverse-complement columns: {sorted(missing)}")
    augmented = frame.copy()
    augmented["Reverse_Complement_Augmented"] = False
    training = frame.loc[frame[split_column].eq("train")].copy()
    training[reference_column] = training[reference_column].map(reverse_complement)
    training[alternate_column] = training[alternate_column].map(reverse_complement)
    training["Reverse_Complement_Augmented"] = True
    return pd.concat([augmented, training], ignore_index=True)


def split_assignment_fingerprint(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = DEFAULT_VARIANT_KEY,
    split_column: str = "Split",
) -> str:
    """Hash variant-to-split assignments independently of row order."""
    columns = [*key_columns, split_column]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split fingerprint columns: {sorted(missing)}")
    canonical = frame.loc[:, columns].astype(str).sort_values(columns, kind="stable")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def enforce_fixed_split(
    frame: pd.DataFrame,
    *,
    expected_fingerprint: str | None = None,
    split_version: str = "gene-split-v1",
    key_columns: Sequence[str] = DEFAULT_VARIANT_KEY,
    gene_column: str = "Gene",
    split_column: str = "Split",
) -> str:
    """Validate Phase 9's immutable split and return its fingerprint."""
    if split_version != "gene-split-v1":
        raise ValueError(f"Phase 9 requires gene-split-v1, got {split_version!r}")
    if split_column not in frame or gene_column not in frame:
        raise ValueError(f"Dataset must contain {split_column!r} and {gene_column!r}")
    observed = set(frame[split_column].dropna().unique())
    if observed != EXPECTED_SPLITS:
        raise ValueError(f"Expected splits {sorted(EXPECTED_SPLITS)}, got {sorted(observed)}")
    gene_counts = frame.groupby(gene_column, dropna=False)[split_column].nunique()
    if (gene_counts > 1).any():
        examples = gene_counts.loc[gene_counts > 1].index.astype(str).tolist()[:5]
        raise ValueError(f"Gene-level split leakage detected: {examples}")
    fingerprint = split_assignment_fingerprint(
        frame, key_columns=key_columns, split_column=split_column
    )
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("Split assignment fingerprint differs from the fixed manifest")
    return fingerprint


@dataclass(frozen=True)
class ConservationFeatureSpec:
    """Training-derived imputation values for conservation features."""

    columns: tuple[str, ...]
    medians: tuple[float, ...]

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "medians": list(self.medians)}

    @classmethod
    def from_dict(cls, values: dict) -> "ConservationFeatureSpec":
        return cls(tuple(values["columns"]), tuple(float(x) for x in values["medians"]))


def fit_conservation_features(
    training_frame: pd.DataFrame, columns: Iterable[str]
) -> ConservationFeatureSpec:
    """Fit median imputation on training rows; never treat missing as zero."""
    names = tuple(columns)
    if not names:
        raise ValueError("At least one conservation-score column is required")
    missing = set(names) - set(training_frame.columns)
    if missing:
        raise ValueError(f"Missing conservation columns: {sorted(missing)}")
    numeric = training_frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0, skipna=True)
    unavailable = medians.index[medians.isna()].tolist()
    if unavailable:
        raise ValueError(f"No valid training scores for columns: {unavailable}")
    return ConservationFeatureSpec(names, tuple(float(medians[name]) for name in names))


def transform_conservation_features(
    frame: pd.DataFrame, spec: ConservationFeatureSpec
) -> torch.Tensor:
    """Return values plus explicit missingness flags for every score column."""
    missing = set(spec.columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing conservation columns: {sorted(missing)}")
    numeric = frame.loc[:, spec.columns].apply(pd.to_numeric, errors="coerce")
    missing_flags = numeric.isna().to_numpy(dtype=np.float32)
    filled = numeric.fillna(dict(zip(spec.columns, spec.medians))).to_numpy(dtype=np.float32)
    return torch.from_numpy(np.concatenate([filled, missing_flags], axis=1))


def write_experiment_report(path: Path, report: dict) -> None:
    """Persist the durable metrics record required before a phase is closed."""
    if "validation_auprc" not in report:
        raise ValueError("Phase 9 reports must include validation_auprc")
    if any(key.startswith("test") for key in report):
        raise ValueError("Configuration-selection reports must not contain test metrics")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
