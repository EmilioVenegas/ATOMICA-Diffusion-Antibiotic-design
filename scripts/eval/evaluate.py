"""Utilities to compare Boltz-2 affinity runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_affinity_values(summary_dir: Path) -> np.ndarray:
    """Load `affinity_pred_value` scores from a Boltz summary directory."""

    scores_path = summary_dir / "affinity_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"Could not find affinity scores at {scores_path}")

    with scores_path.open("r", encoding="utf-8") as handle:
        affinity_scores = json.load(handle)

    values = [entry["affinity_pred_value"] for entry in affinity_scores.values()]
    if not values:
        raise ValueError(f"No affinity_pred_value entries found in {scores_path}")

    return np.asarray(values, dtype=float)


def ks_test_results(dir_a: Path, dir_b: Path) -> dict[str, float]:
    """Compute the two-sample Kolmogorov–Smirnov statistic between two runs."""
    """Interpretation:
    statistic is the Kolmogorov–Smirnov D value (0–1). 
    Largest gap between the cumulative distributions; 
    0: curves on top of each other, 
    1: completely disjoint.
    pvalue significance level to reject “both samples come from the same distribution.” 
    Conventional thresholds: p < 0.05 (or stricter)
    Small p (close to 0): distributions likely differ.
    Large p (close to 1): differences could just be sampling noise.
    n_a and n_b are the number of affinity scores in each set; small n makes the test less sensitive.
    """

    values_a = load_affinity_values(dir_a)
    values_b = load_affinity_values(dir_b)

    statistic, pvalue = ks_2samp(values_a, values_b)
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "n_a": int(len(values_a)),
        "n_b": int(len(values_b)),
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
    }

