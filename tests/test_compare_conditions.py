"""Guard the headline numbers.

The README and docs/results.md quote specific values from the A/B ablation. Those
are derived from the committed per-condition metrics, so they can silently drift
out of sync if either the metrics or the comparison logic changes. These tests
recompute them from the committed evidence and fail if the published claims no
longer hold.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline_A"
CONDITIONED = ROOT / "results" / "cond_B"

pytestmark = pytest.mark.skipif(
    not (BASELINE / "metrics.json").exists(),
    reason="ablation metrics not present in this checkout",
)


def load(path):
    return json.loads((path / "metrics.json").read_text())


def test_both_arms_cover_the_same_pockets():
    """A per-pocket comparison is only meaningful over a shared pocket set."""
    a, b = load(BASELINE), load(CONDITIONED)
    assert a["n_pockets"] == b["n_pockets"] == 100


def test_both_arms_have_comparable_sample_counts():
    a, b = load(BASELINE), load(CONDITIONED)
    # Generation drops a few samples per pocket; arms must still be within 5%
    # of each other or the distributional metrics are not comparable.
    assert abs(a["n_valid"] - b["n_valid"]) / a["n_valid"] < 0.05


def test_conditioning_improves_drug_likeness():
    """The core claim: QED rises under ATOMICA conditioning."""
    a, b = load(BASELINE), load(CONDITIONED)
    relative_gain = (b["qed_mean"] - a["qed_mean"]) / a["qed_mean"]
    assert relative_gain > 0.10, f"QED gain fell to {relative_gain:.1%}"


def test_diversity_cost_is_reported_not_hidden():
    """The tradeoff is part of the published result; assert it still holds.

    If diversity ever stops dropping, the honest caveat in the README and
    docs/results.md is stale and must be rewritten rather than quietly kept.
    """
    a, b = load(BASELINE), load(CONDITIONED)
    assert b["diversity_mean"] < a["diversity_mean"]


def test_comparison_script_runs_and_reports_both_arms(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_conditions.py"),
            "--conditions",
            str(BASELINE),
            str(CONDITIONED),
            "--outdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    summary = (tmp_path / "ablation_summary.md").read_text()
    assert "QED" in summary and "Diversity" in summary
    assert (tmp_path / "figures" / "ablation.png").exists()
