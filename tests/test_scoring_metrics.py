"""Tests for the Phase 0 metrics.

These are pure-numpy and run without torch, rdkit or a trained model, so the
statistics behind any go/no-go decision are verified independently of the
environment needed to produce the representations.
"""

import numpy as np
import pytest

from atomica_interface.scoring import (
    enrichment_factor,
    permutation_test,
    roc_auc,
)


def test_perfect_separation():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert roc_auc(labels, scores) == pytest.approx(1.0)


def test_inverted_separation():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert roc_auc(labels, scores) == pytest.approx(0.0)


def test_all_ties_is_chance():
    """Constant scores must give exactly 0.5, not 0 or 1 depending on sort order."""
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert roc_auc(labels, scores) == pytest.approx(0.5)


def test_known_value():
    # One negative outranks one positive out of 2x2 pairs -> 3/4.
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.4, 0.6, 0.1]
    assert roc_auc(labels, scores) == pytest.approx(0.75)


def test_matches_reference_implementation_on_random_data():
    """Cross-check the rank-sum formula against a brute-force pair count."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        labels = rng.integers(0, 2, size=40)
        if labels.sum() in (0, len(labels)):
            continue
        scores = rng.normal(size=40).round(1)  # rounding forces ties

        pos, neg = scores[labels == 1], scores[labels == 0]
        brute = np.mean(
            [(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]
        )
        assert roc_auc(labels, scores) == pytest.approx(brute)


def test_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_enrichment_factor_perfect_and_random():
    labels = np.array([1] * 10 + [0] * 90)
    perfect = np.concatenate([np.ones(10), np.zeros(90)])
    # Top 10% is all positives; base rate is 10% -> 10x enrichment.
    assert enrichment_factor(labels, perfect, fraction=0.10) == pytest.approx(10.0)

    # A constant score ranks arbitrarily; enrichment should sit at chance.
    flat = np.zeros(100)
    assert enrichment_factor(labels, flat, fraction=0.10) == pytest.approx(1.0)


def test_enrichment_factor_ignores_input_ordering():
    """Constant scores must give EF = 1 even when the input is grouped by class.

    Regression test. Inputs here are built by concatenating actives then decoys,
    so an order-dependent tie-break ranks every active first and reports perfect
    enrichment for a completely uninformative score.
    """
    labels = np.array([1] * 50 + [0] * 50)
    assert enrichment_factor(labels, np.zeros(100), fraction=0.10) == pytest.approx(1.0)


def test_enrichment_factor_tie_straddling_cutoff():
    """A tie group crossing the cutoff contributes its positives pro rata.

    Top 2 of 4: the first slot is the uniquely top-scored positive, the second is
    drawn from three tied molecules of which one is positive. Expected positives
    = 1 + 1/3, hit rate = (4/3)/2, base rate = 1/2, so EF = 4/3.
    """
    labels = np.array([1, 1, 0, 0])
    scores = np.array([1.0, 0.0, 0.0, 0.0])
    assert enrichment_factor(labels, scores, fraction=0.50) == pytest.approx(4 / 3)


def test_permutation_test_flags_chance_and_signal():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 20 + [1] * 20)

    noise = rng.normal(size=40)
    assert permutation_test(labels, noise, n_permutations=500, seed=2) > 0.05

    separated = np.concatenate([rng.normal(-3, 0.5, 20), rng.normal(3, 0.5, 20)])
    assert permutation_test(labels, separated, n_permutations=500, seed=2) < 0.01
