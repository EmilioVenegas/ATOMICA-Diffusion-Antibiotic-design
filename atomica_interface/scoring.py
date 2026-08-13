"""Load ATOMICA's pretrained encoder and evaluate what its representations encode.

`DenoisePretrainModel` cannot be used for inference directly: its `forward`
requires the denoising targets (`tor_edges`, `tor_score`, `receptor_segment`, ...)
and asserts on them, and it defines no `infer`. ATOMICA's supported route to
pretrained representations is `PredictionModel._load_from_pretrained`, which
rebuilds the same encoder with the denoising heads disabled and exposes
`infer()` -> (unit_repr, block_repr, graph_repr).

The metrics below are deliberately dependency-light (numpy only) so they can be
tested without a torch environment.
"""

from typing import Optional

import numpy as np


def load_encoder(config_path: str, weights_path: str, device: str = "cpu"):
    """Return a pretrained ATOMICA encoder usable for representation extraction."""
    from ATOMICA.models.prediction_model import PredictionModel
    from ATOMICA.models.pretrain_model import DenoisePretrainModel

    pretrained = DenoisePretrainModel.load_from_config_and_weights(config_path, weights_path)
    model = PredictionModel._load_from_pretrained(pretrained)
    model.eval()
    return model.to(device)


def interface_representation(model, batch, level: str = "graph") -> np.ndarray:
    """Run the encoder and return one representation per complex.

    ``level`` selects ``graph`` (whole interface, pooled), ``block`` (per residue
    or fragment) or ``unit`` (per atom).
    """
    import torch

    with torch.no_grad():
        out = model.infer(batch)

    repr_tensor = {"graph": out.graph_repr, "block": out.block_repr, "unit": out.unit_repr}[level]
    return repr_tensor.detach().cpu().numpy()


# --------------------------------------------------------------------------
# Metrics -- pure numpy so they are testable without torch/sklearn installed.
# --------------------------------------------------------------------------


def roc_auc(labels, scores) -> float:
    """AUROC via the rank-sum identity, with ties averaged.

    Equivalent to the probability that a randomly chosen positive outranks a
    randomly chosen negative.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one positive and one negative")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    # Average ranks within tied score groups so ties contribute 0.5, not 0 or 1.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def enrichment_factor(labels, scores, fraction: float = 0.05) -> float:
    """Enrichment of positives in the top ``fraction`` of ranked scores.

    EF = 1.0 means no better than random. With small sets the top slice is only a
    few molecules, so report this alongside its sample size rather than alone.

    Ties are resolved to their expected value rather than by sort order. This
    matters: inputs are normally built by concatenating actives then decoys, so an
    order-dependent tie-break would rank every active first and report perfect
    enrichment for a constant (i.e. uninformative) score.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_top = max(1, int(round(len(scores) * fraction)))
    hit_rate_all = labels.mean()
    if hit_rate_all == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]

    # Walk groups of equal score. Groups fully inside the cutoff contribute all
    # their positives; the group straddling the cutoff contributes its positives
    # pro rata, which is the expectation over random orderings within the tie.
    expected_hits, filled, start = 0.0, 0, 0
    while filled < n_top and start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_size = end - start
        group_hits = int(sorted_labels[start:end].sum())
        take = min(group_size, n_top - filled)
        expected_hits += group_hits * (take / group_size)
        filled += take
        start = end

    return float((expected_hits / n_top) / hit_rate_all)


def permutation_test(labels, scores, n_permutations: int = 10000, seed: int = 0) -> float:
    """One-sided p-value for AUROC under label permutation.

    With tens of samples per class a linear probe can reach a flattering AUROC by
    chance; this reports how often shuffled labels do as well.
    """
    rng = np.random.default_rng(seed)
    observed = roc_auc(labels, scores)
    labels = np.asarray(labels).astype(int)

    count = sum(
        1 for _ in range(n_permutations) if roc_auc(rng.permutation(labels), scores) >= observed
    )
    return float((count + 1) / (n_permutations + 1))


def cross_validated_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    C: float = 1.0,
) -> Optional[np.ndarray]:
    """Out-of-fold decision scores from a stratified-CV logistic probe.

    Every score is produced by a model that never saw that sample, so the AUROC
    computed from the returned scores is not optimistically biased. Returns None
    if either class is too small to stratify.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels).astype(int)
    if min(np.bincount(labels)) < n_splits:
        return None

    oof = np.zeros(len(labels), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(features, labels):
        scaler = StandardScaler().fit(features[train_idx])
        clf = LogisticRegression(max_iter=5000, C=C).fit(
            scaler.transform(features[train_idx]), labels[train_idx]
        )
        oof[test_idx] = clf.decision_function(scaler.transform(features[test_idx]))
    return oof
