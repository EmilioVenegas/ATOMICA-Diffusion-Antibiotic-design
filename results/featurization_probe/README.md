# Featurization probe — the diagnosis is only half right

Tests the claim that `scripts/process_expert_atomica.py` destroyed the
conditioning signal by collapsing each pocket into a single `UNK` block. Same 99
pockets, same single-segment pocket-only setup, differing **only** in block
vocabulary.

| | old `[GLB, UNK]` | new per-residue |
|---|---|---|
| blocks per pocket | 2 (1 distinct type) | 56.3 (17.9 distinct types) |
| mean pairwise cosine, graph repr | **1.0000** | **0.9917** |
| mean pairwise cosine, unit repr | 1.0000 | 0.9999 |
| composition probe R², graph | 0.201 | **0.176** |
| composition probe R², unit | 0.165 | **0.108** |

## Confirmed: the old featurization destroys pocket identity

Mean pairwise cosine similarity between *different* pockets is **1.0000**. Every
pocket maps to the same direction in embedding space. Whatever the adapter was
conditioned on, it was not pocket identity — which is exactly why arm B produced a
generic drug-likeness shift with no pocket specificity.

(The residual R² ≈ 0.20 comes from vector *magnitude*, which still varies; the
directions are degenerate.)

## Not confirmed: that fixing the block vocabulary repairs it

Per-residue blocks with real amino-acid types give 56 blocks and 18 distinct
types instead of 2 and 1 — the input is unquestionably richer. The representation
is barely different: cosine falls only from 1.0000 to 0.9917, still near
degenerate, and recoverable composition gets *worse*, not better (0.201 → 0.176
on graph, 0.165 → 0.108 on unit).

**A pocket-only ATOMICA encoding does not usefully distinguish pockets, and
correcting the block vocabulary does not change that.**

## Why, and what it implies

ATOMICA is pretrained on *interfaces* — two interacting segments. Encoding a lone
protein fragment is out of distribution no matter how well its blocks are typed;
there is no interaction for an interaction model to represent. The block
vocabulary was a real defect, but it was not the binding one.

This kills the cheapest plan — "fix the pocket featurization, re-run the same
conditioning" — before any GPU time was spent on it. Conditioning on a corrected
pocket-only embedding would reproduce the original result.

It leaves the two-segment approaches, which supply the missing partner:

- a **training-time critic**, comparing `ATOMICA(pocket, x̂₀)` against
  `ATOMICA(pocket, x_true)` — both are genuine two-segment interfaces, and this
  is the within-system regime measured at AUROC 1.000 in Phase 0;
- **conditioning on the partially-denoised ligand** during sampling, where the
  second segment is the model's own `x̂₀`.

Neither ever asks ATOMICA to represent a pocket in isolation.

## Reproduce

```bash
python scripts/featurization_probe.py --benchmark data/pose_benchmark
```

One target (`9WT9`) is skipped: the PS_300 tokenizer has no valence entry for iron.
