# Critic sanity gate — does the loss have a gradient worth following?

`scripts/critic_sanity_gate.py`, `gate_cache.json`, `gate_random.json`.

The proposed objective is

```
L = L_diffusion + lambda(t) * d( ATOMICA(pocket, x0_hat), ATOMICA(pocket, x_true) )
```

which is only worth GPU-days if `d` grows as `x0_hat` moves away from `x_true`,
and grows *where the critic is applied*. This is the cheap check, run before any
training.

**Verdict: the gate passes for `graph_cosine`, and the gap over a permuted-weight
control shows pretraining is what carries the signal.**

## Setup

`data/pose_benchmark`, 92 targets, 1,662 poses (of 100 targets / 1,691 poses;
`9WT9` raises `KeyError: 'Fe'` in the PS_300 tokenizer, and targets with fewer
than 5 scored poses are dropped). Every pose of a target is the *same molecule*
rigidly displaced by a known symmetry-aware in-place RMSD, so composition and
chemistry are controlled by construction and only interaction geometry varies.
That is the within-system Phase 0 regime, which is the regime the critic
operates in — it is never asked for a transferable score across pockets.

Every statistic is **per target**, then averaged over targets. Pooling poses
across targets would measure the cross-system transfer that Phase 2 already
resolved as negative.

- `rho` — Spearman between the distance and pose RMSD.
- `rho(<4A)` — the same, restricted to poses within 4 Å. **This is the number
  that matters.** A loss separating 0.5 Å from 30 Å is useless for refinement;
  at low `t` the denoiser's `x0_hat` is already close.
- `AUROC` — separates RMSD < 2 Å from > 4 Å, poses in between excluded.

## Results

| metric | rho(all) | rho(<4 Å) | AUROC | frac targets AUROC > 0.5 |
|---|---|---|---|---|
| **graph_cosine (pretrained)** | **+0.386 ± 0.034** | **+0.558 ± 0.059** | **0.926 ± 0.022** | 0.97 |
| graph_cosine (random weights) | +0.149 ± 0.040 | +0.061 ± 0.075 | 0.697 ± 0.032 | 0.77 |
| contacts (no-learning floor) | +0.253 ± 0.034 | +0.355 ± 0.066 | 0.837 ± 0.026 | 0.94 |
| smina (established reference) | +0.281 ± 0.032 | +0.465 ± 0.060 | 0.844 ± 0.029 | 0.88 |
| ligand_pool_l2 (pretrained) | +0.303 ± 0.031 | +0.183 ± 0.080 | 0.823 ± 0.032 | 0.85 |

`graph_cosine` beats the no-learning floor by +0.089 AUROC and the permuted-weight
control by +0.229. In the low-RMSD regime the gap against the control is the
whole signal: +0.558 against +0.061.

Normalised distance by RMSD bin, mean over targets:

| metric | 0–1 Å | 1–2 Å | 2–4 Å | 4–8 Å | 8+ Å |
|---|---|---|---|---|---|
| graph_cosine (pretrained) | 0.007 | 0.099 | 0.284 | 0.504 | 0.611 |
| graph_cosine (random weights) | 0.247 | 0.220 | 0.311 | 0.365 | 0.445 |

The pretrained profile rises monotonically from zero. The control does not even
rise from the first bin to the second, which is exactly the region the critic is
weighted toward.

## The control that was wrong, and why it matters

The first version of this gate used `pocket_pool` — the pooled block
representations of the pocket segment — as its negative control, on the
assumption stated in `scripts/featurize_block_level.py` that it is "identical for
every pose of a target". **That assumption is false.** The input pocket blocks are
identical, but their representations are computed with message passing from the
ligand, so they move with the pose like everything else.

It scored **0.949 AUROC against the real metric's 0.926** — the intended floor
came out on top, because it was never a floor.

The permuted-weight run then explains it: `pocket_pool` scores **0.923 with random
weights**, against `graph_cosine`'s 0.697. Almost all of `pocket_pool`'s apparent
strength is architecture and geometry rather than learned interaction chemistry.
Selecting it as the critic metric — which its raw number invites — would have
produced a loss that is mostly a geometric penalty wearing a foundation model,
and the eventual result would have been uninterpretable.

This is the same failure the hotspot phase hit: a trivial buriedness baseline
reaching the 98.2nd percentile while the ATOMICA field sat at random. **A control
has to be shown to be a floor, not assumed to be one.**

## Consequences for the loss

1. **Use `graph_cosine`.** It is the metric with the largest margin over both
   floors. `pocket_pool` is excluded despite the best raw number.
2. **Ramp `lambda` off at high noise.** Not tuning — necessity. The distance
   saturates and turns over once the ligand leaves the pocket
   (`tests/test_critic_roundtrip.py`: at a rigid displacement of 2 Å the distance
   is larger than at 5 Å), so a gradient taken there can point the wrong way.
   ATOMICA has never seen a half-formed ligand.
3. **`max_weight: 1.0` is too small.** Measured on a real batch: critic distance
   0.0114, weighted 0.0070, against a diffusion nll of 0.60 — about 0.3% of the
   objective. Cosine distance on a 32-d representation is intrinsically small.
   Expect to need 1e1–1e2; sweep it and confirm `critic_distance/train` falls.

## Reproducing

```bash
python scripts/critic_sanity_gate.py --encoder cache  --device cuda   # pretrained + contacts floor
python scripts/critic_sanity_gate.py --encoder random --device cuda --skip_contacts
```

`--encoder cache` reads pose vectors from `results/pose_scorer/features_block.npz`
and encodes only the 99 natives, so it costs one forward pass per target.
`--encoder random` must encode everything (~5 min on GPU; it took 5 hours on CPU).
