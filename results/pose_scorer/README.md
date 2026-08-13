# Pose scorer — cross-system result

Does a head on ATOMICA interface representations rank docked poses on pockets it
has never seen? This is the gate on whether a usable tool exists
(`docs/experiment-plan.md`, Phase 2).

All numbers are out-of-fold under `GroupKFold` **by target**, so no test target
contributes to its own prediction. Metric is CASF docking power: for each target,
is the top-ranked pose within 2 Å of the crystal pose. The 8 targets where
docking never produced a sub-2 Å pose are excluded — no scorer can solve them.

## Result: matches smina, does not beat it

| Scorer | docking power | hits |
|---|---|---|
| random (floor) | 9.4% | — |
| **smina (baseline)** | **63.6%** | **14/22** |
| ATOMICA 32-d + ridge | 59.1% | 13/22 |
| ATOMICA 32-d + gradient boosting | 45.5% | 10/22 |
| ATOMICA + smina combined | 54.5% | 12/22 |

Mean per-target Spearman(predicted, true RMSD) for the ridge head: **+0.370**.

The head is far above the 9.4% random floor, so it has learned something real
about pose quality that transfers to unseen targets. It does not beat smina.

**The gap is not measurable at this sample size.** Ridge and smina differ on 3 of
22 targets (ATOMICA alone gets `9TMX`; smina alone gets `29LA` and `9SZK`), giving
McNemar exact **p = 1.000**. Reporting 59.1% as "worse than 63.6%" would be
over-reading one target.

Adding capacity hurts (gradient boosting, 45.5%) and so does combining with smina
(54.5%), both consistent with overfitting 520 poses through 32 features.

## What this means for the tool

As built, it does not justify itself. A scorer that matches smina while being
slower and requiring a GPU has no reason to exist. Under the decision gate in
`docs/experiment-plan.md` — beat smina, or match it with a *different* error
profile — it fails both: 12 of smina's 14 hits are shared, so the errors are
largely the same errors.

## The untested lever

`graph_repr` is **32-dimensional**. Everything above reads pose quality through
those 32 numbers, and both attempts to add head capacity overfit immediately,
which is what a representation bottleneck looks like.

ATOMICA also exposes `block_repr` (per residue/fragment) and `unit_repr` (per
atom). Pose quality is a local, per-contact property, so pooling block-level
features — separately for the pocket and ligand segments — is the obvious next
attempt and is cheap: `atomica_interface.scoring.interface_representation` already
takes a `level` argument, and the benchmark and harness are built.

Also worth noting before concluding anything about ATOMICA: at 6% of poses within
2 Å this benchmark is harder than CASF-2016, whose decoys are curated to be
balanced across RMSD bins. n = 22 targets is small; CASF-2016 has 285.

## Reproduce

```bash
conda activate ~/.conda/envs/atomica-interface
python scripts/train_pose_scorer.py --benchmark data/pose_benchmark
```

Representations are cached in `features.npz`; delete it to re-featurize.
