# Pose scorer — cross-system result

Does a head on ATOMICA interface representations rank docked poses on pockets it
has never seen? This is the gate on whether a usable tool exists
(`docs/experiment-plan.md`, Phase 2).

All numbers are out-of-fold under `GroupKFold` **by target**, so no test target
contributes to its own prediction; ridge alpha is chosen by inner CV inside each
training fold. Metric is CASF docking power: for each target, is the top-ranked
pose within 2 Å of the crystal pose. Targets where docking never produced a
sub-2 Å pose are excluded — no scorer can solve them.

## Result (100-target benchmark, 72 solvable, 1674 poses)

| Scorer | docking power | hits | mean per-target Spearman | vs smina |
|---|---|---|---|---|
| random (floor) | 15.7% | — | — | — |
| **smina (baseline)** | **59.7%** | **43/72** | — | — |
| graph (32-d) | 55.6% | 40/72 | +0.371 ± 0.084 | p = 0.70 |
| pocket_pool (96-d) | 41.7% | 30/72 | +0.344 ± 0.077 | **p = 0.019, worse** |
| all-block (288-d) | 63.9% | 46/72 | +0.400 ± 0.074 | p = 0.65 |

**Conclusion: the head is comparable to smina and does not beat it.** All-block
is 4 points ahead but wins 11 targets and loses 8 (McNemar p = 0.65), which is
not a difference. The scorer clearly learns something real — every variant is far
above the 15.7% floor, on targets never seen in training — but nothing here
justifies a tool that is slower than smina and needs a GPU.

## Retraction: the 22-target block-level result did not replicate

An earlier revision of this file reported, from the 22-target benchmark, that
block-level features "fixed the bottleneck": `pocket_pool` at 68.2% with Spearman
+0.521, and a paired gain over graph-level of +0.127 (t = 1.83, p ≈ 0.07) that
was described as supported.

At 72 targets `pocket_pool` measures **41.7%**, significantly *worse* than smina
(p = 0.019), and the paired Spearman gain is **−0.027 (p = 0.52)**. The effect did
not merely shrink, it reversed. That earlier conclusion is withdrawn.

Two things went wrong, both worth remembering:

- **Six feature sets were evaluated and the best reported.** The caveat was
  recorded at the time, but recording a selection bias does not remove it.
- **22 targets could not resolve differences of two or three targets.** Every
  comparison at that size had McNemar p ≥ 0.69; the ranking between variants was
  arbitrary, and the one that happened to lead was the one that reversed hardest.

The Spearman intervals tell the same story: ±0.140 at 22 targets against ±0.074
here. What survives the larger sample is narrow — all three variants sit between
+0.34 and +0.40, indistinguishable from each other.

## Where this leaves the tool question

It fails the decision gate in `docs/experiment-plan.md`, which asked for beating
smina or matching it with a complementary error profile. All-block matches, and
the errors are not complementary enough for the difference to register at
n = 72. Extraction into its own repository is not justified on this evidence.

What would change the answer is a better-posed learning problem, not more feature
engineering on pooled representations. Pose quality is a per-contact property and
every variant here pools it into a fixed-length vector before the head sees it;
an architecture that scores contacts directly is a different proposition. That is
a research project, not a packaging exercise.

Worth noting for comparability: at 6% of poses within 2 Å this benchmark is
harder than CASF-2016, whose decoys are curated across RMSD bins, and CASF has
285 targets to this set's 100.

## Reproduce

```bash
conda activate ~/.conda/envs/atomica-interface
python scripts/build_pose_benchmark.py --n_targets 100 --candidate_pool 2500
python scripts/featurize_block_level.py --benchmark data/pose_benchmark
```

One target (`9WT9`) is skipped: ATOMICA's PS_300 tokenizer has no valence entry
for iron and raises on haem-like ligands.
