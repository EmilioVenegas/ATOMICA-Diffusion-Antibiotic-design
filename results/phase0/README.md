# Phase 0 results

Go/no-go for [docs/experiment-plan.md](../../docs/experiment-plan.md).

## Pose sensitivity — PASSED

Is ATOMICA's two-segment interface representation sensitive to interaction
geometry? Native 4SP (NU6102) pose in CDK2 (1H1S, chain A) versus rigid
perturbations of it, 30 poses per class, out-of-fold linear probe.

The binding site is defined once from the native pose and **held fixed** across
every pose. Otherwise a displaced ligand contacts a different number of residues
and the representation can separate the classes on complex size alone — worth
AUROC ~0.70 here. With the site fixed, that shortcut measures exactly 0.500.

| Test | displaced RMSD | size confound | AUROC | perm p | Spearman(drift, RMSD) |
|---|---|---|---|---|---|
| easy | 4.5–8.3 Å | 0.500 | **1.000** | 0.0005 | +0.855 |
| hard | 1.1–2.3 Å | 0.500 | **0.999** | 0.0005 | +0.919 |

The hard row is the meaningful one: sub-2 Å is the conventional threshold for a
"correct" docked pose, and the representation separates that from near-native
almost perfectly while drifting monotonically with RMSD.

**Featurization sanity check.** Records now encode as 2 segments, 55 pocket
residue blocks + 8 ligand fragment blocks, 23 distinct block types — against the
`[GLB, UNK]` 2-block single-segment input the original pipeline produced.

## What this does and does not establish

Established: the representation encodes interaction geometry, it is finely
graded rather than merely detecting gross displacement, and the effect is not an
artifact of complex composition. The premise behind Phases 1–4 holds.

Not established:

- **One pocket, one ligand.** Rigid perturbations of a single chemotype in a
  single kinase. Confirm across pockets and ligands before generalising.
- **Pose sensitivity is not affinity discrimination.** This compares poses of the
  *same* molecule; it says nothing yet about ranking *different* molecules.
- Implication for the binder/decoy test: because the representation is this
  pose-sensitive, feeding it arbitrarily-oriented generated conformers will
  likely swamp any binding signal with pose noise. That test needs **docked**
  poses to be interpretable.

## Reproduce

```bash
conda activate ~/.conda/envs/atomica-interface
curl -o data/1h1s.pdb https://files.rcsb.org/download/1H1S.pdb

python scripts/phase0_pose_sensitivity.py \
    --pocket data/1h1s.pdb --chains A --ligand 4SP --ligand_chain A --n_poses 30

python scripts/phase0_pose_sensitivity.py \
    --pocket data/1h1s.pdb --chains A --ligand 4SP --ligand_chain A --n_poses 30 \
    --far_shift 0.5 1.5 --far_angle 10 30      # hard variant
```
