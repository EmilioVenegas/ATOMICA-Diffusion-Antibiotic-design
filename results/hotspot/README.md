# Hotspot fields — negative result

Does scoring chemical probes on a grid with ATOMICA's pretrained denoising energy
produce a useful interaction hotspot field? Protocol from Radoux et al. 2016
(median percentile rank of crystal ligand atoms in the matching probe's field).

## Result: the field is indistinguishable from random

CDK2 / NU6102 (1H1S chain A), 5 Å site, 1.5 Å grid, 1683 accessible non-clashing
points, 6 probes × 2 orientations, scored with the training-free denoising energy.

| Measure | Value | Reference |
|---|---|---|
| median percentile, matched probe | **52.4** | Radoux: 97 (fragments), 72 (leads) |
| median percentile, **buriedness control** | **98.2** | the confound |
| median percentile, random placement | 52.2 | the floor |
| type specificity (matching probe wins) | **0.107** | chance = 0.167 |

The field carries no information about where ligand atoms sit: 52.4 against a
random floor of 52.2. Type specificity is **below chance**, so it does not even
weakly assign the right chemistry to the right subpocket.

## The buriedness number is the useful part

Protein neighbour count alone reaches the 98.2nd percentile — it beats our field,
and it would beat Radoux's published fragment number too. That is not a result
about ATOMICA; it is confirmation that the harness works. The metric *can*
detect a field that predicts ligand positions, and it detects one here. The
signal is simply absent from the ATOMICA field.

It also confirms the survey's central warning: buriedness dominates this
literature, and any hotspot method that does not report it as a baseline can look
excellent while measuring nothing but enclosure. Had we skipped that control and
reported only "ligand atoms land in the 98th percentile of buriedness-weighted
scores", the method would have looked like a success.

## What this does and does not rule out

**Ruled out:** the training-free denoising energy as a probe score. This is the
third measurement pointing the same way — it scored 0.787 against a 0.727 trivial
baseline on pose discrimination, and now sits at the random floor here. The
pretrained heads do not surface a usable interaction score.

**Not ruled out, but not worth pursuing here:** a *trained* readout on probe
representations. That is supervised hotspot prediction from protein structure,
which is exactly PharmacoNet (Chem Sci 2024, MIT licensed, protein-only,
generalises to unseen targets). Training our own would be reimplementing a
published method with a weaker backbone.

**Caveats.** One pocket, one ligand. Coarse settings chosen for tractability
(5 Å site, 1.5 Å spacing, 2 orientations) — a 10 Å site is 20× the compute
(898 vs 76 ms per forward). But the failure is total rather than marginal, and
below-chance type specificity is not a settings problem.

## Reproduce

```bash
python scripts/hotspot_validate.py --pocket data/1h1s.pdb --chains A \
    --ligand 4SP --ligand_chain A --site_radius 5.0 --spacing 1.5 --n_rotations 2
```
