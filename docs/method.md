# Method

## Motivation

Structure-based generative models condition on a pocket's raw geometry: atom
coordinates and element types. That representation is complete but uninformative —
it says where the atoms are, not what kind of chemistry the pocket rewards. A model
must rediscover interaction preferences (hydrophobic burial, hydrogen-bond geometry,
π-stacking) from coordinates alone, for every pocket, from a training set of tens of
thousands of complexes.

ATOMICA is pretrained on ~2M molecular interaction interfaces, so its per-atom
embeddings already encode that vocabulary. The question this project asks is narrow
and testable:

> Does replacing part of the geometric conditioning signal with a pretrained
> *interaction* representation improve the molecules a diffusion model generates?

The design is deliberately an **adapter**, not a new architecture: the pretrained
DiffSBDD backbone is frozen and the ATOMICA encoder is frozen, so any measured
difference isolates the conditioning signal itself.

## Pipeline

```
CrossDocked2020 (164k complexes)
    │  process_expert_atomica.py — filter + precompute embeddings
    │    druggability (15–80 heavy atoms), steric clashes,
    │    pocket proximity (<6 Å), Vina < −8.5
    ▼
~40k complexes as .pt  (coords, one-hots, 32-d ATOMICA pocket embeddings)
    │  LigandPocketDatasetPT
    ▼
LigandPocketDDPM  (PyTorch Lightning)
  EGNNDynamics + SE3EquivariantCrossAttention  →  ConditionalDDPM
    ▼
generate_ligands.py / optimize.py  →  SDF
    │  rl_loop/RL_loop.py (optional)
    ▼
ADMET-AI scoring → REOS filter → Tanimoto diversity → top-k
```

## The conditioning mechanism

Full detail, including masking and initialisation, is in
[MODIFICATIONS.md](../MODIFICATIONS.md#the-conditioning-mechanism). In brief, one
cross-attention block per denoising step:

| | |
|---|---|
| **Query** | ligand scalar features `h_l` ⊕ 16-d timestep embedding |
| **Key / Value** | frozen ATOMICA per-atom pocket embeddings `h_p` |
| **Output** | delta on ligand *invariant scalar* features only |

Generating ligand atoms attend to pocket atoms by learned chemical compatibility
rather than by distance alone.

### Why this preserves SE(3) equivariance

The diffusion model is equivariant: rotate or translate the pocket, and generated
coordinates transform identically. A conditioning module can easily destroy that.

This one cannot, because of a type restriction. ATOMICA embeddings are **invariant
scalars** — they do not change under rotation. The module reads invariant features
and writes invariant features; coordinates never enter the attention computation and
never receive its output. Equivariance is preserved structurally rather than by
penalty or augmentation, so it holds exactly and needs no test to defend it.

The alternative — letting attention write coordinate updates — would require the
values to be equivariant vectors and the attention weights to be invariant, a
strictly more complex design that this project did not need.

### Variable-sized pockets

Pockets differ in atom count, so a batch cannot be a dense tensor without padding,
and padded positions must not receive attention mass. Rather than pad, the batch is
kept ragged and a block-diagonal mask built from the batch-index vectors
(`mask_l[:, None] == mask_p[None, :]`) confines attention within each complex.
Masked positions are set to −1e4 before the softmax, so they contribute no gradient.
This avoids padding artifacts entirely at the cost of an N_lig × N_pocket score
matrix per batch.

## Training

Two parameter groups: the backbone at `lr`, the adapter at `adapter_lr` (10× higher,
since it trains from a zero init while the backbone is already converged). In arm B
the backbone is frozen outright (`freeze_backbone: True`).

Full-atom pockets exceed 16 GB VRAM at useful batch sizes, so training uses gradient
checkpointing, `bf16-mixed` precision, and gradient accumulation. Prototyping ran on
a single local GPU; the reported runs were retrained on an HPC cluster once the
architecture stabilised.

## Evaluation

Distributional metrics (validity, QED, SA, Lipinski, diversity, uniqueness, novelty)
via `scripts/evaluate.py`, compared across arms with `scripts/compare_conditions.py`.

A separate "three judges" pipeline under `scripts/eval/deep_dive_eval/` scores
candidate molecules with a PAINS filter, AutoDock Vina, and Boltz-2 affinity
prediction. It was validated on CDK2 but has **not** been run as a matched A/B
comparison — see [results.md](results.md) for why that gap is the one that matters.
