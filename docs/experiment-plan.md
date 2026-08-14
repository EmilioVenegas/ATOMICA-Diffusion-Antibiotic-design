# Research plan: using ATOMICA's interaction semantics for ligand design

**This is a handoff document.** It assumes no memory of the work. It records what
was measured, what those measurements ruled out, what the current plan is and why,
and the traps that have already cost time. Every claim points at the file it comes
from. Numbers in this document were re-verified against the repository on
2026-08-13; where a number elsewhere in the tree disagrees, that is noted.

## Status

| Phase | Question | Status |
|---|---|---|
| **0** | Is the interface representation geometry-sensitive? | **passed** — `results/phase0/README.md` |
| **1** | Featurize ATOMICA the way it was pretrained | **done** — `atomica_interface/`, back-ported to preprocessing on `fix/expert-preprocessing-featurization` |
| **2** | Does a pose scorer generalise to unseen systems? | **resolved: no** — `results/pose_scorer/README.md` |
| **3** | Interaction hotspot fields | **resolved: no** — `results/hotspot/README.md` |
| **3b** | Does correcting the pocket block vocabulary restore conditioning? | **resolved: no** — `results/featurization_probe/README.md` |
| **4** | **ATOMICA as a training-time critic** | **current work** — implemented and gated; `results/critic_gate/README.md`. Not yet trained |
| 5 | Conditioning on the partially-denoised ligand, low-noise steps only | after 4, and only if 4 shows signal |
| 6 | ATOMICA as a selector over generated molecules | blocked by 2 |
| 7 | Distillation to a pocket-only encoder | dead — see Phase 3b |

Three independent negatives and one positive. The positive is narrow and the
negatives are well controlled, which is what makes them worth publishing.

## The one result that should drive everything

Two measurements that look contradictory and are not:

| | Task | Result |
|---|---|---|
| Phase 0 | rank poses **within one pocket** | AUROC 1.000, clash- and composition-controlled |
| Phase 2 | rank poses **across unseen pockets** | 63.9% docking power vs smina 59.7%, McNemar p = 0.65 |

The representation discriminates interaction geometry sharply *inside* a system and
barely transfers *between* systems. Read as failure that is discouraging; read as a
constraint it is directive, because applications divide cleanly along exactly that
line:

- **Within-system** — comparing two states of *the same* complex: a predicted
  ligand against the true ligand, one pose against another, a probe placement
  against a neighbouring one. The Phase 0 regime. Viable.
- **Cross-system** — a universal scoring function, a transferable selector, a
  conditioning encoder trained over thousands of different pockets that must produce
  comparable absolute embeddings. The Phase 2 regime. Not supported by this evidence.

The original conditioning approach was cross-system — the harder regime — and it was
attempted through a featurization that destroyed the signal entirely. Everything
current is within-system.

**Governing principle: ATOMICA scores interfaces, so only ever use it on interfaces,
and only ever ask it to compare two interfaces of the same complex.**

## The central obstacle, stated honestly

ATOMICA needs two segments. At generation time the second segment — the ligand — is
what we are trying to produce. Every design here is a different resolution of that
tension:

| Approach | Resolution | Verdict |
|---|---|---|
| Selection | score the interface *after* generating | blocked by Phase 2 |
| Hotspot field | probe the pocket with surrogate fragments | **failed**, Phase 3 |
| Distillation | learn a pocket-only encoder that anticipates the interface | **dead**, Phase 3b |
| **Critic loss** | **the true ligand is a training label, not an input** | **current plan** |
| Guidance | the model's own `x̂₀` is segment 1, at low noise | secondary, after the critic |

## Resolved negative 1 — cross-system pose scoring (Phase 2)

`results/pose_scorer/README.md`, `scripts/train_pose_scorer.py`,
`scripts/featurize_block_level.py`.

100-target benchmark, 72 solvable targets, 1,674 poses. Out-of-fold under
`GroupKFold` **by target**, ridge alpha chosen by inner CV. Metric is CASF docking
power: per target, is the top-ranked pose within 2 Å of the crystal pose.

| Scorer | docking power | hits | mean per-target Spearman | vs smina |
|---|---|---|---|---|
| random (floor) | 15.7% | — | — | — |
| **smina (baseline)** | **59.7%** | **43/72** | — | — |
| graph (32-d) | 55.6% | 40/72 | +0.371 ± 0.084 | p = 0.70 |
| pocket_pool (96-d) | 41.7% | 30/72 | +0.344 ± 0.077 | **p = 0.019, worse** |
| all-block (288-d) | 63.9% | 46/72 | +0.400 ± 0.074 | p = 0.65 |

Every variant is far above the 15.7% floor on targets never seen in training, so the
head learns something real and transferable. It is simply not competitive with a
fast, free, established baseline. Nothing here justifies a tool that is slower than
smina and needs a GPU.

**An earlier 22-target result was retracted.** `pocket_pool` had measured 68.2% with
a paired Spearman gain of +0.127 (p ≈ 0.07); at 72 targets it measures 41.7% and the
gain is −0.027 (p = 0.52). The effect reversed, not merely shrank. Two causes worth
carrying: six feature sets were evaluated and the best reported, and 22 targets could
not resolve differences of two or three targets (every McNemar p ≥ 0.69). Recording a
selection bias does not remove it.

### The benchmark

`scripts/build_pose_benchmark.py` builds a CASF-style set from open RCSB data
(CASF-2016 itself sits behind registration): single-protein X-ray complexes, each
ligand redocked into its own pocket with smina, every pose labelled by symmetry-aware
in-place RMSD. Output in `data/pose_benchmark/`, indexed by `manifest.csv`.

Verified from `manifest.csv` as it stands on disk:

| | |
|---|---|
| targets / poses / distinct ligands | 100 / 1,691 / 76 |
| RMSD range | 0.17 – 34.96 Å |
| poses within 2 Å | 6.2% |
| targets with a near-native pose (solvable) | 72 / 100 |

`data/pose_benchmark/README.md` still describes the earlier 30-target / 520-pose
build and should be updated. The pose-scorer figures use 1,674 poses because target
`9WT9` (17 poses) is excluded — the PS_300 tokenizer raises on its haem ligand.

At 6.2% of poses within 2 Å this set is harder than CASF-2016, whose decoys are
curated across RMSD bins, and CASF has 285 targets. If cross-system scoring is ever
revisited, registering for CASF-2016 is the way to settle comparability.

### Why the training-free route was insufficient

ATOMICA was pretrained to predict the rigid-body noise applied to a segment, so
passing a **zero** noise target makes `translation_loss` equal the magnitude of the
predicted correction — a pose energy with no labels and no fitting
(`atomica_interface/energy.py`). Measured on the clash-controlled Phase 0 benchmark,
30 poses per class:

| Scorer | AUROC | Spearman vs RMSD | Needs fitting? |
|---|---|---|---|
| min contact distance (trivial baseline) | 0.727 | — | no |
| **training-free denoising energy** | **0.787** | +0.476 | **no** |
| linear probe on the representation | 1.000 | +0.927 | yes, per system |

0.787 sits marginally above a baseline that only measures how close the ligand is to
the protein. The probe's 1.000 is not a competing number — it was fitted on the
system it scores — it is an upper bound on what a head could extract. The rotation
head is worse than useless: displaced poses score *lower* rotational correction than
native ones, which is backwards.

## Resolved negative 2 — interaction hotspot fields (Phase 3)

`results/hotspot/README.md`, `results/hotspot/hotspot_1h1s_4SP.json`,
`scripts/hotspot_validate.py`, `atomica_interface/hotspot.py`.

Chemical probes on a grid through the pocket, scored with the training-free denoising
energy; validation protocol from Radoux et al. 2016 (median percentile rank of
crystal ligand atoms in the matching probe's field). CDK2 / NU6102 (1H1S chain A),
5 Å site, 1.5 Å grid, 1,683 accessible non-clashing points, 6 probes × 2 orientations,
28 ligand atoms scored.

| Measure | Value | Reference |
|---|---|---|
| median percentile, matched probe | **52.4** | Radoux: 97 (fragments), 72 (leads) |
| median percentile, **buriedness control** | **98.2** | the confound |
| median percentile, random placement | 52.2 | the floor |
| type specificity (matching probe wins) | **0.107** | chance = 0.167 |

52.4 against a random floor of 52.2 is no information at all, and type specificity is
*below* chance. The buriedness control at 98.2 is the useful part: protein neighbour
count alone beats our field and would beat Radoux's published fragment number, which
proves the harness can detect a field that predicts ligand positions. It detects one;
the signal is simply absent from the ATOMICA field. Had that control been skipped,
"ligand atoms land in the 98th percentile" would have looked like a success.

This is the **third** independent measurement placing the training-free denoising
energy at a trivial baseline. Treat the pretrained heads as unusable as a scoring
function and stop probing them.

Not ruled out but not worth pursuing here: a *trained* readout on probe
representations. That is supervised hotspot prediction from protein structure, which
is PharmacoNet (Chem Sci 2024, MIT, protein-only, generalises to unseen targets).
Reimplementing a published method with a weaker backbone is not a contribution.

## Resolved negative 3 — the featurization diagnosis was only half right (Phase 3b)

This is the most consequential correction in this document. **The diagnosis below was
right about what was broken and wrong about what fixing it would buy.**

### What was broken

`scripts/process_expert_atomica.py` originally fed ATOMICA:

```python
pocket_segment_ids   = np.array([0, 0])                  # one segment, no partner
pocket_B_types       = np.array([GLB, UNK])              # ALL pocket atoms in ONE block
pocket_block_lengths = np.array([1, n_pocket_atoms])     #   typed UNK
```

Two defects. **No interaction is present** — with a single segment there is no
partner, so none of ATOMICA's interaction semantics are engaged; this is the analogue
of asking a model trained on dialogue to embed one sentence with the speaker
stripped. And **block-level chemistry is erased** — ATOMICA's vocabulary is residue-
and fragment-level (`abrv2idx`), so every pocket is described at block level as one
unknown entity. What survives is per-atom element and local geometry, which the EGNN
already derives from coordinates: exactly the input from which one would expect a
generic drug-likeness shift with no pocket specificity.

That is what the A/B ablation measured (`results/ablation_summary.md`, 100 pockets,
9,328 vs 9,246 valid molecules): QED 0.424 → 0.483 (+13.9% relative), diversity
0.731 → 0.684 (−6.4% relative), no target-aware gain.

### What the probe measured

`scripts/featurization_probe.py`, `results/featurization_probe/README.md`. Same 99
pockets, same pocket-only setup, differing **only** in block vocabulary.

| | old `[GLB, UNK]` | new per-residue |
|---|---|---|
| blocks per pocket | 2.0 | 56.3 (17.9 distinct types) |
| mean pairwise cosine between *different* pockets, graph repr | **1.0000** | **0.9917** |
| mean pairwise cosine, unit repr | 1.0000 | 0.9999 |
| composition probe R², graph | 0.201 | **0.176** |
| composition probe R², unit | 0.165 | **0.108** |

**Confirmed:** cosine 1.0000 between different pockets means every pocket mapped to
the same direction in embedding space. The adapter was never conditioned on pocket
identity — not attenuated, absent. (The residual R² ≈ 0.20 comes from vector
*magnitude*, which still varied.)

**Refuted:** per-residue blocks with real amino-acid types barely move the
representation — cosine falls only to 0.9917, and recoverable amino-acid composition
gets *worse*, not better. A separate check on genuine two-segment records
(pocket + ligand) over six unrelated targets gives 0.9248 (commit `dd1c756`); that is
the only configuration that meaningfully de-degenerates, and it is a small sample.

**Conclusion: the missing interaction partner was the binding defect, not the block
vocabulary. Pocket-only ATOMICA encodings are degenerate regardless of block typing.**

Two things follow, and both save GPU time:

1. "Fix the pocket featurization and re-run the same conditioning" is ruled out. It
   would reproduce the original result.
2. **Phase 7 (distillation to a pocket-only encoder) is dead.** Its target — a
   pocket-only embedding that anticipates interaction — is the object just measured
   to be degenerate.

## The current direction — ATOMICA as a training-time critic

Not an inference-time conditioning encoder. An auxiliary loss during fine-tuning:

```
L = L_diffusion + lambda * d( ATOMICA(pocket, x0_hat), ATOMICA(pocket, x_true) )
```

where `x0_hat` is the denoiser's predicted clean ligand at step *t*, `x_true` is the
reference ligand, and ATOMICA is frozen.

Why this and not the alternatives, grounded in what has been measured:

- **It only asks ATOMICA to distinguish right from wrong within one complex.** That
  is the Phase 0 regime, measured at AUROC 1.000 with the composition and clash
  confounds controlled. It never asks for a transferable absolute score, which is the
  Phase 2 regime that failed (63.9% vs 59.7%, p = 0.65).
- **Both encodings are genuine two-segment interfaces**, which the featurization
  probe shows are the discriminative ones (0.9248 two-segment against 0.9917
  pocket-only).
- **Using the true ligand in a *loss* is not leakage — it is a label**, exactly as in
  any supervised objective. Leakage would be using it as a *conditioning input* at
  sampling time. This distinction is the crux and it is easy to get wrong; the
  existing preprocessing cache gets it wrong (see below).
- **ATOMICA is a frozen teacher, so inference needs no ATOMICA at all.** The
  generator internalises the signal, stays fast, and cannot be bottlenecked by
  ATOMICA failing to generalise at test time — which, per Phase 2, it does.

Practical notes for whoever implements it:

- Fine-tune from the existing checkpoints in `my_logs/` (`condB_static_t_r0`,
  `condC_timestep_adaptive_r0`, `condD_lora_r0`) or `checkpoints/`. Backbone frozen,
  adapter-only or LoRA. Note that despite the directory names, no LoRA is implemented
  anywhere in the tree (`MODIFICATIONS.md`); arm D is full backbone fine-tuning and
  arm C is architecturally identical to B.
- `d(·,·)` is a choice to make and report: cosine on the graph representation, or an
  L2 on pooled unit representations. The graph representation is the one the probe
  found least degenerate for two-segment records.
- Apply the term over low-*t* steps where `x0_hat` is chemically plausible, or ramp
  `lambda` with the noise level. ATOMICA has never seen a half-formed ligand.
- Sanity gate before training anything: check that
  `d(ATOMICA(pocket, x_true), ATOMICA(pocket, decoy))` is reliably larger than
  `d(ATOMICA(pocket, x_true), ATOMICA(pocket, near-native))` on the pose benchmark.
  If the distance is not ordered, the loss has no gradient worth following, and this
  costs minutes rather than GPU-days to check.

### The gate has been run and it passes — `results/critic_gate/README.md`

92 targets, 1,662 poses, per target. Every pose of a target is the same molecule
rigidly displaced, so composition is controlled by construction.

| metric | rho(all) | rho(<4 Å) | AUROC |
|---|---|---|---|
| **`graph_cosine` (pretrained)** | **+0.386** | **+0.558** | **0.926** |
| `graph_cosine` (permuted weights) | +0.149 | +0.061 | 0.697 |
| `contacts` (no-learning floor) | +0.253 | +0.355 | 0.837 |
| smina (reference) | +0.281 | +0.465 | 0.844 |

Pretraining is what carries it: in the low-RMSD regime the critic is weighted
toward, the permuted-weight control has essentially no signal. Three consequences
are already encoded in `DiffSBDD/configs/crossdock_fullatom_critic.yml`: use
`graph_cosine`, ramp `lambda` off at high noise, and raise `max_weight` well above
1.0.

**`pocket_pool` must not be used as the critic metric**, despite scoring the best
raw AUROC of any variant (0.949). It scores 0.923 with *random* weights, so almost
all of that is architecture and geometry rather than learned interaction
chemistry — see the gotcha below.

### Secondary — conditioning on `x̂₀` during sampling

Only if the critic shows signal. At step *t*, form the two-segment complex from the
pocket and the model's predicted clean ligand `x̂₀`, and condition on (or take
gradients through) its ATOMICA representation. Restrict to **low-noise steps only**,
for the reason above. Falls back gracefully: at strength 0 it is exactly baseline
DiffSBDD.

### Three architecture defects that must be fixed first, if conditioning is attempted

All in `DiffSBDD/equivariant_diffusion/dynamics.py`, all verified in the code as it
stands:

1. **No distance term.** `SE3EquivariantCrossAttention.forward(h_l, h_p, mask_l,
   mask_p, t)` receives no coordinates; scores are `q · kᵀ` on scalar features only.
   The module is geometry-blind and *structurally cannot express spatial
   specificity* — it can say "this pocket wants a donor", never "a donor here".
2. **It fires once, at the input.** The update is applied in `EGNNDynamics.forward`
   as `h_atoms = h_atoms + self.adapter_scale * h_update`, *before* features are
   concatenated and handed to the EGNN. The signal is then diluted through
   `n_layers: 6` (`DiffSBDD/configs/crossdock_fullatom_cond_B.yml`).
3. **Three overlapping magnitude controls.** `out_proj` is zero-initialised, a
   sigmoid `gate` on the timestep multiplies the output, and `adapter_scale` is a
   learned scalar initialised to 0.1. Three knobs on one quantity, one of which
   forces a cold start; they interact and none is individually interpretable.

## State of the data and infrastructure

Read this section before touching anything under `data/`.

### CrossDocked was never lost

An earlier revision of this plan, and some commit messages, claimed the CrossDocked
LMDB had been lost. **That was wrong.** Both files are on disk:

| File | Size | Entries |
|---|---:|---:|
| `data/crossdocked_pocket10_processed.lmdb` | 7.29 GB | 164,814 |
| `data/crossdocked_filtered.lmdb` | 6.23 GB | 132,469 |

### Split integrity

`data/crossdocked_split.pt` indexes the pocket10 LMDB in **cursor order**, not by
numeric key (LMDB orders keys lexicographically: `'0'`, `'1'`, `'10'`, `'100'`, …).

| | indices | distinct targets |
|---|---:|---:|
| `train` | 98,995 | 1,893 |
| `val` | 100 | 93 |
| `test` | 100 | 93 |

**Both defects below are now fixed** (`scripts/build_holdout_split.py`,
`data/holdout_target_split.pt`); the description is kept because the reasoning
still governs how the splits may be used.

Good news: the split is **target-aware** — zero targets shared between train and the
held-out set. Two problems:

1. **`val` and `test` are the identical index list.** There is no independent test
   set at all.
2. **The preprocessing val/test buckets are not target-disjoint from train.**
   `scripts/process_expert_atomica.py` *skips* everything in the official holdout
   ("held out upstream"), then fills `val` and `test` up to
   `--target_val_size` / `--target_test_size` (default 1,000 each) from complexes in
   *neither* official split. That leftover pool is 65,719 entries spanning 1,770
   targets, of which **1,327 targets (32,771 entries) also appear in train**. Any
   validation number computed on the current `val`/`test` directories is
   substantially within-target.

The LMDB holds 2,346 distinct targets; the official split covers 1,986 of them.

#### How they were fixed

`scripts/build_holdout_split.py` partitions the holdout's 93 targets into two
disjoint groups — by target, not by complex, because CrossDocked holds many docked
poses per target and a per-complex split puts the same protein on both sides. It
asserts the official split really is target-aware before relying on it.

The official holdout names only 100 complexes, which is thin. Those 93 targets carry
**8,330 LMDB entries** between them, **none in train's index list**, so every one is
as clean as the official 100. The script emits an expanded list alongside the strict
one — all entries on a held-out target, capped at 30 per target so a single
2,567-entry target cannot dominate — with the official complexes ranked first, so
the expanded list is always a superset and the strict holdout stays recoverable.

`process_expert_atomica.py --holdout_expand --splits val,test` then produced:

| | targets | complexes | shares targets with train |
|---|---:|---:|---:|
| `val` | 41 | 499 | **0** |
| `test` | 44 | 544 | **0** |
| *old `val` (kept as `val_contaminated_legacy/`)* | 17 | 1,000 | **9 of 17** |

The 9-of-17 overlap is measured against a 4,000-file sample of train, so the true
figure is higher. The old directories are retained rather than deleted.

`--legacy_fill` restores the old behaviour behind an explicit flag. The assertions
that val/test are disjoint from each other and from train now run on every build.

`affinity_info.pkl` has 184,087 entries keyed by ligand filename without extension,
each `{'rmsd', 'pk', 'vina'}`. Every one of the 164,814 LMDB complexes has a Vina
score. Two entries carry the sentinel `vina = 999.0`; filter `vina >= 900` before any
statistics.

### The expert filter was dropped, and why that matters

`expert_split.pt` (built by `scripts/create_expert_split.py`) keeps complexes with
Vina < −8.5: 77,638 of 164,814, i.e. **52.9% of the data discarded**. Recomputed
directly from the LMDB and `affinity_info.pkl`:

| Measure | All 164,813 scored complexes | Within the 10–40 heavy-atom band the preprocessor keeps |
|---|---:|---:|
| Pearson r(heavy atoms, Vina) | **−0.61** | −0.55 |
| mean heavy atoms, kept | 28.8 | 27.7 |
| mean heavy atoms, discarded | 19.2 | 20.1 |
| equal-size cohort by ligand efficiency (Vina / heavy atoms): overlap with expert cohort | 46% | 51% |
| that cohort's mean heavy atoms | 19.6 | 19.6 |

**It is largely a size filter.** Half the cohort changes if you control for ligand
size, and the size-controlled cohort averages ten fewer heavy atoms. It also makes
docking-based evaluation circular: selecting training data by Vina and then reporting
Vina improvements measures the filter.

The `--no_expert_filter` flag (commit `9b3279c`) keeps every complex in the standard
split. Its help text quotes r = −0.58 and 27.8 / 20.5 heavy atoms; the recomputation
above gives −0.61 and 28.8 / 19.2 over all complexes, −0.55 and 27.7 / 20.1 within
the size band. Same conclusion, slightly different arithmetic — worth correcting in
the source string when someone next touches that file.

### Preprocessing rebuilt (branch `fix/expert-preprocessing-featurization`, unmerged)

Two commits ahead of `main`: `dd1c756` (rebuild on the two-segment featurization) and
`9b3279c` (`--no_expert_filter`). What changed:

- Routes through `atomica_interface.scoring.load_encoder`, i.e.
  `PredictionModel._load_from_pretrained` — the supported path, with denoising heads
  disabled and `infer()` exposed. The old script called `.infer()` on a
  `DenoisePretrainModel`, which defines no such method, so **it could not run at
  all**.
- Pocket is one block per residue with its real amino-acid type (segment 0), ligand
  is PS_300 fragment blocks (segment 1). Verified structure: 67–101 blocks and 20–27
  distinct block types per complex over 2 segments, against the old 2 blocks in 1
  segment.
- Asserts the alignment invariant that keeps `pocket_atomica_embeddings` row-aligned
  with `pocket_coords` (the encoder prepends a synthetic global atom per segment; if
  that is not accounted for, every downstream row is off by one and silently wrong).
- The old element rejection list (`{H, C, N, O, F, P, S, Cl, Br, I}`) is preserved
  verbatim so dataset composition stays comparable to the run it replaces.

**The cache it writes contains ground-truth ligand information.** Each `.pt` holds
`lig_coords`, `lig_one_hot`, `pocket_coords`, `pocket_one_hot` and
`pocket_atomica_embeddings` `[n_pocket_atoms, 32]` — and that embedding is read off a
two-segment encoding whose segment 1 is the *reference* ligand. It is therefore
**not** usable as a sampling-time conditioning input; it is usable as a critic
target. This is the leakage distinction above, made concrete. The script's docstring
flags it; do not let it be mistaken for an oversight or quietly used as conditioning.

**That run has finished.** `data/processed_expert_atomica/` now holds:

| split | complexes | targets |
|---|---:|---:|
| `train` | 83,921 | ~1,893 |
| `val` | 499 | 41 |
| `test` | 544 | 44 |

83,921 against 85,206 candidates passing the size filter, the ~1,285 difference
being per-complex fragmentation and encoder failures. `val_contaminated_legacy/`
and `test_contaminated_legacy/` are the superseded directories, kept, not deleted.

### `size_distribution.npy` was regenerated, and the script had two bugs

`scripts/create_2d_histogram.py` hardcoded its axes to `(100, 500)` while the
preprocessing filter keeps 10–40 ligand atoms and 350–800 pocket atoms. That
**clamped 72.3% of complexes into the final pocket bin** and set
`max_num_nodes = len(histogram) - 1` to 99 for ligands that cannot exceed 40. The
file it was overwriting was `(41, 801)`, so the wrong shape would also have been a
silent regression. Defaults are now 41 / 801 and any clamping is counted and
reported. It also globbed train + val + test; since this histogram conditions
sampled ligand size, that let the held-out sets inform the generator's size prior.
It now defaults to `train` alone.

The old histogram was biased large exactly as predicted:

| | complexes | mean ligand heavy atoms | p5 / p50 / p95 |
|---|---:|---:|---|
| old (Vina-filtered) | 23,149 | 27.44 | 19 / 27 / 37 |
| **new (no expert filter)** | **83,921** | **22.61** | 10 / 23 / 34 |

A shift of −4.83 heavy atoms — the size confound the expert filter introduced.

## Immediate next steps

Steps 1–5 of the previous revision are **done**: the preprocessing run finished
(83,921 train complexes), val/test were rebuilt target-disjoint (499 / 544 over
41 / 44 targets), `size_distribution.npy` was regenerated on correct axes,
`fix/expert-preprocessing-featurization` was merged into `main`, and the critic
loss is implemented and gated. What remains:

1. **Finish `scripts/add_critic_targets.py` over all three splits.** It caches
   `ATOMICA(pocket, x_true)` and the coordinate-independent record structure that
   makes the encoder differentiable. Complexes without those fields still train,
   just without the critic term. ~0.16 s/complex on GPU.
2. **Sweep `critic_params.max_weight` on a short run.** At the config's 1.0 the
   critic is ~0.3% of the objective (measured: distance 0.0114, weighted 0.0070,
   diffusion nll 0.60) and will not move the model. Expect 1e1–1e2. The thing to
   watch is `critic_distance/train` actually falling, not the total loss.
3. **Fine-tune from `my_logs/` with the backbone frozen**, and run a
   `critic_params.enabled: False` arm from the same checkpoint as its control.
   Because ATOMICA is frozen and appears only in the loss, both arms sample
   identically and any difference is attributable to the objective.
4. **Evaluate with pocket-aware metrics** — `scripts/cross_dock_specificity.py`,
   see below. Needs `smina` (`conda install -c conda-forge smina`), which is not
   installed, and regenerated receptor PDBs (next item).
5. **Regenerate `data/receptor_pdbs/`** with `scripts/extract_pocket_pdbs.py`.
   The 100 pockets there are named `complex_000001…` against the *old,
   contaminated* test set. Their ids no longer correspond to the rebuilt `test/`,
   so cross-docking would silently score molecules against the wrong pockets.

### Environment (resolved)

`~/.conda/envs/atomica-interface` now runs both ATOMICA and DiffSBDD training.
`DiffSBDD/environment.yaml` and the root `pyproject.toml` describe different
things and the difference has caused confusion: the yaml is the DiffSBDD training
env (python 3.10.4 / torch 2.0.1 cu118 — the same constraints ATOMICA needs, so
there is **no** version conflict), while `pyproject.toml` (python 3.12 / torch
2.2.2) belongs to the Boltz tooling under `scripts/eval/`.

Installed into the conda env under a constraints file pinning `torch==2.0.1` and
`numpy==1.26.4`, so the CUDA build could not be swapped: `pytorch-lightning==2.3.3`
(newest line accepting torch 2.0), `torchmetrics==1.4.2`, `wandb`, `imageio`,
`seaborn`, `PyYAML`, and `setuptools<81` — lightning imports `pkg_resources`,
which setuptools 81+ drops, and the env had no setuptools at all. Verified after
install: torch 2.0.1 / CUDA 11.8 available, `torch_scatter` 2.1.2 working.

Note `DiffSBDD/environment.yaml` pins `pytorch-lightning=1.8.4`, which is stale —
the existing checkpoints were written by 2.5.5.

## Evaluation, for every generative phase

The A/B evaluation could not have detected pocket specificity: QED is
target-independent, so a QED gain is consistent with "conditioning narrowed the model
toward generically drug-like chemistry" and with "conditioning improved pocket fit".
Replace it:

1. **Pocket-aware primary metrics.** Matched docking across arms, and **cross-docking
   specificity** — dock each pocket's molecules against its own pocket and against *m*
   others. A molecule designed for its pocket should beat arbitrary pockets; a
   generically drug-like molecule scores ≈ 0. This is the metric that separates the
   two hypotheses.
2. **Analyse per pocket, not per molecule.** The ~9,300 valid molecules per arm are
   nested within 100 pockets; treating them as independent is pseudo-replication and
   inflates significance. Use paired tests across pockets and report the *fraction of
   pockets improved*, not just the mean shift.
3. **Docking-based evaluation is only non-circular now that the Vina filter is
   dropped.** With `--no_expert_filter` the training set is not selected on the metric
   being reported. Do not reintroduce the filter and then report Vina.
4. **Retain the ligand-only metrics** (QED / SA / Lipinski) as guardrails, never as
   evidence of pocket specificity.

For any scoring work the equivalent rule is the target-wise split: report per-target
figures over held-out targets, never pose-level accuracy pooled across a set where
the same protein appears in train and test.

## Gotchas

Accumulated the hard way. Each of these has already cost time.

**Analysis**

- `rdMolAlign.GetBestRMS` superimposes before measuring and therefore deletes exactly
  the rigid-body displacement that decides whether a pose is correct. Poses displaced
  by 1.9, 3.5 and 6.3 Å all measure **0.00**. Use `CalcRMS`. The first pose-benchmark
  build used `GetBestRMS` and reported 71% of poses within 2 Å; the same poses under
  `CalcRMS` give 6%. Both are symmetry-aware.
- Any hotspot or pocket-scoring result must report a **buriedness baseline**. Protein
  neighbour count alone reaches the 98.2nd percentile on the standard protocol and
  will make a method that measures nothing look excellent.
- **A control has to be shown to be a floor, not assumed to be one.** The critic gate
  first used `pocket_pool` as its negative control, because
  `scripts/featurize_block_level.py` states it is "identical for every pose of a
  target". The *input* pocket blocks are; their representations are not, being
  computed with message passing from the ligand. The intended floor scored 0.949
  AUROC against the real metric's 0.926. A permuted-weight run settled it —
  `pocket_pool` scores 0.923 with random weights — but only because that control was
  added. Any "representation X carries signal" claim needs a same-architecture,
  same-scale, no-learned-information comparison; permuting each weight tensor's
  entries preserves every marginal and destroys only what was learned.
- **Check that histogram and array axes cover the data they are built from.**
  `create_2d_histogram.py` clamped 72.3% of complexes into its final pocket bin for
  a year because its hardcoded `(100, 500)` did not match a filter admitting 800
  pocket atoms. Nothing errored.
- Report per-target/per-pocket, and beware evaluating several feature sets and
  reporting the best — that is how the retracted 22-target result happened.

**ATOMICA**

- The PS_300 tokenizer has no valence entry for iron and raises on haem-like ligands
  (`KeyError: 'Fe'`). Target `9WT9` is skipped in both the pose-scorer and
  featurization-probe runs for this reason. Catch it per complex; do not let it kill
  a long run.
- `DenoisePretrainModel` has **no** `infer()` method and its `forward` asserts on
  denoising targets it is not given. The supported route to pretrained
  representations is `PredictionModel._load_from_pretrained`, wrapped as
  `atomica_interface.scoring.load_encoder`.
- `blocks_to_data` prepends a synthetic global atom to each segment. Any code mapping
  encoder output rows back to input atoms must account for it or every row is
  misaligned — silently, with plausible-looking values.
- Use the fragmentation scheme named in `ATOMICA/pretrain/pretrain_model_config.json`
  (**PS_300**). Any other scheme silently degrades the block vocabulary against the
  checkpoint.

**Environment** — `~/.conda/envs/atomica-interface`, spec in `environment-atomica.yml`

- `pip install` of anything that depends on torch silently replaces the CUDA build
  with a CPU wheel and breaks `torch_scatter`. Use `--no-deps`.
- Anaconda's default channels (`pkgs/main`, `pkgs/r`) require accepting a Terms of
  Service and leak in from global conda config. The spec sets `nodefaults`; on the
  command line use `--override-channels`.
- MKL ≥ 2024.1 breaks torch 2.0.1 with `undefined symbol: iJIT_NotifyEvent`. Pin
  below it.
- e3nn must be **0.5.1** for torch 2.0.1. The installed environment has 0.5.1 but
  `environment-atomica.yml` currently lists `e3nn` unpinned in its pip section — pin
  it before anyone rebuilds the environment.
- Working versions: python 3.10.4, torch 2.0.1 / CUDA 11.8, e3nn 0.5.1,
  pytorch-scatter 2.1.2, pytorch-cluster 1.6.3, rdkit 2022.03.2, numpy 1.26.4.

**Shell**

- Piping a command into `grep` or `tail` masks its exit code — the pipeline reports
  the *last* command's status. This has already hidden two silent failures. Redirect
  to a file and inspect it, or check `PIPESTATUS`.

## What to do with the existing work

Keep it and report it. The value of this project now rests substantially on a
**well-controlled negative result about foundation-model transfer**, and that is a
more useful contribution than a marginal QED improvement:

- A pretrained interaction foundation model discriminates geometry within a complex
  essentially perfectly (AUROC 1.000, with the two confounds that independently faked
  the result identified and driven to chance).
- The same model, with a head trained across systems, does not beat a 2010 scoring
  function on held-out targets (63.9% vs 59.7%, p = 0.65).
- Its training-free energy sits at the random floor as a hotspot field (52.4 vs 52.2),
  while a trivial buriedness baseline reaches 98.2.
- Its pocket-only encodings are degenerate (cosine 1.0000 between different pockets),
  and correcting the block vocabulary does not repair it (0.9917) — only supplying the
  second segment does (0.9248).
- The A/B ablation is the documented consequence: naively-extracted single-segment
  embeddings yield a generic drug-likeness prior (QED +13.9%) with no pocket
  specificity, at a diversity cost (−6.4%).

Each negative is paired with a control that proves the measurement could have
detected the positive. That is what makes them evidence rather than absence of
evidence.

The adapter code stays in history. It should not be extended without fixing the three
architecture defects first, and the current plan does not require it at all.
