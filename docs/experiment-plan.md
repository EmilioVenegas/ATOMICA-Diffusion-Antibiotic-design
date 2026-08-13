# Research plan: using ATOMICA's interaction semantics for ligand design

## Where this stands

| Phase | Question | Status |
|---|---|---|
| **0** | Is the interface representation geometry-sensitive? | **passed** — `results/phase0/README.md` |
| **1** | Featurize ATOMICA the way it was pretrained | **done for the scoring path** — `atomica_interface/` |
| **2** | Does a pose scorer built on it generalise to unseen systems? | **in progress** — benchmark built, head not yet trained |
| 3 | ATOMICA as a selector over generated molecules | not started |
| 4 | Interaction hotspot fields | not started |
| 5 | ATOMICA as sampling guidance | not started |
| 6 | Distillation to a pocket-only encoder | conditional on 4–5 |

Phase 2 is new. It was not in the original plan because the original plan assumed
the pretrained model could be used as a scorer directly; measurement showed it
cannot (see below), and everything downstream depends on a scorer that works on a
pocket it has never seen. It is the near-term line of work.

## Diagnosis: why the adapter approach could not have worked

The cross-attention adapter is not the problem. The **featurization** is.

ATOMICA is pretrained on *intermolecular interaction interfaces*. Its pretraining
dataset is built around two interacting segments — `ATOMICA/data/dataset_pretrain.py`
splits on `segment_ids == 0` / `== 1` and masks blocks within each separately. The
representation it learned is of an **interaction between two entities**, organised
over a hierarchy of chemically-typed **blocks** (residues, functional groups).

`scripts/process_expert_atomica.py` feeds it:

```python
pocket_segment_ids = np.array([0, 0])                    # one segment, no partner
pocket_B_types     = np.array([GLB, UNK])                # ALL pocket atoms in ONE
pocket_block_lengths = np.array([1, n_pocket_atoms])     #   block, typed UNK
```

Two consequences, both fatal to the intended claim:

1. **No interaction is present.** With a single segment there is no partner to
   interact with, so none of ATOMICA's interaction semantics are engaged. This is
   the analogue of asking a model trained on dialogue to embed one sentence with
   the speaker stripped.
2. **Block-level chemistry is erased.** ATOMICA's vocabulary is residue- and
   fragment-level (`abrv2idx`, amino-acid symbols). Collapsing the pocket into one
   `UNK` block means every pocket — regardless of composition — is described at
   block level as a single unknown entity.

What survives is per-atom element and local geometric context, which the EGNN
already derives from coordinates. That is precisely the input from which one would
expect a **generic drug-likeness shift with no pocket specificity** — which is what
the A/B ablation measured (QED +13.9%, diversity −6.4%, no target-aware gain).

The A/B result is not wasted: it is a clean documented negative for
"naively-extracted foundation-model embeddings as a conditioning signal." It should
be reported as such.

**Governing principle for everything below: ATOMICA scores interfaces, so use it on
interfaces.**

## The central obstacle, stated honestly

ATOMICA needs two segments. At generation time the second segment — the ligand — is
what we are trying to produce. Every design below is a different resolution of that
tension, ordered by how much has to work for it to pay off:

| Approach | Resolution of the obstacle |
|---|---|
| Selection | Score the interface *after* generating (ligand exists) |
| Hotspot field | Probe the pocket with *surrogate* ligands (small fragments) |
| Guidance | Use the partially-denoised ligand as segment 1 during sampling |
| Distillation | Train a pocket-only encoder to predict the interface representation |

## Phase 0 — Is the representation geometry-sensitive? — PASSED

The go/no-go gate. Full writeup in `results/phase0/README.md`; the result in short:
native 4SP (NU6102) in CDK2 (1H1S, chain A) against rigid perturbations of it, 30
poses per class, out-of-fold linear probe, via `scripts/phase0_pose_sensitivity.py`.

| Test | displaced RMSD | size conf. | steric conf. | AUROC | perm p | Spearman |
|---|---|---|---|---|---|---|
| uncontrolled | 4.5–8.3 Å | 0.500 | **1.000** | 1.000 | 0.0005 | +0.855 |
| uncontrolled, hard | 1.1–2.3 Å | 0.500 | 0.983 | 0.999 | 0.0005 | +0.919 |
| **clash-controlled** | 1.8–3.5 Å | 0.500 | **0.696** | **1.000** | 0.0005 | **+0.927** |

Only the last row is evidence, and getting to it is the substance of the phase. Two
confounds each independently faked the result:

- **Complex composition.** Re-trimming the pocket to each pose's contacts lets a
  displaced ligand touch a different number of residues, so the classes separate on
  complex size alone — worth AUROC ~0.70 here. Holding the binding site fixed across
  every pose drives that shortcut to exactly 0.500.
- **Steric clash.** Rigid perturbation pushes the ligand into the protein, so
  minimum ligand–protein distance alone reaches AUROC 1.000, matching ATOMICA and
  making the test meaningless. `--clash_free` rejects poses that clash worse than the
  crystal pose, keeping both classes physically plausible and dropping the shortcut
  to 0.696.

Established: the representation encodes interaction geometry, is finely graded
rather than merely detecting gross displacement, and is not reading complex size or
steric overlap. Not established: anything beyond one pocket and one ligand, and
nothing at all about ranking *different* molecules.

**Still unrun:** the binder/decoy arm (`scripts/phase0_discriminate.py`). Phase 0
implies it needs *docked* poses to be interpretable — a representation this
pose-sensitive will swamp any binding signal with pose noise if fed arbitrarily
oriented conformers. The benchmark built in Phase 2 supplies exactly the docked
poses that test needs, which is a further reason to do Phase 2 first.

## Phase 1 — Fix the featurization — DONE for the scoring path

Implemented as the first-party package `atomica_interface/`, which builds inputs the
way ATOMICA's own dataset pipeline does rather than reimplementing them:

- `featurize.pocket_blocks_from_pdb` — one block per residue with real amino-acid
  block types, via ATOMICA's `pdb_to_list_blocks`.
- `featurize.ligand_blocks_from_mol` — ligand fragmented with **PS_300**, the scheme
  named in `ATOMICA/pretrain/pretrain_model_config.json`; using any other scheme
  silently degrades the block vocabulary against the checkpoint.
- `featurize.ligand_from_pdb_het` — assigns bond orders from the component SMILES
  template, since PDB records carry none and fragmentation without them is wrong.
- `featurize.interface_data` — genuine two-segment records through
  `blocks_to_data(pocket, ligand)`, pocket = segment 0, ligand = segment 1.
- `scoring.load_encoder` — the supported route to pretrained representations is
  `PredictionModel._load_from_pretrained`, which rebuilds the encoder with the
  denoising heads off and exposes `infer()`. `DenoisePretrainModel` has no `infer`.

Sanity check on a real complex: 2 segments, 55 pocket residue blocks + 8 ligand
fragment blocks, 23 distinct block types — against the `[GLB, UNK]` 2-block
single-segment input the original pipeline produced.

**Not back-ported.** `scripts/process_expert_atomica.py` is still the old, broken
path: it builds single-segment `UNK` inputs and calls `atomica_model.infer()` on a
`DenoisePretrainModel`, which has no such method. Anything that regenerates the
DiffSBDD conditioning cache must be rewritten onto `atomica_interface` first. Until
then the Phase 3–6 conditioning work cannot be run at all.

## Phase 2 — A pose scorer that generalises across systems (current work)

### Why this exists: the training-free route is insufficient

ATOMICA was pretrained to predict the rigid-body noise applied to a segment, so
passing a **zero** noise target makes `translation_loss` equal the magnitude of the
predicted correction — a pose energy with no labels and no fitting
(`atomica_interface/energy.py`). If that worked, a tool would exist today with no
training at all. Measured on the clash-controlled Phase 0 benchmark, 30 poses per
class:

| Scorer | AUROC | Spearman vs RMSD | Needs fitting? |
|---|---|---|---|
| min contact distance (trivial baseline) | 0.727 | — | no |
| **training-free denoising energy** | **0.787** | +0.476 | **no** |
| linear probe on the representation | 1.000 | +0.927 | yes, per system |

The free lunch is not there. 0.787 sits marginally above a baseline that only
measures how close the ligand is to the protein. The probe's 1.000 is not a
competing number — it was fitted on the system it scores — it is an upper bound on
what a head could extract. (The rotation head is worse than useless here: displaced
poses score *lower* rotational correction than native ones, which is backwards.)

**Conclusion to carry forward: the information is in the representation, but the
pretrained heads do not surface it. A usable scorer needs a head trained across
systems, not a wrapper around the pretrained model.** That makes cross-system
generalisation the deciding experiment rather than an optional check.

### The benchmark

CASF-2016 is the standard docking-power benchmark but sits behind registration, so
`scripts/build_pose_benchmark.py` constructs an equivalent from open RCSB data:
single-protein X-ray complexes, each ligand redocked into its own pocket with smina,
every pose labelled by symmetry-aware in-place RMSD to the crystal pose. Output in
`data/pose_benchmark/`.

| | |
|---|---|
| targets / poses / distinct ligands | 30 / 520 / 25 |
| RMSD range | 0.27 – 13.68 Å |
| poses within 2 Å | 6% |
| targets with a near-native pose (solvable) | 22 / 30 |
| targets with both correct and incorrect poses (usable) | 22 / 30 |

Two properties matter. Decoys come from a docking engine, so they are physically
plausible and clash-free — unlike the Phase 0 perturbations, which were separable on
steric overlap alone. And RMSD is computed with `rdMolAlign.CalcRMS`, never
`GetBestRMS`: superimposing before measuring deletes exactly the rigid-body
displacement that decides whether a pose is correct, and under it poses displaced by
1.9, 3.5 and 6.3 Å all measure 0.00. The 8 targets where docking never recovered a
sub-2 Å pose are unsolvable by any scorer and are excluded from docking-power
figures, as CASF does.

### The next experiment, in order

1. **Featurize** all 520 (pocket, pose) pairs with `atomica_interface` and cache the
   representations.
2. **Train a small head** on those representations with **target-wise splits**
   (GroupKFold grouped by target), so no test protein is ever seen in training. This
   is the cross-system generalisation test, and it is the gate on everything
   downstream — a head that only works within a target is the per-system probe again.
3. **Evaluate docking power**: per target, is the top-ranked pose within 2 Å.
   Excluding the 8 unsolvable targets, i.e. over the 22 usable ones.
4. **Baseline to beat:** the smina score already stored per pose in
   `data/pose_benchmark/manifest.csv` (`smina_score` column). No extra runs needed.

### Decision gate

- **Beats or matches smina, with a different error profile** → a real tool exists.
  Extract `atomica_interface` into its own repository, packaged properly: ATOMICA as
  a git submodule rather than a vendored copy, weights as a GitHub Release asset.
- **Fails** → do not conclude anything about ATOMICA yet. Check whether the
  benchmark's difficulty is the cause first: at 6% of poses within 2 Å this set is
  harder than CASF-2016, whose decoys are curated to be balanced across RMSD bins, so
  a scorer can lose here on the pose distribution rather than on the representation.
  Registering for CASF-2016 would give standard, comparable numbers and settle it.

### How this serves the original research question

The pose scorer is not a detour from *"does ATOMICA help the diffusion model?"* —
it is the component both of the downstream phases are missing:

- **Phase 3 (selector)** replaces the ADMET composite, which the earlier
  evolutionary experiment already identified as the binding-blind component of the
  selection criterion. Selection needs a score over ligand–pocket interfaces; there
  isn't one yet.
- **Phase 5 (guidance)** needs a differentiable interaction score over the
  pocket and a candidate pose. That is the same object.

Two caveats, stated plainly so they are not quietly dropped later:

- **Recognition is necessary but not sufficient for generation.** Scoring poses well
  does not show that a generator can be conditioned to produce good geometry. A
  scorer that passes Phase 2 makes Phases 3–5 worth running; it does not predict
  their outcome.
- **Docking power is not screening or ranking power.** CASF separates them: docking
  power is picking the right pose *of one molecule* (what Phase 5 guidance needs);
  screening and ranking power are ordering *different molecules* (what Phase 3
  selection actually needs). Only docking power is addressed so far. Phase 0 already
  flagged this — pose sensitivity says nothing about ranking different molecules.

## Phase 3 — ATOMICA as a selector

No retraining, no architecture. Generate with unmodified DiffSBDD, then rank
candidates by ATOMICA interface score against the target pocket and keep the top
fraction.

This directly answers *"does ATOMICA help us design better molecules?"* without
entangling that question with generative modelling. It composes with the existing
`rl_loop/` selection stage, replacing or augmenting the ADMET composite.

- Compare: random selection vs ADMET-composite vs Vina vs ATOMICA vs Vina+ATOMICA.
- Evaluate the *selected set*, using the pocket-aware metrics below.
- Cost: inference only.

**Blocked on Phase 2**, and on the *ranking* side of it: selection ranks different
molecules, which the pose benchmark does not measure. Expect to need a ranking-power
evaluation (affinity-labelled complexes, e.g. PDBbind) before this phase means
anything. If ATOMICA adds nothing over Vina here, that is a strong and cheap
negative that saves Phases 4–5.

## Phase 4 — Interaction hotspot fields (the novel contribution)

The idea with the most scientific upside, and it converts ATOMICA's interface
knowledge into something both spatially specific and interpretable.

**Construction.** Place small chemical probes — methane (hydrophobic), water or
methanol (H-bond donor/acceptor), benzene (aromatic), ammonium/acetate (charged) —
at grid points throughout the pocket. For each (probe type, position), evaluate the
ATOMICA interface representation with pocket as segment 0 and probe as segment 1.
The result is a field over space × probe type: *this subpocket favours a donor here,
a hydrophobe there.*

This is a learned analogue of GRID / FTMap / SILCS hotspot mapping, derived from a
foundation model rather than a force field, and it uses ATOMICA exactly as
pretrained — two segments, real interaction, chemically-typed blocks.

**Validation, and this is the part that makes it a paper.** It is falsifiable
without any generation at all: on held-out co-crystal complexes, compute the hotspot
field from the *pocket alone*, then ask whether the true ligand's functional groups
sit where the matching probe scores highest.

- Metric: enrichment of true ligand atoms in the top-k hotspot voxels, by
  functional-group type, against a random-placement null.
- Data: abundant (PDBbind, CrossDocked co-crystals). No docking required.
- Baselines: distance-to-pocket-surface, buriedness, and a classical hotspot method.

A hotspot map that recovers real ligand contacts is a useful medicinal-chemistry
tool on its own, independent of the diffusion model.

**Then use it for conditioning.** The field is spatially resolved, so conditioning
on it is structurally capable of the spatial specificity the previous adapter could
not express — the conditioning signal is now indexed by *position*, not just pocket
identity. Concretely: voxelise the field, or attach per-probe-type scores to pocket
atoms, and feed it to the denoiser with a distance-aware attention bias.

Note that Phase 2's finding applies here too: if the pretrained heads do not surface
a usable pose signal, a probe field read straight off the pretrained model may be
just as flat, and this phase may need the trained head as its scoring function.

## Phase 5 — ATOMICA as sampling guidance

Use ATOMICA as a differentiable interaction potential during denoising. At step *t*,
take the model's predicted clean ligand `x̂₀`, form the two-segment complex with the
pocket, and take the gradient of the ATOMICA interaction score with respect to
ligand coordinates to steer sampling.

- Requires no DiffSBDD retraining, but does require the Phase 2 head as the scoring
  function — the training-free energy is too weak to guide on.
- **Key risk:** ATOMICA has never seen noisy or partially-formed ligands. Guide on
  `x̂₀` rather than `x_t`, and only over low-noise steps. Sweep guidance strength and
  the step window; report the diversity/affinity tradeoff rather than one setting.
- Falls back gracefully: at guidance strength 0 it is exactly baseline DiffSBDD.

This is the phase docking power is directly relevant to: steering a pose toward a
near-native geometry is the same task as recognising one.

## Phase 6 — Distillation (only if feed-forward conditioning is needed)

If Phases 4–5 show the interface representation is valuable but too slow for in-loop
use, train a pocket-only encoder to regress the ATOMICA interface representation
obtained from true complexes. That yields a pocket embedding which *anticipates*
interaction, and it is the principled version of what the original adapter
attempted — it solves the missing-second-segment problem by learning it rather than
ignoring it.

## Evaluation (applies to every generative phase)

The previous evaluation could not have detected pocket specificity. Replace it:

1. **Pocket-aware primary metrics.** Matched docking across arms, and **cross-docking
   specificity** — dock each pocket's molecules against its own pocket and against
   *m* others; a molecule designed for its pocket should beat arbitrary pockets. A
   generically drug-like molecule scores ≈ 0. This is the metric that separates the
   two hypotheses.
2. **Correct statistics.** Analyse per pocket, not per molecule. The ~9,700
   molecules per arm are nested within 100 pockets; treating them as independent is
   pseudo-replication and inflates significance. Use paired tests across the 100
   pockets and report the fraction of pockets improved, not just the mean shift.
3. **Retain the ligand-only metrics** (QED/SA/Lipinski) as guardrails, never as
   evidence of pocket specificity.

For Phase 2 the equivalent rule is the target-wise split: report per-target docking
power over held-out targets, never pose-level accuracy pooled across a set where the
same protein appears in train and test.

## Sequencing and cost

| Phase | Work | Cost | Gate |
|---|---|---|---|
| **0** | Pose-sensitivity gate | hours, inference | **passed** (AUROC 1.000 clash-controlled) |
| **1** | Residue blocks + two-segment featurization | days, CPU | **done** for scoring; `process_expert_atomica.py` still to port |
| **2** | Cross-system pose scorer on the 520-pose benchmark | days, GPU | **beats smina docking power on held-out targets?** |
| **3** | ATOMICA as selector | inference only | beats Vina/ADMET selection? (needs ranking power, not just docking power) |
| **4** | Hotspot field + co-crystal validation | weeks | hotspots recover true contacts? |
| **5** | Sampling guidance | weeks, GPU | affinity gain at acceptable diversity cost |
| **6** | Distillation | optional | only if 4–5 succeed but are too slow |

Phase 2 is the current gate. If it fails and CASF-2016 confirms the failure is not
the benchmark's difficulty, then the representation does not support a usable
interface score and Phases 3–6 are not worth their cost.

## What to do with the existing work

Keep it and report it. The A/B ablation becomes the documented baseline showing that
naive single-segment embeddings yield a drug-likeness prior rather than pocket
conditioning — with the featurization diagnosis above as the explanation. That is a
more useful contribution than a marginal QED improvement, and it motivates
everything that follows.

The adapter code stays in history; it should not be extended.
