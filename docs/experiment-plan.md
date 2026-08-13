# Research plan: using ATOMICA's interaction semantics for ligand design

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

## Phase 0 — Does ATOMICA discriminate binders at all? (go/no-go)

**Nothing downstream is worth building until this passes.**

For a fixed pocket, compute ATOMICA interface representations for (pocket, ligand)
pairs where the ligand is a known active, a property-matched decoy, or a random
drug-like molecule. If ATOMICA's representation carries binding-relevant
information, actives should separate from decoys.

- Data on hand: `scripts/eval/cdk2_test_data/` (30 binders / 30 decoys / 30 random).
- Metric: AUROC and early enrichment (EF@5%) for actives vs decoys.
- Ligand conformers: use the co-crystal pose where available; otherwise a docked or
  RDKit-generated conformer. **Report pose sensitivity** — if the signal only exists
  for crystal poses, it will not survive contact with generated molecules.

Feed it *correctly*: pocket as segment 0 with one block per residue and true
amino-acid block types, ligand as segment 1 with fragment-level blocks.

### Running it

```bash
conda activate ~/.conda/envs/atomica-interface
curl -o data/1h1s.pdb https://files.rcsb.org/download/1H1S.pdb   # CDK2/cyclin A + NU6102

# Primary gate: is the representation geometry-sensitive at all?
python scripts/phase0_pose_sensitivity.py \
    --pocket data/1h1s.pdb --chains A --ligand 4SP --ligand_chain A

# Secondary: does it separate binders from property-matched decoys?
python scripts/phase0_discriminate.py \
    --pocket data/1h1s.pdb --chains A --ref_ligand 4SP --ref_ligand_chain A \
    --actives scripts/eval/cdk2_test_data/binders.csv \
    --decoys scripts/eval/cdk2_test_data/decoys.csv \
    --random scripts/eval/cdk2_test_data/random_molecules.csv
```

Run the pose test first: it needs only the one structure, invents no poses, and if
it fails the binder/decoy result cannot be interpreted anyway.

**Gate.** n = 30/30 gives a wide confidence interval, so treat CDK2 as a smoke test
and confirm on a larger retrospective set (DUD-E or LIT-PCBA, several targets)
before committing to Phases 2–4.

- **AUROC ≈ 0.5** → ATOMICA's representation does not transfer to this task as used.
  Stop and reconsider the premise; do not build conditioning on top of it.
- **AUROC comfortably > 0.7** with enrichment → the signal is real, and Phases 1–4
  are justified.

An informative intermediate outcome: strong separation of actives from *random*
molecules but weak separation from *property-matched decoys* means ATOMICA is
recapitulating drug-likeness rather than binding — the same failure mode as the
adapter, diagnosed cheaply.

## Phase 1 — Fix the featurization (prerequisite)

Rewrite pocket construction in `scripts/process_expert_atomica.py`:

- **One block per residue**, with the correct amino-acid block type from `VOCAB`,
  instead of a single `UNK` block.
- **Two-segment inputs** wherever a ligand is present (pocket = 0, ligand = 1).
- Keep the global node, but verify it is not producing the degenerate embeddings
  behind the LayerNorm NaN cascade in the current adapter.

**Standalone check:** compute pocket embeddings for a diverse set of pockets under
old and new featurization, and measure how much pocket identity is recoverable —
e.g. can a linear probe predict pocket family, or do embeddings of the same pocket
family cluster? If the old featurization yields embeddings that barely distinguish
pockets, that is the direct, quantitative confirmation of the diagnosis above, and
it belongs in the writeup.

## Phase 2 — ATOMICA as a selector (fastest path to better molecules)

No retraining, no architecture. Generate with unmodified DiffSBDD, then rank
candidates by ATOMICA interface score against the target pocket and keep the top
fraction.

This is worth doing early because it directly answers *"does ATOMICA help us design
better molecules?"* without entangling that question with generative modelling. It
also composes with the existing `rl_loop/` selection stage, replacing or augmenting
the ADMET composite — which the earlier evolutionary experiment already identified
as the binding-blind component.

- Compare: random selection vs ADMET-composite vs Vina vs ATOMICA vs Vina+ATOMICA.
- Evaluate the *selected set*, using the pocket-aware metrics below.
- Cost: inference only.

If ATOMICA adds nothing over Vina here, that is a strong and cheap negative that
saves Phases 3–4.

## Phase 3 — Interaction hotspot fields (the novel contribution)

This is the idea with the most scientific upside, and it converts ATOMICA's
interface knowledge into something both spatially specific and interpretable.

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

## Phase 4 — ATOMICA as sampling guidance

Use ATOMICA as a differentiable interaction potential during denoising. At step *t*,
take the model's predicted clean ligand `x̂₀`, form the two-segment complex with the
pocket, and take the gradient of the ATOMICA interaction score with respect to
ligand coordinates to steer sampling.

- Uses ATOMICA as pretrained; requires no DiffSBDD retraining.
- **Key risk:** ATOMICA has never seen noisy or partially-formed ligands. Guide on
  `x̂₀` rather than `x_t`, and only over low-noise steps. Sweep guidance strength and
  the step window; report the diversity/affinity tradeoff rather than one setting.
- Falls back gracefully: at guidance strength 0 it is exactly baseline DiffSBDD.

## Phase 5 — Distillation (only if feed-forward conditioning is needed)

If Phases 3–4 show the interface representation is valuable but too slow for
in-loop use, train a pocket-only encoder to regress the ATOMICA interface
representation obtained from true complexes. That yields a pocket embedding which
*anticipates* interaction, and it is the principled version of what the original
adapter attempted — it solves the missing-second-segment problem by learning it
rather than ignoring it.

## Evaluation (applies to every phase)

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

## Sequencing and cost

| Phase | Work | Cost | Gate |
|---|---|---|---|
| **0** | Binder/decoy discrimination | hours, inference | **AUROC ≈ 0.5 → stop** |
| **1** | Residue blocks + two-segment featurization | days, CPU | pocket identity recoverable from embeddings |
| **2** | ATOMICA as selector | inference only | beats Vina/ADMET selection? |
| **3** | Hotspot field + co-crystal validation | weeks | hotspots recover true contacts? |
| **4** | Sampling guidance | weeks, GPU | affinity gain at acceptable diversity cost |
| **5** | Distillation | optional | only if 3–4 succeed but are too slow |

Phase 0 is a day's work and can invalidate the entire premise. Do it first.

## What to do with the existing work

Keep it and report it. The A/B ablation becomes the documented baseline showing that
naive single-segment embeddings yield a drug-likeness prior rather than pocket
conditioning — with the featurization diagnosis above as the explanation. That is a
more useful contribution than a marginal QED improvement, and it motivates
everything that follows.

The adapter code stays in history; it should not be extended.
