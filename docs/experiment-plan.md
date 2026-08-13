# Experiment plan

## The problem with the current ablation

The A→B→C→D progression varies **training capacity**, not **information**:

| Arm | What it changes |
|---|---|
| A | frozen pretrained backbone, no conditioning |
| B | + adapter (backbone frozen) |
| C | *nothing* — identical to B (see [MODIFICATIONS.md](../MODIFICATIONS.md)) |
| D | + unfrozen backbone |

Every arm receives the same conditioning signal (true ATOMICA embeddings) or
none at all. So the ladder can tell us *that* B beats A, but not *why*. The
result — QED +13.9%, diversity −6.4% — is equally consistent with two very
different stories:

1. **The interaction hypothesis.** The adapter learned pocket-specific chemistry
   from ATOMICA's interaction representation.
2. **The prior hypothesis.** The adapter learned a generic drug-likeness prior
   that raises QED for *any* pocket, and the diversity drop is the mode collapse
   that implies.

No experiment run so far separates these. Since every reported metric (QED, SA,
Lipinski) is computed from the ligand alone and never sees the pocket, the
current evidence cannot distinguish them even in principle.

**The fix: hold capacity fixed and vary the conditioning signal.**

## Proposed arms

Same architecture, same parameter count, same training budget. Only the tensor
in `pocket_atomica_embeddings` changes.

| Arm | Conditioning signal | Question it answers |
|---|---|---|
| A | none (coordinates only) | baseline — *done* |
| B | true ATOMICA embeddings | does conditioning help? — *done* |
| **S** | **true embeddings from a *different* pocket** | is the gain pocket-specific, or generic? |
| **R** | **random vectors, matched mean/covariance** | is it information, or just adapter capacity? |
| **P** | **cheap pocket features** (residue-type composition + physicochemical descriptors, projected to 32-d) | does ATOMICA beat a trivial featurization? |
| D | true embeddings, backbone unfrozen | capacity vs. information — *trained, not evaluated* |

**S is the decisive one and it is nearly free** — no retraining. Take the trained
arm-B checkpoint and permute the `pocket_atomica_embeddings` field across test
complexes at generation time. `scripts/run_baseline.py` already reads that field
directly from the `.pt` records (`build_pocket_dict`, line ~82), so this is a
`--shuffle_embeddings` flag, not new machinery.

The logic is clean:

- **S ≈ B** → the adapter ignores which pocket it was given. The benefit is a
  drug-likeness prior. The interaction hypothesis is dead, and you have a crisp,
  publishable negative result with a diagnosis rather than a shrug.
- **S ≈ A**, or S degrades on pocket-aware metrics → conditioning is genuinely
  pocket-specific. Proceed to R and P to establish *what* ATOMICA contributes.

R and P cost one training run each and are only worth spending after S resolves.

## Metrics must become pocket-aware

QED, SA and Lipinski cannot answer a pocket-specificity question at any sample
size. Two additions:

### 1. Matched docking (necessary)

Vina/smina for both arms over the same 100 pockets, identical protocol. This is
the already-identified gap. It gives a target-aware measure, but on its own it
still conflates "good binder" with "good binder *for this pocket*."

### 2. Cross-docking specificity (the informative one)

For each pocket *i*, dock its generated molecules against pocket *i* (self) and
against *m* other pockets *j≠i* (cross):

```
specificity_i = mean(affinity_cross) − mean(affinity_self)
```

A molecule designed for its pocket should bind that pocket better than it binds
arbitrary others. A generically drug-like molecule docks about equally well
everywhere and scores ≈ 0.

This metric is immune to the drug-likeness confound, which is exactly why it is
worth the compute. **The claim to test is that specificity is larger for B than
for A** — not merely that B's absolute affinity is better.

Suggested scale: 50 pockets × top-10 molecules × (1 self + 5 cross) ≈ 3,000 docking
runs. Embarrassingly parallel; hours on the cluster.

## Fix the statistics first (no GPU required)

The current comparison pools ~9,700 molecules per arm and compares means. Those
molecules are **not independent** — they are nested within 100 pockets, ~97 per
pocket. Treating them as independent samples is pseudo-replication and inflates
significance; the effective sample size is closer to 100 than 9,700.

The correct unit of analysis is the pocket, and since both arms generate for the
same pockets the design is naturally **paired**:

1. Have `scripts/evaluate.py` emit **per-pocket** metrics, not just pooled scalars
   (it already keeps per-pocket structure internally in `pocket_mol_lists`; it just
   doesn't write it out).
2. Compare arms with a **Wilcoxon signed-rank test** on per-pocket median QED,
   n = 100 pairs.
3. Report effect size with a confidence interval, plus the fraction of pockets
   where B beats A — a mean shift driven by 10 pockets is a different finding from
   one that holds in 85 of 100.

This costs no GPU time and may change how strong the headline looks. Do it before
spending on new runs.

## Sequencing

| Phase | Work | Cost | Decision gate |
|---|---|---|---|
| **0** | Per-pocket metrics + paired statistics | hours, CPU | Does the effect survive proper analysis? |
| **1** | Arm S (shuffle control); evaluate trained arm D; matched docking A vs B | ~1–2 days, no retraining | **S ≈ B kills the interaction hypothesis** and redirects the whole project |
| **2** | Arms R and P; cross-docking specificity | 2 training runs + docking | What specifically does ATOMICA contribute? |
| **3** | 3 seeds of the surviving arm; PBP3 case study | 3 training runs | Error bars; the application claim |

Phase 1 is where the scientific value is concentrated, and almost all of it needs
no retraining. **Run S before anything else.**

## What a paper looks like either way

- **If conditioning is pocket-specific:** "Pretrained interaction representations
  transfer to structure-based generation" — with S, R and P as the controls that
  make the claim credible, and cross-docking specificity as the headline metric.
- **If it is a drug-likeness prior:** "Foundation-model conditioning improves
  drug-likeness but not pocket-specificity in geometric diffusion" — a genuine
  negative result with a clean diagnosis, controls that localise the failure to
  the conditioning signal rather than the architecture, and a concrete statement
  of what a useful conditioning signal would need to carry.

The second is a real contribution and is currently under-reported in the field.
Both outcomes are worth writing up; the current state — QED up, mechanism
unknown — is the only one that is not.

## Caveats to carry into any of this

- Arm B froze the backbone, so all conclusions are about what an adapter adds to a
  fixed pretrained model.
- Evaluation pockets are CrossDocked; no ESKAPE/PBP3 target appears in the
  benchmark. The PBP3 application is motivation, not evidence.
- Single seed throughout. Run-to-run variance of the adapter is unmeasured, so a
  small effect could be noise until Phase 3.
