# Results

## The A/B ablation

Two arms, evaluated on the same 100 held-out CrossDocked pockets with 100 samples
drawn per pocket (100 diffusion timesteps, no resampling):

- **A — baseline.** The pretrained DiffSBDD denoiser, conditioned on pocket
  coordinates only (`atomica_nf: 0`, `--no_atomica`).
- **B — ATOMICA-conditioned.** The same backbone, frozen, plus the cross-attention
  adapter reading frozen ATOMICA pocket embeddings.

Because the backbone is frozen in arm B and the adapter is zero-initialised, the
two arms start from *identical* behaviour; every difference below is attributable
to the adapter.

![Relative change from the unconditioned baseline](../results/figures/ablation.png)

| Metric | A-baseline | cond_B | Δ |
| --- | --- | --- | --- |
| QED | 0.424 ± 0.214 | 0.483 ± 0.208 | +0.059 ↑ |
| SA | 0.581 ± 0.130 | 0.585 ± 0.112 | +0.004 ↑ |
| Lipinski | 4.417 ± 0.970 | 4.690 ± 0.696 | +0.273 ↑ |
| Validity | 0.962 | 0.950 | −0.012 ↓ |
| Diversity | 0.731 ± 0.042 | 0.684 ± 0.025 | −0.046 ↓ |
| Uniqueness | 1.000 | 0.999 | −0.001 ↓ |
| Novelty | 1.000 | 1.000 | ≈ |

- **A-baseline**: 9,328 valid / 9,698 generated across 100 pockets
- **cond_B**: 9,246 valid / 9,736 generated across 100 pockets

Regenerate with:

```bash
python scripts/compare_conditions.py \
    --conditions results/baseline_A results/cond_B --outdir results
```

## What this does and does not show

**Supported by the data.** Conditioning on a pretrained interaction-foundation-model
embedding measurably shifts the generated distribution toward drug-likeness:
QED +13.9% relative, Lipinski compliance +6.2%, and the spread of both tightens
(QED std 0.214→0.208, Lipinski std 0.970→0.696). The effect is consistent enough
across ~9,300 molecules per arm that it is not sampling noise.

**Not yet supported.** That this reflects better *pocket-specific* fit. Every metric
in the table above is computed from the ligand alone — QED, SA and Lipinski do not
know which pocket the molecule was generated for. So the result is equally
consistent with a less interesting explanation: the adapter narrowed the model
toward a generically drug-like region of chemical space, which would raise QED for
any pocket. The 6.4% diversity drop and the tightened standard deviations are what
that explanation predicts.

Distinguishing the two requires a pocket-aware measure. The decisive experiment is
a **matched docking comparison** — Vina scores for both arms over the same 100
pockets:

- If affinity improves alongside QED, the conditioning is doing structural work.
- If affinity is flat while QED rises, the adapter is a drug-likeness prior, not a
  binding prior. That is a genuine and more interesting negative result, and it
  localises the bottleneck to the conditioning signal rather than the architecture.

Either outcome is worth reporting. The current state — QED up, affinity unmeasured —
is the one state that supports no conclusion.

`docking/scores.csv` holds Vina scores for 50 ligands from a single generation run.
It is **not** the matched comparison and should not be read as one.

## Root cause of the null pocket-specificity

The likely explanation is upstream of the model. `scripts/process_expert_atomica.py`
feeds ATOMICA a **single segment** (`segment_ids = [0, 0]`) with the entire pocket
as **one `UNK` block**. ATOMICA is pretrained on two interacting segments over
chemically-typed residue/fragment blocks, so this invokes none of its interaction
semantics and erases its block vocabulary. What remains is per-atom element and
local geometry -- exactly the input from which a generic drug-likeness shift with no
pocket specificity would be expected.

Read that way, this table is a clean negative result for naively-extracted
foundation-model embeddings as a conditioning signal. See
[experiment-plan.md](experiment-plan.md).

## Caveats

- Single seed per arm; no confidence intervals over repeated training runs. The
  per-pocket spread is reported, but run-to-run variance of the adapter is unknown.
- Arm B freezes the backbone, so this measures what the adapter adds *on top of* a
  fixed pretrained model, not what end-to-end training with ATOMICA would achieve.
- Evaluation pockets come from CrossDocked; no ESKAPE/PBP3 target is represented in
  this table. The PBP3 application is the motivation, not the benchmark.
- Conditions C and D of the planned progression were never run — see
  [MODIFICATIONS.md](../MODIFICATIONS.md#status-of-the-planned-ablation-arms).
