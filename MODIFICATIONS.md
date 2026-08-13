# Modifications to vendored DiffSBDD

`DiffSBDD/` is a fork of [arneschneuing/DiffSBDD](https://github.com/arneschneuing/DiffSBDD),
not a clean copy. This file records exactly what was changed, so the boundary
between upstream work and the contribution of this project is inspectable
without diffing the trees by hand.

`ATOMICA/` is vendored **unmodified** and used only as a frozen encoder.

Regenerate this map at any time:

```bash
git clone --depth 1 https://github.com/arneschneuing/DiffSBDD /tmp/upstream-diffsbdd
python scripts/diff_upstream.py --upstream /tmp/upstream-diffsbdd --fork DiffSBDD
```

## Changed files

Changed-line counts are `diff -u` added+removed lines against upstream `main`.

| File | ± lines | What changed |
|---|---:|---|
| `equivariant_diffusion/dynamics.py` | 239 | **The core contribution.** Adds `SE3EquivariantCrossAttention` and wires it into `EGNNDynamics`. |
| `lightning_modules.py` | 144 | Threads `atomica_nf` through model construction; splits the optimiser into backbone and adapter parameter groups (`lr` vs `adapter_lr`); adds `freeze_backbone`; BioPython 1.8x compatibility shim for `three_to_one`. |
| `utils.py` | 101 | `load_atomica_model()` and `format_atomica_batch()` — load the pretrained ATOMICA encoder and convert a DiffSBDD pocket into ATOMICA's block/segment batch format. |
| `equivariant_diffusion/conditional_model.py` | 97 | Threads `h_atomica` through every step of the reverse process so the pocket embedding reaches the denoiser at each timestep. |
| `train.py` | 73 | Repo-root `sys.path` setup; graceful resume when a checkpoint's architecture predates the adapter (loads the backbone, leaves adapter weights fresh) instead of crashing on a key mismatch. |
| `dataset.py` | 59 | `LigandPocketDatasetPT` — loads preprocessed `.pt` complexes carrying precomputed ATOMICA embeddings, centres on centre of mass, caches in memory. |
| `optimize.py` | 45 | ATOMICA imports and path setup for the property-guided optimisation entry point. |
| `constants.py` | 36 | ATOMICA↔DiffSBDD atom vocabulary mapping (`atomica_atom_encoder`, `atomica_block_encoder`, `ATOMICA_TO_DRUGLIKE_MAP`). |
| `generate_ligands.py` | 27 | `--atomica_config`, `--atomica_weights`, `--no_atomica` flags; path setup. |
| `analysis/molecule_builder.py` | 24 | OpenBabel import path fixed for `openbabel-wheel`. |
| `analysis/SA_Score/sascorer.py` | 8 | Import compatibility for current RDKit. |

New files: `__init__.py`, `equivariant_diffusion/__init__.py` (make the fork importable as a package).

`equivariant_diffusion/egnn_new.py` and `equivariant_diffusion/en_diffusion.py` are
**unchanged** from upstream — the EGNN layers and the base diffusion process are
used as-is. All conditioning enters through `dynamics.py`.

## The conditioning mechanism

`SE3EquivariantCrossAttention` (`equivariant_diffusion/dynamics.py`) inserts one
cross-attention block into each denoising step:

- **Queries** — intermediate ligand scalar features `h_l`, concatenated with a
  16-dimensional learned timestep embedding.
- **Keys / values** — the frozen per-atom ATOMICA pocket embeddings `h_p`.
- **Masking** — a `mask_l[:, None] == mask_p[None, :]` block-diagonal mask keeps
  attention inside each complex, so variable-sized pockets batch together without
  padding artifacts.
- **Output** — a delta on the ligand's *invariant scalar* features only. Coordinates
  are never touched by this module, which is what preserves SE(3) equivariance:
  invariant inputs updating invariant features cannot introduce a frame dependence.
- **Warm start** — `out_proj` is zero-initialised, so at step 0 the adapter is an
  exact no-op and training begins from the pretrained baseline's behaviour. A
  sigmoid `gate` on the timestep learns how much conditioning to apply per noise
  level.

Numerical guards (`nan_to_num` on inputs and post-softmax, attention scores clamped
to ±100) were added after NaN losses during early full-atom training runs.

## Status of the planned ablation arms

The configs describe a four-arm progression (A→D). **All four were trained, but
only A and B were sampled and evaluated** — `results/` contains no `cond_C` or
`cond_D`, so the published comparison is A vs B only. Trained checkpoints for the
other arms are in `my_logs/`.

| Arm | Config | Trained | Evaluated | Notes |
|---|---|:--:|:--:|---|
| **A** — no conditioning (`atomica_nf: 0`) | `crossdock_fullatom_cond.yml` | yes | **yes** | `results/baseline_A/` |
| **B** — adapter, backbone frozen | `crossdock_fullatom_cond_B.yml` | yes | **yes** | `results/cond_B/` |
| C — "timestep-adaptive" adapter | `crossdock_fullatom_cond_C.yml` | yes | no | **Architecturally identical to B.** The two configs differ only in `timestep_adaptive` (False vs True), and no code reads that key — the timestep embedding and gate are unconditionally active in `SE3EquivariantCrossAttention`. Both arms therefore trained the same model; C is a duplicate of B under a different run name. |
| D — "LoRA" fine-tuning | `crossdock_fullatom_cond_D.yml` | yes | no | **Mislabeled, not absent.** `lora_rank`/`lora_alpha` are read by no code and `egnn_new.py` is unmodified from upstream, so no low-rank adaptation happens. What actually distinguishes D is `freeze_backbone: False` — it is full backbone fine-tuning. (Consistent with its 65 MB checkpoint against 18–19 MB for the frozen-backbone arms: an unfrozen backbone carries optimizer state for every parameter.) |

Remaining work for these arms is listed in [run_scripts.md](run_scripts.md):
generation for C and D, `scripts/evaluate.py` for both, docking for C, and a
re-run of all four arms under `--sa_max 0.7`.

## Known gap

The A/B comparison reports distributional and drug-likeness metrics only. It does
**not** yet include a matched binding-affinity comparison across the same pockets:
`docking/scores.csv` covers 50 ligands from a single run, not both arms over the
100-pocket evaluation set. Since QED is target-independent, the current result
cannot separate "conditioning improved pocket-specific fit" from "conditioning
narrowed the model toward generically drug-like chemistry" — and the 6.4% diversity
drop is consistent with the latter. See [docs/results.md](docs/results.md).
