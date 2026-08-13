# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ATOMICA-Diffusion-Antibiotic-design** integrates two models for *de novo* antibiotic design:

1. **ATOMICA**: A geometric AI model pretrained on 2M+ molecular interaction interfaces. Generates fixed-size (32-dim) pocket embeddings from protein structures.
2. **DiffSBDD**: An SE(3)-equivariant diffusion model for structure-based drug design that generates ligand molecules conditioned on protein pockets.

The core innovation is a `SE3EquivariantCrossAttention` module in `DiffSBDD/equivariant_diffusion/dynamics.py` that lets the diffusion denoiser attend to ATOMICA pocket embeddings, providing a richer conditioning signal than raw pocket coordinates alone.

### Data Flow

```
CrossDocked LMDB
    ↓ process_expert_atomica.py  (filter Vina < -8.5, compute ATOMICA embeddings)
    ↓
.pt files (data/processed_expert_atomica/)
    ↓ LigandPocketDatasetPT
    ↓
LigandPocketDDPM  (PyTorch Lightning, wraps EGNNDynamics + ConditionalDDPM)
    ↓
generate_ligands.py / optimize.py  →  SDF
    ↓ rl_loop/RL_loop.py  (optional: ADMET-AI scoring, REOS filter, top-k)
```

## Environment Setup

Two environments exist; use the **conda** one for all training/inference:

```bash
# Conda (DiffSBDD core — Python 3.10.4, PyTorch 2.0.1, CUDA 11.8)
conda env create -f DiffSBDD/environment.yaml
conda activate diffsbdd

# Poetry (newer tooling, Python 3.12 — for scripts using boltz/openbabel-wheel)
pip install poetry && poetry install
```

The conda env provides: PyTorch Lightning 1.8.4, RDKit, BioPython, torch-scatter, WandB, admet-ai.

## Key Commands

### Data Preprocessing
```bash
# From repo root — filter LMDB by Vina score, compute ATOMICA embeddings → .pt files
python scripts/process_expert_atomica.py
```

### Training
All training runs from inside `DiffSBDD/` (train.py adds repo root to `sys.path`):
```bash
cd DiffSBDD

# Condition A — baseline, no ATOMICA (set atomica_nf: 0 in config)
python train.py --config configs/crossdock_fullatom_cond.yml \
  --resume checkpoints/pretrained_baseline.ckpt

# Condition B — adapter only, backbone frozen
python train.py --config configs/crossdock_fullatom_cond_B.yml

# Condition C — timestep-adaptive adapter (set timestep_adaptive: True)
python train.py --config configs/crossdock_fullatom_cond_C.yml

# Condition D — LoRA fine-tuning (set lora_rank: 8, lora_alpha: 16)
python train.py --config configs/crossdock_fullatom_cond.yml

# Resume from checkpoint
python train.py --config configs/crossdock_fullatom_cond.yml \
  --resume ../my_logs/SE3-cond-full-atomica-v9/checkpoints/last.ckpt
```

### Inference
```bash
# Generate ligands with ATOMICA conditioning
python DiffSBDD/generate_ligands.py \
  checkpoints/crossdocked_fullatom_cond.ckpt \
  --pdbfile example/5ndu.pdb \
  --ref_ligand example/5ndu_linked_mols.sdf \
  --atomica_config ATOMICA/pretrain/pretrain_model_config.json \
  --atomica_weights ATOMICA/pretrain/pretrain_model_weights.pt \
  --outfile generated_ligands.sdf \
  --n_samples 20 --resamplings 10

# Baseline: no ATOMICA
python DiffSBDD/generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
  --pdbfile example/5ndu.pdb --ref_ligand example/5ndu_linked_mols.sdf \
  --outfile generated_ligands_baseline.sdf --no_atomica

# Batch baseline for test set
python scripts/run_baseline.py \
  --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
  --test_dir DiffSBDD/data/processed_expert_atomica/test \
  --outdir results/baseline \
  --n_pockets 100 --n_samples 100 --batch_size 20 --timesteps 100 --no_atomica

# Optimize with QED guidance
python DiffSBDD/optimize.py \
  --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \
  --pdbfile example/5ndu.pdb --ref_ligand example/5ndu_linked_mols.sdf \
  --property qed --outfile optimized_ligands.sdf
```

### Tests
```bash
# Ablation-result regression tests (no GPU or dataset needed)
pytest tests -q

# RL loop tests
python rl_loop/test_rl_loop.py
```
There is no unit test covering `SE3EquivariantCrossAttention` itself — an
equivariance/masking test for it is the most valuable one to add next.

## Architecture

### Core: SE3EquivariantCrossAttention (`DiffSBDD/equivariant_diffusion/dynamics.py`)

Integrates ATOMICA embeddings into each denoising step:
- **Query**: ligand features `h_l` concatenated with a 16-dim timestep embedding (always on — there is no `timestep_adaptive` switch in the code)
- **Key/Value**: per-pocket ATOMICA embeddings `h_p` (32-dim invariant scalars)
- Masked scaled dot-product attention (batch-aware, variable-sized molecules)
- Output: ligand feature delta. `out_proj` is zero-initialized, so the adapter is an exact no-op at step 0; a sigmoid `gate` on the timestep (init ~0.5) learns how much conditioning to apply per noise level.

Cross-attention preserves SE(3) equivariance: invariant ATOMICA scalars update invariant ligand scalar features `h`, never coordinates.

### Diffusion (`DiffSBDD/equivariant_diffusion/`)

- `dynamics.py` — `EGNNDynamics`: SE(3)-equivariant graph neural network, denoises ligand and pocket coordinates jointly. Contains the ATOMICA integration via `SE3EquivariantCrossAttention`.
- `egnn_new.py` — EGNN layer. **Unmodified from upstream.** LoRA is *not* implemented here or anywhere else; `lora_rank`/`lora_alpha` appear only in configs and are read by no code.
- `en_diffusion.py` — `EnVariationalDiffusion`: base diffusion class (forward/reverse process, noise schedule).
- `conditional_model.py` — `ConditionalDDPM`: only ligand is denoised; pocket is fixed context.

### Training (`DiffSBDD/lightning_modules.py`)

`LigandPocketDDPM` (PyTorch Lightning module, ~1200 lines):
- Creates EGNNDynamics → wraps in ConditionalDDPM
- Atom type encoders/decoders from `DiffSBDD/constants.py`
- Two optimizer param groups: `lr` for backbone, `adapter_lr` for adapter layers
- Validation: samples molecules from pure noise every `eval_epochs`, computes validity/QED/SA/docking, logs to WandB

### Data (`DiffSBDD/dataset.py`)

- `LigandPocketDatasetPT`: loads `.pt` files, centers to CoM, in-memory cache
- Each `.pt` file: `lig_coords`, `pocket_coords`, `lig_one_hot`, `pocket_one_hot`, `pocket_atomica_embeddings` (32-dim)
- Custom `collate_fn`: per-atom masks for variable-sized molecules in a batch

### RL Loop (`rl_loop/`)

Post-generation pipeline: ADMET-AI scoring → REOS filter (Lipinski-like) → Tanimoto diversity → top-k extraction. Entry point: `rl_loop/RL_loop.py`.

## Configuration

Configs live in `DiffSBDD/configs/`. Key parameters:

```yaml
# Experimental conditions
freeze_backbone: False        # True = only adapter trains (Condition B)
egnn_params:
  atomica_nf: 32              # 0 = disable ATOMICA
  lora_rank: 0                # NOT IMPLEMENTED -- see MODIFICATIONS.md
  lora_alpha: 16
  timestep_adaptive: True     # NOT READ BY ANY CODE -- see MODIFICATIONS.md
  gradient_checkpointing: True  # required for <16GB VRAM with full-atom pockets

# Training
lr: 1.0e-5
adapter_lr: 1.0e-4            # higher than backbone LR in adapter-only mode
accumulate_grad_batches: 4    # effective batch = batch_size × this
val_check_interval: 200       # validate every N steps (useful for short epochs)
precision: bf16-mixed

# Diffusion
diffusion_params:
  diffusion_steps: 500
  diffusion_noise_schedule: 'polynomial_2'
```

The ablation progression A→B→C→D was the plan, but **only A and B have been run**.
C is not distinguishable from B (its timestep behaviour is unconditionally active),
and D's LoRA knobs are config-only with no implementation. See MODIFICATIONS.md.

## Repo Layout Notes

- Preprocessed `.pt` files go in `data/processed_expert_atomica/` (not committed).
- Checkpoints: `my_logs/<run_name>/checkpoints/` during training; `checkpoints/` for pretrained.
- WandB entity: `emiliovenegas10-tecnol-gico-de-monterrey`, project: `atomica-enhanced`.
- `constants.py` defines atom vocabularies (`DRUGLIKE_ATOMS_DECODER`, `ATOMICA_TO_DRUGLIKE_MAP`, `atomica_block_encoder`).
- `DiffSBDD/analysis/` has post-generation tools: `metrics.py` (validity/QED/SA), `molecule_builder.py` (RDKit mol), `docking.py` (smina scores).

## Debugging

| Issue | Fix |
|-------|-----|
| CUDA OOM | Add `gradient_checkpointing: True` to `egnn_params`, reduce `batch_size`, use `precision: bf16-mixed` |
| NaN loss | Check clash detection in preprocessing; reduce `lr`; `SE3EquivariantCrossAttention` has `nan_to_num` guards and attention score clamping `[-100, 100]` |
| Slow generation | Reduce `resamplings` or `--timesteps` in `generate_ligands.py` |
| ATOMICA not loading | Verify `--atomica_config` / `--atomica_weights` paths; check `ATOMICA/pretrain/` |
| RL loop fails | `pip install admet-ai`; validate SDF with RDKit before passing to RL loop |
| `5.0n` typo | `crossdock_fullatom_cond_B.yml` line 46 has `edge_cutoff_interaction: 5.0n` — fix to `5.0` |
