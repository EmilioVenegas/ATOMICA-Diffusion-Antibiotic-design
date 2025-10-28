# Transfer Learning Guide for PDBBind Binding Affinity Prediction

This guide covers the transfer learning approach for protein-ligand binding affinity prediction using PDBBind dataset.

## Overview

Transfer learning approach:
1. **Pre-train** or load pre-trained ligand encoder (e.g., from molecular property prediction tasks)
2. **Freeze** the ligand encoder and train complex-level layers on PDBBind
3. **Unfreeze** all layers and fine-tune end-to-end
4. **Evaluate** on held-out core set

## Quick Start

### 1. Data Preparation

```bash
# Load PDBBind refined set (v2015, automatically downloaded)
python data_prep_pdbbind.py --subset refined --version v2015

# Or use local PDBBind data
python data_prep_pdbbind.py --subset refined --version v2015 --local_path /path/to/pdbbind

# For evaluation, load core set
python data_prep_pdbbind.py --subset core --version v2015
```

**Output**: Dataset metadata saved to `./data/pdbbind_refined_v2015_metadata.json`

### 2. Training with Transfer Learning

```bash
# TODO: After implementing train_transfer.py
python train_transfer.py --config configs/pdbbind_refined_v2015_transfer.yaml
```

### 3. Evaluation

```bash
# TODO: After implementing eval_and_export.py
python eval_and_export.py --config configs/pdbbind_core_eval.yaml
```

### 4. Visualization

```bash
# TODO: After implementing Transfer_Playground.ipynb
jupyter notebook Transfer_Playground.ipynb
```

---

## Data Preparation Details

### Supported PDBBind Versions

| Version | Refined Set Size | Core Set Size | Availability |
|---------|-----------------|---------------|--------------|
| v2007   | ~1,300          | ~200          | Auto-download via DGL |
| v2015   | 3,706           | 195           | Auto-download via DGL |
| v2020   | ~5,000          | ~300          | Manual download required |

### Downloading PDBBind v2020 (Latest)

PDBBind v2020 is the most recent and comprehensive version but requires manual download:

1. **Register and Download**:
   - Visit: http://www.pdbbind.org.cn/download.php
   - Register for an account (free for academic use)
   - Download the refined set and/or core set

2. **Extract the Dataset**:
   ```bash
   tar -xzf PDBbind_v2020_refined.tar.gz
   tar -xzf PDBbind_v2020_core.tar.gz
   ```

3. **Verify Directory Structure**:
   ```
   PDBBind_v2020/
   ├── refined-set/
   │   ├── index/
   │   │   └── INDEX_refined_data.2020
   │   └── [PDB_ID]/
   │       ├── [PDB_ID]_ligand.mol2
   │       ├── [PDB_ID]_protein.pdb
   │       └── [PDB_ID]_pocket.pdb
   └── core-set/
       └── (similar structure)
   ```

4. **Use with data_prep_pdbbind.py**:
   ```bash
   python data_prep_pdbbind.py \
       --subset refined \
       --version v2020 \
       --local_path /path/to/PDBBind_v2020
   ```

### Data Preparation Options

```bash
python data_prep_pdbbind.py --help
```

Key arguments:
- `--subset`: `refined` (larger, for training) or `core` (smaller, for testing)
- `--version`: PDBBind version (`v2007`, `v2015`, `v2020`)
- `--local_path`: Path to local PDBBind directory (if not using auto-download)
- `--load_binding_pocket`: Load binding pocket only (default: True, faster)
- `--remove_coreset_from_refinedset`: Remove overlap when training on refined, testing on core (default: True)
- `--num_processes`: Number of worker processes (default: number of CPUs)
- `--output_dir`: Where to save metadata (default: `./data`)

---

## Dataset Information

### PDBBind Subsets

**Refined Set**:
- Larger dataset (~3,700+ complexes in v2015)
- High-quality binding affinity data
- Used for training and validation
- Includes diverse protein targets and ligands

**Core Set**:
- Smaller, carefully curated dataset (~200 complexes)
- Non-redundant protein families
- High-resolution crystal structures
- Used for final evaluation and benchmarking

### Binding Affinity Measurements

PDBBind provides experimentally measured binding affinities:
- **Kd** (dissociation constant)
- **Ki** (inhibition constant)  
- **IC50** (half-maximal inhibitory concentration)

All converted to **-log(Kd/Ki)** in molar units (pKd/pKi) for regression.

### Data Splits

Common splitting strategies:
1. **Random**: Random 80/10/10 train/val/test split
2. **Scaffold**: Split by ligand scaffold (tests generalization to new chemotypes)
3. **Temporal**: Split by publication date (simulates real-world deployment)
4. **Stratified**: Stratify by affinity ranges
5. **Structure**: Cluster by protein structure similarity (PDBBind specific)
6. **Sequence**: Cluster by protein sequence similarity (PDBBind specific)

---

## Transfer Learning Strategy

### Stage 1: Freeze Ligand Encoder (5-10 epochs)

- **Freeze**: Ligand GNN encoder weights
- **Train**: Complex-level layers, protein encoder (if separate)
- **Goal**: Learn protein-ligand interaction patterns without destroying pre-trained ligand representations
- **Learning Rate**: Higher (e.g., 1e-3)

### Stage 2: Fine-tune All Layers (40-100 epochs)

- **Unfreeze**: All model parameters
- **Train**: End-to-end fine-tuning
- **Goal**: Jointly optimize all components for binding affinity prediction
- **Learning Rate**: Lower (e.g., 1e-4 to 1e-5)

### Pre-trained Weights Sources

Potential sources for ligand encoder pre-training:
1. **DGLLife pre-trained models**: Trained on molecular property prediction
2. **MoleculeNet tasks**: BBBP, Tox21, HIV, etc.
3. **Self-supervised learning**: Masked atom prediction, graph-level contrastive learning
4. **ChEMBL pre-training**: Large-scale bioactivity data

---

## Models

### PotentialNet (Recommended for Transfer Learning)

3-stage architecture:
1. **Stage 1**: Covalent-only propagation (ligand + protein separately)
2. **Stage 2**: Dual noncovalent + covalent propagation (KNN graphs)
3. **Stage 3**: Ligand-based feature gathering and prediction

**Advantages for TL**:
- Clear separation of ligand and protein encoders
- Stage 1 ligand encoder can be pre-trained independently
- Proven performance on PDBBind

### ACNN (Atomic Convolutional Networks)

Simpler architecture using nearest-neighbor graphs.

**Advantages**:
- Faster training
- Fewer hyperparameters
- Good baseline performance

---

## Expected Performance

### PotentialNet on PDBBind v2015

| Training Set | Test Set | Test R² | Test MAE |
|--------------|----------|---------|----------|
| Refined      | Core     | 0.586   | ~1.1     |
| Refined (core removed) | Core | 0.451 | ~1.3 |

### Transfer Learning Expectations

With pre-trained ligand encoder:
- **Faster convergence**: 30-50% fewer epochs
- **Better generalization**: +5-10% R² on core set
- **Lower overfitting**: More stable validation curves

---

## CPU Performance Notes

Training on CPU (no GPU):
- **Batch size**: Reduce to 8-16 (vs. 32-64 on GPU)
- **Training time**: 10-30x slower per epoch
- **Recommended**: Use subset of data for prototyping
  ```bash
  # TODO: Add --max_samples flag to data_prep_pdbbind.py
  python data_prep_pdbbind.py --subset refined --version v2015 --max_samples 500
  ```

---

## Next Steps

### TODO List for Complete Implementation

- [ ] Implement `model_transfer.py` with PotentialNet/ACNN architectures
- [ ] Implement `train_transfer.py` with two-stage training loop
- [ ] Create YAML configuration files for different experiments
- [ ] Implement `eval_and_export.py` for evaluation and result export
- [ ] Create `Transfer_Playground.ipynb` for visualization
- [ ] Add pre-trained weight loading utilities
- [ ] Implement data splitting strategies
- [ ] Add logging and checkpointing
- [ ] Add early stopping based on validation metrics

### Suggested Workflow

1. **Prototype on v2015**: Fast iteration with auto-downloaded data
2. **Scale to v2020**: Use latest data for final experiments
3. **Try both models**: Compare ACNN (fast) vs. PotentialNet (accurate)
4. **Experiment with splits**: Test generalization with scaffold/temporal splits
5. **Ablation studies**: Compare transfer learning vs. training from scratch

---

## Troubleshooting

### Issue: RDKit installation fails
**Solution**: Install via conda instead of pip:
```bash
conda install -c conda-forge rdkit
```

### Issue: Out of memory during data loading
**Solution**: Reduce `num_processes` or use binding pocket only:
```bash
python data_prep_pdbbind.py --load_binding_pocket --num_processes 2
```

### Issue: PDBBind download is very slow
**Solution**: 
1. Download manually from http://www.pdbbind.org.cn/
2. Use `--local_path` to point to the extracted directory

### Issue: Dataset loading fails with "File not found"
**Solution**: Ensure directory structure matches PDBBind v2015 format (see above)

---

## References

- **PDBBind Database**: http://www.pdbbind.org.cn/
- **DGL-LifeSci Documentation**: https://lifesci.dgl.ai/
- **PotentialNet Paper**: Feinberg et al. (2018) ACS Central Science
- **ACNN Paper**: Gomes et al. (2017) arXiv:1703.10603
- **MoleculeNet Paper**: Wu et al. (2018) Chemical Science

---

## Citation

If you use this transfer learning approach, please cite:

```bibtex
@article{feinberg2018potentialnet,
  title={PotentialNet for molecular property prediction},
  author={Feinberg, Evan N and Sur, Debnil and Wu, Zhenqin and Husic, Brooke E and Mai, Huanghao and Li, Yang and Sun, Saisai and Yang, Jianyi and Ramsundar, Bharath and Pande, Vijay S},
  journal={ACS central science},
  volume={4},
  number={11},
  pages={1520--1530},
  year={2018},
  publisher={ACS Publications}
}

@article{wu2018moleculenet,
  title={MoleculeNet: a benchmark for molecular machine learning},
  author={Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N and Gomes, Joseph and Geniesse, Caleb and Pappu, Aneesh S and Leswing, Karl and Pande, Vijay},
  journal={Chemical science},
  volume={9},
  number={2},
  pages={513--530},
  year={2018},
  publisher={Royal Society of Chemistry}
}
```
