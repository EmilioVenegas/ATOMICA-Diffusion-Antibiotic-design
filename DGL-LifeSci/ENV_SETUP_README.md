# DGL-LifeSci Environment Setup Guide

## CPU-Only Environment for Binding Affinity Prediction

This guide covers setting up a CPU-only environment for transfer learning on PDBBind binding affinity prediction.

### Quick Start

#### On Linux/Mac:
```bash
bash setup_env_cpu.sh
```

#### On Windows (PowerShell):
```powershell
.\setup_env_cpu.ps1
```

### Manual Installation

If the automated scripts don't work, follow these steps:

#### 1. Create Conda Environment
```bash
conda create -n dgl-bap python=3.10 -y
conda activate dgl-bap
```

#### 2. Install PyTorch (CPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Install DGL (CPU)
```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

#### 4. Install Dependencies
```bash
pip install -r requirements_cpu.txt
```

#### 5. Install DGLLife from Source
```bash
cd python
pip install -e .
cd ..
```

### Verify Installation

After setup, verify everything is working:

```python
# Test PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be False for CPU

# Test DGL
import dgl
print(f"DGL version: {dgl.__version__}")

# Test DGLLife
import dgllife
print(f"DGLLife version: {dgllife.__version__}")

# Test RDKit
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')
print(f"RDKit working: {mol is not None}")
```

### Environment Details

- **Environment name**: `dgl-bap`
- **Python version**: 3.10
- **PyTorch**: CPU-only version (latest)
- **DGL**: CPU-only version (latest)
- **RDKit**: For molecular featurization
- **Key packages**: scikit-learn, pandas, numpy, scipy, matplotlib, pyyaml

### Troubleshooting

#### Issue: Conda command not found
**Solution**: Ensure Conda is installed and in your PATH. If using Miniconda or Miniforge, restart your shell after installation.

#### Issue: pip install fails for rdkit-pypi
**Solution**: Try installing RDKit via conda instead:
```bash
conda install -c conda-forge rdkit
```

#### Issue: DGL installation fails
**Solution**: Check the DGL website for the latest installation instructions: https://www.dgl.ai/pages/start.html

#### Issue: Permission denied on setup scripts
**Solution** (Linux/Mac): Make the script executable:
```bash
chmod +x setup_env_cpu.sh
```

### CPU Performance Notes

Training on CPU will be significantly slower than GPU. For large-scale experiments:
- Reduce batch size (e.g., 8-16 instead of 32-64)
- Use fewer training epochs for initial experiments
- Consider using a subset of PDBBind for prototyping
- Enable PyTorch's optimizations: `torch.set_num_threads()`

### Next Steps

After environment setup:
1. Download PDBBind dataset (see `examples/binding_affinity_prediction/README_TRANSFER.md`)
2. Run data preparation: `python examples/binding_affinity_prediction/data_prep_pdbbind.py`
3. Start training: `python examples/binding_affinity_prediction/train_transfer.py --config configs/pdbbind_refined_v2015_transfer.yaml`

### Deactivating Environment

When done:
```bash
conda deactivate
```

### Removing Environment

To completely remove the environment:
```bash
conda env remove -n dgl-bap
```
