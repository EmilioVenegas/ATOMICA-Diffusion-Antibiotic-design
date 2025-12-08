# Installation Guide for RL Loop

## Prerequisites

You need Python 3.8+ and either conda or pip.

## Option 1: Using Existing DiffSBDD Environment (Recommended)

If you already have the DiffSBDD conda environment set up:

```bash
# Activate the environment
conda activate diffsbdd

# Add RL loop dependencies
pip install admet-ai pyarrow

# Test installation
cd rl_loop
python RL_loop.py --help
```

## Option 2: Create New Conda Environment

```bash
# Create new environment
conda create -n rl-loop python=3.10
conda activate rl-loop

# Install dependencies
conda install -c conda-forge rdkit pandas
pip install admet-ai pyarrow

# Test installation
cd rl_loop
python RL_loop.py --help
```

## Option 3: Using pip (Virtual Environment)

```bash
# Create virtual environment
python -m venv rl_loop_env
source rl_loop_env/bin/activate  # On Windows: rl_loop_env\Scripts\activate

# Install dependencies
pip install rdkit-pypi pandas pyarrow admet-ai

# Test installation
cd rl_loop
python RL_loop.py --help
```

## Verify Installation

Run the test suite to verify everything is working:

```bash
cd rl_loop
python test_rl_loop.py
```

Expected output:
```
============================================================
Testing RL Loop with DiffSBDD Example Ligands
============================================================
Loading ADMET-AI model...
✓ ADMET-AI model loaded successfully
...
```

## Troubleshooting

### RDKit Import Error

**Error:** `ModuleNotFoundError: No module named 'rdkit'`

**Solution:**
```bash
# If using conda
conda install -c conda-forge rdkit

# If using pip
pip install rdkit-pypi
```

### ADMET-AI Not Found

**Error:** `ModuleNotFoundError: No module named 'admet_ai'`

**Solution:**
```bash
pip install admet-ai pyarrow
```

### PyArrow Compatibility Issues

**Error:** Version conflicts with pyarrow

**Solution:**
```bash
pip install --upgrade pyarrow
# Or specify a compatible version
pip install "pyarrow>=11.0.0,<15.0.0"
```

### PATH Issues on Windows

If commands aren't found, make sure to activate your environment:

```powershell
# For conda
conda activate diffsbdd

# For venv
.\rl_loop_env\Scripts\activate
```

## Dependencies Summary

**Core Requirements:**
- Python 3.8+
- rdkit (or rdkit-pypi)
- pandas
- pathlib (included in Python 3.4+)

**Optional but Recommended:**
- admet-ai (for ADMET property prediction)
- pyarrow (required by admet-ai)
- matplotlib (for visualization in tests)

## Testing Your Setup

### Quick Test
```bash
python -c "from rdkit import Chem; print('RDKit:', Chem.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "from admet_ai import ADMETModel; print('ADMET-AI: OK')"
```

### Full Test
```bash
cd rl_loop
python test_rl_loop.py
```

## Next Steps

Once installed, see:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start examples
- `SUMMARY.md` - Implementation overview

