#!/bin/bash
# Setup script for DGL-LifeSci CPU-only environment
# For binding affinity prediction transfer learning on PDBBind
#
# Usage:
#   bash setup_env_cpu.sh
#
# This creates a conda environment named 'dgl-bap' (DGL Binding Affinity Prediction)
# optimized for CPU-only training.

set -e

ENV_NAME="dgl-bap"

echo "================================================"
echo "DGL-LifeSci CPU Environment Setup"
echo "Environment name: ${ENV_NAME}"
echo "================================================"

# Create conda environment with Python 3.10
echo ""
echo "[1/4] Creating conda environment with Python 3.10..."
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo ""
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install PyTorch CPU version
echo ""
echo "[3/4] Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install DGL CPU version
echo ""
echo "Installing DGL (CPU version)..."
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Install core dependencies
echo ""
echo "[4/4] Installing core dependencies..."
pip install -r requirements_cpu.txt

# Install dgllife from source (current directory)
echo ""
echo "Installing dgllife from source..."
cd python
pip install -e .
cd ..

echo ""
echo "================================================"
echo "Environment setup complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import dgl; print(f\"DGL: {dgl.__version__}\")'"
echo "  python -c 'import dgllife; print(f\"DGLLife: {dgllife.__version__}\")'"
echo ""
