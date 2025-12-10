#!/bin/bash
# Setup script for DiffSBDD using Python venv with CUDA support
# Requires NVIDIA GPU with CUDA 11.8 support

set -e  # Exit on error

echo "Setting up DiffSBDD virtual environment with CUDA support..."

# Require Python 3.10 - error if not found
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
    PYTHON_VERSION=$(python3.10 --version 2>&1 | awk '{print $2}')
    echo "Found Python 3.10 ($PYTHON_VERSION), using it for venv"
else
    echo "ERROR: Python 3.10 is required but not found."
    echo ""
    echo "Please install Python 3.10:"
    echo "  brew install python@3.10  # macOS"
    echo "  # or use your Linux package manager"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo "⚠ WARNING: nvidia-smi not found. CUDA may not be available."
    echo "  This script will install CUDA-enabled PyTorch, but it may not work without a GPU."
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
else
    echo "Virtual environment already exists. Skipping creation."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CUDA 11.8 version)
echo "Installing PyTorch (CUDA 11.8 version)..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements (torch-scatter is commented out)
echo "Installing remaining requirements..."
pip install -r requirements.txt

# Install torch-scatter and torch-cluster separately after PyTorch is installed
# Temporarily disable exit on error for this step
set +e
echo ""
echo "Installing torch-scatter and torch-cluster (required for DiffSBDD)..."
echo "Attempting to install from pre-built wheels..."

# Try installing torch-scatter with --no-build-isolation (allows build to see installed torch)
echo "Installing torch-scatter..."
if pip install torch-scatter==2.1.2 --no-build-isolation 2>/dev/null; then
    echo "✓ torch-scatter installed successfully"
else
    echo "Standard installation failed, trying pre-built wheels..."
    if pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html --no-build-isolation 2>/dev/null; then
        echo "✓ torch-scatter installed successfully from pre-built wheels"
    else
        echo "⚠ WARNING: torch-scatter installation failed!"
        echo "  This may cause issues. Try installing manually:"
        echo "    pip install torch-scatter==2.1.2 --no-build-isolation"
        echo "    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html --no-build-isolation"
        echo "  Or use conda: conda install -c pyg pytorch-scatter=2.1.2"
    fi
fi

# Try installing torch-cluster with --no-build-isolation
echo ""
echo "Installing torch-cluster..."
if pip install torch-cluster --no-build-isolation 2>/dev/null; then
    echo "✓ torch-cluster installed successfully"
else
    echo "Standard installation failed, trying pre-built wheels..."
    if pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html --no-build-isolation 2>/dev/null; then
        echo "✓ torch-cluster installed successfully from pre-built wheels"
    else
        echo "⚠ WARNING: torch-cluster installation failed!"
        echo "  This may cause issues. Try installing manually:"
        echo "    pip install torch-cluster --no-build-isolation"
        echo "    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html --no-build-isolation"
        echo "  Or use conda: conda install -c pyg pytorch-cluster"
    fi
fi
set -e  # Re-enable exit on error

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Verify CUDA is available:"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"

