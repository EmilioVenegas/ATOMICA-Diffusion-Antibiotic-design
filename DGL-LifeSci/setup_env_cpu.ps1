# Setup script for DGL-LifeSci CPU-only environment (Windows PowerShell)
# For binding affinity prediction transfer learning on PDBBind
#
# Usage:
#   .\setup_env_cpu.ps1
#
# This creates a conda environment named 'dgl-bap' (DGL Binding Affinity Prediction)
# optimized for CPU-only training.

$ErrorActionPreference = "Stop"

$ENV_NAME = "dgl-bap"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "DGL-LifeSci CPU Environment Setup" -ForegroundColor Cyan
Write-Host "Environment name: $ENV_NAME" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Create conda environment with Python 3.10
Write-Host ""
Write-Host "[1/4] Creating conda environment with Python 3.10..." -ForegroundColor Yellow
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
Write-Host ""
Write-Host "[2/4] Activating environment..." -ForegroundColor Yellow
conda activate $ENV_NAME

# Install PyTorch CPU version
Write-Host ""
Write-Host "[3/4] Installing PyTorch (CPU version)..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install DGL CPU version
Write-Host ""
Write-Host "Installing DGL (CPU version)..." -ForegroundColor Yellow
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Install core dependencies
Write-Host ""
Write-Host "[4/4] Installing core dependencies..." -ForegroundColor Yellow
pip install -r requirements_cpu.txt

# Install dgllife from source (current directory)
Write-Host ""
Write-Host "Installing dgllife from source..." -ForegroundColor Yellow
Set-Location python
pip install -e .
Set-Location ..

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Cyan
Write-Host "  conda activate $ENV_NAME" -ForegroundColor White
Write-Host ""
Write-Host "To verify installation:" -ForegroundColor Cyan
Write-Host "  python -c 'import torch; print(f`"PyTorch: {torch.__version__}`")'" -ForegroundColor White
Write-Host "  python -c 'import dgl; print(f`"DGL: {dgl.__version__}`")'" -ForegroundColor White
Write-Host "  python -c 'import dgllife; print(f`"DGLLife: {dgllife.__version__}`")'" -ForegroundColor White
Write-Host ""
