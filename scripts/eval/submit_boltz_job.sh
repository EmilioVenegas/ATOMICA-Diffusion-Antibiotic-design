#!/bin/bash
# Submit Boltz-2 scoring job to MIT Engaging Cloud cluster
#
# Usage:
#   ./submit_boltz_job.sh <chemical-space-file> <sample-size> [options]
#
# Example:
#   ./submit_boltz_job.sh data/chemical_space.csv 100 --template templates/target.yaml

set -e

# Default values
PARTITION="${PARTITION:-mit_normal_gpu}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
CPUS="${CPUS:-16}"
TIME="${TIME:-6:00:00}"
MEM="${MEM:-32GB}"
GPU="${GPU:-1}"
ACCELERATOR="${ACCELERATOR:-gpu}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/boltz_cluster_score.py"

# Check if running in tmux (recommended)
if [ -z "$TMUX" ]; then
    echo "Warning: Not running in tmux. Consider starting a tmux session first:"
    echo "tmux new -s boltz-scoring"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Parse required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <chemical-space-file> <sample-size> [additional-args...]"
    echo ""
    echo "Required arguments:"
    echo "  chemical-space-file  Path to CSV/SMI/TXT file with SMILES"
    echo "  sample-size          Number of molecules to sample and score"
    echo ""
    echo "Optional arguments (passed to boltz_cluster_score.py):"
    echo "  --column COLUMN              Column name for SMILES (default: smiles)"
    echo "  --template PATH              Template YAML for Boltz"
    echo "  --binder-id ID               Ligand identifier (default: LIG)"
    echo "  --output-dir PATH            Output directory (default: outputs/boltz_<timestamp>)"
    echo "  --seed SEED                  Random seed"
    echo "  --sampling-steps N           Diffusion steps (default: 25)"
    echo "  --sampling-steps-affinity N  Affinity sampling steps (default: 50)"
    echo "  --keep-inputs                Keep generated YAML files"
    echo ""
    echo "Environment variables (for resource customization):"
    echo "  PARTITION    Slurm partition (default: mit_normal_gpu)"
    echo "  CPUS         CPUs per task (default: 16)"
    echo "  MEM          Memory (default: 32GB)"
    echo "  GPU          Number of GPUs (default: 1)"
    echo "  TIME         Time limit (default: 6:00:00)"
    echo ""
    exit 1
fi

CHEMICAL_SPACE="$1"
SAMPLE_SIZE="$2"
shift 2

# Validate chemical space file
if [ ! -f "$CHEMICAL_SPACE" ]; then
    echo "Error: Chemical space file not found: $CHEMICAL_SPACE"
    exit 1
fi

# Generate default output directory if not provided
HAS_OUTPUT_DIR=false
for arg in "$@"; do
    if [[ "$arg" == "--output-dir" ]]; then
        HAS_OUTPUT_DIR=true
        break
    fi
done

if [ "$HAS_OUTPUT_DIR" = false ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/boltz_${TIMESTAMP}"
    EXTRA_ARGS=("--output-dir" "${OUTPUT_DIR}")
else
    EXTRA_ARGS=()
fi

# Build srun command
SRUN_CMD=(
    srun
    --nodes="${NODES}"
    --ntasks-per-node="${NTASKS}"
    --cpus-per-task="${CPUS}"
    --time="${TIME}"
    --partition="${PARTITION}"
    --gres="gpu:${GPU}"
    --mem="${MEM}"
    --pty
    python "${PYTHON_SCRIPT}"
    --chemical-space "${CHEMICAL_SPACE}"
    --sample-size "${SAMPLE_SIZE}"
    --accelerator "${ACCELERATOR}"
    "${EXTRA_ARGS[@]}"
    "$@"
)

echo "=" | tr -d '\n' | head -c 60
echo ""
echo "Submitting Boltz-2 scoring job"
echo "=" | tr -d '\n' | head -c 60
echo ""
echo "Chemical space: ${CHEMICAL_SPACE}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Partition: ${PARTITION}"
echo "GPUs: ${GPU}"
echo "CPUs: ${CPUS}"
echo "Memory: ${MEM}"
echo "Time limit: ${TIME}"
echo "=" | tr -d '\n' | head -c 60
echo ""
echo "Command: ${SRUN_CMD[*]}"
echo ""
echo "Press Ctrl+C to interrupt the job"
echo ""

# Change to project root and run
cd "${PROJECT_ROOT}"
exec "${SRUN_CMD[@]}"

