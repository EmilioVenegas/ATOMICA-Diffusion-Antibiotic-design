# Running Boltz-2 Scoring on MIT Engaging Cloud Cluster

This guide explains how to submit Boltz-2 affinity scoring jobs to the MIT Engaging Cloud cluster for faster processing.

## Quick Start

1. **SSH into the cluster** and navigate to your project directory, more details and tutorial here: https://docs.google.com/document/d/1pMI8h0wAyodo5yKyb9dOdgp9vyAwfXTE3SZFHSUen8Q/edit?tab=t.0

2. **Start a tmux session** (recommended):
   ```bash
   tmux new -s boltz-scoring
   ```

3. **Activate your conda/micromamba environment**:
   ```bash
   micromamba activate <your-environment-name>
   ```

4. **Submit the job**:
   ```bash
   cd scripts/eval
   ./submit_boltz_job.sh <chemical-space-file> <sample-size> [options]
   ```

## Example Usage

### Basic example:
```bash
./submit_boltz_job.sh \
    ../../data/chemical_space.csv \
    100
```

### With template YAML:
```bash
./submit_boltz_job.sh \
    ../../data/chemical_space.csv \
    100 \
    --template templates/target.yaml \
    --binder-id LIG
```

### Custom output directory:
```bash
./submit_boltz_job.sh \
    ../../data/chemical_space.csv \
    200 \
    --output-dir ../../outputs/my_experiment \
    --seed 42 \
    --keep-inputs
```

## Arguments

### Required:
- `chemical-space-file`: Path to CSV/SMI/TXT file containing SMILES strings
- `sample-size`: Number of molecules to randomly sample and score

### Optional:
- `--column COLUMN`: Column name for SMILES in CSV (default: `smiles`)
- `--template PATH`: Template YAML file for Boltz-2 (protein sequence, etc.)
- `--binder-id ID`: Ligand identifier in template (default: `LIG`)
- `--output-dir PATH`: Output directory (default: `outputs/boltz_<timestamp>`)
- `--seed SEED`: Random seed for reproducibility
- `--sampling-steps N`: Diffusion sampling steps (default: 25)
- `--sampling-steps-affinity N`: Affinity sampling steps (default: 50)
- `--keep-inputs`: Keep generated YAML input files

## Resource Configuration

You can customize cluster resources via environment variables:

```bash
# Use more CPUs
CPUS=32 ./submit_boltz_job.sh data/chemical_space.csv 100

# Request A100 GPU
GPU=a100:1 ./submit_boltz_job.sh data/chemical_space.csv 100

# More memory
MEM=64GB ./submit_boltz_job.sh data/chemical_space.csv 100

# Different partition
PARTITION=mit_gpu ./submit_boltz_job.sh data/chemical_space.csv 100
```

### Default Resources:
- **Partition**: `mit_normal_gpu`
- **CPUs**: 16
- **Memory**: 32GB
- **GPUs**: 1
- **Time limit**: 6:00:00 (6 hours)

## Output Files

After the job completes, you'll find:

```
<output-dir>/
├── summaries/
│   ├── affinity_summary.json      # Summary statistics
│   ├── affinity_scores.json       # Individual scores
│   └── affinity_histogram.png     # Histogram visualization
└── boltz_results_inputs/          # Generated YAML files (if --keep-inputs)
```

## Monitoring Jobs

### Check job status:
```bash
squeue --me
```

### Attach to tmux session:
```bash
tmux a -t boltz-scoring
```

### Cancel a job:
```bash
scancel <JOBID>
```

## Direct Python Script Usage

If you prefer to run the Python script directly with custom `srun` arguments:

```bash
srun \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --time=6:00:00 \
    --partition=mit_normal_gpu \
    --gres=gpu:1 \
    --mem=32GB \
    --pty \
    python scripts/eval/boltz_cluster_score.py \
        --chemical-space data/chemical_space.csv \
        --sample-size 100 \
        --accelerator gpu \
        --output-dir outputs/my_experiment
```

## Tips

1. **Use GPU**: The `--accelerator gpu` option (default) significantly speeds up Boltz-2 predictions
2. **Start with small samples**: Test with `sample-size=10` before running large batches
3. **Use tmux**: Keeps jobs running if you disconnect
4. **Check partition availability**: Some partitions have different time limits and GPU availability
5. **Monitor resource usage**: Adjust `CPUS` and `MEM` based on your workload

## Troubleshooting

### Job fails immediately:
- Check that your conda environment is activated
- Verify all dependencies are installed (boltz, torch, etc.)
- Ensure input files exist and are readable

### Out of memory:
- Increase `MEM` environment variable
- Reduce `sample-size` or process in batches

### Job times out:
- Increase `TIME` limit or use a different partition
- Reduce `sampling-steps` or `sampling-steps-affinity` for faster runs

### GPU not available:
- Check partition availability: `sinfo`
- Try a different partition or wait for resources

