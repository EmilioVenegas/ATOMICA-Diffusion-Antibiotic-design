RL Loop for ADMET-AI Guided Ligand Optimization
Purpose

This directory implements the first fully working end-to-end pipeline for evaluating diffusion-generated ligands using ADMET-AI. This pipeline was validated on the MIT ORCD GPU cluster and successfully performs:

Ligand generation using DiffSBDD (training-refactor branch)

SDF export to rl_loop/examples/

ADMET-AI scoring & ranking using RL_loop.py

CSV output to rl_loop/results/

This enables iterative ligand design workflows and establishes the foundation for a future reinforcement learning optimization loop.

Directory Structure
rl_loop/
├── README.md                # This file
├── RL_loop.py               # Main scoring and ranking logic
├── mock_generate_ligands.py # Utility for testing without DiffSBDD
├── examples/                # Input ligand SDF files (generated externally)
│   ├── .gitkeep
│   └── generated_from_refactor.sdf      # Example generated ligands (committed manually)
└── results/                 # Scored output CSV files
    ├── .gitkeep
    └── generated_from_refactor_scored.csv  # Example scored results


Note:
Large generated files are normally ignored by .gitignore, but the above two example files were intentionally force-added using git add -f to serve as a canonical demonstration of the working pipeline.

Requirements

Inside the DiffSBDD conda environment:

conda activate diffsbdd
pip install admet-ai pyarrow rdkit-pypi pandas


(ADMET-AI requires Python ≥3.8 and works fine inside the diffsbdd environment.)

Full Working Pipeline (validated)
1. Generate ligands using training-refactor + last_ckpt.ckpt:

From the cluster GPU node:

cd ~/ATOMICA-Diffusion-Antibiotic-design/DiffSBDD

python generate_ligands.py checkpoints/last_ckpt.ckpt \
    --pdbfile example/3rfm.pdb \
    --ref_ligand A:330 \
    --outfile ../rl_loop/examples/generated_from_refactor.sdf \
    --n_samples 100


This produces:

rl_loop/examples/generated_from_refactor.sdf

2. Score ligands with ADMET-AI:

From repo root:

cd ~/ATOMICA-Diffusion-Antibiotic-design

python rl_loop/RL_loop.py \
    --input rl_loop/examples/generated_from_refactor.sdf \
    --output rl_loop/results/generated_from_refactor_scored.csv


Which produces:

rl_loop/results/generated_from_refactor_scored.csv


Both of these files were included in this branch as working examples.

Command-Line Options
--input, -i      Input SDF file (single file)
--output, -o     Output CSV path
--top_k, -k      How many top molecules to keep (default: all)
--admet_props    Comma-separated ADMET properties to rank by (default: all)

Output Format

The CSV contains:

Column	Description
molecule_id	Unique identifier for each molecule
smiles	SMILES representation
rank	Rank sorted by composite_score
composite_score	Weighted score across ADMET properties
[ADMET props]	Individual ADMET-AI predictions
Example Python API Usage
from rl_loop.RL_loop import RLLoop

rl = RLLoop()

mols = rl.load_molecules_from_sdf("examples/generated_from_refactor.sdf")
smiles = rl.convert_to_smiles(mols)
scores = rl.score_with_admet(smiles)

ranked = rl.rank_by_score(scores)
rl.save_results(ranked, "results/example_output.csv")

## Additional Tools

### Testing

**`test_rl_loop.py`** - Test script to verify RL loop functionality

# Run tests with example ligands
python rl_loop/test_rl_loop.pyThis script:
- Tests loading molecules from SDF files
- Tests SMILES conversion
- Tests ADMET-AI scoring (if available)
- Runs a full pipeline test
- Outputs results to `rl_loop/results/test_output.csv`

### Diversity Analysis

**`tanimoto_similarity.py`** - Analyze molecular diversity using Tanimoto similarity

# Analyze top-10 from scored CSV
python rl_loop/tanimoto_similarity.py \
    --input rl_loop/results/generated_from_refactor_scored.csv \
    --top_k 10 \
    --threshold 0.7 \
    --output rl_loop/results/top10_diversity.csv

# Analyze SDF file directly
python rl_loop/tanimoto_similarity.py \
    --input rl_loop/examples/generated_from_refactor.sdf \
    --top_k 20 \
    --threshold 0.6
**Options:**
- `--input, -i` - Input CSV (scored) or SDF file (required)
- `--top_k, -k` - Number of top molecules to analyze (default: 10)
- `--threshold, -t` - Similarity threshold for clustering (default: 0.7)
- `--fingerprint, -f` - Fingerprint type: `RDKit` or `Morgan` (default: RDKit)
- `--output, -o` - Output CSV path for cluster assignments (optional)
- `--quiet, -q` - Suppress progress messages

**Output:**
- Console report with diversity metrics
- CSV file with cluster assignments (if `--output` specified)
- Metrics: number of clusters, average pairwise similarity, cluster sizes

### REOS Filtering

**`reos_filter.py`** - Filter ligands using REOS (Rapid Elimination Of Swill) criteria

# Filter with default REOS criteria (strict)
python rl_loop/reos_filter.py \
    rl_loop/examples/generated_from_refactor.sdf \
    rl_loop/results/generated_ligands_filtered.sdf

# Allow 1 violation
python rl_loop/reos_filter.py \
    rl_loop/examples/generated_from_refactor.sdf \
    rl_loop/results/generated_ligands_filtered.sdf \
    --max_violations 1

# Custom criteria
python rl_loop/reos_filter.py \
    rl_loop/examples/generated_from_refactor.sdf \
    rl_loop/results/generated_ligands_filtered.sdf \
    --mw_min 250 --mw_max 450 \
    --logp_max 4.0 \
    --rot_bonds_max 7**Options:**
- `input_sdf` - Input SDF file (required, positional)
- `output_sdf` - Output filtered SDF file (required, positional)
- `--mw_min, --mw_max` - Molecular weight range (default: 200-500)
- `--logp_min, --logp_max` - LogP range (default: -5.0 to +5.0)
- `--hbd_min, --hbd_max` - H-bond donors range (default: 0-5)
- `--hba_min, --hba_max` - H-bond acceptors range (default: 0-10)
- `--charge_min, --charge_max` - Formal charge range (default: -2 to +2)
- `--rot_bonds_min, --rot_bonds_max` - Rotatable bonds range (default: 0-8)
- `--heavy_atoms_min, --heavy_atoms_max` - Heavy atom count range (default: 15-50)
- `--max_violations` - Maximum violations allowed (default: 0 = strict)
- `--quiet, -q` - Suppress output

**Default REOS Criteria:**
- Molecular weight: 200-500
- LogP: -5.0 to +5.0
- H-bond donors: 0-5
- H-bond acceptors: 0-10
- Formal charge: -2 to +2
- Rotatable bonds: 0-8
- Heavy atoms: 15-50

### Complete Pipeline Example

Full workflow with filtering and diversity analysis:

# 1. Generate ligands
python DiffSBDD/generate_ligands.py checkpoints/last_ckpt.ckpt \
    --pdbfile example/3rfm.pdb \
    --ref_ligand A:330 \
    --outfile rl_loop/examples/generated.sdf \
    --n_samples 100

# 2. Filter with REOS
python rl_loop/reos_filter.py \
    rl_loop/examples/generated.sdf \
    rl_loop/examples/generated_filtered.sdf

# 3. Score with ADMET-AI
python rl_loop/RL_loop.py \
    --input rl_loop/examples/generated_filtered.sdf \
    --output rl_loop/results/generated_scored.csv

# 4. Analyze diversity of top-10
python rl_loop/tanimoto_similarity.py \
    --input rl_loop/results/generated_scored.csv \
    --top_k 10 \
    --threshold 0.7 \
    --output rl_loop/results/top10_diversity.csv