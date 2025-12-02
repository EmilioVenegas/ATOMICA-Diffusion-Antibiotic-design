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

Next Steps (Roadmap)

These items were discussed but NOT implemented yet in this working branch.

High Priority

 Integrate SA score & Brenk structural alerts

 Define multi-objective composite scoring

 Stabilize scoring on large ligand sets

 Add unit tests for ranking

Medium Priority

 Weighted scoring configuration

 Property-specific filtering (e.g., hERG)

 Visualization tools for property distributions

Low Priority

 Logging / progress indicators

 Batch scoring optimization