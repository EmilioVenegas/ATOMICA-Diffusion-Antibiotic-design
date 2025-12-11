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
├── tanimoto_similarity.py   # Diversity analysis using Tanimoto similarity
├── reos_filter.py           # REOS filter for drug-likeness
├── drugbank_reference.py    # Compare ligands against DrugBank reference set
├── test_rl_loop.py          # Test suite for RL loop
├── INSTALLATION.md          # Installation instructions
├── QUICKSTART.md            # Quick start guide
├── SUMMARY.md               # Implementation summary
├── examples/                # Input ligand SDF files (generated externally)
│   ├── .gitkeep
│   └── generated_from_refactor.sdf      # Example generated ligands (committed manually)
└── results/                 # Scored output CSV files and SDF exports
    ├── .gitkeep
    └── generated_from_refactor_scored.csv  # Example scored results
└── plots/                   # DrugBank reference comparison plots
    └── .gitkeep


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
    -i rl_loop/examples/generated_from_refactor.sdf \
    -o rl_loop/results/generated_from_refactor_scored.csv \
    --save_top_sdf \
    --top_sdf_count 1

Which produces:

rl_loop/results/generated_from_refactor_scored.csv
rl_loop/results/generated_from_refactor_scored_top1.sdf  # Top molecule for optimization

3. (Optional) Filter with REOS criteria:

python rl_loop/reos_filter.py \
    rl_loop/examples/generated_from_refactor.sdf \
    rl_loop/examples/generated_filtered.sdf \
    --max_violations 1

4. Optimize top ADMET-scored ligand:

Use the top molecule from step 2 as the reference ligand for optimization:

python DiffSBDD/optimize.py \
      --checkpoint DiffSBDD/checkpoints/last_ckpt.ckpt \
      --pdbfile DiffSBDD/example/3rfm.pdb \
      --ref_ligand rl_loop/results/iter1_scored_top1_top1.sdf \
      --objective qed \
      --timesteps 50 \
      --population_size 10 \
      --evolution_steps 3 \
      --top_k 3 \
      --outfile rl_loop/examples/iter2_optimized.sdf

This optimizes the top ADMET-scored molecule for the specified objective (QED or SA) and produces:

rl_loop/examples/iter2_optimized.sdf

You can then re-score the optimized molecules and iterate:

python rl_loop/RL_loop.py \
    -i rl_loop/examples/iter2_optimized.sdf \
    -o rl_loop/results/iter2_scored.csv \
    --save_top_sdf


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

### DrugBank Reference Comparison

**`drugbank_reference.py`** - Compare generated ligands against DrugBank reference set (2,579 approved drugs)

This script uses ADMET-AI's built-in plotting functionality to visualize how generated molecules compare to known approved drugs in ADMET property space. The plots show:
- **Blue cloud**: DrugBank reference drugs (all 2,579 or filtered by ATC code)
- **Red stars**: Your generated/experimental molecules
- **Marginal histograms**: Distribution of each property

# Compare against all DrugBank drugs
python rl_loop/drugbank_reference.py \
    --input rl_loop/results/generated_from_refactor_scored.csv \
    --x_property "Human Intestinal Absorption" \
    --y_property "Clinical Toxicity" \
    --output rl_loop/plots/hia_vs_clintox_all.svg

# Compare against antibiotics only (ATC J01)
python rl_loop/drugbank_reference.py \
    --input rl_loop/results/generated_from_refactor_scored.csv \
    --atc_filter J01 \
    --x_property "Human Intestinal Absorption" \
    --y_property "Clinical Toxicity" \
    --output rl_loop/plots/hia_vs_clintox_antibiotics.svg

# Compare bioavailability vs hERG block
python rl_loop/drugbank_reference.py \
    --input rl_loop/results/generated_from_refactor_scored.csv \
    --x_property "Oral Bioavailability" \
    --y_property "hERG Block" \
    --output rl_loop/plots/bioavailability_vs_herg.svg

**Options:**
- `--input, -i` - Input scored CSV file from `RL_loop.py` (required)
- `--x_property` - ADMET property name for x-axis (required)
- `--y_property` - ADMET property name for y-axis (required)
- `--output, -o` - Output plot path (SVG format recommended, required)
- `--atc_filter` - Optional ATC code prefix to filter DrugBank (e.g., "J01" for antibiotics)
- `--quiet, -q` - Suppress progress messages

**ADMET Property Names:**
- Androgen Receptor (Ligand Binding Domain)
  Aqueous Solubility
  Aromatase
  Aryl Hydrocarbon Receptor
  Blood-Brain Barrier Penetration
  CYP1A2 Inhibition
  CYP2C19 Inhibition
  CYP2C9 Inhibition
  CYP2C9 Substrate
  CYP2D6 Inhibition
  CYP2D6 Substrate
  CYP3A4 Inhibition
  CYP3A4 Substrate
  Carcinogenicity
  Cell Effective Permeability
  Clinical Toxicity
  Drug Clearance (Hepatocyte)
  Drug Clearance (Microsome)
  Drug Induced Liver Injury
  Estrogen Receptor (Full Length)
  Estrogen Receptor (Ligand Binding Domain)
  Half Life
  Heat Shock Factor Response Element
  Human Intestinal Absorption
  Hydration Free Energy
  Hydrogen Bond Acceptors
  Hydrogen Bond Donors
  Lipinski Rule of 5
  Lipophilicity
  LogP
  Mitochondrial Membrane Potential
  Molecular Weight
  Mutagenicity
  Nuclear Factor (Erythroid-Derived 2)-Like 2/Antioxidant Responsive Element
  Oral Bioavailability
  P-glycoprotein Inhibition
  PAMPA Permeability
  Peroxisome Proliferator-Activated Receptor Gamma
  Plasma Protein Binding Rate
  Quantitative Estimate of Druglikeness (QED)
  Skin Reaction
  Stereo Centers
  Topological Polar Surface Area (TPSA)
  Tumor Protein p53
  Volume of Distribution at Steady State
  hERG Blocking

You can also use the exact CSV column names (e.g., `HIA_Hou`, `ClinTox`) directly.

**ATC Code Examples:**
- `J01` - Antibacterials for systemic use
- `L01` - Antineoplastic agents
- `C09` - Agents acting on the renin-angiotensin system
- `A10` - Drugs used in diabetes

**Output:**
- SVG plot file showing DrugBank reference (blue) vs. generated ligands (red)
- Same styling and marginal histograms as ADMET-AI web interface

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