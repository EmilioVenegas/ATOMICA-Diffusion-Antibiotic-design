# RL Loop for ADMET-AI Guided Ligand Optimization

## Purpose

This directory contains the initial implementation of a reinforcement learning (RL) feedback loop for optimizing generated ligands. The pipeline:

1. Loads ligand molecules (from SDF files)
2. Converts them to SMILES strings
3. Scores them using ADMET-AI predictions
4. Ranks molecules by their predicted ADMET properties

This is the **first step** toward a full RL optimization loop that will guide the diffusion model to generate molecules with improved drug-like properties.

## Directory Structure

```
rl_loop/
├── README.md              # This file
├── RL_loop.py            # Main RL loop implementation
├── examples/             # Example input ligands (SDF files)
│   └── .gitkeep
└── results/              # Scored and ranked output
    └── .gitkeep
```

## Requirements

Install required packages:

```bash
# If using the DiffSBDD conda environment:
conda activate diffsbdd
pip install admet-ai pyarrow

# Or install standalone:
pip install rdkit-pypi admet-ai pandas pyarrow
```

## Usage

### Basic Example

Score example ligands from DiffSBDD:

```bash
python rl_loop/RL_loop.py \
  --input_dir DiffSBDD/example \
  --output results/scored_ligands.csv
```

### Score Generated Ligands

After running `generate_ligands.py`:

```bash
# 1. Generate ligands (from DiffSBDD directory)
python generate_ligands.py checkpoints/model.ckpt \
  --pdbfile example/3rfm.pdb \
  --outfile ../rl_loop/examples/generated_ligands.sdf \
  --ref_ligand A:330 \
  --n_samples 100

# 2. Score them
python rl_loop/RL_loop.py \
  --input ../rl_loop/examples/generated_ligands.sdf \
  --output results/scored_generated.csv
```

### Command-Line Options

```
--input, -i          Input SDF file or directory containing SDF files
--output, -o         Output CSV file with scores and rankings
--top_k, -k          Number of top molecules to save (default: all)
--admet_props        Comma-separated list of ADMET properties to use for ranking
                     (default: all available)
```

## Output Format

The output CSV contains:

| Column | Description |
|--------|-------------|
| `molecule_id` | Unique identifier (filename + mol index) |
| `smiles` | SMILES string representation |
| `rank` | Overall rank (1 = best) |
| `composite_score` | Weighted average of ADMET properties |
| `[ADMET properties]` | Individual ADMET-AI predictions (e.g., Solubility, Clearance, hERG, etc.) |

## Next Steps

Future enhancements for the RL loop:

1. **Scoring Integration**: Add SA score and Brenk structural alerts
2. **Multi-objective Optimization**: Define composite reward function
3. **Diffusion Model Feedback**: Use scores to guide generation
4. **Active Learning**: Iteratively generate and score batches
5. **Visualization**: Plot property distributions and trends

## Example Workflow

```python
from rl_loop.RL_loop import RLLoop

# Initialize
rl = RLLoop()

# Load and score molecules
molecules = rl.load_molecules_from_sdf("examples/ligands.sdf")
smiles = rl.convert_to_smiles(molecules)
scores = rl.score_with_admet(smiles)

# Rank and save
ranked = rl.rank_by_score(scores)
rl.save_results(ranked, "results/output.csv")
```

## TODO / Known Issues

### High Priority

1. **Dependency Resolution & Dockerization**
   - [ ] Resolve torch-scatter compatibility issues with PyTorch 2.0.1 + CUDA 11.8
   - [ ] Create Docker container for unified DiffSBDD + RL loop environment
   - [ ] Ensure `generate_ligands.py` can run reliably to produce ligand distributions
   - [ ] Test end-to-end pipeline: `generate_ligands.py` → SDF → `RL_loop.py`
   - [ ] Document Windows-specific installation workarounds

2. **Ranking Validation**
   - [ ] Validate composite score ranking aligns with expected drug-likeness on larger molecule sets
   - [ ] Test ranking consistency across different ADMET property combinations
   - [ ] Compare rankings with expert-curated reference sets
   - [ ] Add unit tests for ranking logic with known good/bad molecules

### Medium Priority

3. **Scoring Enhancements**
   - [ ] Integrate SA (Synthetic Accessibility) score
   - [ ] Add Brenk structural alerts detection
   - [ ] Implement weighted composite scoring (custom property importance)
   - [ ] Add property-specific filtering (e.g., exclude high hERG risk)

4. **Integration**
   - [ ] Direct integration with `generate_ligands.py` (avoid intermediate SDF files)
   - [ ] Real-time scoring during generation
   - [ ] Feedback loop to guide diffusion model sampling

### Low Priority

5. **Performance & Usability**
   - [ ] Batch processing for large ligand sets
   - [ ] Progress bars and logging improvements
   - [ ] Visualization of property distributions
   - [ ] Export top molecules back to SDF format


## References

- **DiffSBDD**: Structure-based diffusion model for drug design
- **ADMET-AI**: Deep learning for ADMET property prediction
- **RDKit**: Cheminformatics toolkit for molecule processing

