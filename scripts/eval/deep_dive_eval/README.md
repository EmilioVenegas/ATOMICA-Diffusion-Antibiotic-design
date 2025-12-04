# Deep Dive Evaluation Pipeline

The "Three Judges" evaluation strategy for assessing generated molecules:

1. **Judge 1 (The Chemist)**: RDKit PAINS filter - detects toxic/problematic molecules
2. **Judge 2 (The Physicist)**: AutoDock Vina scoring - checks if molecules fit in the pocket without steric clashes
3. **Judge 3 (The AI)**: Boltz-2 Affinity - deep learning model predicts binding affinity

## Components

### `judge1_pains.py`
RDKit PAINS filter to identify problematic molecules.

**Usage:**
```bash
python scripts/eval/deep_dive_eval/judge1_pains.py \
    --input molecules.csv \
    --smiles-column smiles \
    --output pains_results.csv
```

### `judge2_vina.py`
AutoDock Vina scoring for protein-ligand docking.

**Requirements:**
- AutoDock Vina installed (`vina` command)
- AutoDockTools (`prepare_receptor4.py`, `prepare_ligand4.py`)
- OpenBabel (`obabel` command)

**Usage:**
```bash
python scripts/eval/deep_dive_eval/judge2_vina.py \
    --sdf ligands.sdf \
    --receptor protein.pdb \
    --output vina_scores.csv \
    --center 0 0 0 \
    --size 20 20 20 \
    --exhaustiveness 8
```

### `judge3_boltz.py`
Boltz-2 affinity prediction wrapper.

**Usage:**
```bash
python scripts/eval/deep_dive_eval/judge3_boltz.py \
    --input molecules.csv \
    --template template.yaml \
    --output-dir boltz_results/ \
    --accelerator cpu \
    --fast
```

### `consistency_plot.py`
Creates scatter plot of Boltz-2 affinity vs Vina score, colored by PAINS status.

**Interpretation:**
- **Bottom-right quadrant**: Ideal (high Boltz affinity + low Vina energy)
- **Red points**: PAINS alerts (problematic molecules)
- **Blue points**: Safe molecules

**Usage:**
```bash
python scripts/eval/deep_dive_eval/consistency_plot.py \
    --boltz-scores boltz_scores.csv \
    --vina-scores vina_scores.csv \
    --pains-status pains_results.csv \
    --output consistency_plot.png
```

### `interaction_heatmap.py`
Creates heatmap showing protein-ligand interactions for top molecules.

**Usage:**
```bash
python scripts/eval/deep_dive_eval/interaction_heatmap.py \
    --sdf complexes.sdf \
    --output heatmap.png \
    --top-n 50 \
    --residues His57 Ser195 Asp102
```

## Workflow

### Option 1: Reuse Boltz Results from `compare_benchmarks.py` (Recommended)

If you've already run `compare_benchmarks.py`, you can extract the Boltz scores and run the other judges:

```bash
# 1. Extract Boltz scores from existing results (e.g., from compare_benchmarks.py)
python scripts/eval/deep_dive_eval/judge3_boltz.py \
    --extract-from outputs/binders_results/summaries \
    --output boltz_scores.csv

# 2. Check PAINS on the same binders
python scripts/eval/deep_dive_eval/judge1_pains.py \
    --input binders.csv \
    --output pains_results.csv

# 3. Score with Vina on the same binders
python scripts/eval/deep_dive_eval/judge2_vina.py \
    --sdf binders.sdf \
    --receptor protein.pdb \
    --output vina_scores.csv

# 4. Create consistency plot
python scripts/eval/deep_dive_eval/consistency_plot.py \
    --boltz-scores boltz_scores.csv \
    --vina-scores vina_scores.csv \
    --pains-status pains_results.csv \
    --output consistency_plot.png

# 5. Create interaction heatmap
python scripts/eval/deep_dive_eval/interaction_heatmap.py \
    --sdf complexes.sdf \
    --output heatmap.png \
    --top-n 50
```

### Option 2: Run All Judges Independently

If you haven't run `compare_benchmarks.py`, you can run all three judges:

```bash
# 1. Check PAINS
python scripts/eval/deep_dive_eval/judge1_pains.py \
    --input generated_molecules.csv \
    --output pains_results.csv

# 2. Score with Vina
python scripts/eval/deep_dive_eval/judge2_vina.py \
    --sdf generated_ligands.sdf \
    --receptor protein.pdb \
    --output vina_scores.csv

# 3. Score with Boltz-2
python scripts/eval/deep_dive_eval/judge3_boltz.py \
    --input generated_molecules.csv \
    --template protein_template.yaml \
    --output-dir boltz_results/

# 4. Create consistency plot
python scripts/eval/deep_dive_eval/consistency_plot.py \
    --boltz-scores boltz_results/summaries/affinity_scores.json \
    --vina-scores vina_scores.csv \
    --pains-status pains_results.csv \
    --output consistency_plot.png
```

## Relationship to `compare_benchmarks.py`

- **`compare_benchmarks.py`**: Compares two datasets (e.g., binders vs decoys) using Boltz-2, creates comparison plots
- **`deep_dive_eval`**: Evaluates a single set of molecules using three different scoring methods (PAINS, Vina, Boltz-2)

You can reuse Boltz results from `compare_benchmarks.py` by using `judge3_boltz.py --extract-from` to avoid re-running expensive Boltz scoring.

## Notes

### Molecule ID Matching

When combining results from different judges, ensure molecule IDs match:

- **Boltz results**: Molecule IDs are like `sample_0001`, `sample_0002`, etc. (from sample directories)
- **PAINS results**: Uses SMILES as keys (or molecule_id if provided)
- **Vina results**: Uses `mol_0`, `mol_1`, etc. by default (from SDF order)

**To align IDs:**
1. Ensure your input CSV/SDF files have molecules in the same order
2. Or add explicit `molecule_id` columns that match across all files
3. The consistency plot will attempt to match by `molecule_id` column, or by index order if IDs don't match

### Other Notes

- The interaction heatmap currently uses simplified interaction detection. For production use, integrate with proper tools like PLIP or PyMOL.
- Vina scoring requires prepared PDBQT files. The script handles conversion automatically.
- Boltz-2 scoring can be slow; use `--fast` flag for quicker (lower accuracy) results.
- When reusing Boltz results from `compare_benchmarks.py`, the summaries directory is at `outputs/{dataset_name}_results/summaries/`

