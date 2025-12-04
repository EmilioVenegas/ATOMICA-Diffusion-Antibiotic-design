# 3PBQ Evaluation Inputs

This folder contains all necessary inputs for running the evaluation pipeline on 3PBQ generated binders.

## Files

- `3pbq_template.yaml`: Boltz-2 template YAML file for 3PBQ protein
  - Protein sequence: Chain A (493 residues)
  - Template uses `{{SMILES}}` placeholder that will be replaced during Boltz scoring
  
- `3pbq_imipenem_iter3_all_filtered_scored.csv`: Generated binders with SMILES and ADMET scores
  - Contains 50 molecules with their SMILES strings
  - Column: `smiles` contains the SMILES for each molecule
  
- `3PBQ.pdb`: Protein structure file for docking (Vina) and interaction analysis

- `3pbq_imipenem_iter3_all_filtered_scored_top50.sdf`: SDF file containing top 50 generated molecules
  - Used for Vina scoring (Judge 2)
  - Used for interaction heatmap analysis

## Usage

### For compare_benchmarks.py:
```bash
python scripts/eval/compare_benchmarks.py \
  --dataset1 3pbq_evaluation/inputs/3pbq_imipenem_iter3_all_filtered_scored.csv \
  --dataset2 <comparison_dataset.csv> \
  --template1 3pbq_evaluation/inputs/3pbq_template.yaml \
  --template2 <comparison_template.yaml> \
  --output-dir 3pbq_evaluation/outputs/compare_benchmarks
```

### For Three Judges Pipeline:

#### Judge 1 (PAINS):
```bash
python scripts/eval/deep_dive_eval/judge1_pains.py \
  --input 3pbq_evaluation/inputs/3pbq_imipenem_iter3_all_filtered_scored.csv \
  --smiles-column smiles \
  --output 3pbq_evaluation/outputs/pains_scores.csv
```

#### Judge 2 (Vina):
```bash
python scripts/eval/deep_dive_eval/judge2_vina.py \
  --sdf 3pbq_evaluation/inputs/3pbq_imipenem_iter3_all_filtered_scored_top50.sdf \
  --receptor 3pbq_evaluation/inputs/3PBQ.pdb \
  --output 3pbq_evaluation/outputs/vina_scores.csv
```

#### Judge 3 (Boltz-2):
```bash
python scripts/eval/deep_dive_eval/judge3_boltz.py \
  --input 3pbq_evaluation/inputs/3pbq_imipenem_iter3_all_filtered_scored.csv \
  --template 3pbq_evaluation/inputs/3pbq_template.yaml \
  --smiles-column smiles \
  --output-dir 3pbq_evaluation/outputs/boltz_scores
```

#### Consistency Plot:
```bash
python scripts/eval/deep_dive_eval/consistency_plot.py \
  --boltz-scores 3pbq_evaluation/outputs/boltz_scores/boltz_scores.csv \
  --vina-scores 3pbq_evaluation/outputs/vina_scores.csv \
  --pains-status 3pbq_evaluation/outputs/pains_scores.csv \
  --output 3pbq_evaluation/outputs/consistency_plot.png
```

#### Interaction Heatmap:
```bash
python scripts/eval/deep_dive_eval/interaction_heatmap.py \
  --sdf 3pbq_evaluation/inputs/3pbq_imipenem_iter3_all_filtered_scored_top50.sdf \
  --protein-pdb 3pbq_evaluation/inputs/3PBQ.pdb \
  --output 3pbq_evaluation/outputs/interaction_heatmap.png \
  --top-n 50
```

## Notes

- The YAML template uses `{{SMILES}}` placeholder which will be replaced by actual SMILES during Boltz scoring
- Make sure the SDF file contains docked complexes (protein-ligand complexes), not just ligands
- For Vina scoring, you may need to specify docking center and box size if not using default values

