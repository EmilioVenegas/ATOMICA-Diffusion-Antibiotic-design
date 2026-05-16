 python scripts/dock.py \
    --filtered_dir results/cond_C/filtered \
    --pdb_dir data/receptor_pdbs \
    --out results/cond_C/docking.csv \
    --condition cond_C --n_jobs 4


python scripts/evaluate.py \
    --sdf_dir results/baseline_A \
    --pdb_dir DiffSBDD/data/processed_expert_atomica/test \
    --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \
    --out results/baseline_A/metrics.json \
    --label A-baseline

This will:
  1. Compute chemistry metrics (validity, QED, SA, etc.) on all 100 pockets
  2. Run the full filter pipeline (valid → PoseBusters → QED≥0.5, SA≤5)
  3. Save filtered SDFs to results/baseline_A/filtered/ + manifest.csv
  4. Print a summary table and save metrics.json

  Before you can evaluate B, C, D you need to run inference first (the run_baseline.py step from earlier). Right now only
  baseline_A has SDF results, so that's the one to start with.

  One note: PoseBusters on 100 pockets × 100 molecules each takes a while (20–40 min depending on your CPU). If you want just
  the chemistry metrics fast to sanity-check, you can run without --pdb_dir:



python scripts/evaluate.py \
    --sdf_dir results/cond_B \
    --pdb_dir DiffSBDD/data/processed_expert_atomica/test \
    --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \
    --out results/cond_B/metrics.json \
    --label cond_B

python scripts/evaluate.py \
    --sdf_dir results/cond_C \
    --pdb_dir DiffSBDD/data/processed_expert_atomica/test \
    --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \
    --out results/cond_C/metrics.json \
    --label cond_C


python scripts/evaluate.py \
    --sdf_dir results/cond_D \
    --pdb_dir DiffSBDD/data/processed_expert_atomica/test \
    --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \
    --out results/cond_D/metrics.json \
    --label cond_D

for cond in baseline_A cond_B cond_C cond_D; do
    python scripts/evaluate.py \
      --sdf_dir results/$cond \
      --pdb_dir data/receptor_pdbs \
      --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \
      --out results/$cond/metrics.json \
      --label $cond \
      --sa_max 0.7
  done

python scripts/evaluate.py \
    --sdf_dir results/cond_B \
    --pdb_dir data/receptor_pdbs \
    --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \
    --out results/cond_B/metrics.json \
    --label cond_B
    --sa_max 0.7