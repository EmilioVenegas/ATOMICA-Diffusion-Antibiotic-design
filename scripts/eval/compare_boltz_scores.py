#!/usr/bin/env python3
"""Run Boltz scoring on two datasets and collect scores.

This script runs boltz_scoring.py on two CSV files, collects scores,
calculates pIC50, and saves summary statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.eval.boltz_scoring import run_boltz_scoring


def calculate_pic50(affinity_log10_ic50_um: float) -> float:
    """Calculate pIC50 from log10 IC50 in µM.
    
    pIC50 = 6.0 - log10(IC50_µM)
    where log10(IC50_µM) = affinity_log10_IC50_uM
    """
    return 6.0 - affinity_log10_ic50_um


def collect_scores_from_boltz_output(
    output_dir: Path,
    yaml_paths: list[Path],
) -> pd.DataFrame:
    """Collect affinity scores from Boltz output directory.
    
    Returns a DataFrame with columns:
    - molecule_id
    - affinity_log10_IC50_uM (renamed from affinity_pred_value)
    - binder_probability (renamed from affinity_probability_binary)
    - pIC50 (calculated)
    - pLDDT, PAE (if available)
    """
    scores_list = []
    
    # Find the boltz_results directory
    boltz_results_dir = None
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("boltz_results"):
            boltz_results_dir = item
            break
    
    if boltz_results_dir is None:
        raise FileNotFoundError(f"No boltz_results directory found in {output_dir}")
    
    predictions_dir = boltz_results_dir / "predictions"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    # Map YAML paths to molecule IDs
    yaml_to_id = {}
    for yaml_path in yaml_paths:
        # Extract molecule ID from YAML filename
        # Format: target_name_molecule_id.yaml
        stem = yaml_path.stem
        parts = stem.split("_", 1)
        if len(parts) > 1:
            mol_id = parts[1]
        else:
            mol_id = stem
        yaml_to_id[yaml_path.name] = mol_id
    
    # Collect scores from each prediction
    for sample_dir in sorted(predictions_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        # Find affinity JSON
        affinity_json = sample_dir / f"affinity_{sample_dir.name}.json"
        if not affinity_json.exists():
            continue
        
        # Load affinity data
        with affinity_json.open("r", encoding="utf-8") as f:
            affinity_data = json.load(f)
        
        # Extract molecule ID from sample name or YAML mapping
        mol_id = sample_dir.name
        
        # Get affinity values
        affinity_pred_value = affinity_data.get("affinity_pred_value")
        affinity_probability = affinity_data.get("affinity_probability_binary")
        
        if affinity_pred_value is None:
            continue
        
        # Build score record
        record = {
            "molecule_id": mol_id,
            "affinity_log10_IC50_uM": float(affinity_pred_value),
            "binder_probability": float(affinity_probability) if affinity_probability is not None else None,
        }
        
        # Calculate pIC50
        record["pIC50"] = calculate_pic50(record["affinity_log10_IC50_uM"])
        
        # Add confidence metrics if available
        confidence_json = sample_dir / f"confidence_{sample_dir.name}_model_0.json"
        if confidence_json.exists():
            try:
                with confidence_json.open("r", encoding="utf-8") as f:
                    confidence_data = json.load(f)
                
                if "complex_plddt" in confidence_data:
                    record["pLDDT"] = float(confidence_data["complex_plddt"])
                if "complex_pde" in confidence_data:
                    record["PAE"] = float(confidence_data["complex_pde"])
            except Exception:
                pass
        
        scores_list.append(record)
    
    if not scores_list:
        raise ValueError(f"No scores found in {predictions_dir}")
    
    return pd.DataFrame(scores_list)


def main():
    parser = argparse.ArgumentParser(
        description="Run Boltz scoring on two datasets and collect scores"
    )
    
    parser.add_argument(
        "--protein-pdb",
        type=Path,
        required=True,
        help="Path to PDB file"
    )
    parser.add_argument(
        "--csv-a",
        type=Path,
        required=True,
        help="Path to first CSV file"
    )
    parser.add_argument(
        "--csv-b",
        type=Path,
        required=True,
        help="Path to second CSV file"
    )
    parser.add_argument(
        "--name-a",
        type=str,
        required=True,
        help="Name for first dataset"
    )
    parser.add_argument(
        "--name-b",
        type=str,
        required=True,
        help="Name for second dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Hardware accelerator"
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="Column name containing SMILES"
    )
    parser.add_argument(
        "--name-col",
        default="molecule_id",
        help="Column name for molecule identifiers"
    )
    parser.add_argument(
        "--skip-boltz",
        action="store_true",
        help="Skip running Boltz and extract scores from existing results directories"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_a = output_dir / f"{args.name_a}_results"
    output_dir_b = output_dir / f"{args.name_b}_results"
    yamls_dir_a = output_dir_a / "yamls"
    yamls_dir_b = output_dir_b / "yamls"
    
    print("=" * 60)
    print("Running Boltz Scoring on Two Datasets")
    print("=" * 60)
    print(f"Dataset A: {args.name_a}")
    print(f"Dataset B: {args.name_b}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print()
    
    # Import functions
    from scripts.eval.boltz_scoring import make_yamls_from_protein_and_csv
    from scripts.eval.get_protein_sequence import get_protein_sequence_from_pdb
    
    if args.skip_boltz:
        print("⚠ Skipping Boltz scoring - extracting from existing results...")
        # Check that results directories exist
        if not output_dir_a.exists():
            raise FileNotFoundError(f"Results directory not found: {output_dir_a}. Cannot skip Boltz scoring.")
        if not output_dir_b.exists():
            raise FileNotFoundError(f"Results directory not found: {output_dir_b}. Cannot skip Boltz scoring.")
        
        # Get YAML paths (they should already exist)
        yaml_paths_a = list(yamls_dir_a.glob("*.yaml")) if yamls_dir_a.exists() else []
        yaml_paths_b = list(yamls_dir_b.glob("*.yaml")) if yamls_dir_b.exists() else []
        print(f"✓ Found {len(yaml_paths_a)} YAML files for {args.name_a}")
        print(f"✓ Found {len(yaml_paths_b)} YAML files for {args.name_b}\n")
    else:
        # Import functions
        from scripts.eval.boltz_scoring import make_yamls_from_protein_and_csv
        from scripts.eval.get_protein_sequence import get_protein_sequence_from_pdb
        
        # Get protein sequence
        protein_sequence = get_protein_sequence_from_pdb(args.protein_pdb)
        
        # Generate YAMLs and run scoring for dataset A
        print(f"Running Boltz scoring for {args.name_a}...")
        run_boltz_scoring(
            ligands_csv_path=args.csv_a,
            output_dir=output_dir_a,
            yamls_dir=yamls_dir_a,
            protein_pdb_path=args.protein_pdb,
            accelerator=args.accelerator,
            target_name=args.name_a.lower(),
            smiles_col=args.smiles_col,
            name_col=args.name_col,
        )
        
        # Get YAML paths after generation
        yaml_paths_a = list(yamls_dir_a.glob("*.yaml"))
        print(f"✓ {args.name_a} scoring complete\n")
        
        # Generate YAMLs and run scoring for dataset B
        print(f"Running Boltz scoring for {args.name_b}...")
        run_boltz_scoring(
            ligands_csv_path=args.csv_b,
            output_dir=output_dir_b,
            yamls_dir=yamls_dir_b,
            protein_pdb_path=args.protein_pdb,
            accelerator=args.accelerator,
            target_name=args.name_b.lower(),
            smiles_col=args.smiles_col,
            name_col=args.name_col,
        )
        
        # Get YAML paths after generation
        yaml_paths_b = list(yamls_dir_b.glob("*.yaml"))
        print(f"✓ {args.name_b} scoring complete\n")
    
    # Collect scores
    print("Collecting scores...")
    scores_a = collect_scores_from_boltz_output(output_dir_a, yaml_paths_a)
    scores_b = collect_scores_from_boltz_output(output_dir_b, yaml_paths_b)
    
    print(f"✓ Collected {len(scores_a)} scores for {args.name_a}")
    print(f"✓ Collected {len(scores_b)} scores for {args.name_b}\n")
    
    # Save scores to CSV
    scores_a_path = output_dir / f"{args.name_a}_scores.csv"
    scores_b_path = output_dir / f"{args.name_b}_scores.csv"
    scores_a.to_csv(scores_a_path, index=False)
    scores_b.to_csv(scores_b_path, index=False)
    print(f"✓ Saved scores: {scores_a_path}")
    print(f"✓ Saved scores: {scores_b_path}\n")
    
    # Save summary statistics
    summary = {
        "dataset_a": args.name_a,
        "dataset_b": args.name_b,
        "n_a": int(len(scores_a)),
        "n_b": int(len(scores_b)),
        "statistics": {}
    }
    
    for metric in ["affinity_log10_IC50_uM", "binder_probability", "pIC50", "pLDDT", "PAE"]:
        if metric in scores_a.columns and metric in scores_b.columns:
            values_a = scores_a[metric].dropna().values
            values_b = scores_b[metric].dropna().values
            
            if len(values_a) > 0 and len(values_b) > 0:
                ks_stat, ks_pval = ks_2samp(values_a, values_b)
                summary["statistics"][metric] = {
                    "mean_a": float(values_a.mean()),
                    "mean_b": float(values_b.mean()),
                    "std_a": float(values_a.std()),
                    "std_b": float(values_b.std()),
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pval),
                }
    
    summary_path = output_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}\n")
    
    print("=" * 60)
    print("Boltz scoring complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

