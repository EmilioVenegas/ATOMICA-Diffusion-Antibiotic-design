#!/usr/bin/env python3
"""Judge 3: The AI - Boltz-2 affinity prediction (wrapper around existing functionality)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

# Change to project root and set up path
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
os.chdir(_project_root)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

# Only import if needed (for scoring mode)
try:
    from scripts.eval.boltz_random import run_random_affinity_workflow
except ImportError:
    # If import fails, we can still use extract mode
    run_random_affinity_workflow = None


def extract_boltz_scores(summary_dir: Path, output_csv: Path | None = None) -> pd.DataFrame:
    """Extract Boltz-2 affinity scores from a summary directory.
    
    Args:
        summary_dir: Path to directory containing affinity_scores.json
        output_csv: Optional path to save results as CSV
    
    Returns:
        DataFrame with molecule IDs and affinity scores
    """
    import json
    
    scores_path = summary_dir / "affinity_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"Could not find affinity scores at {scores_path}")
    
    with scores_path.open("r", encoding="utf-8") as f:
        scores = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for mol_id, score_dict in scores.items():
        rows.append({
            'molecule_id': mol_id,
            'boltz_affinity': score_dict.get('affinity_pred_value'),
            'boltz_probability': score_dict.get('affinity_probability_binary'),
        })
    
    df = pd.DataFrame(rows)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved Boltz scores to {output_csv}")
    
    return df


def score_molecules_with_boltz(
    smiles_list: list[str],
    template_yaml: Path,
    output_dir: Path,
    binder_id: str = "LIG",
    accelerator: str = "cpu",
    fast: bool = False,
) -> pd.DataFrame:
    """Score molecules using Boltz-2.
    
    Args:
        smiles_list: List of SMILES strings to score
        template_yaml: Path to Boltz template YAML file
        output_dir: Directory to save Boltz results
        binder_id: Ligand identifier in template
        accelerator: 'cpu' or 'gpu'
        fast: Use faster, lower-accuracy settings
    
    Returns:
        DataFrame with molecule IDs and Boltz scores
    """
    # Create temporary CSV with SMILES
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df_temp = pd.DataFrame({'smiles': smiles_list})
        df_temp.to_csv(f.name, index=False)
        temp_csv = Path(f.name)
    
    try:
        # Run Boltz workflow
        sampling_steps = 10 if fast else 25
        sampling_steps_affinity = 10 if fast else 50
        
        run_random_affinity_workflow(
            chemical_space=temp_csv,
            sample_size=len(smiles_list),
            column="smiles",
            seed=42,
            template_path=template_yaml,
            output_dir=output_dir,
            binder_id=binder_id,
            cache_dir=Path("~/.boltz").expanduser(),
            accelerator=accelerator,
            sampling_steps=sampling_steps,
            diffusion_samples=1,
            sampling_steps_affinity=sampling_steps_affinity,
            diffusion_samples_affinity=1,
            keep_inputs=False,
        )
        
        # Extract scores
        summary_dir = output_dir / "summaries"
        return extract_boltz_scores(summary_dir)
    
    finally:
        # Clean up temp file
        temp_csv.unlink()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Judge 3: Boltz-2 affinity prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from existing Boltz results (e.g., from compare_benchmarks.py):
  python judge3_boltz.py --extract-from outputs/binders_results/summaries --output boltz_scores.csv
  
  # Run new Boltz scoring:
  python judge3_boltz.py --input molecules.csv --template template.yaml --output-dir boltz_results/
        """
    )
    
    # Two modes: extract from existing results OR run new scoring
    parser.add_argument("--extract-from", type=Path, help="Extract scores from existing Boltz summaries directory (skip running Boltz)")
    parser.add_argument("--output", type=Path, help="Output CSV file (for extract mode)")
    
    # Arguments for running new Boltz scoring
    parser.add_argument("--input", type=Path, help="Input CSV file with SMILES column (for new scoring)")
    parser.add_argument("--template", type=Path, help="Boltz template YAML file (for new scoring)")
    parser.add_argument("--output-dir", type=Path, help="Output directory for Boltz results (for new scoring)")
    parser.add_argument("--smiles-column", default="smiles", help="Name of SMILES column")
    parser.add_argument("--binder-id", default="LIG", help="Ligand identifier in template")
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu", help="Hardware accelerator")
    parser.add_argument("--fast", action="store_true", help="Use faster, lower-accuracy settings")
    
    args = parser.parse_args()
    
    # Mode 1: Extract from existing results
    if args.extract_from:
        if not args.output:
            raise ValueError("--output is required when using --extract-from")
        
        scores_df = extract_boltz_scores(args.extract_from, args.output)
        
        print(f"\nBoltz-2 Scores Extracted:")
        print(f"  Total molecules: {len(scores_df)}")
        print(f"  Mean affinity: {scores_df['boltz_affinity'].mean():.4f}")
        print(f"  Mean probability: {scores_df['boltz_probability'].mean():.4f}")
    
    # Mode 2: Run new Boltz scoring
    elif args.input and args.template and args.output_dir:
        if run_random_affinity_workflow is None:
            raise ImportError("Cannot run Boltz scoring: scripts.eval.boltz_random not available. Use --extract-from mode instead.")
        
        df = pd.read_csv(args.input)
        smiles_list = df[args.smiles_column].tolist()
        
        scores_df = score_molecules_with_boltz(
            smiles_list,
            args.template,
            args.output_dir,
            binder_id=args.binder_id,
            accelerator=args.accelerator,
            fast=args.fast,
        )
        
        print(f"\nBoltz-2 Scoring Results:")
        print(f"  Total molecules scored: {len(scores_df)}")
        print(f"  Mean affinity: {scores_df['boltz_affinity'].mean():.4f}")
        print(f"  Mean probability: {scores_df['boltz_probability'].mean():.4f}")
    
    else:
        parser.error("Either --extract-from or (--input, --template, --output-dir) must be provided")

