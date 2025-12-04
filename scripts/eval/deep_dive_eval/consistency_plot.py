#!/usr/bin/env python3
"""Consistency Plot: Boltz-2 Affinity vs Vina Score with PAINS status coloring."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_boltz_scores(boltz_path: Path) -> pd.DataFrame:
    """Load Boltz scores from CSV or JSON file.
    
    Args:
        boltz_path: Path to CSV file or JSON file (from Boltz summaries)
    
    Returns:
        DataFrame with 'molecule_id' and 'boltz_affinity' columns
    """
    if boltz_path.suffix == '.json':
        # Load from Boltz JSON format
        import json
        
        with boltz_path.open("r", encoding="utf-8") as f:
            scores = json.load(f)
        
        # Convert to DataFrame
        rows = []
        for mol_id, score_dict in scores.items():
            rows.append({
                'molecule_id': mol_id,
                'boltz_affinity': score_dict.get('affinity_pred_value'),
                'boltz_probability': score_dict.get('affinity_probability_binary'),
            })
        
        return pd.DataFrame(rows)
    else:
        # Load from CSV
        return pd.read_csv(boltz_path)


def create_consistency_plot(
    boltz_scores: pd.DataFrame | Path,
    vina_scores: pd.DataFrame | Path,
    pains_status: pd.DataFrame | Path,
    output_path: Path,
    title: str = "Consistency Plot: Boltz-2 vs Vina",
) -> None:
    """Create scatter plot of Boltz-2 affinity vs Vina score with PAINS coloring.
    
    Args:
        boltz_scores: DataFrame or Path to CSV/JSON with 'molecule_id' and 'boltz_affinity' columns
        vina_scores: DataFrame or Path to CSV with 'molecule_id' and 'vina_score' columns
        pains_status: DataFrame or Path to CSV with 'molecule_id' (or SMILES) and 'pains_alert' column
        output_path: Path to save the plot
        title: Plot title
    """
    # Load data if paths are provided
    if isinstance(boltz_scores, Path):
        boltz_scores = load_boltz_scores(boltz_scores)
    if isinstance(vina_scores, Path):
        vina_scores = pd.read_csv(vina_scores)
    if isinstance(pains_status, Path):
        pains_status = pd.read_csv(pains_status)
    # Merge dataframes
    df = boltz_scores.merge(vina_scores, on='molecule_id', how='inner')
    
    # Handle PAINS status - might be keyed by SMILES or molecule_id
    if 'smiles' in pains_status.columns:
        # If we have SMILES, we need to match differently
        # For now, assume pains_status has molecule_id or can be matched by index
        if 'molecule_id' not in pains_status.columns:
            # Assume same order
            df['pains_alert'] = pains_status['pains_alert'].values[:len(df)]
        else:
            df = df.merge(pains_status[['molecule_id', 'pains_alert']], on='molecule_id', how='left')
    else:
        df = df.merge(pains_status, on='molecule_id', how='left')
    
    # Fill missing PAINS status as False (safe)
    df['pains_alert'] = df['pains_alert'].fillna(False)
    
    # Filter out NaN values
    df = df.dropna(subset=['boltz_affinity', 'vina_score'])
    
    if len(df) == 0:
        raise ValueError("No valid data points after merging and filtering")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate PAINS alerts and safe molecules
    pains_df = df[df['pains_alert'] == True]
    safe_df = df[df['pains_alert'] == False]
    
    # Plot safe molecules in blue
    if len(safe_df) > 0:
        ax.scatter(
            safe_df['boltz_affinity'],
            safe_df['vina_score'],
            c='steelblue',
            alpha=0.6,
            s=50,
            label=f'Safe (n={len(safe_df)})',
            edgecolors='white',
            linewidth=0.5,
        )
    
    # Plot PAINS alerts in red
    if len(pains_df) > 0:
        ax.scatter(
            pains_df['boltz_affinity'],
            pains_df['vina_score'],
            c='coral',
            alpha=0.6,
            s=50,
            label=f'PAINS Alert (n={len(pains_df)})',
            edgecolors='white',
            linewidth=0.5,
        )
    
    # Add quadrant lines (optional: highlight ideal region)
    # Ideal: high Boltz affinity (right) + low Vina energy (bottom)
    ax.axvline(x=df['boltz_affinity'].median(), color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=df['vina_score'].median(), color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Boltz-2 Affinity (Predicted pKd)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vina Score (Minimization Energy, kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add interpretation text
    interpretation = (
        "Ideal region: Bottom-right quadrant\n"
        "(High Boltz affinity + Low Vina energy)\n"
        "Red = PAINS alert (problematic), Blue = Safe"
    )
    ax.text(
        0.02, 0.98, interpretation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Consistency plot saved to {output_path}")
    print(f"  Total points: {len(df)}")
    print(f"  Safe molecules: {len(safe_df)}")
    print(f"  PAINS alerts: {len(pains_df)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create consistency plot: Boltz-2 vs Vina with PAINS colors")
    parser.add_argument("--boltz-scores", type=Path, required=True, help="CSV or JSON (from Boltz summaries) with Boltz scores")
    parser.add_argument("--vina-scores", type=Path, required=True, help="CSV with Vina scores (molecule_id, vina_score)")
    parser.add_argument("--pains-status", type=Path, required=True, help="CSV with PAINS status (molecule_id or smiles, pains_alert)")
    parser.add_argument("--output", type=Path, required=True, help="Output plot path")
    parser.add_argument("--title", default="Consistency Plot: Boltz-2 vs Vina", help="Plot title")
    
    args = parser.parse_args()
    
    create_consistency_plot(args.boltz_scores, args.vina_scores, args.pains_status, args.output, args.title)

