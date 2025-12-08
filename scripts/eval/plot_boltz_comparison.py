#!/usr/bin/env python3
"""Plot comparison histograms from existing Boltz scoring outputs.

This script extracts scores from already-completed Boltz predictions
and generates comparison histograms without re-running Boltz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def calculate_pic50(affinity_log10_ic50_um: float) -> float:
    """Calculate pIC50 from log10 IC50 in µM.
    
    pIC50 = 6.0 - log10(IC50_µM)
    where log10(IC50_µM) = affinity_log10_IC50_uM
    """
    return 6.0 - affinity_log10_ic50_um


def collect_scores_from_boltz_output(
    output_dir: Path,
) -> pd.DataFrame:
    """Collect affinity scores from Boltz output directory.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing boltz_results_* subdirectory
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
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
    
    # Collect scores from each prediction
    for sample_dir in sorted(predictions_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        # Find affinity JSON
        affinity_json = sample_dir / f"affinity_{sample_dir.name}.json"
        if not affinity_json.exists():
            # Try alternative naming
            json_files = list(sample_dir.glob("affinity_*.json"))
            if json_files:
                affinity_json = json_files[0]
            else:
                print(f"⚠ Warning: No affinity JSON found for {sample_dir.name}, skipping...")
                continue
        
        # Load affinity data
        try:
            with affinity_json.open("r", encoding="utf-8") as f:
                affinity_data = json.load(f)
        except Exception as e:
            print(f"⚠ Warning: Error loading {affinity_json}: {e}, skipping...")
            continue
        
        # Extract molecule ID from sample name
        mol_id = sample_dir.name
        
        # Get affinity values
        affinity_pred_value = affinity_data.get("affinity_pred_value")
        affinity_probability = affinity_data.get("affinity_probability_binary")
        
        if affinity_pred_value is None:
            print(f"⚠ Warning: No affinity_pred_value in {affinity_json}, skipping...")
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


def generate_comparison_histograms(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    output_dir: Path,
    name_a: str,
    name_b: str,
) -> None:
    """Generate comparison histograms for two datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base metrics from Boltz
    base_metrics = ["affinity_log10_IC50_uM", "binder_probability", "pIC50", "pLDDT", "PAE"]
    
    # Molecular descriptors (if available)
    descriptor_metrics = ["mw", "alogp", "hbd", "hba"]
    
    # Combine all metrics
    metrics = base_metrics + descriptor_metrics
    
    # Filter out None values
    for metric in metrics:
        if metric not in scores_a.columns or metric not in scores_b.columns:
            print(f"⚠ Warning: Metric '{metric}' not found in one or both datasets. Skipping.")
            continue
        
        values_a = scores_a[metric].dropna().values
        values_b = scores_b[metric].dropna().values
        
        if len(values_a) == 0 or len(values_b) == 0:
            print(f"⚠ Warning: Not enough data for {metric} to generate histograms. Skipping.")
            continue
        
        # Create figure with side-by-side and overlay
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Side-by-side
        ax1, ax2 = axes
        
        counts_a, bins_a, _ = ax1.hist(values_a, bins=20, color="steelblue", edgecolor="white", alpha=0.7)
        ax1.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title(f"{name_a}\n(n={len(values_a)}, μ={values_a.mean():.3f})", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)
        
        try:
            kde_a = gaussian_kde(values_a)
            xs_a = np.linspace(values_a.min(), values_a.max(), 200)
            scale_a = counts_a.max() / kde_a(xs_a).max() if kde_a(xs_a).max() > 0 else 1.0
            ax1.plot(xs_a, kde_a(xs_a) * scale_a, color="navy", linewidth=1.5)
        except Exception:
            pass
        
        counts_b, bins_b, _ = ax2.hist(values_b, bins=20, color="coral", edgecolor="white", alpha=0.7)
        ax2.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title(f"{name_b}\n(n={len(values_b)}, μ={values_b.mean():.3f})", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        
        try:
            kde_b = gaussian_kde(values_b)
            xs_b = np.linspace(values_b.min(), values_b.max(), 200)
            scale_b = counts_b.max() / kde_b(xs_b).max() if kde_b(xs_b).max() > 0 else 1.0
            ax2.plot(xs_b, kde_b(xs_b) * scale_b, color="darkred", linewidth=1.5)
        except Exception:
            pass
        
        plt.tight_layout()
        side_by_side_path = output_dir / f"comparison_{metric}_side_by_side.png"
        plt.savefig(side_by_side_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {side_by_side_path}")
        
        # Overlay histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        counts_a_ov, bins_a_ov, _ = ax.hist(values_a, bins=20, label=name_a, color="steelblue", alpha=0.6, edgecolor="white")
        counts_b_ov, bins_b_ov, _ = ax.hist(values_b, bins=20, label=name_b, color="coral", alpha=0.6, edgecolor="white")
        
        # KS test
        ks_stat, ks_pval = ks_2samp(values_a, values_b)
        
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison\nKS D={ks_stat:.4f}, p={ks_pval:.4e}", 
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        
        try:
            kde_a_ov = gaussian_kde(values_a)
            kde_b_ov = gaussian_kde(values_b)
            xs = np.linspace(min(values_a.min(), values_b.min()),
                             max(values_a.max(), values_b.max()), 300)
            scale_a_ov = counts_a_ov.max() / kde_a_ov(xs).max() if kde_a_ov(xs).max() > 0 else 1.0
            scale_b_ov = counts_b_ov.max() / kde_b_ov(xs).max() if kde_b_ov(xs).max() > 0 else 1.0
            ax.plot(xs, kde_a_ov(xs) * scale_a_ov, color="navy", linewidth=1.5)
            ax.plot(xs, kde_b_ov(xs) * scale_b_ov, color="darkred", linewidth=1.5)
        except Exception:
            pass
        
        plt.tight_layout()
        overlay_path = output_dir / f"comparison_{metric}_overlay.png"
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {overlay_path}")


def generate_all_metrics_subplot(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    output_dir: Path,
    name_a: str,
    name_b: str,
) -> None:
    """Generate a single figure with subplots for all metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base metrics from Boltz
    base_metrics = ["affinity_log10_IC50_uM", "binder_probability", "pIC50", "pLDDT", "PAE"]
    
    # Molecular descriptors (if available)
    descriptor_metrics = ["mw", "alogp", "hbd", "hba"]
    
    # Combine all metrics
    metrics = base_metrics + descriptor_metrics
    
    # Collect available metrics with data
    available_metrics = []
    for metric in metrics:
        if metric not in scores_a.columns or metric not in scores_b.columns:
            continue
        
        values_a = scores_a[metric].dropna().values
        values_b = scores_b[metric].dropna().values
        
        if len(values_a) > 0 and len(values_b) > 0:
            available_metrics.append((metric, values_a, values_b))
    
    if not available_metrics:
        print("⚠ Warning: No metrics with sufficient data for combined subplot.")
        return
    
    # Calculate grid size (2 columns, rows as needed)
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    
    # Flatten axes array if needed
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, (metric, values_a, values_b) in enumerate(available_metrics):
        ax = axes_flat[idx]
        
        # Plot overlay histograms
        counts_a, bins_a, _ = ax.hist(values_a, bins=20, label=name_a, color="steelblue", alpha=0.6, edgecolor="white")
        counts_b, bins_b, _ = ax.hist(values_b, bins=20, label=name_b, color="coral", alpha=0.6, edgecolor="white")
        
        # KS test
        ks_stat, ks_pval = ks_2samp(values_a, values_b)
        
        # Add KDE overlays
        try:
            kde_a = gaussian_kde(values_a)
            kde_b = gaussian_kde(values_b)
            xs = np.linspace(min(values_a.min(), values_b.min()),
                           max(values_a.max(), values_b.max()), 300)
            scale_a = counts_a.max() / kde_a(xs).max() if kde_a(xs).max() > 0 else 1.0
            scale_b = counts_b.max() / kde_b(xs).max() if kde_b(xs).max() > 0 else 1.0
            ax.plot(xs, kde_a(xs) * scale_a, color="navy", linewidth=1.5)
            ax.plot(xs, kde_b(xs) * scale_b, color="darkred", linewidth=1.5)
        except Exception:
            pass
        
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{metric.replace('_', ' ').title()}\nKS D={ks_stat:.4f}, p={ks_pval:.4e}", 
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes_flat)):
        axes_flat[idx].axis("off")
    
    plt.tight_layout()
    all_metrics_path = output_dir / "comparison_all_metrics_subplot.png"
    plt.savefig(all_metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {all_metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from existing Boltz scoring outputs"
    )
    
    parser.add_argument(
        "--output-dir-a",
        type=Path,
        required=True,
        help="Path to first dataset's Boltz output directory (contains boltz_results_*)"
    )
    parser.add_argument(
        "--output-dir-b",
        type=Path,
        required=True,
        help="Path to second dataset's Boltz output directory (contains boltz_results_*)"
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
        help="Output directory for plots and scores"
    )
    parser.add_argument(
        "--descriptors-csv-a",
        type=Path,
        default=None,
        help="Optional CSV file with molecular descriptors for dataset A (must have molecule_id column)"
    )
    parser.add_argument(
        "--descriptors-csv-b",
        type=Path,
        default=None,
        help="Optional CSV file with molecular descriptors for dataset B (must have molecule_id column)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Plotting Boltz Score Comparison")
    print("=" * 60)
    print(f"Dataset A: {args.name_a} from {args.output_dir_a}")
    print(f"Dataset B: {args.name_b} from {args.output_dir_b}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print()
    
    # Collect scores
    print(f"Collecting scores from {args.name_a}...")
    scores_a = collect_scores_from_boltz_output(args.output_dir_a)
    print(f"✓ Collected {len(scores_a)} scores for {args.name_a}\n")
    
    print(f"Collecting scores from {args.name_b}...")
    scores_b = collect_scores_from_boltz_output(args.output_dir_b)
    print(f"✓ Collected {len(scores_b)} scores for {args.name_b}\n")
    
    # Merge molecular descriptors if provided
    if args.descriptors_csv_a and args.descriptors_csv_a.exists():
        print(f"Loading descriptors for {args.name_a} from {args.descriptors_csv_a}...")
        desc_a = pd.read_csv(args.descriptors_csv_a)
        if "molecule_id" in desc_a.columns:
            # Extract molecule ID from directory name (may have prefix like "decoys_dude_ZINC123")
            # Try to match by extracting the last part after underscore, or full match
            def extract_mol_id(dir_name):
                # If directory name contains underscores, try to extract the molecule ID
                # Common patterns: "target_mol_id" or just "mol_id"
                parts = dir_name.split("_")
                # Try the last part first (most likely to be the molecule ID)
                if len(parts) > 1:
                    # Check if last part looks like a molecule ID (starts with common prefixes)
                    last_part = parts[-1]
                    if last_part.startswith(("ZINC", "CHEMBL", "CHEM")):
                        return last_part
                # Otherwise return the full name
                return dir_name
            
            # Create a mapping from directory name to molecule ID
            scores_a["mol_id_for_merge"] = scores_a["molecule_id"].apply(extract_mol_id)
            
            # Try exact match first
            merged_a = scores_a.merge(desc_a, left_on="mol_id_for_merge", right_on="molecule_id", 
                                     how="left", suffixes=("", "_desc"))
            
            # Check how many matched (check if mw column exists first)
            if "mw" in merged_a.columns:
                matched_count = merged_a["mw"].notna().sum()
            else:
                matched_count = 0
            print(f"  Matched {matched_count}/{len(scores_a)} molecules with descriptors")
            
            if matched_count == 0:
                # Try matching by checking if CSV molecule_id is contained in directory name
                print("  Trying alternative matching: checking if CSV molecule_id is contained in directory name...")
                def find_matching_id(dir_name, desc_df):
                    for mol_id in desc_df["molecule_id"]:
                        if str(mol_id) in dir_name:
                            return mol_id
                    return None
                
                scores_a["matched_mol_id"] = scores_a["molecule_id"].apply(
                    lambda x: find_matching_id(x, desc_a)
                )
                merged_a = scores_a.merge(desc_a, left_on="matched_mol_id", right_on="molecule_id",
                                         how="left", suffixes=("", "_desc"))
                if "mw" in merged_a.columns:
                    matched_count = merged_a["mw"].notna().sum()
                else:
                    matched_count = 0
                print(f"  Matched {matched_count}/{len(scores_a)} molecules with alternative matching")
            
            # Drop helper columns
            scores_a = merged_a.drop(columns=["mol_id_for_merge", "matched_mol_id"], errors="ignore")
            print(f"✓ Merged descriptors for {args.name_a}\n")
        else:
            print(f"⚠ Warning: {args.descriptors_csv_a} does not have 'molecule_id' column, skipping descriptor merge.\n")
    
    if args.descriptors_csv_b and args.descriptors_csv_b.exists():
        print(f"Loading descriptors for {args.name_b} from {args.descriptors_csv_b}...")
        desc_b = pd.read_csv(args.descriptors_csv_b)
        if "molecule_id" in desc_b.columns:
            # Extract molecule ID from directory name (may have prefix like "decoys_dude_ZINC123")
            def extract_mol_id(dir_name):
                parts = dir_name.split("_")
                if len(parts) > 1:
                    last_part = parts[-1]
                    if last_part.startswith(("ZINC", "CHEMBL", "CHEM")):
                        return last_part
                return dir_name
            
            scores_b["mol_id_for_merge"] = scores_b["molecule_id"].apply(extract_mol_id)
            
            # Try exact match first
            merged_b = scores_b.merge(desc_b, left_on="mol_id_for_merge", right_on="molecule_id",
                                     how="left", suffixes=("", "_desc"))
            
            if "mw" in merged_b.columns:
                matched_count = merged_b["mw"].notna().sum()
            else:
                matched_count = 0
            print(f"  Matched {matched_count}/{len(scores_b)} molecules with descriptors")
            
            if matched_count == 0:
                # Try alternative matching
                print("  Trying alternative matching: checking if CSV molecule_id is contained in directory name...")
                def find_matching_id(dir_name, desc_df):
                    for mol_id in desc_df["molecule_id"]:
                        if str(mol_id) in dir_name:
                            return mol_id
                    return None
                
                scores_b["matched_mol_id"] = scores_b["molecule_id"].apply(
                    lambda x: find_matching_id(x, desc_b)
                )
                merged_b = scores_b.merge(desc_b, left_on="matched_mol_id", right_on="molecule_id",
                                         how="left", suffixes=("", "_desc"))
                if "mw" in merged_b.columns:
                    matched_count = merged_b["mw"].notna().sum()
                else:
                    matched_count = 0
                print(f"  Matched {matched_count}/{len(scores_b)} molecules with alternative matching")
            
            scores_b = merged_b.drop(columns=["mol_id_for_merge", "matched_mol_id"], errors="ignore")
            print(f"✓ Merged descriptors for {args.name_b}\n")
        else:
            print(f"⚠ Warning: {args.descriptors_csv_b} does not have 'molecule_id' column, skipping descriptor merge.\n")
    
    # Save scores to CSV
    scores_a_path = output_dir / f"{args.name_a}_scores.csv"
    scores_b_path = output_dir / f"{args.name_b}_scores.csv"
    scores_a.to_csv(scores_a_path, index=False)
    scores_b.to_csv(scores_b_path, index=False)
    print(f"✓ Saved scores: {scores_a_path}")
    print(f"✓ Saved scores: {scores_b_path}\n")
    
    # Generate histograms
    print("Generating comparison histograms...")
    histograms_dir = output_dir / "histograms"
    generate_comparison_histograms(
        scores_a, scores_b, histograms_dir, args.name_a, args.name_b
    )
    
    # Generate combined subplot with all metrics
    print("Generating combined subplot with all metrics...")
    generate_all_metrics_subplot(
        scores_a, scores_b, histograms_dir, args.name_a, args.name_b
    )
    
    # Save summary statistics
    summary = {
        "dataset_a": args.name_a,
        "dataset_b": args.name_b,
        "n_a": int(len(scores_a)),
        "n_b": int(len(scores_b)),
        "statistics": {}
    }
    
    # Base metrics + descriptors
    all_metrics = ["affinity_log10_IC50_uM", "binder_probability", "pIC50", "pLDDT", "PAE", "mw", "alogp", "hbd", "hba"]
    
    for metric in all_metrics:
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
    print("Plotting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

