#!/usr/bin/env python3
"""Run Boltz-2 scoring and generate histograms on MIT cluster."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Change to project root and set up path
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
os.chdir(_project_root)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

from scripts.eval.boltz_random import run_random_affinity_workflow


def generate_histogram(
    scores_path: Path,
    output_path: Path,
    title: str = "Boltz-2 Affinity Distribution",
    bins: int | str = "auto",
) -> None:
    """Generate and save a histogram of affinity scores."""
    with scores_path.open("r", encoding="utf-8") as handle:
        affinity_scores = json.load(handle)

    values = np.array([entry["affinity_pred_value"] for entry in affinity_scores.values()], dtype=float)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, color="steelblue", edgecolor="white", alpha=0.7)
    plt.xlabel("affinity_pred_value", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)

    # Add statistics text box
    stats_text = f"n = {len(values)}\n"
    stats_text += f"μ = {values.mean():.3f}\n"
    stats_text += f"σ = {values.std():.3f}\n"
    stats_text += f"min = {values.min():.3f}\n"
    stats_text += f"max = {values.max():.3f}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Boltz-2 scoring on cluster and generate histograms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chemical-space", type=Path, required=True, help="Path to chemical space file (CSV/SMI/TXT)")
    parser.add_argument("--column", default="smiles", help="Column name for SMILES in CSV/TSV")
    parser.add_argument("--sample-size", type=int, required=True, help="Number of molecules to sample and score")
    parser.add_argument("--template", type=Path, help="Template YAML for Boltz (optional)")
    parser.add_argument("--binder-id", default="LIG", help="Ligand/binder identifier")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.boltz").expanduser(), help="Boltz cache directory")
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Hardware accelerator (use gpu on cluster)",
    )
    parser.add_argument("--sampling-steps", type=int, default=25, help="Diffusion sampling steps for structure")
    parser.add_argument("--diffusion-samples", type=int, default=1, help="Number of diffusion samples")
    parser.add_argument("--sampling-steps-affinity", type=int, default=50, help="Sampling steps for affinity head")
    parser.add_argument("--diffusion-samples-affinity", type=int, default=1, help="Diffusion samples for affinity")
    parser.add_argument("--keep-inputs", action="store_true", help="Keep generated YAML input files")
    parser.add_argument("--histogram-bins", type=int, help="Number of histogram bins (default: auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Boltz-2 Cluster Scoring Job")
    print("=" * 60)
    print(f"Chemical space: {args.chemical_space}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Accelerator: {args.accelerator}")
    print("=" * 60)
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run Boltz-2 scoring workflow
    summary = run_random_affinity_workflow(
        chemical_space=args.chemical_space,
        sample_size=args.sample_size,
        column=args.column,
        seed=args.seed,
        template_path=args.template,
        output_dir=args.output_dir,
        binder_id=args.binder_id,
        cache_dir=args.cache_dir,
        accelerator=args.accelerator,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        sampling_steps_affinity=args.sampling_steps_affinity,
        diffusion_samples_affinity=args.diffusion_samples_affinity,
        keep_inputs=args.keep_inputs,
    )

    # Generate histogram
    scores_path = args.output_dir / "summaries" / "affinity_scores.json"
    if scores_path.exists():
        histogram_path = args.output_dir / "summaries" / "affinity_histogram.png"
        bins = args.histogram_bins if args.histogram_bins else "auto"
        generate_histogram(scores_path, histogram_path, bins=bins)
    else:
        print(f"Warning: No affinity scores found at {scores_path}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("Job completed successfully!")
    print("=" * 60)
    print(f"Results directory: {args.output_dir}")
    print(f"Summary JSON: {args.output_dir / 'summaries' / 'affinity_summary.json'}")
    print(f"Histogram: {args.output_dir / 'summaries' / 'affinity_histogram.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

