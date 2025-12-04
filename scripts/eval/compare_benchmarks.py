#!/usr/bin/env python3
"""Run Boltz scoring on two datasets and compare their affinity distributions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.eval.boltz_random import run_random_affinity_workflow
from scripts.eval.evaluate import ks_test_results, load_affinity_values


def run_comparison(
    dataset_a_path: Path,
    dataset_a_name: str,
    dataset_b_path: Path,
    dataset_b_name: str,
    template_path: Path,
    output_base_dir: Path,
    binder_id: str = "LIG",
    sample_size: int | None = None,
    seed: int = 42,
    accelerator: str = "cpu",
    fast: bool = False,
) -> None:
    """Run Boltz scoring on two datasets and compare results.
    
    Args:
        dataset_a_path: Path to first CSV file
        dataset_a_name: Name for first dataset (for labeling)
        dataset_b_path: Path to second CSV file
        dataset_b_name: Name for second dataset (for labeling)
        template_path: Path to Boltz template YAML
        output_base_dir: Base directory for outputs
        binder_id: Ligand identifier in template
        sample_size: Number of molecules to sample (None = all)
        seed: Random seed
        accelerator: 'cpu' or 'gpu'
    """
    print("=" * 60)
    print("BENCHMARK COMPARISON: Boltz-2 Affinity Scoring")
    print("=" * 60)
    print()
    
    # Create output directories
    output_dir_a = output_base_dir / f"{dataset_a_name}_results"
    output_dir_b = output_base_dir / f"{dataset_b_name}_results"
    comparison_dir = output_base_dir / "comparison"
    
    output_dir_a.mkdir(parents=True, exist_ok=True)
    output_dir_b.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose sampling parameters (fast vs default)
    if fast:
        sampling_steps = 10
        sampling_steps_affinity = 10
    else:
        sampling_steps = 25
        sampling_steps_affinity = 50
    
    # Step 1: Run Boltz scoring on dataset A
    print("=" * 60)
    print(f"STEP 1: Scoring {dataset_a_name}")
    print("=" * 60)
    summary_a = run_random_affinity_workflow(
        chemical_space=dataset_a_path,
        sample_size=sample_size or 1000,  # Use all if not specified
        column="smiles",
        seed=seed,
        template_path=template_path,
        output_dir=output_dir_a,
        binder_id=binder_id,
        cache_dir=Path("~/.boltz").expanduser(),
        accelerator=accelerator,
        sampling_steps=sampling_steps,
        diffusion_samples=1,
        sampling_steps_affinity=sampling_steps_affinity,
        diffusion_samples_affinity=1,
        keep_inputs=False,
    )
    print(f"✓ {dataset_a_name} scoring complete")
    print()
    
    # Step 2: Run Boltz scoring on dataset B
    print("=" * 60)
    print(f"STEP 2: Scoring {dataset_b_name}")
    print("=" * 60)
    summary_b = run_random_affinity_workflow(
        chemical_space=dataset_b_path,
        sample_size=sample_size or 1000,
        column="smiles",
        seed=seed + 1,  # Different seed for dataset B
        template_path=template_path,
        output_dir=output_dir_b,
        binder_id=binder_id,
        cache_dir=Path("~/.boltz").expanduser(),
        accelerator=accelerator,
        sampling_steps=sampling_steps,
        diffusion_samples=1,
        sampling_steps_affinity=sampling_steps_affinity,
        diffusion_samples_affinity=1,
        keep_inputs=False,
    )
    print(f"✓ {dataset_b_name} scoring complete")
    print()
    
    # Step 3: Compare results
    print("=" * 60)
    print("STEP 3: Statistical Comparison")
    print("=" * 60)
    
    summary_dir_a = output_dir_a / "summaries"
    summary_dir_b = output_dir_b / "summaries"
    
    # Check that summaries exist before proceeding
    scores_path_a = summary_dir_a / "affinity_scores.json"
    scores_path_b = summary_dir_b / "affinity_scores.json"
    
    if not scores_path_a.exists():
        raise FileNotFoundError(
            f"Missing affinity scores for {dataset_a_name}. "
            f"Expected file: {scores_path_a}\n"
            f"Did the Boltz scoring complete successfully? Check {output_dir_a}"
        )
    if not scores_path_b.exists():
        raise FileNotFoundError(
            f"Missing affinity scores for {dataset_b_name}. "
            f"Expected file: {scores_path_b}\n"
            f"Did the Boltz scoring complete successfully? Check {output_dir_b}"
        )
    
    ks_results = ks_test_results(summary_dir_a, summary_dir_b)
    
    print(f"Kolmogorov-Smirnov Test Results:")
    print(f"  Statistic (D): {ks_results['statistic']:.4f}")
    print(f"  p-value: {ks_results['pvalue']:.4e}")
    print(f"  {dataset_a_name} (n={ks_results['n_a']}): mean = {ks_results['mean_a']:.4f}")
    print(f"  {dataset_b_name} (n={ks_results['n_b']}): mean = {ks_results['mean_b']:.4f}")
    print()
    
    # Step 4: Create comparison histogram for main metric (affinity_pred_value)
    print("=" * 60)
    print("STEP 4: Generating Comparison Histogram (affinity_pred_value)")
    print("=" * 60)
    
    values_a = load_affinity_values(summary_dir_a)
    values_b = load_affinity_values(summary_dir_b)
    
    # Create side-by-side histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram A
    counts_a, bins_a, _ = ax1.hist(values_a, bins=20, color="steelblue", edgecolor="white", alpha=0.7)
    ax1.set_xlabel("Affinity Predicted Value", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"{dataset_a_name}\n(n={len(values_a)}, μ={values_a.mean():.3f})", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    # KDE for A
    try:
        from scipy.stats import gaussian_kde
        kde_a = gaussian_kde(values_a)
        xs_a = np.linspace(values_a.min(), values_a.max(), 200)
        scale_a = counts_a.max() / kde_a(xs_a).max() if kde_a(xs_a).max() > 0 else 1.0
        ax1.plot(xs_a, kde_a(xs_a) * scale_a, color="navy", linewidth=1.5)
    except Exception:
        pass
    
    # Histogram B
    counts_b, bins_b, _ = ax2.hist(values_b, bins=20, color="coral", edgecolor="white", alpha=0.7)
    ax2.set_xlabel("Affinity Predicted Value", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"{dataset_b_name}\n(n={len(values_b)}, μ={values_b.mean():.3f})", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    # KDE for B
    try:
        from scipy.stats import gaussian_kde
        kde_b = gaussian_kde(values_b)
        xs_b = np.linspace(values_b.min(), values_b.max(), 200)
        scale_b = counts_b.max() / kde_b(xs_b).max() if kde_b(xs_b).max() > 0 else 1.0
        ax2.plot(xs_b, kde_b(xs_b) * scale_b, color="darkred", linewidth=1.5)
    except Exception:
        pass
    
    plt.tight_layout()
    comparison_hist_path = comparison_dir / "comparison_histogram.png"
    plt.savefig(comparison_hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Comparison histogram saved to {comparison_hist_path}")
    
    # Create overlay histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    counts_a_ov, bins_a_ov, _ = ax.hist(values_a, bins=20, label=dataset_a_name, color="steelblue", alpha=0.6, edgecolor="white")
    counts_b_ov, bins_b_ov, _ = ax.hist(values_b, bins=20, label=dataset_b_name, color="coral", alpha=0.6, edgecolor="white")
    ax.set_xlabel("Affinity Predicted Value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Affinity Distribution Comparison\nKS D={ks_results['statistic']:.4f}, p={ks_results['pvalue']:.4e}", 
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # KDE overlays on combined plot
    try:
        from scipy.stats import gaussian_kde
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
    overlay_hist_path = comparison_dir / "overlay_histogram.png"
    plt.savefig(overlay_hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Overlay histogram saved to {overlay_hist_path}")

    # Step 4b: Histograms for affinity_probability_binary (binder probability)
    print("=" * 60)
    print("STEP 4b: Generating Comparison Histogram (affinity_probability_binary)")
    print("=" * 60)

    scores_path_a = summary_dir_a / "affinity_scores.json"
    scores_path_b = summary_dir_b / "affinity_scores.json"
    if scores_path_a.exists() and scores_path_b.exists():
        with scores_path_a.open("r", encoding="utf-8") as f:
            scores_a_main = json.load(f)
        with scores_path_b.open("r", encoding="utf-8") as f:
            scores_b_main = json.load(f)

        if scores_a_main and scores_b_main:
            try:
                probs_a = np.array(
                    [entry["affinity_probability_binary"] for entry in scores_a_main.values()],
                    dtype=float,
                )
                probs_b = np.array(
                    [entry["affinity_probability_binary"] for entry in scores_b_main.values()],
                    dtype=float,
                )

                # Side-by-side probability histograms
                fig, (axp1, axp2) = plt.subplots(1, 2, figsize=(14, 5))

                counts_pa, bins_pa, _ = axp1.hist(
                    probs_a, bins=20, color="steelblue", edgecolor="white", alpha=0.7
                )
                axp1.set_xlabel("affinity_probability_binary", fontsize=12)
                axp1.set_ylabel("Count", fontsize=12)
                axp1.set_title(
                    f"{dataset_a_name}\n(n={len(probs_a)}, μ={probs_a.mean():.3f})",
                    fontsize=12,
                    fontweight="bold",
                )
                axp1.grid(axis="y", alpha=0.3)

                counts_pb, bins_pb, _ = axp2.hist(
                    probs_b, bins=20, color="coral", edgecolor="white", alpha=0.7
                )
                axp2.set_xlabel("affinity_probability_binary", fontsize=12)
                axp2.set_ylabel("Count", fontsize=12)
                axp2.set_title(
                    f"{dataset_b_name}\n(n={len(probs_b)}, μ={probs_b.mean():.3f})",
                    fontsize=12,
                    fontweight="bold",
                )
                axp2.grid(axis="y", alpha=0.3)

                plt.tight_layout()
                prob_hist_path = comparison_dir / "comparison_histogram_probability.png"
                plt.savefig(prob_hist_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"✓ Probability comparison histogram saved to {prob_hist_path}")

                # Overlay probability histogram with KDE + KS in subtitle
                from scipy.stats import ks_2samp
                ks_prob_stat, ks_prob_p = ks_2samp(probs_a, probs_b)

                fig, axp = plt.subplots(figsize=(10, 6))
                counts_pa_ov, bins_pa_ov, _ = axp.hist(
                    probs_a,
                    bins=20,
                    label=dataset_a_name,
                    color="steelblue",
                    alpha=0.6,
                    edgecolor="white",
                )
                counts_pb_ov, bins_pb_ov, _ = axp.hist(
                    probs_b,
                    bins=20,
                    label=dataset_b_name,
                    color="coral",
                    alpha=0.6,
                    edgecolor="white",
                )
                axp.set_xlabel("affinity_probability_binary", fontsize=12)
                axp.set_ylabel("Count", fontsize=12)
                axp.set_title(
                    f"Binder Probability Distribution Comparison\n"
                    f"KS D={ks_prob_stat:.4f}, p={ks_prob_p:.4e}",
                    fontsize=14,
                    fontweight="bold",
                )
                axp.legend(fontsize=11)
                axp.grid(axis="y", alpha=0.3)

                try:
                    from scipy.stats import gaussian_kde

                    kde_pa = gaussian_kde(probs_a)
                    kde_pb = gaussian_kde(probs_b)
                    xs_p = np.linspace(min(probs_a.min(), probs_b.min()), max(probs_a.max(), probs_b.max()), 300)
                    scale_pa_ov = counts_pa_ov.max() / kde_pa(xs_p).max() if kde_pa(xs_p).max() > 0 else 1.0
                    scale_pb_ov = counts_pb_ov.max() / kde_pb(xs_p).max() if kde_pb(xs_p).max() > 0 else 1.0
                    axp.plot(xs_p, kde_pa(xs_p) * scale_pa_ov, color="navy", linewidth=1.5)
                    axp.plot(xs_p, kde_pb(xs_p) * scale_pb_ov, color="darkred", linewidth=1.5)
                except Exception:
                    pass

                plt.tight_layout()
                prob_overlay_path = comparison_dir / "overlay_histogram_probability.png"
                plt.savefig(prob_overlay_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"✓ Probability overlay histogram saved to {prob_overlay_path}")
            except KeyError:
                print("Warning: affinity_probability_binary not found; skipping probability histograms.")
        else:
            print("Warning: affinity_scores.json is empty; skipping probability histograms.")
    else:
        print("Warning: affinity_scores.json not found; skipping probability histograms.")
    
    # Step 5: Per-metric side-by-side and overlay histograms
    print("=" * 60)
    print("STEP 5: Generating per-metric side-by-side overlays")
    print("=" * 60)

    scores_path_a = summary_dir_a / "affinity_scores.json"
    scores_path_b = summary_dir_b / "affinity_scores.json"

    if scores_path_a.exists() and scores_path_b.exists():
        with scores_path_a.open("r", encoding="utf-8") as f:
            scores_a = json.load(f)
        with scores_path_b.open("r", encoding="utf-8") as f:
            scores_b = json.load(f)

        if scores_a and scores_b:
            first_key = next(iter(scores_a))
            all_metric_names = list(scores_a[first_key].keys())
            # Focus on a curated subset of metrics
            wanted_metrics = {
                "affinity_pred_value",
                "affinity_probability_binary",
                "pLDDT",
                "PAE",
            }
            metric_names = [m for m in all_metric_names if m in wanted_metrics]

            n_metrics = len(metric_names)
            n_cols = 2
            n_rows = 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            per_metric_ks: dict[str, dict[str, float]] = {}

            for idx, metric in enumerate(metric_names):
                ax = axes[idx]

                vals_a = np.array([entry[metric] for entry in scores_a.values()], dtype=float)
                vals_b = np.array([entry[metric] for entry in scores_b.values()], dtype=float)

                # Overlaid histograms
                counts_ma, bins_ma, _ = ax.hist(vals_a, bins=20, alpha=0.6, color="steelblue",
                                                label=dataset_a_name, edgecolor="white")
                counts_mb, bins_mb, _ = ax.hist(vals_b, bins=20, alpha=0.6, color="coral",
                                                label=dataset_b_name, edgecolor="white")

                ax.set_title(metric, fontsize=10, fontweight="bold")
                ax.set_xlabel("value")
                ax.set_ylabel("count")
                ax.grid(axis="y", alpha=0.3)

                # KDE overlays for this metric
                try:
                    from scipy.stats import gaussian_kde
                    kde_ma = gaussian_kde(vals_a)
                    kde_mb = gaussian_kde(vals_b)
                    xs_m = np.linspace(min(vals_a.min(), vals_b.min()),
                                       max(vals_a.max(), vals_b.max()), 200)
                    scale_ma = counts_ma.max() / kde_ma(xs_m).max() if kde_ma(xs_m).max() > 0 else 1.0
                    scale_mb = counts_mb.max() / kde_mb(xs_m).max() if kde_mb(xs_m).max() > 0 else 1.0
                    ax.plot(xs_m, kde_ma(xs_m) * scale_ma, color="navy", linewidth=1.0)
                    ax.plot(xs_m, kde_mb(xs_m) * scale_mb, color="darkred", linewidth=1.0)
                except Exception:
                    pass

                # KS for this metric
                from scipy.stats import ks_2samp  # local import to avoid top-level dependency surprises
                stat, pval = ks_2samp(vals_a, vals_b)
                per_metric_ks[metric] = {
                    "statistic": float(stat),
                    "pvalue": float(pval),
                    "mean_a": float(vals_a.mean()),
                    "mean_b": float(vals_b.mean()),
                    "n_a": int(len(vals_a)),
                    "n_b": int(len(vals_b)),
                }

                # Add KS summary in subtitle and legend
                ax.set_title(
                    f"{metric}\nD={stat:.3f}, p={pval:.1e}",
                    fontsize=10,
                    fontweight="bold",
                )
                ax.legend(fontsize=8)

            # Hide any unused subplots
            for j in range(idx + 1, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(
                f"{dataset_a_name} vs {dataset_b_name} – per-metric distributions",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            multi_hist_path = comparison_dir / "all_metrics_overlay.png"
            plt.savefig(multi_hist_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✓ All-metrics comparison saved to {multi_hist_path}")
        else:
            per_metric_ks = {}
            print("Warning: affinity_scores.json is empty; skipping per-metric plots.")
    else:
        per_metric_ks = {}
        print("Warning: affinity_scores.json not found; skipping per-metric plots.")
    
    # Save comparison results
    comparison_results = {
        "dataset_a": dataset_a_name,
        "dataset_b": dataset_b_name,
        "ks_test": ks_results,
        "dataset_a_stats": {
            "mean": float(values_a.mean()),
            "std": float(values_a.std()),
            "min": float(values_a.min()),
            "max": float(values_a.max()),
            "n": int(len(values_a)),
        },
        "dataset_b_stats": {
            "mean": float(values_b.mean()),
            "std": float(values_b.std()),
            "min": float(values_b.min()),
            "max": float(values_b.max()),
            "n": int(len(values_b)),
        },
        "per_metric_ks": per_metric_ks,
    }
    
    results_path = comparison_dir / "comparison_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"✓ Comparison results saved to {results_path}")

    # Also write a human-readable KS report per metric
    if per_metric_ks:
        report_path = comparison_dir / "per_metric_ks_report.txt"
        with report_path.open("w", encoding="utf-8") as rep:
            rep.write(f"Per-metric KS comparison for {dataset_a_name} vs {dataset_b_name}\n")
            rep.write("=" * 80 + "\n\n")
            for metric, stats in per_metric_ks.items():
                rep.write(f"Metric: {metric}\n")
                rep.write(f"  n_a = {stats['n_a']}, mean_a = {stats['mean_a']:.4f}\n")
                rep.write(f"  n_b = {stats['n_b']}, mean_b = {stats['mean_b']:.4f}\n")
                rep.write(f"  KS statistic D = {stats['statistic']:.4f}\n")
                rep.write(f"  p-value = {stats['pvalue']:.4e}\n")
                if stats["pvalue"] < 0.05:
                    rep.write("  Interpretation: distributions likely differ (p < 0.05)\n")
                else:
                    rep.write("  Interpretation: no strong evidence of difference (p >= 0.05)\n")
                rep.write("-" * 80 + "\n")
        print(f"✓ Per-metric KS text report saved to {report_path}")
    
    print()
    print("=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Results directory: {comparison_dir}")
    print(f"KS Test: D={ks_results['statistic']:.4f}, p={ks_results['pvalue']:.4e}")
    if ks_results['pvalue'] < 0.05:
        print("✓ Distributions are significantly different (p < 0.05)")
    else:
        print("✗ Distributions are not significantly different (p >= 0.05)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Boltz-2 affinity distributions between two datasets")
    parser.add_argument("--dataset-a", type=Path, required=True, help="Path to first CSV file")
    parser.add_argument("--name-a", type=str, required=True, help="Name for first dataset")
    parser.add_argument("--dataset-b", type=Path, required=True, help="Path to second CSV file")
    parser.add_argument("--name-b", type=str, required=True, help="Name for second dataset")
    parser.add_argument("--template", type=Path, required=True, help="Path to Boltz template YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"), help="Output directory")
    parser.add_argument("--binder-id", default="LIG", help="Ligand identifier")
    parser.add_argument("--sample-size", type=int, help="Number of molecules to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu", help="Hardware accelerator")
    parser.add_argument("--fast", action="store_true", help="Use faster, lower-accuracy Boltz settings")
    
    args = parser.parse_args()
    
    run_comparison(
        dataset_a_path=args.dataset_a,
        dataset_a_name=args.name_a,
        dataset_b_path=args.dataset_b,
        dataset_b_name=args.name_b,
        template_path=args.template,
        output_base_dir=args.output_dir,
        binder_id=args.binder_id,
        sample_size=args.sample_size,
        seed=args.seed,
        accelerator=args.accelerator,
        fast=args.fast,
    )

