"""Compare generation metrics across ablation conditions.

Reads the ``metrics.json`` emitted by ``scripts/evaluate.py`` for each condition
and renders the comparison as a markdown table plus a summary figure. This is
the script that produces the headline table in the README, so the reported
numbers can always be regenerated from the raw per-condition metrics rather
than being transcribed by hand.

Usage (from repo root)::

    python scripts/compare_conditions.py \
        --conditions results/baseline_A results/cond_B \
        --outdir results
"""

import argparse
import json
from pathlib import Path

# (json key, display name, "higher is better")
METRICS = [
    ("qed_mean", "QED", True),
    ("sa_mean", "SA", True),
    ("lipinski_mean", "Lipinski", True),
    ("validity", "Validity", True),
    ("diversity_mean", "Diversity", True),
    ("uniqueness_mean", "Uniqueness", True),
    ("novelty", "Novelty", True),
]


def load_condition(path):
    """Load one condition's metrics.json, keeping its std fields where present."""
    with open(Path(path) / "metrics.json") as fh:
        return json.load(fh)


def fmt(value, std=None):
    if std is None:
        return f"{value:.3f}"
    return f"{value:.3f} ± {std:.3f}"


def build_table(conditions):
    """Render the condition comparison as a markdown table.

    The delta column is only meaningful against a single reference, so it is
    emitted only for the two-condition case (the A/B ablation).
    """
    names = [c["label"] for c in conditions]
    pairwise = len(conditions) == 2

    header = ["Metric"] + names + (["Δ"] if pairwise else [])
    rows = [header, ["---"] * len(header)]

    for key, display, higher_better in METRICS:
        if not all(key in c for c in conditions):
            continue
        row = [display]
        for cond in conditions:
            # Only *_mean metrics carry a matching *_std; scalars like validity
            # and novelty are reported as a single number.
            std = cond.get(key[: -len("_mean")] + "_std") if key.endswith("_mean") else None
            row.append(fmt(cond[key], std))
        if pairwise:
            delta = conditions[1][key] - conditions[0][key]
            # Mark direction against intent, not against raw sign.
            arrow = "↑" if (delta > 0) == higher_better else "↓"
            row.append(f"{delta:+.3f} {arrow}" if abs(delta) >= 5e-4 else "≈")
        rows.append(row)

    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


# Diverging pair (validated: adjacent CVD ΔE 21.6, normal-vision 32.3, both >3:1
# on the light surface) plus chart chrome. Sign is carried by bar direction and
# the printed value as well as by hue, so color never encodes alone.
SURFACE = "#fcfcfb"
IMPROVED = "#2a78d6"
REGRESSED = "#e34948"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def build_figure(conditions, outpath):
    """Plot each metric's relative change from the reference condition.

    Absolute values span incommensurate scales (Lipinski runs 0-5, everything
    else 0-1), so plotting them on one axis buries the effect that matters. The
    ablation question is directional -- did conditioning help, and by how much --
    which is a polarity job: relative change against a neutral zero.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(conditions) != 2:
        raise ValueError("the relative-change figure compares exactly two conditions")

    ref, cond = conditions
    rows = [
        (display, 100.0 * (cond[key] - ref[key]) / ref[key])
        for key, display, _ in METRICS
        if key in ref and key in cond and ref[key]
    ]
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh(
        labels,
        deltas,
        height=0.62,
        color=[IMPROVED if d >= 0 else REGRESSED for d in deltas],
    )

    span = max(abs(d) for d in deltas) or 1.0
    for y, d in enumerate(deltas):
        pad = span * 0.02
        ax.text(
            d + (pad if d >= 0 else -pad),
            y,
            f"{d:+.1f}%",
            va="center",
            ha="left" if d >= 0 else "right",
            fontsize=9,
            color=INK,
        )

    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel("relative change vs. unconditioned baseline (%)", color=INK_MUTED)
    ax.set_title(
        "ATOMICA pocket conditioning shifts drug-likeness up, diversity down\n"
        f"{ref['n_pockets']} pockets · "
        f"{min(c['n_valid'] for c in conditions):,}+ valid molecules per arm",
        color=INK,
        fontsize=11,
        loc="left",
    )

    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="Condition directories, each containing a metrics.json. "
        "The first is treated as the reference for the delta column.",
    )
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    conditions = [load_condition(p) for p in args.conditions]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    table = build_table(conditions)

    counts = "\n".join(
        f"- **{c['label']}**: {c['n_valid']:,} valid / {c['n_generated']:,} generated "
        f"across {c['n_pockets']} pockets"
        for c in conditions
    )
    summary = (
        "# Ablation summary\n\n"
        "Generated by `scripts/compare_conditions.py`. Do not edit by hand.\n\n"
        f"{counts}\n\n"
        f"{table}\n"
    )
    (outdir / "ablation_summary.md").write_text(summary)

    build_figure(conditions, outdir / "figures" / "ablation.png")

    print(summary)
    print(f"wrote {outdir/'ablation_summary.md'} and {outdir/'figures'/'ablation.png'}")


if __name__ == "__main__":
    main()
