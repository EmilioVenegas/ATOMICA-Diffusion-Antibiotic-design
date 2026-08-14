"""Cross-docking specificity: does a molecule prefer the pocket it was designed for?

The A/B ablation could not have detected pocket specificity. QED is
target-independent, so a QED gain is equally consistent with "conditioning
narrowed the model toward generically drug-like chemistry" and with
"conditioning improved pocket fit" (docs/experiment-plan.md, "Evaluation").
This is the measurement that separates them.

Each pocket's molecules are docked into **its own** pocket and into *m* other
pockets from the same evaluation set. The statistic per pocket is

    specificity = mean(score against other pockets) - mean(score against own)

With smina, lower is better, so a molecule that genuinely fits the pocket it was
generated for gives a **positive** specificity. A generically drug-like molecule
docks about as well anywhere and gives approximately zero, however good its
absolute score or its QED.

Two things this script refuses to do, both of which have already produced wrong
conclusions in this project:

- **It does not pool molecules.** The ~9,300 valid molecules per arm are nested
  within ~100 pockets; treating them as independent is pseudo-replication and
  inflates significance. Every statistic here is computed per pocket first, and
  arms are compared by a paired test *across pockets* with the fraction of
  pockets improved reported alongside the mean shift.
- **It re-docks rather than rescoring in place.** A molecule generated for
  pocket A sits in A's coordinate frame; running `--score_only` against pocket B
  would score it wherever it happens to land in B, which measures the frame
  offset, not the fit. Each pair is docked into a box centred on the target
  pocket.

Docking-based evaluation is only non-circular because the Vina expert filter was
dropped (`--no_expert_filter`). Do not reintroduce that filter and then report
docking scores.

Requires the smina binary (`conda install -c conda-forge smina`, or set
SMINA_BIN). Receptor PDBs and their boxes come from
`scripts/extract_pocket_pdbs.py`; regenerate them whenever the test split is
rebuilt, or the pocket ids here will not match the ones the molecules were
generated for.

Usage (from repo root):

    python scripts/cross_dock_specificity.py \\
        --arms baseline_A=results/baseline_A/filtered \\
               critic=results/critic/filtered \\
        --pdb_dir data/receptor_pdbs \\
        --out results/specificity/specificity.csv \\
        --n_decoy_pockets 3 --max_mols_per_pocket 20 --n_jobs 8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

SMINA_BIN = os.environ.get("SMINA_BIN", "smina")


def read_box(box_path):
    """Parse a `key = value` box file into smina arguments."""
    box = {}
    with open(box_path) as fh:
        for line in fh:
            if "=" in line:
                key, value = line.split("=", 1)
                box[key.strip()] = float(value.strip())
    return box


def dock_one(args):
    """Dock every molecule of one SDF into one receptor's box.

    Returns ``(pocket, receptor, [scores])``. A failed pair yields NaNs rather
    than raising: over thousands of pairs a handful will fail, and losing the
    whole run to one of them is worse than a gap in the table.
    """
    sdf_path, receptor_pdb, box_path, pocket, receptor, exhaustiveness, seed = args
    box = read_box(box_path)

    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True)
    mols = [m for m in suppl if m is not None]
    if not mols:
        return pocket, receptor, []

    with tempfile.TemporaryDirectory() as tmpdir:
        ligand_file = Path(tmpdir) / "ligands.sdf"
        writer = Chem.SDWriter(str(ligand_file))
        for mol in mols:
            writer.write(mol)
        writer.close()

        cmd = [
            SMINA_BIN,
            "--receptor", str(receptor_pdb),
            "--ligand", str(ligand_file),
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",
            "--seed", str(seed),
            "--cpu", "1",
            "--out", str(Path(tmpdir) / "out.sdf"),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except FileNotFoundError:
            raise RuntimeError(
                f"smina not found at '{SMINA_BIN}'. Install with "
                "`conda install -c conda-forge smina` or set SMINA_BIN."
            )
        except subprocess.TimeoutExpired:
            return pocket, receptor, [float("nan")] * len(mols)

        if proc.returncode != 0:
            return pocket, receptor, [float("nan")] * len(mols)

        scores = []
        out_sdf = Path(tmpdir) / "out.sdf"
        if out_sdf.exists():
            for mol in Chem.SDMolSupplier(str(out_sdf), sanitize=False):
                if mol is None:
                    scores.append(float("nan"))
                    continue
                value = mol.GetPropsAsDict().get("minimizedAffinity")
                scores.append(float(value) if value is not None else float("nan"))
    return pocket, receptor, scores


def build_pairs(sdf_by_pocket, receptors, n_decoy, rng, exhaustiveness, seed,
                pocket_targets=None):
    """(molecule set, receptor) pairs: each pocket against its own and n_decoy others.

    Decoy pockets are drawn from **different targets** when `pocket_targets` is
    available. CrossDocked holds many complexes per target, so without that
    constraint a "decoy" pocket can be the same protein in another docked pose --
    which a pocket-specific molecule should fit, washing out the contrast the
    metric exists to measure.
    """
    pockets = sorted(sdf_by_pocket)
    pairs = []
    for pocket in pockets:
        if pocket not in receptors:
            continue
        own_target = (pocket_targets or {}).get(pocket)
        others = [
            p for p in pockets
            if p != pocket and p in receptors
            and (own_target is None
                 or (pocket_targets or {}).get(p) != own_target)
        ]
        chosen = rng.choice(others, size=min(n_decoy, len(others)), replace=False) \
            if others else []
        for receptor in [pocket, *chosen]:
            pdb, box = receptors[receptor]
            pairs.append((sdf_by_pocket[pocket], pdb, box, pocket, receptor,
                          exhaustiveness, seed))
    return pairs


def subsample_sdf(src, dest, limit, rng):
    """Write at most `limit` molecules from `src` into `dest`.

    Cross-docking cost is (pockets x molecules x receptors), so the molecule
    count is the only term worth cutting. Sampling is random rather than
    top-of-file, since generation order is not arbitrary.
    """
    mols = [m for m in Chem.SDMolSupplier(str(src), sanitize=True) if m is not None]
    if not mols:
        return None
    if len(mols) > limit:
        mols = [mols[i] for i in sorted(rng.choice(len(mols), limit, replace=False))]
    writer = Chem.SDWriter(str(dest))
    for mol in mols:
        writer.write(mol)
    writer.close()
    return dest


def collect_arm(arm_dir, receptors, workdir, limit, rng):
    """Map pocket id -> a (possibly subsampled) SDF of that pocket's molecules."""
    arm_dir = Path(arm_dir)
    out = {}
    for sdf in sorted(arm_dir.glob("*.sdf")):
        pocket = sdf.stem.replace("_filtered", "")
        if pocket not in receptors:
            continue
        dest = subsample_sdf(sdf, workdir / f"{pocket}.sdf", limit, rng)
        if dest is not None:
            out[pocket] = dest
    return out


def per_pocket_specificity(rows):
    """Collapse raw pair scores into one specificity value per pocket."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["score"])
    matched = frame[frame["pocket"] == frame["receptor"]]
    crossed = frame[frame["pocket"] != frame["receptor"]]

    own = matched.groupby("pocket")["score"].mean().rename("own_score")
    other = crossed.groupby("pocket")["score"].mean().rename("cross_score")
    counts = matched.groupby("pocket")["score"].size().rename("n_mols")

    result = pd.concat([own, other, counts], axis=1).dropna()
    # smina: lower is better, so cross - own > 0 means the molecule prefers the
    # pocket it was designed for.
    result["specificity"] = result["cross_score"] - result["own_score"]
    return result.reset_index()


def paired_report(arms):
    """Paired comparison across the pockets common to every arm."""
    names = list(arms)
    common = set(arms[names[0]]["pocket"])
    for name in names[1:]:
        common &= set(arms[name]["pocket"])
    common = sorted(common)
    if not common:
        print("No pockets common to all arms; nothing to compare.")
        return None

    table = pd.DataFrame({"pocket": common})
    for name in names:
        sub = arms[name].set_index("pocket").loc[common]
        table[f"{name}__own"] = sub["own_score"].values
        table[f"{name}__cross"] = sub["cross_score"].values
        table[f"{name}__specificity"] = sub["specificity"].values

    print(f"\n{len(common)} pockets common to all arms. "
          f"n is the number of pockets, not molecules.\n")
    header = f"{'arm':<20} {'own':>10} {'cross':>10} {'specificity':>22} {'frac > 0':>10}"
    print(header)
    print("-" * len(header))
    for name in names:
        spec = table[f"{name}__specificity"].values
        sem = spec.std(ddof=1) / np.sqrt(len(spec)) if len(spec) > 1 else 0.0
        print(f"{name:<20} {table[f'{name}__own'].mean():>10.3f} "
              f"{table[f'{name}__cross'].mean():>10.3f} "
              f"{spec.mean():>+13.3f} +- {sem:<5.3f} "
              f"{(spec > 0).mean():>10.2f}")

    if len(names) > 1:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            print("\n(scipy not available; skipping paired tests)")
            return table
        print(f"\nPaired across pockets, against '{names[0]}':")
        print(f"{'arm':<20} {'d(specificity)':>16} {'pockets improved':>18} {'wilcoxon p':>12}")
        print("-" * 70)
        base = table[f"{names[0]}__specificity"].values
        for name in names[1:]:
            other = table[f"{name}__specificity"].values
            delta = other - base
            try:
                p = wilcoxon(other, base).pvalue
            except ValueError:
                p = float("nan")
            print(f"{name:<20} {delta.mean():>+16.3f} "
                  f"{f'{(delta > 0).sum()}/{len(delta)}':>18} {p:>12.4g}")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arms", nargs="+", required=True,
                   help="name=path/to/filtered pairs, one per arm")
    p.add_argument("--pdb_dir", default="data/receptor_pdbs")
    p.add_argument("--out", default="results/specificity/specificity.csv")
    p.add_argument("--n_decoy_pockets", type=int, default=3,
                   help="other pockets each molecule set is docked into")
    p.add_argument("--max_mols_per_pocket", type=int, default=20)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pdb_dir = Path(args.pdb_dir)
    receptors = {}
    for pdb in sorted(pdb_dir.glob("*.pdb")):
        box = pdb.with_suffix(".box.txt")
        if box.exists():
            receptors[pdb.stem] = (pdb, box)
    if not receptors:
        print(f"No receptor/box pairs under {pdb_dir}. Run "
              f"scripts/extract_pocket_pdbs.py first.")
        return
    # Written by scripts/extract_pocket_pdbs.py. Without it, decoy pockets can
    # be other poses of the same protein.
    targets_path = pdb_dir / "pocket_targets.json"
    pocket_targets = None
    if targets_path.exists():
        with open(targets_path) as fh:
            pocket_targets = json.load(fh)
        print(f"{len(receptors)} receptors with boxes, "
              f"{len(set(pocket_targets.values()))} distinct targets.")
    else:
        print(f"{len(receptors)} receptors with boxes. WARNING: no "
              f"{targets_path.name}; decoy pockets may be other poses of the "
              f"same protein, which understates specificity. Regenerate with "
              f"scripts/extract_pocket_pdbs.py.")

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arm_specificity, raw_rows = {}, []
    with tempfile.TemporaryDirectory() as tmpdir:
        for spec in args.arms:
            if "=" not in spec:
                print(f"Skipping malformed arm {spec!r}; expected name=path")
                continue
            name, path = spec.split("=", 1)
            workdir = Path(tmpdir) / name
            workdir.mkdir(parents=True, exist_ok=True)

            sdf_by_pocket = collect_arm(path, receptors, workdir,
                                        args.max_mols_per_pocket, rng)
            if not sdf_by_pocket:
                print(f"Arm {name!r}: no usable SDFs under {path}")
                continue
            pairs = build_pairs(sdf_by_pocket, receptors, args.n_decoy_pockets,
                                rng, args.exhaustiveness, args.seed,
                                pocket_targets=pocket_targets)
            print(f"\nArm {name!r}: {len(sdf_by_pocket)} pockets, {len(pairs)} "
                  f"docking jobs ({1 + args.n_decoy_pockets} receptors each)")

            rows = []
            with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
                futures = [pool.submit(dock_one, pair) for pair in pairs]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"docking {name}"):
                    pocket, receptor, scores = future.result()
                    for i, score in enumerate(scores):
                        rows.append({"arm": name, "pocket": pocket,
                                     "receptor": receptor, "mol_idx": i,
                                     "score": score})
            raw_rows.extend(rows)
            arm_specificity[name] = per_pocket_specificity(rows)

    if not arm_specificity:
        print("Nothing docked.")
        return

    pd.DataFrame(raw_rows).to_csv(out_path.with_name(out_path.stem + "_raw.csv"),
                                  index=False)
    combined = pd.concat(
        [frame.assign(arm=name) for name, frame in arm_specificity.items()],
        ignore_index=True)
    combined.to_csv(out_path, index=False)

    table = paired_report(arm_specificity)
    if table is not None:
        table.to_csv(out_path.with_name(out_path.stem + "_paired.csv"), index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
