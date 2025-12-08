#!/usr/bin/env python3
"""
Minimal AutoDock Vina scorer

Assumes:
- receptor is already in PDBQT format
- ligands are PDBQT files (e.g. prepared with Meeko's mk_prepare_ligand.py)

Example:

  # 1) Prepare ligands from multi_mol.sdf
  # mk_prepare_ligand.py -i multi_mol.sdf --multimol_outdir lig_pdbqt

  # 2) Run Vina scoring
  # python vina_score_simple.py \
  #   --receptor rec.pdbqt \
  #   --ligand_dir lig_pdbqt \
  #   --center 10.5 25.0 15.0 \
  #   --size   20 20 20 \
  #   --output vina_scores.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List

import pandas as pd


def parse_vina_score(stdout: str) -> float:
    """
    Parse the best Vina score from vina stdout.
    Looks for the first line in the result table (mode 1).
    """
    for line in stdout.splitlines():
        # Typical table line: "   1       -7.5      0.000      0.000"
        if line.strip().startswith("1 "):  # mode index 1
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass

    # Fallback: look for "Affinity: XXX"
    m = re.search(r"Affinity:\s+([+-]?\d+(?:\.\d+)?)", stdout)
    if m:
        return float(m.group(1))

    raise RuntimeError(f"Could not parse Vina score from output:\n{stdout}")


def run_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    exhaustiveness: int = 8,
    num_modes: int = 1,
) -> float:
    """
    Run AutoDock Vina for a single receptor/ligand pair and return best score.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_pdbqt = Path(tmpdir) / "out.pdbqt"

        cmd = [
            "vina",
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
            "--out", str(out_pdbqt),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Vina failed for ligand {ligand_pdbqt}:\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        return parse_vina_score(result.stdout)


def score_ligands_in_dir(
    ligand_dir: Path,
    receptor_pdbqt: Path,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
) -> pd.DataFrame:
    """
    Score all *.pdbqt ligands in a directory with Vina.
    """
    lig_files: List[Path] = sorted(ligand_dir.glob("*.pdbqt"))
    if not lig_files:
        raise FileNotFoundError(f"No .pdbqt ligands found in {ligand_dir}")

    results = []
    for lig in lig_files:
        try:
            score = run_vina(
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=lig,
                center=center,
                size=size,
                exhaustiveness=exhaustiveness,
                num_modes=1,
            )
            results.append({"ligand_file": lig.name, "vina_score": score})
        except Exception as e:
            print(f"[WARN] Failed to score {lig.name}: {e}")
            results.append({"ligand_file": lig.name, "vina_score": None})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Minimal AutoDock Vina scorer for a directory of ligand PDBQT files"
    )
    parser.add_argument(
        "--receptor",
        type=Path,
        required=True,
        help="Receptor PDBQT file (already prepared)",
    )
    parser.add_argument(
        "--ligand_dir",
        type=Path,
        required=True,
        help="Directory containing ligand .pdbqt files",
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        required=True,
        help="Search space center (x y z)",
    )
    parser.add_argument(
        "--size",
        nargs=3,
        type=float,
        default=[20.0, 20.0, 20.0],
        help="Search space size (x y z, default: 20 20 20)",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=8,
        help="Vina exhaustiveness (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write scores",
    )

    args = parser.parse_args()
    center = tuple(args.center)
    size = tuple(args.size)

    df = score_ligands_in_dir(
        ligand_dir=args.ligand_dir,
        receptor_pdbqt=args.receptor,
        center=center,
        size=size,
        exhaustiveness=args.exhaustiveness,
    )

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved scores to {args.output}")

    print("\nVina scoring summary:")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
