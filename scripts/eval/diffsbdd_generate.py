"""Utility to run a small DiffSBDD generation for a given protein pocket."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFSBDD_DIR = PROJECT_ROOT / "DiffSBDD"
if str(DIFFSBDD_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFSBDD_DIR))

from lightning_modules import LigandPocketDDPM  # noqa: E402
import utils as diff_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ligand binders with DiffSBDD for a supplied PDB file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a DiffSBDD checkpoint (.ckpt).")
    parser.add_argument("pdbfile", type=Path, help="Protein pocket PDB file used for conditioning.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ref-ligand",
        type=Path,
        help="Reference ligand (SDF or PDB) to define the pocket.",
    )
    group.add_argument(
        "--resi-list",
        nargs="+",
        help="Residue list defining the pocket, e.g. A:123 B:45.",
    )
    parser.add_argument("--outfile", type=Path, required=True, help="Location to save generated SDF.")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of ligands to generate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Generation batch size.")
    parser.add_argument("--timesteps", type=int, default=50, help="Diffusion timesteps (None uses default).")
    parser.add_argument("--resamplings", type=int, default=5, help="Number of resampling iterations.")
    parser.add_argument("--sanitize", action="store_true", help="Run RDKit sanitization on outputs.")
    parser.add_argument("--relax", action="store_true", help="Run force-field relaxation (200 steps).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device.")
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = args.checkpoint.resolve()
    pdb_path = args.pdbfile.resolve()
    outfile = args.outfile.resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_size is None:
        if args.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        batch_size = args.n_samples
    else:
        batch_size = args.batch_size
    if args.n_samples % batch_size != 0:
        raise ValueError("n_samples must be divisible by batch_size")

    model = LigandPocketDDPM.load_from_checkpoint(checkpoint, map_location=device)
    model = model.to(device)

    if args.ref_ligand:
        ref_ligand = str(args.ref_ligand.resolve())
        resi_list = None
    else:
        ref_ligand = None
        resi_list = args.resi_list

    num_nodes_lig = None
    molecules = []
    for _ in range(args.n_samples // batch_size):
        batch = model.generate_ligands(
            str(pdb_path),
            batch_size,
            resi_list,
            ref_ligand,
            num_nodes_lig,
            sanitize=args.sanitize,
            largest_frag=True,
            relax_iter=(200 if args.relax else 0),
            resamplings=args.resamplings,
            timesteps=args.timesteps,
        )
        molecules.extend(batch)

    if not molecules:
        raise RuntimeError("DiffSBDD did not return any molecules. Try adjusting generation parameters.")

    diff_utils.write_sdf_file(outfile, molecules)
    print(f"Wrote {len(molecules)} molecules to {outfile}")


if __name__ == "__main__":
    main()


