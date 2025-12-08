#!/usr/bin/env python3
from rdkit import Chem
from pathlib import Path
import argparse

def add_hs_to_sdf(in_sdf: Path, out_sdf: Path):
    # Load all molecules (keep original coords)
    suppl = Chem.SDMolSupplier(str(in_sdf), removeHs=False)
    writer = Chem.SDWriter(str(out_sdf))

    for i, mol in enumerate(suppl):
        if mol is None:
            print(f"[WARN] Skipping mol {i} (RDKit failed to read it)")
            continue

        # Add explicit H atoms; addCoords=True fills coordinates for Hs
        mol_H = Chem.AddHs(mol, addCoords=True)
        writer.write(mol_H)

    writer.close()
    print(f"✔ Finished: wrote SDF with explicit Hs to {out_sdf}")


if __name__ == "__main__":
    print("Hi")
    parser = argparse.ArgumentParser(description="Add explicit hydrogens to all molecules in an SDF")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input SDF file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output SDF with explicit Hs")

    args = parser.parse_args()
    add_hs_to_sdf(args.input, args.output)
