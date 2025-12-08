#!/usr/bin/env python3
"""
Extract SMILES from an SDF file and save to CSV.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def extract_smiles_from_sdf(sdf_path: Path) -> pd.DataFrame:
    """Extract SMILES from SDF file."""
    print(f"Reading SDF file: {sdf_path}")
    
    supplier = Chem.SDMolSupplier(str(sdf_path))
    
    records = []
    mol_idx = 0
    for idx, mol in enumerate(supplier):
        if mol is None:
            print(f"⚠ Warning: Failed to read molecule {idx + 1}, skipping...")
            continue
        
        mol_idx += 1
        
        # Try to get molecule name from title, otherwise generate one
        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name").strip()
            if not mol_name:
                mol_name = f"generated_mol_{mol_idx:04d}"
        else:
            mol_name = f"generated_mol_{mol_idx:04d}"
        
        # Convert to SMILES
        smiles = Chem.MolToSmiles(mol)
        
        # Calculate molecular descriptors
        mw = round(Descriptors.MolWt(mol), 2)
        alogp = round(Descriptors.MolLogP(mol), 2)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        records.append({
            "molecule_id": mol_name,
            "smiles": smiles,
            "mw": mw,
            "alogp": alogp,
            "hbd": hbd,
            "hba": hba,
        })
    
    df = pd.DataFrame(records)
    print(f"✓ Extracted {len(df)} molecules")
    return df


def main():
    sdf_path = Path("/Users/khbelahsen/GitHub/harvard/mlcb/final-project/ATOMICA-Diffusion-Antibiotic-design/3pbq_evaluation/inputs/Dec06_iter1_generated_filtered.sdf")
    output_csv = Path("3pbq_evaluation/inputs/generated_molecules_without_rl.csv")
    
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")
    
    df = extract_smiles_from_sdf(sdf_path)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df)} molecules to {output_csv}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()

