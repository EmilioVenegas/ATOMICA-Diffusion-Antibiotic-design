#!/usr/bin/env python3
"""
Add molecular descriptors to CDK2 test data CSV files.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_descriptors(smiles: str) -> dict:
    """Calculate molecular descriptors from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "mw": None,
                "alogp": None,
                "hbd": None,
                "hba": None,
            }
        
        return {
            "mw": round(Descriptors.MolWt(mol), 2),
            "alogp": round(Descriptors.MolLogP(mol), 2),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
        }
    except Exception as e:
        print(f"⚠ Warning: Error calculating descriptors for {smiles}: {e}")
        return {
            "mw": None,
            "alogp": None,
            "hbd": None,
            "hba": None,
        }


def main():
    binders_csv = Path("scripts/eval/cdk2_test_data/binders.csv")
    decoys_csv = Path("scripts/eval/cdk2_test_data/decoys.csv")
    
    # Process binders
    print(f"Reading {binders_csv}...")
    binders_df = pd.read_csv(binders_csv)
    print(f"Found {len(binders_df)} binders")
    
    print("Calculating descriptors for binders...")
    descriptors_list = []
    for idx, row in binders_df.iterrows():
        desc = calculate_descriptors(row["smiles"])
        descriptors_list.append(desc)
    
    binders_df["mw"] = [d["mw"] for d in descriptors_list]
    binders_df["alogp"] = [d["alogp"] for d in descriptors_list]
    binders_df["hbd"] = [d["hbd"] for d in descriptors_list]
    binders_df["hba"] = [d["hba"] for d in descriptors_list]
    
    # Reorder columns
    binders_df = binders_df[["smiles", "molecule_id", "mw", "alogp", "hbd", "hba"]]
    
    binders_df.to_csv(binders_csv, index=False)
    print(f"✓ Saved {len(binders_df)} binders with descriptors to {binders_csv}\n")
    
    # Process decoys
    print(f"Reading {decoys_csv}...")
    decoys_df = pd.read_csv(decoys_csv)
    print(f"Found {len(decoys_df)} decoys")
    
    print("Calculating descriptors for decoys...")
    descriptors_list = []
    for idx, row in decoys_df.iterrows():
        desc = calculate_descriptors(row["smiles"])
        descriptors_list.append(desc)
    
    decoys_df["mw"] = [d["mw"] for d in descriptors_list]
    decoys_df["alogp"] = [d["alogp"] for d in descriptors_list]
    decoys_df["hbd"] = [d["hbd"] for d in descriptors_list]
    decoys_df["hba"] = [d["hba"] for d in descriptors_list]
    
    # Reorder columns
    decoys_df = decoys_df[["smiles", "molecule_id", "mw", "alogp", "hbd", "hba"]]
    
    decoys_df.to_csv(decoys_csv, index=False)
    print(f"✓ Saved {len(decoys_df)} decoys with descriptors to {decoys_csv}\n")
    
    print("Done!")


if __name__ == "__main__":
    main()

