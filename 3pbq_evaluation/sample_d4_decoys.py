#!/usr/bin/env python3
"""Sample 50 random compounds from D4_exp_data.csv and calculate molecular descriptors."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def calculate_descriptors(smiles: str) -> dict | None:
    """Calculate molecular descriptors from SMILES.
    
    Returns dict with mw, alogp, hbd, hba or None if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mw = Descriptors.ExactMolWt(mol)
        alogp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        return {
            "mw": round(mw, 2),
            "alogp": round(alogp, 2),
            "hbd": int(hbd),
            "hba": int(hba),
        }
    except Exception:
        return None


def main():
    # Read D4_exp_data.csv
    input_csv = Path("3pbq_evaluation/D4_exp_data.csv")
    output_csv = Path("3pbq_evaluation/3pbq_decoys_dude.csv")
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Found {len(df)} total compounds")
    
    # Filter for binders only (binder_nonbinder == 1)
    binders_df = df[df["binder_nonbinder"] == 1].copy()
    print(f"Found {len(binders_df)} binders")
    
    # Calculate MW for all binders and filter by MW range
    print("Calculating molecular weights and filtering by MW (150-400)...")
    filtered_results = []
    for idx, row in binders_df.iterrows():
        smiles = row["smiles"]
        zincid = row["zincid"]
        
        descriptors = calculate_descriptors(smiles)
        if descriptors is None:
            continue
        
        mw = descriptors["mw"]
        # Filter by MW range
        if 150.0 <= mw <= 400.0:
            filtered_results.append({
                "molecule_id": zincid,
                "smiles": smiles,
                "mw": mw,
                "alogp": descriptors["alogp"],
                "hbd": descriptors["hbd"],
                "hba": descriptors["hba"],
            })
    
    print(f"Found {len(filtered_results)} binders with MW between 150-400")
    
    if len(filtered_results) == 0:
        print("⚠ Error: No compounds match the criteria!")
        return
    
    # Randomly sample 50 compounds (or all if fewer than 50)
    random.seed(42)  # For reproducibility
    n_samples = min(50, len(filtered_results))
    sampled_results = random.sample(filtered_results, n_samples)
    
    print(f"Sampled {len(sampled_results)} compounds")
    
    # Create DataFrame with same columns as previous decoys CSV
    output_df = pd.DataFrame(sampled_results)
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(output_df)} compounds to {output_csv}")
    print(f"\nFirst few rows:")
    print(output_df.head())
    print(f"\nMW range: {output_df['mw'].min():.2f} - {output_df['mw'].max():.2f}")


if __name__ == "__main__":
    main()

