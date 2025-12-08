#!/usr/bin/env python3
"""Extract molecular descriptors from 3pbq_dec06_iter2_mol05_scored.csv and save in standard format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main():
    # Input and output paths
    input_csv = Path("3pbq_evaluation/inputs/3pbq_dec06_iter2_mol05_scored.csv")
    output_csv = Path("3pbq_evaluation/inputs/3pbq_dec06_iter2_mol05_descriptors.csv")
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Found {len(df)} molecules")
    print(f"Columns: {list(df.columns)}")
    
    # Map existing columns to standard descriptor names
    # The CSV has: molecular_weight, logP, hydrogen_bond_acceptors, hydrogen_bond_donors
    column_mapping = {
        "molecular_weight": "mw",
        "logP": "alogp",
        "hydrogen_bond_donors": "hbd",
        "hydrogen_bond_acceptors": "hba",
    }
    
    # Extract columns we need
    output_df = df[["molecule_id", "smiles"]].copy()
    
    # Add and rename descriptor columns
    for csv_col, std_name in column_mapping.items():
        if csv_col in df.columns:
            output_df[std_name] = df[csv_col]
        else:
            print(f"⚠ Warning: Column '{csv_col}' not found in CSV")
    
    # Round numeric columns
    if "mw" in output_df.columns:
        output_df["mw"] = output_df["mw"].round(2)
    if "alogp" in output_df.columns:
        output_df["alogp"] = output_df["alogp"].round(2)
    if "hbd" in output_df.columns:
        output_df["hbd"] = output_df["hbd"].astype(int)
    if "hba" in output_df.columns:
        output_df["hba"] = output_df["hba"].astype(int)
    
    # Ensure correct column order
    column_order = ["molecule_id", "smiles", "mw", "alogp", "hbd", "hba"]
    output_df = output_df[[col for col in column_order if col in output_df.columns]]
    
    # Save to new file
    output_df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(output_df)} molecules with descriptors to {output_csv}")
    print(f"\nColumns: {list(output_df.columns)}")
    print(f"\nFirst few rows:")
    print(output_df.head())


if __name__ == "__main__":
    main()

