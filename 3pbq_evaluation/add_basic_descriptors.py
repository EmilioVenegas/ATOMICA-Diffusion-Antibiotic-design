#!/usr/bin/env python3
"""
Add basic molecular descriptors (mw, alogp, hbd, hba) to CSV files.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path

def calculate_basic_descriptors(smiles):
    """
    Calculate basic molecular descriptors from SMILES.
    
    Returns a dictionary with mw, alogp, hbd, hba, or None if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'mw': Descriptors.MolWt(mol),
            'alogp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
        }
    except Exception:
        return None

def add_descriptors_to_csv(input_csv, output_csv=None, smiles_column='smiles'):
    """
    Add basic molecular descriptors to a CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (if None, overwrites input)
        smiles_column: Name of the column containing SMILES
    """
    df = pd.read_csv(input_csv)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    print(f"Processing {len(df)} molecules from {input_csv}...")
    
    # Calculate descriptors for each molecule
    descriptors_list = []
    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        desc = calculate_basic_descriptors(smiles)
        if desc is None:
            print(f"  Warning: Invalid SMILES at row {idx+1}: {smiles}")
            desc = {'mw': None, 'alogp': None, 'hbd': None, 'hba': None}
        descriptors_list.append(desc)
    
    # Add descriptor columns to dataframe
    desc_df = pd.DataFrame(descriptors_list)
    
    # Merge with original dataframe
    # Check if any descriptor columns already exist and remove them first
    existing_cols = [col for col in desc_df.columns if col in df.columns]
    if existing_cols:
        print(f"  Replacing existing columns: {existing_cols}")
        df = df.drop(columns=existing_cols)
    
    df = pd.concat([df, desc_df], axis=1)
    
    # Save to output file
    if output_csv is None:
        output_csv = input_csv
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(df)} molecules with descriptors to {output_path}")
    print(f"  Added descriptors: {', '.join(desc_df.columns)}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add basic molecular descriptors (mw, alogp, hbd, hba) to CSV files")
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file path (if not provided, overwrites input)")
    parser.add_argument("--smiles-column", type=str, default="smiles",
                       help="Name of the SMILES column")
    
    args = parser.parse_args()
    
    add_descriptors_to_csv(args.input, args.output, args.smiles_column)

