#!/usr/bin/env python3
"""
Add molecular descriptors to CSV files containing SMILES.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path

def calculate_descriptors(smiles):
    """
    Calculate molecular descriptors from SMILES.
    
    Returns a dictionary with descriptors, or None if SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logP': Descriptors.MolLogP(mol),
            'hydrogen_bond_donors': Descriptors.NumHDonors(mol),
            'hydrogen_bond_acceptors': Descriptors.NumHAcceptors(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
            'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
            'tpsa': Descriptors.TPSA(mol),  # Topological Polar Surface Area
            'num_rings': Descriptors.RingCount(mol),
        }
    except Exception:
        return None

def add_descriptors_to_csv(input_csv, output_csv=None, smiles_column='smiles'):
    """
    Add molecular descriptors to a CSV file.
    
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
        desc = calculate_descriptors(smiles)
        if desc is None:
            print(f"  Warning: Invalid SMILES at row {idx+1}: {smiles}")
            # Add NaN values for failed molecules
            desc = {k: None for k in [
                'molecular_weight', 'logP', 'hydrogen_bond_donors', 
                'hydrogen_bond_acceptors', 'num_rotatable_bonds',
                'num_aromatic_rings', 'num_saturated_rings', 
                'num_heteroatoms', 'tpsa', 'num_rings'
            ]}
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
    
    parser = argparse.ArgumentParser(description="Add molecular descriptors to CSV files")
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file path (if not provided, overwrites input)")
    parser.add_argument("--smiles-column", type=str, default="smiles",
                       help="Name of the SMILES column")
    
    args = parser.parse_args()
    
    add_descriptors_to_csv(args.input, args.output, args.smiles_column)

