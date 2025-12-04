#!/usr/bin/env python3
"""
Remove duplicate descriptor columns, keeping only mw, alogp, hbd, hba.
"""

import pandas as pd
from pathlib import Path

def remove_duplicate_descriptors(input_csv, output_csv=None):
    """
    Remove duplicate descriptor columns, keeping only mw, alogp, hbd, hba.
    """
    df = pd.read_csv(input_csv)
    
    print(f"Original columns: {df.columns.tolist()}")
    
    # Columns to remove (longer names)
    cols_to_remove = [
        'molecular_weight',
        'logP',
        'hydrogen_bond_donors',
        'hydrogen_bond_acceptors',
        'num_rotatable_bonds',
        'num_aromatic_rings',
        'num_saturated_rings',
        'num_heteroatoms',
        'tpsa',
        'num_rings'
    ]
    
    # Remove columns that exist
    existing_cols_to_remove = [col for col in cols_to_remove if col in df.columns]
    if existing_cols_to_remove:
        print(f"Removing columns: {existing_cols_to_remove}")
        df = df.drop(columns=existing_cols_to_remove)
    
    # Save to output file
    if output_csv is None:
        output_csv = input_csv
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {len(df)} molecules to {output_path}")
    print(f"  Remaining columns: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove duplicate descriptor columns")
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file path (if not provided, overwrites input)")
    
    args = parser.parse_args()
    
    remove_duplicate_descriptors(args.input, args.output)

