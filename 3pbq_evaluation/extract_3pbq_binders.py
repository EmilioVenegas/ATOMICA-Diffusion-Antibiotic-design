#!/usr/bin/env python3
"""
Extract known binders for 3PBQ from the antibiotic pockets CSV file.
"""

import pandas as pd
from pathlib import Path

def extract_3pbq_binders(input_csv, output_csv):
    """
    Extract 3PBQ binders from the antibiotic pockets CSV.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    print(f"Total rows in input file: {len(df)}")
    
    # Find rows where:
    # 1. PDB column contains "3PBQ", OR
    # 2. The protein name is "PBP3 (P. aeruginosa)" (which is the section for 3PBQ)
    
    # First, identify the section for PBP3 (P. aeruginosa)
    # The first column contains the protein name
    protein_col = df.columns[0]  # Usually the first column
    
    # Find rows in the PBP3 (P. aeruginosa) section
    pbp3_pa_rows = []
    in_pbp3_pa_section = False
    
    for idx, row in df.iterrows():
        # Check if this row starts a new section
        protein_name = str(row[protein_col]) if pd.notna(row[protein_col]) else ""
        
        if "PBP3 (P. aeruginosa)" in protein_name:
            in_pbp3_pa_section = True
            continue
        
        # Check if we've moved to a new section
        if protein_name and protein_name != "nan" and "PBP3 (P. aeruginosa)" not in protein_name:
            if in_pbp3_pa_section:
                break  # We've left the PBP3 (P. aeruginosa) section
        
        # If we're in the PBP3 (P. aeruginosa) section and have a binder name
        if in_pbp3_pa_section:
            binder_name = str(row['Binder']) if pd.notna(row['Binder']) else ""
            if binder_name and binder_name != "nan":
                pbp3_pa_rows.append(idx)
    
    # Also find rows where PDB contains "3PBQ"
    pdb_col = 'PDB' if 'PDB' in df.columns else df.columns[7]  # Column 7 is PDB
    pdb_3pbq_rows = df[df[pdb_col].astype(str).str.contains('3PBQ', case=False, na=False)].index.tolist()
    
    # Combine both sets of rows
    all_3pbq_rows = list(set(pbp3_pa_rows + pdb_3pbq_rows))
    
    print(f"Found {len(pbp3_pa_rows)} rows in PBP3 (P. aeruginosa) section")
    print(f"Found {len(pdb_3pbq_rows)} rows with PDB containing '3PBQ'")
    print(f"Total unique rows: {len(all_3pbq_rows)}")
    
    # Extract the relevant rows
    df_3pbq = df.loc[all_3pbq_rows].copy()
    
    # Get SMILES column (Canonical SMILES is column 4)
    smiles_col = 'Canonical SMILES' if 'Canonical SMILES' in df.columns else df.columns[4]
    
    # Create output dataframe with required columns for compare_benchmarks.py
    # It expects: molecule_id (or similar) and smiles
    output_data = []
    
    for idx, row in df_3pbq.iterrows():
        binder_name = str(row['Binder']) if pd.notna(row['Binder']) else f"binder_{idx}"
        smiles = str(row[smiles_col]) if pd.notna(row[smiles_col]) else ""
        
        # Skip if no SMILES
        if not smiles or smiles == "nan":
            continue
        
        output_data.append({
            'molecule_id': f"3pbq_{binder_name.lower().replace(' ', '_').replace('/', '_')}",
            'smiles': smiles,
            'binder_name': binder_name,
        })
    
    # Create output DataFrame
    df_output = pd.DataFrame(output_data)
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved {len(df_output)} binders to {output_path}")
    print(f"\nBinder names:")
    for name in df_output['binder_name'].unique():
        print(f"  - {name}")
    
    return df_output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract 3PBQ binders from antibiotic pockets CSV")
    parser.add_argument("--input", type=str,
                       default="3pbq_evaluation/inputs/antibotic_pockets_known_binders.csv",
                       help="Input CSV file path")
    parser.add_argument("--output", type=str,
                       default="3pbq_evaluation/inputs/3pbq_known_binders.csv",
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    extract_3pbq_binders(args.input, args.output)

