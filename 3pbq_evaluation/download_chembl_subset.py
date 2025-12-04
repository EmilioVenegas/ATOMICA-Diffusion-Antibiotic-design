#!/usr/bin/env python3
"""
Download a subset of ChEMBL compounds using the ChEMBL web service API.
This is more reliable than downloading huge SDF files.
"""

import requests
import pandas as pd
from pathlib import Path
import time
import random

def download_chembl_subset(output_csv, n_samples=50, seed=42, max_compounds=10000):
    """
    Download a random subset of ChEMBL compounds via the web service API.
    
    Args:
        output_csv: Path to output CSV file
        n_samples: Number of compounds to sample (will filter for drug-like)
        seed: Random seed for reproducibility
        max_compounds: Maximum compounds to fetch before filtering (to ensure we have enough)
    """
    random.seed(seed)
    
    print(f"Fetching ChEMBL compounds via web service API...")
    print(f"This may take a few minutes...")
    
    # ChEMBL web service endpoint for compounds
    base_url = "https://www.ebi.ac.uk/chembl/api/data"
    
    # Fetch compounds in batches
    compounds = []
    page_size = 1000
    total_pages = (max_compounds + page_size - 1) // page_size
    
    for page in range(1, min(total_pages + 1, 11)):  # Limit to 10 pages (10k compounds)
        url = f"{base_url}/molecule.json?limit={page_size}&offset={(page-1)*page_size}"
        
        try:
            print(f"  Fetching page {page}/{total_pages}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            molecules = data.get('molecules', [])
            compounds.extend(molecules)
            
            print(f"    Fetched {len(molecules)} compounds (total: {len(compounds)})")
            
            # Be nice to the API
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching page {page}: {e}")
            break
    
    print(f"\nTotal compounds fetched: {len(compounds)}")
    
    # Filter for drug-like compounds and extract SMILES
    print("Filtering for drug-like compounds...")
    druglike_data = []
    
    for mol in compounds:
        try:
            # Extract properties
            props = mol.get('molecule_properties', {})
            if not props:
                continue
                
            mw = props.get('molecular_weight')
            logp = props.get('alogp')
            hbd = props.get('num_hbd')
            hba = props.get('num_hba')
            
            # Check drug-likeness (Lipinski-like)
            if mw and logp and hbd is not None and hba is not None:
                if (150 < mw < 600) and (logp < 5) and (hbd <= 5) and (hba <= 10):
                    smiles = mol.get('molecule_structures', {}).get('canonical_smiles')
                    if smiles:
                        druglike_data.append({
                            'molecule_id': mol.get('molecule_chembl_id', f"chembl_{len(druglike_data)+1}"),
                            'smiles': smiles,
                            'molecular_weight': mw,
                            'logP': logp,
                            'hydrogen_bond_donors': hbd,
                            'hydrogen_bond_acceptors': hba,
                        })
        except Exception as e:
            continue
    
    print(f"Drug-like compounds found: {len(druglike_data)}")
    
    if len(druglike_data) < n_samples:
        print(f"Warning: Only {len(druglike_data)} drug-like molecules found, but {n_samples} requested.")
        print(f"Using all {len(druglike_data)} available molecules.")
        n_samples = len(druglike_data)
    
    # Sample random subset
    print(f"Sampling {n_samples} random compounds...")
    random_subset = random.sample(druglike_data, n_samples)
    
    # Save to CSV
    df = pd.DataFrame(random_subset)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} decoy compounds to {output_path}")
    print(f"  Columns: {', '.join(df.columns)}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ChEMBL subset via web API")
    parser.add_argument("--output", type=str, 
                       default="3pbq_evaluation/inputs/3pbq_decoys.csv",
                       help="Output CSV file path")
    parser.add_argument("--n-samples", type=int, default=50,
                       help="Number of compounds to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max-compounds", type=int, default=10000,
                       help="Maximum compounds to fetch from API")
    
    args = parser.parse_args()
    
    download_chembl_subset(args.output, args.n_samples, args.seed, args.max_compounds)

