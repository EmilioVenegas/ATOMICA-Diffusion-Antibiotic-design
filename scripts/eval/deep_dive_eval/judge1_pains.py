#!/usr/bin/env python3
"""Judge 1: The Chemist - RDKit PAINS filter to detect toxic/problematic molecules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import FilterCatalog


def load_pains_filter() -> FilterCatalog.FilterCatalog:
    """Load the RDKit PAINS filter catalog."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return catalog


def check_pains(smiles: str, catalog: FilterCatalog.FilterCatalog | None = None) -> bool:
    """Check if a SMILES string matches any PAINS filter.
    
    Args:
        smiles: SMILES string to check
        catalog: Optional pre-loaded PAINS catalog (for efficiency when checking many molecules)
    
    Returns:
        True if molecule matches PAINS (problematic), False if safe
    """
    if catalog is None:
        catalog = load_pains_filter()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # Invalid molecules are considered problematic
    
    return catalog.HasMatch(mol)


def check_pains_batch(smiles_list: List[str]) -> Dict[str, bool]:
    """Check multiple SMILES strings for PAINS.
    
    Args:
        smiles_list: List of SMILES strings
    
    Returns:
        Dictionary mapping SMILES to PAINS status (True = problematic, False = safe)
    """
    catalog = load_pains_filter()
    results = {}
    
    for smiles in smiles_list:
        results[smiles] = check_pains(smiles, catalog)
    
    return results


def check_pains_from_csv(csv_path: Path, smiles_column: str = "smiles", output_path: Path | None = None) -> pd.DataFrame:
    """Check PAINS for molecules in a CSV file.
    
    Args:
        csv_path: Path to CSV file with SMILES column
        smiles_column: Name of the column containing SMILES
        output_path: Optional path to save results CSV
    
    Returns:
        DataFrame with original data plus 'pains_alert' column (True = problematic)
    """
    df = pd.read_csv(csv_path)
    
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    catalog = load_pains_filter()
    df['pains_alert'] = df[smiles_column].apply(lambda s: check_pains(s, catalog))
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved PAINS results to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge 1: PAINS filter for toxic/problematic molecules")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file with SMILES column")
    parser.add_argument("--smiles-column", default="smiles", help="Name of SMILES column (default: smiles)")
    parser.add_argument("--output", type=Path, help="Output CSV file with PAINS results")
    
    args = parser.parse_args()
    
    df = check_pains_from_csv(args.input, args.smiles_column, args.output)
    
    n_total = len(df)
    n_pains = df['pains_alert'].sum()
    n_safe = n_total - n_pains
    
    print(f"\nPAINS Filter Results:")
    print(f"  Total molecules: {n_total}")
    print(f"  PAINS alerts (problematic): {n_pains} ({100*n_pains/n_total:.1f}%)")
    print(f"  Safe molecules: {n_safe} ({100*n_safe/n_total:.1f}%)")

