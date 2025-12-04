#!/usr/bin/env python3

import random
import pandas as pd
from pathlib import Path
from chembl_webresource_client.new_client import new_client

# -----------------------------
# DRUG-LIKENESS FILTER FUNCTION
# -----------------------------
def is_druglike(m):
    """Return True if compound meets simple drug-like criteria."""
    try:
        mw = float(m["molecule_properties"]["full_mwt"])
        logp = float(m["molecule_properties"]["alogp"])
        hbd = int(m["molecule_properties"]["hbd"])
        hba = int(m["molecule_properties"]["hba"])
    except (TypeError, ValueError):
        return False
    
    return (150 < mw < 600) and (logp < 5) and (hbd <= 5) and (hba <= 10)

# -----------------------------
# SAMPLER FUNCTION
# -----------------------------
def generate_random_druglike_csv(
    output_csv="3pbq_evaluation/inputs/3pbq_decoys.csv",
    n_samples=100,
    seed=42,
):
    random.seed(seed)

    molecule = new_client.molecule

    # Query molecules that have structures + props
    results = molecule.filter(
        molecule_properties__isnull=False,
        molecule_structures__isnull=False
    ).only(
        "molecule_chembl_id",
        "molecule_properties",
        "molecule_structures"
    )

    print("Querying ChEMBL database...")
    rows = []
    max_iterations = 10000  # Limit iterations to avoid infinite loops
    target_collect = n_samples * 3  # Collect 3x for good random sampling
    
    for i, m in enumerate(results):
        if i >= max_iterations:
            break
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i} molecules, found {len(rows)} drug-like...")
        
        if "molecule_properties" not in m or "molecule_structures" not in m:
            continue
        if is_druglike(m):
            smiles = m["molecule_structures"].get("canonical_smiles")
            if smiles:
                rows.append({
                    "chembl_id": m["molecule_chembl_id"],
                    "smiles": smiles,
                    "mw": m["molecule_properties"]["full_mwt"],
                    "alogp": m["molecule_properties"]["alogp"],
                    "hbd": m["molecule_properties"]["hbd"],
                    "hba": m["molecule_properties"]["hba"],
                })
                # Stop early if we have enough candidates for sampling
                if len(rows) >= target_collect:
                    print(f"  Collected {len(rows)} drug-like molecules")
                    break

    if len(rows) == 0:
        print("Error: No drug-like molecules found!")
        return None
    
    if len(rows) < n_samples:
        print(f"Only found {len(rows)} drug-like molecules, returning all.")
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(random.sample(rows, n_samples))

    # Create output directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} molecules → {output_csv}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    generate_random_druglike_csv("3pbq_evaluation/inputs/3pbq_decoys.csv", n_samples=50)
