#!/usr/bin/env python3
"""
Generate random drug-like decoy compounds from ChEMBL using the web resource client.

Requirements:
    pip install chembl_webresource_client rdkit-pypi pandas
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import List, Optional

import pandas as pd
from chembl_webresource_client.new_client import new_client


def is_druglike_from_props(mw, alogp, hbd, hba) -> bool:
    """
    Simple Lipinski-like filter + MW 150–400.
    """
    if mw is None or alogp is None or hbd is None or hba is None:
        return False

    return (
        150.0 <= mw <= 400.0 and  # your requested range
        alogp < 5.0 and
        hbd <= 5 and
        hba <= 10
    )


def generate_decoys_from_chembl_api(
    n_samples: int = 100,
    seed: int = 42,
    max_query: int = 20000,
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Sample random drug-like compounds from ChEMBL with MW 150–400.

    Parameters
    ----------
    n_samples : int
        Number of decoys you want.
    seed : int
        RNG seed for reproducibility.
    max_query : int
        Max number of ChEMBL molecules to inspect before stopping.
    output_csv : str or Path or None
        If provided, save the decoys to this CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: molecule_id, smiles, mw, alogp, hbd, hba
    """
    random.seed(seed)

    mol_client = new_client.molecule

    # Pull a bunch of molecules with some lightweight server-side filtering
    # (MW window handled again in is_druglike_from_props to be safe)
    res = mol_client.filter(
        molecule_properties__full_mwt__gte=140,   # a bit looser to be safe
        molecule_properties__full_mwt__lte=450,
    ).only(
        "molecule_chembl_id",
        "molecule_structures",
        "molecule_properties"
    )[:max_query]

    decoys = []

    for entry in res:
        props = entry.get("molecule_properties") or {}
        structs = entry.get("molecule_structures") or {}

        mw = float(props.get("full_mwt") or 0)
        alogp = props.get("alogp")
        alogp = float(alogp) if alogp is not None else None
        hbd = props.get("hbd")
        hba = props.get("hba")

        hbd = int(hbd) if hbd is not None else None
        hba = int(hba) if hba is not None else None

        if not is_druglike_from_props(mw, alogp, hbd, hba):
            continue

        smiles = structs.get("canonical_smiles")
        if not smiles:
            continue

        decoys.append(
            {
                "molecule_id": entry["molecule_chembl_id"],
                "smiles": smiles,
                "mw": mw,
                "alogp": alogp,
                "hbd": hbd,
                "hba": hba,
            }
        )

    if len(decoys) == 0:
        raise RuntimeError("No drug-like molecules found in the queried subset.")

    # Randomly sample requested number (or all if fewer)
    if len(decoys) > n_samples:
        decoys = random.sample(decoys, n_samples)

    df = pd.DataFrame(decoys)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


if __name__ == "__main__":
    # Example run: generate 100 decoys and save to decoys.csv
    df_decoys = generate_decoys_from_chembl_api(
        n_samples=50,
        seed=123,
        max_query=30000,
        output_csv="3pbq_evaluation/3pbq_decoys_150_400mw.csv",
    )
    print(df_decoys.head())
    print(f"Generated {len(df_decoys)} decoys.")
