"""
Rescore PoseBusters-filtered molecules with smina.

Reads the output of evaluate.py (filtered/ directory containing per-pocket
SDFs and manifest.csv), scores each molecule against its receptor PDB using
smina --score_only, and writes a docking.csv.

Usage (from project root):
    python scripts/dock.py \\
        --filtered_dir results/cond_C/filtered \\
        --pdb_dir data/receptor_pdbs \\
        --out results/cond_C/docking.csv \\
        --condition cond_C \\
        --n_jobs 4

    # Compare all conditions in one CSV:
    for cond in baseline_A cond_B cond_C cond_D; do
        python scripts/dock.py \\
            --filtered_dir results/$cond/filtered \\
            --pdb_dir data/receptor_pdbs \\
            --out results/$cond/docking.csv \\
            --condition $cond
    done

Output CSV columns:
    condition, pocket, mol_idx_in_sdf, smiles, qed, sa, logp, pb_pass,
    passes_all, smina_score

Requirements:
    smina binary on PATH  (conda install -c conda-forge smina)
    or set SMINA_BIN env var to point to a smina.static binary.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm

_ROOT     = Path(__file__).resolve().parent.parent
_DIFFSBDD = _ROOT / 'DiffSBDD'
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DIFFSBDD))


SMINA_BIN = os.environ.get('SMINA_BIN', 'smina.static')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pdb(pdb_dir: Path, pocket_name: str):
    exact = pdb_dir / f'{pocket_name}.pdb'
    if exact.exists():
        return exact
    matches = list(pdb_dir.glob(f'{pocket_name}_*.pdb'))
    return matches[0] if matches else None


def _smina_score_sdf(sdf_path: Path, pdb_path: Path):
    """
    Run smina --score_only on sdf_path against pdb_path.
    Returns a list of float scores, one per molecule in the SDF.
    Raises RuntimeError if the smina binary is not found.
    """
    import re
    cmd = [
        SMINA_BIN,
        '-l', str(sdf_path),
        '-r', str(pdb_path),
        '--score_only',
        '--cpu', '1',
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"smina not found at '{SMINA_BIN}'.\n"
            "Install with:  conda install -c conda-forge smina\n"
            "or set:        export SMINA_BIN=/path/to/smina.static"
        )
    except subprocess.TimeoutExpired:
        return []

    matches = re.findall(
        r'Affinity:\s+([+-]?\d+\.?\d*)\s+\(kcal/mol\)',
        result.stdout,
    )
    return [float(x) for x in matches]


def _score_pocket(pocket_name: str, sdf_file: Path, pdb_file: Path):
    """
    Sanitize, batch-score all molecules in one pocket SDF with smina.
    Returns a list of row dicts: {pocket, mol_idx_in_sdf, smina_score}.
    """
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    valid = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            valid.append((i, mol))
        except Exception:
            pass

    if not valid:
        return []

    tmp = Path(tempfile.mktemp(suffix='.sdf'))
    try:
        w = Chem.SDWriter(str(tmp))
        for _, mol in valid:
            w.write(mol)
        w.close()
        scores = _smina_score_sdf(tmp, pdb_file)
    finally:
        tmp.unlink(missing_ok=True)

    rows = []
    for j, (orig_idx, mol) in enumerate(valid):
        rows.append({
            'pocket':         pocket_name,
            'mol_idx_in_sdf': orig_idx,
            'smina_score':    scores[j] if j < len(scores) else float('nan'),
        })
    return rows


# ---------------------------------------------------------------------------
# Main docking routine
# ---------------------------------------------------------------------------

def run_docking(filtered_dir: Path, pdb_dir: Path, n_jobs: int = 1):
    """
    Score all filtered molecules (passes_all == True in manifest.csv)
    against their receptor PDBs.

    Returns a DataFrame merging manifest columns with smina_score.
    """
    manifest_path = filtered_dir / 'manifest.csv'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.csv not found in {filtered_dir}.\n"
            "Run evaluate.py with --pdb_dir first to generate the filtered set."
        )

    manifest = pd.read_csv(manifest_path)
    if 'passes_all' in manifest.columns:
        manifest = manifest[manifest['passes_all']].copy()

    if manifest.empty:
        print("No molecules passed all filters in the manifest.")
        return manifest

    pocket_names = manifest['pocket'].unique().tolist()
    sdf_files    = {p: filtered_dir / f'{p}.sdf' for p in pocket_names}
    pdb_files    = {p: _find_pdb(pdb_dir, p)     for p in pocket_names}

    missing = [p for p, f in pdb_files.items() if f is None]
    if missing:
        print(f"Warning: no PDB found for {len(missing)} pocket(s): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        pocket_names = [p for p in pocket_names if pdb_files[p] is not None]

    scored_rows = []

    if n_jobs == 1:
        for name in tqdm(pocket_names, desc='Docking'):
            sdf = sdf_files[name]
            pdb = pdb_files[name]
            if not sdf.exists():
                continue
            scored_rows.extend(_score_pocket(name, sdf, pdb))
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for name in pocket_names:
                sdf = sdf_files[name]
                pdb = pdb_files[name]
                if not sdf.exists():
                    continue
                f = pool.submit(_score_pocket, name, sdf, pdb)
                futures[f] = name

            for f in tqdm(as_completed(futures), total=len(futures), desc='Docking'):
                try:
                    scored_rows.extend(f.result())
                except Exception as e:
                    print(f"Error scoring {futures[f]}: {e}")

    if not scored_rows:
        return pd.DataFrame()

    scores_df = pd.DataFrame(scored_rows)

    # Merge smina_score back onto the full manifest (keeps QED, SA, etc.)
    merged = manifest.merge(
        scores_df[['pocket', 'mol_idx_in_sdf', 'smina_score']],
        on=['pocket', 'mol_idx_in_sdf'],
        how='left',
    )
    return merged.sort_values('smina_score', na_position='last')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Rescore PoseBusters-filtered molecules with smina.'
    )
    p.add_argument('--filtered_dir', type=Path, required=True,
                   help='filtered/ directory produced by evaluate.py')
    p.add_argument('--pdb_dir',      type=Path, required=True,
                   help='Directory of receptor PDB files')
    p.add_argument('--out',          type=Path, default=None,
                   help='Output CSV (default: <filtered_dir>/docking.csv)')
    p.add_argument('--condition',    type=str,  default='',
                   help='Label added as first column, e.g. "cond_C"')
    p.add_argument('--n_jobs',       type=int,  default=1,
                   help='Parallel smina workers (default: 1)')
    args = p.parse_args()

    out_path = args.out or (args.filtered_dir / 'docking.csv')

    print(f"Scoring filtered molecules from {args.filtered_dir} ...")
    results = run_docking(args.filtered_dir, args.pdb_dir, n_jobs=args.n_jobs)

    if results.empty:
        print("No molecules scored — nothing to save.")
        return

    if args.condition:
        results.insert(0, 'condition', args.condition)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    valid_scores = results['smina_score'].dropna()
    print(f"\n{'='*52}")
    print(f"  Condition          : {args.condition or args.filtered_dir.parent.name}")
    print(f"  Molecules scored   : {len(results)}")
    print(f"  Smina score mean   : {valid_scores.mean():.2f} kcal/mol")
    print(f"  Smina score median : {valid_scores.median():.2f} kcal/mol")
    print(f"  Best score         : {valid_scores.min():.2f} kcal/mol")
    print(f"\nResults → {out_path}")


if __name__ == '__main__':
    main()
