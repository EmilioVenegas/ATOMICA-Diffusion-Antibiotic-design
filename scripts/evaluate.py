"""
Evaluate a directory of per-pocket SDF files.

Computes: Validity, Connectivity, Uniqueness, Novelty, QED, SA, LogP,
Lipinski, Diversity, and optionally PoseBusters pass rate.

Usage (from project root):
    # Chemistry-only metrics (no protein structure needed):
    python scripts/evaluate.py \\
        --sdf_dir results/baseline_A \\
        --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \\
        --out results/baseline_A/metrics.json

    # With PoseBusters (requires per-pocket PDB files):
    python scripts/evaluate.py \\
        --sdf_dir results/cond_C \\
        --pdb_dir data/receptor_pdbs \\
        --out results/cond_C/metrics.json

PDB matching: looks for <pdb_dir>/<pocket_name>.pdb or <pdb_dir>/<pocket_name>_*.pdb.
SDF files are expected to match the naming produced by run_baseline.py:
    <sdf_dir>/pocket_0000.sdf, pocket_0001.sdf, ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm

_ROOT    = Path(__file__).resolve().parent.parent
_DIFFSBDD = _ROOT / 'DiffSBDD'
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DIFFSBDD))

from analysis.metrics import BasicMolecularMetrics, MoleculeProperties


def load_sdf_dir(sdf_dir: Path):
    """Return (pocket_name, [RDKit mol, ...]) for every SDF in the directory."""
    pockets = []
    for sdf_file in sorted(sdf_dir.glob('*.sdf')):
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
        mols = [m for m in suppl if m is not None]
        pockets.append((sdf_file.stem, mols))
    return pockets


def compute_chemistry_metrics(pockets, smiles_ref=None):
    """
    Run validity, connectivity, uniqueness, novelty, and per-molecule
    druglikeness metrics.  Returns a flat dict of mean ± std values.
    """
    all_mols_flat  = [m for _, mols in pockets for m in mols]
    pocket_mol_lists = [mols for _, mols in pockets]

    metrics = MoleculeProperties()

    results = {}

    # Per-pocket druglikeness: QED, SA, LogP, Lipinski, Diversity
    all_qed, all_sa, all_logp, all_lipo, per_div = [], [], [], [], []
    valid_count = 0
    for mols in tqdm(pocket_mol_lists, desc='Chemistry metrics'):
        valid_mols = []
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
                valid_mols.append(mol)
                valid_count += 1
            except Exception:
                pass
        if not valid_mols:
            continue
        all_qed.append([metrics.calculate_qed(m)      for m in valid_mols])
        all_sa.append( [metrics.calculate_sa(m)       for m in valid_mols])
        all_logp.append([metrics.calculate_logp(m)    for m in valid_mols])
        all_lipo.append([metrics.calculate_lipinski(m) for m in valid_mols])
        per_div.append(metrics.calculate_diversity(valid_mols))

    total_generated = sum(len(mols) for _, mols in pockets)
    results['validity']    = valid_count / total_generated if total_generated else 0.0
    results['n_pockets']   = len(pockets)
    results['n_generated'] = total_generated
    results['n_valid']     = valid_count

    def _stats(lists, name):
        flat = [x for sublist in lists for x in sublist]
        results[f'{name}_mean'] = float(np.mean(flat))  if flat else 0.0
        results[f'{name}_std']  = float(np.std(flat))   if flat else 0.0
        results[f'{name}_median'] = float(np.median(flat)) if flat else 0.0

    _stats(all_qed,  'qed')
    _stats(all_sa,   'sa')
    _stats(all_logp, 'logp')
    _stats(all_lipo, 'lipinski')
    results['diversity_mean'] = float(np.mean(per_div)) if per_div else 0.0
    results['diversity_std']  = float(np.std(per_div))  if per_div else 0.0

    # Uniqueness (within each pocket)
    unique_counts = []
    for mols in pocket_mol_lists:
        smiles_set = set()
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol)
                smiles_set.add(smi)
            except Exception:
                pass
        unique_counts.append(len(smiles_set) / len(mols) if mols else 0.0)
    results['uniqueness_mean'] = float(np.mean(unique_counts)) if unique_counts else 0.0

    # Novelty vs reference SMILES
    if smiles_ref is not None:
        ref_smiles = set(np.load(smiles_ref, allow_pickle=True).tolist())
        gen_smiles = set()
        for mols in pocket_mol_lists:
            for mol in mols:
                try:
                    Chem.SanitizeMol(mol)
                    gen_smiles.add(Chem.MolToSmiles(mol))
                except Exception:
                    pass
        novel = gen_smiles - ref_smiles
        results['novelty'] = len(novel) / len(gen_smiles) if gen_smiles else 0.0
    else:
        results['novelty'] = None

    return results


def compute_posebusters(pockets, sdf_dir: Path, pdb_dir: Path):
    """
    Run PoseBusters on each pocket's SDF file against its receptor PDB.
    sdf_dir: directory containing <pocket_name>.sdf files (from run_baseline.py)
    pdb_dir: directory containing <pocket_name>.pdb files (from extract_pocket_pdbs.py)
    Returns per-pocket pass rates and the overall mean.
    """
    try:
        from posebusters import PoseBusters
    except ImportError:
        print("PoseBusters not installed. Run: pip install posebusters")
        return None

    pb = PoseBusters(config='dock')
    pass_rates = []
    missing_pdb, missing_sdf = 0, 0

    for pocket_name, _ in tqdm(pockets, desc='PoseBusters'):
        pdb_file = pdb_dir / f'{pocket_name}.pdb'
        sdf_file = sdf_dir / f'{pocket_name}.sdf'

        if not pdb_file.exists():
            missing_pdb += 1
            continue
        if not sdf_file.exists():
            missing_sdf += 1
            continue

        try:
            results = pb.bust(str(sdf_file),
                              mol_cond=str(pdb_file),
                              full_report=False)
            # v0.6.x returns individual check columns; 'all_pass' only exists in older versions
            if 'all_pass' in results.columns:
                all_pass = results['all_pass']
            else:
                all_pass = results.select_dtypes(bool).all(axis=1)
            rate = float(all_pass.mean())
            pass_rates.append({'pocket': pocket_name, 'pb_pass_rate': rate})
        except Exception as e:
            print(f"PoseBusters failed for {pocket_name}: {e}")

    if missing_pdb:
        print(f"Warning: {missing_pdb} pockets had no matching PDB in {pdb_dir}")
    if missing_sdf:
        print(f"Warning: {missing_sdf} pockets had no matching SDF in {sdf_dir}")

    if not pass_rates:
        return None

    mean_rate = float(np.mean([r['pb_pass_rate'] for r in pass_rates]))
    return {'pb_pass_rate_mean': mean_rate, 'per_pocket': pass_rates}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sdf_dir',    type=Path, required=True,
                   help='Directory of per-pocket SDF files from run_baseline.py')
    p.add_argument('--pdb_dir',    type=Path, default=None,
                   help='Directory of receptor PDB files (enables PoseBusters)')
    p.add_argument('--smiles_ref', type=Path,
                   default='DiffSBDD/data/crossdocked_smiles.npy',
                   help='Reference SMILES .npy for novelty calculation')
    p.add_argument('--out',        type=Path, default=None,
                   help='Output JSON path (default: <sdf_dir>/metrics.json)')
    p.add_argument('--label',      type=str, default='',
                   help='Short label for this condition (e.g. "A-baseline")')
    args = p.parse_args()

    out_path = args.out or (args.sdf_dir / 'metrics.json')

    smiles_ref = args.smiles_ref if args.smiles_ref.exists() else None
    if smiles_ref is None:
        print(f"Warning: reference SMILES not found at {args.smiles_ref} — skipping novelty")

    print(f"Loading SDF files from {args.sdf_dir} ...")
    pockets = load_sdf_dir(args.sdf_dir)
    if not pockets:
        raise FileNotFoundError(f"No SDF files found in {args.sdf_dir}")
    print(f"  {len(pockets)} pockets, "
          f"{sum(len(m) for _, m in pockets)} total molecules")

    metrics = {'label': args.label, 'sdf_dir': str(args.sdf_dir)}

    print("\nComputing chemistry metrics ...")
    chem = compute_chemistry_metrics(pockets, smiles_ref=smiles_ref)
    metrics.update(chem)

    if args.pdb_dir is not None:
        print("\nRunning PoseBusters ...")
        pb_results = compute_posebusters(pockets, args.sdf_dir, args.pdb_dir)
        if pb_results:
            metrics['posebusters'] = pb_results

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary table
    print(f"\n{'='*50}")
    print(f"  Condition: {args.label or args.sdf_dir.name}")
    print(f"{'='*50}")
    print(f"  Pockets evaluated : {metrics['n_pockets']}")
    print(f"  Molecules generated: {metrics['n_generated']}")
    print(f"  Validity           : {metrics['validity']:.1%}")
    print(f"  Uniqueness         : {metrics['uniqueness_mean']:.1%}")
    if metrics['novelty'] is not None:
        print(f"  Novelty            : {metrics['novelty']:.1%}")
    print(f"  QED                : {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}")
    print(f"  SA                 : {metrics['sa_mean']:.3f} ± {metrics['sa_std']:.3f}")
    print(f"  LogP               : {metrics['logp_mean']:.3f} ± {metrics['logp_std']:.3f}")
    print(f"  Lipinski           : {metrics['lipinski_mean']:.3f} ± {metrics['lipinski_std']:.3f}")
    print(f"  Diversity          : {metrics['diversity_mean']:.3f} ± {metrics['diversity_std']:.3f}")
    if 'posebusters' in metrics:
        print(f"  PoseBusters        : {metrics['posebusters']['pb_pass_rate_mean']:.1%}")
    print(f"\nFull metrics saved to {out_path}")


if __name__ == '__main__':
    main()
