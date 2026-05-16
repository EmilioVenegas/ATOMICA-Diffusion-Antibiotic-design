"""
Evaluate a directory of per-pocket SDF files.

Computes: Validity, Uniqueness, Novelty, QED, SA, LogP, Lipinski, Diversity,
and optionally runs the three-stage filter pipeline:

    valid (RDKit sanitize)
        → PoseBusters pass          ← geometry / protein-ligand clash
        → QED > qed_min AND SA < sa_max  ← druglikeness

Filtered molecules are written to <filtered_dir>/<pocket_name>.sdf and a
manifest.csv is saved there for use by scripts/dock.py.

Usage (from project root):
    # Chemistry-only metrics (no protein structure needed):
    python scripts/evaluate.py \\
        --sdf_dir results/baseline_A \\
        --smiles_ref DiffSBDD/data/crossdocked_smiles.npy \\
        --out results/baseline_A/metrics.json

    # Full pipeline with PoseBusters + filtering:
    python scripts/evaluate.py \\
        --sdf_dir results/cond_C \\
        --pdb_dir data/receptor_pdbs \\
        --out results/cond_C/metrics.json \\
        --qed_min 0.5 --sa_max 5.0

PDB matching: looks for <pdb_dir>/<pocket_name>.pdb or <pdb_dir>/<pocket_name>_*.pdb.
Filtered molecules land in <sdf_dir>/filtered/ by default (override with --filtered_dir).
"""

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
# Some RDKit routines (tautomer enumerator, kekulization) emit via the
# Boost logger even when rdApp.* is disabled — silence the whole logger.
RDLogger.logger().setLevel(RDLogger.CRITICAL)
from tqdm import tqdm

_ROOT     = Path(__file__).resolve().parent.parent
_DIFFSBDD = _ROOT / 'DiffSBDD'
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DIFFSBDD))

from analysis.metrics import BasicMolecularMetrics, MoleculeProperties


# ---------------------------------------------------------------------------
# Standardization helpers
# ---------------------------------------------------------------------------

# Lazy singleton — TautomerEnumerator construction is expensive
_TE = None

def _tautomer_enumerator():
    global _TE
    if _TE is None:
        _TE = rdMolStandardize.TautomerEnumerator()
        # Default limits (1000) cause multi-second stalls on complex molecules.
        # 100 transforms is enough for canonical tautomer selection.
        _TE.SetMaxTautomers(100)
        _TE.SetMaxTransforms(100)
    return _TE


# Any 3-membered ring (cyclopropane, epoxide, aziridine …)
_STRAINED_3_SMARTS = Chem.MolFromSmarts('[R]1~[R]~[R]~1')


def _standardize_mol(mol):
    """
    Apply Cleanup (charge/valence normalisation) then tautomer canonicalisation.
    Returns the standardised mol, or the original mol if standardisation fails.
    """
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _tautomer_enumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        pass  # keep whatever we had
    return mol


def _has_strained_ring(mol):
    """True if the molecule contains any 3-membered ring."""
    return mol.HasSubstructMatch(_STRAINED_3_SMARTS)


def _load_checkpoint(ckpt_path: Path):
    """Return (completed_pockets, manifest_rows, pass_rates, totals) from a checkpoint file."""
    if not ckpt_path.exists():
        return set(), [], [], {'n_valid': 0, 'n_pb_pass': 0, 'n_filtered': 0}
    import json as _json
    with open(ckpt_path) as f:
        ckpt = _json.load(f)
    return (
        set(ckpt['completed_pockets']),
        ckpt['manifest_rows'],
        ckpt['pass_rates'],
        ckpt['totals'],
    )


def _save_checkpoint(ckpt_path: Path, completed, manifest_rows, pass_rates, totals):
    """Atomically write checkpoint — write to .tmp then rename so a crash leaves the old file intact."""
    import json as _json
    tmp = ckpt_path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        _json.dump({
            'completed_pockets': sorted(completed),
            'manifest_rows':     manifest_rows,
            'pass_rates':        pass_rates,
            'totals':            totals,
        }, f)
    tmp.replace(ckpt_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sdf_dir(sdf_dir: Path):
    """Return [(pocket_name, [mol, ...]), ...] for every SDF in sdf_dir."""
    pockets = []
    for sdf_file in sorted(sdf_dir.glob('*.sdf')):
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
        mols = [m for m in suppl if m is not None]
        pockets.append((sdf_file.stem, mols))
    return pockets


def _find_pdb(pdb_dir: Path, pocket_name: str):
    """Return the PDB path for a pocket, or None if not found."""
    exact = pdb_dir / f'{pocket_name}.pdb'
    if exact.exists():
        return exact
    matches = list(pdb_dir.glob(f'{pocket_name}_*.pdb'))
    return matches[0] if matches else None


def _get_valid_mols(sdf_file: Path):
    """
    Load, sanitize, and standardize molecules from an SDF.
    Returns [(original_idx, mol), ...] — original_idx is the 0-based position
    in the file (including None/failed entries so indices stay aligned).
    Standardization (Cleanup + tautomer canonicalization) runs after sanitization;
    if it raises, the sanitized-but-unstandardized mol is kept.
    """
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    valid = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        mol = _standardize_mol(mol)
        valid.append((i, mol))
    return valid


# ---------------------------------------------------------------------------
# Chemistry metrics (no protein structure required)
# ---------------------------------------------------------------------------

def compute_chemistry_metrics(pockets, smiles_ref=None):
    """
    Validity, uniqueness, novelty, QED, SA, LogP, Lipinski, Diversity.
    Returns a flat dict of aggregate statistics.
    """
    pocket_mol_lists = [mols for _, mols in pockets]
    mp = MoleculeProperties()
    results = {}

    all_qed, all_sa, all_logp, all_lipo, per_div = [], [], [], [], []
    valid_count = 0

    for mols in tqdm(pocket_mol_lists, desc='Chemistry metrics'):
        valid_mols = []
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            mol = _standardize_mol(mol)
            valid_mols.append(mol)
            valid_count += 1
        if not valid_mols:
            continue
        all_qed.append([mp.calculate_qed(m)      for m in valid_mols])
        all_sa.append( [mp.calculate_sa(m)        for m in valid_mols])
        all_logp.append([mp.calculate_logp(m)     for m in valid_mols])
        all_lipo.append([mp.calculate_lipinski(m)  for m in valid_mols])
        per_div.append(mp.calculate_diversity(valid_mols))

    total_generated = sum(len(mols) for _, mols in pockets)
    results['validity']    = valid_count / total_generated if total_generated else 0.0
    results['n_pockets']   = len(pockets)
    results['n_generated'] = total_generated
    results['n_valid']     = valid_count

    def _stats(lists, name):
        flat = [x for sub in lists for x in sub]
        results[f'{name}_mean']   = float(np.mean(flat))   if flat else 0.0
        results[f'{name}_std']    = float(np.std(flat))    if flat else 0.0
        results[f'{name}_median'] = float(np.median(flat)) if flat else 0.0

    _stats(all_qed,  'qed')
    _stats(all_sa,   'sa')
    _stats(all_logp, 'logp')
    _stats(all_lipo, 'lipinski')
    results['diversity_mean'] = float(np.mean(per_div)) if per_div else 0.0
    results['diversity_std']  = float(np.std(per_div))  if per_div else 0.0

    # Uniqueness within each pocket
    unique_counts = []
    for mols in pocket_mol_lists:
        smiles_set = set()
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            smiles_set.add(Chem.MolToSmiles(_standardize_mol(mol)))
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
                except Exception:
                    continue
                gen_smiles.add(Chem.MolToSmiles(_standardize_mol(mol)))
        novel = gen_smiles - ref_smiles
        results['novelty'] = len(novel) / len(gen_smiles) if gen_smiles else 0.0
    else:
        results['novelty'] = None

    return results


# ---------------------------------------------------------------------------
# Per-pocket worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _run_pocket(pocket_name, sdf_file, pdb_file, qed_min, sa_max, filtered_dir):
    """
    Run all three filter stages for one pocket and return a result dict.
    Creates its own PoseBusters + MoleculeProperties so it is safe to call
    from forked worker processes.
    """
    from posebusters import PoseBusters
    # Re-apply suppression: posebusters re-enables RDKit logging on import.
    from rdkit import RDLogger as _RDL
    _RDL.DisableLog('rdApp.*')
    _RDL.logger().setLevel(_RDL.CRITICAL)
    mp = MoleculeProperties()
    pb = PoseBusters(config='dock')

    valid = _get_valid_mols(Path(sdf_file))
    if not valid:
        return {'pocket': pocket_name, 'rows': [], 'pass_rate': None,
                'n_valid': 0, 'n_pb_pass': 0, 'n_filtered': 0}

    tmp_sdf = Path(tempfile.mktemp(suffix='.sdf'))
    try:
        w = Chem.SDWriter(str(tmp_sdf))
        for _, mol in valid:
            w.write(mol)
        w.close()
        pb_df = pb.bust(str(tmp_sdf), mol_cond=str(pdb_file), full_report=False)
        pb_pass_arr = (
            pb_df['all_pass'] if 'all_pass' in pb_df.columns
            else pb_df.select_dtypes(bool).all(axis=1)
        ).to_numpy(dtype=bool)
    except Exception:
        pb_pass_arr = np.zeros(len(valid), dtype=bool)
    finally:
        tmp_sdf.unlink(missing_ok=True)

    if len(pb_pass_arr) < len(valid):
        pb_pass_arr = np.pad(pb_pass_arr, (0, len(valid) - len(pb_pass_arr)),
                             constant_values=False)
    else:
        pb_pass_arr = pb_pass_arr[:len(valid)]

    rows      = []
    n_filtered = 0
    fdir = Path(filtered_dir) if filtered_dir else None
    writer = Chem.SDWriter(str(fdir / f'{pocket_name}.sdf')) if fdir else None

    for j, (orig_idx, mol) in enumerate(valid):
        pb_pass  = bool(pb_pass_arr[j])
        qed_val  = mp.calculate_qed(mol)
        sa_val   = mp.calculate_sa(mol)
        logp_val = mp.calculate_logp(mol)
        smiles   = Chem.MolToSmiles(mol)
        passes   = pb_pass and qed_val >= qed_min and sa_val <= sa_max
        rows.append({
            'pocket':         pocket_name,
            'mol_idx_in_sdf': orig_idx,
            'smiles':         smiles,
            'qed':            round(qed_val,  4),
            'sa':             round(sa_val,   4),
            'logp':           round(logp_val, 4),
            'pb_pass':        pb_pass,
            'passes_all':     passes,
            'near_miss':      pb_pass and not passes,
            'strained_ring':  _has_strained_ring(mol),
        })
        if passes:
            n_filtered += 1
            if writer:
                writer.write(mol)

    if writer:
        writer.close()

    return {
        'pocket':    pocket_name,
        'rows':      rows,
        'pass_rate': {'pocket': pocket_name, 'pb_pass_rate': float(pb_pass_arr.mean())},
        'n_valid':   len(valid),
        'n_pb_pass': int(pb_pass_arr.sum()),
        'n_filtered': n_filtered,
    }


# ---------------------------------------------------------------------------
# Three-stage filter pipeline (requires receptor PDB files)
# ---------------------------------------------------------------------------

def compute_posebusters_and_filter(
    pockets,
    sdf_dir: Path,
    pdb_dir: Path,
    qed_min: float = 0.5,
    sa_max: float = 5.0,
    filtered_dir: Path = None,
    n_jobs: int = 1,
):
    """
    Per-pocket pipeline:
      Stage 1 — RDKit sanitize + standardize (validity)
      Stage 2 — PoseBusters (geometry, clashes, protein contacts)
      Stage 3 — QED >= qed_min AND SA <= sa_max (druglikeness)

    Filtered molecules are written to filtered_dir/<pocket_name>.sdf.
    Progress is checkpointed after every pocket; rerun the same command to resume.
    Use n_jobs > 1 to parallelise across pockets (each worker owns its own
    PoseBusters instance — no shared state).

    Returns:
        summary  dict  — pb_pass_rate_mean, per_pocket list, filter_counts
        manifest DataFrame — one row per valid molecule
    """
    try:
        from posebusters import PoseBusters  # noqa — verify install before spawning workers
    except ImportError:
        print("PoseBusters not installed. Run: pip install posebusters")
        return None, None

    if filtered_dir is not None:
        filtered_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = (filtered_dir / 'checkpoint.json') if filtered_dir else None
    completed, manifest_rows, pass_rates, totals = (
        _load_checkpoint(ckpt_path) if ckpt_path else
        (set(), [], [], {'n_valid': 0, 'n_pb_pass': 0, 'n_filtered': 0})
    )
    total_valid    = totals['n_valid']
    total_pb_pass  = totals['n_pb_pass']
    total_filtered = totals['n_filtered']

    # Build work list: skip completed pockets and those with no PDB
    work        = []
    missing_pdb = 0
    for pocket_name, _ in pockets:
        if pocket_name in completed:
            continue
        sdf_file = sdf_dir / f'{pocket_name}.sdf'
        if not sdf_file.exists():
            continue
        pdb_file = _find_pdb(pdb_dir, pocket_name)
        if pdb_file is None:
            missing_pdb += 1
            continue
        work.append((pocket_name, sdf_file, pdb_file))

    if completed:
        print(f"  Resuming: {len(completed)} pockets already done, {len(work)} remaining.")
    if missing_pdb:
        print(f"Warning: {missing_pdb} pockets had no matching PDB in {pdb_dir}")

    def _collect(result):
        nonlocal total_valid, total_pb_pass, total_filtered
        if result['pass_rate']:
            pass_rates.append(result['pass_rate'])
        manifest_rows.extend(result['rows'])
        total_valid    += result['n_valid']
        total_pb_pass  += result['n_pb_pass']
        total_filtered += result['n_filtered']
        completed.add(result['pocket'])
        if ckpt_path:
            _save_checkpoint(ckpt_path, completed, manifest_rows, pass_rates,
                             {'n_valid': total_valid, 'n_pb_pass': total_pb_pass,
                              'n_filtered': total_filtered})

    fdir_str = str(filtered_dir) if filtered_dir else None

    if n_jobs == 1:
        for pocket_name, sdf_file, pdb_file in tqdm(work, desc='PoseBusters + filter'):
            _collect(_run_pocket(pocket_name, sdf_file, pdb_file,
                                 qed_min, sa_max, fdir_str))
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for pocket_name, sdf_file, pdb_file in work:
                f = pool.submit(_run_pocket, pocket_name, sdf_file, pdb_file,
                                qed_min, sa_max, fdir_str)
                futures[f] = pocket_name
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f'PoseBusters + filter ({n_jobs} workers)'):
                try:
                    _collect(f.result())
                except Exception as e:
                    print(f"Error on pocket {futures[f]}: {e}")

    manifest_df_tmp = pd.DataFrame(manifest_rows)
    n_near_miss     = int(manifest_df_tmp['near_miss'].sum())      if not manifest_df_tmp.empty else 0
    n_strained      = int(manifest_df_tmp['strained_ring'].sum())  if not manifest_df_tmp.empty else 0
    n_salvageable   = int(
        (manifest_df_tmp['near_miss'] & manifest_df_tmp['strained_ring']).sum()
    ) if not manifest_df_tmp.empty else 0

    summary = {
        'pb_pass_rate_mean': (
            float(np.mean([r['pb_pass_rate'] for r in pass_rates]))
            if pass_rates else 0.0
        ),
        'per_pocket': pass_rates,
        'filter_counts': {
            'n_valid':       total_valid,
            'n_pb_pass':     total_pb_pass,
            'n_filtered':    total_filtered,
            'n_near_miss':   n_near_miss,
            'n_strained':    n_strained,
            'n_salvageable': n_salvageable,
        },
    }

    manifest = pd.DataFrame(manifest_rows)
    if filtered_dir is not None and not manifest.empty:
        manifest_path = filtered_dir / 'manifest.csv'
        manifest.to_csv(manifest_path, index=False)
        print(f"Manifest saved → {manifest_path}")

    # Clean up checkpoint — run finished cleanly.
    if ckpt_path and ckpt_path.exists():
        ckpt_path.unlink()

    return summary, manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sdf_dir',      type=Path, required=True,
                   help='Directory of per-pocket SDF files from run_baseline.py')
    p.add_argument('--pdb_dir',      type=Path, default=None,
                   help='Receptor PDB directory — enables PoseBusters + filter pipeline')
    p.add_argument('--smiles_ref',   type=Path,
                   default='DiffSBDD/data/crossdocked_smiles.npy',
                   help='Reference SMILES .npy for novelty (default: crossdocked set)')
    p.add_argument('--out',          type=Path, default=None,
                   help='Output JSON (default: <sdf_dir>/metrics.json)')
    p.add_argument('--filtered_dir', type=Path, default=None,
                   help='Where to write filtered SDFs + manifest.csv '
                        '(default: <sdf_dir>/filtered/ when --pdb_dir is given)')
    p.add_argument('--qed_min',      type=float, default=0.5,
                   help='Minimum QED for druglikeness filter (default: 0.5)')
    p.add_argument('--sa_max',       type=float, default=5.0,
                   help='Maximum SA score for druglikeness filter (default: 5.0)')
    p.add_argument('--label',        type=str,  default='',
                   help='Short label for this condition (e.g. "C-timestep")')
    p.add_argument('--n_jobs',       type=int,  default=1,
                   help='Parallel PoseBusters workers (default: 1). '
                        'Set to number of CPU cores for a large speedup.')
    args = p.parse_args()

    out_path     = args.out or (args.sdf_dir / 'metrics.json')
    filtered_dir = args.filtered_dir
    if filtered_dir is None and args.pdb_dir is not None:
        filtered_dir = args.sdf_dir / 'filtered'

    # ── Cache hit: chemistry metrics already done → skip straight to PoseBusters
    if out_path.exists():
        print(f"Chemistry metrics already computed → {out_path}")
        with open(out_path) as _f:
            metrics = json.load(_f)
        # PoseBusters needs to run if:
        #   (a) the 'posebusters' key is absent, OR
        #   (b) a checkpoint file exists meaning a previous run was interrupted
        ckpt_path = (filtered_dir / 'checkpoint.json') if filtered_dir else None
        pb_incomplete = ckpt_path is not None and ckpt_path.exists()
        if args.pdb_dir is not None and ('posebusters' not in metrics or pb_incomplete):
            if pb_incomplete:
                print(f"  PoseBusters checkpoint found → resuming from {ckpt_path}")
            print(f"Loading SDF files from {args.sdf_dir} ...")
            pockets = load_sdf_dir(args.sdf_dir)
            print(f"  {len(pockets)} pockets, "
                  f"{sum(len(m) for _, m in pockets)} total molecules")
        else:
            pockets = None
    # ── Fresh run: compute chemistry metrics first ────────────────────────────
    else:
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
        metrics.update(compute_chemistry_metrics(pockets, smiles_ref=smiles_ref))

    if args.pdb_dir is not None and pockets is not None:
        print(f"\nRunning PoseBusters + filter "
              f"(QED≥{args.qed_min}, SA≤{args.sa_max}) ...")
        pb_summary, _ = compute_posebusters_and_filter(
            pockets, args.sdf_dir, args.pdb_dir,
            qed_min=args.qed_min, sa_max=args.sa_max,
            filtered_dir=filtered_dir, n_jobs=args.n_jobs,
        )
        if pb_summary:
            metrics['posebusters'] = pb_summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Condition : {args.label or args.sdf_dir.name}")
    print(f"{'='*52}")
    print(f"  Pockets evaluated  : {metrics['n_pockets']}")
    print(f"  Molecules generated: {metrics['n_generated']}")
    print(f"  Validity           : {metrics['validity']:.1%}")
    print(f"  Uniqueness         : {metrics['uniqueness_mean']:.1%}")
    if metrics.get('novelty') is not None:
        print(f"  Novelty            : {metrics['novelty']:.1%}")
    print(f"  QED                : {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}")
    print(f"  SA                 : {metrics['sa_mean']:.3f} ± {metrics['sa_std']:.3f}")
    print(f"  LogP               : {metrics['logp_mean']:.3f} ± {metrics['logp_std']:.3f}")
    print(f"  Lipinski           : {metrics['lipinski_mean']:.3f} ± {metrics['lipinski_std']:.3f}")
    print(f"  Diversity          : {metrics['diversity_mean']:.3f} ± {metrics['diversity_std']:.3f}")
    if 'posebusters' in metrics:
        pb  = metrics['posebusters']
        fc  = pb.get('filter_counts', {})
        nv  = fc.get('n_valid', 0)
        print(f"  PoseBusters pass   : {pb['pb_pass_rate_mean']:.1%}")
        if fc and nv:
            print(f"  After all filters  : "
                  f"{fc['n_filtered']} / {nv} valid  "
                  f"({fc['n_filtered']/nv:.1%})")
            print(f"  Near-misses        : {fc['n_near_miss']}  "
                  f"(PB pass, failed QED/SA — inpaint candidates)")
            print(f"  Strained rings     : {fc['n_strained']}  "
                  f"(3-membered rings present)")
            print(f"  Salvageable        : {fc['n_salvageable']}  "
                  f"(near-miss AND strained ring — highest priority)")
        if filtered_dir:
            print(f"  Filtered SDFs      : {filtered_dir}/")
            print(f"  Manifest           : {filtered_dir}/manifest.csv")
    print(f"\nFull metrics → {out_path}")


if __name__ == '__main__':
    main()
