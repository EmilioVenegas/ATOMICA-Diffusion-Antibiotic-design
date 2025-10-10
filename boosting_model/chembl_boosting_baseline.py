"""
chembl_boosting_baseline.py
Minimal ligand-only boosting baseline on ChEMBL for pAff (≈ pKi/pKd/pIC50 in molar).
- Fast runtime
- Clear scaffolding
- Few dependencies (RDKit + scikit-learn; LightGBM optional)

Usage:
    python chembl_boosting_baseline.py

Prereqs:
    pip install rdkit-pypi scikit-learn pandas numpy
    # optional (faster/better trees):
    pip install lightgbm
"""

from __future__ import annotations
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

# ---- Config ------------------------------------------------------------------

@dataclass
class Config:
    chembl_sqlite_path: str  # absolute path to chembl_XX.sqlite
    n_bits: int = 1024        # Morgan FP size (smaller for speed)
    radius: int = 2
    seed: int = 13
    kfolds: int = 5
    use_lightgbm_if_available: bool = True
    save_dir: str = "./artifacts_baseline"
    max_rows: Optional[int] = 150_000      # cap for speed; set None for all
    standard_types: Tuple[str, ...] = ("Kd", "Ki", "IC50")  # normalized to nM in query


# ---- Utilities ---------------------------------------------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def column_exists(con: sqlite3.Connection, table: str, col: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r[1] == col for r in rows)


def pick_target_id_column(con: sqlite3.Connection) -> str:
    return "chembl_id" if column_exists(con, "target_dictionary", "chembl_id") else "target_chembl_id"


# ---- Data Loading ------------------------------------------------------------

def load_chembl_activities(sqlite_path: str,
                           standard_types: Iterable[str],
                           limit: Optional[int] = None) -> pd.DataFrame:
    """
    Pull standardized binding activities with SMILES and target ID.
    """
    with sqlite3.connect(sqlite_path) as con:
        tgt_col = pick_target_id_column(con)
        q = f"""
        SELECT a.activity_id, a.assay_id, a.standard_type, a.standard_relation,
               a.standard_value, a.standard_units, a.pchembl_value,
               cs.canonical_smiles AS smiles, a.molregno,
               s.tid, td.{tgt_col} AS target_id, td.pref_name AS target_name
        FROM activities a
        JOIN assays s               ON a.assay_id = s.assay_id
        JOIN compound_structures cs ON a.molregno = cs.molregno
        JOIN target_dictionary td   ON s.tid = td.tid
        WHERE a.data_validity_comment IS NULL
          AND a.standard_units = 'nM'
          AND a.standard_relation = '='
          AND a.standard_type IN ({",".join(f"'{t}'" for t in standard_types)});
        """
        df = pd.read_sql_query(q, con)
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=13).reset_index(drop=True)
    return df


# ---- Featurization (Ligand-only) --------------------------------------------

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def smiles_to_features(smiles: str, n_bits: int, radius: int) -> Optional[np.ndarray]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
    # small physchem set (kept short for speed)
    phys = np.array([
        Descriptors.MolWt(m),
        Descriptors.MolLogP(m),
        Descriptors.TPSA(m),
        Descriptors.NumHAcceptors(m),
        Descriptors.NumHDonors(m),
        Descriptors.NumRotatableBonds(m),
        Descriptors.RingCount(m),
        Chem.GetFormalCharge(m)
    ], dtype=float)
    return np.concatenate([np.frombuffer(fp.ToBitString().encode(), dtype='S1').astype(int), phys])


def featurize_df(df: pd.DataFrame, n_bits: int, radius: int) -> pd.DataFrame:
    feats = []
    keep_rows = []
    for i, s in enumerate(df["smiles"].astype(str)):
        v = smiles_to_features(s, n_bits=n_bits, radius=radius)
        if v is None:
            keep_rows.append(False)
            feats.append(None)
        else:
            keep_rows.append(True)
            feats.append(v)
    kept = df.loc[keep_rows].reset_index(drop=True)
    X = np.vstack([v for v in feats if v is not None])
    kept["features"] = list(X)
    return kept


# ---- Targets / Labels --------------------------------------------------------

def compute_pAff(df: pd.DataFrame) -> pd.Series:
    """
    pAff priority:
    - use pchembl_value when present;
    - fallback to -log10(standard_value[nM] * 1e-9)
    """
    y = pd.to_numeric(df["pchembl_value"], errors="coerce")
    fallback_mask = y.isna() & df["standard_value"].notna()
    if fallback_mask.any():
        nM = pd.to_numeric(df.loc[fallback_mask, "standard_value"], errors="coerce")
        y.loc[fallback_mask] = -np.log10(nM * 1e-9)
    return y


# ---- Splits (Scaffold-aware GroupKFold) -------------------------------------

from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

def bemis_murcko_scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        return ""


def make_splits(df: pd.DataFrame, k: int, seed: int):
    """
    Group by both scaffold and target to reduce leakage across protein families and chemotypes.
    """
    df = df.copy()
    df["scaffold"] = df["smiles"].astype(str).apply(bemis_murcko_scaffold)
    # Combine groups: (target_id, scaffold)
    groups = df["target_id"].astype(str) + "|" + df["scaffold"].astype(str)
    gkf = GroupKFold(n_splits=k)
    idx = np.arange(len(df))
    for tr_idx, va_idx in gkf.split(idx, groups=groups):
        yield tr_idx, va_idx


# ---- Models ------------------------------------------------------------------

def get_model(light=True, seed=13):
    """
    Return a light-weight regressor.
    - If LightGBM is available and `light=True`, use it with small trees.
    - Else fall back to scikit-learn HistGradientBoostingRegressor.
    """
    if light:
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1
            )
        except Exception:
            pass
    # Fallback: very fast, built-in
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.06, max_iter=600,
        l2_regularization=1.0, random_state=seed
    )


# ---- Metrics / Logging -------------------------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    pr   = pearsonr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    sr   = spearmanr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    return {"RMSE": rmse, "MAE": mae, "Pearson_r": pr, "Spearman_r": sr}


# ---- Main --------------------------------------------------------------------

def main(cfg: Config):
    ensure_dir(cfg.save_dir)
    print(">> Loading ChEMBL from:", cfg.chembl_sqlite_path)
    df = load_chembl_activities(cfg.chembl_sqlite_path, cfg.standard_types, limit=cfg.max_rows)

    # Clean and target
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    df["pAff"] = compute_pAff(df)
    df = df.dropna(subset=["pAff"]).reset_index(drop=True)

    print(f">> Rows after filtering: {len(df):,}")

    # Featurize (ligand-only)
    print(">> Featurizing ligands (Morgan FP + small physchem)…")
    df = featurize_df(df, n_bits=cfg.n_bits, radius=cfg.radius)
    X = np.vstack(df["features"].values)
    y = df["pAff"].values

    # Cross-validated training
    print(">> Training folds…")
    all_metrics = []
    fold_models = []
    for fold, (tr_idx, va_idx) in enumerate(make_splits(df, cfg.kfolds, cfg.seed), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        model = get_model(light=cfg.use_lightgbm_if_available, seed=cfg.seed)
        model.fit(X_tr, y_tr)

        y_hat = model.predict(X_va)
        metrics = eval_metrics(y_va, y_hat)
        all_metrics.append(metrics)
        fold_models.append(model)

        print(f"Fold {fold}: " +
              ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # Aggregate metrics
    agg = {k: np.nanmean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print("\n== CV Mean Metrics ==")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}")

    # Save minimal artifacts
    np.save(os.path.join(cfg.save_dir, "X_shape.npy"), np.array(X.shape))
    pd.DataFrame(all_metrics).to_csv(os.path.join(cfg.save_dir, "cv_metrics.csv"), index=False)
    df[["activity_id", "smiles", "pAff"]].head(1000).to_csv(
        os.path.join(cfg.save_dir, "sample_rows.csv"), index=False
    )
    try:
        import joblib
        joblib.dump(fold_models[0], os.path.join(cfg.save_dir, "model_fold1.joblib"))
    except Exception:
        pass
    print(f"\nArtifacts saved to: {cfg.save_dir}")


if __name__ == "__main__":
    # EDIT THIS PATH: point to your local ChEMBL SQLite file
    cfg = Config(
        chembl_sqlite_path=r"C:\Users\khcod\.data\chembl\35\chembl_35.db",
        n_bits=1024,
        radius=2,
        seed=13,
        kfolds=5,
        use_lightgbm_if_available=True,
        save_dir="./artifacts_baseline",
        max_rows=150_000,   # lower for faster runs; None for full set
    )
    main(cfg)
