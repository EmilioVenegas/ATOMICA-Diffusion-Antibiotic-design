"""
analyze_baseline.py
Load the saved LightGBM/HistGB model and visualize diagnostics:
- Pred vs Actual
- Residual histogram
- Residuals vs Target (heteroscedasticity)
- Learning curve (bias/variance view)
- Simple validation curve over n_estimators
- Feature importance bar chart

Run:
    python analyze_baseline.py
"""

import os, sqlite3, joblib, numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import argparse
import matplotlib
matplotlib.use("Agg")   # must be before importing pyplot
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_rows", type=int, default=150_000, help="subset for speed")
    p.add_argument("--rebuild_cache", action="store_true", help="force recompute X,y")
    p.add_argument("--learning_curves", action="store_true", help="run learning curve")
    p.add_argument("--validation_curve", action="store_true", help="run validation curve")
    return p.parse_args()

args = parse_args()
ARTIFACT_DIR = "./artifacts_baseline"
CACHE_X = os.path.join(ARTIFACT_DIR, "X.npy")
CACHE_Y = os.path.join(ARTIFACT_DIR, "y.npy")
CACHE_ROWS = os.path.join(ARTIFACT_DIR, "rows.parquet")



# ---------- CONFIG ----------
CHEMBL_SQLITE = r"C:\Users\khcod\.data\chembl\35\chembl_35.db"
MODEL_PATH    = os.path.join(ARTIFACT_DIR, "model_fold1.joblib")
N_BITS        = 1024
RADIUS        = 2
SEED          = 13
MAX_ROWS      = 150_000   # keep consistent with training
STANDARD_TYPES = ("Kd","Ki","IC50")

# ---------- DATA LOADING (same as training, ligand-only) ----------

def column_exists(con, table, col):
    rows = con.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r[1] == col for r in rows)

def pick_target_id_column(con):
    return "chembl_id" if column_exists(con, "target_dictionary", "chembl_id") else "target_chembl_id"

def load_chembl_df(sqlite_path, limit=None):
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
          AND a.standard_type IN ({",".join(f"'{t}'" for t in STANDARD_TYPES)});
        """
        df = pd.read_sql_query(q, con)
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=SEED).reset_index(drop=True)
    return df

def smiles_to_features(smiles: str, n_bits: int, radius: int):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)  # fast C++ path, no bitstring
    phys = np.array([
        Descriptors.MolWt(m),
        Descriptors.MolLogP(m),
        Descriptors.TPSA(m),
        Descriptors.NumHAcceptors(m),
        Descriptors.NumHDonors(m),
        Descriptors.NumRotatableBonds(m),
        Descriptors.RingCount(m),
        Chem.GetFormalCharge(m),
    ], dtype=float)
    return np.concatenate([arr, phys])

def featurize_df(df):
    out = []
    keep = []
    for s in df["smiles"].astype(str):
        v = smiles_to_features(s, N_BITS, RADIUS)
        if v is None:
            keep.append(False); out.append(None)
        else:
            keep.append(True); out.append(v)
    df = df.loc[keep].reset_index(drop=True)
    X = np.vstack([v for v in out if v is not None])
    df["features"] = list(X)
    return df

def compute_pAff(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df["pchembl_value"], errors="coerce")
    mask = y.isna() & df["standard_value"].notna()
    if mask.any():
        nM = pd.to_numeric(df.loc[mask, "standard_value"], errors="coerce")
        y.loc[mask] = -np.log10(nM * 1e-9)
    return y

# SPEED PATCH

os.makedirs(ARTIFACT_DIR, exist_ok=True)

if (not args.rebuild_cache) and os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    print(">> Loading cached features…")
    X = np.load(CACHE_X, mmap_mode="r")
    y = np.load(CACHE_Y, mmap_mode="r")
    rows = pd.read_parquet(CACHE_ROWS)
else:
    print(">> Building features (first run may take a while)…")
    raw = load_chembl_df(CHEMBL_SQLITE, limit=args.max_rows).dropna(subset=["smiles"]).reset_index(drop=True)
    raw["pAff"] = compute_pAff(raw)
    raw = raw.dropna(subset=["pAff"]).reset_index(drop=True)
    rows = featurize_df(raw)                         # adds rows['features']

    X = np.vstack(rows["features"].values)
    y = rows["pAff"].values

    print(">> Caching features to disk")
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    rows[["activity_id","smiles","pAff"]].to_parquet(CACHE_ROWS, index=False)


# ---------- LOAD MODEL ----------
model = joblib.load(MODEL_PATH)
print(f"Loaded model from: {MODEL_PATH}")

# ---------- REBUILD DATA (X, y) ----------
raw = load_chembl_df(CHEMBL_SQLITE, limit=MAX_ROWS).dropna(subset=["smiles"]).reset_index(drop=True)
raw["pAff"] = compute_pAff(raw)
raw = raw.dropna(subset=["pAff"]).reset_index(drop=True)

df = featurize_df(raw)
X = np.vstack(df["features"].values)
y = df["pAff"].values

# ---------- PREDICTIONS ----------
y_pred = model.predict(X)

# ---------- PLOTS ----------
ensure = os.makedirs
ensure(os.path.join(ARTIFACT_DIR, "figs"), exist_ok=True)

# 1) Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, s=6, alpha=0.5)
lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual pAff")
plt.ylabel("Predicted pAff")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "figs", "pred_vs_actual.png"), dpi=200)
plt.show()

# 2) Residuals histogram
res = y_pred - y
plt.figure(figsize=(6,4))
plt.hist(res, bins=60)
plt.axvline(0, linestyle="--")
plt.xlabel("Residual (Pred - Actual)")
plt.ylabel("Count")
plt.title("Residuals Histogram")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "figs", "residuals_hist.png"), dpi=200)
plt.show()

# 3) Residuals vs Actual (heteroscedasticity)
plt.figure(figsize=(6,4))
plt.scatter(y, res, s=6, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Actual pAff")
plt.ylabel("Residual (Pred - Actual)")
plt.title("Residuals vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "figs", "residuals_vs_target.png"), dpi=200)
plt.show()

# 4) Feature importance (top 25)
def plot_feature_importance(model, top_k=25):
    # Works for LightGBM and sklearn HistGB
    if hasattr(model, "feature_importances_"):
        fi = np.array(model.feature_importances_, dtype=float)
        idx = np.argsort(fi)[::-1][:top_k]
        vals = fi[idx]
        # name features: fp_0..fp_{N_BITS-1}, then physchem_*
        names = [f"fp_{i}" for i in range(N_BITS)] + ["MW","logP","TPSA","HBA","HBD","RotB","Rings","FormalCharge"]
        labels = [names[i] for i in idx]
        plt.figure(figsize=(7,6))
        plt.barh(range(len(vals))[::-1], vals[np.argsort(vals)], align="center")
        plt.yticks(range(len(labels))[::-1], [labels[j] for j in np.argsort(vals)])
        plt.xlabel("Feature importance")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, "figs", "feature_importance_top25.png"), dpi=200)
        plt.show()

plot_feature_importance(model, top_k=25)

# 5) Calibration / binning plot (mean predicted vs mean actual per bin)
def calibration_plot(y_true, y_hat, n_bins=20):
    order = np.argsort(y_hat)
    yb = y_hat[order]
    tb = y_true[order]
    bins = np.array_split(np.arange(len(yb)), n_bins)
    mean_pred = np.array([yb[idx].mean() for idx in bins])
    mean_true = np.array([tb[idx].mean() for idx in bins])

    plt.figure(figsize=(6,6))
    plt.plot(mean_true, mean_pred, marker="o")
    lims = [min(mean_true.min(), mean_pred.min()), max(mean_true.max(), mean_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("Bin mean Actual pAff")
    plt.ylabel("Bin mean Predicted pAff")
    plt.title("Calibration (binned means)")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "figs", "calibration_binned.png"), dpi=200)
    plt.show()

calibration_plot(y, y_pred, n_bins=25)

# 6) Learning curve (bias-variance view): grow sample size and track RMSE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def learning_curve_rmse(X, y, steps=8, test_size=0.2, seed=SEED):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    sizes = np.linspace(0.1, 1.0, steps)
    rmse_tr, rmse_te = [], []
    for s in sizes:
        n = int(len(X_tr) * s)
        # re-init a small model for speed
        try:
            import lightgbm as lgb
            mdl = lgb.LGBMRegressor(
                n_estimators=600, learning_rate=0.06, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor
            mdl = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=600, random_state=SEED)

        mdl.fit(X_tr[:n], y_tr[:n])
        yhat_tr = mdl.predict(X_tr[:n])
        yhat_te = mdl.predict(X_te)
        rmse_tr.append(mean_squared_error(y_tr[:n], yhat_tr, squared=False))
        rmse_te.append(mean_squared_error(y_te, yhat_te, squared=False))
    return sizes, np.array(rmse_tr), np.array(rmse_te)

sizes, rmse_tr, rmse_te = learning_curve_rmse(X, y, steps=8)
plt.figure(figsize=(6,4))
plt.plot(sizes, rmse_tr, marker="o", label="Train RMSE")
plt.plot(sizes, rmse_te, marker="o", label="Validation RMSE")
plt.xlabel("Relative training set size")
plt.ylabel("RMSE")
plt.title("Learning Curve (bias/variance view)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "figs", "learning_curve.png"), dpi=200)
plt.show()

# 7) Simple validation curve on n_estimators (capacity sweep)
def validation_curve_n_estimators(X, y, n_list=(100, 300, 600, 900, 1200)):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rmses = []
    try:
        import lightgbm as lgb
        for n in n_list:
            mdl = lgb.LGBMRegressor(
                n_estimators=n, learning_rate=0.06, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED
            )
            mdl.fit(X_tr, y_tr)
            yhat = mdl.predict(X_te)
            rmses.append(mean_squared_error(y_te, yhat, squared=False))
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        for n in n_list:
            mdl = HistGradientBoostingRegressor(
                max_iter=n, max_depth=6, learning_rate=0.06, random_state=SEED
            )
            mdl.fit(X_tr, y_tr)
            yhat = mdl.predict(X_te)
            rmses.append(mean_squared_error(y_te, yhat, squared=False))
    return np.array(n_list), np.array(rmses)

n_list, val_rmse = validation_curve_n_estimators(X, y)
plt.figure(figsize=(6,4))
plt.plot(n_list, val_rmse, marker="o")
plt.xlabel("n_estimators / max_iter")
plt.ylabel("Validation RMSE")
plt.title("Validation Curve (model capacity)")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "figs", "validation_curve_n_estimators.png"), dpi=200)
plt.show()

print("\nSaved figures under:", os.path.join(ARTIFACT_DIR, "figs"))
