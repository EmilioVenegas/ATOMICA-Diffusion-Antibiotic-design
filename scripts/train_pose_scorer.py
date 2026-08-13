"""Train and evaluate a pose scorer on ATOMICA interface representations.

The Phase 0 gate showed the representation encodes interaction geometry, but it
was read by a probe fitted on the very system it scored. That is not a scorer. A
usable one must rank poses in a pocket it has never seen, so every number here
comes from target-wise splits: the targets a fold is tested on contribute nothing
to its training.

The metric is CASF's **docking power** -- for each target, is the top-ranked pose
within 2 A of the crystal pose. Targets where docking never produced a sub-2 A
pose are excluded, since no scorer can succeed on them. The baseline to beat is
smina's own score, already stored in the benchmark manifest.

Representations are cached, because featurization dominates the runtime and the
head is cheap to retrain.

Usage (from repo root):

    python scripts/train_pose_scorer.py --benchmark data/pose_benchmark
"""

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

NEAR_NATIVE = 2.0  # A, the conventional threshold for a correct pose


def featurize(manifest_rows, site_radius, dist_th, device, cache_path):
    """Encode every (fixed site, pose) pair as a two-segment ATOMICA record.

    The binding site is derived once per target from its crystal ligand and held
    fixed across that target's poses. Re-deriving it per pose would let the
    representation separate poses by how many residues they happen to contact --
    a shortcut worth AUROC ~0.70 on the Phase 0 benchmark.
    """
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        print(f"      loaded cached features {cached['X'].shape} from {cache_path}")
        return cached["X"], cached["y"], cached["groups"], cached["smina"]

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")

    from ATOMICA.data.dataset import blocks_interface
    from atomica_interface.featurize import (
        component_smiles,
        interface_data,
        ligand_blocks_from_mol,
        pocket_blocks_from_pdb,
        to_batch,
    )
    from atomica_interface.scoring import interface_representation, load_encoder

    model = load_encoder(
        "ATOMICA/pretrain/pretrain_model_config.json",
        "ATOMICA/pretrain/pretrain_model_weights.pt",
        device,
    )

    by_target = collections.defaultdict(list)
    for row in manifest_rows:
        by_target[row["target"]].append(row)

    X, y, groups, smina = [], [], [], []
    for i, (target, rows) in enumerate(sorted(by_target.items()), 1):
        first = rows[0]
        template = Chem.MolFromSmiles(component_smiles(first["ligand"]) or "")
        native = Chem.SDMolSupplier(first["native_sdf"])[0]
        if template is None or native is None:
            continue

        # Guarded for the same reason as in featurize_block_level.py: the
        # tokenizer raises on ligands whose elements are outside its valence
        # table, and that must skip a target rather than kill the run.
        try:
            pocket = pocket_blocks_from_pdb(first["receptor"])
            site, _ = blocks_interface(pocket, ligand_blocks_from_mol(native), site_radius)
        except Exception as exc:
            print(f"      [{i}/{len(by_target)}] {target}: SKIPPED ({type(exc).__name__})")
            continue

        poses = list(Chem.SDMolSupplier(first["poses_file"], sanitize=False))
        kept = 0
        for row in rows:
            pose = poses[int(row["pose_index"])]
            if pose is None:
                continue
            try:
                fixed = AllChem.AssignBondOrdersFromTemplate(
                    template, Chem.RemoveAllHs(pose, sanitize=False)
                )
                record = interface_data(
                    site, ligand_blocks_from_mol(fixed), dist_th, trim=False
                )
                vec = interface_representation(model, to_batch(record, device))
            except Exception:
                continue
            X.append(vec.reshape(-1))
            y.append(float(row["rmsd"]))
            groups.append(target)
            smina.append(float(row["smina_score"]) if row["smina_score"] else np.nan)
            kept += 1
        print(f"      [{i}/{len(by_target)}] {target}: {kept} poses, "
              f"site {len(site)} blocks")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    groups = np.asarray(groups)
    smina = np.asarray(smina, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, groups=groups, smina=smina)
    return X, y, groups, smina


def out_of_fold_predictions(X, y, groups, alpha, seed):
    """Predict RMSD for every pose from a model that never saw its target."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n_targets = len(set(groups))
    n_splits = min(5, n_targets)
    preds = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        head = Ridge(alpha=alpha, random_state=seed).fit(
            scaler.transform(X[train_idx]), y[train_idx]
        )
        preds[test_idx] = head.predict(scaler.transform(X[test_idx]))
    return preds


def docking_power(y_true, score, groups, solvable):
    """Fraction of solvable targets whose top-ranked pose is near-native.

    ``score`` is ranked ascending, so pass predicted RMSD or smina affinity
    directly (both are lower-is-better).
    """
    hits, considered, per_target = 0, 0, {}
    for target in sorted(set(groups)):
        if target not in solvable:
            continue
        mask = groups == target
        best = np.argmin(score[mask])
        top_rmsd = float(y_true[mask][best])
        per_target[target] = round(top_rmsd, 2)
        hits += top_rmsd <= NEAR_NATIVE
        considered += 1
    return (hits / considered if considered else float("nan")), considered, per_target


def spearman(a, b):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="data/pose_benchmark")
    p.add_argument("--site_radius", type=float, default=10.0)
    p.add_argument("--dist_th", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=10.0, help="ridge regularisation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/pose_scorer")
    args = p.parse_args()

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    bench = Path(args.benchmark)
    rows = list(csv.DictReader(open(bench / "manifest.csv")))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] featurizing {len(rows)} poses on {device}")
    X, y, groups, smina = featurize(
        rows, args.site_radius, args.dist_th, device, outdir / "features.npz"
    )
    print(f"      {X.shape[0]} poses x {X.shape[1]}-d over {len(set(groups))} targets")

    # A target with no sub-2 A pose cannot be solved by any scorer; CASF excludes
    # these rather than counting them as universal failures.
    solvable = {t for t in set(groups) if y[groups == t].min() <= NEAR_NATIVE}
    print(f"      solvable targets: {len(solvable)}/{len(set(groups))}")

    print("[2/3] training head with target-wise splits")
    preds = out_of_fold_predictions(X, y, groups, args.alpha, args.seed)

    print("[3/3] evaluating")
    atomica_dp, n_considered, atomica_per = docking_power(y, preds, groups, solvable)
    smina_dp, _, smina_per = docking_power(y, smina, groups, solvable)
    # Ranking poses at random gives the rate at which a target's poses are
    # near-native -- the floor any scorer must clear.
    random_floor = float(
        np.mean([(y[groups == t] <= NEAR_NATIVE).mean() for t in sorted(solvable)])
    )

    per_target_rho = [
        spearman(preds[groups == t], y[groups == t]) for t in sorted(solvable)
    ]
    mean_rho = float(np.nanmean(per_target_rho))

    report = {
        "n_poses": int(len(y)),
        "n_targets": int(len(set(groups))),
        "n_solvable": len(solvable),
        "feature_dim": int(X.shape[1]),
        "near_native_threshold_A": NEAR_NATIVE,
        "docking_power": {
            "atomica_head": round(atomica_dp, 4),
            "smina_baseline": round(smina_dp, 4),
            "random_floor": round(random_floor, 4),
            "targets_considered": n_considered,
        },
        "mean_per_target_spearman": round(mean_rho, 4),
        "top1_rmsd_by_target": {"atomica": atomica_per, "smina": smina_per},
    }
    (outdir / "pose_scorer_report.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"{'scorer':<22}{'docking power':>15}")
    print(f"{'random (floor)':<22}{random_floor:>14.1%}")
    print(f"{'smina (baseline)':<22}{smina_dp:>14.1%}")
    print(f"{'ATOMICA head':<22}{atomica_dp:>14.1%}")
    print(f"\nmean per-target Spearman(predicted, true RMSD): {mean_rho:+.3f}")
    print(f"over {n_considered} solvable targets, out-of-fold by target")
    print(f"\nwrote {outdir/'pose_scorer_report.json'}")


if __name__ == "__main__":
    main()
