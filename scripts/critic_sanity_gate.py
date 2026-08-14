"""Cheap gate on the critic loss, before spending GPU-days on it.

The proposed objective is

    L = L_diffusion + lambda * d( ATOMICA(pocket, x0_hat), ATOMICA(pocket, x_true) )

which is only worth training if `d` actually grows as `x0_hat` moves away from
`x_true`. This measures exactly that on `data/pose_benchmark`, where every pose
of a target is the *same molecule* rigidly displaced by a known RMSD -- so
composition is controlled by construction and the only thing varying is
interaction geometry. That is the within-system Phase 0 regime the whole plan
rests on, and it is the regime the critic operates in.

Two things are checked, and the second matters more:

1. **Separation.** Does `d` rank near-native poses below decoys? Reported as a
   per-target AUROC over poses, never pooled across targets -- pooling would
   mix systems and measure the cross-system transfer that Phase 2 already
   resolved as negative.
2. **Gradient near the target.** A loss that only separates 0.5 A from 30 A is
   useless for refinement; the denoiser's `x0_hat` at low `t` is already close.
   So the RMSD < 4 A subset is analysed separately, and `d` is profiled across
   RMSD bins. If `d` is flat below a few angstrom, the critic has nothing to
   pull on where it is applied.

## The controls, and why the obvious one is wrong

An earlier version of this script used `pocket_pool` -- the pooled block
representations of the pocket segment -- as its negative control, on the
assumption (stated in `scripts/featurize_block_level.py`) that it is "identical
for every pose of a target". It is not. The *input* pocket blocks are identical,
but their representations are computed with message passing from the ligand, so
they move with the pose like everything else. It scored 0.949 AUROC against the
real metric's 0.926: not a floor, just another readout of the same encoding.

That leaves the question the gate actually has to answer. Any encoder that is
sensitive to geometry at all will report a large distance when a ligand is
dragged out of its pocket, so separation alone proves nothing about ATOMICA.
Two real floors are measured instead:

- **`contacts`** -- no learning whatsoever. The change in the number of
  protein-ligand atom contacts between the pose and the native. This is the
  buriedness baseline that `docs/experiment-plan.md` requires of any pocket
  scoring result, and it is the same class of trivial measure that reached the
  98.2nd percentile in the hotspot phase while the ATOMICA field sat at random.
- **`random_encoder`** -- the same ATOMICA architecture with every weight tensor
  randomly permuted (seeded). Permuting rather than re-initialising preserves
  each layer's exact weight distribution and scale while destroying all learned
  structure, so any gap against the pretrained model is attributable to
  pretraining rather than to architecture or scale.

If the pretrained encoder does not beat both, the critic is a geometric loss
wearing a foundation model, and a cheaper differentiable geometric term would do
the same job.

Usage (from repo root):

    python scripts/critic_sanity_gate.py --encoder cache      # pretrained, fast
    python scripts/critic_sanity_gate.py --encoder random     # the control
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

NEAR_NATIVE_A = 2.0
DECOY_A = 4.0
CONTACT_A = 4.5


def pool(matrix):
    """mean/max/std over rows -- must match scripts/featurize_block_level.py."""
    if matrix.shape[0] == 0:
        return np.zeros(3 * matrix.shape[1], dtype=np.float32)
    return np.concatenate([matrix.mean(0), matrix.max(0), matrix.std(0)]).astype(np.float32)


def spearman(a, b):
    """Rank correlation, ties averaged. nan when either side is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return np.nan

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        sv = v[order]
        start = 0
        for i in range(1, len(sv) + 1):
            if i == len(sv) or sv[i] != sv[start]:
                if i - start > 1:
                    r[order[start:i]] = r[order[start:i]].mean()
                start = i
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(labels, scores):
    """P(a random positive outranks a random negative), ties at 0.5."""
    from atomica_interface.scoring import roc_auc

    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return roc_auc(labels, scores)


def permute_weights(model, seed):
    """Randomly permute the entries of every weight tensor, in place.

    A scale-preserving destruction of learned structure: each tensor keeps its
    exact multiset of values, hence its norm, mean and variance, so the control
    differs from the pretrained model in *what the weights encode* and in
    nothing else. Re-initialising instead would confound the comparison with a
    change of scale.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            flat = param.detach().reshape(-1).cpu()
            if flat.numel() < 2:
                continue
            shuffled = flat[torch.randperm(flat.numel(), generator=generator)]
            param.copy_(shuffled.reshape(param.shape).to(param.device))
    return model


def load_target_rows(bench):
    rows = list(csv.DictReader(open(bench / "manifest.csv")))
    by_target = collections.defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)
    return by_target


def encode_benchmark(by_target, args, randomise_seed=None):
    """Encode every native and pose with one encoder.

    Returns ``(refs, poses)``: ``refs[target][block]`` is the native reference
    vector and ``poses[target]`` is a dict of stacked per-pose arrays plus the
    matching RMSD and smina columns. Mirrors
    `scripts/featurize_block_level.py` exactly -- same site radius, same
    dist_th, same trim=False, same pooling -- so cached pretrained vectors and
    freshly-encoded control vectors are directly comparable.
    """
    import torch
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
    from atomica_interface.scoring import load_encoder

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_encoder(
        "ATOMICA/pretrain/pretrain_model_config.json",
        "ATOMICA/pretrain/pretrain_model_weights.pt",
        device,
    )
    if randomise_seed is not None:
        print(f"CONTROL: permuting all weight tensors (seed {randomise_seed}).")
        model = permute_weights(model, randomise_seed)
    print(f"Encoding {len(by_target)} targets on {device}...")

    def encode(site, mol):
        record = interface_data(site, ligand_blocks_from_mol(mol), args.dist_th, trim=False)
        batch = to_batch(record, device)
        with torch.no_grad():
            out = model.infer(batch)
        block_repr = out.block_repr.detach().cpu().numpy()
        graph_repr = out.graph_repr.detach().cpu().numpy().reshape(-1)
        segments = batch["segment_ids"].detach().cpu().numpy()
        if block_repr.shape[0] != segments.shape[0]:
            raise ValueError("block/segment length mismatch")
        return {
            "graph": graph_repr.astype(np.float32),
            "ligand_pool": pool(block_repr[segments == 1]),
            "pocket_pool": pool(block_repr[segments == 0]),
        }

    refs, poses = {}, {}
    for i, (target, rows) in enumerate(sorted(by_target.items()), 1):
        first = rows[0]
        template = Chem.MolFromSmiles(component_smiles(first["ligand"]) or "")
        native = Chem.SDMolSupplier(first["native_sdf"])[0]
        if template is None or native is None:
            continue
        try:
            # PS_300 has no valence entry for iron and raises on haem ligands;
            # one such target must not abort the run.
            pocket = pocket_blocks_from_pdb(first["receptor"])
            site, _ = blocks_interface(pocket, ligand_blocks_from_mol(native), args.site_radius)
            native_fixed = AllChem.AssignBondOrdersFromTemplate(
                template, Chem.RemoveAllHs(native, sanitize=False)
            )
            refs[target] = encode(site, native_fixed)
        except Exception as exc:
            print(f"  [{i}/{len(by_target)}] {target}: SKIPPED "
                  f"({type(exc).__name__}: {exc})")
            continue

        pose_mols = list(Chem.SDMolSupplier(first["poses_file"], sanitize=False))
        collected = collections.defaultdict(list)
        rmsd, smina = [], []
        for row in rows:
            pose = pose_mols[int(row["pose_index"])]
            if pose is None:
                continue
            try:
                fixed = AllChem.AssignBondOrdersFromTemplate(
                    template, Chem.RemoveAllHs(pose, sanitize=False)
                )
                vectors = encode(site, fixed)
            except Exception:
                continue
            for k, v in vectors.items():
                collected[k].append(v)
            rmsd.append(float(row["rmsd"]))
            smina.append(float(row["smina_score"]) if row["smina_score"] else np.nan)

        if not rmsd:
            refs.pop(target, None)
            continue
        poses[target] = {k: np.asarray(v, dtype=np.float32) for k, v in collected.items()}
        poses[target]["rmsd"] = np.asarray(rmsd, dtype=float)
        poses[target]["smina"] = np.asarray(smina, dtype=float)
        print(f"  [{i}/{len(by_target)}] {target}: {len(rmsd)} poses")

    return refs, poses


def contact_counts(by_target, args):
    """Protein-ligand atom contacts per pose, and for the native. No encoder.

    The buriedness-class baseline `docs/experiment-plan.md` requires: it uses
    coordinates only, so whatever it scores is what this protocol reports for a
    measure containing no interaction chemistry at all.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from atomica_interface.featurize import pocket_blocks_from_pdb

    out = {}
    for target, rows in sorted(by_target.items()):
        first = rows[0]
        native = Chem.SDMolSupplier(first["native_sdf"])[0]
        if native is None:
            continue
        try:
            blocks = pocket_blocks_from_pdb(first["receptor"])
        except Exception:
            continue
        pocket_xyz = np.array(
            [a.get_coord() for b in blocks for a in b.units], dtype=float
        )
        if pocket_xyz.size == 0:
            continue

        def n_contacts(mol):
            xyz = mol.GetConformer().GetPositions()
            d = np.linalg.norm(pocket_xyz[:, None, :] - xyz[None, :, :], axis=2)
            return float((d < CONTACT_A).sum())

        native_n = n_contacts(Chem.RemoveAllHs(native, sanitize=False))
        pose_mols = list(Chem.SDMolSupplier(first["poses_file"], sanitize=False))
        counts, rmsd = [], []
        for row in rows:
            pose = pose_mols[int(row["pose_index"])]
            if pose is None:
                continue
            try:
                counts.append(n_contacts(Chem.RemoveAllHs(pose, sanitize=False)))
            except Exception:
                continue
            rmsd.append(float(row["rmsd"]))
        if counts:
            out[target] = {
                "distance": np.abs(np.asarray(counts) - native_n),
                "rmsd": np.asarray(rmsd, dtype=float),
            }
    return out


def cosine_distance(ref, mat):
    rn = ref / (np.linalg.norm(ref) + 1e-12)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return 1.0 - mn @ rn


def l2_distance(ref, mat):
    return np.linalg.norm(mat - ref[None, :], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--benchmark", default="data/pose_benchmark")
    p.add_argument("--encoder", choices=("cache", "pretrained", "random"), default="cache",
                   help="'cache' reads the pretrained pose vectors from "
                        "--features and encodes only the natives; 'random' is "
                        "the weight-permuted control and must encode everything")
    p.add_argument("--features", default="results/pose_scorer/features_block.npz")
    p.add_argument("--seed", type=int, default=0, help="weight-permutation seed")
    p.add_argument("--out", default=None)
    p.add_argument("--site_radius", type=float, default=10.0)
    p.add_argument("--dist_th", type=float, default=8.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min_poses", type=int, default=5,
                   help="targets with fewer scored poses are dropped from the "
                        "per-target statistics")
    p.add_argument("--skip_contacts", action="store_true",
                   help="skip the no-learning contact baseline")
    args = p.parse_args()
    out_path = Path(args.out or f"results/critic_gate/gate_{args.encoder}.json")

    bench = Path(args.benchmark)
    by_target = load_target_rows(bench)

    if args.encoder == "cache":
        refs, poses = encode_natives_and_cache(by_target, args)
    else:
        seed = args.seed if args.encoder == "random" else None
        refs, poses = encode_benchmark(by_target, args, randomise_seed=seed)

    metrics = {
        "graph_cosine": ("graph", cosine_distance),
        "graph_l2": ("graph", l2_distance),
        "ligand_pool_l2": ("ligand_pool", l2_distance),
        "pocket_pool_l2": ("pocket_pool", l2_distance),
    }

    contacts = {} if args.skip_contacts else contact_counts(by_target, args)

    names = list(metrics) + ["smina (reference)"]
    if contacts:
        names.append("contacts (no-learning floor)")
    per_target = {name: [] for name in names}
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 100)]
    binned = {name: collections.defaultdict(list) for name in names}

    def record(name, target, d, y):
        labels_near = (y < NEAR_NATIVE_A).astype(int)
        # Poses between the two thresholds are neither clearly right nor clearly
        # wrong; including them would blur the separation being measured.
        decisive = (y < NEAR_NATIVE_A) | (y > DECOY_A)
        per_target[name].append({
            "target": target,
            "n_poses": int(len(y)),
            "spearman_all": spearman(d, y),
            "spearman_low": spearman(d[y < DECOY_A], y[y < DECOY_A]),
            # Lower distance should mean lower RMSD, so the positive class
            # (near-native) is ranked by -d.
            "auroc": auroc(labels_near[decisive], -d[decisive]),
        })
        dn = (d - d.min()) / (d.max() - d.min() + 1e-12)
        for lo, hi in bins:
            sel = (y >= lo) & (y < hi)
            if sel.any():
                binned[name][(lo, hi)].append(float(dn[sel].mean()))

    usable = []
    for target in sorted(poses):
        if target not in refs:
            continue
        y = poses[target]["rmsd"]
        if len(y) < args.min_poses:
            continue
        usable.append(target)
        for name, (block, dist_fn) in metrics.items():
            record(name, target, dist_fn(refs[target][block], poses[target][block].astype(float)), y)

        s = poses[target]["smina"]
        ok = ~np.isnan(s)
        if ok.sum() >= args.min_poses:
            record("smina (reference)", target, s[ok], y[ok])

        if contacts and target in contacts:
            c = contacts[target]
            if len(c["rmsd"]) == len(y):
                record("contacts (no-learning floor)", target, c["distance"], c["rmsd"])

    def summarize(entries, key):
        vals = np.array([e[key] for e in entries], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return {"mean": float("nan"), "sem": float("nan"), "n": 0,
                    "frac_positive": float("nan")}
        threshold = 0.5 if key == "auroc" else 0.0
        return {
            "mean": float(vals.mean()),
            "sem": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
            "frac_positive": float((vals > threshold).mean()),
        }

    total_poses = int(sum(len(poses[t]["rmsd"]) for t in usable))
    print("\n" + "=" * 86)
    print(f"CRITIC SANITY GATE [{args.encoder}] -- {len(usable)} targets, {total_poses} poses")
    print("=" * 86)
    print("Per-target statistics. Positive Spearman = distance grows with RMSD (wanted).")
    print(f"AUROC separates RMSD < {NEAR_NATIVE_A} A from > {DECOY_A} A.\n")
    header = f"{'metric':<32} {'rho(all)':>16} {'rho(<4A)':>16} {'AUROC':>16}"
    print(header)
    print("-" * len(header))

    report = {}
    for name in names:
        entries = per_target[name]
        if not entries:
            continue
        stats = {k: summarize(entries, k) for k in ("spearman_all", "spearman_low", "auroc")}
        report[name] = {"per_target_summary": stats, "n_targets": len(entries)}
        cells = [f"{stats[k]['mean']:+.3f}+-{stats[k]['sem']:.3f}"
                 for k in ("spearman_all", "spearman_low", "auroc")]
        print(f"{name:<32} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")

    print(f"\n{'metric':<32} {'frac targets rho>0':>20} {'frac AUROC>0.5':>18}")
    print("-" * 72)
    for name in names:
        if name not in report:
            continue
        s = report[name]["per_target_summary"]
        print(f"{name:<32} {s['spearman_all']['frac_positive']:>20.2f} "
              f"{s['auroc']['frac_positive']:>18.2f}")

    print("\nNormalised distance by RMSD bin (mean over targets).")
    print("The critic needs this to keep rising in the low bins, where x0_hat lives.\n")
    label = f"{'metric':<32}" + "".join(f"{f'{lo}-{hi}A':>12}" for lo, hi in bins)
    print(label)
    print("-" * len(label))
    for name in names:
        if name not in report:
            continue
        cells, profile = "", {}
        for lo, hi in bins:
            vals = binned[name][(lo, hi)]
            cells += f"{np.mean(vals):>12.3f}" if vals else f"{'-':>12}"
            profile[f"{lo}-{hi}"] = float(np.mean(vals)) if vals else None
        report[name]["rmsd_bin_profile"] = profile
        print(f"{name:<32}{cells}")

    report["_meta"] = {
        "encoder": args.encoder,
        "seed": args.seed,
        "n_targets": len(usable),
        "n_poses": total_poses,
        "near_native_threshold_A": NEAR_NATIVE_A,
        "decoy_threshold_A": DECOY_A,
        "contact_threshold_A": CONTACT_A,
        "note": "pocket_pool is NOT a negative control -- its representations "
                "are computed with message passing from the ligand and move "
                "with the pose. The floors are 'contacts' (no learning) and the "
                "--encoder random run (same architecture, permuted weights).",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


def encode_natives_and_cache(by_target, args):
    """Pretrained path: encode natives, take pose vectors from the feature cache.

    One forward pass per target instead of one per pose. Only valid because the
    cache was written by `scripts/featurize_block_level.py` under the same
    encoding conventions this script uses for the natives.
    """
    import torch
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
    from atomica_interface.scoring import load_encoder

    refs_path = Path("results/critic_gate/native_refs.npz")
    blocks_wanted = ("graph", "ligand_pool", "pocket_pool")
    if refs_path.exists():
        print(f"Reusing native references: {refs_path}")
        cached = np.load(refs_path, allow_pickle=True)
        refs = {t: {k: cached[f"{t}__{k}"] for k in blocks_wanted}
                for t in cached["targets"]}
    else:
        device = args.device if torch.cuda.is_available() else "cpu"
        print(f"Encoding native references on {device}...")
        model = load_encoder(
            "ATOMICA/pretrain/pretrain_model_config.json",
            "ATOMICA/pretrain/pretrain_model_weights.pt",
            device,
        )
        refs = {}
        for i, (target, rows) in enumerate(sorted(by_target.items()), 1):
            first = rows[0]
            template = Chem.MolFromSmiles(component_smiles(first["ligand"]) or "")
            native = Chem.SDMolSupplier(first["native_sdf"])[0]
            if template is None or native is None:
                continue
            try:
                pocket = pocket_blocks_from_pdb(first["receptor"])
                site, _ = blocks_interface(
                    pocket, ligand_blocks_from_mol(native), args.site_radius
                )
                fixed = AllChem.AssignBondOrdersFromTemplate(
                    template, Chem.RemoveAllHs(native, sanitize=False)
                )
                record = interface_data(
                    site, ligand_blocks_from_mol(fixed), args.dist_th, trim=False
                )
                batch = to_batch(record, device)
                with torch.no_grad():
                    out = model.infer(batch)
                block_repr = out.block_repr.detach().cpu().numpy()
                graph_repr = out.graph_repr.detach().cpu().numpy().reshape(-1)
                segments = batch["segment_ids"].detach().cpu().numpy()
            except Exception as exc:
                print(f"  [{i}/{len(by_target)}] {target}: SKIPPED "
                      f"({type(exc).__name__}: {exc})")
                continue
            if block_repr.shape[0] != segments.shape[0]:
                continue
            refs[target] = {
                "graph": graph_repr.astype(np.float32),
                "ligand_pool": pool(block_repr[segments == 1]),
                "pocket_pool": pool(block_repr[segments == 0]),
            }
        refs_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            refs_path, targets=np.array(sorted(refs)),
            **{f"{t}__{k}": v for t, d in refs.items() for k, v in d.items()},
        )
        print(f"Encoded {len(refs)} native references -> {refs_path}")

    data = np.load(args.features, allow_pickle=True)
    groups = data["groups"].astype(str)
    poses = {}
    for target in sorted(set(groups)):
        mask = groups == target
        poses[target] = {
            "graph": data["graph"][mask],
            "ligand_pool": data["ligand_pool"][mask],
            "pocket_pool": data["pocket_pool"][mask],
            "rmsd": data["y"][mask].astype(float),
            "smina": data["smina"][mask].astype(float),
        }
    return refs, poses


if __name__ == "__main__":
    main()
