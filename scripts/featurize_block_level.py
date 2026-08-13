"""Encode benchmark poses using ATOMICA's block-level representation.

`graph_repr` compresses a whole complex into 32 numbers. Pose quality is a local,
per-contact property, and a head reading it through 32 channels overfits as soon
as capacity is added (see `results/pose_scorer/README.md`) -- the signature of a
representation bottleneck rather than an undertrained head.

`infer` returns per-atom, per-block and pooled representations from a single
forward pass, so this extracts all of them and pools the block level **separately
for the pocket and ligand segments**. Pooling across both at once would average
the two sides of the interface together and discard exactly the asymmetry that
distinguishes a good pose from a bad one.

Writes a cache with several feature blocks so the head can be trained on each
without re-encoding:

    graph          32   pooled whole-complex representation (the current baseline)
    ligand_pool    96   mean/max/std over ligand blocks
    pocket_pool    96   mean/max/std over pocket blocks
    contact_pool   64   mean/max over the pocket blocks nearest the ligand

Usage (from repo root):

    python scripts/featurize_block_level.py --benchmark data/pose_benchmark
"""

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


def pool(matrix):
    """mean/max/std over rows -> a fixed-length descriptor."""
    if matrix.shape[0] == 0:
        width = matrix.shape[1]
        return np.zeros(3 * width, dtype=np.float32)
    return np.concatenate(
        [matrix.mean(0), matrix.max(0), matrix.std(0)]
    ).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="data/pose_benchmark")
    p.add_argument("--site_radius", type=float, default=10.0)
    p.add_argument("--dist_th", type=float, default=8.0)
    p.add_argument("--n_contact", type=int, default=8,
                   help="pocket blocks nearest the ligand to pool as the contact shell")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/pose_scorer/features_block.npz")
    args = p.parse_args()

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
    bench = Path(args.benchmark)
    rows = list(csv.DictReader(open(bench / "manifest.csv")))
    by_target = collections.defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)

    model = load_encoder(
        "ATOMICA/pretrain/pretrain_model_config.json",
        "ATOMICA/pretrain/pretrain_model_weights.pt",
        device,
    )

    out = {k: [] for k in ("graph", "ligand_pool", "pocket_pool", "contact_pool")}
    y, groups, smina = [], [], []

    for i, (target, target_rows) in enumerate(sorted(by_target.items()), 1):
        first = target_rows[0]
        template = Chem.MolFromSmiles(component_smiles(first["ligand"]) or "")
        native = Chem.SDMolSupplier(first["native_sdf"])[0]
        if template is None or native is None:
            continue

        # Guarded: ATOMICA's PS_300 tokenizer raises on elements outside its
        # valence table (metal-containing ligands such as haem raise
        # KeyError: 'Fe'). Unguarded, one such target aborts the whole run.
        try:
            pocket = pocket_blocks_from_pdb(first["receptor"])
            site, _ = blocks_interface(
                pocket, ligand_blocks_from_mol(native), args.site_radius
            )
        except Exception as exc:
            print(f"  [{i}/{len(by_target)}] {target}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        site_centroids = np.array(
            [np.mean([a.get_coord() for a in b.units], axis=0) for b in site]
        )

        poses = list(Chem.SDMolSupplier(first["poses_file"], sanitize=False))
        kept = 0
        for row in target_rows:
            pose = poses[int(row["pose_index"])]
            if pose is None:
                continue
            try:
                fixed = AllChem.AssignBondOrdersFromTemplate(
                    template, Chem.RemoveAllHs(pose, sanitize=False)
                )
                record = interface_data(
                    site, ligand_blocks_from_mol(fixed), args.dist_th, trim=False
                )
                batch = to_batch(record, device)
                with torch.no_grad():
                    result = model.infer(batch)
                block_repr = result.block_repr.detach().cpu().numpy()
                graph_repr = result.graph_repr.detach().cpu().numpy().reshape(-1)
                segments = batch["segment_ids"].detach().cpu().numpy()
            except Exception:
                continue

            if block_repr.shape[0] != segments.shape[0]:
                continue  # a mismatch would silently misassign the segments

            pocket_blocks = block_repr[segments == 0]
            ligand_blocks = block_repr[segments == 1]

            # Contact shell: the site blocks closest to this pose specifically.
            # Unlike the pocket pool, which is identical for every pose of a
            # target, this varies with where the ligand actually sits.
            ligand_xyz = fixed.GetConformer().GetPositions()
            # site_centroids excludes the per-segment global node that
            # blocks_to_data prepends, so drop it before aligning indices.
            pocket_no_global = pocket_blocks[1:]
            n = min(len(site_centroids), len(pocket_no_global))
            if n == 0:
                continue
            dist = np.linalg.norm(
                site_centroids[:n, None, :] - ligand_xyz[None, :, :], axis=2
            ).min(axis=1)
            nearest = np.argsort(dist)[: args.n_contact]
            contact = pocket_no_global[nearest]
            contact_pool = np.concatenate(
                [contact.mean(0), contact.max(0)]
            ).astype(np.float32)

            out["graph"].append(graph_repr.astype(np.float32))
            out["ligand_pool"].append(pool(ligand_blocks))
            out["pocket_pool"].append(pool(pocket_blocks))
            out["contact_pool"].append(contact_pool)
            y.append(float(row["rmsd"]))
            groups.append(target)
            smina.append(float(row["smina_score"]) if row["smina_score"] else np.nan)
            kept += 1

        print(f"  [{i}/{len(by_target)}] {target}: {kept} poses")

    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        y=np.asarray(y, dtype=np.float32),
        groups=np.asarray(groups),
        smina=np.asarray(smina, dtype=np.float32),
        **arrays,
    )
    print("\nfeature blocks:")
    for k, v in arrays.items():
        print(f"  {k:<14} {v.shape}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
