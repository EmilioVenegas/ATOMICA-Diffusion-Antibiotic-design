"""Measure how much pocket information each featurization preserves.

The claim underpinning this project's redirection is that
`scripts/process_expert_atomica.py` destroyed the conditioning signal by
collapsing every pocket into a single `UNK` block in one segment. So far that is
an argument from how ATOMICA was pretrained. This turns it into a measurement.

Controlled comparison: the same pockets, the same single-segment setup (there is
no ligand at generation time, so both are pocket-only), differing **only** in
block vocabulary.

    old   [GLB, UNK]                 -- every pocket is "one unknown thing"
    new   [GLB, RES, RES, RES, ...]  -- one block per residue, real amino-acid types

Two readouts, neither requiring labels:

1. **Pairwise cosine similarity between different pockets.** If the old encoding
   maps every pocket to nearly the same vector, pocket identity is gone, and no
   conditioning mechanism downstream can recover it.
2. **Linear probe for amino-acid composition.** A concrete, verifiable property
   the pocket embedding ought to carry. Scored by cross-validated R^2.

Usage (from repo root):

    python scripts/featurization_probe.py --benchmark data/pose_benchmark
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

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


def old_style_record(site_blocks):
    """Reproduce the original featurization: one segment, one `UNK` block.

    Mirrors `scripts/process_expert_atomica.py`: a global node followed by every
    pocket atom in a single block typed `UNK`, all in segment 0.
    """
    from ATOMICA.data.pdb_utils import VOCAB

    coords, atom_types = [], []
    for block in site_blocks:
        for atom in block.units:
            coords.append(atom.get_coord())
            atom_types.append(VOCAB.atom_to_idx(atom.get_element()))

    coords = np.asarray(coords, dtype=np.float32)
    global_pos = coords.mean(axis=0)

    X = np.vstack([global_pos, coords]).tolist()
    A = [VOCAB.get_atom_global_idx()] + atom_types
    B = [VOCAB.symbol_to_idx(VOCAB.GLB), VOCAB.abrv_to_idx("UNK")]
    block_lengths = [1, len(atom_types)]
    segment_ids = [0, 0]
    return {"X": X, "A": A, "B": B, "block_lengths": block_lengths,
            "segment_ids": segment_ids}


def new_style_record(site_blocks):
    """The corrected featurization: one block per residue, real block types."""
    from ATOMICA.data.dataset import blocks_to_data

    return blocks_to_data(site_blocks)


def encode(model, record, device):
    """Return (graph_repr, mean unit_repr) for one pocket record."""
    import torch

    from atomica_interface.featurize import to_batch

    batch = to_batch(record, device)
    with torch.no_grad():
        out = model.infer(batch)
    graph = out.graph_repr.detach().cpu().numpy().reshape(-1)
    unit = out.unit_repr.detach().cpu().numpy().mean(axis=0)
    return graph, unit


def cosine_offdiag(matrix):
    """Mean cosine similarity between distinct rows."""
    norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    n = len(sim)
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def probe_r2(features, targets, seed=0):
    """Cross-validated R^2 for predicting composition from the embedding."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    preds = np.zeros_like(targets, dtype=float)
    for train, test in KFold(n_splits=5, shuffle=True, random_state=seed).split(features):
        scaler = StandardScaler().fit(features[train])
        model = RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(
            scaler.transform(features[train]), targets[train]
        )
        preds[test] = model.predict(scaler.transform(features[test]))
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="data/pose_benchmark")
    p.add_argument("--site_radius", type=float, default=10.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/featurization_probe")
    args = p.parse_args()

    import torch
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    from ATOMICA.data.dataset import blocks_interface
    from atomica_interface.featurize import ligand_blocks_from_mol, pocket_blocks_from_pdb
    from atomica_interface.scoring import load_encoder

    device = args.device if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader(open(Path(args.benchmark) / "manifest.csv")))
    by_target = collections.defaultdict(list)
    for row in rows:
        by_target[row["target"]].append(row)

    model = load_encoder("ATOMICA/pretrain/pretrain_model_config.json",
                         "ATOMICA/pretrain/pretrain_model_weights.pt", device)

    old_g, old_u, new_g, new_u, comps, names = [], [], [], [], [], []
    old_blocks, new_blocks, new_types = [], [], []

    for i, (target, target_rows) in enumerate(sorted(by_target.items()), 1):
        first = target_rows[0]
        try:
            native = Chem.SDMolSupplier(first["native_sdf"])[0]
            pocket = pocket_blocks_from_pdb(first["receptor"])
            site, _ = blocks_interface(
                pocket, ligand_blocks_from_mol(native), args.site_radius
            )
            if not site:
                continue
            og, ou = encode(model, old_style_record(site), device)
            ng, nu = encode(model, new_style_record(site), device)
        except Exception as exc:
            print(f"  [{i}] {target}: skipped ({type(exc).__name__})")
            continue

        counts = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
        for block in site:
            symbol = getattr(block, "symbol", "")
            try:
                abrv = __import__(
                    "ATOMICA.data.pdb_utils", fromlist=["VOCAB"]
                ).VOCAB.symbol_to_abrv(symbol)
            except Exception:
                abrv = None
            if abrv in AMINO_ACIDS:
                counts[AMINO_ACIDS.index(abrv)] += 1
        if counts.sum() == 0:
            continue

        old_g.append(og); old_u.append(ou); new_g.append(ng); new_u.append(nu)
        comps.append(counts / counts.sum())
        names.append(target)
        old_blocks.append(2)
        new_blocks.append(len(new_style_record(site)["B"]))
        new_types.append(len(set(new_style_record(site)["B"])))
        if i % 20 == 0:
            print(f"  [{i}/{len(by_target)}] encoded {len(names)} pockets")

    if len(names) < 10:
        raise SystemExit(f"only {len(names)} pockets encoded -- too few")

    old_g = np.asarray(old_g); new_g = np.asarray(new_g)
    old_u = np.asarray(old_u); new_u = np.asarray(new_u)
    comps = np.asarray(comps)

    report = {
        "n_pockets": len(names),
        "blocks_per_pocket": {
            "old": {"mean_blocks": float(np.mean(old_blocks)), "distinct_types": 2},
            "new": {"mean_blocks": round(float(np.mean(new_blocks)), 1),
                    "mean_distinct_types": round(float(np.mean(new_types)), 1)},
        },
        "mean_pairwise_cosine_between_pockets": {
            "old_graph": round(cosine_offdiag(old_g), 4),
            "new_graph": round(cosine_offdiag(new_g), 4),
            "old_unit_mean": round(cosine_offdiag(old_u), 4),
            "new_unit_mean": round(cosine_offdiag(new_u), 4),
        },
        "composition_probe_r2": {
            "old_graph": round(probe_r2(old_g, comps, args.seed), 4),
            "new_graph": round(probe_r2(new_g, comps, args.seed), 4),
            "old_unit_mean": round(probe_r2(old_u, comps, args.seed), 4),
            "new_unit_mean": round(probe_r2(new_u, comps, args.seed), 4),
        },
    }
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "featurization_probe.json").write_text(json.dumps(report, indent=2))

    c = report["mean_pairwise_cosine_between_pockets"]
    r = report["composition_probe_r2"]
    print(f"\n{len(names)} pockets")
    print(f"blocks per pocket: old 2 (1 type) | new {report['blocks_per_pocket']['new']['mean_blocks']} "
          f"({report['blocks_per_pocket']['new']['mean_distinct_types']} distinct types)")
    print(f"\n{'readout':<14}{'old':>10}{'new':>10}")
    print(f"{'cos (graph)':<14}{c['old_graph']:>10.4f}{c['new_graph']:>10.4f}   lower = pockets more distinguishable")
    print(f"{'cos (unit)':<14}{c['old_unit_mean']:>10.4f}{c['new_unit_mean']:>10.4f}")
    print(f"{'R2 (graph)':<14}{r['old_graph']:>10.4f}{r['new_graph']:>10.4f}   higher = composition recoverable")
    print(f"{'R2 (unit)':<14}{r['old_unit_mean']:>10.4f}{r['new_unit_mean']:>10.4f}")
    print(f"\nwrote {outdir/'featurization_probe.json'}")


if __name__ == "__main__":
    main()
