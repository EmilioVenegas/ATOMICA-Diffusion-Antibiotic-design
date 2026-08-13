"""Phase 0 primary gate: is ATOMICA's interface representation geometry-sensitive?

This is a cheaper and more fundamental test than binder/decoy discrimination, and
it needs only a single co-crystal structure -- no docking, no decoy set, no poses
to invent.

Take the native ligand pose from a complex, generate rigid perturbations of it,
and ask whether the two-segment interface representation distinguishes near-native
poses from displaced ones. The logic is unavoidable:

  * If ATOMICA cannot tell a native pose from one displaced by a few angstroms,
    its interface representation is not sensitive to interaction geometry, and no
    conditioning, guidance or hotspot scheme built on it can be either. Stop.
  * If it can, the representation encodes something about interaction geometry,
    and the binder/decoy test (phase0_discriminate.py) becomes worth running.

Perturbations are kept small enough that the ligand stays in contact with the
pocket. A ligand displaced clear of the protein would be trivially separable and
would inflate the result without saying anything about interface quality.

Usage (from repo root):

    python scripts/phase0_pose_sensitivity.py \
        --pocket data/1h1s.pdb --chains A --ligand 4SP --ligand_chain A \
        --out results/phase0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from atomica_interface.scoring import (  # noqa: E402
    cross_validated_probe,
    permutation_test,
    roc_auc,
)


def random_rotation(rng):
    """Uniformly random 3D rotation via QR of a Gaussian matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))  # fix signs so the factorisation is unique
    if np.linalg.det(q) < 0:  # reflection -> rotation
        q[:, 0] *= -1
    return q


def axis_angle_rotation(rng, angle_range_deg):
    """Rotation about a random axis by an angle drawn from ``angle_range_deg``.

    Rotation magnitude has to be bounded separately from translation. A uniformly
    random rotation displaces atoms far from the centroid by many angstroms
    regardless of any translation limit, so using one for "near-native" poses
    produces nothing of the kind.

    The range needs a floor as well as a ceiling: sampling the displaced class
    from [0, 180] degrees occasionally returns a near-native pose and mislabels
    it, which is label noise that depresses the measured separation.
    """
    lo, hi = angle_range_deg
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(np.radians(lo), np.radians(hi))
    # Rodrigues' rotation formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def perturb(coords, rng, shift_range, angle_range_deg):
    """Rigidly perturb a pose about its own centroid; return (coords, RMSD).

    Both magnitudes are ranges with a floor, so the two pose classes stay
    separated rather than overlapping through occasional small draws. RMSD is
    measured against the input pose without realignment, so it reflects the
    actual displacement rather than a best-fit residual.
    """
    centre = coords.mean(axis=0)
    local = (coords - centre) @ axis_angle_rotation(rng, angle_range_deg).T
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    moved = local + centre + direction * rng.uniform(*shift_range)
    rmsd = float(np.sqrt(((moved - coords) ** 2).sum(axis=1).mean()))
    return moved, rmsd


def set_coords(mol, coords):
    from copy import deepcopy

    out = deepcopy(mol)
    conf = out.GetConformer()
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pocket", required=True)
    p.add_argument("--chains", nargs="*", default=None)
    p.add_argument("--ligand", required=True, help="HETATM residue name, e.g. 4SP")
    p.add_argument("--ligand_chain", default=None)
    p.add_argument("--atomica_config", default="ATOMICA/pretrain/pretrain_model_config.json")
    p.add_argument("--atomica_weights", default="ATOMICA/pretrain/pretrain_model_weights.pt")
    p.add_argument("--n_poses", type=int, default=60, help="per class")
    p.add_argument("--near_shift", type=float, nargs=2, default=[0.0, 0.5],
                   help="translation range for the near-native class (A)")
    p.add_argument("--near_angle", type=float, nargs=2, default=[0.0, 10.0],
                   help="rotation range for the near-native class (deg)")
    p.add_argument("--far_shift", type=float, nargs=2, default=[2.0, 4.0],
                   help="translation range for the displaced class (A)")
    p.add_argument("--far_angle", type=float, nargs=2, default=[60.0, 180.0],
                   help="rotation range for the displaced class (deg)")
    p.add_argument("--dist_th", type=float, default=8.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/phase0")
    args = p.parse_args()

    import torch

    from atomica_interface.featurize import (
        interface_data,
        ligand_blocks_from_mol,
        ligand_from_pdb_het,
        pocket_blocks_from_pdb,
        summarize,
        to_batch,
    )
    from atomica_interface.scoring import interface_representation, load_encoder

    device = args.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] pocket {args.pocket} chains={args.chains or 'all'}")
    pocket_blocks = pocket_blocks_from_pdb(args.pocket, args.chains)
    print(f"      {len(pocket_blocks)} residue blocks")

    print(f"[2/4] native ligand {args.ligand}")
    native = ligand_from_pdb_het(args.pocket, args.ligand, args.ligand_chain)
    native_coords = native.GetConformer().GetPositions()
    print(f"      {native.GetNumAtoms()} atoms, centroid {np.round(native_coords.mean(0), 1)}")

    print(f"[3/4] encoder on {device}")
    model = load_encoder(args.atomica_config, args.atomica_weights, device)

    def encode(coords):
        mol = set_coords(native, coords)
        blocks = ligand_blocks_from_mol(mol)
        data = interface_data(pocket_blocks, blocks, args.dist_th)
        return interface_representation(model, to_batch(data, device)).reshape(-1), data

    print("[4/4] encoding perturbed poses")
    feats, labels, rmsds = [], [], []
    lost_contact = 0
    structure = None

    # Classes are defined by perturbation magnitude, and the resulting RMSD is
    # measured rather than targeted -- no rejection sampling, so no pose is
    # silently discarded for landing outside a band.
    # The native pose itself leads the near-native class.
    jobs = (
        [(None, None, 1)]
        + [(args.near_shift, args.near_angle, 1) for _ in range(args.n_poses - 1)]
        + [(args.far_shift, args.far_angle, 0) for _ in range(args.n_poses)]
    )

    for shift, angle, label in jobs:
        if shift is None:  # the native pose itself
            coords, rmsd = native_coords.copy(), 0.0
        else:
            coords, rmsd = perturb(native_coords, rng, shift, angle)
        try:
            vec, data = encode(coords)
        except ValueError:
            lost_contact += 1
            continue
        if structure is None:
            structure = summarize(data)
            print(f"      example record: {structure}")
        feats.append(vec)
        labels.append(label)
        rmsds.append(rmsd)

    labels = np.asarray(labels)
    feats = np.asarray(feats)
    rmsds = np.asarray(rmsds)
    print(
        f"      encoded {len(labels)} poses "
        f"({int(labels.sum())} near-native, {int((labels == 0).sum())} displaced); "
        f"{lost_contact} dropped for losing pocket contact"
    )
    if labels.sum() == 0 or (labels == 0).sum() == 0:
        raise SystemExit("need both classes -- widen the displacement bands")

    scores = cross_validated_probe(feats, labels, seed=args.seed)
    if scores is None:
        raise SystemExit("too few poses per class to cross-validate")

    auroc = roc_auc(labels, scores)
    p_value = permutation_test(labels, scores, 2000, args.seed)

    # Monotonicity: does the representation drift steadily away from native as the
    # pose degrades? Rank correlation avoids assuming the relationship is linear.
    native_vec = feats[0]
    drift = np.linalg.norm(feats - native_vec, axis=1)
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    a, b = rank(drift), rank(rmsds)
    spearman = float(np.corrcoef(a, b)[0, 1])

    report = {
        "pocket": args.pocket,
        "ligand": args.ligand,
        "n_poses": int(len(labels)),
        "near_class": {"shift_A": args.near_shift, "angle_deg": args.near_angle},
        "displaced_class": {"shift_A": args.far_shift, "angle_deg": args.far_angle},
        "rmsd_by_class": {
            "near_native": [
                round(float(rmsds[labels == 1].min()), 3),
                round(float(rmsds[labels == 1].max()), 3),
            ],
            "displaced": [
                round(float(rmsds[labels == 0].min()), 3),
                round(float(rmsds[labels == 0].max()), 3),
            ],
        },
        "dropped_lost_contact": lost_contact,
        "record_structure": structure,
        "auroc_near_vs_displaced": round(auroc, 4),
        "permutation_p": round(p_value, 5),
        "spearman_repr_drift_vs_rmsd": round(spearman, 4),
        "rmsd_range": [round(float(rmsds.min()), 3), round(float(rmsds.max()), 3)],
    }
    (outdir / "phase0_pose_sensitivity.json").write_text(json.dumps(report, indent=2))

    print(f"\nAUROC near-native vs displaced : {auroc:.3f}  (permutation p = {p_value:.4f})")
    print(f"Spearman(repr drift, RMSD)     : {spearman:+.3f}")
    print()
    if auroc < 0.65:
        print(
            "VERDICT: representation is not geometry-sensitive at the interface.\n"
            "         Conditioning, guidance and hotspot mapping all inherit this.\n"
            "         Stop and reconsider the premise."
        )
    elif spearman < 0.3:
        print(
            "VERDICT: separable but not smoothly ordered by pose quality.\n"
            "         Usable for ranking, weak as a guidance signal."
        )
    else:
        print(
            "VERDICT: geometry-sensitive and monotonic in pose quality.\n"
            "         Proceed to phase0_discriminate.py (binder vs decoy)."
        )
    print(f"\nwrote {outdir/'phase0_pose_sensitivity.json'}")


if __name__ == "__main__":
    main()
