"""Compute ATOMICA hotspot fields for a pocket and validate against its ligand.

Protocol follows Radoux et al. (J Med Chem 2016, 59:4314), the accepted test for
this class of method: assign every heavy atom of the crystal ligand a probe type,
look up its position in the *matching* probe's field, and express that as a
percentile of all grid scores in that field. Published reference values are a
median 97th percentile for fragment atoms and 72nd for elaborated lead atoms, so
the number here is directly comparable rather than self-invented.

Three controls decide whether a good number means anything:

* **buriedness** -- protein neighbour count, a field that knows no chemistry.
  Every method in this literature has scores that correlate with enclosure, and
  Fragment Hotspot Maps weights by buriedness for exactly this reason. If
  buriedness alone reaches the same percentile, the field is an enclosure
  detector.
* **type specificity** -- score each ligand atom against *all* probe fields and
  check the matching one wins. Without this, a field that only says "this voxel
  is enclosed" passes the primary metric.
* **random placement** -- percentile of randomly chosen accessible grid points,
  the floor.

Usage (from repo root):

    python scripts/hotspot_validate.py --pocket data/1h1s.pdb --chains A \
        --ligand 4SP --ligand_chain A --spacing 1.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from atomica_interface.hotspot import (  # noqa: E402
    DEFAULT_PROBES,
    buriedness,
    build_probe,
    classify_ligand_atom,
    percentile_rank,
    place,
    pocket_grid,
    random_rotations,
)


def score_field(model, site_blocks, probe_mol, points, rotations, dist_th, device,
                score_fn):
    """Best score over orientations at each grid point."""
    from atomica_interface.featurize import (
        interface_data,
        ligand_blocks_from_mol,
        to_batch,
    )

    out = np.full(len(points), np.nan, dtype=np.float32)
    for i, centre in enumerate(points):
        best = None
        for rot in rotations:
            posed = place(probe_mol, centre, rot)
            try:
                # trim=False: the pocket stays fixed across every grid point, so
                # the field cannot vary with how many residues a placement
                # happens to contact.
                record = interface_data(
                    site_blocks, ligand_blocks_from_mol(posed, None), dist_th, trim=False
                )
                value = score_fn(model, to_batch(record, device))
            except Exception:
                continue
            if best is None or value > best:
                best = value
        if best is not None:
            out[i] = best
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pocket", required=True)
    p.add_argument("--chains", nargs="*", default=None)
    p.add_argument("--ligand", required=True, help="HETATM residue name")
    p.add_argument("--ligand_chain", default=None)
    p.add_argument("--spacing", type=float, default=1.0,
                   help="grid spacing (A). 0.5 is the Fragment Hotspot Maps standard; "
                        "1.0 is used here for tractability and swept separately.")
    p.add_argument("--n_rotations", type=int, default=4)
    p.add_argument("--site_radius", type=float, default=10.0)
    p.add_argument("--dist_th", type=float, default=8.0)
    p.add_argument("--atomica_config", default="ATOMICA/pretrain/pretrain_model_config.json")
    p.add_argument("--atomica_weights", default="ATOMICA/pretrain/pretrain_model_weights.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/hotspot")
    args = p.parse_args()

    import torch
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    from ATOMICA.data.dataset import blocks_interface
    from atomica_interface.energy import load_denoiser, pose_energy
    from atomica_interface.featurize import (
        ligand_blocks_from_mol,
        ligand_from_pdb_het,
        pocket_blocks_from_pdb,
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] pocket {args.pocket}")
    pocket = pocket_blocks_from_pdb(args.pocket, args.chains)
    native = ligand_from_pdb_het(args.pocket, args.ligand, args.ligand_chain)
    site, _ = blocks_interface(pocket, ligand_blocks_from_mol(native), args.site_radius)
    site_coords = np.array([a.get_coord() for b in site for a in b.units], dtype=float)
    print(f"      site {len(site)} residue blocks, {len(site_coords)} atoms")

    print(f"[2/5] grid at {args.spacing} A")
    points = pocket_grid(site_coords, spacing=args.spacing)
    if len(points) == 0:
        raise SystemExit("empty grid -- loosen the contact bounds")
    bur = buriedness(points, site_coords)
    print(f"      {len(points)} accessible, non-clashing points")

    print(f"[3/5] encoder on {device}")
    # The training-free denoising energy: no fitting, no labels. Phase 0 measured
    # it at AUROC 0.787 against a 0.727 trivial baseline, so a flat field here is
    # a live possibility rather than a surprise.
    model = load_denoiser(args.atomica_config, args.atomica_weights, device)
    rotations = random_rotations(args.n_rotations, rng)

    def score_fn(m, batch):
        # Lower predicted correction = better placement, so negate for
        # higher-is-better.
        return -pose_energy(m, batch)["translation"]

    print(f"[4/5] scoring {len(DEFAULT_PROBES)} probes x {len(points)} points "
          f"x {args.n_rotations} orientations")
    fields, ranks = {}, {}
    for name, smiles in DEFAULT_PROBES.items():
        probe = build_probe(smiles, seed=args.seed)
        values = score_field(model, site, probe, points, rotations, args.dist_th,
                             device, score_fn)
        ok = ~np.isnan(values)
        fields[name] = values
        # Percentile rank within this probe's own field: removes both the
        # cross-pocket scale problem and the probe-size confound.
        r = np.full(len(values), np.nan)
        r[ok] = percentile_rank(values[ok])
        ranks[name] = r
        print(f"      {name:<9} scored {ok.sum()}/{len(points)}")

    print("[5/5] validating against the crystal ligand")
    lig_coords = native.GetConformer().GetPositions()
    atom_types = [classify_ligand_atom(a) for a in native.GetAtoms()]

    bur_rank = percentile_rank(bur)
    matched, buried_ctrl, specificity = [], [], []
    for xyz, kind in zip(lig_coords, atom_types):
        nearest = int(np.argmin(np.linalg.norm(points - xyz, axis=1)))
        if kind not in ranks or np.isnan(ranks[kind][nearest]):
            continue
        matched.append(ranks[kind][nearest])
        buried_ctrl.append(bur_rank[nearest])
        # Type specificity: does the matching probe outrank the others here?
        others = [ranks[k][nearest] for k in ranks
                  if k != kind and not np.isnan(ranks[k][nearest])]
        if others:
            specificity.append(ranks[kind][nearest] >= max(others))

    if not matched:
        raise SystemExit("no ligand atom landed on a scored grid point")

    random_ctrl = float(np.median(rng.choice(bur_rank, size=200)))
    report = {
        "pocket": args.pocket, "ligand": args.ligand,
        "grid_points": int(len(points)), "spacing_A": args.spacing,
        "n_ligand_atoms_scored": len(matched),
        "median_percentile_matched_probe": round(float(np.median(matched)), 1),
        "median_percentile_buriedness_control": round(float(np.median(buried_ctrl)), 1),
        "median_percentile_random_control": round(random_ctrl, 1),
        "type_specificity_rate": (round(float(np.mean(specificity)), 3)
                                  if specificity else None),
        "reference_radoux_fragments": 97,
        "reference_radoux_leads": 72,
    }
    (outdir / f"hotspot_{Path(args.pocket).stem}_{args.ligand}.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print(f"median percentile, matched probe : {report['median_percentile_matched_probe']}")
    print(f"median percentile, buriedness    : {report['median_percentile_buriedness_control']}   <- confound")
    print(f"median percentile, random        : {report['median_percentile_random_control']}   <- floor")
    print(f"type specificity (matching wins) : {report['type_specificity_rate']}")
    print(f"\nRadoux reference: 97 (fragments) / 72 (leads)")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
