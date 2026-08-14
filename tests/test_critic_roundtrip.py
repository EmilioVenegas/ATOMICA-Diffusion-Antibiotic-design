"""Correctness checks for the ATOMICA critic, on real CrossDocked complexes.

Unlike the other tests here this one needs the full environment (torch, ATOMICA,
the LMDB), so it is an integration test rather than a numpy unit test. It exists
because every failure mode of the critic is silent: a misapplied permutation, a
stale global-atom centroid or a broken gradient path all produce finite,
plausible-looking numbers.

Run from the repo root:

    python tests/test_critic_roundtrip.py --n 5
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import lmdb
import numpy as np
import torch

from atomica_interface.critic import ATOMICACritic, lambda_schedule
from atomica_interface.featurize import (
    atom_segment_ids,
    interface_data,
    ligand_blocks_from_arrays,
    pocket_blocks_from_arrays,
    to_batch,
)
from DiffSBDD.constants import ATOMICA_TO_DRUGLIKE_MAP

CONFIG = "ATOMICA/pretrain/pretrain_model_config.json"
WEIGHTS = "ATOMICA/pretrain/pretrain_model_weights.pt"


def build_case(record, fragmentation="PS_300"):
    """One complex: record metadata, stored-style coordinates, reference X."""
    to_np = lambda v: v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    pocket_blocks, _ = pocket_blocks_from_arrays(
        to_np(record["protein_pos"]), to_np(record["protein_element"]),
        to_np(record["protein_atom2residue"]), to_np(record["amino_acid"]),
    )
    ligand_blocks, lig_order = ligand_blocks_from_arrays(
        to_np(record["ligand_pos"]), to_np(record["ligand_element"]),
        to_np(record["ligand_bond_index"]), to_np(record["ligand_bond_type"]),
        fragmentation_method=fragmentation, return_atom_order=True,
    )
    data = interface_data(pocket_blocks, ligand_blocks, trim=False)

    atom_types = np.asarray(data["A"], dtype=np.int64)
    coords = np.asarray(data["X"], dtype=np.float32)
    segments = atom_segment_ids(data)

    pocket_rows = segments == 0
    ligand_rows = segments == 1
    pocket_valid = ATOMICA_TO_DRUGLIKE_MAP[atom_types[pocket_rows]] != -1
    ligand_valid = ATOMICA_TO_DRUGLIKE_MAP[atom_types[ligand_rows]] != -1

    meta = {
        "A": torch.from_numpy(atom_types),
        "B": torch.from_numpy(np.asarray(data["B"], dtype=np.int64)),
        "block_lengths": torch.from_numpy(
            np.asarray(data["block_lengths"], dtype=np.int64)),
        "segment_ids": torch.from_numpy(
            np.asarray(data["segment_ids"], dtype=np.int64)),
        "pocket_atom_order": torch.arange(int(pocket_valid.sum()), dtype=torch.long),
        "lig_atom_order": torch.from_numpy(lig_order),
    }
    stored_pocket = torch.from_numpy(coords[pocket_rows][pocket_valid])
    stored_lig = torch.from_numpy(
        np.asarray(to_np(record["ligand_pos"]), dtype=np.float32)
    )
    return meta, stored_pocket, stored_lig, torch.from_numpy(coords), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/crossdocked_pocket10_processed.lmdb")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    critic = ATOMICACritic(CONFIG, WEIGHTS, distance="cosine", level="graph")
    critic.to(args.device)

    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    cases, failures = 0, []
    with env.begin() as txn:
        for idx, (_, value) in enumerate(txn.cursor()):
            if cases >= args.n:
                break
            record = pickle.loads(value)
            try:
                meta, pocket, lig, reference_X, data = build_case(record)
            except Exception as exc:
                continue

            name = record.get("ligand_filename", f"idx{idx}")
            print(f"\n--- {name}  ({len(meta['B'])} blocks, {len(lig)} ligand atoms)")
            cases += 1

            def check(label, ok, detail=""):
                print(f"    {'PASS' if ok else 'FAIL'}  {label}{detail}")
                if not ok:
                    failures.append(f"{name}: {label}")

            # 1. Assembling X from stored coordinates must reproduce the record
            #    that blocks_to_data built, including both global atoms.
            assembled = ATOMICACritic.assemble_coords(meta, pocket, lig)
            max_dev = float((assembled - reference_X).abs().max())
            check("assemble_coords reproduces the record X",
                  max_dev < 1e-4, f"   (max deviation {max_dev:.2e} A)")

            # 2. Encoding x_true must reproduce the cached target exactly, so a
            #    perfect prediction really does score zero.
            with torch.no_grad():
                target = critic.encode([meta], [pocket], [lig])
                d_self = critic.pairwise_distance(
                    critic.encode([meta], [pocket], [lig]), target)
            check("d(x_true, x_true) == 0", float(d_self) < 1e-6,
                  f"   (d = {float(d_self):.2e})")

            # 3. Translating the whole complex must not change the
            #    representation. The training loader centres every complex on
            #    its joint centre of mass, so a translation-sensitive encoding
            #    would make the cached target wrong for every sample.
            shift = torch.tensor([12.0, -5.0, 3.0])
            with torch.no_grad():
                moved = critic.encode([meta], [pocket + shift], [lig + shift])
                d_shift = float(critic.pairwise_distance(moved, target))
            check("translation invariance", d_shift < 1e-5, f"   (d = {d_shift:.2e})")

            # 4. Distance profile against per-atom RMSD, plus the gradient
            #    magnitude at each displacement.
            #
            #    The gradient is deliberately NOT evaluated at x_true. Cosine
            #    distance is exactly zero there, which is its global minimum, so
            #    the gradient vanishes for the right reason and measuring it
            #    there says nothing about trainability. What matters is the
            #    gradient at a displaced ligand, which is where x0_hat actually
            #    sits during training.
            generator = torch.Generator().manual_seed(0)
            profile = []
            for rmsd in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
                noise = torch.randn(lig.shape, generator=generator)
                noise = noise / noise.pow(2).sum(1).mean().sqrt() * rmsd
                nudged = (lig + noise).requires_grad_(True)
                d = critic([meta], [pocket], [nudged], target)
                d.sum().backward()
                profile.append((rmsd, float(d), float(nudged.grad.norm())))
                print(f"           RMSD {rmsd:>4.2f} A -> d = {float(d):.5f}   "
                      f"||grad|| = {float(nudged.grad.norm()):.3e}")

            # Only the low-RMSD regime is asserted. That is where the critic is
            # applied (lambda is ramped off at high noise) and it is the only
            # regime ATOMICA has seen: past a few angstrom the ligand leaves the
            # pocket entirely, the interface stops existing, and the distance
            # saturates and turns over -- visible in the profile above.
            low = [d for r, d, _ in profile if r <= 2.0]
            check("distance increases monotonically below 2 A",
                  all(b > a for a, b in zip(low, low[1:])))
            check("gradient is non-zero where the critic is applied",
                  all(g > 0 for r, _, g in profile if r <= 2.0),
                  f"   (min ||grad|| below 2 A = "
                  f"{min(g for r, _, g in profile if r <= 2.0):.3e})")
    env.close()

    # The weight schedule is pure arithmetic but decides where the critic is
    # applied at all, so it is checked here rather than left implicit.
    t = torch.tensor([0, 250, 500, 750, 1000])
    ramp = lambda_schedule(t, T=1000, max_weight=1.0, mode="ramp", cutoff=0.5)
    cut = lambda_schedule(t, T=1000, max_weight=1.0, mode="cutoff", cutoff=0.5)
    print(f"\n--- lambda schedule (T=1000, max_weight=1.0, cutoff=0.5)")
    print(f"    t          {t.tolist()}")
    print(f"    ramp       {[round(v, 3) for v in ramp.tolist()]}")
    print(f"    cutoff     {[round(v, 3) for v in cut.tolist()]}")
    schedule_ok = (ramp[0] == 1.0 and ramp[-1] == 0.0 and bool((ramp[1:] <= ramp[:-1]).all())
                   and cut[0] == 1.0 and cut[-1] == 0.0)
    print(f"    {'PASS' if schedule_ok else 'FAIL'}  schedule is monotone and "
          f"switches off above the cutoff")
    if not schedule_ok:
        failures.append("lambda schedule")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"All checks passed over {cases} complexes.")


if __name__ == "__main__":
    main()
