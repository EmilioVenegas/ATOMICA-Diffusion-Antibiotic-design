"""Cache what the ATOMICA critic needs onto an already-processed dataset.

The critic loss

    L = L_diffusion + lambda * d( ATOMICA(pocket, x0_hat), ATOMICA(pocket, x_true) )

needs two things per complex that `scripts/process_expert_atomica.py` does not
currently write:

1. **The target** `ATOMICA(pocket, x_true)`. It depends only on fixed inputs, so
   it is a constant -- caching it makes the critic cost one encoder pass per
   training step instead of two.
2. **The record's static structure** -- `A`, `B`, `block_lengths`,
   `segment_ids`, and the permutations mapping stored coordinates into record
   rows. None of it depends on coordinates (block identity comes from the
   amino-acid sequence and the PS_300 fragmentation, not geometry), so it can be
   computed once here and reused every step, which is exactly what makes the
   encoder differentiable with respect to `x0_hat` at training time.

This runs as a post-pass over the written `.pt` files rather than as part of
preprocessing, so an in-flight preprocessing run does not have to be restarted.
It re-reads each complex's original LMDB record -- keyed by the `name` field the
preprocessor stored -- because the ligand bond table needed for fragmentation is
not carried in the `.pt`.

The two alignment invariants are asserted per complex rather than trusted:
`blocks_to_data` prepends a synthetic global atom to each segment, and
fragmentation permutes the ligand's atoms. Either one, unaccounted for, misaligns
every row silently with plausible-looking values.

Usage (from repo root):

    python scripts/add_critic_targets.py --data_dir data/processed_expert_atomica/train
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from atomica_interface.featurize import (
    atom_segment_ids,
    interface_data,
    ligand_blocks_from_arrays,
    pocket_blocks_from_arrays,
    to_batch,
)
from atomica_interface.scoring import load_encoder
from DiffSBDD.constants import ATOMICA_TO_DRUGLIKE_MAP, DRUGLIKE_ATOMS_DECODER

POCKET_SEGMENT = 0
LIGAND_SEGMENT = 1


def build_name_index(manifest_path):
    """ligand_filename -> LMDB cursor index, from the cached manifest."""
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    index = {}
    for i, name in enumerate(manifest):
        if name is not None and name not in index:
            index[name] = i
    return index, manifest


def critic_metadata(data, encoder, device, args):
    """Static record structure plus ATOMICA(pocket, x_true) for one complex.

    Returns ``(payload, reason)``; ``payload`` is None when the complex cannot be
    rebuilt, in which case it is left without critic fields and the training
    loader skips the critic term for it.
    """
    def as_numpy(value):
        return value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    pocket_coords = as_numpy(data["protein_pos"])
    pocket_elements = as_numpy(data["protein_element"])
    atom2residue = as_numpy(data["protein_atom2residue"])
    residue_aa = as_numpy(data["amino_acid"])
    ligand_coords = as_numpy(data["ligand_pos"])
    ligand_elements = as_numpy(data["ligand_element"])

    pocket_blocks, pocket_atom_index = pocket_blocks_from_arrays(
        pocket_coords, pocket_elements, atom2residue, residue_aa
    )
    ligand_blocks, lig_atom_order = ligand_blocks_from_arrays(
        ligand_coords, ligand_elements,
        as_numpy(data["ligand_bond_index"]), as_numpy(data["ligand_bond_type"]),
        fragmentation_method=args.fragmentation, return_atom_order=True,
    )

    # trim=False is required: trimming selects blocks by distance, which would
    # make the block structure depend on coordinates and invalidate caching it.
    record = interface_data(pocket_blocks, ligand_blocks, dist_th=args.interface_dist,
                            trim=False)

    record_atom_types = np.asarray(record["A"], dtype=np.int64)
    record_coords = np.asarray(record["X"], dtype=np.float32)
    per_atom_segment = atom_segment_ids(record)

    with torch.no_grad():
        out = encoder.infer(to_batch(record, device=device))
    graph_repr = out.graph_repr.detach().cpu().numpy().reshape(-1)
    unit_repr = out.unit_repr.detach().cpu().numpy()

    if not (unit_repr.shape[0] == len(record_atom_types) == len(per_atom_segment)):
        return None, "row_count_mismatch"

    # --- Pocket rows ---
    # Reproduce exactly the mask the preprocessor used, so the cached
    # permutation indexes the *stored* pocket_coords rather than the record.
    pocket_rows = per_atom_segment == POCKET_SEGMENT
    pocket_druglike = ATOMICA_TO_DRUGLIKE_MAP[record_atom_types[pocket_rows]]
    valid_pocket = pocket_druglike != -1

    # The only row the druglike mask should drop is the synthetic global atom
    # that blocks_to_data prepends. CrossDocked pockets carry no hydrogens, so
    # anything else being dropped means the assumption has changed and the
    # permutation below would be wrong.
    if int((~valid_pocket).sum()) != 1 or bool(valid_pocket[0]):
        return None, "unexpected_pocket_mask"
    if not np.allclose(record_coords[pocket_rows][valid_pocket],
                       np.asarray(data["_stored_pocket_coords"], dtype=np.float32),
                       atol=1e-3):
        return None, "pocket_coords_mismatch"

    ligand_rows = per_atom_segment == LIGAND_SEGMENT
    ligand_druglike = ATOMICA_TO_DRUGLIKE_MAP[record_atom_types[ligand_rows]]
    valid_ligand = ligand_druglike != -1
    if int((~valid_ligand).sum()) != 1 or bool(valid_ligand[0]):
        return None, "unexpected_ligand_mask"
    if len(lig_atom_order) != int(valid_ligand.sum()):
        return None, "ligand_order_length_mismatch"

    # The permutation must reproduce the record's ligand coordinates from the
    # stored ones. This is the check that catches a silent misalignment.
    stored_lig = np.asarray(data["_stored_lig_coords"], dtype=np.float32)
    if not np.allclose(record_coords[ligand_rows][valid_ligand],
                       stored_lig[lig_atom_order], atol=1e-3):
        return None, "ligand_order_mismatch"

    payload = {
        "critic_A": torch.from_numpy(record_atom_types),
        "critic_B": torch.from_numpy(np.asarray(record["B"], dtype=np.int64)),
        "critic_block_lengths": torch.from_numpy(
            np.asarray(record["block_lengths"], dtype=np.int64)),
        "critic_segment_ids": torch.from_numpy(
            np.asarray(record["segment_ids"], dtype=np.int64)),
        # Identity for the pocket (verified above), explicit so the critic never
        # has to assume it.
        "critic_pocket_atom_order": torch.arange(int(valid_pocket.sum()), dtype=torch.long),
        "critic_lig_atom_order": torch.from_numpy(lig_atom_order),
        "critic_graph_repr_true": torch.from_numpy(graph_repr.astype(np.float32)),
        "critic_unit_repr_true": torch.from_numpy(
            unit_repr.mean(axis=0).astype(np.float32)),
    }
    return payload, "success"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_dir", required=True,
                   help="a processed split directory holding complex_*.pt")
    p.add_argument("--lmdb", default="data/crossdocked_pocket10_processed.lmdb")
    p.add_argument("--manifest", default="data/lmdb_index_manifest.json",
                   help="written by scripts/build_holdout_split.py")
    p.add_argument("--model_config", default="ATOMICA/pretrain/pretrain_model_config.json")
    p.add_argument("--model_weights", default="ATOMICA/pretrain/pretrain_model_weights.pt")
    p.add_argument("--fragmentation", default="PS_300")
    p.add_argument("--interface_dist", type=float, default=8.0)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even for files that already carry critic fields")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("complex_*.pt"))
    if not files:
        print(f"No complex_*.pt under {data_dir}")
        return
    print(f"{len(files)} complexes in {data_dir}")

    name_index, _ = build_name_index(args.manifest)
    print(f"Manifest covers {len(name_index)} distinct ligand filenames.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = load_encoder(args.model_config, args.model_weights, device=str(device))
    print(f"Loaded encoder on {device}.")

    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)

    # One cursor pass, collecting only the records these files need. Random
    # access by cursor position is not possible, and re-scanning per file would
    # be quadratic.
    wanted = {}
    for path in files:
        stored = torch.load(path)
        name = stored.get("name")
        if name is None or name not in name_index:
            continue
        if not args.overwrite and "critic_graph_repr_true" in stored:
            continue
        wanted.setdefault(name_index[name], []).append(path)
    print(f"{len(wanted)} LMDB records to re-read.")
    if not wanted:
        print("Nothing to do.")
        return

    counts = {}
    written = 0
    with env.begin() as txn:
        for idx, (_, value) in enumerate(tqdm(txn.cursor(), total=env.stat()["entries"],
                                              desc="Scanning")):
            if idx not in wanted:
                continue
            record = pickle.loads(value)
            for path in wanted[idx]:
                stored = torch.load(path)
                record["_stored_pocket_coords"] = stored["pocket_coords"].numpy()
                record["_stored_lig_coords"] = stored["lig_coords"].numpy()
                try:
                    payload, reason = critic_metadata(record, encoder, device, args)
                except Exception as exc:
                    payload, reason = None, f"exception_{type(exc).__name__}"
                counts[reason] = counts.get(reason, 0) + 1
                if payload is None:
                    continue
                stored.update(payload)
                torch.save(stored, path)
                written += 1
            if args.limit is not None and written >= args.limit:
                break
    env.close()

    print(f"\nAugmented {written} complexes.")
    for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {n}")
    if written < len(wanted):
        print("\nComplexes without critic fields keep training normally; the "
              "critic term is simply skipped for them.")


if __name__ == "__main__":
    main()
