"""Preprocess the expert CrossDocked subset into .pt complexes with ATOMICA embeddings.

Reads `data/crossdocked_pocket10_processed.lmdb` (the TargetDiff/Pocket2Mol-lineage
processed CrossDocked2020 pocket-10 set), keeps the "expert" complexes selected by
`expert_split.pt` (Vina < -8.5, see `scripts/create_expert_split.py`), encodes each
pocket with the frozen pretrained ATOMICA encoder, and writes one `.pt` per complex
for `DiffSBDD/dataset.py::LigandPocketDatasetPT`.

Two things were wrong with the original version of this script and are fixed here.

1. **It could not run.** It called `.infer()` on a model built by
   `DenoisePretrainModel.load_from_config_and_weights`, but `DenoisePretrainModel`
   defines no `infer` and its `forward` asserts on denoising targets. The supported
   route to pretrained representations is `PredictionModel._load_from_pretrained`,
   which rebuilds the same encoder with the denoising heads disabled; that is what
   `atomica_interface.scoring.load_encoder` does, and it is reused here.

2. **The featurization defeated the point of using ATOMICA.** The pocket was passed
   as a SINGLE segment with every pocket atom collapsed into ONE block of type
   `UNK`. ATOMICA is pretrained on two interacting segments over chemically-typed
   residue/fragment blocks (`ATOMICA/data/dataset_pretrain.py` splits on
   `segment_ids == 0` / `== 1`), so that input engaged none of its interaction
   semantics and erased its block vocabulary -- leaving only per-atom element and
   local geometry, which the EGNN already derives from coordinates. Here the pocket
   becomes one block per residue carrying its real amino-acid type (segment 0) and
   the ligand becomes PS_300 fragment blocks (segment 1). See docs/experiment-plan.md.

NOTE ON THE LIGAND SEGMENT: segment 1 is the *reference* ligand from the crystal /
docked pose. ATOMICA requires two segments, and at training time the ligand is
available -- but at generation time it is the thing being generated. The conditioning
signal therefore carries information about the ground-truth ligand that will not be
available at sampling time. This is the "central obstacle" in docs/experiment-plan.md
and it is a property of the approach, not of this script; it is called out here so it
is not mistaken for an oversight.
"""

import argparse
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

# Repo root, so `ATOMICA.*` and `DiffSBDD.*` resolve regardless of working dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from atomica_interface.featurize import (
    ATOMIC_NUMBER_TO_SYMBOL,
    atom_segment_ids,
    interface_data,
    ligand_blocks_from_arrays,
    pocket_blocks_from_arrays,
    summarize,
    to_batch,
)
from atomica_interface.scoring import load_encoder
from DiffSBDD.constants import ATOMICA_TO_DRUGLIKE_MAP, DRUGLIKE_ATOMS_DECODER

POCKET_SEGMENT = 0

# The original script built its own atomic-number table and rejected any complex
# containing an element outside it. That rejection is a real filter -- it is what
# excluded selenomethionine pockets, metal sites and so on from the training set --
# so it is preserved verbatim here rather than widened, keeping the dataset
# composition comparable to the run this replaces.
ALLOWED_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})

# The subset of the above that survives the ATOMICA -> druglike remapping, i.e. the
# atoms that actually become rows in the saved arrays. (Hydrogen is absent from the
# druglike decoder, so it is dropped.) Used to size-filter before encoding.
DRUGLIKE_ATOMIC_NUMBERS = np.array(
    sorted(
        z for z in ALLOWED_ATOMIC_NUMBERS
        if ATOMIC_NUMBER_TO_SYMBOL[z] in set(DRUGLIKE_ATOMS_DECODER)
    ),
    dtype=np.int64,
)


def clash_reason(ligand_coords, pocket_coords, threshold, complex_id=""):
    """Return a skip reason if any interatomic distance is below ``threshold``.

    Unchanged in behaviour from the original script; only refactored out of the
    featurization path so the two concerns can be read separately.
    """
    if threshold <= 0:
        return None

    with torch.no_grad():
        lig = torch.from_numpy(np.ascontiguousarray(ligand_coords)).float()
        pocket = torch.from_numpy(np.ascontiguousarray(pocket_coords)).float()

        if lig.shape[0] > 1:
            dists = torch.pdist(lig)
            if dists.numel() and dists.min() < threshold:
                return "intra_ligand_clash"

        if pocket.shape[0] > 1:
            try:
                dists = torch.pdist(pocket)
                if dists.numel() and dists.min() < threshold:
                    return "intra_pocket_clash"
            except RuntimeError as exc:
                if "out of memory" in str(exc) or "too large" in str(exc):
                    print(f"WARNING: skipping intra-pocket clash check for {complex_id} (too large).")
                else:
                    raise

        if lig.shape[0] and pocket.shape[0]:
            dists = torch.cdist(lig, pocket)
            if dists.numel() and dists.min() < threshold:
                return "inter_clash"

    return None


def process_complex(key, data, encoder, device, clash_threshold, trim_interface,
                    interface_dist, fragmentation, size_filter=None, collect_stats=False):
    """Encode one LMDB record into the dict `LigandPocketDatasetPT` consumes.

    Returns ``(payload, reason)``; ``payload`` is None when the complex is skipped.
    """
    vocab_size = len(DRUGLIKE_ATOMS_DECODER)
    complex_id = key.decode() if isinstance(key, bytes) else str(key)

    def as_numpy(value):
        return value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    try:
        pocket_coords = as_numpy(data["protein_pos"])
        pocket_elements = as_numpy(data["protein_element"])
        atom2residue = as_numpy(data["protein_atom2residue"])
        residue_aa = as_numpy(data["amino_acid"])
        ligand_coords = as_numpy(data["ligand_pos"])
        ligand_elements = as_numpy(data["ligand_element"])
        bond_index = as_numpy(data["ligand_bond_index"])
        bond_type = as_numpy(data["ligand_bond_type"])
    except KeyError as exc:
        return None, f"missing_field_{exc.args[0]}"
    except Exception as exc:
        print(f"Skipping {complex_id} due to parsing error: {exc}")
        return None, "parsing_error"

    if len(pocket_coords) == 0:
        return None, "no_pocket_atoms"
    if len(ligand_coords) == 0:
        return None, "no_ligand_atoms"

    reason = clash_reason(ligand_coords, pocket_coords, clash_threshold, complex_id)
    if reason is not None:
        return None, reason

    if not set(np.unique(pocket_elements).tolist()) <= ALLOWED_ATOMIC_NUMBERS:
        return None, "unknown_pocket_atom"
    if not set(np.unique(ligand_elements).tolist()) <= ALLOWED_ATOMIC_NUMBERS:
        return None, "unknown_ligand_atom"

    # --- Size filter, applied BEFORE encoding ---
    # The counts below are exact, not approximate: every element is known to be in
    # ALLOWED_ATOMIC_NUMBERS (so no atom is dropped when blocks are built), and the
    # only rows the druglike mask removes afterwards are non-druglike elements plus
    # the synthetic global atom. So this is the same predicate the caller re-checks
    # on the finished payload -- just evaluated before the expensive encoder call,
    # which is what stops us encoding ~half the candidates only to discard them.
    # Trimming would shrink the pocket further, so the shortcut is skipped then.
    if size_filter is not None and not trim_interface:
        n_pocket = int(np.isin(pocket_elements, DRUGLIKE_ATOMIC_NUMBERS).sum())
        n_ligand = int(np.isin(ligand_elements, DRUGLIKE_ATOMIC_NUMBERS).sum())
        if not (size_filter["min_lig"] <= n_ligand <= size_filter["max_lig"]):
            return None, "size_filter_ligand"
        if not (size_filter["min_pocket"] <= n_pocket <= size_filter["max_pocket"]):
            return None, "size_filter_pocket"

    # --- Build the two-segment ATOMICA record ---
    try:
        pocket_blocks, _ = pocket_blocks_from_arrays(
            pocket_coords, pocket_elements, atom2residue, residue_aa
        )
    except Exception as exc:
        print(f"ERROR building pocket blocks for {complex_id}: {exc}")
        return None, "pocket_block_error"

    try:
        ligand_blocks = ligand_blocks_from_arrays(
            ligand_coords, ligand_elements, bond_index, bond_type,
            fragmentation_method=fragmentation,
        )
    except Exception:
        # Fragmentation needs a chemically sane bond table; some docked records do
        # not sanitize. Skip rather than fall back to per-atom blocks, which would
        # silently mix two block vocabularies across the dataset.
        return None, "ligand_fragmentation_error"

    try:
        record = interface_data(
            pocket_blocks, ligand_blocks, dist_th=interface_dist, trim=trim_interface
        )
    except ValueError:
        # Raised when trimming leaves no contacts, i.e. the ligand is not in the
        # pocket. This is the pocket-proximity filter.
        return None, "no_interface_contacts"

    # --- Encode ---
    try:
        with torch.no_grad():
            output = encoder.infer(to_batch(record, device=device))
        unit_repr = output.unit_repr.cpu().numpy()
    except Exception as exc:
        print(f"ERROR encoding {complex_id}: {exc}")
        return None, "encoder_error"

    record_atom_types = np.asarray(record["A"], dtype=np.int64)
    record_coords = np.asarray(record["X"], dtype=np.float32)
    per_atom_segment = atom_segment_ids(record)

    # The encoder must return exactly one row per atom in the record, or every
    # downstream alignment is meaningless.
    assert unit_repr.shape[0] == len(record_atom_types) == len(per_atom_segment), (
        f"{complex_id}: encoder returned {unit_repr.shape[0]} rows for "
        f"{len(record_atom_types)} atoms"
    )

    # --- Pocket rows: coords, embeddings and one-hots from ONE mask ---
    # Selecting all three from the same boolean mask over the same atom ordering is
    # what keeps `pocket_atomica_embeddings` row-aligned with `pocket_coords`.
    # `blocks_to_data` prepends a synthetic global atom to each segment; its atom
    # type ('g') has no druglike counterpart, so the druglike mask drops it. That
    # matches the original script, where the same mask removed it.
    pocket_rows = per_atom_segment == POCKET_SEGMENT
    pocket_druglike = ATOMICA_TO_DRUGLIKE_MAP[record_atom_types[pocket_rows]]
    valid_pocket_mask = pocket_druglike != -1
    if valid_pocket_mask.sum() == 0:
        return None, "no_valid_pocket_atoms"

    filtered_pocket_coords = record_coords[pocket_rows][valid_pocket_mask]
    filtered_pocket_embeddings = unit_repr[pocket_rows][valid_pocket_mask]
    pocket_one_hot = np.eye(vocab_size, dtype=np.float32)[pocket_druglike[valid_pocket_mask]]

    assert (
        filtered_pocket_coords.shape[0]
        == filtered_pocket_embeddings.shape[0]
        == pocket_one_hot.shape[0]
    ), f"{complex_id}: pocket coord/embedding/one-hot rows disagree"

    # --- Ligand rows: coordinates only, straight from the record ---
    # No embeddings are stored for the ligand, so its original LMDB atom order is
    # kept (the same convention the training data used before).
    ligand_atomica_types = np.array(
        [_element_to_atomica_index(int(z)) for z in ligand_elements], dtype=np.int64
    )
    ligand_druglike = ATOMICA_TO_DRUGLIKE_MAP[ligand_atomica_types]
    valid_ligand_mask = ligand_druglike != -1
    if valid_ligand_mask.sum() == 0:
        return None, "no_valid_ligand_atoms"

    filtered_ligand_coords = np.asarray(ligand_coords, dtype=np.float32)[valid_ligand_mask]
    ligand_one_hot = np.eye(vocab_size, dtype=np.float32)[ligand_druglike[valid_ligand_mask]]

    assert filtered_ligand_coords.shape[0] == ligand_one_hot.shape[0], (
        f"{complex_id}: ligand coord/one-hot rows disagree"
    )

    payload = {
        "lig_coords": filtered_ligand_coords.astype(np.float32),
        "lig_one_hot": ligand_one_hot,
        "pocket_coords": filtered_pocket_coords.astype(np.float32),
        "pocket_atomica_embeddings": filtered_pocket_embeddings.astype(np.float32),
        "pocket_one_hot": pocket_one_hot.astype(np.float32),
        "name": data.get("ligand_filename", complex_id),
    }

    if collect_stats:
        payload["_stats"] = summarize(record)

    return payload, "success"


def _build_atomic_number_to_atomica_index():
    """Atomic number -> ATOMICA atom-vocabulary index, built once at import."""
    from ATOMICA.data.pdb_utils import VOCAB

    from atomica_interface.featurize import ATOMIC_NUMBER_TO_SYMBOL

    return {z: VOCAB.atom_to_idx(sym) for z, sym in ATOMIC_NUMBER_TO_SYMBOL.items()}


_ATOMIC_NUMBER_TO_ATOMICA_INDEX = _build_atomic_number_to_atomica_index()


def _element_to_atomica_index(atomic_number):
    """Map an atomic number to ATOMICA's atom-vocabulary index (mask if unknown)."""
    from ATOMICA.data.pdb_utils import VOCAB

    return _ATOMIC_NUMBER_TO_ATOMICA_INDEX.get(atomic_number, VOCAB.get_atom_mask_idx())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lmdb", default="data/crossdocked_pocket10_processed.lmdb")
    parser.add_argument("--split", default="data/crossdocked_split.pt",
                        help="standard CrossDocked split (indices into LMDB cursor order)")
    parser.add_argument("--expert_split", default="expert_split.pt")
    parser.add_argument("--output", default="data/processed_expert_atomica")
    parser.add_argument("--model_config", default="ATOMICA/pretrain/pretrain_model_config.json")
    parser.add_argument("--model_weights", default="ATOMICA/pretrain/pretrain_model_weights.pt")

    parser.add_argument("--limit", "--max_complexes", dest="limit", type=int, default=None,
                        help="stop after writing this many complexes (for smoke runs)")
    parser.add_argument("--scan_limit", type=int, default=None,
                        help="stop after scanning this many LMDB entries")

    parser.add_argument("--clash_threshold", type=float, default=0.5)
    parser.add_argument("--min_lig_atoms", type=int, default=10)
    parser.add_argument("--max_lig_atoms", type=int, default=40)
    parser.add_argument("--min_pocket_atoms", type=int, default=350)
    parser.add_argument("--max_pocket_atoms", type=int, default=800)

    parser.add_argument("--trim_interface", action="store_true",
                        help="keep only blocks within --interface_dist of the other entity. "
                             "Off by default: the pocket10 set is already a 10 A pocket, and "
                             "trimming would change the pocket the diffusion model sees.")
    parser.add_argument("--interface_dist", type=float, default=8.0)
    parser.add_argument("--fragmentation", default="PS_300",
                        help="ligand fragmentation scheme; must match the checkpoint's "
                             "fragmentation_method")
    parser.add_argument("--target_val_size", type=int, default=1000)
    parser.add_argument("--target_test_size", type=int, default=1000)
    parser.add_argument("--stats_every", type=int, default=0,
                        help="print block/segment structure for every Nth written complex")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_base = Path(args.output)
    dirs = {name: output_base / name for name in ("train", "val", "test")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Loading ATOMICA encoder...")
    try:
        encoder = load_encoder(args.model_config, args.model_weights, device=str(device))
    except Exception as exc:
        print(f"FATAL: failed to load ATOMICA encoder: {exc}")
        traceback.print_exc()
        return
    print(f"Loaded {type(encoder).__name__} (denoising heads disabled).")

    print(f"Loading expert split from {args.expert_split}...")
    expert_split = torch.load(args.expert_split)
    if isinstance(expert_split, dict) and "train" in expert_split:
        expert_filenames = set(expert_split["train"])
    elif isinstance(expert_split, list):
        expert_filenames = set(expert_split)
    else:
        print("Unknown expert split format; expected dict with 'train' or a list.")
        return
    print(f"Loaded {len(expert_filenames)} expert filenames.")

    print(f"Loading standard split from {args.split}...")
    std_split = torch.load(args.split)
    std_train = set(std_split["train"])
    std_holdout = set(std_split["val"]) | set(std_split["test"])
    print(f"Standard split: train={len(std_train)}, val+test={len(std_holdout)}")

    print(f"Opening LMDB: {args.lmdb}")
    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    total_entries = env.stat()["entries"]

    size_filter = {
        "min_lig": args.min_lig_atoms, "max_lig": args.max_lig_atoms,
        "min_pocket": args.min_pocket_atoms, "max_pocket": args.max_pocket_atoms,
    }

    counts = {"train": 0, "val": 0, "test": 0}
    skipped = {}
    total_success = 0
    encode_seconds = 0.0
    started = time.time()

    def note(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    with env.begin() as txn:
        cursor = txn.cursor()
        # `idx` is the position in LMDB cursor order, which is what both
        # `expert_split.pt`'s `train_indices` and the standard split index -- LMDB
        # orders keys lexicographically ('0', '1', '10', '100', ...), so the numeric
        # key is NOT the same thing. Membership below is matched on
        # `ligand_filename` anyway, which is order-independent.
        for idx, (key, value) in tqdm(enumerate(cursor), total=total_entries, desc="Processing"):
            if args.scan_limit is not None and idx >= args.scan_limit:
                break

            data = pickle.loads(value)
            if "ligand_filename" not in data:
                note("no_ligand_filename")
                continue
            if data["ligand_filename"] not in expert_filenames:
                note("not_expert")
                continue

            if idx in std_train:
                bucket = "train"
            elif idx in std_holdout:
                # Held out upstream: skip to avoid leaking it into our train set.
                note("in_standard_val_test")
                continue
            elif counts["val"] < args.target_val_size:
                bucket = "val"
            elif counts["test"] < args.target_test_size:
                bucket = "test"
            else:
                note("excess_candidate")
                continue

            want_stats = bool(args.stats_every) and total_success % args.stats_every == 0
            t0 = time.time()
            try:
                payload, reason = process_complex(
                    key, data, encoder, device, args.clash_threshold,
                    args.trim_interface, args.interface_dist, args.fragmentation,
                    size_filter=size_filter, collect_stats=want_stats,
                )
            except Exception as exc:
                print(f"\n--- UNHANDLED ERROR on key {key} ---\n{exc}")
                traceback.print_exc()
                payload, reason = None, "unhandled_exception"
            encode_seconds += time.time() - t0

            if payload is None:
                note(reason)
                continue

            n_lig = payload["lig_coords"].shape[0]
            n_pocket = payload["pocket_coords"].shape[0]
            if not (args.min_lig_atoms <= n_lig <= args.max_lig_atoms):
                note("size_filter_ligand")
                continue
            if not (args.min_pocket_atoms <= n_pocket <= args.max_pocket_atoms):
                note("size_filter_pocket")
                continue

            stats = payload.pop("_stats", None)
            if stats is not None:
                print(f"\n  [{payload['name']}] {stats}")

            counts[bucket] += 1
            tensor_data = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in payload.items()
            }
            torch.save(tensor_data, dirs[bucket] / f"complex_{counts[bucket]:06d}.pt")
            total_success += 1

            if args.limit is not None and total_success >= args.limit:
                print(f"\nReached --limit of {args.limit}")
                break

    env.close()
    elapsed = time.time() - started

    print("\n" + "=" * 60)
    print("Processing complete")
    print("=" * 60)
    print(f"Written: {total_success}  (train={counts['train']}, val={counts['val']}, test={counts['test']})")
    print(f"Wall time: {elapsed:.1f}s; in process_complex: {encode_seconds:.1f}s")
    if total_success:
        print(f"Per written complex: {encode_seconds / total_success:.3f}s of featurize+encode")
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {reason}: {n}")


if __name__ == "__main__":
    main()
