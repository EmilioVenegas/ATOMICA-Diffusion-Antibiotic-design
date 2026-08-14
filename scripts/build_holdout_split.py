"""Split the official CrossDocked holdout into target-disjoint val and test sets.

`data/crossdocked_split.pt` has two defects that have to be fixed together
(docs/experiment-plan.md, "Split integrity"):

1. Its `val` and `test` entries are the *identical* list of 100 indices, so there
   is no independent test set at all.
2. `scripts/process_expert_atomica.py` never used that holdout. It skipped every
   index in it and filled its own val/test buckets from complexes in *neither*
   official split -- a leftover pool of 65,719 entries spanning 1,770 targets, of
   which 1,327 targets also appear in train. Any number computed on those
   directories is substantially within-target.

This script takes the 100 official holdout indices, groups them by target, and
partitions the *targets* into disjoint val and test halves. Splitting by target
rather than by complex is the whole point: CrossDocked holds many docked
complexes per target, so a random per-complex split would put the same protein on
both sides, which is defect 2 in miniature.

The index -> target mapping needs `ligand_filename`, which lives inside each
pickled LMDB record, and the split indexes the LMDB in *cursor* order (LMDB sorts
keys lexicographically -- '0', '1', '10', '100' -- so the numeric key is not the
position). The full scan is therefore unavoidable; its result is cached to
`--manifest` so nothing downstream has to repeat it.

The official holdout names only 100 complexes, which is thin for a validation
set. Those 100 sit on 93 targets, and the LMDB holds 8,330 entries on those same
targets -- none of which appear in the official train index list. Since the split
is target-disjoint, every one of those 8,330 is as clean as the official 100, so
this script also emits an *expanded* index list (all entries on the holdout
targets, optionally capped per target so a single 2,567-entry target cannot
dominate). Both lists are written; the consumer picks.

Writes `data/holdout_target_split.pt`:

    {'val_targets', 'test_targets'}   sorted lists of target directory names
    {'val_indices', 'test_indices'}   the official holdout indices only
    {'val_indices_expanded', 'test_indices_expanded'}
                                      every LMDB entry on those targets, capped
    {'val_filenames', 'test_filenames'}
    'provenance'                      how it was built, for the record
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import lmdb
import torch
from tqdm import tqdm


def target_of(ligand_filename):
    """Target identity is the top-level directory of the ligand path.

    CrossDocked lays out `<POCKET_DIR>/<rec>_<lig>_..._docked.sdf`, one directory
    per target, holding every docked pose for it.
    """
    return str(ligand_filename).split("/")[0]


def build_manifest(lmdb_path, manifest_path):
    """Cursor index -> ligand_filename for every LMDB record, cached as JSON."""
    if manifest_path.exists():
        print(f"Reusing cached manifest: {manifest_path}")
        with open(manifest_path) as fh:
            return json.load(fh)

    print(f"Scanning {lmdb_path} in cursor order (this is the slow part)...")
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    total = env.stat()["entries"]
    manifest = []
    missing = 0
    with env.begin() as txn:
        for _, value in tqdm(txn.cursor(), total=total, desc="Scanning"):
            record = pickle.loads(value)
            name = record.get("ligand_filename")
            if name is None:
                missing += 1
            manifest.append(name)
    env.close()

    if missing:
        print(f"WARNING: {missing} records carry no ligand_filename (stored as null).")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh)
    print(f"Wrote manifest for {len(manifest)} entries to {manifest_path}")
    return manifest


def partition_targets(target_to_indices):
    """Greedily split targets into two groups of near-equal complex count.

    Deterministic, no RNG: targets are ordered by descending complex count with
    the name as tiebreak, then each is assigned to whichever side currently holds
    fewer complexes. With 93 targets holding one or two complexes each this is
    close to an even cut, and it never depends on a seed nobody recorded.
    """
    ordered = sorted(target_to_indices.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    val_targets, test_targets = [], []
    n_val = n_test = 0
    for target, indices in ordered:
        if n_val <= n_test:
            val_targets.append(target)
            n_val += len(indices)
        else:
            test_targets.append(target)
            n_test += len(indices)
    return sorted(val_targets), sorted(test_targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lmdb", default="data/crossdocked_pocket10_processed.lmdb")
    parser.add_argument("--split", default="data/crossdocked_split.pt")
    parser.add_argument("--manifest", default="data/lmdb_index_manifest.json",
                        help="cache of cursor index -> ligand_filename")
    parser.add_argument("--output", default="data/holdout_target_split.pt")
    parser.add_argument("--max_per_target", type=int, default=30,
                        help="cap on complexes per target in the EXPANDED lists. "
                             "Holdout target sizes run 1 to 2,567; without a cap "
                             "two targets would supply most of the set and the "
                             "per-pocket analysis would be dominated by them.")
    args = parser.parse_args()

    manifest = build_manifest(Path(args.lmdb), Path(args.manifest))

    split = torch.load(args.split)
    train_indices = set(split["train"])
    holdout_indices = sorted(set(split["val"]) | set(split["test"]))
    print(f"\nOfficial split: train={len(train_indices)}, "
          f"val={len(split['val'])}, test={len(split['test'])}, "
          f"val==test: {list(split['val']) == list(split['test'])}")
    print(f"Distinct holdout indices: {len(holdout_indices)}")

    def targets_for(indices):
        mapping = defaultdict(list)
        for idx in indices:
            name = manifest[idx]
            if name is None:
                continue
            mapping[target_of(name)].append(idx)
        return mapping

    # Every LMDB entry grouped by target, used to expand the holdout beyond the
    # 100 officially-named complexes.
    all_by_target = targets_for(range(len(manifest)))

    train_targets = set(targets_for(train_indices))
    holdout_by_target = targets_for(holdout_indices)
    print(f"Train targets: {len(train_targets)}")
    print(f"Holdout targets: {len(holdout_by_target)}")

    # The official split is target-aware; if that ever stops being true, every
    # number computed downstream is within-target and worthless. Fail loudly.
    contaminated = train_targets & set(holdout_by_target)
    assert not contaminated, (
        f"{len(contaminated)} targets appear in BOTH train and the official "
        f"holdout, e.g. {sorted(contaminated)[:5]}"
    )
    print("Verified: zero targets shared between train and the official holdout.")

    val_targets, test_targets = partition_targets(holdout_by_target)
    val_indices = sorted(i for t in val_targets for i in holdout_by_target[t])
    test_indices = sorted(i for t in test_targets for i in holdout_by_target[t])

    assert not (set(val_targets) & set(test_targets)), "val/test targets overlap"
    assert not (set(val_indices) & set(test_indices)), "val/test indices overlap"
    assert len(val_indices) + len(test_indices) == sum(
        len(v) for v in holdout_by_target.values()
    ), "partition lost complexes"

    # --- Expanded lists: every LMDB entry on a holdout target ---
    # Safe by construction. The targets are disjoint from train's targets (just
    # asserted), so nothing here can share a protein with training data. The cap
    # is taken deterministically -- lowest cursor indices first -- so reruns
    # agree.
    # The officially-named complexes are taken first, so the expanded list is
    # always a superset of the strict one and the cap only ever trims the
    # complexes we added. That keeps the strict holdout recoverable from an
    # expanded build by filtering on ligand_filename.
    official = set(holdout_indices)

    def expand(targets):
        out = []
        for t in sorted(targets):
            entries = sorted(all_by_target[t])
            ranked = ([i for i in entries if i in official]
                      + [i for i in entries if i not in official])
            out.extend(ranked[: args.max_per_target])
        return sorted(out)

    val_expanded = expand(val_targets)
    test_expanded = expand(test_targets)

    assert not (set(val_expanded) & set(test_expanded)), "expanded val/test overlap"
    assert not (set(val_expanded) & train_indices), "expanded val hits train indices"
    assert not (set(test_expanded) & train_indices), "expanded test hits train indices"
    assert set(val_indices) <= set(val_expanded), "expanded val lost official complexes"
    assert set(test_indices) <= set(test_expanded), "expanded test lost official complexes"

    payload = {
        "val_targets": val_targets,
        "test_targets": test_targets,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "val_indices_expanded": val_expanded,
        "test_indices_expanded": test_expanded,
        "val_filenames": [manifest[i] for i in val_indices],
        "test_filenames": [manifest[i] for i in test_indices],
        "provenance": {
            "lmdb": args.lmdb,
            "split": args.split,
            "method": "official val|test holdout indices, targets partitioned "
                      "greedily by descending complex count into two disjoint "
                      "groups of near-equal size",
            "expanded_method": "every LMDB entry whose target is in the group, "
                               f"capped at {args.max_per_target} per target",
            "max_per_target": args.max_per_target,
            "n_holdout_indices": len(holdout_indices),
            "n_holdout_targets": len(holdout_by_target),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print("\n" + "=" * 60)
    print(f"{'':6s} {'targets':>8s} {'official':>9s} {'expanded':>9s}")
    print(f"{'val':6s} {len(val_targets):8d} {len(val_indices):9d} {len(val_expanded):9d}")
    print(f"{'test':6s} {len(test_targets):8d} {len(test_indices):9d} {len(test_expanded):9d}")
    print("=" * 60)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
