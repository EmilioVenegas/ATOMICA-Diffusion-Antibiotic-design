#!/usr/bin/env python3
# downselect_embeddings.py — minimal, works with your dict format
# - Loads .pt files, a dict where key 'pocket_atomica_embeddings' holds [N_atoms, 32] tensor
# - Mean-pools to [32]
# - Farthest-first (k-center) on cosine distance to pick <budget> items
# - Writes: subset_indices.pt and subset_paths.txt

import sys
import glob
import torch

KEY = "pocket_atomica_embeddings"

def safe_load(path):
    # Use safe loader if supported; fall back for older torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def main():
    if len(sys.argv) != 3:
        print('Usage: python downselect_embeddings.py "<glob/of/*.pt>" <budget_int>')
        raise SystemExit(1)
    pattern, budget = sys.argv[1], int(sys.argv[2])

    # Glob (supports ** with recursive=True)
    paths = sorted(glob.glob(pattern, recursive=True))
    print(f"[info] matched_files={len(paths)}")
    if not paths:
        print("[error] No files matched. If there are subfolders, use **/*.pt")
        raise SystemExit(1)

    # Load -> mean-pool
    means, good_idx = [], []
    skipped_load = skipped_key = skipped_bad = 0
    for i, p in enumerate(paths, 1):
        try:
            obj = safe_load(p)
        except Exception:
            print("what the heck")
            skipped_load += 1
            continue
        if not isinstance(obj, dict) or KEY not in obj:
            skipped_key += 1
            continue
        Z = obj[KEY]
        if not isinstance(Z, torch.Tensor) or Z.ndim != 2 or Z.numel() == 0:
            skipped_bad += 1
            continue
        means.append(Z.mean(dim=0).to(torch.float32).reshape(1, -1))
        good_idx.append(i - 1)
        if i % 1000 == 0:
            print(f"[info] scanned {i}/{len(paths)} loaded={len(means)} skipped={skipped_load+skipped_key+skipped_bad}")

    if not means:
        print(f"[error] Loaded zero embeddings. "
              f"failed_load={skipped_load}, missing_key={skipped_key}, bad_tensor={skipped_bad}")
        raise SystemExit(1)

    E = torch.cat(means, dim=0)  # [M, 32]
    # Normalize for cosine space
    E = torch.nn.functional.normalize(E, dim=1)

    # Farthest-first (k-center) in cosine distance (1 - cos sim)
    M = E.size(0)
    budget = min(budget, M)
    torch.manual_seed(0)
    i0 = int(torch.randint(0, M, ()).item())
    chosen_local = [i0]
    dmin = 1 - (E @ E[i0])  # [M]
    for _ in range(1, budget):
        j = int(torch.argmax(dmin).item())
        chosen_local.append(j)
        dmin = torch.minimum(dmin, 1 - (E @ E[j]))

    # Map back to original indices
    chosen_global = sorted({good_idx[j] for j in chosen_local})

    # Save outputs
    torch.save(torch.tensor(chosen_global, dtype=torch.long), "subset_indices.pt")
    with open("subset_paths.txt", "w") as f:
        for idx in chosen_global:
            f.write(paths[idx] + "\n")

    print(f"[done] kept={len(chosen_global)} of matched={len(paths)} "
          f"(loaded={M}, skipped_load={skipped_load}, missing_key={skipped_key}, bad_tensor={skipped_bad})")
    print("Wrote subset_indices.pt and subset_paths.txt")

if __name__ == "__main__":
    main()
