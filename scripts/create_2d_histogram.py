# File: create_2d_histogram.py (Corrected Version)

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Bin counts must cover the preprocessing size filter exactly:
# process_expert_atomica.py keeps 10-40 ligand atoms and 350-800 pocket atoms,
# so the axes need 41 and 801 bins. The previous defaults here were (100, 500),
# which silently clamped 72% of complexes into the last pocket bin -- destroying
# the pocket axis this histogram exists to represent -- and set
# `max_num_nodes = len(histogram) - 1` to 99 for ligands that cannot exceed 40.
DEFAULT_MAX_LIG_NODES = 41
DEFAULT_MAX_POCKET_NODES = 801


def create_histogram(data_dir, max_lig_nodes=DEFAULT_MAX_LIG_NODES,
                     max_pocket_nodes=DEFAULT_MAX_POCKET_NODES,
                     splits=('train',)):
    """
    Creates a 2D histogram of (ligand_size, pocket_size) from processed .pt files.

    Built from `train` only by default. `DiffSBDD/train.py` loads this file and
    it is what conditions the size of sampled ligands, so including val or test
    would let the held-out sets inform the generator's size prior -- a small
    leak, but a free one to avoid. The earlier version globbed all three.
    """
    print(f"Creating 2D histogram with max sizes: Ligand={max_lig_nodes}, Pocket={max_pocket_nodes}")
    print(f"Splits: {', '.join(splits)}")

    # Initialize a 2D numpy array to store the counts
    # Axis 0: Ligand size, Axis 1: Pocket size
    histogram = np.zeros((max_lig_nodes, max_pocket_nodes), dtype=int)

    processed_dir = Path(data_dir)
    pt_files = [f for split in splits for f in processed_dir.glob(f'{split}/*.pt')]

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in subdirectories of {data_dir}. Please check the path.")

    print(f"Found {len(pt_files)} files to process.")
    
    skipped_files = 0
    clamped_lig = 0
    clamped_pocket = 0
    for file_path in tqdm(pt_files, desc="Processing files"):
        data = torch.load(file_path)
        
        # --- START: CORRECTED LOGIC ---
        lig_coords = data.get('lig_coords')
        pocket_coords = data.get('pocket_coords')

        # Check if the tensors exist and are not empty
        if lig_coords is not None and pocket_coords is not None and lig_coords.shape[0] > 0 and pocket_coords.shape[0] > 0:
            lig_nodes = lig_coords.shape[0]
            pocket_nodes = pocket_coords.shape[0]
            # --- END: CORRECTED LOGIC ---

            # Clamp the values to the max size of our histogram. Clamping is a
            # last resort, not normal operation: every clamped complex is
            # recorded at the wrong size, so the counts are reported below and
            # the axes should be widened rather than the warning ignored.
            lig_idx = min(lig_nodes - 1, max_lig_nodes - 1)
            pocket_idx = min(pocket_nodes - 1, max_pocket_nodes - 1)
            clamped_lig += lig_nodes - 1 > max_lig_nodes - 1
            clamped_pocket += pocket_nodes - 1 > max_pocket_nodes - 1
            
            if lig_idx >= 0 and pocket_idx >= 0:
                histogram[lig_idx, pocket_idx] += 1
        else:
            skipped_files += 1

    # Save the histogram to the root of the data directory
    # IMPORTANT: The histogram must be in the datadir itself, not a sub-folder.
    save_path = processed_dir / 'size_distribution.npy'
    np.save(save_path, histogram)
    
    print("\n--- Histogram Creation Complete ---")
    print(f"Successfully saved 2D histogram to: {save_path}")
    print(f"Histogram shape: {histogram.shape}")
    print(f"Total valid pairs counted: {np.sum(histogram)}")
    if skipped_files > 0:
        print(f"Warning: Skipped {skipped_files} files due to missing or empty coordinate tensors.")
    total = max(int(np.sum(histogram)), 1)
    if clamped_lig or clamped_pocket:
        print(f"WARNING: clamped {clamped_lig} ligand sizes "
              f"({100 * clamped_lig / total:.1f}%) and {clamped_pocket} pocket "
              f"sizes ({100 * clamped_pocket / total:.1f}%) into the final bin. "
              f"Those complexes are recorded at the wrong size -- widen "
              f"--max_lig_nodes / --max_pocket_nodes.")
    else:
        print("No complexes were clamped; both axes cover the data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a 2D size histogram for the dataset.")
    parser.add_argument('--datadir', type=str, required=True,
                        help="Path to the PROCESSED and FILTERED data directory (e.g., 'data/processed_atomica_filtered/')")
    parser.add_argument('--max_lig_nodes', type=int, default=DEFAULT_MAX_LIG_NODES)
    parser.add_argument('--max_pocket_nodes', type=int, default=DEFAULT_MAX_POCKET_NODES)
    parser.add_argument('--splits', type=str, default='train',
                        help="comma-separated splits to count. Defaults to train "
                             "alone: this histogram conditions sampled ligand "
                             "size, so val/test should not inform it.")
    args = parser.parse_args()

    create_histogram(args.datadir,
                     max_lig_nodes=args.max_lig_nodes,
                     max_pocket_nodes=args.max_pocket_nodes,
                     splits=tuple(s.strip() for s in args.splits.split(',') if s.strip()))