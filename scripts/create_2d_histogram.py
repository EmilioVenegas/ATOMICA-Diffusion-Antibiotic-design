# File: create_2d_histogram.py (Corrected Version)

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def create_histogram(data_dir, max_lig_nodes=100, max_pocket_nodes=500):
    """
    Creates a 2D histogram of (ligand_size, pocket_size) from processed .pt files.
    """
    print(f"Creating 2D histogram with max sizes: Ligand={max_lig_nodes}, Pocket={max_pocket_nodes}")
    
    # Initialize a 2D numpy array to store the counts
    # Axis 0: Ligand size, Axis 1: Pocket size
    histogram = np.zeros((max_lig_nodes, max_pocket_nodes), dtype=int)

    # Find all .pt files in the train, val, and test directories
    processed_dir = Path(data_dir)
    pt_files = list(processed_dir.glob('train/*.pt')) + \
               list(processed_dir.glob('val/*.pt')) + \
               list(processed_dir.glob('test/*.pt'))

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in subdirectories of {data_dir}. Please check the path.")

    print(f"Found {len(pt_files)} files to process.")
    
    skipped_files = 0
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

            # Clamp the values to the max size of our histogram
            lig_idx = min(lig_nodes - 1, max_lig_nodes - 1)
            pocket_idx = min(pocket_nodes - 1, max_pocket_nodes - 1)
            
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a 2D size histogram for the dataset.")
    parser.add_argument('--datadir', type=str, required=True,
                        help="Path to the PROCESSED and FILTERED data directory (e.g., 'data/processed_atomica_filtered/')")
    args = parser.parse_args()
    
    create_histogram(args.datadir)