import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import sys
import argparse

try:
    from DiffSBDD.constants import dataset_params
except ImportError:
    print("Error: Could not import DiffSBDD.constants.")
    print("Please run this script from the 'ATOMICA-Diffusion-Antibiotic-design-training-optimization' directory.")
    sys.exit(1)
# ---------------------

def create_histograms(data_dir, out_path, dataset_name):
    data_dir = Path(data_dir)
    out_path = Path(out_path)

    if not data_dir.is_dir():
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)

    files = sorted(data_dir.glob('*.pt'))
    if not files:
        print(f"Error: No .pt files found in {data_dir}")
        sys.exit(1)

    if dataset_name not in dataset_params:
        print(f"Error: Dataset '{dataset_name}' not found in constants.py")
        print(f"Available datasets: {list(dataset_params.keys())}")
        sys.exit(1)

    # Get definitions from constants.py
    params = dataset_params[dataset_name]
    num_lig_atom_types = len(params['atom_decoder'])
    num_pocket_node_types = len(params['aa_decoder'])

    print(f"Using {num_lig_atom_types} ligand atom types and {num_pocket_node_types} pocket node types for '{dataset_name}'.")

    lig_sizes = []
    
    # Initialize histogram counters
    atom_hist = torch.zeros(num_lig_atom_types, dtype=torch.float64)
    aa_hist = torch.zeros(num_pocket_node_types, dtype=torch.float64)

    for f in tqdm(files, desc="Processing data files"):
        try:
            data = torch.load(f)
            
            # 1. Get ligand size
            lig_sizes.append(len(data['lig_coords']))

            # 2. Get ligand atom type counts
            if 'lig_one_hot' in data:
                if data['lig_one_hot'].shape[1] == num_lig_atom_types:
                    atom_hist += data['lig_one_hot'].sum(dim=0)
                else:
                    print(f"Warning: Skipping {f}, lig_one_hot shape mismatch.")
            
            # 3. Get pocket node type counts
            if 'pocket_one_hot' in data:
                if data['pocket_one_hot'].shape[1] == num_pocket_node_types:
                    aa_hist += data['pocket_one_hot'].sum(dim=0)
                else:
                    print(f"Warning: Skipping {f}, pocket_one_hot shape mismatch.")

        except Exception as e:
            print(f"Warning: Could not load file {f}. Skipping. Error: {e}")

    # --- Create 1D Ligand Size Histogram (as you had before) ---
    max_size = max(lig_sizes)
    histogram = np.zeros(max_size + 1, dtype=int)
    for size in lig_sizes:
        histogram[size] += 1
    # ---------------------------------

    # Save the 1D histogram
    np.save(out_path, histogram)
    
    print(f"\nSuccessfully saved 1D ligand size histogram to {out_path}")
    print(f"Total complexes: {len(lig_sizes)}")
    print(f"Max ligand size: {max_size}")
    print(f"Histogram shape: {histogram.shape}")

    # --- Print the histograms for constants.py ---
    print("\n--- Copy and paste these into DiffSBDD/constants.py ---")
    
    # Ligand atom histogram
    print("\n# Ligand atom probability (druglike_atom_hist_array)")
    lig_atom_prob = atom_hist / (atom_hist.sum() + 1e-8) # Add epsilon for safety
    print(f"lig_atom_hist_counts = {atom_hist.tolist()}")
    print(f"druglike_atom_hist_array = np.array({np.round(lig_atom_prob.numpy(), 8).tolist()})")
    
    # Pocket node histogram
    print("\n# Pocket node probability (atomica_aa_hist)")
    aa_prob = aa_hist / (aa_hist.sum() + 1e-8) # Add epsilon for safety
    print(f"pocket_node_hist_counts = {aa_hist.tolist()}")
    print(f"atomica_aa_hist = np.array({np.round(aa_prob.numpy(), 8).tolist()})")
    
    print("\n--- End of histogram data ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create histograms for DiffSBDD.")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to the directory containing processed .pt files (e.g., data/processed_atomica/train)")
    parser.add_argument('--out_path', type=str, required=True,
                        help="Full path to save the output .npy file (e.g., data/processed_atomica/size_distribution.npy)")
    parser.add_argument('--dataset', type=str, default='atomica_PL',
                        help="Name of the dataset (e.g., 'atomica_PL') as defined in constants.py")
    args = parser.parse_args()

   
    print(f"Running histogram creation...")
    print(f"Data directory: {args.data_path}")
    print(f"Output file: {args.out_path}")
    print(f"Dataset: {args.dataset}")
    
    create_histograms(args.data_path, args.out_path, args.dataset)