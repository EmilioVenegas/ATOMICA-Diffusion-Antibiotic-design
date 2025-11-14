import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

def filter_dataset(source_dir, dest_dir, 
                   min_lig_atoms, max_lig_atoms, 
                   min_pocket_atoms, max_pocket_atoms):
    """
    Copies .pt files from source to destination only if they meet size criteria.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    print(f"Filtering dataset from '{source_path}' to '{dest_path}'")
    print(f"Criteria: {min_lig_atoms} <= Ligand atoms <= {max_lig_atoms}, "
          f"{min_pocket_atoms} <= Pocket atoms <= {max_pocket_atoms}")

    for split in ['train', 'val', 'test']:
        source_split_path = source_path / split
        dest_split_path = dest_path / split
        
        if not source_split_path.is_dir():
            print(f"Warning: Source directory '{source_split_path}' not found. Skipping.")
            continue
            
        dest_split_path.mkdir(parents=True, exist_ok=True)
        
        files_to_process = list(source_split_path.glob('*.pt'))
        
        if not files_to_process:
            print(f"No .pt files found in '{source_split_path}'.")
            continue

        kept_count = 0
        for file_path in tqdm(files_to_process, desc=f"Filtering {split} split"):
            data = torch.load(file_path)
            
            lig_nodes = data['lig_coords'].shape[0]
            pocket_nodes = data['pocket_coords'].shape[0]

            lig_criteria_met = (min_lig_atoms <= lig_nodes <= max_lig_atoms)
            pocket_criteria_met = (min_pocket_atoms <= pocket_nodes <= max_pocket_atoms)

            if lig_criteria_met and pocket_criteria_met:
                shutil.copy(file_path, dest_split_path)
                kept_count += 1
        
        print(f"Finished '{split}': Kept {kept_count} / {len(files_to_process)} files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter a processed dataset based on molecule and pocket size.")
    parser.add_argument('--source_dir', type=str, required=True,
                        help="Path to the original processed data directory (e.g., 'data/processed_atomica_centered/')")
    parser.add_argument('--dest_dir', type=str, required=True,
                        help="Path to the new directory for the filtered dataset (e.g., 'data/processed_atomica_filtered/')")
    
    parser.add_argument('--min_lig', type=int, default=10, help="Minimum number of ligand atoms.")
    parser.add_argument('--max_lig', type=int, default=100, help="Maximum number of ligand atoms.")
    parser.add_argument('--min_pocket', type=int, default=50, help="Minimum number of pocket atoms.")
    parser.add_argument('--max_pocket', type=int, default=500, help="Maximum number of pocket atoms.")
    
    args = parser.parse_args()
    
    filter_dataset(args.source_dir, args.dest_dir, 
                   args.min_lig, args.max_lig, 
                   args.min_pocket, args.max_pocket)

# python filter_processed_data.py --source_dir data/processed_atomica_pocketcoords/ --dest_dir data/processed_atomica_filtered_pocketcoords/