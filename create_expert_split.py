import lmdb
import pickle
import torch
import os
import numpy as np
from tqdm import tqdm

def create_expert_split(lmdb_path, affinity_pkl_path, output_path, threshold=-8.5):
    print(f"Loading affinity info from {affinity_pkl_path}...")
    with open(affinity_pkl_path, 'rb') as f:
        affinity_data = pickle.load(f)
    
    print(f"Opening LMDB: {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    
    expert_indices = []
    expert_filenames = []
    all_filenames = []
    
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        total_entries = env.stat()['entries']
        
        for i, (key, value) in tqdm(enumerate(cursor), total=total_entries, desc="Filtering"):
            try:
                data = pickle.loads(value)
                
                if 'ligand_filename' in data:
                    lig_filename = data['ligand_filename']
                    all_filenames.append(lig_filename)
                    
                    # Remove extension to match affinity_info keys
                    key_candidate = os.path.splitext(lig_filename)[0]
                    
                    if key_candidate in affinity_data:
                        score = affinity_data[key_candidate].get('vina')
                        if score is not None and score < threshold:
                            expert_indices.append(i)
                            expert_filenames.append(lig_filename)
                            
            except Exception as e:
                pass

    # Calculate statistics
    original_count = len(all_filenames)
    expert_count = len(expert_filenames)
    reduction_percentage = (1 - expert_count / original_count) * 100
    
    print("\n" + "="*40)
    print("FILTERING RESULTS")
    print("="*40)
    print(f"Original Dataset Size: {original_count}")
    print(f"Expert Dataset Size:   {expert_count}")
    print(f"Reduction:             {reduction_percentage:.2f}% removed")
    print(f"Kept:                  {expert_count / original_count * 100:.2f}%")
    print(f"Threshold used:        < {threshold} kcal/mol")
    
    # Save to .pt file
    # Saving as a dictionary with 'train', 'val', 'test' keys is standard
    # We'll put everything in 'train' for now, or just save the list if preferred.
    # The user asked for "filtered list of filenames/indices".
    # I'll save a dictionary with both for maximum utility.
    
    output_data = {
        'train': expert_filenames, # Use filenames as they are more robust than indices
        'train_indices': expert_indices,
        'val': [],
        'test': []
    }
    
    print(f"\nSaving split to {output_path}...")
    torch.save(output_data, output_path)
    print("Done.")

if __name__ == "__main__":
    create_expert_split(
        lmdb_path='data/crossdocked_pocket10_processed.lmdb',
        affinity_pkl_path='affinity_info.pkl',
        output_path='expert_split.pt',
        threshold=-8.5
    )
