
import gzip
import json
import sys
from collections import defaultdict
from tqdm import tqdm

# --- Configuration ---
FILE_NAME = '/home/emiliovenegas/Downloads/PL.jsonl.gz'

def analyze_dataset():
    """
    Reads the gzipped JSON Lines file to analyze its contents based on the
    provided sample structure.
    """
    
    # === Part 1: Full Dataset Scan ===
    print(f"--- STARTING FULL ANALYSIS of {FILE_NAME} ---")
    total_complexes = 0
    top_level_keys = defaultdict(int)
    data_sub_keys = defaultdict(int)
    
    protein_atom_counts = []
    ligand_atom_counts = []
    total_atom_counts = []
    
    affinity_values = []
    
    try:
        # We re-open the file to iterate from the beginning
        with gzip.open(FILE_NAME, 'rt', encoding='utf-8') as f:
            
            # tqdm will create a progress bar
            for line in tqdm(f, desc="Analyzing complexes"):
                try:
                    data = json.loads(line)
                    total_complexes += 1

                    # Count top-level keys
                    for key in data.keys():
                        top_level_keys[key] += 1
                    
                    # --- Atom Counting Logic ---
                    # This is now based on the sample you provided
                    p_atoms = 0
                    l_atoms = 0
                    
                    # Get the main data payload
                    main_data = data.get('data', {})
                    
                    # Get the lists we need
                    block_lengths = main_data.get('block_lengths', [])
                    segment_ids = main_data.get('segment_ids', [])
                    
                    # Get total atoms from the coordinate list 'X'
                    all_atoms_x = main_data.get('X', [])
                    total_atom_counts.append(len(all_atoms_x))

                    # Count keys inside the 'data' object
                    for key in main_data.keys():
                        data_sub_keys[key] += 1

                    # Iterate over blocks to assign atoms to protein or ligand
                    if len(block_lengths) == len(segment_ids):
                        for length, seg_id in zip(block_lengths, segment_ids):
                            if seg_id == 0:
                                p_atoms += length
                            elif seg_id == 1:
                                l_atoms += length
                        
                        if p_atoms > 0:
                            protein_atom_counts.append(p_atoms)
                        if l_atoms > 0:
                            ligand_atom_counts.append(l_atoms)
                    
                    # Get affinity
                    affinity_val = data.get('affinity', {}).get('neglog_aff')
                    if affinity_val is not None:
                        affinity_values.append(affinity_val)

                except json.JSONDecodeError:
                    print(f"\nWarning: Skipping a line that was not valid JSON.")
                except Exception as e:
                    print(f"\nWarning: Skipping line due to unexpected error: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at '{FILE_NAME}'")
        print("Please make sure the file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        return

    # === Part 2: Print Statistics ===
    print("\n--- ANALYSIS COMPLETE ---")
    print(f"\nTotal complexes found: {total_complexes:,}")

    print("\nTop-level JSON keys found (and frequency):")
    for key, count in top_level_keys.items():
        print(f"  - '{key}': {count:,} times")

    print("\nSub-keys inside 'data' object (and frequency):")
    for key, count in data_sub_keys.items():
        print(f"  - '{key}': {count:,} times")

    if total_atom_counts:
        print(f"\n--- Total Atom Statistics (from 'data.X') ---")
        print(f"  Min atoms:    {min(total_atom_counts)}")
        print(f"  Max atoms:    {max(total_atom_counts)}")
        print(f"  Avg atoms:    {sum(total_atom_counts) / len(total_atom_counts):.2f}")

    if protein_atom_counts:
        print(f"\n--- Protein Atom Statistics (from 'segment_id == 0') ---")
        print(f"  Min atoms:    {min(protein_atom_counts)}")
        print(f"  Max atoms:    {max(protein_atom_counts)}")
        print(f"  Avg atoms:    {sum(protein_atom_counts) / len(protein_atom_counts):.2f}")
    
    if ligand_atom_counts:
        print(f"\n--- Ligand Atom Statistics (from 'segment_id == 1') ---")
        print(f"  Min atoms:    {min(ligand_atom_counts)}")
        print(f"  Max atoms:    {max(ligand_atom_counts)}")
        print(f"  Avg atoms:    {sum(ligand_atom_counts) / len(ligand_atom_counts):.2f}")

    if affinity_values:
        print(f"\n--- Affinity Statistics ('neglog_aff') ---")
        print(f"  Min affinity:    {min(affinity_values)}")
        print(f"  Max affinity:    {max(affinity_values)}")
        print(f"  Avg affinity:    {sum(affinity_values) / len(affinity_values):.2f}")
        
    if not protein_atom_counts or not ligand_atom_counts:
        print("\n*** WARNING: Could not find any protein/ligand atoms. ***")
        print("This might mean the 'segment_ids' are not 0 or 1, or the")
        print("'block_lengths' / 'segment_ids' keys were not found.")


if __name__ == "__main__":
    analyze_dataset()