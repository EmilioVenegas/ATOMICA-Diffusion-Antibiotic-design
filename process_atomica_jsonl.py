import sys
import os
# Add project root to path to find DiffSBDD
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import argparse
import gzip
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from DiffSBDD.utils import format_atomica_batch, load_atomica_model
import traceback

# --- ATOMICA Imports ---
from ATOMICA.models.prediction_model import PredictionModel
from ATOMICA.models.pretrain_model import DenoisePretrainModel
from ATOMICA.models.prot_interface_model import ProteinInterfaceModel
from DiffSBDD.constants import ATOMICA_TO_DRUGLIKE_MAP, DRUGLIKE_ATOMS_DECODER

#
# --- MODIFIED FUNCTION ---
#
def process_complex(line, atomica_model, device, clash_threshold):
    """
    Processes a single line (complex) from the .jsonl file.
    This version is corrected to handle the per-block data structure
    and safely skip lines with missing data.
    
    NEW: Includes a clash check to filter out complexes with atom distances
    below 'clash_threshold'.
    """
    DRUGLIKE_VOCAB_SIZE = len(DRUGLIKE_ATOMS_DECODER)
    complex_data = json.loads(line)
    complex_id = complex_data.get('id', 'Unknown')
    
    # 1. Access nested "data" object 
    try:
        # Use .get() to safely access the 'data' object
        data = complex_data.get('data')
        if data is None:
            # This line will print if the 'data' key is missing
            # print(f"Skipping complex {complex_id}: Top-level 'data' key is missing.")
            return None

        # Use .get() for all required keys. 
        x_list = data.get('X')
        a_list = data.get('A')
        b_blocks_list = data.get('B')
        block_lengths_list = data.get('block_lengths')
        seg_blocks_list = data.get('segment_ids')

        if any(v is None for v in [x_list, a_list, b_blocks_list, block_lengths_list, seg_blocks_list]):
            # This line will print if 'block_lengths' or any other key is missing
            # print(f"Skipping complex {complex_id}: Missing one or more required data keys (X, A, B, block_lengths, or segment_ids).")
            return None

        # --- All keys exist,convert to numpy arrays ---
        x = np.array(x_list)
        a = np.array(a_list)
        b_blocks = np.array(b_blocks_list)
        block_lengths = np.array(block_lengths_list)
        seg_blocks = np.array(seg_blocks_list)
        
        # --- Perform validations ---
        if not (len(b_blocks) == len(block_lengths) == len(seg_blocks)):
            print(f"Skipping complex {complex_id}: Block array lengths mismatch (B={len(b_blocks)}, len={len(block_lengths)}, seg={len(seg_blocks)}).")
            return None
        
        # Check for empty complexes before sum()
        if len(block_lengths) == 0:
             # print(f"Skipping complex {complex_id}: 'block_lengths' is empty.")
             return None

        if not (sum(block_lengths) == len(x) == len(a)):
            print(f"Skipping complex {complex_id}: Atom array length mismatch (sum(block_lengths)={sum(block_lengths)}, len(x)={len(x)}, len(a)={len(a)}).")
            return None

    except Exception as e:
        # This will catch any other unexpected errors (like bad data types)
        print(f"Skipping complex {complex_id} due to parsing/validation error: {e}")
        return None

    # Reconstruct per-atom arrays from blocks
    pocket_coords_list = []
    pocket_atom_types_list = []
    pocket_B_types_list = []
    ligand_coords_list = []
    ligand_atom_types_list = []
    pocket_block_lengths_list = []
    
    atom_count = 0
    for i in range(len(seg_blocks)):
        length = block_lengths[i]
        # Boundary check for safety
        if atom_count + length > len(x):
             print(f"Skipping complex {complex_id}: 'block_lengths' sum mismatch during reconstruction.")
             return None
             
        block_atoms_x = x[atom_count : atom_count + length]
        block_atoms_a = a[atom_count : atom_count + length]
        block_type_b = b_blocks[i]
        
        if seg_blocks[i] == 0:  # This is a Pocket block
            pocket_coords_list.append(block_atoms_x)
            pocket_atom_types_list.append(block_atoms_a)
            pocket_B_types_list.append(block_type_b)
            pocket_block_lengths_list.append(length) 
        
        elif seg_blocks[i] == 1:  # This is a Ligand block
            ligand_coords_list.append(block_atoms_x)
            ligand_atom_types_list.append(block_atoms_a)
            
        atom_count += length

    # Concatenate lists into final numpy arrays
    if not pocket_coords_list:
        # print(f"Warning: Complex {complex_id} has no pocket atoms. Skipping.")
        return None
    if not ligand_coords_list:
        # print(f"Warning: Complex {complex_id} has no ligand atoms. Skipping.")
        return None

    # Create pocket arrays
    pocket_coords = np.concatenate(pocket_coords_list, axis=0)
    pocket_atom_types = np.concatenate(pocket_atom_types_list, axis=0)
    pocket_block_lengths = np.array(pocket_block_lengths_list)
    pocket_B_types = np.array(pocket_B_types_list)
    
    # Create ligand arrays
    ligand_coords = np.concatenate(ligand_coords_list, axis=0)
    ligand_atom_types = np.concatenate(ligand_atom_types_list, axis=0)
    
    num_pocket_blocks = len(pocket_block_lengths)
    pocket_segment_ids = np.zeros(num_pocket_blocks, dtype=np.int64) # Dummy var for format_atomica_batch

    # --- NEW: ATOM CLASH CHECK ---
    if clash_threshold > 0:
        with torch.no_grad():
            # Move to CPU for distance checks to avoid GPU OOM
            lig_coords_t = torch.from_numpy(ligand_coords).to('cpu')
            pocket_coords_t = torch.from_numpy(pocket_coords).to('cpu')
            
            # 1. Intra-ligand
            if lig_coords_t.shape[0] > 1:
                lig_dists = torch.pdist(lig_coords_t)
                if lig_dists.shape[0] > 0 and lig_dists.min() < clash_threshold:
                    print(f"Skipping complex {complex_id}: Intra-ligand clash (min: {lig_dists.min():.4f}Å < {clash_threshold}Å)")
                    return None
            
            # 2. Intra-pocket (Warning: can be slow/memory-intensive)
            if pocket_coords_t.shape[0] > 1:
                try:
                    pocket_dists = torch.pdist(pocket_coords_t) 
                    if pocket_dists.shape[0] > 0 and pocket_dists.min() < clash_threshold:
                        print(f"Skipping complex {complex_id}: Intra-pocket clash (min: {pocket_dists.min():.4f}Å < {clash_threshold}Å)")
                        return None
                except RuntimeError as e:
                    if "out of memory" in str(e) or "too large" in str(e):
                         print(f"WARNING: Skipping intra-pocket clash check for {complex_id} (too large).")
                    else:
                        raise e # Re-raise other errors

            # 3. Inter-ligand-pocket
            if lig_coords_t.shape[0] > 0 and pocket_coords_t.shape[0] > 0:
                inter_dists = torch.cdist(lig_coords_t, pocket_coords_t)
                if inter_dists.numel() > 0 and inter_dists.min() < clash_threshold:
                    print(f"Skipping complex {complex_id}: Ligand-pocket clash (min: {inter_dists.min():.4f}Å < {clash_threshold}Å)")
                    return None
    # --- END CLASH CHECK ---

    # generate Embeddings
    atomica_batch = format_atomica_batch(pocket_coords, pocket_atom_types, 
                                       pocket_B_types, pocket_block_lengths, 
                                       pocket_segment_ids, device)
    
    with torch.no_grad():
        atomica_output = atomica_model.infer(atomica_batch)
    
    pocket_atomica_embeddings = atomica_output.unit_repr.cpu().numpy()

    # Save data, with error handling
    try:
        # 1. Map ATOMICA indices (e.g., 0-120) to DRUGLIKE indices (0-8 or -1)
        #    This uses the map you defined in constants.py
        remapped_ligand_indices = ATOMICA_TO_DRUGLIKE_MAP[ligand_atom_types]

        # 2. Create a mask to find *valid* atoms (where index is not -1)
        #    This mask filters out 'p', 'm', 'g', and unwanted heavy atoms
        valid_atom_mask = (remapped_ligand_indices != -1)

        # 3. Filter both coordinates and indices using this mask
        filtered_ligand_coords = ligand_coords[valid_atom_mask]
        filtered_ligand_indices = remapped_ligand_indices[valid_atom_mask]

        # 4. Check if any ligand atoms remain
        if filtered_ligand_indices.shape[0] == 0:
            # print(f"Skipping complex {complex_id}: No valid drug-like ligand atoms found after filtering.")
            return None

        # 5. Create one-hot encoding using the *new* vocab size and *filtered* indices
        ligand_one_hot = np.eye(DRUGLIKE_VOCAB_SIZE, dtype=np.float32)[filtered_ligand_indices]

    except IndexError as e:
        print(f"ERROR during atom remapping/one-hot encoding for {complex_id}: {e}")
        # This error can happen if an atom index in your jsonl is > 120
        print(f"Original max index: {np.max(ligand_atom_types)}, Map size: {len(ATOMICA_TO_DRUGLIKE_MAP)}")
        return None
    except Exception as e:
        print(f"Unhandled ERROR during atom remapping for {complex_id}: {e}")
        return None

    # 6. Create the final data object with filtered data
    new_data = {
        'lig_coords': filtered_ligand_coords.astype(np.float32),
        'lig_one_hot': ligand_one_hot,
        'pocket_coords': pocket_coords.astype(np.float32),
        'pocket_atomica_embeddings': pocket_atomica_embeddings.astype(np.float32),
        'name': complex_id
    }

    return new_data


def main(args):
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading ATOMICA model...")
    try:
        atomica_model = load_atomica_model(args).to(device).eval()
        print("ATOMICA model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load ATOMICA model: {e}")
        print("Please check your --model_ckpt or --model_config/--model_weights paths.")
        return

    print(f"Starting processing of {input_file}...")
    print(f"Saving processed files to {output_dir}")
    print(f"Using ATOM vocab size: {len(DRUGLIKE_ATOMS_DECODER)} (from constants.DRUGLIKE_ATOMS_DECODER)")
    if args.clash_threshold > 0:
        print(f"Filtering enabled: Skipping complexes with any atoms closer than {args.clash_threshold} Å")
    else:
        print("Filtering disabled (--clash_threshold is 0.0)")


    count = 0
    skipped_count = 0
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            
            try:
                # Pass the clash_threshold argument
                processed_data = process_complex(line, atomica_model, device, 
                                 args.clash_threshold)
            
            except Exception as e:
                # ---  DETAILED ERROR BLOCK ---
                print("\n---!!! UNHANDLED ERROR ENCOUNTERED !!!---")
                print(f"Exception Type: {type(e)}")
                print(f"Exception Details: {e}")
                print(f"Failed to process line: {line.strip()[:150]}...") 
                print("--- STACK TRACE ---")
                traceback.print_exc() 
                print("------------------------------------------\n")
                # -----
                processed_data = None
                
            if processed_data:
                file_name = f"complex_{count:06d}.pt"
                file_path = output_dir / file_name
                
                tensor_data = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                               for k, v in processed_data.items()}
                
                torch.save(tensor_data, file_path)
                count += 1
            else:
                skipped_count += 1


    print(f"\nProcessing complete.")
    print(f"Successfully processed and saved {count} complexes to {output_dir}.")
    print(f"Skipped {skipped_count} complexes (due to errors or filtering).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Pre-process PL.jsonl.gz with ATOMICA embeddings")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input PL.jsonl.gz file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new .pt files.")
    parser.add_argument('--model_ckpt', type=str, default=None, 
                        help='Path of the full model .ckpt file to load.')
    parser.add_argument('--model_config', type=str, default=None, 
                        help='Path of the model .json config to load (if not using ckpt).')
    parser.add_argument('--model_weights', type=str, default=None, 
                        help='Path of the model .pt weights to load (if not using ckpt).')
       
    # --- NEW ARGUMENT ---
    parser.add_argument("--clash_threshold", type=float, default=0.0,
                        help="Minimum atom distance threshold (in Å). Complexes with any atom pair "
                             "(lig-lig, pocket-pocket, or lig-pocket) closer than this value "
                             "will be skipped. Default: 0.0 (no filtering). Recommended: 0.5")
    
    args = parser.parse_args()
    
    if not (args.model_ckpt or (args.model_config and args.model_weights)):
        parser.error("You must provide either --model_ckpt or both --model_config and --model_weights.")
        
    main(args)

