import sys
import os
# Add project root to path to find DiffSBDD
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
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

def process_complex(line, atomica_model, device, clash_threshold):
    """
    Processes a single line (complex) from the .jsonl file.
    Includes a clash check and filters atoms to DRUGLIKE vocabulary.
    """
    DRUGLIKE_VOCAB_SIZE = len(DRUGLIKE_ATOMS_DECODER)
    complex_data = json.loads(line)
    complex_id = complex_data.get('id', 'Unknown')
    
    # 1. Access nested "data" object 
    try:
        data = complex_data.get('data')
        if data is None:
            return None, "data_key_missing"

        x_list = data.get('X')
        a_list = data.get('A')
        b_blocks_list = data.get('B')
        block_lengths_list = data.get('block_lengths')
        seg_blocks_list = data.get('segment_ids')

        if any(v is None for v in [x_list, a_list, b_blocks_list, block_lengths_list, seg_blocks_list]):
            return None, "required_keys_missing"

        x = np.array(x_list)
        a = np.array(a_list)
        b_blocks = np.array(b_blocks_list)
        block_lengths = np.array(block_lengths_list)
        seg_blocks = np.array(seg_blocks_list)
        
        if not (len(b_blocks) == len(block_lengths) == len(seg_blocks)):
            print(f"Skipping complex {complex_id}: Block array lengths mismatch.")
            return None, "block_length_mismatch"
        
        if len(block_lengths) == 0:
             return None, "block_lengths_empty"

        if not (sum(block_lengths) == len(x) == len(a)):
            print(f"Skipping complex {complex_id}: Atom array length mismatch.")
            return None, "atom_length_mismatch"

    except Exception as e:
        print(f"Skipping complex {complex_id} due to parsing/validation error: {e}")
        return None, "parsing_error"

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
        if atom_count + length > len(x):
             print(f"Skipping complex {complex_id}: 'block_lengths' sum mismatch during reconstruction.")
             return None, "reconstruction_mismatch"
             
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

    if not pocket_coords_list:
        return None, "no_pocket_atoms"
    if not ligand_coords_list:
        return None, "no_ligand_atoms"

    # Create pocket arrays
    pocket_coords = np.concatenate(pocket_coords_list, axis=0)
    pocket_atom_types = np.concatenate(pocket_atom_types_list, axis=0)
    pocket_block_lengths = np.array(pocket_block_lengths_list)
    pocket_B_types = np.array(pocket_B_types_list)
    
    # Create ligand arrays
    ligand_coords = np.concatenate(ligand_coords_list, axis=0)
    ligand_atom_types = np.concatenate(ligand_atom_types_list, axis=0)
    
    num_pocket_blocks = len(pocket_block_lengths)
    pocket_segment_ids = np.zeros(num_pocket_blocks, dtype=np.int64)

    # --- ATOM CLASH CHECK ---
    if clash_threshold > 0:
        with torch.no_grad():
            lig_coords_t = torch.from_numpy(ligand_coords).to('cpu')
            pocket_coords_t = torch.from_numpy(pocket_coords).to('cpu')
            
            if lig_coords_t.shape[0] > 1:
                lig_dists = torch.pdist(lig_coords_t)
                if lig_dists.shape[0] > 0 and lig_dists.min() < clash_threshold:
                    print(f"Skipping complex {complex_id}: Intra-ligand clash")
                    return None, "intra_ligand_clash"
            
            if pocket_coords_t.shape[0] > 1:
                try:
                    pocket_dists = torch.pdist(pocket_coords_t) 
                    if pocket_dists.shape[0] > 0 and pocket_dists.min() < clash_threshold:
                        print(f"Skipping complex {complex_id}: Intra-pocket clash")
                        return None, "intra_pocket_clash"
                except RuntimeError as e:
                     if "out of memory" in str(e) or "too large" in str(e):
                         print(f"WARNING: Skipping intra-pocket clash check for {complex_id} (too large).")
                     else: raise e

            if lig_coords_t.shape[0] > 0 and pocket_coords_t.shape[0] > 0:
                inter_dists = torch.cdist(lig_coords_t, pocket_coords_t)
                if inter_dists.numel() > 0 and inter_dists.min() < clash_threshold:
                    print(f"Skipping complex {complex_id}: Ligand-pocket clash")
                    return None, "inter_clash"

    # --- Generate Embeddings ---
    atomica_batch = format_atomica_batch(pocket_coords, pocket_atom_types, 
                                       pocket_B_types, pocket_block_lengths, 
                                       pocket_segment_ids, device)
    
    with torch.no_grad():
        atomica_output = atomica_model.infer(atomica_batch)
    
    pocket_atomica_embeddings = atomica_output.unit_repr.cpu().numpy()

    # --- Filter pocket atoms to match DRUGLIKE vocabulary ---
    try:
        remapped_pocket_indices = ATOMICA_TO_DRUGLIKE_MAP[pocket_atom_types]
        valid_pocket_mask = (remapped_pocket_indices != -1)
        if valid_pocket_mask.sum() == 0:
            return None, "no_valid_pocket_atoms"

        filtered_pocket_coords = pocket_coords[valid_pocket_mask]
        filtered_pocket_embeddings = pocket_atomica_embeddings[valid_pocket_mask]
        filtered_pocket_indices = remapped_pocket_indices[valid_pocket_mask]
        pocket_one_hot = np.eye(DRUGLIKE_VOCAB_SIZE, dtype=np.float32)[filtered_pocket_indices]

    except Exception as e:
        print(f"ERROR during POCKET atom remapping for {complex_id}: {e}")
        return None, "pocket_remapping_error"

    # --- Filter ligand atoms to match DRUGLIKE vocabulary ---
    try:
        remapped_ligand_indices = ATOMICA_TO_DRUGLIKE_MAP[ligand_atom_types]
        valid_atom_mask = (remapped_ligand_indices != -1)
        filtered_ligand_coords = ligand_coords[valid_atom_mask]
        filtered_ligand_indices = remapped_ligand_indices[valid_atom_mask]

        if filtered_ligand_indices.shape[0] == 0:
            return None, "no_valid_ligand_atoms"

        ligand_one_hot = np.eye(DRUGLIKE_VOCAB_SIZE, dtype=np.float32)[filtered_ligand_indices]

    except Exception as e:
        print(f"ERROR during LIGAND atom remapping for {complex_id}: {e}")
        return None, "ligand_remapping_error"

    # --- Create final data object ---
    new_data = {
        'lig_coords': filtered_ligand_coords.astype(np.float32),
        'lig_one_hot': ligand_one_hot,
        'pocket_coords': filtered_pocket_coords.astype(np.float32),          
        'pocket_atomica_embeddings': filtered_pocket_embeddings.astype(np.float32), 
        'pocket_one_hot': pocket_one_hot.astype(np.float32),               
        'name': complex_id
    }

    return new_data, "success"


def main():
   
    # 1. File Paths
    input_file = "PLjsonl.gz"
    # *** UNIFIED: Output dir is now the FINAL filtered directory ***
    output_dir = "data/processed_atomica_filtered2"

    # 2. Model Loading
    model_ckpt = None 
    model_config = "ATOMICA/pretrain/pretrain_model_config.json"
    model_weights = "ATOMICA/pretrain/pretrain_model_weights.pt"

    # 3. Processing Parameters
    clash_threshold = 0.5  # Min atom distance (Å). 0.0 to disable.
    
    # 4. Dataset Splitting
    train_split = 0.8  
    val_split = 0.1   
    test_split = 0.1 

    # 5. Max Process Count
    max_success_count = 10000 
    
    # 6. *** UNIFIED: Size Filter Parameters ***
    min_lig_atoms = 15
    max_lig_atoms = 75
    min_pocket_atoms = 50
    max_pocket_atoms = 380

    # --- Setup and Validation ---
    
    if not (model_ckpt or (model_config and model_weights)):
        print("FATAL: You must provide either --model_ckpt or both --model_config and --model_weights.")
        return

    if not np.isclose(train_split + val_split + test_split, 1.0):
        print(f"FATAL: Split ratios must sum to 1.0")
        return

    input_file_path = Path(input_file)
    output_dir_path = Path(output_dir)

    train_dir = output_dir_path / "train"
    val_dir = output_dir_path / "val"
    test_dir = output_dir_path / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading ATOMICA model...")
    try:
        class MockArgs:
            def __init__(self):
                self.model_ckpt = model_ckpt
                self.model_config = model_config
                self.model_weights = model_weights
        
        atomica_model = load_atomica_model(MockArgs()).to(device).eval()
        print("ATOMICA model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load ATOMICA model: {e}")
        return

    print(f"Starting processing of {input_file_path}...")
    print(f"Saving *filtered* files to: {output_dir_path}")
    print(f"Size filter criteria:")
    print(f"  - Ligand: {min_lig_atoms} <= atoms <= {max_lig_atoms}")
    print(f"  - Pocket: {min_pocket_atoms} <= atoms <= {max_pocket_atoms}")
    
    if clash_threshold > 0:
        print(f"Clash filter enabled: Skipping complexes with atoms closer than {clash_threshold} Å")
    
    if max_success_count is not None:
        print(f"Processing will STOP after {max_success_count} successfully processed and *filtered* complexes.")


    # --- Counters ---
    total_success_count = 0
    train_count = 0
    val_count = 0
    test_count = 0
    skipped_count_processing = 0
    skipped_count_size_filter = 0 # New counter for size-based filtering
    
    # --- Main Processing Loop ---
    with gzip.open(input_file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            
            processed_data = None
            skip_reason = "unhandled_exception"
            try:
                processed_data, skip_reason = process_complex(line, atomica_model, device, clash_threshold)
            
            except Exception as e:
                print("\n---!!! UNHANDLED ERROR ENCOUNTERED !!!---")
                print(f"Exception Type: {type(e)}")
                print(f"Exception Details: {e}")
                print(f"Failed to process line: {line.strip()[:150]}...") 
                print("--- STACK TRACE ---")
                traceback.print_exc() 
                print("------------------------------------------\n")
                
            if processed_data:
                # --- UNIFICATION: APPLY SIZE FILTER HERE ---
                lig_nodes = processed_data['lig_coords'].shape[0]
                pocket_nodes = processed_data['pocket_coords'].shape[0]

                lig_criteria_met = (min_lig_atoms <= lig_nodes <= max_lig_atoms)
                pocket_criteria_met = (min_pocket_atoms <= pocket_nodes <= max_pocket_atoms)

                if lig_criteria_met and pocket_criteria_met:
                    # --- All criteria met, proceed to save ---
                    r = np.random.rand()  # Random float 0.0 to 1.0
                    
                    if r < train_split:
                        target_dir = train_dir
                        train_count += 1
                        file_name = f"complex_{train_count:06d}.pt"
                    elif r < (train_split + val_split):
                        target_dir = val_dir
                        val_count += 1
                        file_name = f"complex_{val_count:06d}.pt"
                    else:
                        target_dir = test_dir
                        test_count += 1
                        file_name = f"complex_{test_count:06d}.pt"

                    file_path = target_dir / file_name
                    
                    tensor_data = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                                   for k, v in processed_data.items()}
                    
                    torch.save(tensor_data, file_path)
                    
                    total_success_count += 1
                    
                    if max_success_count is not None and total_success_count >= max_success_count:
                        print(f"\nReached max success count of {max_success_count}. Stopping processing.")
                        break
                else:
                    # --- Failed size filter ---
                    skipped_count_size_filter += 1
                
            else:
                # --- Failed processing (clash, no atoms, etc.) ---
                skipped_count_processing += 1


    print(f"\nProcessing complete.")
    print(f"Total successfully processed and saved: {total_success_count}")
    print(f"  - Train: {train_count}")
    print(f"  - Val:   {val_count}")
    print(f"  - Test:  {test_count}")
    print(f"Skipped {skipped_count_processing} complexes (due to processing errors, clashes, or atom filtering).")
    print(f"Skipped {skipped_count_size_filter} complexes (due to size filter criteria).")


if __name__ == "__main__":
    main()