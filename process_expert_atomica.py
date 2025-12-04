"""
Process Expert CrossDocked LMDB dataset with ATOMICA embeddings.
Filters dataset based on expert_split.pt (Vina score < -8.5).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import lmdb
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback

# Import ATOMICA models first
from ATOMICA.models.prediction_model import PredictionModel
from ATOMICA.models.pretrain_model import DenoisePretrainModel
from ATOMICA.models.prot_interface_model import ProteinInterfaceModel

from DiffSBDD.utils import format_atomica_batch
from DiffSBDD.constants import ATOMICA_TO_DRUGLIKE_MAP, DRUGLIKE_ATOMS_DECODER, atomica_block_encoder

def process_complex(key, data, atomica_model, device, clash_threshold):
    """
    Process a single complex from LMDB.
    Includes clash check and filters atoms to DRUGLIKE vocabulary.
    """
    DRUGLIKE_VOCAB_SIZE = len(DRUGLIKE_ATOMS_DECODER)
    complex_id = key.decode() if isinstance(key, bytes) else str(key)
    
    try:
        # Extract pocket/protein data - LMDB data is in Tensor format
        pocket_coords_raw = data['protein_pos']
        pocket_atom_types_raw = data['protein_element']
        
        # Convert to numpy if needed
        if isinstance(pocket_coords_raw, torch.Tensor):
            pocket_coords = pocket_coords_raw.cpu().numpy()
        else:
            pocket_coords = pocket_coords_raw
        
        if isinstance(pocket_atom_types_raw, torch.Tensor):
            pocket_atom_types = pocket_atom_types_raw.cpu().numpy()
        else:
            pocket_atom_types = pocket_atom_types_raw
        
        # Extract ligand data
        ligand_coords_raw = data['ligand_pos']
        ligand_atom_types_raw = data['ligand_element']
        
        if isinstance(ligand_coords_raw, torch.Tensor):
            ligand_coords = ligand_coords_raw.cpu().numpy()
        else:
            ligand_coords = ligand_coords_raw
        
        if isinstance(ligand_atom_types_raw, torch.Tensor):
            ligand_atom_types = ligand_atom_types_raw.cpu().numpy()
        else:
            ligand_atom_types = ligand_atom_types_raw
        
        # Validate we have data
        if len(pocket_coords) == 0:
            return None, "no_pocket_atoms"
        if len(ligand_coords) == 0:
            return None, "no_ligand_atoms"
            
    except Exception as e:
        print(f"Skipping complex {complex_id} due to parsing error: {e}")
        return None, "parsing_error"
    
    # --- ATOM CLASH CHECK ---
    if clash_threshold > 0:
        with torch.no_grad():
            lig_coords_t = torch.from_numpy(ligand_coords).to('cpu')
            pocket_coords_t = torch.from_numpy(pocket_coords).to('cpu')
            
            # Check intra-ligand clashes
            if lig_coords_t.shape[0] > 1:
                lig_dists = torch.pdist(lig_coords_t)
                if lig_dists.shape[0] > 0 and lig_dists.min() < clash_threshold:
                    # print(f"Skipping complex {complex_id}: Intra-ligand clash")
                    return None, "intra_ligand_clash"
            
            # Check intra-pocket clashes
            if pocket_coords_t.shape[0] > 1:
                try:
                    pocket_dists = torch.pdist(pocket_coords_t)
                    if pocket_dists.shape[0] > 0 and pocket_dists.min() < clash_threshold:
                        # print(f"Skipping complex {complex_id}: Intra-pocket clash")
                        return None, "intra_pocket_clash"
                except RuntimeError as e:
                    if "out of memory" in str(e) or "too large" in str(e):
                        print(f"WARNING: Skipping intra-pocket clash check for {complex_id} (too large).")
                    else:
                        raise e
            
            # Check inter-molecular clashes
            if lig_coords_t.shape[0] > 0 and pocket_coords_t.shape[0] > 0:
                inter_dists = torch.cdist(lig_coords_t, pocket_coords_t)
                if inter_dists.numel() > 0 and inter_dists.min() < clash_threshold:
                    # print(f"Skipping complex {complex_id}: Ligand-pocket clash")
                    return None, "inter_clash"
    
    # --- Convert atomic numbers to ATOMICA atom types ---
    atomic_num_to_symbol = {
        1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
        15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
    }
    
    # Get ATOMICA atom encoder from constants
    from DiffSBDD.constants import atomica_atom_encoder
    
    # Convert pocket atomic numbers to ATOMICA indices
    try:
        pocket_atomica_types = []
        for atomic_num in pocket_atom_types:
            symbol = atomic_num_to_symbol.get(int(atomic_num), None)
            if symbol and symbol in atomica_atom_encoder:
                pocket_atomica_types.append(atomica_atom_encoder[symbol])
            else:
                # Unknown atom type, skip this complex
                # print(f"Skipping complex {complex_id}: Unknown pocket atom {atomic_num}")
                return None, "unknown_pocket_atom"
        pocket_atomica_types = np.array(pocket_atomica_types, dtype=np.int64)
    except Exception as e:
        print(f"ERROR converting pocket atoms for {complex_id}: {e}")
        return None, "pocket_conversion_error"
    
    # Convert ligand atomic numbers to ATOMICA indices
    try:
        ligand_atomica_types = []
        for atomic_num in ligand_atom_types:
            symbol = atomic_num_to_symbol.get(int(atomic_num), None)
            if symbol and symbol in atomica_atom_encoder:
                ligand_atomica_types.append(atomica_atom_encoder[symbol])
            else:
                # print(f"Skipping complex {complex_id}: Unknown ligand atom {atomic_num}")
                return None, "unknown_ligand_atom"
        ligand_atomica_types = np.array(ligand_atomica_types, dtype=np.int64)
    except Exception as e:
        print(f"ERROR converting ligand atoms for {complex_id}: {e}")
        return None, "ligand_conversion_error"
    
    # --- Generate ATOMICA embeddings for pocket ---
    from ATOMICA.data.pdb_utils import VOCAB
    
    # Compute global node position as center of mass
    global_pos = np.mean(pocket_coords, axis=0)
    
    # Prepend global atom to coordinates and atom types
    pocket_coords_with_global = np.vstack([global_pos, pocket_coords])
    global_atom_idx = VOCAB.get_atom_global_idx()
    pocket_atomica_types_with_global = np.concatenate([[global_atom_idx], pocket_atomica_types])
    
    # Create block structure with global block first
    n_pocket_atoms_with_global = len(pocket_atomica_types_with_global)
    global_block_idx = VOCAB.symbol_to_idx(VOCAB.GLB)
    
    # Block types: [GLOBAL, UNK for the main pocket]
    pocket_B_types = np.array([global_block_idx, atomica_block_encoder.get('UNK', 21)], dtype=np.int64)
    
    # Block lengths: [1 for global atom, rest for pocket atoms]
    n_pocket_atoms = len(pocket_atomica_types)
    pocket_block_lengths = np.array([1, n_pocket_atoms], dtype=np.int64)
    
    # Segment IDs: all belong to segment 0 (pocket segment)
    pocket_segment_ids = np.array([0, 0], dtype=np.int64)
    
    atomica_batch = format_atomica_batch(
        pocket_coords_with_global, pocket_atomica_types_with_global,
        pocket_B_types, pocket_block_lengths,
        pocket_segment_ids, device
    )
    
    with torch.no_grad():
        atomica_output = atomica_model.infer(atomica_batch)
    
    # Get all atom embeddings (including global atom)
    pocket_atomica_embeddings_all = atomica_output.unit_repr.cpu().numpy()
    
    # --- Filter pocket atoms to DRUGLIKE vocabulary ---
    try:
        # Map all atom types (including global) to druglike vocab
        remapped_pocket_indices = ATOMICA_TO_DRUGLIKE_MAP[pocket_atomica_types_with_global]
        valid_pocket_mask = (remapped_pocket_indices != -1)
        
        if valid_pocket_mask.sum() == 0:
            return None, "no_valid_pocket_atoms"
        
        # Filter coords, embeddings, and indices using the mask
        filtered_pocket_coords = pocket_coords_with_global[valid_pocket_mask]
        filtered_pocket_embeddings = pocket_atomica_embeddings_all[valid_pocket_mask]
        filtered_pocket_indices = remapped_pocket_indices[valid_pocket_mask]
        pocket_one_hot = np.eye(DRUGLIKE_VOCAB_SIZE, dtype=np.float32)[filtered_pocket_indices]
        
    except Exception as e:
        print(f"ERROR during POCKET atom remapping for {complex_id}: {e}")
        return None, "pocket_remapping_error"
    
    # --- Filter ligand atoms to DRUGLIKE vocabulary ---
    try:
        remapped_ligand_indices = ATOMICA_TO_DRUGLIKE_MAP[ligand_atomica_types]
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
    # === Configuration ===
    lmdb_path = "data/crossdocked_pocket10_processed.lmdb"
    split_path = "data/crossdocked_split.pt"
    expert_split_path = "expert_split.pt"
    output_base = Path("data/processed_expert_atomica")
    
    # Model paths
    model_ckpt = None
    model_config = "ATOMICA/pretrain/pretrain_model_config.json"
    model_weights = "ATOMICA/pretrain/pretrain_model_weights.pt"
    
    # Processing parameters
    clash_threshold = 0.5  # Min atom distance (Å)
    
    # Size filter parameters
    min_lig_atoms = 10
    max_lig_atoms = 40
    min_pocket_atoms = 350
    max_pocket_atoms = 800
    
    # Max samples to process (None = all)
    max_success_count = None 
    
    # === Setup ===
    if not (model_ckpt or (model_config and model_weights)):
        print("FATAL: Must provide model_ckpt or both model_config and model_weights")
        return
    
    # Create output directories
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    test_dir = output_base / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ATOMICA model
    print("Loading ATOMICA model...")
    try:
        atomica_model = PredictionModel.load_from_config_and_weights(model_config, model_weights).to(device).eval()
        print("ATOMICA model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load ATOMICA model: {e}")
        traceback.print_exc()
        return
    
    # Load Expert Split
    print(f"Loading expert split from {expert_split_path}...")
    try:
        expert_split = torch.load(expert_split_path)
        if isinstance(expert_split, dict) and 'train' in expert_split:
            expert_filenames = set(expert_split['train'])
        elif isinstance(expert_split, list):
            expert_filenames = set(expert_split)
        else:
            print("Unknown split file format. Expected dict with 'train' key or list.")
            return
        print(f"Loaded {len(expert_filenames)} expert filenames.")
    except Exception as e:
        print(f"Error loading expert split: {e}")
        return

    # Load Standard Split
    print(f"Loading standard split from {split_path}...")
    try:
        std_split = torch.load(split_path)
        std_train_idxs = set(std_split['train'])
        std_val_idxs = set(std_split['val'])
        std_test_idxs = set(std_split['test'])
        all_std_idxs = std_train_idxs | std_val_idxs | std_test_idxs
        print(f"Loaded standard split: Train={len(std_train_idxs)}, Val={len(std_val_idxs)}, Test={len(std_test_idxs)}")
    except Exception as e:
        print(f"Error loading standard split: {e}")
        return

    # Target sizes for Val/Test from the "Candidate Pool" (Expert items NOT in standard split)
    target_val_size = 1000
    target_test_size = 1000
    
    print(f"Hybrid Split Strategy:")
    print(f"  - Train: Expert items present in Standard Train")
    print(f"  - Val:   First {target_val_size} Expert items NOT in Standard Split")
    print(f"  - Test:  Next {target_test_size} Expert items NOT in Standard Split")
    
    # Open LMDB
    print(f"Opening LMDB: {lmdb_path}")
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    
    # === Processing ===
    print(f"\nStarting processing...")
    print(f"Output directory: {output_base}")
    
    # Counters
    total_success = 0
    train_count = 0
    val_count = 0
    test_count = 0
    skipped_processing = 0
    skipped_size_filter = 0
    skipped_not_expert = 0
    skipped_std_val_test = 0 # Expert items that were in Std Val/Test (we skip to avoid leakage)
    skipped_excess_candidate = 0 # Expert items not in Std Split, but Val/Test buckets are full
    
    with env.begin() as txn:
        cursor = txn.cursor()
        
        # Use enumerate to get index
        for idx, (key, value) in tqdm(enumerate(cursor), total=env.stat()['entries'], desc="Processing"):
            # Deserialize data
            data = pickle.loads(value)
            
            # Check if in expert set (by filename)
            if 'ligand_filename' not in data:
                continue
                
            if data['ligand_filename'] not in expert_filenames:
                skipped_not_expert += 1
                continue
            
            # --- Determine Split Assignment ---
            target_dir = None
            is_candidate = False
            
            if idx in std_train_idxs:
                target_dir = train_dir
                # Will increment train_count later on success
            elif idx in std_val_idxs or idx in std_test_idxs:
                # Skip to avoid leakage (even though overlap is minimal/zero)
                skipped_std_val_test += 1
                continue
            else:
                # Not in standard split -> Candidate for Val/Test
                if val_count < target_val_size:
                    target_dir = val_dir
                elif test_count < target_test_size:
                    target_dir = test_dir
                else:
                    skipped_excess_candidate += 1
                    continue
            
            # Process complex
            processed_data = None
            skip_reason = "unhandled_exception"
            try:
                processed_data, skip_reason = process_complex(
                    key, data, atomica_model, device, clash_threshold
                )
            except Exception as e:
                print(f"\n--- UNHANDLED ERROR ---")
                print(f"Key: {key}")
                print(f"Exception: {e}")
                traceback.print_exc()
                print("----------------------\n")
            
            if processed_data:
                # Apply size filter
                lig_nodes = processed_data['lig_coords'].shape[0]
                pocket_nodes = processed_data['pocket_coords'].shape[0]
                
                lig_ok = (min_lig_atoms <= lig_nodes <= max_lig_atoms)
                pocket_ok = (min_pocket_atoms <= pocket_nodes <= max_pocket_atoms)
                
                if lig_ok and pocket_ok:
                    # Assign filename based on target dir
                    if target_dir == train_dir:
                        train_count += 1
                        file_name = f"complex_{train_count:06d}.pt"
                    elif target_dir == val_dir:
                        val_count += 1
                        file_name = f"complex_{val_count:06d}.pt"
                    elif target_dir == test_dir:
                        test_count += 1
                        file_name = f"complex_{test_count:06d}.pt"
                    
                    file_path = target_dir / file_name
                    
                    # Convert to tensors
                    tensor_data = {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in processed_data.items()
                    }
                    
                    torch.save(tensor_data, file_path)
                    total_success += 1
                    
                    if max_success_count and total_success >= max_success_count:
                        print(f"\nReached max count of {max_success_count}")
                        break
                else:
                    skipped_size_filter += 1
            else:
                skipped_processing += 1
    
    env.close()
    
    # === Summary ===
    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}")
    print(f"Total successfully processed and saved: {total_success}")
    print(f"  - Train: {train_count}")
    print(f"  - Val:   {val_count}")
    print(f"  - Test:  {test_count}")
    print(f"Skipped {skipped_not_expert} complexes (not in expert split)")
    print(f"Skipped {skipped_std_val_test} complexes (in standard val/test)")
    print(f"Skipped {skipped_excess_candidate} complexes (excess candidates)")
    print(f"Skipped {skipped_processing} complexes (processing errors/clashes)")
    print(f"Skipped {skipped_size_filter} complexes (size filter)")


if __name__ == "__main__":
    main()
