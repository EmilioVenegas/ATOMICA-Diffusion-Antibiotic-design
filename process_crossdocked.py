"""
Process CrossDocked LMDB dataset with ATOMICA embeddings.
Adapted from process_and_filter.py to work with LMDB input.
Maps atoms to druglike vocabulary and generates compatible output format.
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

from DiffSBDD.utils import format_atomica_batch, load_atomica_model
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
                    print(f"Skipping complex {complex_id}: Intra-ligand clash")
                    return None, "intra_ligand_clash"
            
            # Check intra-pocket clashes
            if pocket_coords_t.shape[0] > 1:
                try:
                    pocket_dists = torch.pdist(pocket_coords_t)
                    if pocket_dists.shape[0] > 0 and pocket_dists.min() < clash_threshold:
                        print(f"Skipping complex {complex_id}: Intra-pocket clash")
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
                    print(f"Skipping complex {complex_id}: Ligand-pocket clash")
                    return None, "inter_clash"
    
    # --- Convert atomic numbers to ATOMICA atom types ---
    # CrossDocked uses atomic numbers (6=C, 7=N, etc.)
    # ATOMICA uses indices where 'C' is at position 8 (after 'p','m','g' + He,Li,Be,B,C...)
    # We need to map: atomic_number -> element_symbol -> atomica_index
    
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
                print(f"Skipping complex {complex_id}: Unknown pocket atom {atomic_num}")
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
                print(f"Skipping complex {complex_id}: Unknown ligand atom {atomic_num}")
                return None, "unknown_ligand_atom"
        ligand_atomica_types = np.array(ligand_atomica_types, dtype=np.int64)
    except Exception as e:
        print(f"ERROR converting ligand atoms for {complex_id}: {e}")
        return None, "ligand_conversion_error"
    
    # --- Generate ATOMICA embeddings for pocket ---
    # ATOMICA requires a global node (block + atom) at the start of each segment!
    # This was the missing piece causing the dtype error.
    
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
    
    # Get all atom embeddings (including global atom, matching antibiotic processing)
    # The antibiotic data structures include the global atom in their pocket representations
    pocket_atomica_embeddings_all = atomica_output.unit_repr.cpu().numpy()
    
    # --- Filter pocket atoms to DRUGLIKE vocabulary ---
    # Note: pocket_atomica_types_with_global includes the global atom at index 0
    # We need to filter to match it with the embeddings
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
    
    # --- Create final data object (matching process_and_filter.py format) ---
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
    output_base = Path("data/processed_crossdocked_atomica")
    
    # Model paths
    model_ckpt = None
    model_config = "ATOMICA/pretrain/pretrain_model_config.json"
    model_weights = "ATOMICA/pretrain/pretrain_model_weights.pt"
    
    # Processing parameters
    clash_threshold = 0.5  # Min atom distance (Å)
    
    # Size filter parameters (adjusted for CrossDocked pocket10 representation)
    # CrossDocked pockets are much larger than antibiotic pockets!
    # Analysis shows: pocket median=542, range=199-869
    min_lig_atoms = 10   # 5th percentile: 12, being slightly lenient
    max_lig_atoms = 40   # 95th percentile: 36, keeping some margin
    min_pocket_atoms = 350  # Well below 5th percentile (395) to be safe
    max_pocket_atoms = 800  # Above 95th percentile (756) to keep 95%+ of data
    
    # Max samples to process (None = all)
    max_success_count = 10000  # Process 1000 for testing, then run full dataset
    
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
        class MockArgs:
            def __init__(self):
                self.model_ckpt = model_ckpt
                self.model_config = model_config
                self.model_weights = model_weights
        
        atomica_model = load_atomica_model(MockArgs()).to(device).eval()
        print("ATOMICA model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load ATOMICA model: {e}")
        traceback.print_exc()
        return
    
    # Split ratios (using random splitting instead of predefined split file)
    # The provided split file has too few val/test samples (only 100 each total)
    train_split = 0.95   # 95% train
    val_split = 0.025    # 2.5% val
    test_split = 0.025   # 2.5% test
    
    if not np.isclose(train_split + val_split + test_split, 1.0):
        print(f"FATAL: Split ratios must sum to 1.0")
        return
    
    print(f"Using random splitting: train={train_split}, val={val_split}, test={test_split}")
    print(f"  (ignoring split file - it has insufficient val/test samples)")
    
    # Open LMDB
    print(f"Opening LMDB: {lmdb_path}")
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    
    # === Processing ===
    print(f"\nStarting processing...")
    print(f"Output directory: {output_base}")
    print(f"Size filter criteria:")
    print(f"  - Ligand: {min_lig_atoms} <= atoms <= {max_lig_atoms}")
    print(f"  - Pocket: {min_pocket_atoms} <= atoms <= {max_pocket_atoms}")
    
    if clash_threshold > 0:
        print(f"Clash filter: atoms closer than {clash_threshold} Å will be rejected")
    
    if max_success_count:
        print(f"Will stop after {max_success_count} successful complexes")
    
    # Counters
    total_success = 0
    train_count = 0
    val_count = 0
    test_count = 0
    skipped_processing = 0
    skipped_size_filter = 0
    
    with env.begin() as txn:
        cursor = txn.cursor()
        
        for key, value in tqdm(cursor, desc="Processing"):
            # Deserialize data
            data = pickle.loads(value)
            
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
                    # Random split assignment
                    r = np.random.rand()
                    
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
                    
                    # Convert to tensors
                    tensor_data = {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in processed_data.items()
                    }
                    
                    torch.save(tensor_data, file_path)
                    total_success += 1
                    
                    # Check if we've hit max count
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
    print(f"  - Train: {train_count} ({100*train_count/total_success if total_success > 0 else 0:.1f}%)")
    print(f"  - Val:   {val_count} ({100*val_count/total_success if total_success > 0 else 0:.1f}%)")
    print(f"  - Test:  {test_count} ({100*test_count/total_success if total_success > 0 else 0:.1f}%)")
    print(f"Skipped {skipped_processing} complexes (processing errors/clashes)")
    print(f"Skipped {skipped_size_filter} complexes (size filter)")


if __name__ == "__main__":
    main()
